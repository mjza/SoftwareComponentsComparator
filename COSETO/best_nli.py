# Imports and Setup
import os
import re
import pandas as pd
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv
from tqdm import tqdm


# Load environment variables
load_dotenv()

# Database URL from environment variable
DATABASE_URL = os.getenv('DATABASE_URL')

# Database connection (adjust accordingly)
engine = create_engine(DATABASE_URL)

# Set the device for PyTorch
# This will use GPU if available, otherwise fallback to CPU.
# Using GPU can significantly speed up the NLI model inference, especially for larger models.
# Ensure that the PyTorch library is installed with CUDA support if you want to use GPU.
# If CUDA is not available, it will automatically use CPU, which is slower but still functional
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define all tuning parameters
# NLI models to evaluate
# These models are selected based on their performance in various NLI tasks.
# They include a mix of transformer-based models that are known for their effectiveness in natural language inference
# tasks, particularly in the context of software quality attributes.
# The models are chosen to cover a range of architectures and sizes, from smaller models like DeBERTa-v3-base to larger ones like RoBERTa-large.
# The models are expected to handle the nuances of software-related text and quality attributes effectively.
# The list can be extended with more models as needed, depending on the evaluation requirements and advancements in NLI research.
# The models are expected to be pre-trained and fine-tuned on relevant datasets for NLI tasks.
# The models are expected to be compatible with the Hugging Face Transformers library.
nli_models = [
    'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
    'microsoft/deberta-v3-large',    
    'google-t5/t5-base',
    'cross-encoder/nli-roberta-base',
    'FacebookAI/roberta-large-mnli',
]

# Define chunking strategies
# These strategies determine how the text is split into smaller parts for NLI processing.
# "sentence" splits by sentences, "paragraph" by paragraphs, and "newline" by newlines.
# The choice of strategy can significantly affect the performance of the NLI model.
# "sentence" is often the most effective for NLI tasks, as it preserves context better,
# while "paragraph" and "newline" can be useful for longer texts or specific formats.
# The "sentence" strategy is the default and generally recommended for most cases.
# However, the other strategies can be useful depending on the nature of the text and the hypotheses
chunk_strategies = ["sentence", "paragraph", "newline"]

# Define threshold configurations
# These thresholds are used to determine the confidence level for labeling an issue as related to a quality attribute.
# The thresholds are designed to be flexible, allowing for different levels of strictness in labeling.
# The first set is more lenient, while the second set is stricter.
# The third set is even stricter, allowing for fine-tuning based on the model's performance.
threshold_configs = [
    [0.60, 0.55, 0.50, 0.45, 0.40],
    [0.70, 0.65, 0.60, 0.55],    
    [0.80, 0.70, 0.60],
]

# Define max tokens and aggregation logics
# These parameters control the maximum length of input sequences for the NLI model and how scores are aggregated.
# The max_tokens_list defines the maximum number of tokens for each input sequence.
# The aggregation_logic_list defines how the scores from multiple hypotheses are combined.
# "max" aggregates by taking the maximum score across hypotheses, while "mean" averages the scores.
# The max_tokens_list allows for flexibility in handling longer texts, with different limits based on the model's capabilities.
# The choice of max tokens can affect the model's performance, especially for longer texts.
max_tokens_list = [128, 
                   256,
                   512, 
                   1024 # does not work for 'cross-encoder/nli-roberta-base' and 'FacebookAI/roberta-large-mnli'
                   ]

# Define aggregation logics
# "max" for maximum score, "mean" for average score
# "max" is often more robust for NLI tasks, but "mean" can be useful for averaging over multiple chunks
# depending on the nature of the text and hypotheses.
# Here we include both to allow flexibility in evaluation.
aggregation_logic_list = ["max", "mean"]

# Function to fetch data from the database
# This function retrieves labeled issues, quality attributes, and related words from the database.
# It uses SQL queries to join the combined issues with ground truth data and fetch quality attributes.
# The resulting DataFrames contain the necessary information for evaluating NLI models against quality attributes.
def fetch_data():
    """Fetch labeled issues, quality attributes, and related words from DB."""
    with engine.connect() as conn:
        issues_df = pd.read_sql("""
            SELECT ci.issue_id, ci.clean_issue_text, ngt.attribute_id, ngt.is_related
            FROM combined_issues ci
            JOIN ground_truth ngt ON ci.issue_id = ngt.issue_id
        """, conn)

        attributes_df = pd.read_sql("""
            SELECT attribute_id, attribute, definition, related_words
            FROM quality_attributes
        """, conn)

    return issues_df, attributes_df
# End of fetch_data function

# Function to generate hypotheses based on the attribute, definition, and related words
# This function creates a set of hypotheses for each quality attribute.
# The hypotheses are designed to cover various aspects of the attribute, including direct references,
# definition-based references, related words, and question-style hypotheses.
# The hypotheses are structured to be compatible with NLI models, allowing them to evaluate the relevance
# of issue texts to the quality attributes.
# The function returns a list of hypothesis sets, where each set contains multiple hypotheses.
def generate_hypothesis_sets(attribute, definition, related_words):
    related_words_list = [w.strip() for w in related_words.split(',') if w.strip()]
    related_words_text = ', '.join(related_words_list)
    first_word = related_words_list[0] if related_words_list else attribute

    return [
        # Set 0: Several aspects
        [
            f"The text is related to the quality attribute '{attribute}'.",
            f"The text is about {attribute} as defined by: {definition}.",
            f"The text is about one or some of these related words: {related_words_text}.",
            f"The text mentions {attribute} or is relevant to {attribute}.",
            f"The text concerns aspects like: {related_words_text}."
        ],
                
        # Set 1: Direct / literal
        [
            f"This text is related to the quality attribute '{attribute}'.",
            f"This text discusses '{attribute}' or is relevant to {attribute}.",
            f"This text focuses on '{first_word}', a key aspect of '{attribute}'."
        ],

        # Set 2: Definition-based
        [
            f"This text is related to the quality attribute '{attribute}'.",
            f"This text is about '{attribute}' as defined by: {definition}.",
            f"The content aligns with the definition of {attribute}: {definition}.",
        ],

        # Set 3: Related words emphasis
        [
            f"This text is related to the quality attribute '{attribute}'.",
            f"This text is about one or some of these related words: {related_words_text}.",
            f"The content discusses aspects like {related_words_text}.",
        ],

        # Set 4: Question-style (NLI supports questions as hypotheses)
        [
            f"Is this text about {attribute}?",
            f"Is this text related to topics like {related_words_text}?",
            f"Does this concern {attribute} according to the definition: {definition}?",
        ],

        # Set 5: Instruction-style
        [
            f"Determine whether the text concerns {attribute}.",
            f"Check if this text is related to the quality attribute '{attribute}'.",
            f"Check if this text speaks about a concern that is defined like '{definition}'.",
        ],

        # Set 5: Paraphrased & slightly verbose
        [
            f"This text raises concerns typically associated with '{attribute}', which includes topics like {related_words_text}.",
            f"This text appears to be discussing '{attribute}', possibly referencing: {related_words_text}.",
        ]
    ]
# End of generate_hypothesis_sets function

# Function to split text into chunks based on the specified strategy
# This function takes a text and splits it into smaller chunks based on the specified strategy.
# The strategies include splitting by sentences, paragraphs, or newlines.
# The function ensures that each chunk does not exceed the maximum token limit specified by the tokenizer.
# It returns a list of text chunks that are suitable for processing by the NLI model.
def split_text_into_chunks(text, tokenizer, max_tokens=512, strategy="sentence"):
    if strategy == "paragraph":
        chunks = text.split("\n\n")
    elif strategy == "newline":
        chunks = text.split("\n")
    else:  # sentence-based (default)
        split_pattern = re.compile(r'(?<=[\.\n])\s+')
        chunks = split_pattern.split(text)

    result_chunks = []
    current = ""
    for s in chunks:
        temp = (current + " " + s).strip()
        if len(tokenizer(temp)['input_ids']) <= max_tokens:
            current = temp
        else:
            if current:
                result_chunks.append(current)
            current = s
    if current:
        result_chunks.append(current.strip())
    return result_chunks
# End of split_text_into_chunks function

# Function to execute NLI on hypotheses against issue texts
# This function takes a list of hypotheses and a list of issue texts, tokenizes them, and runs the NLI model.
# It processes the issue texts in chunks based on the specified chunking strategy and maximum token limit.
# For each chunk, it computes the entailment scores for each hypothesis using the NLI model.
# The scores are aggregated based on the specified aggregation logic (e.g., max or mean).
# The function returns a list of entailment scores for each hypothesis across all issue texts.
def execute_nli(hypotheses, issue_texts, tokenizer, model, max_tokens=512, chunking_strategy="sentence", aggregation_logic="max"):
    entailment_scores = []
    model.eval()
    with torch.no_grad():
        for issue in tqdm(issue_texts, desc="Evaluating Issues"):
            chunks = split_text_into_chunks(issue, tokenizer, max_tokens=max_tokens, strategy=chunking_strategy)

            # Store scores per hypothesis
            all_scores = [[] for _ in hypotheses]  # all_scores[i] = list of scores for hypothesis i

            for chunk in chunks:
                for i, hypo in enumerate(hypotheses):
                    inputs = tokenizer(chunk, hypo, truncation=True, max_length=max_tokens, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    logits = torch.softmax(outputs.logits, dim=-1)
                    entailment_index = 2 if logits.shape[-1] >= 3 else 1
                    score = logits[0][entailment_index].item()
                    all_scores[i].append(score)

            # Apply aggregation
            final_scores = []
            for scores in all_scores:
                if not scores:
                    final_scores.append(0.0)
                elif aggregation_logic == "mean":
                    final_scores.append(sum(scores) / len(scores))
                else:  # default to "max"
                    final_scores.append(max(scores))

            entailment_scores.append(final_scores)

    return entailment_scores
# End of execute_nli function

# Function to apply heuristics based on entailment scores
# This function takes the entailment scores and applies heuristics to determine the final labels.
# It uses predefined thresholds to classify the scores into binary labels (1.0 for related, 0.0 for not related).
# The thresholds are applied in order, and the first score that meets or exceeds a threshold results in a label of 1.0.
# If none of the scores meet the thresholds, the label is set to 0.0.
# The function returns a list of labels corresponding to each set of scores.
def apply_heuristics(entailment_scores, thresholds=[0.60, 0.55, 0.50, 0.45, 0.40]):
    labels = []
    for score_list in entailment_scores:
        sorted_scores = sorted(score_list, reverse=True)
        for i, threshold in enumerate(thresholds):
            if i < len(sorted_scores) and sorted_scores[i] >= threshold:
                labels.append(1.0)
                break
        else:
            labels.append(0.0)
    return labels
# End of apply_heuristics function

# Function to calculate precision, recall, specificity, accuracy, and F1 score
# This function takes the predictions and ground truth labels and calculates various evaluation metrics.
# It computes true positives (TP), false positives (FP), false negatives (FN), and true negatives (TN).
# Based on these counts, it calculates precision, recall, specificity, accuracy, and F1 score.
# The metrics are returned as a tuple.
def calculate_metrics(predictions, ground_truth):
    TP = sum((p == 1 and g == 1) for p, g in zip(predictions, ground_truth))
    FP = sum((p == 1 and g == 0) for p, g in zip(predictions, ground_truth))
    FN = sum((p == 0 and g == 1) for p, g in zip(predictions, ground_truth))
    TN = sum((p == 0 and g == 0) for p, g in zip(predictions, ground_truth))

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    specificity = TN / (TN + FP) if (TN + FP) else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    return precision, recall, specificity, accuracy, f1_score
# End of calculate_metrics function

# Function to ensure the NLI metrics table exists
# This function checks if the table `nli_metrics_results` exists in the database.
def ensure_nli_metrics_table_exists():
    with engine.begin() as conn:
        inspector = inspect(conn)
        if "nli_metrics_results" not in inspector.get_table_names():
            print("🔧 Creating table: nli_metrics_results")

            conn.execute(text("""
                CREATE TABLE nli_metrics_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    attribute_id INTEGER NOT NULL,
                    attribute TEXT NOT NULL,
                    model TEXT NOT NULL,
                    chunking_strategy TEXT,
                    threshold_config TEXT,
                    hypothesis_version TEXT,
                    max_tokens INTEGER,
                    aggregation_logic TEXT,
                    duration_seconds REAL,
                    precision REAL,
                    recall REAL,
                    specificity REAL,
                    accuracy REAL,
                    f1_score REAL,
                    CONSTRAINT FK_nli_metrics_results_quality_attributes
                        FOREIGN KEY (attribute_id) REFERENCES quality_attributes(attribute_id)
                        ON DELETE CASCADE ON UPDATE CASCADE
                );
            """))

            conn.execute(text("""
                CREATE UNIQUE INDEX nli_metrics_results_unique_idx
                ON nli_metrics_results (
                    attribute_id, model,
                    chunking_strategy, threshold_config, hypothesis_version,
                    max_tokens, aggregation_logic
                );
            """))
        else:
            print("✅ Table already exists: nli_metrics_results")
# End of ensure_nli_metrics_table_exists function

# Function to store metrics in the database
# This function inserts or updates the evaluation metrics in the database.
# It uses an SQL INSERT statement with ON CONFLICT to handle duplicate entries based on the attribute_id, model, chunking_strategy, threshold_config, hypothesis_version, max_tokens, and aggregation_logic.
# If a conflict occurs, it updates the existing record with the new metrics.
# The function takes various parameters including attribute_id, model_name, chunking_strategy, threshold_config, hypothesis_version, max_tokens, aggregation_logic, and the calculated metrics (precision, recall, specificity, accuracy, f1_score, duration_seconds).
# The duration_seconds parameter is used to store the time taken for the evaluation.
# The function uses SQLAlchemy's text module to execute the SQL command with the provided parameters.
def store_metrics(
    attribute_id, attribute, model_name,
    chunking_strategy, threshold_config, hypothesis_version,
    max_tokens, aggregation_logic,
    precision, recall, specificity, accuracy, f1_score,
    duration_seconds
):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO nli_metrics_results (
                attribute_id, attribute, model,
                chunking_strategy, threshold_config, hypothesis_version,
                max_tokens, aggregation_logic,
                duration_seconds,
                precision, recall, specificity, accuracy, f1_score
            )
            VALUES (
                :attribute_id, :attribute, :model,
                :chunking_strategy, :threshold_config, :hypothesis_version,
                :max_tokens, :aggregation_logic,
                :duration_seconds,
                :precision, :recall, :specificity, :accuracy, :f1_score
            )
            ON CONFLICT(attribute_id, model, chunking_strategy, threshold_config, hypothesis_version, max_tokens, aggregation_logic)
            DO UPDATE SET
                attribute = excluded.attribute,
                duration_seconds = excluded.duration_seconds,
                precision = excluded.precision,
                recall = excluded.recall,
                specificity = excluded.specificity,
                accuracy = excluded.accuracy,
                f1_score = excluded.f1_score;
        """), {
            'attribute_id': attribute_id,
            'attribute': attribute,
            'model': model_name,
            'chunking_strategy': chunking_strategy,
            'threshold_config': str(threshold_config),
            'hypothesis_version': hypothesis_version,
            'max_tokens': max_tokens,
            'aggregation_logic': aggregation_logic,
            'duration_seconds': duration_seconds,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'accuracy': accuracy,
            'f1_score': f1_score
        })
# End of store_metrics function

# Main function to run the evaluation
# This function orchestrates the entire evaluation process.
# It fetches the data, iterates over each NLI model, and evaluates it against the quality attributes.
# For each model, it generates hypotheses, runs NLI, applies heuristics, calculates metrics, and stores the results in the database.
# The function handles different configurations such as chunking strategies, threshold configurations, max token limits, and aggregation logics.
# It prints the evaluation results for each attribute and configuration, including precision, recall, specificity, accuracy, and F1 score.
# The results are stored in the database for further analysis.
def evaluate_nlis():
    issues_df, attributes_df = fetch_data()

    for model_name in nli_models:
        print(f"Evaluating model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        
        for chunk_strategy in chunk_strategies:
            for thresholds in threshold_configs:
                for max_tokens in max_tokens_list:
                    for aggregation_logic in aggregation_logic_list:                        
                        config_id = (
                            f"{model_name} | Chunk: {chunk_strategy} | "
                            f"Thresholds: {thresholds} | MaxTokens: {max_tokens} | "
                            f"Aggregation: {aggregation_logic}"
                        )
                        print(f"\n📦 Running config: {config_id}")
                        for _, attr_row in attributes_df.iterrows():
                            attr_id = attr_row['attribute_id']
                            attr_name = attr_row['attribute']
                            definition = attr_row['definition']
                            related_words = attr_row['related_words']

                            # Filter relevant issues
                            relevant_issues = issues_df[issues_df['attribute_id'] == attr_id]
                            if relevant_issues.empty:
                                continue

                            issue_texts = relevant_issues['clean_issue_text'].tolist()
                            ground_truth = relevant_issues['is_related'].tolist()

                            # Generate hypotheses
                            
                            hypothesis_sets = generate_hypothesis_sets(attr_name, definition, related_words)

                            for hypothesis_version, hypotheses in enumerate(hypothesis_sets):
                            
                                start_time = time.time()

                                # Run NLI
                                entailment_scores = execute_nli(
                                    hypotheses, issue_texts, tokenizer, model,
                                    max_tokens=max_tokens,
                                    chunking_strategy=chunk_strategy,
                                    aggregation_logic=aggregation_logic
                                )

                                predictions = apply_heuristics(entailment_scores, thresholds)

                                # Calculate metrics
                                precision, recall, specificity, accuracy, f1_score = calculate_metrics(predictions, ground_truth)
                            
                                duration_seconds = round(time.time() - start_time, 3)  # ⏱️ Time spent per attribute config

                                print(f"🔎 Attribute: {attr_name}, "
                                    f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
                                    f"Specificity: {specificity:.4f}, Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}")

                                # Store results                                
                                store_metrics(
                                    attr_id, attr_name, model_name,
                                    chunk_strategy, thresholds, f"v{hypothesis_version}",
                                    max_tokens, aggregation_logic,
                                    precision, recall, specificity, accuracy, f1_score, duration_seconds
                                )    

                                print(f"✅ Stored: Model {model_name}, Attr {attr_name}, Hypo v{hypothesis_version}, F1: {f1_score:.4f}")

# End of main function                           
if __name__ == "__main__":
    ensure_nli_metrics_table_exists()
    evaluate_nlis()


# This script evaluates various NLI models on quality attributes using different configurations to find the best NLI model.
# It fetches data from a database, generates hypotheses, runs NLI, applies heuristics, calculates metrics, and stores results.
# It supports multiple models, chunking strategies, threshold configurations, max token limits, and aggregation logics.
# The results are stored in a database for further analysis.
# The script is designed to be modular and extensible for future enhancements.

# This script is part of the COSETO project for evaluating NLI models on software quality attributes.
# It is designed to be run in a Python environment with the necessary libraries installed.
# Ensure that the environment variables for the database connection are set correctly.