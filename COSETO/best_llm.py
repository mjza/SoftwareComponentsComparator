from builtins import print
import os
import torch
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from sqlalchemy import create_engine, text, inspect
from huggingface_hub import login
from dotenv import load_dotenv
from tqdm import tqdm
from time import time

# Load environment variables
load_dotenv()

# Database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

# Database connection
engine = create_engine(DATABASE_URL)

# Login to Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN, new_session=True)
print("Hugging Face login successful.")

# Set the device for PyTorch
# This will use GPU if available, otherwise fallback to CPU.
# Using GPU can significantly speed up the NLI model inference, especially for larger models.
# Ensure that the PyTorch library is installed with CUDA support if you want to use GPU.
# If CUDA is not available, it will automatically use CPU, which is slower but still functional
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# List of LLM models to evaluate
# These models are expected to be pre-trained and fine-tuned on relevant datasets for LLM tasks.
# The models are expected to be available on the Hugging Face Model Hub.
# The models are expected to be capable of generating text based on the provided prompts.
# The models are expected to be capable of understanding and processing the hypotheses generated from quality attributes.
# The models are expected to be capable of handling the tokenization and generation tasks efficiently.
# The models are expected to be capable of handling the context length required for the evaluation.
# The models are expected to be capable of generating responses that can be classified as "yes" or "no" based on the hypotheses.
# The models are expected to be capable of handling the input text and hypotheses in a structured manner.
models = [
    'meta-llama/Llama-3.1-8B-Instruct',
    'meta-llama/Meta-Llama-3-8B-Instruct',    
    'mistralai/Mistral-7B-Instruct-v0.3',
    'deepseek-ai/deepseek-coder-6.7b-instruct',
    'microsoft/Phi-3-mini-4k-instruct',
    'Qwen/Qwen1.5-7B-Chat', 
    'meta-llama/Llama-2-70b-chat-hf',
    'tiiuae/falcon-7b-instruct',
]

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


# Function to calculate the maximum number of text tokens for the LLM
# This function calculates the maximum number of tokens that can be used for the text input
# based on the tokenizer's model maximum length and the number of tokens used in the prompt.
def max_text_tokens(tokenizer, hypotheses):
    prompt = prompt = [
        {
            'role': 'system',
            'content': f"You are a scholarly researcher. You will receive an issue text and a list of quality attribute-related hypotheses. Respond with 'yes' if the text matches any of the hypotheses. Otherwise, respond with 'no'.\nHypotheses:\n - {'\n - '.join(hypotheses)}"
        },
        {
            'role': 'user',
            'content': f"Text: \nIs this text relevant?"
        }
    ]
    ids = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, truncation=False
    )
    
    # Leave room for “yes/no” + a tiny margin
    # It's a buffer to ensure the model has enough room to:
    # 1.Generate a response ("yes" or "no")
    # 2.Include any special tokens (like EOS)
    # 3.Avoid accidental truncation by pushing the prompt right up to the model's maximum length
    safety = 8

    # Clamp for LLaMA-3 weird tokenizer behavior
    max_len = min(getattr(tokenizer, "model_max_length", 4096), 8192)

    return max(32, max_len - len(ids) - safety)
# End of max_text_tokens function


# Function to split text into chunks based on token count
# This function takes a text input and splits it into smaller chunks based on the maximum number of tokens allowed.
# It uses the tokenizer to convert the text into tokens and then splits the tokens into chunks.
# Each chunk is then decoded back into text format.
def split_text_into_chunks(text, tokenizer, max_tokens):
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_ids = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_ids))
        start = end
    return chunks
# End of split_text_into_chunks function


# Function to truncate input_ids and attention_mask for context length
# This function ensures that the input_ids and attention_mask do not exceed the maximum context length of the model.
# It removes the overflow tokens from the beginning of the input_ids and attention_mask.
# The context length is defined by the model's max_position_embeddings.
# The function returns the truncated input_ids and attention_mask.    
def truncate_for_ctx(input_ids, attention_mask, ctx):
    overflow = input_ids.shape[-1] + 2 - ctx     # 2 → future reply
    if overflow > 0:
        input_ids      = input_ids[:, overflow:]
        attention_mask = attention_mask[:, overflow:]
    return input_ids, attention_mask
# End of truncate_for_ctx function


# Function to analyze issue text against hypotheses
# This function takes an issue text and a list of hypotheses, and uses the LLM to determine if the text is relevant to any of the hypotheses.
# It constructs a prompt for the LLM, which includes the hypotheses and the issue text.
# The LLM generates a response indicating whether the text matches any of the hypotheses.
def analyze_issue_text(text, hypotheses, model, tokenizer, max_tokens):
    prompt = prompt = [
        {
            'role': 'system',
            'content': f"You are a scholarly researcher. You will receive an issue text and a list of quality attribute-related hypotheses. Respond with 'yes' if the text matches any of the hypotheses. Otherwise, respond with 'no'.\nHypotheses:\n - {'\n - '.join(hypotheses)}"
        },
        {
            'role': 'user',
            'content': f"Text: {text}\nIs this text relevant?"
        }
    ]
    input_ids = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        padding=True  # ensure pad token used
    ).to(model.device)
    
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    input_ids, attention_mask = truncate_for_ctx(input_ids, attention_mask,
                                             model.config.max_position_embeddings)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2,  # Keeps output length short (just "yes"/"no")
        do_sample=False,   # Keeps generation deterministic; needed for binary ("yes"/"no") answers
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True).strip().lower()
# End of analyze_issue_text function

# Function to classify issue text based on hypotheses
# This function processes the issue text by splitting it into manageable chunks,
# analyzing each chunk with the LLM, and aggregating the results.
def classify_issue_text(text, hypotheses, model, tokenizer, max_tokens, num_trials=5):
    trial_votes = []

    for _ in range(num_trials):
        chunks = split_text_into_chunks(text, tokenizer, max_tokens)

        # A single trial is 'yes' if ANY chunk is 'yes'
        # because it might happen that the relevant data being in any chunk.
        trial_result = any(
            analyze_issue_text(chunk, hypotheses, model, tokenizer, max_tokens) == 'yes'
            for chunk in chunks
        )
        trial_votes.append("yes" if trial_result else "no")
    
    yes_count = trial_votes.count("yes")
    total = len(trial_votes)    

    # Aggregate the trial-level results
    result = {
        "majority_vote": 1.0 if Counter(trial_votes).most_common(1)[0][0] == "yes" else 0.0,
        "unanimous": 1.0 if all(v == "yes" for v in trial_votes) else 0.0,
        "first_yes": 1.0 if "yes" in trial_votes else 0.0,
        "trial_ratio_threshold": 1.0 if (yes_count / total) >= 0.6 else 0.0
    }
    return result
# End of classify_issue_text function

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


# Function to ensure the LLM metrics table exists
# This function checks if the table `llm_metrics_results` exists in the database.
def ensure_llm_metrics_table_exists():
    with engine.begin() as conn:
        inspector = inspect(conn)
        if "llm_metrics_results" not in inspector.get_table_names():
            print("🔧 Creating table: llm_metrics_results")

            conn.execute(text("""
                CREATE TABLE llm_metrics_results (
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
                    CONSTRAINT FK_llm_metrics_results_quality_attributes
                        FOREIGN KEY (attribute_id) REFERENCES quality_attributes(attribute_id)
                        ON DELETE CASCADE ON UPDATE CASCADE
                );
            """))

            conn.execute(text("""
                CREATE UNIQUE INDEX llm_metrics_results_unique_idx
                ON llm_metrics_results (
                    attribute_id, model,
                    chunking_strategy, threshold_config, hypothesis_version,
                    aggregation_logic
                );
            """))
        else:
            print("✅ Table already exists: llm_metrics_results")
# End of ensure_llm_metrics_table_exists function

# Function to check if a result already exists in the database
# This function checks if a result for the given combination of attribute_id, model_name, hypothesis_version,
# aggregation_logic, chunking_strategy, threshold_config, and max_tokens already exists in the `llm_metrics_results` table.
# It returns True if a result exists, otherwise False.
def check_if_exists(attribute_id, model_name, hypothesis_version,
                    aggregation_logic, chunking_strategy,
                    threshold_config):
    """
    Check if a result for the given combination already exists in the llm_metrics_results table.
    """
    with engine.begin() as conn:
        result = conn.execute(text("""
            SELECT 1
            FROM llm_metrics_results
            WHERE attribute_id = :attribute_id
              AND model = :model
              AND chunking_strategy = :chunking_strategy
              AND threshold_config = :threshold_config
              AND hypothesis_version = :hypothesis_version
              AND aggregation_logic = :aggregation_logic
            LIMIT 1
        """), {
            "attribute_id": attribute_id,
            "model": model_name,
            "chunking_strategy": chunking_strategy,
            "threshold_config": threshold_config,
            "hypothesis_version": hypothesis_version,
            "aggregation_logic": aggregation_logic
        })

        return result.fetchone() is not None
# End of check_if_exists function

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
            INSERT INTO llm_metrics_results (
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
            ON CONFLICT(attribute_id, model, chunking_strategy, threshold_config, hypothesis_version, aggregation_logic)
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

# Function to evaluate LLMs on quality attributes
# This function iterates over the list of LLM models and evaluates each model against the quality attributes.
# For each model, it loads the tokenizer and model, generates hypothesis sets for each attribute,
# and classifies the issue texts based on the hypotheses.
# It calculates the evaluation metrics (precision, recall, specificity, accuracy, F1 score)
# and stores the results in the database.
# The function uses the Hugging Face Transformers library to handle the model and tokenizer.
# It also uses the tqdm library for progress tracking during the evaluation.
def evaluate_llms():
    issues_df, attributes_df = fetch_data()
    chunking_strategy = "token"  # Stick to token chunking for LLMs
    threshold_config = "default"  # we don't have scores in LLM to define thresholds
    num_trials = 5

    for model_name in models:
        print(f"Evaluating {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if torch.cuda.is_available():
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            dtype = torch.float16
        else:
            dtype = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto", 
            trust_remote_code=True
        )
        
        for _, attr_row in attributes_df.iterrows():
            attr_id = attr_row['attribute_id']
            attr_name = attr_row['attribute']
            definition = attr_row['definition']
            related_words = attr_row['related_words']

            hypothesis_sets = generate_hypothesis_sets(attr_name, definition, related_words)

            for hypothesis_version, hypotheses in enumerate(hypothesis_sets):
                relevant_issues = issues_df[issues_df['attribute_id'] == attr_id]
                if relevant_issues.empty:
                    continue
                
                hypothesis_version_name = f"v{hypothesis_version}"
                aggregation_logics = ["majority_vote", "unanimous", "first_yes", "trial_ratio_threshold"]
                
                # Check for all aggregations before doing any processing
                all_done = True
                for aggregation_logic in aggregation_logics:
                    if not check_if_exists(
                        attribute_id=attr_id,
                        model_name=model_name,
                        hypothesis_version=hypothesis_version_name,
                        aggregation_logic=aggregation_logic,
                        chunking_strategy=chunking_strategy,
                        threshold_config=threshold_config
                    ):
                        all_done = False
                        break

                if all_done:
                    print(f"✅ Skipping: {attr_name} ({hypothesis_version_name}) – All aggregations already done")
                    continue
                
                texts = relevant_issues['clean_issue_text'].tolist()
                ground_truth = relevant_issues['is_related'].tolist()
                max_tokens = max_text_tokens(tokenizer, hypotheses)

                # Compute predictions (since at least one aggregation is missing)
                all_predictions = {
                    agg: [] for agg in aggregation_logics
                }
                
                start_time = time()
                for text in tqdm(texts, desc=f"{attr_name} ({hypothesis_version_name})"):
                    result_dict = classify_issue_text(
                        text, hypotheses, model, tokenizer, max_tokens, num_trials=num_trials
                    )
                    for agg in all_predictions:
                        all_predictions[agg].append(result_dict[agg])
                duration_seconds = time() - start_time

                for aggregation_logic, predictions in all_predictions.items():
                    precision, recall, specificity, accuracy, f1 = calculate_metrics(predictions, ground_truth)
                    print(f"{attr_name} ({hypothesis_version_name}) - {aggregation_logic}: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")

                    store_metrics(
                        attribute_id=attr_id,
                        attribute=attr_name,
                        model_name=model_name,
                        chunking_strategy=chunking_strategy,
                        threshold_config=threshold_config,
                        hypothesis_version=hypothesis_version_name,
                        max_tokens=max_tokens,
                        aggregation_logic=aggregation_logic,
                        precision=precision,
                        recall=recall,
                        specificity=specificity,
                        accuracy=accuracy,
                        f1_score=f1,
                        duration_seconds=duration_seconds
                    )
                    
# End of evaluate_llms function    

if __name__ == "__main__":
    ensure_llm_metrics_table_exists()
    evaluate_llms()
