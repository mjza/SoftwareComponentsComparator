import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
PAGE_SIZE = int(os.getenv('PAGE_SIZE', 100))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_nli_model = os.getenv('BEST_NLI_MODEL')

tokenizer = AutoTokenizer.from_pretrained(best_nli_model)
model = AutoModelForSequenceClassification.from_pretrained(best_nli_model)
model.to(device)

def generate_hypotheses(attribute, definition, related_words):
    """
    Generates domain-specific hypotheses using attribute, definition, and related words.
    """
    related_words_text = ', '.join(related_words)
    hypotheses = [
        f"The text is related to the quality attribute '{attribute}'.",
        f"The text is about {attribute} as defined by: {definition}.",
        f"The text is about one of these related words: {related_words_text}.",
        f"The text mentions {attribute} or is relevant to {attribute}.",
        f"The text concerns aspects like: {related_words_text}."
    ]
    return hypotheses


def apply_heuristics(entailment_scores):
    results = []

    for scores in entailment_scores:
        sorted_scores = sorted(scores, reverse=True)
        candidate_scores = []

        # Collect candidate scores from all threshold-relevant positions
        for i in [0, 1, 2, 3, 4]:
            if len(sorted_scores) > i:
                candidate_scores.append(sorted_scores[i])

        top_score = max(candidate_scores) if candidate_scores else 0.0

        # Apply thresholds
        if (
            (len(sorted_scores) > 0 and sorted_scores[0] >= 0.90) or
            (len(sorted_scores) > 1 and sorted_scores[1] >= 0.85) or
            (len(sorted_scores) > 2 and sorted_scores[2] >= 0.80) or
            (len(sorted_scores) > 3 and sorted_scores[3] >= 0.75) or
            (len(sorted_scores) > 4 and sorted_scores[4] >= 0.70)
        ):
            label = 'maybe-relevant'
        else:
            label = 'maybe-not-relevant'

        results.append((label, top_score))

    return results


def chunk_text(text, tokenizer, max_tokens=512):
    """
    Splits text into smaller chunks that fit within the model's token limit.
    Prefers splitting at newlines or periods.
    """
    tokens = tokenizer.encode(text, truncation=False)
    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    current_chunk = ""
    for sentence in text.split('\n'):
        if len(tokenizer.encode(current_chunk + sentence, truncation=False)) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # fallback: split on period
                subsentences = sentence.split('.')
                for sub in subsentences:
                    if sub.strip():
                        tokenized = tokenizer.encode(sub, truncation=False)
                        if len(tokenized) > max_tokens:
                            # hard truncation fallback
                            chunks.append(tokenizer.decode(tokenized[:max_tokens]))
                        else:
                            chunks.append(sub.strip())
                current_chunk = ""
        else:
            current_chunk += sentence + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def execute_nli(hypotheses, texts):
    entailment_scores = []
    for text in texts:
        if not isinstance(text, str):
            entailment_scores.append([0.0] * len(hypotheses))
            continue

        chunks = chunk_text(text, tokenizer)
        hypo_scores = []

        for hypo in hypotheses:
            max_score = 0.0  # OR logic: use max entailment across chunks
            for chunk in chunks:
                inputs = tokenizer(chunk, hypo, truncation=True, max_length=512, return_tensors="pt")
                outputs = model(**inputs.to(device))
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                entailment_prob = probs[0].item()  # Class 0 = entailment
                max_score = max(max_score, entailment_prob)
            hypo_scores.append(max_score)
        entailment_scores.append(hypo_scores)
    return entailment_scores


def main():
    engine = create_engine(DATABASE_URL)
    conn = engine.connect()

    # Read quality attributes
    query_attrs = text("SELECT attribute_id, attribute, definition, related_words FROM quality_attributes")
    quality_df = pd.read_sql(query_attrs, conn)

    for _, row in quality_df.iterrows():
        attribute_id = row['attribute_id']
        attribute = row['attribute']
        definition = row['definition']
        related_words = eval(row['related_words'])  # List from string

        # Pre-filter issues with simple keyword search
        conditions = [f"clean_issue_text LIKE '%{attribute}%'"]
        for word in related_words:
            conditions.append(f"clean_issue_text LIKE '%{word}%'")
        where_clause = " OR ".join(conditions)
        
        # Determine total number of matching issues
        count_query = f"SELECT COUNT(*) FROM combined_issues WHERE {where_clause}"
        total_issues = pd.read_sql(count_query, conn).iloc[0, 0]
        print(f"🔍 Found {total_issues} matching issues for attribute '{attribute}'.")

        offset = 0
        while offset < total_issues:
            query_issues = f"""
                SELECT issue_id, clean_issue_text, project_id
                FROM combined_issues
                WHERE {where_clause}
                ORDER BY issue_id
                LIMIT {PAGE_SIZE}
                OFFSET {offset}
            """
            issues_df = pd.read_sql(query_issues, conn)

            if issues_df.empty:
                break

            # NLI step
            hypotheses = generate_hypotheses(attribute, definition, related_words)
            entailment_scores = execute_nli(hypotheses, issues_df['clean_issue_text'])
            label_score_tuples = apply_heuristics(entailment_scores)
            labels = [x[0] for x in label_score_tuples]
            
            # Convert each tuple to JSON string
            json_results = [json.dumps({'label': label, 'score': score}) for label, score in label_score_tuples]

            result_df = issues_df.copy()
            result_df['attribute_id'] = attribute_id
            result_df['nli_label'] = labels
            result_df['nli_json'] = json_results
            
            # Remove 'clean_issue_text' column before saving
            result_df = result_df.drop(columns=['clean_issue_text'])
            
            with engine.begin() as connection:
                # Save results
                result_df.to_sql('nli_results', connection, if_exists='append', index=False)
                print(f"📦 Stored {len(result_df)} results for attribute '{attribute}' (offset {offset}).")

            offset += PAGE_SIZE

    conn.close()
    print("✅ All processing complete.")

if __name__ == "__main__":
    main()
