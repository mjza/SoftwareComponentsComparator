"""
Quality Attribute to GitHub Issues Similarity Analysis using Phi-2

This module demonstrates how to use Microsoft's Phi-2 language model to find semantic
relationships between software quality attribute definitions and GitHub issues.

The core idea is to leverage Phi-2's deep understanding of technical content to create
meaningful embeddings that capture the essence of both quality definitions and issues,
then measure their similarity to identify relationships.

Author: [Your Name]
Date: April 2025
"""

import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import logging
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gc  # Garbage collection for memory management
from sqlalchemy import create_engine
from sqlalchemy.sql import text
import os
from dotenv import load_dotenv

# load environment parameters
load_dotenv()

# Configure logging to provide detailed runtime information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Phi2OptimizedAnalyzer:
    """
    A class that uses Phi-2 to analyze the semantic similarity between software
    quality attribute definitions and GitHub issues.

    The analyzer works by:
    1. Loading and preprocessing the data
    2. Converting texts to embeddings using Phi-2's hidden states
    3. Computing similarity scores between embeddings
    4. Performing sentiment analysis on the issues
    5. Generating a structured output with the relationships found

    Why Phi-2? It offers an excellent balance between:
    - Performance (faster than Phi-3, slower than SBERT)
    - Understanding (better technical understanding than SBERT)
    - Resource usage (moderate memory requirements)
    """
    
    def __init__(self, db_engine, model_name='microsoft/phi-2', similarity_threshold=0.3):
        """
        Initialize the analyzer with configuration parameters.

        Args:
            db_engine (): 
            model_name (str): The identifier for the Phi-2 model on Hugging Face
            similarity_threshold (float): Minimum similarity score to consider a match
                                        (0.3 means 30% semantic similarity)

        The similarity threshold is crucial: too low and you get many false positives,
        too high and you might miss valid relationships. 0.3 is a good starting point
        that can be adjusted based on your results.
        """        
        self.db_engine = db_engine
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        
        # Model and tokenizer will be loaded later for better memory management
        self.model = None
        self.tokenizer = None
        
        # VADER is specifically designed for social media text sentiment analysis
        # It handles technical jargon better than general sentiment tools
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Automatically select GPU if available, otherwise fall back to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        
    def initialize(self):
        self.load_model()
        
        # create resulting tables
        with self.db_engine.begin() as con:
            con.execute(text("""
                CREATE TABLE IF NOT EXISTS quality_attribute_analysis_v2 (
                    id SERIAL PRIMARY KEY,
                    project_id BIGINT,
                    attribute TEXT,
                    sentiment TEXT,
                    similarity_score FLOAT,
                    issue_id BIGINT,
                    reason TEXT
                );
            """))
            con.execute(text("""
                CREATE TABLE IF NOT EXISTS quality_attribute_analysis_tracker_v2 (
                    project_id     BIGINT PRIMARY KEY,
                    last_issue_id  BIGINT NOT NULL DEFAULT 0
                );
            """))

    
    def load_quality_attributes(self):
        logging.info("Loading quality attributes from DB...")
        self.quality_df = pd.read_sql("""
            SELECT DISTINCT attribute, definition
            FROM quality_attributes_v2
            WHERE definition IS NOT NULL
        """, self.db_engine)
        

    def track_last_issue_id(self, project_id: int, last_issue_id: int):
        # Convert to native Python int to avoid psycopg2 'can't adapt type' error
        project_id = int(project_id)
        last_issue_id = int(last_issue_id)
    
        with self.db_engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO quality_attribute_analysis_tracker_v2 (project_id, last_issue_id)
                VALUES (:pid, :iid)
                ON CONFLICT (project_id)
                DO UPDATE SET last_issue_id = EXCLUDED.last_issue_id
            """), {"pid": project_id, "iid": last_issue_id})


    def store_results(self, results_df: pd.DataFrame, project_id: int):
        if results_df.empty:
            return

        results_df['project_id'] = project_id  # attach project ID to each row

        # Reorder and select required columns only
        records = results_df[['project_id', 'attribute', 'sentiment', 'similarity_score', 'issue_id']].to_dict(orient='records')

        with self.db_engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO quality_attribute_analysis_v2 (project_id, attribute, sentiment, similarity_score, issue_id)
                    VALUES (:project_id, :attribute, :sentiment, :similarity_score, :issue_id)
                """),
                records
            )


    def analyze_batch(self, issues_df: pd.DataFrame) -> pd.DataFrame:
        if issues_df.empty:
            return pd.DataFrame()
        
        # Generate embeddings for issues
        logging.info("Computing embeddings for issues...")
        issue_embeddings = self.batch_get_embeddings(issues_df['combined_text'].tolist(), batch_size=8)
        
        # Generate embeddings for quality attributes
        logging.info("Computing embeddings for quality attributes...")
        quality_embeddings = self.batch_get_embeddings(self.quality_df['definition'].tolist(), batch_size=4)

        # Calculate similarity matrix
        # Cosine similarity measures the angle between vectors
        # Values range from 0 (orthogonal/unrelated) to 1 (identical)
        logging.info("Calculating similarity matrix...")
        similarity_matrix = cosine_similarity(quality_embeddings, issue_embeddings)
        
        results = []

        # Process results
        logging.info("Processing results...")
        for i, quality_row in self.quality_df.iterrows():
            # Get similarity scores for this quality attribute
            similarities = similarity_matrix[i]

            # Find indices where similarity exceeds threshold
            top_indices = np.where(similarities >= self.similarity_threshold)[0]

            # Create result entries for each match
            for idx in top_indices:
                issue_row = issues_df.iloc[idx]
                sentiment = self.analyze_sentiment(issue_row['combined_text'])

                results.append({
                    'attribute': quality_row['attribute'],
                    'reason': None,
                    'issue_id': issue_row['issue_id'],
                    'project_id': issue_row['project_id'],
                    'similarity_score': round(float(similarities[idx]), 4),
                    'sentiment': sentiment
                })

        return pd.DataFrame(results)
    

    def preprocess_issues_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        if batch_df.empty:
            return pd.DataFrame()

        batch_df['comment_text'] = batch_df['comment_text'].fillna('')
        batch_df['body_text'] = batch_df['body_text'].fillna('')
        batch_df['title'] = batch_df['title'].fillna('')

        batch_df['combined_text'] = (
            batch_df['title'] + ' ' +
            batch_df['body_text'] + ' ' +
            batch_df['comment_text']
        ).str.strip()

        return batch_df


    def process_all_projects(self):
        self.load_quality_attributes()
        
        project_ids = pd.read_sql(text("""
            SELECT i.project_id
            FROM issues i
            LEFT JOIN quality_attribute_analysis_tracker_v2 t
            ON i.project_id = t.project_id
            WHERE i.issue_id > COALESCE(t.last_issue_id, 0)
            GROUP BY i.project_id
            HAVING COUNT(i.issue_id) > 100
            ORDER BY COUNT(i.issue_id) DESC
        """), self.db_engine)["project_id"].tolist()

        for pid in project_ids:
            self.process_project(pid)


    def process_project(self, project_id: int, issue_batch_size: int = 100):
        conn = self.db_engine.connect()

        res = conn.execute(text("""
            SELECT last_issue_id FROM quality_attribute_analysis_tracker_v2
            WHERE project_id = :pid
        """), {"pid": project_id}).fetchone()

        last_issue_id = res[0] if res else 0
        logging.info(f"📂 Project {project_id}: starting from issue {last_issue_id}")

        while True:
            batch_df = pd.read_sql(text("""
                SELECT i.issue_id, i.project_id, i.title, i.body_text, i.state_reason,
                    STRING_AGG(c.body_text, '\n') AS comment_text
                FROM issues i
                LEFT JOIN comments c ON c.issue_id = i.issue_id
                WHERE i.project_id = :pid AND i.issue_id > :last_id
                AND (i.title IS NOT NULL OR i.body_text IS NOT NULL)
                GROUP BY i.issue_id
                ORDER BY i.issue_id
                LIMIT :batch_size
            """), conn, params={"pid": project_id, "last_id": last_issue_id, "batch_size": issue_batch_size})

            if batch_df.empty:
                logging.info(f"✅ Done with project {project_id}")
                break

            batch_df = self.preprocess_issues_batch(batch_df)
            results_df = self.analyze_batch(batch_df)
            self.store_results(results_df, project_id)  
            last_issue_id = batch_df["issue_id"].max()
            self.track_last_issue_id(project_id, last_issue_id)


    def load_model(self): # OK
        """
        Load the Phi-2 model and tokenizer with optimized settings.

        This method handles several important considerations:
        1. Memory management (using low_cpu_mem_usage)
        2. Precision selection (float16 for GPU, float32 for CPU)
        3. Tokenizer configuration (setting pad token)
        4. Model evaluation mode (disabling dropout for inference)

        The trust_remote_code=True parameter allows loading custom model code,
        which is necessary for Phi-2 as it includes custom implementations.
        """
        logging.info(f"Loading Phi-2 model: {self.model_name}")

        # Load tokenizer first (it's smaller and loads faster)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True  # Required for Phi-2's custom code
        )

        # Load model with optimized settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            # Use float16 on GPU for faster computation and lower memory usage
            # Use float32 on CPU as CPUs don't benefit from float16
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",       # Automatically distribute model across available devices
            low_cpu_mem_usage=True   # Minimize CPU memory usage during loading
        )

        # Many models don't have a pad token defined, so we use the EOS token
        # This prevents errors when tokenizing batches of different lengths
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Switch to evaluation mode (disables dropout and other training-specific behaviors)
        self.model.eval()

        logging.info("Model loaded successfully")


    def get_embedding(self, text, max_length=512): # OK
        """
        Convert a text string into an embedding vector using Phi-2.

        Unlike traditional embedding models, we use Phi-2's hidden states to create
        embeddings. This gives us rich semantic representations that understand
        technical content better than word-level embeddings.

        Args:
            text (str): The input text to embed
            max_length (int): Maximum number of tokens to process

        Returns:
            numpy.ndarray: The embedding vector (flattened from the model's hidden state)

        How it works:
        1. Tokenize the text (convert to token IDs)
        2. Pass through the model to get hidden states
        3. Take the last layer's hidden state
        4. Average across all tokens (mean pooling) to get a fixed-size vector
        """
        # Truncate very long texts before tokenization to avoid memory issues
        # We take the first 1024 characters as a reasonable compromise
        text_truncated = text[:1024]

        # Tokenize the text
        inputs = self.tokenizer(
            text_truncated,
            return_tensors="pt",      # Return PyTorch tensors
            truncation=True,          # Truncate to max_length if needed
            max_length=max_length,    # Maximum number of tokens
            padding=True              # Pad shorter sequences
        )

        # Move all tensors to the appropriate device (GPU or CPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model outputs without computing gradients (inference mode)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True  # We need the hidden states for embeddings
            )

            # Extract the last layer's hidden state
            # Shape: [batch_size, sequence_length, hidden_size]
            hidden_states = outputs.hidden_states[-1]

            # Average across all tokens to get a single vector per text
            # This is called "mean pooling" and gives equal weight to all tokens
            pooled_output = hidden_states.mean(dim=1)

        # Convert from PyTorch tensor to numpy array and move to CPU
        return pooled_output.cpu().numpy()

    
    def batch_get_embeddings(self, texts, batch_size=8): #OK
        """
        Process multiple texts in batches to manage memory efficiently.

        Processing texts one by one is slow, but processing all at once can cause
        out-of-memory errors. Batch processing provides the optimal balance.

        Args:
            texts (list): List of text strings to embed
            batch_size (int): Number of texts to process at once

        Returns:
            numpy.ndarray: Matrix of embeddings, one row per text

        Memory management strategy:
        1. Process texts in small batches
        2. Periodically clear GPU cache
        3. Force garbage collection to free memory
        """
        embeddings = []

        # Process texts in batches with progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []

            # Process each text in the batch
            for text in batch_texts:
                embedding = self.get_embedding(text)
                batch_embeddings.append(embedding)

            embeddings.extend(batch_embeddings)

            # Periodically clear memory to prevent accumulation
            if i % (batch_size * 5) == 0:
                gc.collect()  # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear GPU memory cache

        # Stack all embeddings into a single matrix
        return np.vstack(embeddings)

    
    def analyze_sentiment(self, text): #OK
        """
        Analyze the sentiment of a text using VADER.

        VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically
        attuned to sentiments expressed in social media and technical text.

        Args:
            text (str): The text to analyze

        Returns:
            str: '+' for positive, '-' for negative, '0' for neutral

        The compound score from VADER ranges from -1 (most negative) to +1 (most positive).
        We use thresholds of ±0.05 to classify sentiment, which helps avoid false
        positives/negatives for borderline cases.
        """
        scores = self.sentiment_analyzer.polarity_scores(str(text))

        # Compound score is a normalized, weighted composite score
        if scores['compound'] > 0:
            return '+'
        elif scores['compound'] < 0:
            return '-'
        else:
            return '0'


# Entry point when running as a script
if __name__ == "__main__":
    # Create DB engine
    # Connect to the database
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)

    # Create analyzer instance with default parameters
    analyzer = Phi2OptimizedAnalyzer(
        db_engine=engine,
        model_name="microsoft/phi-2",
        similarity_threshold=0.3
    )

    # Run the analysis
    analyzer.initialize()
    analyzer.process_all_projects()

