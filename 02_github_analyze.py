import pandas as pd
import numpy as np
import torch
import re
import os
import multiprocessing
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import argparse
from transformers import AutoTokenizer,AutoModel,pipeline
from datasets import Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import defaultdict
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# load environment parameters
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)


# Class for analyzing software quality attributes
class QualityAttributeAnalyzer:
    def __init__(self,
                 similarity_threshold=0.05,
                 batch_size=32,
                 use_gpu=True,
                 parallel=False,
                 max_workers=None,
                 embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                 sentiment_model='distilbert-base-uncased-finetuned-sst-2-english',
                 max_matches_per_project=1000,
                 sample_matches=True):
    
        self.similarity_threshold=similarity_threshold
        self.batch_size=batch_size
        self.parallel=parallel
        self.max_workers=max_workers or max(1,multiprocessing.cpu_count()-1)
        self.device='cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.max_matches_per_project=max_matches_per_project
        self.sample_matches=sample_matches

        logger.info(f"Using device: {self.device}")
        logger.info(f"Using threshold: {self.similarity_threshold}")

        # Load embedding and sentiment models
        self.tokenizer=AutoTokenizer.from_pretrained(embedding_model)
        self.model=AutoModel.from_pretrained(embedding_model).to(self.device)

        self.sentiment_tokenizer=AutoTokenizer.from_pretrained(sentiment_model)
        self.sentiment_model=AutoModel.from_pretrained(sentiment_model).to(self.device)
        self.sentiment_analyzer=pipeline(
            'sentiment-analysis',
            model=sentiment_model,tokenizer=self.sentiment_tokenizer,
            device=0 if self.device=='cuda:0' else -1,
            batch_size=batch_size,
            max_length=512,
            truncation=True
        )

        # Predefined data structures
        self.w2v_scores={}
        self.similar_word_embeddings=None
        self.similar_word_index={}
        self.similar_word_to_criteria={}
        self.quality_dict={}
        self.reason_cache={}

        # Simple positive/negative words for override heuristics
        self.pos_words=["good","great","excellent","benefit","like","love","improve","better","fix","solve","easy","useful","solved","success","fast"]
        self.neg_words=["bad","poor","issue","bug","problem","difficult","fail","crash","error","slow","breaks","broken","missing","cannot","wrong"]

    # Additional methods go here (check_valid_text, determine_content_type, etc.)
    
    def check_valid_text(self, text):
        """
        Checks whether the input text is valid for further processing.

        Criteria for invalid text:
        - Not a string or empty/whitespace-only string
        - Contains more than 5 [UNK] tokens (likely low-quality text from tokenizer)
        - More than 30% of characters are non-ASCII (possibly gibberish or non-English)

        Args:
            text (str): The text to be validated.

        Returns:
            bool: True if the text is valid, False otherwise.
        """
        # Check if text is a string and not empty
        if not isinstance(text, str) or not text.strip():
            return False

        # Count the number of unknown tokens (e.g., [UNK])
        unk_count = text.count("[UNK]")
        if unk_count > 5:
            return False

        # Check proportion of non-ASCII characters
        non_ascii = len([c for c in text if ord(c) > 127])
        if non_ascii > len(text) * 0.3:
            return False

        # Passed all checks, text is valid
        return True
    

    def determine_content_type(self, text):
        """
        Classifies the type of content in a given text based on keyword presence.

        Categories:
        - bug_fix: if related to bugs, errors, or issues
        - feature: if suggesting or describing new functionality
        - dependency: if related to library or version updates
        - documentation: if related to documentation or examples
        - general: if no specific category matches

        Args:
            text (str): The input text to classify.

        Returns:
            str: The detected content type (one of the above categories).
        """
        text_lower = text.lower()

        if any(w in text_lower for w in ["fix", "issue", "bug", "crash", "error", "problem", "vulnerability"]):
            return "bug_fix"
        elif any(w in text_lower for w in ["add", "feature", "implement", "new", "enhancement", "request"]):
            return "feature"
        elif any(w in text_lower for w in ["bump", "update", "upgrade", "dependency", "version"]):
            return "dependency"
        elif any(w in text_lower for w in ["doc", "documentation", "example", "guide", "manual"]):
            return "documentation"
        else:
            return "general"
        
    
    def check_context_relevance(self, text, quality_attr):
        """
        Checks whether a given text is contextually relevant to a specific quality attribute.

        Relevance is determined by:
        - Direct match of the full quality attribute string (case insensitive)
        - Partial match if the attribute consists of multiple words (e.g., "response time")

        Args:
            text (str): The text to analyze.
            quality_attr (str): The quality attribute to check for.

        Returns:
            bool: True if the attribute is mentioned directly or partially in the text.
        """
        qa_lower = quality_attr.lower()
        text_lower = text.lower()

        # Direct match
        if qa_lower in text_lower:
            return True

        # Partial word match (e.g., individual words of a multi-word attribute)
        words = qa_lower.split()
        if len(words) > 1:
            for word in words:
                if len(word) > 3 and word in text_lower:
                    return True

        return False
    
    
    def override_sentiment(self, text, model_sentiment, confidence):
        """
        Adjusts the sentiment prediction using heuristic keyword matching.

        If there is a strong presence of known positive or negative words that
        contradict the model's prediction and the model's confidence is low,
        the sentiment may be overridden.

        Args:
            text (str): The original input text.
            model_sentiment (str): The model's predicted sentiment ('+' or '-').
            confidence (float): The model's confidence score (0 to 1).

        Returns:
            tuple: (corrected sentiment, adjusted confidence)
        """
        text_lower = text.lower()

        # Count positive and negative word occurrences
        pos_count = sum(1 for w in self.pos_words if w in text_lower)
        neg_count = sum(1 for w in self.neg_words if w in text_lower)

        # Heuristic overrides if positive words dominate but model predicted negative
        if pos_count > neg_count * 2 and confidence < 0.9 and model_sentiment == '-':
            return '+', 0.7

        # Heuristic overrides if negative words dominate but model predicted positive
        if neg_count > pos_count * 2 and confidence < 0.9 and model_sentiment == '+':
            return '-', 0.7

        # No override needed
        return model_sentiment, confidence    


    def extract_meaningful_context(self, text, quality_attr):
        """
        Extracts the most relevant and meaningful context from the given text for a specific quality attribute.

        Prioritizes sentences that explicitly mention the attribute, then uses fallback heuristics:
        - Short sentences with strong positive/negative terms
        - Content-type-based keyword matching
        - Title (if available)
        - First 30 words as last resort

        Heuristics:
        - Prefers short, relevant sentences containing the quality attribute
        - Falls back to content-type-based keyword matching
        - Falls back to title or truncated version of the text if nothing is found

        Args:
            text (str): Raw text to extract context from
            quality_attr (str): Target quality attribute to focus on

        Returns:
            str: A concise and relevant context string
        """
        if not self.check_valid_text(text):
            return "Content not available or contains invalid characters"

        # Clean text
        clean_text = re.sub(r'\[CLS\]|\[SEP\]', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Try to extract title
        title_match = re.search(r'title:\s*([^\n]+)', clean_text, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else None
        if title and len(title) > 150:
            title = title[:150] + "..."

        # Determine content type for fallback later
        content_type = self.determine_content_type(clean_text)

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        qa_lower = quality_attr.lower()
        qa_words = qa_lower.split()
        qa_sentences = []

        # Look for relevant short sentences
        for s in sentences:
            if len(s) > 200:
                continue
            if len(s.split()) < 3:
                continue
            s_lower = s.lower()
            if qa_lower in s_lower:
                qa_sentences.append(s)
            elif len(qa_words) > 1 and any(w in s_lower for w in qa_words if len(w) > 3):
                qa_sentences.append(s)

        if qa_sentences:
            best_sentences = sorted(
                qa_sentences,
                key=lambda s: sum(1 for w in self.pos_words + self.neg_words if w in s.lower()),
                reverse=True
            )[:2]
            return " ".join(best_sentences)

        # Fallback: based on content type keywords
        type_keywords = {
            "bug_fix": ["fix", "issue", "bug", "problem"],
            "feature": ["add", "feature", "implement", "new"],
            "dependency": ["bump", "update", "upgrade", "dependency"],
            "documentation": ["doc", "guide", "example"]
        }
        kw = type_keywords.get(content_type, [])
        context_sentences = [
            s for s in sentences
            if len(s.split()) >= 3 and len(s) < 200 and any(w in s.lower() for w in kw)
        ][:2]
        if context_sentences:
            return " ".join(context_sentences)

        # Fallback: use title
        if title:
            return title

        # Fallback: use first 30 words of the cleaned text
        brief_text = re.sub(r'\s+', ' ', clean_text).strip()
        return " ".join(brief_text.split()[:30]) + "..."    
    
    
    def generate_reason(self, text, qa, model_sentiment, confidence):
        """
        Constructs a human-readable explanation for why a sentiment was assigned to a quality attribute mention.

        Steps:
        - Returns a fallback message if the text is invalid
        - Optionally overrides the model sentiment using keyword-based heuristics
        - Checks if the content is relevant to the quality attribute
        - Extracts a relevant context snippet from the text
        - Returns a sentence that includes sentiment, confidence, type of content, and context

        Args:
            text (str): The original input text
            qa (str): The quality attribute in question
            model_sentiment (str): The original sentiment predicted by the model ('+' or '-')
            confidence (float): The confidence of the model's prediction

        Returns:
            str: A formatted string describing the sentiment reasoning
        """
        # Return early if text is invalid
        if not self.check_valid_text(text):
            return f"{'Positive' if model_sentiment == '+' else 'Negative'} sentiment ({confidence:.2f}) about {qa}: Content not available or contains invalid characters"

        # Determine content type and potentially override sentiment
        content_type = self.determine_content_type(text)
        sentiment, adj_confidence = self.override_sentiment(text, model_sentiment, confidence)
        sentiment_label = "Positive" if sentiment == '+' else "Negative"

        # Check contextual relevance and extract supporting context
        is_relevant = self.check_context_relevance(text, qa)
        context = self.extract_meaningful_context(text, qa)

        # If the content is not clearly related to the quality attribute
        if not is_relevant:
            return f"{sentiment_label} sentiment ({adj_confidence:.2f}) about {qa}: Content may indirectly relate to {qa.lower()}. {context}"

        # Define sentence patterns per content type and sentiment
        content_prefix = {
            "bug_fix": f"{sentiment_label} sentiment ({adj_confidence:.2f}) about {qa}: {'Fixed issue improving' if sentiment == '+' else 'Problem affecting'} {qa.lower()}.",
            "feature": f"{sentiment_label} sentiment ({adj_confidence:.2f}) about {qa}: {'Feature enhancing' if sentiment == '+' else 'Feature request for'} {qa.lower()}.",
            "dependency": f"{sentiment_label} sentiment ({adj_confidence:.2f}) about {qa}: {'Dependency update improving' if sentiment == '+' else 'Dependency issue affecting'} {qa.lower()}.",
            "documentation": f"{sentiment_label} sentiment ({adj_confidence:.2f}) about {qa}: {'Documentation clarifying' if sentiment == '+' else 'Documentation needed for'} {qa.lower()}.",
            "general": f"{sentiment_label} sentiment ({adj_confidence:.2f}) about {qa}: {'Content highlights good' if sentiment == '+' else 'Content indicates issues with'} {qa.lower()}."
        }

        # Combine prefix and context for final output
        return f"{content_prefix.get(content_type, content_prefix['general'])} {context}"

    
    def batch_sentiment_analysis(self, texts, quality_attrs):
        """
        Performs batch sentiment analysis on a list of texts with associated quality attributes.

        For each text:
        - Validates and cleans it
        - Truncates to 512 characters if necessary
        - Uses a preloaded sentiment pipeline to assign sentiment
        - Generates a human-readable reason for the sentiment

        Args:
            texts (List[str]): List of text inputs
            quality_attrs (List[str]): Corresponding quality attributes

        Returns:
            Tuple[List[str], List[str]]: List of sentiment labels ('+' or '-') and reasoning strings
        """
        if not texts:
            return [], []

        data = []
        for text, qa in zip(texts, quality_attrs):
            if not self.check_valid_text(text):
                clean_text = "Invalid content"
            else:
                clean_text = re.sub(r'\[CLS\]|\[SEP\]', '', text)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                if len(clean_text) > 512:
                    clean_text = clean_text[:512]

            data.append({"text": clean_text, "quality_attr": qa})

        # Create dataset and run sentiment analysis
        dataset = Dataset.from_list(data)
        texts_to_process = [d["text"] for d in data]
        results = self.sentiment_analyzer(texts_to_process, batch_size=self.batch_size)

        sentiments = []
        reasons = []

        for i, (result, qa) in enumerate(zip(results, quality_attrs)):
            sentiment = '+' if result['label'] == 'POSITIVE' else '-'
            confidence = result['score']
            reason = self.generate_reason(texts_to_process[i], qa, sentiment, confidence)

            sentiments.append(sentiment)
            reasons.append(reason)

        return sentiments, reasons
    

    def get_bert_embeddings(self, texts):
        """
        Generates BERT embeddings for a list of input texts using the specified transformer model.

        Steps:
        - Converts the list of texts into a HuggingFace dataset
        - Tokenizes the texts with padding and truncation
        - Passes batches of tokens through the BERT model
        - Computes the mean of the token embeddings, weighted by the attention mask

        Args:
            texts (List[str]): A list of input texts to embed

        Returns:
            torch.Tensor: A 2D tensor containing BERT embeddings for each input text
        """
        dataset = Dataset.from_dict({"text": texts})

        # Tokenize with padding/truncation and convert to tensors
        tokenized_dataset = dataset.map(
            lambda ex: self.tokenizer(
                ex["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ),
            batched=True,
            batch_size=self.batch_size,
            remove_columns=["text"]
        )
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=self.batch_size)
        all_embeddings = []

        for batch in tqdm(dataloader, desc="Computing embeddings"):
            # Move batch to correct device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(**batch)

            # Extract token embeddings and attention masks
            token_embeddings = outputs[0]  # shape: (batch_size, seq_len, hidden_dim)
            attention_mask = batch["attention_mask"]

            # Expand mask and compute weighted average of token embeddings
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

            all_embeddings.append(embeddings.cpu())

            # Cleanup to free GPU memory
            del batch, outputs, token_embeddings, attention_mask, mask_expanded
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0)


    def prepare_quality_attributes(self, quality_attr_df):
        """
        Prepares internal data structures for quality attribute analysis.

        - Extracts quality criteria and similar words from the input DataFrame
        - Optionally uses word2vec scores if provided
        - Computes BERT embeddings for each similar word

        Args:
            quality_attr_df (pd.DataFrame): DataFrame containing 'criteria', 'similar_word', and optional score columns
        """
        logger.info("Preparing quality attributes...")
        logger.info(f"Quality attributes columns: {quality_attr_df.columns.tolist()}")

        self.quality_dict = {}
        self.w2v_scores = {}
        self.similar_word_to_criteria = {}
        score_column = None

        # Try to find a score column with 'w2v' and 'score' in its name
        for col in quality_attr_df.columns:
            if 'w2v' in col.lower() and 'score' in col.lower():
                score_column = col
                logger.info(f"Found score column: {col}")
                break

        if not score_column:
            logger.warning("Could not find max_w2v_score column, using default value 1.0")

        all_similar_words = []

        # Build mappings for each quality attribute and its similar words
        for _, row in quality_attr_df.iterrows():
            criteria = row['criteria']
            similar_word = row['similar_word']

            if criteria not in self.quality_dict:
                self.quality_dict[criteria] = []

            self.quality_dict[criteria].append(similar_word)
            self.similar_word_to_criteria[similar_word] = criteria
            all_similar_words.append(similar_word)

            if score_column:
                self.w2v_scores[(criteria, similar_word)] = row[score_column]
            else:
                self.w2v_scores[(criteria, similar_word)] = 1.0

        logger.info(f"Computing embeddings for {len(all_similar_words)} similar words")

        # Create embedding inputs and compute embeddings
        similar_word_texts = [f"Software quality attribute: {word}" for word in all_similar_words]
        self.similar_word_embeddings = self.get_bert_embeddings(similar_word_texts)
        self.similar_word_index = {word: idx for idx, word in enumerate(all_similar_words)}


    def process_project(self, project_id, project_df):
        """
        Processes a single project to detect quality attribute mentions with sentiment.

        Steps:
        - Aggregates text content from issues
        - Computes BERT embeddings for all texts
        - Compares with pre-computed quality attribute embeddings
        - Identifies similar words above a threshold
        - Computes overall sentiment for matching texts

        Args:
            project_id (str): Identifier for the project
            project_df (pd.DataFrame): DataFrame with issue data

        Returns:
            List[dict]: A list of detected mentions with attributes, sentiment, and scores
        """
        start_time = pd.Timestamp.now()
        logger.info(f"Started processing project {project_id} at {start_time}")

        issue_dict = {}
        project_texts = []
        text_to_idx = {}

        # Extract and normalize issue texts
        for _, row in project_df.iterrows():
            issue_id = row['issue_id']
            text_parts = []

            if pd.notna(row['title']):
                text_parts.append(f"title: {str(row['title'])}")
            if pd.notna(row['body_text']):
                text_parts.append(str(row['body_text']))
            if pd.notna(row['comment_text']):
                text_parts.append(f"comment: {str(row['comment_text'])}")

            if not text_parts:
                continue

            text = " ".join(text_parts)
            if len(text) > 5000:
                text = text[:5000]

            text_hash = hash(text)
            if text_hash not in text_to_idx:
                text_to_idx[text_hash] = len(project_texts)
                project_texts.append(text)

            if issue_id not in issue_dict:
                issue_dict[issue_id] = []
            issue_dict[issue_id].append(text_to_idx[text_hash])

        if not project_texts:
            logger.info(f"No texts found for project {project_id}")
            return []

        project_embeddings = self.get_bert_embeddings(project_texts)
        matches = {}
        chunk_size = 32

        # Compute cosine similarities in chunks
        for i in range(0, len(project_texts), chunk_size):
            end_idx = min(i + chunk_size, len(project_texts))
            proj_emb_chunk = project_embeddings[i:end_idx].to(self.device)
            proj_emb_chunk = F.normalize(proj_emb_chunk, p=2, dim=1)
            sim_word_emb = F.normalize(self.similar_word_embeddings.to(self.device), p=2, dim=1)

            similarities = torch.mm(proj_emb_chunk, sim_word_emb.t())

            for text_pos in range(similarities.shape[0]):
                text_idx = i + text_pos
                for word_idx in range(similarities.shape[1]):
                    similarity = similarities[text_pos, word_idx].item()
                    if similarity > self.similarity_threshold:
                        word = list(self.similar_word_index.keys())[word_idx]
                        criteria = self.similar_word_to_criteria[word]
                        w2v_score = self.w2v_scores.get((criteria, word), 1.0)
                        adjusted_sim = similarity * w2v_score

                        for issue_id, indices in issue_dict.items():
                            if text_idx in indices:
                                key = (criteria, issue_id)
                                if key not in matches:
                                    matches[key] = {"similar_words": [], "scores": [], "text_idx": text_idx}
                                matches[key]["similar_words"].append(word)
                                matches[key]["scores"].append(adjusted_sim)

            del proj_emb_chunk, sim_word_emb, similarities
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()

        # Prepare for sentiment analysis
        all_texts = []
        all_words = []
        all_keys = []

        for key, data in matches.items():
            text_idx = data["text_idx"]
            text = project_texts[text_idx]
            for word in data["similar_words"]:
                all_texts.append(text)
                all_words.append(word)
                all_keys.append(key)

        logger.info(f"Processing sentiment for {len(all_texts)} samples in project {project_id}")

        # Perform sentiment analysis in batches
        batch_size = min(200, len(all_texts))
        sentiments_all = []
        for i in range(0, len(all_texts), batch_size):
            end_idx = min(i + batch_size, len(all_texts))
            batch_texts = all_texts[i:end_idx]
            batch_words = all_words[i:end_idx]
            batch_sentiments, _ = self.batch_sentiment_analysis(batch_texts, batch_words)
            sentiments_all.extend(batch_sentiments)
            logger.info(f"Processed sentiment for {min(end_idx, len(all_texts))}/{len(all_texts)} texts")

        # Aggregate sentiments
        sentiment_map = {}
        for i, key in enumerate(all_keys):
            if key not in sentiment_map:
                sentiment_map[key] = []
            sentiment_map[key].append((all_words[i], sentiments_all[i]))

        results = []
        for (criteria, issue_id), sentiments_list in sentiment_map.items():
            data = matches[(criteria, issue_id)]
            total_score = 0
            for i, (word, sentiment) in enumerate(sentiments_list):
                idx = data["similar_words"].index(word)
                score = data["scores"][idx]
                total_score += score if sentiment == '+' else -score

            main_sentiment = '+' if total_score > 0 else '-'

            if abs(total_score) > self.similarity_threshold:
                results.append({
                    'project_id': project_id,
                    'quality_attribute': criteria,
                    'sentiment': main_sentiment,
                    'similarity_score': abs(total_score),
                    'issue_id': issue_id
                })

        end_time = pd.Timestamp.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Completed project {project_id} in {duration:.2f} seconds. Found {len(results)} quality attribute mentions")

        return results
    

    def analyze_projects_parallel(self, result_df):
        """
        Analyzes multiple projects in parallel using multiprocessing.

        Spawns one process per project (up to max_workers), each running the
        process_project function. Results from all projects are collected
        and combined into a single DataFrame.

        Args:
            result_df (pd.DataFrame): The full dataset containing all issues from all projects

        Returns:
            pd.DataFrame: Combined results from all processed projects
        """
        logger.info(f"Analyzing projects in parallel with {self.max_workers} workers")

        project_ids = result_df['project_id'].unique()
        all_results = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for all projects
            future_to_project = {
                executor.submit(
                    self.process_project,
                    pid,
                    result_df[result_df['project_id'] == pid]
                ): pid for pid in project_ids
            }

            # As futures complete, gather their results
            for future in tqdm(as_completed(future_to_project), total=len(project_ids), desc="Projects completed"):
                project_id = future_to_project[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    logger.info(f"Project {project_id}: Added {len(results)} quality attribute mentions")
                except Exception as e:
                    logger.error(f"Project {project_id} failed with error: {e}")

        return pd.DataFrame(all_results)

    

    def analyze_projects_sequential(self, result_df):
        """
        Analyzes projects sequentially, one at a time.

        Iterates over each unique project ID in the DataFrame,
        processes each project, and collects the results.

        Args:
            result_df (pd.DataFrame): The full dataset containing all issues from all projects

        Returns:
            pd.DataFrame: Combined results from all processed projects
        """
        logger.info("Analyzing projects sequentially")

        project_ids = result_df['project_id'].unique()
        all_results = []

        for project_id in tqdm(project_ids, desc="Analyzing projects"):
            project_df = result_df[result_df['project_id'] == project_id]
            try:
                results = self.process_project(project_id, project_df)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Project {project_id} failed with error: {e}")

        return pd.DataFrame(all_results)

    

    def analyze(self, result_df, quality_attr_df):
        """
        Runs the full quality attribute analysis pipeline on a given dataset.

        Steps:
        - Prepares quality attributes and their embeddings
        - Processes projects in parallel or sequentially
        - Formats the resulting DataFrame with standard columns and identifiers

        Args:
            result_df (pd.DataFrame): Dataset containing project issues
            quality_attr_df (pd.DataFrame): Dataset containing quality criteria and similar words

        Returns:
            pd.DataFrame: Analysis results with detected quality attributes, sentiment, and similarity scores
        """
        # Prepare internal mappings and embeddings
        self.prepare_quality_attributes(quality_attr_df)

        # Analyze projects in parallel or sequential mode
        if self.parallel:
            results_df = self.analyze_projects_parallel(result_df)
        else:
            results_df = self.analyze_projects_sequential(result_df)

        # Format results if not empty
        if not results_df.empty:
            results_df['id'] = range(len(results_df))
            results_df = results_df[['project_id', 'quality_attribute', 'sentiment', 'similarity_score', 'issue_id']]
            results_df.rename(columns={
                'quality_attribute': 'criteria',
                'sentiment': 'semantic'
            }, inplace=True)

        return results_df
    

    def create_visualizations(self, results_df, output_dir='output/visualizations'):
        """
        Creates and saves several visualizations based on the analysis results.

        Visualizations include:
        - Top 15 most mentioned quality attributes
        - Sentiment distribution for top 5 attributes
        - Distribution of similarity scores
        - Top 10 projects with most quality attribute mentions

        Args:
            results_df (pd.DataFrame): The analysis results DataFrame
            output_dir (str): Directory to save the plots to
        """
        if results_df.empty:
            return

        os.makedirs(output_dir, exist_ok=True)

        # Top 15 quality attributes by frequency
        plt.figure(figsize=(12, 8))
        top_attrs = results_df['criteria'].value_counts().head(15)
        sns.barplot(x=top_attrs.values, y=top_attrs.index)
        plt.title('Top 15 Quality Attributes Mentioned')
        plt.xlabel('Number of Mentions')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/top_quality_attributes.png')
        plt.close()

        # Sentiment distribution for top 5 attributes
        plt.figure(figsize=(12, 8))
        top_5_attrs = results_df['criteria'].value_counts().head(5).index
        sentiment_by_attr = pd.crosstab(
            results_df[results_df['criteria'].isin(top_5_attrs)]['criteria'],
            results_df[results_df['criteria'].isin(top_5_attrs)]['semantic']
        )
        sentiment_by_attr.plot(kind='bar', stacked=True)
        plt.title('Sentiment Distribution for Top 5 Quality Attributes')
        plt.xlabel('Quality Attribute')
        plt.ylabel('Count')
        plt.legend(title='Sentiment')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sentiment_by_attribute.png')
        plt.close()

        # Similarity score distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['similarity_score'], bins=20)
        plt.title('Distribution of Similarity Scores')
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/similarity_distribution.png')
        plt.close()

        # Top 10 projects by number of mentions
        plt.figure(figsize=(12, 8))
        top_projects = results_df['project_id'].value_counts().head(10)
        sns.barplot(x=top_projects.values, y=top_projects.index.astype(str))
        plt.title('Top 10 Projects by Quality Attribute Mentions')
        plt.xlabel('Number of Mentions')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/top_projects.png')
        plt.close()
# end of class


def main(sim_threshold=0.05,
         batch_size=32,
         use_gpu=True,
         parallel=False,
         max_workers=None,
         max_matches_per_project=1000,
         sample_matches=True):
    """
    Main entry point for running the quality attribute analysis pipeline.

    Args:
        sim_threshold (float): Cosine similarity threshold for matching similar words (default: 0.05).
        batch_size (int): Number of samples processed per batch for embeddings and sentiment analysis.
        use_gpu (bool): Whether to use GPU acceleration (if available).
        parallel (bool): Whether to process projects in parallel (True) or sequentially (False).
        max_workers (int or None): Max number of workers for parallel processing. Defaults to CPU count - 1.
        max_matches_per_project (int): Maximum number of matches to extract per project.
        sample_matches (bool): Whether to sample matches or take the top-ranked ones (if False, takes top by score).
    """

    print("Starting quality attributes analysis pipeline...", flush=True)

    # Connect to the database
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)
    conn = engine.connect()

    # Load the quality attributes (read only once)
    print("Loading quality attributes from 'similar_words' table...", flush=True)
    quality_attr_df = pd.read_sql("SELECT criteria, similar_word, max_w2v_score FROM similar_words", conn)

    # Initialize the analyzer
    analyzer = QualityAttributeAnalyzer(
        similarity_threshold=sim_threshold,
        batch_size=batch_size,
        use_gpu=use_gpu,
        parallel=parallel,
        max_workers=max_workers,
        max_matches_per_project=max_matches_per_project,
        sample_matches=sample_matches
    )

    # Prepare quality attribute embeddings
    analyzer.prepare_quality_attributes(quality_attr_df)

    # Create result table if not exists
    with engine.begin() as con:
        con.execute(text("""
            CREATE TABLE IF NOT EXISTS quality_attribute_analysis (
                id SERIAL PRIMARY KEY,
                project_id BIGINT,
                criteria TEXT,
                semantic TEXT,
                similarity_score FLOAT,
                issue_id BIGINT
            );
        """))
        con.execute(text("""
            CREATE TABLE IF NOT EXISTS quality_attribute_analysis_tracker (
                last_issue_id BIGINT
            );
        """))
        # Insert a progress row if it doesn't exist
        result = con.execute(text("SELECT COUNT(*) FROM quality_attribute_analysis_tracker")).scalar()
        if result == 0:
            con.execute(text("INSERT INTO quality_attribute_analysis_tracker (last_issue_id) VALUES (0);"))

    # Determine starting point
    last_issue_id = conn.execute(text("SELECT MAX(last_issue_id) FROM quality_attribute_analysis_tracker")).scalar()
    print(f"Resuming from issue_id > {last_issue_id}", flush=True)

    # Batch size for issues to analyze
    issue_batch_size = 1000

    while True:
        # Read next batch of issues with comments
        query = f"""
            SELECT i.issue_id, i.project_id, i.title, i.body_text, i.state_reason,
                   STRING_AGG(c.body_text, '\n') AS comment_text
            FROM issues i
            LEFT JOIN comments c ON c.issue_id = i.issue_id
            WHERE i.issue_id > :last_id
            AND (i.title IS NOT NULL OR i.body_text IS NOT NULL)
            GROUP BY i.issue_id
            ORDER BY i.issue_id ASC
            LIMIT {issue_batch_size};
        """
        batch_df = pd.read_sql(text(query), conn, params={"last_id": int(last_issue_id)})

        if batch_df.empty:
            print("No more issues to process.", flush=True)
            break

        print(f"Processing {len(batch_df)} issues...", flush=True)

        # Track the highest issue ID processed
        max_issue_id = batch_df["issue_id"].max()

        # Analyze the batch
        results_df = analyzer.analyze(batch_df, quality_attr_df)

        if not results_df.empty:
            # Append to result table
            results_df.to_sql("quality_attribute_analysis", conn, if_exists="append", index=False)
            print(f"Stored {len(results_df)} results to database.", flush=True)

        # Update progress tracker
        with engine.begin() as con:
            con.execute(text("UPDATE quality_attribute_analysis_tracker SET last_issue_id = :new_id"), {"new_id": int(max_issue_id)})

        last_issue_id = max_issue_id

    print("Pipeline execution complete!", flush=True)

# Entry point for the script    
if __name__=="__main__":
    print("Start of the analyses!", flush=True)

    parser=argparse.ArgumentParser(description='Quality attribute analysis pipeline')

    parser.add_argument('--threshold',type=float,default=0.05,help='Similarity threshold (default: 0.05)')
    parser.add_argument('--batch-size',type=int,default=32,help='Batch size')
    parser.add_argument('--no-gpu',action='store_true',help='Disable GPU acceleration')
    parser.add_argument('--parallel',action='store_true',help='Enable parallel processing')
    parser.add_argument('--workers',type=int,default=None,help='Number of parallel workers')
    parser.add_argument('--max-matches',type=int,default=1000,help='Maximum matches per project (default: 1000)')
    parser.add_argument('--no-sampling',action='store_true',help='Use top matches instead of sampling')

    args=parser.parse_args()

    main(
        sim_threshold=args.threshold,
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu,
        parallel=args.parallel,
        max_workers=args.workers,
        max_matches_per_project=args.max_matches,
        sample_matches=not args.no_sampling
    )
