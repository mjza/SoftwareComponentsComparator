"""
Common imports for an NLP / deep‑learning workflow
--------------------------------------------------
This block gathers all imports in one place, grouped and alphabetised
according to PEP 8 (stdlib, third‑party, then local/project).
"""

# ——— Standard library ————————————————————————————————————————————————
import logging
import multiprocessing
import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Sequence, Tuple, Any, Dict, List
from dotenv import load_dotenv

# ——— Third‑party libraries ————————————————————————————————————————————
import numpy as np                      # Fast numerical arrays / linear algebra
import pandas as pd                     # Tabular data manipulation
import torch                            # Core PyTorch tensor library
import torch.nn.functional as F         # “Functional” NN ops (activations, loss…)
import psycopg2                         # For Postgres database
from psycopg2.extras import execute_values
from datasets import Dataset            # Datasets: memory‑mapped dataset object
from tqdm import tqdm                   # Progress‑bar utility
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Hugging Face Transformers
from transformers import (
    AutoModel,                          # Pre‑trained model loader
    AutoTokenizer,                      # Matching tokenizer loader
    pipeline                            # High‑level task pipelines
)

# Visualisation (note: seaborn is optional—Matplotlib alone is often enough)
import matplotlib.pyplot as plt
import seaborn as sns

# load environment parameters
load_dotenv()

# ——— Logging setup ————————————————————————————————————————————————
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

logger = logging.getLogger(__name__)
logger.info("Imports complete and logger initialised.")


class QualityAttributeAnalyzer:
    """
    Analyse software‑quality attributes by:
      • Embedding text with a Sentence‑Transformers model
      • Computing similarity scores
      • Running sentiment analysis with a DistilBERT SST‑2 head

    Parameters
    ----------
    similarity_threshold : float, default 0.05
        Cosine‑similarity cut‑off below which two items are considered dissimilar.
    batch_size : int, default 2048
        Mini‑batch size for both embedding and sentiment pipelines.
    use_gpu : bool, default True
        If True and a CUDA device is available, computations run on GPU.
    parallel : bool, default False
        Whether to process batches in parallel with `multiprocessing`.
    max_workers : int or None, default None
        Worker count for the process pool.  Falls back to (CPU‑cores − 1).
    embedding_model : str
        model name for text embeddings.
    sentiment_model : str
        model name for sentiment classification.
    max_matches_per_project : int, default 1000
        Cap on stored similarity matches per project.
    sample_matches : bool, default True
        If True, sample from matches when the cap is exceeded.
    """

    # ──────────────────────────────────────────────────────────────────────────────
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        similarity_threshold: float = 0.05,
        batch_size: int = 2048,
        use_gpu: bool = True,
        parallel: bool = False,
        max_workers: int | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sentiment_model: str = (
            "distilbert-base-uncased-finetuned-sst-2-english"
        ),
        max_matches_per_project: int = 1000,
        sample_matches: bool = True,
    ) -> None:
        # ——— Hyper‑parameters ————————————————————————————————
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.parallel = parallel
        self.max_matches_per_project = max_matches_per_project
        self.sample_matches = sample_matches

        # ——— Hardware / execution context ————————————————
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)
        self.device = (
            "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        logger.info("Using device: %s", self.device)
        logger.info("Similarity threshold: %.3f", self.similarity_threshold)

        # ——— Models & tokenisers ————————————————————————————
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model).to(self.device)

        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model)
        self.sentiment_model = AutoModel.from_pretrained(sentiment_model).to(
            self.device
        )

        # Hugging Face pipeline abstracts batching, tokenisation and inference.
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=sentiment_model,
            tokenizer=self.sentiment_tokenizer,
            device=0 if self.device.startswith("cuda") else -1,
            batch_size=batch_size,
            max_length=512,
            truncation=True,
        )

        # ——— Data structures for later use ————————————————————
        self.w2v_scores: dict[str, float] = {}          # word → similarity score
        self.similar_words: list[str] = []              # words near each criterion
        self.similar_word_embeddings: torch.Tensor | None = None
        self.similar_word_to_criteria: dict[str, list[str]] = {}
        self.quality_dict: dict[str, list[str]] = {}    # criterion → matched words
        self.reason_cache: dict[tuple[str, str], str] = {}  # (criterion, word) → explanation

        # Quick polarity lexicons for rule‑based heuristics
        self.pos_words = [
            "good", "great", "excellent", "benefit", "like", "love",
            "improve", "better", "fix", "solve", "easy", "useful",
            "solved", "success", "fast",
        ]
        self.neg_words = [
            "bad", "poor", "issue", "bug", "problem", "difficult",
            "fail", "crash", "error", "slow", "breaks", "broken",
            "missing", "cannot", "wrong",
        ]


    # ──────────────────────────────────────────────────────────────────────────────
    def check_valid_text(self, text: Any) -> bool:
        """
        Heuristically decide whether *text* is “clean” enough for NLP processing.

        Returns
        -------
        bool
            • **True**  – the string passes all sanity checks.  
            • **False** – the string is likely noisy/garbled and should be skipped.

        Rules applied
        -------------
        1. Empty or whitespace‑only strings are rejected.  
        2. More than five explicit “[UNK]” tokens ⇒ reject.  
        3. If > 30 % of characters are non‑ASCII ⇒ reject.
        """
        # Ensure we are working with a string.
        if not isinstance(text, str):
            text = str(text)

        # 1) Non‑empty check
        if not text.strip():
            return False

        # 2) Excessive “[UNK]” tokens usually signal tokenizer failure.
        if text.count("[UNK]") > 5:
            return False

        # 3) Rough filter for binary blobs or non‑English artefacts.
        non_ascii_chars = sum(1 for ch in text if ord(ch) > 127)
        if non_ascii_chars > 0.30 * len(text):
            return False

        return True

    # ──────────────────────────────────────────────────────────────────────────────
    def determine_content_type(self, text: Any) -> str:
        """
        Classify a commit / PR / issue message into a coarse‑grained category.

        Categories returned
        -------------------
        • **"bug_fix"**       – fixing defects, crashes, vulnerabilities …  
        • **"feature"**       – new functionality or enhancements.  
        • **"dependency"**    – version bumps, library upgrades.  
        • **"documentation"** – docs, guides, examples.  
        • **"general"**       – anything that doesn’t match the above.

        Notes
        -----
        This is a keyword‑triggered heuristic; downstream ML models can refine it.
        """
        if not isinstance(text, str):
            text = str(text)

        text_lower = text.lower()

        if any(word in text_lower for word in (
            "fix", "issue", "bug", "crash", "error", "problem", "vulnerability"
        )):
            return "bug_fix"

        if any(word in text_lower for word in (
            "add", "feature", "implement", "new", "enhancement", "request"
        )):
            return "feature"

        if any(word in text_lower for word in (
            "bump", "update", "upgrade", "dependency", "version"
        )):
            return "dependency"

        if any(word in text_lower for word in (
            "doc", "documentation", "example", "guide", "manual"
        )):
            return "documentation"

        return "general"


    # ──────────────────────────────────────────────────────────────────────────────
    def check_context_relevance(self, text: Any, quality_attr: Any) -> bool:
        """
        Decide whether *text* is talking about the given *quality_attr*.

        Heuristic
        ---------
        1. **Exact match** – the full, lower‑cased attribute appears as a
           substring of *text*.
        2. **Partial match** – for multi‑word attributes, any word longer
           than three characters appears in *text*.

        Examples
        --------
        >>> self.check_context_relevance("Improve response time", "response time")
        True
        >>> self.check_context_relevance("Refactor parser", "security")
        False
        """
        # Convert to lowercase strings to avoid AttributeError on non‑str inputs.
        text_lower = str(text).lower()
        qa_lower = str(quality_attr).lower()

        # Fast path: full phrase match.
        if qa_lower in text_lower:
            return True

        # Fallback: component word match for multi‑word attributes.
        words=qa_lower.split()
        if len(words)>1:
            for word in words:
                if len(word) > 3 and word in text_lower:
                    return True

        return False


    # ──────────────────────────────────────────────────────────────────────────────
    def override_sentiment(
        self,
        text: Any,
        model_sentiment: str,
        confidence: float,
    ) -> tuple[str, float]:
        """
        Apply a lightweight, lexicon‑based override to the model’s sentiment.

        Logic
        -----
        • Count positive and negative cue words in *text*.  
        • If one polarity outnumbers the other by **> 2×** *and*  
          the model’s confidence is **< 0.90** *and*  
          the cue‑word polarity conflicts with the model’s label,  
          then flip the label and set confidence to **0.70**.

        Returns
        -------
        tuple[str, float]
            Possibly corrected `(sentiment_label, confidence)` pair.
            Labels: `'+'` (positive), `'-'` (negative).
        """
        text_lower = str(text).lower()

        pos_count = sum(1 for w in self.pos_words if w in text_lower)
        neg_count = sum(1 for w in self.neg_words if w in text_lower)

        # Positive override
        if (
            pos_count > 2 * neg_count
            and confidence < 0.90
            and model_sentiment == "-"
        ):
            return "+", 0.70

        # Negative override
        if (
            neg_count > 2 * pos_count
            and confidence < 0.90
            and model_sentiment == "+"
        ):
            return "-", 0.70

        # No change
        return model_sentiment, confidence


    # ──────────────────────────────────────────────────────────────────────────────
    def extract_meaningful_context(
        self,
        text: Any,
        quality_attr: Any,
    ) -> str:
        """
        Pull out one‑to‑two short sentences that best illustrate *quality_attr*
        within *text* (e.g. a commit message, PR body, or issue description).

        Fallback order
        --------------
        1. Sentences explicitly mentioning *quality_attr* (or its component
           words) and shorter than 200 chars.
        2. Sentences matching the inferred *content_type* keywords.
        3. A trimmed “title: …” line (≤ 150 chars).
        4. First ≈ 30 tokens of the cleaned text, with an ellipsis.

        Returns
        -------
        str
            A concise context string, or a message explaining why none could
            be extracted.
        """
        # Ensure we are working with strings
        text = str(text)
        quality_attr = str(quality_attr)

        # Abort early if the text is obviously corrupted/empty.
        if not self.check_valid_text(text):
            return "Content not available or contains invalid characters"

        # ——— Basic cleaning ————————————————————————————————————————————
        clean_text = re.sub(r"\[CLS\]|\[SEP\]", "", text)           # drop BERT tokens
        clean_text = re.sub(r"\s+", " ", clean_text).strip()        # normalise spaces

        # Extract an optional “title: …” line (common in commit templates)
        title_match = re.search(r"title:\s*([^\n]+)", clean_text, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else None
        if title and len(title) > 150:
            title = f"{title[:150]}..."

        # Coarse classification (bug fix, feature, …)
        content_type = self.determine_content_type(clean_text)

        # ——— Sentence tokenisation ————————————————————————————————
        sentences = re.split(r"(?<=[.!?])\s+", clean_text)

        qa_lower = quality_attr.lower()
        qa_words = qa_lower.split()

        # 1️⃣  Sentences that explicitly mention the quality attribute
        candidate_sents: list[str] = []
        for sent in sentences:
            if len(sent) > 200 or len(sent.split()) < 3:
                continue

            s_lower = sent.lower()
            if qa_lower in s_lower:
                candidate_sents.append(sent)
            elif len(qa_words) > 1 and any(
                w in s_lower for w in qa_words if len(w) > 3
            ):
                candidate_sents.append(sent)

        if candidate_sents:
            # Rank by polarity word count; keep the two most “opinionated”.
            polarity = self.pos_words + self.neg_words
            best = sorted(
                candidate_sents,
                key=lambda s: sum(1 for w in polarity if w in s.lower()),
                reverse=True,
            )[:2]
            return " ".join(best)

        # 2️⃣  Sentences relevant to the inferred content type
        type_keywords = {
            "bug_fix": ["fix", "issue", "bug", "problem"],
            "feature": ["add", "feature", "implement", "new"],
            "dependency": ["bump", "update", "upgrade", "dependency"],
            "documentation": ["doc", "guide", "example"],
        }
        keywords = type_keywords.get(content_type, [])
        context_sents = [
            s
            for s in sentences
            if len(s.split()) >= 3
            and len(s) < 200
            and any(w in s.lower() for w in keywords)
        ][:2]

        if context_sents:
            return " ".join(context_sents)

        # 3️⃣  Use the title line if available
        if title:
            return title

        # 4️⃣  Last resort: return a brief preview of the whole text
        preview = " ".join(clean_text.split()[:30]) + "..."
        return preview


    # ──────────────────────────────────────────────────────────────────────────────
    def generate_reason(
        self,
        text: Any,
        qa: Any,
        model_sentiment: str,
        confidence: float,
    ) -> str:
        """
        Produce a human‑readable explanation string that ties sentiment,
        quality attribute (*qa*), and context together.

        Workflow
        --------
        1. Sanity‑check the raw *text* with `check_valid_text`.
        2. Infer the *content_type* (bug fix, feature, …).
        3. Optionally override the model’s sentiment with a lexicon heuristic.
        4. Assess whether *text* directly references *qa*.
        5. Extract the most relevant sentence(s) via `extract_meaningful_context`.
        6. Compose a templated explanation.

        Parameters
        ----------
        text : Any
            Raw commit/PR/issue text.
        qa : Any
            Quality attribute of interest (e.g. “performance”).
        model_sentiment : {'+', '-'}
            Sentiment label predicted by the model.
        confidence : float
            Model‑reported confidence in the prediction.

        Returns
        -------
        str
            Natural‑language justification string.
        """
        # Coerce inputs to strings early
        text = str(text)
        qa = str(qa)

        # 1️⃣  Validate text
        if not self.check_valid_text(text):
            label = "Positive" if model_sentiment == "+" else "Negative"
            return (
                f"{label} sentiment ({confidence:.2f}) about {qa}: "
                "Content not available or contains invalid characters"
            )

        # 2️⃣  Coarse classification
        content_type = self.determine_content_type(text)

        # 3️⃣  Sentiment override (lexicon heuristic)
        sentiment, adj_conf = self.override_sentiment(
            text, model_sentiment, confidence
        )
        sentiment_label = "Positive" if sentiment == "+" else "Negative"

        # 4️⃣  Relevance check
        is_relevant = self.check_context_relevance(text, qa)

        # 5️⃣  Context extraction
        context = self.extract_meaningful_context(text, qa)

        if not is_relevant:
            return (
                f"{sentiment_label} sentiment ({adj_conf:.2f}) about {qa}: "
                f"Content may indirectly relate to {qa.lower()}. {context}"
            )

        # 6️⃣  Template selection
        prefix_templates = {
            "bug_fix": (
                f"{sentiment_label} sentiment ({adj_conf:.2f}) about {qa}: "
                f"{'Fixed issue improving' if sentiment == '+' else 'Problem affecting'} "
                f"{qa.lower()}."
            ),
            "feature": (
                f"{sentiment_label} sentiment ({adj_conf:.2f}) about {qa}: "
                f"{'Feature enhancing' if sentiment == '+' else 'Feature request for'} "
                f"{qa.lower()}."
            ),
            "dependency": (
                f"{sentiment_label} sentiment ({adj_conf:.2f}) about {qa}: "
                f"{'Dependency update improving' if sentiment == '+' else 'Dependency issue affecting'} "
                f"{qa.lower()}."
            ),
            "documentation": (
                f"{sentiment_label} sentiment ({adj_conf:.2f}) about {qa}: "
                f"{'Documentation clarifying' if sentiment == '+' else 'Documentation needed for'} "
                f"{qa.lower()}."
            ),
            "general": (
                f"{sentiment_label} sentiment ({adj_conf:.2f}) about {qa}: "
                f"{'Content highlights good' if sentiment == '+' else 'Content indicates issues with'} "
                f"{qa.lower()}."
            ),
        }

        return f"{prefix_templates.get(content_type, prefix_templates['general'])} {context}"


    # ──────────────────────────────────────────────────────────────────────────────
    def batch_sentiment_analysis(
        self,
        texts: Sequence[Any],
        quality_attrs: Sequence[Any],
    ) -> Tuple[list[str], list[str]]:
        """
        Run sentiment analysis on a batch of *(text, quality_attribute)* pairs
        and return both the raw sentiment labels and explanatory strings.

        Steps
        -----
        1. **Pre‑clean** each text (remove BERT tokens, collapse whitespace,
           truncate to 512 chars) or mark as “Invalid content”.
        2. Use the Hugging Face `sentiment_analyzer` pipeline in batch mode.
        3. For each result, call `generate_reason` to produce a narrative.
        4. Handle pipeline errors gracefully by logging and returning
           fallback values.

        Parameters
        ----------
        texts : Sequence[Any]
            Iterable of commit/PR/issue texts.
        quality_attrs : Sequence[Any]
            Iterable of quality attributes (same length as *texts*).

        Returns
        -------
        tuple[list[str], list[str]]
            • **sentiments** – list of `'+'` (positive) / `'-'` (negative).  
            • **reasons**    – corresponding explanatory messages.
        """
        # Fast‑exit if nothing to do
        if not texts:
            return [], []

        # ——— Pre‑processing ————————————————————————————————————————————
        records: list[dict[str, str]] = []
        for text, qa in zip(texts, quality_attrs, strict=False):
            # Coerce to strings early
            text = str(text)
            qa = str(qa)

            # Validate and clean
            if not self.check_valid_text(text):
                clean = "Invalid content"
            else:
                clean = re.sub(r"\[CLS\]|\[SEP\]", "", text)
                clean = re.sub(r"\s+", " ", clean).strip()
                clean = clean[:512]  # HF pipeline default max_length is 512

            records.append({"text": clean, "quality_attr": qa})

        # ——— Inference ————————————————————————————————————————————————
        try:
            inputs = [rec["text"] for rec in records]
            results = self.sentiment_analyzer(
                inputs, batch_size=self.batch_size
            )

            sentiments: list[str] = []
            reasons: list[str] = []

            for rec, res in zip(records, results, strict=False):
                sentiment = "+" if res["label"] == "POSITIVE" else "-"
                confidence = res["score"]
                reason = self.generate_reason(
                    rec["text"],
                    rec["quality_attr"],
                    sentiment,
                    confidence,
                )
                sentiments.append(sentiment)
                reasons.append(reason)

            return sentiments, reasons

        # ——— Robust error handling ————————————————————————————————
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error in sentiment analysis: %s", exc)
            fallback_sentiments = ["+"] * len(texts)
            fallback_reasons = ["Error processing sentiment"] * len(texts)
            return fallback_sentiments, fallback_reasons


    # ──────────────────────────────────────────────────────────────────────────────
    def get_bert_embeddings(
        self,
        texts: Sequence[str],
    ) -> torch.Tensor:
        """
        Encode a batch of *texts* with the class’s BERT‑style model and return
        sentence‑level embeddings (mean‑pooled over valid tokens).

        Pipeline
        --------
        1. Build a 🤗 `Dataset` from the raw strings.
        2. Tokenise with padding/truncation to 512 tokens (HF max default).
        3. Create a PyTorch `DataLoader` for efficient batching.
        4. Forward‑pass **without gradients** and mean‑pool over the
           attention‑masked tokens to obtain a single 768‑d (or model‑dim)
           vector per sentence.
        5. Concatenate all batches on the CPU and return a single tensor
           shaped **(len(texts), hidden_size)**.

        Parameters
        ----------
        texts : Sequence[str]
            Iterable of raw strings to embed.

        Returns
        -------
        torch.Tensor
            2‑D tensor of shape *(N, hidden_size)* on **CPU**.
        """
        # ——— Build Hugging Face Dataset ————————————————————————————————
        ds = Dataset.from_dict({"text": texts})

        # Tokenise in bulk; keep tensors on CPU for now
        ds_tok = ds.map(
            lambda ex: self.tokenizer(
                ex["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ),
            batched=True,
            batch_size=self.batch_size,
            remove_columns=["text"],
        )
        ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

        # ——— DataLoader for efficient batching ————————————————————————
        loader = torch.utils.data.DataLoader(
            ds_tok, batch_size=self.batch_size
        )

        all_embeds: list[torch.Tensor] = []

        # ——— Forward pass ——————————————————————————————————————————————
        for batch in tqdm(loader, desc="Computing embeddings"):
            # Move tensors to the target device (CPU or GPU)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(**batch)

            # outputs[0] = (batch, seq_len, hidden_size)
            token_embeds = outputs[0]                       # Token‑level reps
            attn_mask = batch["attention_mask"]             # 1 for real token

            # Mean‑pool: sum hidden states where mask == 1, divide by token count
            mask_exp = attn_mask.unsqueeze(-1).expand(token_embeds.size()).float()
            summed = torch.sum(token_embeds * mask_exp, dim=1)
            counts = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
            sent_embeds = summed / counts                   # (batch, hidden)

            # Collect on CPU to free GPU memory
            all_embeds.append(sent_embeds.cpu())

            # Housekeeping
            del batch, outputs, token_embeds, attn_mask, mask_exp
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        # ——— Stack all batches into one tensor ————————————————————————
        return torch.cat(all_embeds, dim=0)


    # ──────────────────────────────────────────────────────────────────────────────
    def prepare_quality_attributes(
        self,
        quality_attr_df: pd.DataFrame,
    ) -> None:
        """
        Load a *quality‑attribute* table and populate the lookup structures
        used by the analyser.

        Expected columns
        ----------------
        • **criteria**       – canonical quality attribute (e.g. “performance”).  
        • **similar_word**   – lexical variant / synonym.  
        • **<w2v>…<score>**  – (optional) similarity score column; the first
          column whose lowercase name contains both “w2v” and “score” is used.

        Side‑effects
        ------------
        Populates / resets these instance attributes:

        * `self.quality_dict`              – {criteria → [similar_word, …]}  
        * `self.w2v_scores`                – {(criteria, word) → float}  
        * `self.similar_word_to_criteria`  – {word → criteria}  
        * `self.similar_words`             – [word, …]  
        * `self.similar_word_embeddings`   – Tensor of BERT embeddings
        """
        logger.info("Preparing quality attributes…")
        logger.info("Quality‑attribute columns: %s", quality_attr_df.columns.tolist())

        # ——— Reset all dependent data structures ————————————————————————
        self.quality_dict: dict[str, list[str]] = {}
        self.w2v_scores: dict[tuple[str, str], float] = {}
        self.similar_word_to_criteria: dict[str, str] = {}
        self.similar_words: list[str] = []

        # ——— Identify the w2v‑score column (if present) ————————————————
        score_col: str | None = None
        for col in quality_attr_df.columns:
            col_l = col.lower()
            if "w2v" in col_l and "score" in col_l:
                score_col = col
                logger.info("Found score column: %s", score_col)
                break

        if score_col is None:
            logger.warning(
                "Could not find a w2v score column – defaulting all scores to 1.0"
            )

        # ——— Build lookup dictionaries ————————————————————————————————
        for _, row in quality_attr_df.iterrows():
            criteria = str(row["criteria"])
            word = str(row["similar_word"])

            # Map criteria → list of words
            self.quality_dict.setdefault(criteria, []).append(word)

            # Map word → criteria
            self.similar_word_to_criteria[word] = criteria

            # Flat list of all words
            self.similar_words.append(word)

            # (criteria, word) → score
            score = row[score_col] if score_col else 1.0
            self.w2v_scores[(criteria, word)] = float(score)

        # ——— Pre‑compute embeddings for all similar words ——————————————
        logger.info("Computing embeddings for %d similar words", len(self.similar_words))
        prompts = [f"Software quality attribute: {w}" for w in self.similar_words]
        self.similar_word_embeddings = self.get_bert_embeddings(prompts)


    # ──────────────────────────────────────────────────────────────────────────────
    def process_project(
        self,
        project_id: Any,
        project_df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Analyse a single project (repo) and return per‑issue, per‑quality‑attribute
        sentiment findings.

        High‑level workflow
        -------------------
        1. **Gather & deduplicate** textual artefacts (titles, bodies, comments).  
        2. **Embed** each unique text with `get_bert_embeddings`.  
        3. **Similarity search** against pre‑computed quality‑attribute words.  
        4. **Batch sentiment analysis** for every (text, similar_word) match.  
        5. **Aggregate** scores into an overall sentiment per *(criteria, issue)*.  

        Parameters
        ----------
        project_id : Any
            Identifier for logging / result rows.
        project_df : pd.DataFrame
            Must include columns: `issue_id`, `title`, `body_text`, `comment_text`.

        Returns
        -------
        list[dict]
            Each dict contains: `project_id`, `quality_attribute`, `sentiment`,
            `similarity_score`, `issue_id`.
        """
        try:
            start_time = pd.Timestamp.now()
            logger.info("Started processing project %s at %s", project_id, start_time)

            # ─── 1️⃣  Collect and de‑duplicate texts per issue ───────────────────
            issue_text_map: dict[Any, list[int]] = {}
            project_texts: list[str] = []
            text_hash_to_idx: dict[int, int] = {}

            for _, row in project_df.iterrows():
                issue_id = row["issue_id"]

                # Build one long string from title / body / comment
                parts: list[str] = []
                if pd.notna(row["title"]):
                    parts.append(f"title: {str(row['title'])}")
                if pd.notna(row["body_text"]):
                    parts.append(str(row["body_text"]))
                if pd.notna(row["comment_text"]):
                    parts.append(f"comment: {str(row['comment_text'])}")

                if not parts:  # nothing to analyse
                    continue

                text = " ".join(parts)[:5000]  # hard cap length
                text_hash = hash(text)

                # Map unique text → index
                if text_hash not in text_hash_to_idx:
                    text_hash_to_idx[text_hash] = len(project_texts)
                    project_texts.append(text)

                # Map issue → list of text indices
                issue_text_map.setdefault(issue_id, []).append(
                    text_hash_to_idx[text_hash]
                )

            if not project_texts:
                logger.info("No texts found for project %s", project_id)
                return []

            # ─── 2️⃣  Compute embeddings for all unique texts ───────────────────
            project_embeds = self.get_bert_embeddings(project_texts)  # (N, dim)

            # ─── 3️⃣  Similarity search to find quality‑attribute mentions ─────
            matches: dict[
                tuple[str, Any],
                dict[str, Any],
            ] = {}  # key = (criteria, issue_id)

            chunk_size = 32
            qa_embeds = F.normalize(
                self.similar_word_embeddings.to(self.device), p=2, dim=1
            )

            for start in range(0, len(project_texts), chunk_size):
                end = min(start + chunk_size, len(project_texts))

                text_emb_chunk = F.normalize(
                    project_embeds[start:end].to(self.device), p=2, dim=1
                )
                sims = torch.mm(text_emb_chunk, qa_embeds.T)  # (chunk, |words|)

                for pos_in_chunk in range(sims.size(0)):
                    text_idx = start + pos_in_chunk

                    # Indices of attribute words above threshold
                    relevant = torch.where(
                        sims[pos_in_chunk] > self.similarity_threshold
                    )[0]

                    for word_idx_t in relevant:
                        try:
                            word_idx = word_idx_t.item()
                            word = self.similar_words[word_idx]
                            criteria = self.similar_word_to_criteria[word]

                            sim = sims[pos_in_chunk, word_idx].item()
                            w2v = self.w2v_scores.get((criteria, word), 1.0)
                            adjusted = sim * w2v

                            # Record per‑issue
                            for issue_id, indices in issue_text_map.items():
                                if text_idx in indices:
                                    key = (criteria, issue_id)
                                    matches.setdefault(
                                        key,
                                        {
                                            "similar_words": [],
                                            "scores": [],
                                            "text_idx": text_idx,
                                        },
                                    )
                                    matches[key]["similar_words"].append(word)
                                    matches[key]["scores"].append(adjusted)
                        except Exception as exc:  # pylint: disable=broad-except
                            logger.warning(
                                "Error processing word %s in project %s: %s",
                                word_idx_t,
                                project_id,
                                exc,
                            )

                # House‑keeping
                del text_emb_chunk, sims
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()

            # ─── 4️⃣  Run sentiment analysis for each (text, word) pair ─────────
            all_texts: list[str] = []
            all_words: list[str] = []
            all_keys: list[tuple[str, Any]] = []

            for key, data in matches.items():
                text_idx = data["text_idx"]
                if text_idx >= len(project_texts):
                    logger.warning(
                        "Invalid text index %d for project %s (max %d)",
                        text_idx,
                        project_id,
                        len(project_texts) - 1,
                    )
                    continue

                text = project_texts[text_idx]
                for word in data["similar_words"]:
                    all_texts.append(text)
                    all_words.append(word)
                    all_keys.append(key)

            sentiments_all: list[str] = []
            if all_texts:
                logger.info(
                    "Processing sentiment for %d samples in project %s",
                    len(all_texts),
                    project_id,
                )
                batch_size = 256
                for start in range(0, len(all_texts), batch_size):
                    end = min(start + batch_size, len(all_texts))
                    try:
                        batch_sents, _ = self.batch_sentiment_analysis(
                            all_texts[start:end], all_words[start:end]
                        )
                        sentiments_all.extend(batch_sents)
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.error("Error in batch sentiment analysis: %s", exc)
                        sentiments_all.extend(["+"] * (end - start))

                    logger.info("Processed sentiment for %d/%d samples", end, len(all_texts))

            # ─── 5️⃣  Aggregate per‑issue / per‑criteria sentiment ──────────────
            sentiment_map: dict[
                tuple[str, Any], list[tuple[str, str]]
            ] = {}  # key → [(word, sentiment)]

            for idx, key in enumerate(all_keys):
                if idx >= len(sentiments_all):
                    continue
                sentiment_map.setdefault(key, []).append(
                    (all_words[idx], sentiments_all[idx])
                )

            results: list[dict[str, Any]] = []
            for (criteria, issue_id), word_sents in sentiment_map.items():
                data = matches.get((criteria, issue_id))
                if not data:
                    continue

                total_score = 0.0
                for word, sent in word_sents:
                    try:
                        w_idx = data["similar_words"].index(word)
                        score = data["scores"][w_idx]
                        total_score += score if sent == "+" else -score
                    except ValueError:
                        continue

                main_sentiment = "+" if total_score > 0 else "-"
                if abs(total_score) > self.similarity_threshold:
                    results.append(
                        {
                            "project_id": project_id,
                            "quality_attribute": criteria,
                            "sentiment": main_sentiment,
                            "similarity_score": abs(total_score),
                            "issue_id": issue_id,
                        }
                    )

            duration = (pd.Timestamp.now() - start_time).total_seconds()
            logger.info(
                "Completed project %s in %.2f s – found %d quality‑attribute mentions",
                project_id,
                duration,
                len(results),
            )
            return results

        # ─── Top‑level error handling ───────────────────────────────────────────
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error processing project %s: %s", project_id, exc)
            return []


    # ──────────────────────────────────────────────────────────────────────────────
    def analyze_projects_parallel(
        self,
        result_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run `process_project` for every distinct *project_id* in **parallel**
        using a `ProcessPoolExecutor`, then concatenate all results.

        Parameters
        ----------
        result_df : pd.DataFrame
            A data frame containing at least the column **`project_id`** and
            any other columns required by `process_project`.

        Returns
        -------
        pd.DataFrame
            Aggregated rows from all projects, as produced by
            `process_project`.  If no quality‑attribute mentions are found,
            an empty frame is returned.
        """
        logger.info(
            "Analyzing projects in parallel with %d workers", self.max_workers
        )

        project_ids = result_df["project_id"].unique()
        all_results: List[Dict[str, Any]] = []

        # ─── Launch one future per project ─────────────────────────────────────
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.process_project,
                    pid,
                    result_df[result_df["project_id"] == pid],
                ): pid
                for pid in project_ids
            }

            # ─── Collect results as they finish ───────────────────────────────
            for future in tqdm(
                as_completed(futures),
                total=len(project_ids),
                desc="Projects completed",
            ):
                pid = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    logger.info(
                        "Project %s: added %d quality‑attribute mentions",
                        pid,
                        len(results),
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error("Project %s failed with error: %s", pid, exc)

        return pd.DataFrame(all_results)


    # ──────────────────────────────────────────────────────────────────────────────
    def analyze_projects_sequential(
        self,
        result_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Process every distinct *project_id* **one after another** (no parallelism)
        by delegating to `process_project`, then concatenate all results.

        Parameters
        ----------
        result_df : pd.DataFrame
            A data frame that contains at least the column **`project_id`** and
            any other columns expected by `process_project`.

        Returns
        -------
        pd.DataFrame
            Combined rows from all projects.  If no quality‑attribute mentions
            are found, the returned frame is empty.
        """
        logger.info("Analyzing projects sequentially")

        project_ids = result_df["project_id"].unique()
        all_results: List[Dict[str, Any]] = []

        # ─── Iterate over projects one by one ─────────────────────────────────
        for project_id in tqdm(project_ids, desc="Analyzing projects"):
            project_df = result_df[result_df["project_id"] == project_id]
            try:
                results = self.process_project(project_id, project_df)
                all_results.extend(results)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Project %s failed with error: %s", project_id, exc)

        return pd.DataFrame(all_results)


    # ──────────────────────────────────────────────────────────────────────────────
    def analyze(
        self,
        result_df: pd.DataFrame
        #,quality_attr_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        High‑level entry point: prepare the quality‑attribute resources and run
        the project‑level analysis (parallel or sequential).

        Parameters
        ----------
        result_df : pd.DataFrame
            All raw issue / commit / comment data, with at least the column
            **`project_id`** (plus whatever `process_project` expects).
        quality_attr_df : pd.DataFrame
            Table of quality‑attribute criteria, synonyms, and optional
            similarity scores (used by `prepare_quality_attributes`).

        Returns
        -------
        pd.DataFrame
            Final report with columns:

            * `project_id`
            * `criteria`          – canonical quality attribute
            * `semantic`          – `'+'` (positive) or `'-'` (negative)
            * `similarity_score`  – magnitude of aggregated score
            * `issue_id`
        """
        # 1️⃣  Build lookup tables & embeddings for quality attributes
        # self.prepare_quality_attributes(quality_attr_df)

        # 2️⃣  Choose execution mode
        if self.parallel:
            results_df = self.analyze_projects_parallel(result_df)
        else:
            results_df = self.analyze_projects_sequential(result_df)

        # 3️⃣  Post‑process: tidy column names & order
        if not results_df.empty:
            results_df["id"] = range(len(results_df))  # optional unique row id
            results_df = results_df[
                [
                    "project_id",
                    "quality_attribute",
                    "sentiment",
                    "similarity_score",
                    "issue_id",
                ]
            ]
            results_df.rename(
                columns={
                    "quality_attribute": "criteria",
                    "sentiment": "semantic",
                },
                inplace=True,
            )

        return results_df


    # ──────────────────────────────────────────────────────────────────────────────
    def create_visualizations(
        self,
        results_df: pd.DataFrame,
        output_dir: str | Path = "output/visualizations",
    ) -> None:
        """
        Generate and save four static PNG plots that summarise the analysis:

        1. **Top 15 quality attributes** by mention count.  
        2. **Sentiment distribution** for the top 5 attributes.  
        3. **Histogram of similarity scores** across all matches.  
        4. **Top 10 projects** by total attribute mentions.

        Parameters
        ----------
        results_df : pd.DataFrame
            The final report returned by `analyze`, containing at least the
            columns: `criteria`, `semantic`, `similarity_score`, `project_id`.
        output_dir : str | pathlib.Path, default "output/visualizations"
            Directory where the PNG files will be written (created if absent).

        Notes
        -----
        * Uses **Seaborn** for quick styling on top of Matplotlib.  
        * Each figure is closed after saving to free memory in long runs.
        """
        if results_df.empty:
            return  # Nothing to plot

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ——— 1️⃣  Top 15 quality attributes ————————————————————————————
        plt.figure(figsize=(12, 8))
        top_attrs = results_df["criteria"].value_counts().head(15)
        sns.barplot(x=top_attrs.values, y=top_attrs.index, orient="h")
        plt.title("Top 15 Quality Attributes Mentioned")
        plt.xlabel("Number of Mentions")
        plt.tight_layout()
        plt.savefig(output_dir / "top_quality_attributes.png")
        plt.close()

        # ——— 2️⃣  Sentiment distribution for top 5 attributes ————————
        plt.figure(figsize=(12, 8))
        top5 = results_df["criteria"].value_counts().head(5).index
        sentiment_ct = pd.crosstab(
            results_df.loc[results_df["criteria"].isin(top5), "criteria"],
            results_df.loc[results_df["criteria"].isin(top5), "semantic"],
        )
        sentiment_ct.plot(kind="bar", stacked=True, ax=plt.gca())
        plt.title("Sentiment Distribution for Top 5 Quality Attributes")
        plt.xlabel("Quality Attribute")
        plt.ylabel("Count")
        plt.legend(title="Sentiment")
        plt.tight_layout()
        plt.savefig(output_dir / "sentiment_by_attribute.png")
        plt.close()

        # ——— 3️⃣  Histogram of similarity scores ————————————————
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df["similarity_score"], bins=20)
        plt.title("Distribution of Similarity Scores")
        plt.xlabel("Similarity Score")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "similarity_distribution.png")
        plt.close()

        # ——— 4️⃣  Top 10 projects by mention count ————————————————
        plt.figure(figsize=(12, 8))
        top_projects = results_df["project_id"].value_counts().head(10)
        sns.barplot(
            x=top_projects.values,
            y=top_projects.index.astype(str),
            orient="h",
        )
        plt.title("Top 10 Projects by Quality‑Attribute Mentions")
        plt.xlabel("Number of Mentions")
        plt.tight_layout()
        plt.savefig(output_dir / "top_projects.png")
        plt.close()


# --------------------------------------------------------------------------
def persist_results(results_df, conn):
    """
    Insert rows from *results_df* into the `quality_attribute_analysis` table
    using a single batched INSERT.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have columns:
        ['project_id', 'criteria', 'semantic', 'similarity_score', 'issue_id']
    conn : psycopg2 connection OR SQLAlchemy Engine
    """
    if results_df.empty:
        print("⚠️  No rows to persist — skipping INSERT.", flush=True)
        return

    # ------------------------------------------------------------------
    # 1️⃣  Build the VALUES list  (None for the serial `id` column)
    # ------------------------------------------------------------------
    records = [
        (
            int(row.project_id) if pd.notna(row.project_id) else None,
            row.criteria,
            row.semantic,
            float(row.similarity_score) if pd.notna(row.similarity_score) else None,
            int(row.issue_id) if pd.notna(row.issue_id) else None,
        )
        for row in results_df.itertuples(index=False)
    ]

    insert_sql = """
        INSERT INTO public.quality_attribute_analysis
            (project_id, criteria, semantic, similarity_score, issue_id)
        VALUES %s
    """

    # ------------------------------------------------------------------
    # 2️⃣  Get a DB‑API cursor no matter what kind of `conn` we got
    # ------------------------------------------------------------------
    needs_close = False
    if isinstance(conn, Engine):                     # SQLAlchemy engine
        raw_conn = conn.raw_connection()
        needs_close = True
    else:                                            # psycopg2 connection
        raw_conn = conn

    try:
        with raw_conn.cursor() as cur:
            execute_values(cur, insert_sql, records, page_size=10_000)
        raw_conn.commit()
        print(f"🏆  Stored {len(records):,} results to database.", flush=True)
    finally:
        if needs_close:
            raw_conn.close()


# ──────────────────────────────────────────────────────────────────────────────
def main(
    sim_threshold: float = 0.05,
    batch_size: int = 2048,
    use_gpu: bool = True,
    parallel: bool = False,
    max_workers: int | None = None,
    max_matches_per_project: int = 1000,
    sample_matches: bool = True,
) -> None:
    """
    Command‑line entry point for the *Quality Attribute* analysis pipeline.

    Parameters
    ----------
    sim_threshold : float, default 0.05
        Cosine‑similarity threshold for matching attribute words.
    batch_size : int, default 2048
        Batch size for embedding and sentiment pipelines.
    use_gpu : bool, default True
        Enable GPU inference if a CUDA device is available.
    parallel : bool, default False
        Analyse projects in parallel (`ProcessPoolExecutor`) if True.
    max_workers : int | None, default None
        Worker count for the executor; defaults to CPU‑cores − 1.
    max_matches_per_project : int, default 1000
        Upper bound on stored matches per project (for memory control).
    sample_matches : bool, default True
        If True, sample matches when the cap is exceeded.

    Side‑effects
    ------------
    * Writes **`quality_attribute_analysis.csv`** to *output_dir*.  
    * Writes four PNG plots to *output_dir*/visualizations.  
    * Prints a concise progress log to stdout.
    """
    print("🚀  Starting quality‑attribute analysis pipeline…", flush=True)

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

    # ─── Instantiate analyser ────────────────────────────────────────────────
    analyzer = QualityAttributeAnalyzer(
        similarity_threshold=sim_threshold,
        batch_size=batch_size,
        use_gpu=use_gpu,
        parallel=parallel,
        max_workers=max_workers,
        max_matches_per_project=max_matches_per_project,
        sample_matches=sample_matches,
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
    issue_batch_size = 100

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
            print("⚠️  No more issues to process.", flush=True)
            break

        print(f"🏁  Processing {len(batch_df)} issues...", flush=True)

        # Track the highest issue ID processed
        max_issue_id = batch_df["issue_id"].max()

        # Analyze the batch
        results_df = analyzer.analyze(batch_df)#, quality_attr_df)

        # Append to result table
        persist_results(results_df, conn.connection)

        # Update progress tracker
        with engine.begin() as con:
            con.execute(text("UPDATE quality_attribute_analysis_tracker SET last_issue_id = :new_id"), {"new_id": int(max_issue_id)})

        last_issue_id = max_issue_id

    print("🏁  Pipeline execution complete!", flush=True)    



# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Command‑line interface for the *Quality Attribute* analysis pipeline.

    Example
    -------
    $ python analyze.py \
        --threshold 0.05 \
        --batch-size 1024 \
        --parallel \
        --workers 8
    """
    parser = argparse.ArgumentParser(
        description="Run the quality‑attribute analysis pipeline."
    )

    # Core hyper‑parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Cosine‑similarity threshold for matching (default: 0.05).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for embedding & sentiment inference.",
    )

    # Hardware / execution mode
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration even if CUDA is available.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable per‑project parallel processing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (defaults to CPU‑cores − 1).",
    )

    # Memory / sampling controls
    parser.add_argument(
        "--max-matches",
        type=int,
        default=1000,
        help="Maximum matches stored per project (default: 1000).",
    )
    parser.add_argument(
        "--no-sampling",
        action="store_true",
        help="Disable sampling when max‑matches is exceeded (keep top matches).",
    )

    args = parser.parse_args()

    # ─── Dispatch to the main pipeline ───────────────────────────────────────
    main(
        sim_threshold=args.threshold,
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu,
        parallel=args.parallel,
        max_workers=args.workers,
        max_matches_per_project=args.max_matches,
        sample_matches=not args.no_sampling,
    )
