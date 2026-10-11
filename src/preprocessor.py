"""
Text Preprocessing Module for the NLP Document Search Engine.

Handles text normalization, tokenization, stopword removal, and morphological
stemming using NLTK tools. Provides a step-by-step pipeline inspector for
academic demonstration and student learning.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import RegexpTokenizer, word_tokenize


def ensure_nltk_resources() -> None:
    """Ensure essential NLTK data models are present."""
    for resource in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except (LookupError, OSError):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


# Initialize resources on module load
ensure_nltk_resources()


class TextPreprocessor:
    """
    Configurable Classical NLP Preprocessing Pipeline.

    Stages:
    1. Case Normalization & Whitespace Cleansing
    2. Tokenization (word-level extraction)
    3. Noise & Punctuation Filtering
    4. Stopword Elimination (NLTK English lexicon)
    5. Morphological Stemming (Porter or Snowball algorithm)
    """

    def __init__(
        self,
        use_stemming: bool = True,
        remove_stopwords: bool = True,
        lowercase: bool = True,
        min_token_len: int = 2,
        stemmer_type: str = "porter",
    ) -> None:
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.min_token_len = min_token_len

        # Stopwords setup
        try:
            self._stopwords: Set[str] = set(stopwords.words("english"))
        except Exception:
            self._stopwords = {
                "a", "about", "above", "after", "again", "against", "all", "am",
                "an", "and", "any", "are", "aren't", "as", "at", "be", "because",
                "been", "before", "being", "below", "between", "both", "but", "by",
                "can", "could", "did", "do", "does", "doing", "down", "during",
                "each", "few", "for", "from", "further", "had", "has", "have",
                "having", "he", "her", "here", "hers", "herself", "him", "himself",
                "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its",
                "itself", "just", "me", "more", "most", "my", "myself", "no", "nor",
                "not", "now", "of", "off", "on", "once", "only", "or", "other",
                "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
                "should", "so", "some", "such", "than", "that", "the", "their",
                "theirs", "them", "themselves", "then", "there", "these", "they",
                "this", "those", "through", "to", "too", "under", "until", "up",
                "very", "was", "we", "were", "what", "when", "where", "which",
                "while", "who", "whom", "why", "will", "with", "would", "you", "your"
            }

        # Stemmer setup
        if stemmer_type.lower() == "snowball":
            self.stemmer = SnowballStemmer("english")
        else:
            self.stemmer = PorterStemmer()

        # Regex fallback tokenizer
        self.regex_tokenizer = RegexpTokenizer(r"\b[a-zA-Z0-9_-]+\b")

    @property
    def stopword_list(self) -> List[str]:
        """Return sorted list of active stopwords."""
        return sorted(list(self._stopwords))

    def normalize(self, text: str) -> str:
        """Normalize whitespace and optionally case fold."""
        if not text:
            return ""
        normalized = text.strip()
        if self.lowercase:
            normalized = normalized.lower()
        return normalized

    def tokenize(self, text: str) -> List[str]:
        """Tokenize string into individual word tokens."""
        if not text:
            return []
        norm_text = self.normalize(text)
        try:
            tokens = word_tokenize(norm_text)
        except Exception:
            tokens = self.regex_tokenizer.tokenize(norm_text)
        return tokens

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """Filter out punctuation, non-alphanumeric tokens, and short words."""
        cleaned: List[str] = []
        for token in tokens:
            # Strip outer punctuation
            clean_tok = re.sub(r"^[^\w]+|[^\w]+$", "", token)
            if not clean_tok:
                continue
            # Keep tokens with alphanumeric characters and minimum length
            if any(c.isalnum() for c in clean_tok) and len(clean_tok) >= self.min_token_len:
                cleaned.append(clean_tok)
        return cleaned

    def filter_stopwords(self, tokens: List[str]) -> Tuple[List[str], List[str]]:
        """
        Filter out stopwords.
        Returns (retained_tokens, removed_stopwords).
        """
        if not self.remove_stopwords:
            return tokens, []

        retained: List[str] = []
        removed: List[str] = []
        for token in tokens:
            if token.lower() in self._stopwords:
                removed.append(token)
            else:
                retained.append(token)
        return retained, removed

    def stem_tokens(self, tokens: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Stem tokens using the configured morphological stemmer.
        Returns (stemmed_tokens, transformations).
        """
        if not self.use_stemming:
            return tokens, []

        stemmed: List[str] = []
        transformations: List[Tuple[str, str]] = []
        for token in tokens:
            stem = self.stemmer.stem(token)
            stemmed.append(stem)
            if stem != token:
                transformations.append((token, stem))
        return stemmed, transformations

    def preprocess(self, text: str) -> List[str]:
        """
        Run complete preprocessing pipeline.
        Returns final list of processed tokens (stems).
        """
        if not text or not text.strip():
            return []

        tokens = self.tokenize(text)
        filtered = self.filter_tokens(tokens)
        retained, _ = self.filter_stopwords(filtered)
        stemmed, _ = self.stem_tokens(retained)
        return stemmed

    def preprocess_as_text(self, text: str) -> str:
        """Return space-joined processed tokens suitable for vectorizers."""
        tokens = self.preprocess(text)
        return " ".join(tokens)

    def inspect_pipeline_stages(self, text: str) -> Dict[str, Any]:
        """
        Detailed step-by-step breakdown of every preprocessing stage.
        Ideal for UI visualization and pedagogical demonstrations.
        """
        raw_text = text if text else ""
        normalized = self.normalize(raw_text)

        tokens = self.tokenize(raw_text)
        filtered = self.filter_tokens(tokens)
        retained, removed_stopwords = self.filter_stopwords(filtered)
        stemmed, transformations = self.stem_tokens(retained)

        return {
            "raw_text": raw_text,
            "normalized_text": normalized,
            "raw_tokens": tokens,
            "raw_token_count": len(tokens),
            "filtered_tokens": filtered,
            "filtered_token_count": len(filtered),
            "stopwords_removed": removed_stopwords,
            "stopword_count": len(removed_stopwords),
            "tokens_after_stopwords": retained,
            "retained_count": len(retained),
            "stemmed_tokens": stemmed,
            "stemmed_count": len(stemmed),
            "stemming_transformations": transformations,
            "final_processed_text": " ".join(stemmed),
        }
