"""
Corpus State Management and Analytics Engine.

Maintains the active document collection, orchestrates engine re-indexing,
handles file validation and deduplication, and generates corpus statistics
using Pandas.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.document_loader import (
    Document,
    DuplicateDocumentError,
    EmptyDocumentError,
    load_document_from_source,
)
from src.keyword_engine import KeywordSearchEngine
from src.preprocessor import TextPreprocessor
from src.tfidf_engine import TfidfSearchEngine


class CorpusManager:
    """
    Central Corpus Manager maintaining collection state and analytics.
    """

    def __init__(
        self,
        tfidf_engine: TfidfSearchEngine,
        keyword_engine: KeywordSearchEngine,
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> None:
        self.tfidf_engine = tfidf_engine
        self.keyword_engine = keyword_engine
        self.preprocessor = preprocessor or TextPreprocessor()
        self.documents: List[Document] = []
        self._doc_map: Dict[str, Document] = {}  # doc_id -> Document
        self._hash_set: set[str] = set()

    @property
    def document_count(self) -> int:
        return len(self.documents)

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self._doc_map.get(doc_id)

    def get_all_documents(self) -> List[Document]:
        return list(self.documents)

    def add_document_from_source(
        self,
        source: Union[str, bytes],
        filename: str,
        file_type: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[Document]]:
        """
        Ingest a single document from file path or buffer.
        Returns (success: bool, message: str, doc: Optional[Document]).
        """
        try:
            doc = load_document_from_source(
                source=source,
                filename=filename,
                preprocessor=self.preprocessor,
                file_type=file_type,
            )

            # Deduplication check
            if doc.content_hash in self._hash_set:
                existing_doc = next((d for d in self.documents if d.content_hash == doc.content_hash), None)
                msg = f"Duplicate detected: '{filename}' has identical content to '{existing_doc.filename if existing_doc else 'existing'}'."
                return False, msg, None

            # Add document
            self.documents.append(doc)
            self._doc_map[doc.doc_id] = doc
            self._hash_set.add(doc.content_hash)

            # Re-fit search engines
            self._refit_engines()

            return True, f"Successfully indexed '{doc.title}' ({doc.word_count} words).", doc

        except EmptyDocumentError as e:
            return False, f"Empty document error: {str(e)}", None
        except Exception as e:
            return False, f"Failed to ingest '{filename}': {str(e)}", None

    def remove_document(self, doc_id: str) -> Tuple[bool, str]:
        """Remove a document by its doc_id."""
        doc = self._doc_map.get(doc_id)
        if not doc:
            return False, f"Document with ID '{doc_id}' not found."

        self.documents = [d for d in self.documents if d.doc_id != doc_id]
        self._doc_map.pop(doc_id, None)
        self._hash_set.discard(doc.content_hash)

        self._refit_engines()
        return True, f"Removed document '{doc.title}'."

    def clear_corpus(self) -> None:
        """Clear all documents from corpus."""
        self.documents.clear()
        self._doc_map.clear()
        self._hash_set.clear()
        self._refit_engines()

    def _refit_engines(self) -> None:
        """Update both TF-IDF and Keyword search engine models."""
        self.tfidf_engine.fit(self.documents)
        self.keyword_engine.fit(self.documents)

    def load_sample_corpus(self, sample_dir: str) -> Tuple[int, List[str]]:
        """
        Load all sample documents from a directory.
        Returns (loaded_count, status_messages).
        """
        if not os.path.exists(sample_dir):
            return 0, [f"Directory not found: {sample_dir}"]

        messages = []
        loaded_count = 0

        # Sort files for deterministic ordering
        entries = sorted(os.listdir(sample_dir))
        for entry in entries:
            file_path = os.path.join(sample_dir, entry)
            if os.path.isfile(file_path):
                ext = os.path.splitext(entry)[1].lower()
                if ext in (".txt", ".pdf", ".md"):
                    success, msg, _ = self.add_document_from_source(file_path, entry)
                    messages.append(msg)
                    if success:
                        loaded_count += 1

        return loaded_count, messages

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculate aggregate corpus metrics using Pandas.
        """
        if not self.documents:
            return {
                "total_documents": 0,
                "total_raw_words": 0,
                "total_processed_tokens": 0,
                "vocabulary_size": 0,
                "avg_words_per_doc": 0.0,
                "avg_tokens_per_doc": 0.0,
                "min_words": 0,
                "max_words": 0,
            }

        df = pd.DataFrame([d.to_dict() for d in self.documents])

        all_tokens = [tok for d in self.documents for tok in d.processed_tokens]
        unique_tokens = set(all_tokens)

        return {
            "total_documents": len(self.documents),
            "total_raw_words": int(df["word_count"].sum()),
            "total_processed_tokens": len(all_tokens),
            "vocabulary_size": len(unique_tokens),
            "avg_words_per_doc": round(float(df["word_count"].mean()), 1),
            "avg_tokens_per_doc": round(float(df["token_count"].mean()), 1),
            "min_words": int(df["word_count"].min()),
            "max_words": int(df["word_count"].max()),
        }

    def get_documents_dataframe(self) -> pd.DataFrame:
        """
        Return a clean summary DataFrame of all indexed documents.
        """
        if not self.documents:
            return pd.DataFrame(
                columns=["Document Title", "Type", "Words", "Stems", "Preview", "Document ID"]
            )

        data = []
        for d in self.documents:
            data.append({
                "Document Title": d.title,
                "Type": d.file_type.upper(),
                "Words": d.word_count,
                "Stems": d.token_count,
                "Preview": d.get_preview(160),
                "Document ID": d.doc_id,
            })
        return pd.DataFrame(data)

    def get_top_frequent_terms(self, top_n: int = 15) -> pd.DataFrame:
        """
        Compute the most frequent terms across the entire corpus.
        Returns DataFrame with Term, Total Occurrences, and Document Frequency (DF).
        """
        if not self.documents:
            return pd.DataFrame(columns=["Term", "Total Frequency", "Document Frequency (DF)", "Corpus Share (%)"])

        corpus_counter: Counter[str] = Counter()
        doc_freq_counter: Counter[str] = Counter()

        for d in self.documents:
            unique_in_doc = set(d.processed_tokens)
            corpus_counter.update(d.processed_tokens)
            doc_freq_counter.update(unique_in_doc)

        total_tokens = sum(corpus_counter.values())
        top_terms = corpus_counter.most_common(top_n)

        rows = []
        for term, count in top_terms:
            df_val = doc_freq_counter[term]
            share = (count / total_tokens * 100.0) if total_tokens > 0 else 0.0
            rows.append({
                "Term": term,
                "Total Frequency": count,
                "Document Frequency (DF)": f"{df_val} / {len(self.documents)}",
                "Corpus Share (%)": round(share, 2),
            })

        return pd.DataFrame(rows)

    def get_idf_rankings_dataframe(self, top_n: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return DataFrames for the highest IDF (most distinctive)
        and lowest IDF (most ubiquitous) terms.
        """
        high_idf = self.tfidf_engine.get_top_idf_terms(top_n, highest=True)
        low_idf = self.tfidf_engine.get_top_idf_terms(top_n, highest=False)

        high_df = pd.DataFrame(
            [{"Term": t, "IDF Weight": round(w, 4), "Interpretation": "Highly Discriminative"} for t, w in high_idf]
        ) if high_idf else pd.DataFrame(columns=["Term", "IDF Weight", "Interpretation"])

        low_df = pd.DataFrame(
            [{"Term": t, "IDF Weight": round(w, 4), "Interpretation": "Common across Corpus"} for t, w in low_idf]
        ) if low_idf else pd.DataFrame(columns=["Term", "IDF Weight", "Interpretation"])

        return high_df, low_df
