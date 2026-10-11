"""
TF-IDF Vector Space Model & Cosine Similarity Engine.

Implements vector space indexing, TF-IDF calculation, cosine similarity ranking,
and term-level explainability using Scikit-Learn primitives.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.document_loader import Document
from src.preprocessor import TextPreprocessor


class TfidfSearchEngine:
    """
    Classical Vector Space Model Search Engine using TF-IDF and Cosine Similarity.

    Mathematical Formulation:
    - Term Frequency: TF(t, d) = count(t, d)
    - Inverse Document Frequency: IDF(t, D) = ln((1 + |D|) / (1 + df(t))) + 1
    - Raw TF-IDF Weight: w(t, d) = TF(t, d) * IDF(t, D)
    - L2 Normalization: v_norm = v / ||v||_2
    - Cosine Similarity: cos(theta) = (q . d) / (||q|| * ||d||) = q_norm . d_norm
    """

    def __init__(self, preprocessor: Optional[TextPreprocessor] = None) -> None:
        self.preprocessor = preprocessor or TextPreprocessor()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_matrix: Optional[Any] = None
        self.documents: List[Document] = []
        self.vocabulary: Dict[str, int] = {}
        self.feature_names: List[str] = []
        self.idf_weights: Dict[str, float] = {}

    @property
    def is_fitted(self) -> bool:
        """Check if engine is indexed with documents."""
        return self.vectorizer is not None and self.doc_matrix is not None and len(self.documents) > 0

    @property
    def vocab_size(self) -> int:
        """Total number of unique terms in the indexed vocabulary."""
        return len(self.feature_names)

    def fit(self, documents: List[Document]) -> "TfidfSearchEngine":
        """
        Build the TF-IDF index across the document collection.
        """
        self.documents = list(documents)
        if not self.documents:
            self.vectorizer = None
            self.doc_matrix = None
            self.vocabulary = {}
            self.feature_names = []
            self.idf_weights = {}
            return self

        corpus_texts = [doc.processed_text for doc in self.documents]

        # Use scikit-learn TfidfVectorizer with pre-tokenized whitespace-separated text
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
        )

        try:
            self.doc_matrix = self.vectorizer.fit_transform(corpus_texts)
            self.feature_names = self.vectorizer.get_feature_names_out().tolist()
            self.vocabulary = {term: idx for idx, term in enumerate(self.feature_names)}

            # Extract IDF weights
            idf_values = self.vectorizer.idf_
            self.idf_weights = {
                term: float(idf_values[idx])
                for term, idx in self.vocabulary.items()
            }
        except ValueError:
            # Occurs if documents are entirely empty
            self.vectorizer = None
            self.doc_matrix = None
            self.feature_names = []
            self.vocabulary = {}
            self.idf_weights = {}

        return self

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0001,
    ) -> List[Dict[str, Any]]:
        """
        Execute TF-IDF Vector Space search against the indexed documents.

        Returns ranked list of result dictionaries containing:
        - document: Document instance
        - score: Cosine similarity score [0.0, 1.0]
        - rank: 1-indexed rank position
        - query_tokens: list of preprocessed query tokens
        - matched_terms: query terms present in the document
        - top_contributing_terms: terms that contributed most to the similarity score
        """
        if not self.is_fitted or not query or not query.strip():
            return []

        # 1. Preprocess query text
        query_tokens = self.preprocessor.preprocess(query)
        if not query_tokens:
            return []

        processed_query_str = " ".join(query_tokens)

        # 2. Project query into vector space
        query_vec = self.vectorizer.transform([processed_query_str])

        # Check if query is entirely out of vocabulary (zero vector)
        if query_vec.nnz == 0:
            return []

        # 3. Compute Cosine Similarity between query vector and all document vectors
        # Note: Since both vectors are L2-normalized, cosine similarity is the dot product.
        similarity_scores = cosine_similarity(query_vec, self.doc_matrix).flatten()

        # 4. Rank documents by descending score
        sorted_indices = np.argsort(similarity_scores)[::-1]

        results: List[Dict[str, Any]] = []
        rank = 1

        for idx in sorted_indices:
            score = float(similarity_scores[idx])
            if score < min_score:
                break

            doc = self.documents[idx]
            doc_vec = self.doc_matrix[idx]

            # 5. Calculate term-level contributions: (q_i * d_i)
            # Element-wise product of query and document TF-IDF vectors
            contrib_sparse = query_vec.multiply(doc_vec)
            term_indices = contrib_sparse.indices
            term_values = contrib_sparse.data

            contributions = []
            for t_idx, val in zip(term_indices, term_values):
                term_name = self.feature_names[t_idx]
                contributions.append((term_name, float(val)))

            # Sort contributions by impact
            contributions.sort(key=lambda x: x[1], reverse=True)
            matched_terms = [t[0] for t in contributions]

            results.append({
                "document": doc,
                "score": score,
                "rank": rank,
                "query_tokens": query_tokens,
                "matched_terms": matched_terms,
                "top_contributions": contributions[:5],  # top 5 contributing terms
                "doc_index": int(idx),
            })

            rank += 1
            if len(results) >= top_k:
                break

        return results

    def get_term_idf(self, term: str) -> Optional[float]:
        """Get IDF weight for a given term (stemmed)."""
        stemmed = self.preprocessor.preprocess_as_text(term)
        return self.idf_weights.get(stemmed)

    def get_top_idf_terms(self, n: int = 15, highest: bool = True) -> List[Tuple[str, float]]:
        """
        Return the top N terms with highest (rarest) or lowest (most common) IDF weights.
        """
        if not self.idf_weights:
            return []
        sorted_terms = sorted(self.idf_weights.items(), key=lambda x: x[1], reverse=highest)
        return sorted_terms[:n]
