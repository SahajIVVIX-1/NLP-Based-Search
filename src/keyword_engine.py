"""
Basic Keyword & Lexical Matching Engine.

Implements raw term frequency and Boolean/Jaccard overlap scoring
to enable direct pedagogical comparison against TF-IDF search.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from src.document_loader import Document
from src.preprocessor import TextPreprocessor


class KeywordSearchEngine:
    """
    Classical Keyword Matching Search Engine (Non-TF-IDF Baseline).

    Scoring Mechanisms:
    1. Term Occurrence Count: Raw frequency sum of query words in document.
    2. Query Term Coverage (Overlap Ratio): |Query_terms ∩ Doc_terms| / |Query_terms|
    3. Combined Keyword Score: normalized metric for side-by-side comparison.

    Pedagogical Importance:
    Illustrates the flaws of unweighted lexical search:
    - Highly repetitive documents score artificially high.
    - Common words dominate rare informative keywords.
    - Document length bias (longer documents naturally accumulate more matches).
    """

    def __init__(self, preprocessor: Optional[TextPreprocessor] = None) -> None:
        self.preprocessor = preprocessor or TextPreprocessor()
        self.documents: List[Document] = []
        self.doc_term_freqs: List[Counter[str]] = []
        self.doc_term_sets: List[set[str]] = []

    @property
    def is_fitted(self) -> bool:
        return len(self.documents) > 0

    def fit(self, documents: List[Document]) -> "KeywordSearchEngine":
        """Index documents by building term frequencies and unique sets."""
        self.documents = list(documents)
        self.doc_term_freqs = []
        self.doc_term_sets = []

        for doc in self.documents:
            freq = Counter(doc.processed_tokens)
            self.doc_term_freqs.append(freq)
            self.doc_term_sets.append(set(doc.processed_tokens))

        return self

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_matches: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Execute basic keyword matching against documents.
        """
        if not self.is_fitted or not query or not query.strip():
            return []

        query_tokens = self.preprocessor.preprocess(query)
        if not query_tokens:
            return []

        unique_query_tokens = set(query_tokens)
        query_len = len(unique_query_tokens)

        scored_results: List[Dict[str, Any]] = []

        for idx, doc in enumerate(self.documents):
            term_freq = self.doc_term_freqs[idx]
            term_set = self.doc_term_sets[idx]

            # 1. Matched distinct terms
            matched_terms = [t for t in unique_query_tokens if t in term_set]
            overlap_count = len(matched_terms)

            if overlap_count < min_matches:
                continue

            # 2. Raw total occurrences
            total_occurrences = sum(term_freq[t] for t in matched_terms)

            # 3. Term coverage ratio [0.0 to 1.0]
            coverage_ratio = overlap_count / query_len if query_len > 0 else 0.0

            # 4. Keyword score combining coverage (priority) and normalized frequency
            # Score formula: coverage * 0.7 + min(1.0, occurrences / (doc.token_count or 1) * 10) * 0.3
            freq_density = (total_occurrences / max(1, doc.token_count)) * 10.0
            keyword_score = round((coverage_ratio * 0.75) + (min(1.0, freq_density) * 0.25), 4)

            scored_results.append({
                "document": doc,
                "score": keyword_score,
                "raw_match_count": total_occurrences,
                "matched_terms_count": overlap_count,
                "query_coverage": coverage_ratio,
                "query_tokens": query_tokens,
                "matched_terms": matched_terms,
                "term_occurrences": {t: term_freq[t] for t in matched_terms},
                "doc_index": idx,
            })

        # Rank by: 1) coverage ratio, 2) total occurrences, 3) score
        scored_results.sort(
            key=lambda x: (x["matched_terms_count"], x["raw_match_count"], x["score"]),
            reverse=True,
        )

        # Assign ranks
        for r_idx, res in enumerate(scored_results):
            res["rank"] = r_idx + 1

        return scored_results[:top_k]
