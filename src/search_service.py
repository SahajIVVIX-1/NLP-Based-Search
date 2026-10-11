"""
Search Service, Snippet Extraction, Term Highlighting, and History Management.

Provides high-level search execution, smart snippet generation with context-aware
query term highlighting, side-by-side search comparison, and search audit history.
"""

from __future__ import annotations

import html
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.document_loader import Document
from src.keyword_engine import KeywordSearchEngine
from src.preprocessor import TextPreprocessor
from src.tfidf_engine import TfidfSearchEngine


def extract_smart_snippet(
    raw_text: str,
    query_tokens: List[str],
    preprocessor: TextPreprocessor,
    max_length: int = 320,
) -> str:
    """
    Extract the most relevant sentence or window containing query terms.
    If no matches are present, returns the opening text.
    """
    if not raw_text or not raw_text.strip():
        return ""

    # Split into sentences or lines
    raw_sentences = re.split(r"(?<=[.!?\n])\s+", raw_text.strip())
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10]

    if not sentences:
        clean = " ".join(raw_text.split())
        return clean[:max_length] + ("..." if len(clean) > max_length else "")

    query_stem_set = set(query_tokens)

    # Score each sentence by query stem occurrences
    scored_sentences: List[Tuple[int, int, str]] = []  # (score, index, sentence)

    for idx, sentence in enumerate(sentences):
        sent_tokens = preprocessor.preprocess(sentence)
        match_count = sum(1 for tok in sent_tokens if tok in query_stem_set)
        scored_sentences.append((match_count, idx, sentence))

    # Sort descending by match score
    scored_sentences.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    best_score, best_idx, best_sent = scored_sentences[0]

    if best_score > 0:
        # Build window around best sentence if space permits
        snippet = best_sent
        if len(snippet) < max_length and best_idx + 1 < len(sentences):
            next_sent = sentences[best_idx + 1]
            if len(snippet) + len(next_sent) + 1 <= max_length:
                snippet = f"{snippet} {next_sent}"

        if len(snippet) > max_length:
            snippet = snippet[:max_length].rstrip() + "..."
        return snippet

    # Default fallback: first sentence or first characters
    fallback = sentences[0]
    if len(fallback) > max_length:
        fallback = fallback[:max_length].rstrip() + "..."
    return fallback


def highlight_query_terms(
    snippet: str,
    query_tokens: List[str],
    preprocessor: TextPreprocessor,
) -> str:
    """
    Safely highlight matching words in the snippet using HTML <mark> tags.
    Matches words whose stems align with the preprocessed query tokens.
    """
    if not snippet or not query_tokens:
        return html.escape(snippet)

    query_stem_set = set(query_tokens)

    # Tokenize words while preserving boundary delimiters
    tokens_and_delims = re.split(r"(\b[a-zA-Z0-9_-]+\b)", snippet)
    escaped_parts: List[str] = []

    for part in tokens_and_delims:
        if not part:
            continue

        # If it's a word
        if re.match(r"^[a-zA-Z0-9_-]+$", part):
            part_stems = preprocessor.preprocess(part)
            if any(st in query_stem_set for st in part_stems):
                # Highlighted term
                escaped_word = html.escape(part)
                escaped_parts.append(f'<mark class="nlp-highlight">{escaped_word}</mark>')
            else:
                escaped_parts.append(html.escape(part))
        else:
            # Punctuation/whitespace
            escaped_parts.append(html.escape(part))

    return "".join(escaped_parts)


class SearchHistoryManager:
    """Tracks search queries and results for the audit panel."""

    def __init__(self, max_items: int = 50) -> None:
        self.max_items = max_items
        self._history: List[Dict[str, Any]] = []

    def add_entry(
        self,
        query: str,
        engine_type: str,
        result_count: int,
        top_doc_title: Optional[str] = None,
        top_score: Optional[float] = None,
    ) -> None:
        """Record a search action."""
        if not query or not query.strip():
            return

        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query.strip(),
            "engine": engine_type,
            "results_found": result_count,
            "top_document": top_doc_title or "None",
            "top_score": f"{top_score:.4f}" if top_score is not None else "N/A",
        }
        self._history.insert(0, entry)
        if len(self._history) > self.max_items:
            self._history.pop()

    def get_history(self) -> List[Dict[str, Any]]:
        """Return history items."""
        return list(self._history)

    def clear(self) -> None:
        """Clear all search history."""
        self._history.clear()


class SearchService:
    """
    Coordinating service for TF-IDF search, keyword comparison,
    rich result formatting, and history tracking.
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
        self.history = SearchHistoryManager()

    def search_tfidf(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute TF-IDF search with snippets, highlighting, and explainability.
        """
        clean_query = (query or "").strip()
        if not clean_query:
            return {
                "query": "",
                "status": "empty_query",
                "message": "Please enter a non-empty search query.",
                "results": [],
                "query_tokens": [],
            }

        query_tokens = self.preprocessor.preprocess(clean_query)
        if not query_tokens:
            return {
                "query": clean_query,
                "status": "no_valid_tokens",
                "message": "Query consists only of stopwords or non-alphanumeric noise.",
                "results": [],
                "query_tokens": [],
            }

        raw_results = self.tfidf_engine.search(clean_query, top_k=top_k)

        formatted_results: List[Dict[str, Any]] = []
        for res in raw_results:
            doc: Document = res["document"]
            snippet = extract_smart_snippet(doc.raw_text, query_tokens, self.preprocessor)
            highlighted_snippet = highlight_query_terms(snippet, query_tokens, self.preprocessor)

            formatted_results.append({
                "rank": res["rank"],
                "doc_id": doc.doc_id,
                "title": doc.title,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "score": round(res["score"], 4),
                "score_percent": f"{res['score'] * 100:.1f}%",
                "snippet": snippet,
                "highlighted_snippet": highlighted_snippet,
                "word_count": doc.word_count,
                "matched_terms": res["matched_terms"],
                "top_contributions": res["top_contributions"],
            })

        # Record history
        top_title = formatted_results[0]["title"] if formatted_results else None
        top_score = formatted_results[0]["score"] if formatted_results else None
        self.history.add_entry(clean_query, "TF-IDF", len(formatted_results), top_title, top_score)

        return {
            "query": clean_query,
            "status": "success",
            "message": f"Found {len(formatted_results)} matching document(s).",
            "results": formatted_results,
            "query_tokens": query_tokens,
            "total_matches": len(formatted_results),
        }

    def search_keyword(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute basic lexical keyword search.
        """
        clean_query = (query or "").strip()
        if not clean_query:
            return {
                "query": "",
                "status": "empty_query",
                "message": "Please enter a non-empty search query.",
                "results": [],
                "query_tokens": [],
            }

        query_tokens = self.preprocessor.preprocess(clean_query)
        if not query_tokens:
            return {
                "query": clean_query,
                "status": "no_valid_tokens",
                "message": "Query consists only of stopwords or symbols.",
                "results": [],
                "query_tokens": [],
            }

        raw_results = self.keyword_engine.search(clean_query, top_k=top_k)

        formatted_results: List[Dict[str, Any]] = []
        for res in raw_results:
            doc: Document = res["document"]
            snippet = extract_smart_snippet(doc.raw_text, query_tokens, self.preprocessor)
            highlighted_snippet = highlight_query_terms(snippet, query_tokens, self.preprocessor)

            formatted_results.append({
                "rank": res["rank"],
                "doc_id": doc.doc_id,
                "title": doc.title,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "score": round(res["score"], 4),
                "raw_match_count": res["raw_match_count"],
                "matched_terms_count": res["matched_terms_count"],
                "query_coverage": f"{res['query_coverage'] * 100:.1f}%",
                "snippet": snippet,
                "highlighted_snippet": highlighted_snippet,
                "matched_terms": res["matched_terms"],
                "term_occurrences": res["term_occurrences"],
            })

        return {
            "query": clean_query,
            "status": "success",
            "message": f"Found {len(formatted_results)} matching document(s).",
            "results": formatted_results,
            "query_tokens": query_tokens,
            "total_matches": len(formatted_results),
        }

    def compare_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform side-by-side search comparison with analytical commentary.
        """
        tfidf_resp = self.search_tfidf(query, top_k=top_k)
        keyword_resp = self.search_keyword(query, top_k=top_k)

        # Log audit history for comparison
        clean_query = (query or "").strip()
        top_title = tfidf_resp["results"][0]["title"] if tfidf_resp["results"] else None
        top_score = tfidf_resp["results"][0]["score"] if tfidf_resp["results"] else None
        self.history.add_entry(clean_query, "Comparison Mode", len(tfidf_resp["results"]), top_title, top_score)

        # Compare ranking sets
        tfidf_docs = [r["title"] for r in tfidf_resp["results"]]
        kw_docs = [r["title"] for r in keyword_resp["results"]]

        rank_shifts = []
        for r_idx, title in enumerate(tfidf_docs):
            if title in kw_docs:
                kw_idx = kw_docs.index(title)
                shift = kw_idx - r_idx  # positive = ranked higher in TF-IDF
                rank_shifts.append({
                    "title": title,
                    "tfidf_rank": r_idx + 1,
                    "kw_rank": kw_idx + 1,
                    "shift": shift,
                })
            else:
                rank_shifts.append({
                    "title": title,
                    "tfidf_rank": r_idx + 1,
                    "kw_rank": "Unranked",
                    "shift": None,
                })

        return {
            "query": clean_query,
            "query_tokens": tfidf_resp.get("query_tokens", []),
            "tfidf_results": tfidf_resp["results"],
            "keyword_results": keyword_resp["results"],
            "rank_shifts": rank_shifts,
            "analysis": self._generate_comparison_analysis(clean_query, tfidf_resp, keyword_resp),
        }

    def _generate_comparison_analysis(
        self,
        query: str,
        tfidf_resp: Dict[str, Any],
        kw_resp: Dict[str, Any],
    ) -> str:
        """Generate pedagogical analysis comparing the two approaches."""
        if not tfidf_resp["results"] and not kw_resp["results"]:
            return "Neither search method retrieved matching documents for this query."

        points = []
        points.append(
            f"**Query Processed:** `{query}` → Preprocessed Tokens: `{tfidf_resp.get('query_tokens', [])}`"
        )

        top_tfidf = tfidf_resp["results"][0] if tfidf_resp["results"] else None
        top_kw = kw_resp["results"][0] if kw_resp["results"] else None

        if top_tfidf and top_kw:
            if top_tfidf["title"] == top_kw["title"]:
                points.append(
                    f"**Consensus Top Result:** Both methods agreed on `{top_tfidf['title']}` as the most relevant document."
                )
            else:
                points.append(
                    f"**Ranking Divergence:** TF-IDF prioritized `{top_tfidf['title']}` (cosine similarity: {top_tfidf['score']}), "
                    f"while basic keyword matching prioritized `{top_kw['title']}` (raw matches: {top_kw.get('raw_match_count', 'N/A')})."
                )

        points.append(
            "**Key Mechanism Distinction:** Basic keyword matching ranks purely on literal word frequencies and query token coverage, "
            "making it vulnerable to document length bias. In contrast, TF-IDF penalizes ubiquitous corpus terms via IDF weights "
            "and applies L2 normalization to neutralize length disparities."
        )

        return "\n\n".join(points)
