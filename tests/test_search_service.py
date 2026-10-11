"""
Unit Tests for Search Service, Highlighting, and History Tracking.
"""

import pytest
from src.document_loader import load_document_from_source
from src.keyword_engine import KeywordSearchEngine
from src.preprocessor import TextPreprocessor
from src.search_service import (
    SearchHistoryManager,
    SearchService,
    extract_smart_snippet,
    highlight_query_terms,
)
from src.tfidf_engine import TfidfSearchEngine


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


@pytest.fixture
def search_service(preprocessor):
    docs = [
        load_document_from_source(
            "Natural language processing helps computers interpret language. Tokenization and stemming are vital steps.",
            "nlp.txt",
            preprocessor=preprocessor,
        ),
        load_document_from_source(
            "Computer vision analyzes images and videos using convolutional neural networks.",
            "cv.txt",
            preprocessor=preprocessor,
        ),
    ]
    tfidf = TfidfSearchEngine(preprocessor=preprocessor).fit(docs)
    kw = KeywordSearchEngine(preprocessor=preprocessor).fit(docs)
    return SearchService(tfidf_engine=tfidf, keyword_engine=kw, preprocessor=preprocessor)


def test_snippet_extraction_and_highlighting(preprocessor):
    text = "First paragraph here. Natural language processing is very effective. Third paragraph here."
    query_tokens = preprocessor.preprocess("language processing")

    snippet = extract_smart_snippet(text, query_tokens, preprocessor)
    assert "Natural language processing" in snippet

    highlighted = highlight_query_terms(snippet, query_tokens, preprocessor)
    assert '<mark class="nlp-highlight">' in highlighted
    assert "</mark>" in highlighted


def test_search_tfidf_service(search_service):
    res = search_service.search_tfidf("natural language", top_k=2)
    assert res["status"] == "success"
    assert len(res["results"]) > 0
    assert res["results"][0]["filename"] == "nlp.txt"
    assert "highlighted_snippet" in res["results"][0]


def test_search_history_manager():
    mgr = SearchHistoryManager(max_items=3)
    mgr.add_entry("test query 1", "TF-IDF", 2, "Doc 1", 0.85)
    mgr.add_entry("test query 2", "TF-IDF", 1, "Doc 2", 0.65)

    hist = mgr.get_history()
    assert len(hist) == 2
    assert hist[0]["query"] == "test query 2"

    mgr.clear()
    assert len(mgr.get_history()) == 0


def test_compare_search(search_service):
    comp = search_service.compare_search("natural language", top_k=2)
    assert "tfidf_results" in comp
    assert "keyword_results" in comp
    assert "rank_shifts" in comp
    assert "analysis" in comp
