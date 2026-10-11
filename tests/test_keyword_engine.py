"""
Unit Tests for Keyword Matching Engine.
"""

import pytest
from src.document_loader import load_document_from_source
from src.keyword_engine import KeywordSearchEngine
from src.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


@pytest.fixture
def sample_documents(preprocessor):
    docs_data = [
        ("Python programming language is versatile and popular for web and AI development.", "python.txt"),
        ("Database query optimization using SQL indexes and relational algebra.", "db.txt"),
    ]
    return [
        load_document_from_source(text, name, preprocessor=preprocessor)
        for text, name in docs_data
    ]


def test_keyword_search(preprocessor, sample_documents):
    kw_engine = KeywordSearchEngine(preprocessor=preprocessor)
    kw_engine.fit(sample_documents)

    results = kw_engine.search("python programming language", top_k=5)
    assert len(results) == 1
    assert results[0]["document"].filename == "python.txt"
    assert results[0]["raw_match_count"] >= 3
    assert results[0]["matched_terms_count"] >= 2


def test_keyword_search_no_match(preprocessor, sample_documents):
    kw_engine = KeywordSearchEngine(preprocessor=preprocessor)
    kw_engine.fit(sample_documents)

    results = kw_engine.search("astronomy telescope stars")
    assert len(results) == 0
