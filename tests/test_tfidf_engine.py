"""
Unit Tests for TF-IDF Vector Space Search Engine.
"""

import pytest
from src.document_loader import load_document_from_source
from src.preprocessor import TextPreprocessor
from src.tfidf_engine import TfidfSearchEngine


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


@pytest.fixture
def sample_documents(preprocessor):
    docs_data = [
        ("Information retrieval systems index documents using inverted indices and tf-idf.", "doc1.txt"),
        ("Deep learning and neural networks use gradient descent and backpropagation.", "doc2.txt"),
        ("Natural language processing combines computational linguistics with machine learning.", "doc3.txt"),
    ]
    return [
        load_document_from_source(text, name, preprocessor=preprocessor)
        for text, name in docs_data
    ]


def test_fit_and_vocab_size(preprocessor, sample_documents):
    engine = TfidfSearchEngine(preprocessor=preprocessor)
    assert not engine.is_fitted
    engine.fit(sample_documents)
    assert engine.is_fitted
    assert engine.vocab_size > 0
    assert len(engine.idf_weights) == engine.vocab_size


def test_cosine_similarity_search(preprocessor, sample_documents):
    engine = TfidfSearchEngine(preprocessor=preprocessor)
    engine.fit(sample_documents)

    # Search for terms specific to doc1
    results = engine.search("information retrieval indexing", top_k=2)
    assert len(results) > 0
    top_result = results[0]
    assert top_result["document"].filename == "doc1.txt"
    assert 0.0 < top_result["score"] <= 1.0
    assert top_result["rank"] == 1
    assert len(top_result["top_contributions"]) > 0


def test_empty_or_oov_queries(preprocessor, sample_documents):
    engine = TfidfSearchEngine(preprocessor=preprocessor)
    engine.fit(sample_documents)

    # Empty query
    assert engine.search("") == []
    assert engine.search("   ") == []

    # Out of vocabulary query
    oov_results = engine.search("xylophone zeppelin pterodactyl")
    assert oov_results == []


def test_idf_rankings(preprocessor, sample_documents):
    engine = TfidfSearchEngine(preprocessor=preprocessor)
    engine.fit(sample_documents)

    top_rare = engine.get_top_idf_terms(n=5, highest=True)
    assert len(top_rare) > 0
    # Every IDF weight should be positive
    for term, idf in top_rare:
        assert idf > 1.0
