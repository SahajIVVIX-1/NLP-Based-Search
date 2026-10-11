"""
Unit Tests for Text Preprocessor Pipeline.
"""

import pytest
from src.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor(use_stemming=True, remove_stopwords=True, lowercase=True)


def test_normalization(preprocessor):
    text = "   Natural Language PROCESSING! \n\t"
    assert preprocessor.normalize(text) == "natural language processing!"


def test_tokenization(preprocessor):
    text = "Information retrieval uses TF-IDF vectors."
    tokens = preprocessor.tokenize(text)
    assert "information" in tokens
    assert "retrieval" in tokens
    assert "vectors" in tokens


def test_stopword_removal(preprocessor):
    tokens = ["the", "query", "is", "in", "the", "document"]
    retained, removed = preprocessor.filter_stopwords(tokens)
    assert retained == ["query", "document"]
    assert "the" in removed
    assert "is" in removed
    assert "in" in removed


def test_stemming(preprocessor):
    tokens = ["searching", "retrieval", "models", "computational"]
    stemmed, transforms = preprocessor.stem_tokens(tokens)
    assert "search" in stemmed
    assert "model" in stemmed
    assert len(transforms) > 0


def test_preprocess_end_to_end(preprocessor):
    text = "The machine learning algorithm is searching through large databases."
    processed = preprocessor.preprocess(text)
    # Stopwords like 'the', 'is', 'through' should be removed
    assert "the" not in processed
    assert "is" not in processed
    assert "through" not in processed
    # Words should be stemmed
    assert "search" in processed
    assert "algorithm" in processed
    assert "machin" in processed


def test_empty_and_noise_handling(preprocessor):
    assert preprocessor.preprocess("") == []
    assert preprocessor.preprocess("   ") == []
    assert preprocessor.preprocess("!!! ??? ...") == []


def test_pipeline_inspector(preprocessor):
    text = "Searching neural networks."
    inspection = preprocessor.inspect_pipeline_stages(text)
    assert "raw_text" in inspection
    assert "tokens_after_stopwords" in inspection
    assert "stemmed_tokens" in inspection
    assert "final_processed_text" in inspection
    assert inspection["raw_token_count"] >= 3
    assert "search" in inspection["stemmed_tokens"]
