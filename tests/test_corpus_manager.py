"""
Unit Tests for Corpus Manager, Pandas Analytics, and Deduplication.
"""

import os
import pytest
from src.corpus_manager import CorpusManager
from src.keyword_engine import KeywordSearchEngine
from src.preprocessor import TextPreprocessor
from src.tfidf_engine import TfidfSearchEngine


@pytest.fixture
def corpus_manager():
    prep = TextPreprocessor()
    tfidf = TfidfSearchEngine(preprocessor=prep)
    kw = KeywordSearchEngine(preprocessor=prep)
    return CorpusManager(tfidf_engine=tfidf, keyword_engine=kw, preprocessor=prep)


def test_add_and_deduplicate(corpus_manager):
    text = "Machine learning enables computers to learn patterns from training data."
    ok1, msg1, doc1 = corpus_manager.add_document_from_source(text, "doc1.txt")
    assert ok1
    assert corpus_manager.document_count == 1

    # Attempting to add duplicate content should be rejected
    ok2, msg2, doc2 = corpus_manager.add_document_from_source(text, "duplicate_doc1.txt")
    assert not ok2
    assert "Duplicate detected" in msg2
    assert corpus_manager.document_count == 1


def test_corpus_summary_statistics(corpus_manager):
    text1 = "Natural language processing analyzes linguistics and tokens."
    text2 = "Computer algorithms optimize mathematical functions and arrays."
    corpus_manager.add_document_from_source(text1, "text1.txt")
    corpus_manager.add_document_from_source(text2, "text2.txt")

    stats = corpus_manager.get_summary_statistics()
    assert stats["total_documents"] == 2
    assert stats["total_raw_words"] > 10
    assert stats["vocabulary_size"] > 5
    assert stats["avg_words_per_doc"] > 0


def test_remove_document(corpus_manager):
    text = "Information retrieval search query engine document."
    corpus_manager.add_document_from_source(text, "test.txt")
    assert corpus_manager.document_count == 1

    doc_id = corpus_manager.documents[0].doc_id
    ok, msg = corpus_manager.remove_document(doc_id)
    assert ok
    assert corpus_manager.document_count == 0


def test_load_sample_corpus(corpus_manager):
    sample_dir = os.path.join("data", "sample_docs")
    if os.path.exists(sample_dir):
        loaded, msgs = corpus_manager.load_sample_corpus(sample_dir)
        assert loaded >= 7
        assert corpus_manager.document_count >= 7

        # Check top frequent terms dataframe
        freq_df = corpus_manager.get_top_frequent_terms(top_n=10)
        assert len(freq_df) > 0
        assert "Term" in freq_df.columns
