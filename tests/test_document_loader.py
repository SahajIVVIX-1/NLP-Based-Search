"""
Unit Tests for Document Ingestion & Parsing.
"""

import os
import pytest
from src.document_loader import (
    Document,
    EmptyDocumentError,
    compute_content_hash,
    extract_title_from_text,
    load_document_from_source,
)
from src.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


def test_compute_content_hash():
    text1 = "This is a sample document for hash verification."
    text2 = "   THIS IS A SAMPLE   DOCUMENT FOR HASH VERIFICATION.   "
    # Case and whitespace invariant hash
    assert compute_content_hash(text1) == compute_content_hash(text2)


def test_title_extraction():
    doc_text = "# Deep Learning Fundamentals\n\nDeep neural networks process representations."
    title = extract_title_from_text(doc_text, "fallback_name.txt")
    assert title == "Deep Learning Fundamentals"

    # Fallback to filename when text has no clean header
    no_header = "text without a clean title on first line..."
    fallback_title = extract_title_from_text(no_header, "04_machine_learning.txt")
    assert fallback_title == "Machine Learning"


def test_load_text_document(preprocessor):
    content = "Information retrieval algorithms rank documents by relevance scores."
    doc = load_document_from_source(
        source=content,
        filename="ir_test.txt",
        preprocessor=preprocessor,
    )
    assert isinstance(doc, Document)
    assert doc.filename == "ir_test.txt"
    assert doc.file_type == "txt"
    assert doc.word_count == 8
    assert doc.token_count > 0
    assert len(doc.content_hash) == 64


def test_empty_document_error(preprocessor):
    with pytest.raises(EmptyDocumentError):
        load_document_from_source(
            source="   \n\t  ",
            filename="empty.txt",
            preprocessor=preprocessor,
        )


def test_load_sample_pdf(preprocessor):
    pdf_path = os.path.join("data", "sample_docs", "08_corpus_linguistics.pdf")
    if os.path.exists(pdf_path):
        doc = load_document_from_source(
            source=pdf_path,
            filename="08_corpus_linguistics.pdf",
            preprocessor=preprocessor,
        )
        assert doc.file_type == "pdf"
        assert doc.page_count >= 1
        assert "Zipf" in doc.raw_text or "Corpus" in doc.raw_text
        assert doc.token_count > 20
