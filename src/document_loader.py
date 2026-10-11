"""
Document Ingestion & Parsing Module for the NLP Document Search Engine.

Handles TXT and PDF file loading, PyMuPDF-based text extraction,
robust error handling, text validation, metadata extraction, and deduplication.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import BinaryIO, Optional, Union

try:
    import pymupdf as fitz
except ImportError:
    import fitz


class DocumentLoadError(Exception):
    """Raised when a document cannot be read, parsed, or is invalid."""
    pass


class EmptyDocumentError(DocumentLoadError):
    """Raised when an ingested document contains no extractable text."""
    pass


class DuplicateDocumentError(DocumentLoadError):
    """Raised when a document with identical content already exists."""
    pass


@dataclass
class Document:
    """Represents an ingested document with full metadata."""
    doc_id: str
    title: str
    filename: str
    file_type: str  # 'txt', 'pdf', 'sample'
    raw_text: str
    processed_text: str
    processed_tokens: list[str]
    content_hash: str
    char_count: int
    word_count: int
    token_count: int
    page_count: int
    added_at: str

    def to_dict(self) -> dict:
        """Convert document metadata to dictionary."""
        return asdict(self)

    def get_preview(self, max_chars: int = 240) -> str:
        """Generate a clean single-line preview snippet."""
        clean = " ".join(self.raw_text.split())
        if len(clean) <= max_chars:
            return clean
        return clean[:max_chars].rstrip() + "..."


def compute_content_hash(text: str) -> str:
    """Compute deterministic SHA-256 fingerprint for text."""
    normalized = " ".join(text.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_title_from_text(text: str, fallback_filename: str) -> str:
    """
    Attempt to extract a meaningful title from the first heading or line.
    Falls back to a sanitized filename if no prominent title is found.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        first_line = lines[0]
        # If it has explicit markdown header syntax: e.g. # Title
        if re.match(r"^#+\s+", first_line):
            clean_title = re.sub(r"^#+\s*", "", first_line).strip()
            if clean_title:
                return clean_title

        # Check if first line is a short header without prose punctuation
        clean_first = re.sub(r"^[0-9]+[\.\)]\s*", "", first_line).strip()
        is_prose = clean_first.endswith((".", "...", ",", ";", ":", "?", "!"))
        if not is_prose and 3 <= len(clean_first) <= 70 and not clean_first.islower():
            return clean_first

    # Fallback to filename without extension
    base = os.path.splitext(os.path.basename(fallback_filename))[0]
    # Replace underscores/hyphens with spaces and capitalize
    return re.sub(r"^[0-9]+[_-]+", "", base).replace("_", " ").replace("-", " ").title()


def extract_text_from_txt(source: Union[str, bytes, BinaryIO], encoding: str = "utf-8") -> str:
    """Extract raw text from a text file, byte sequence, or file buffer."""
    if isinstance(source, str):
        if os.path.exists(source):
            with open(source, "rb") as f:
                raw_bytes = f.read()
        else:
            raw_bytes = source.encode(encoding)
    elif isinstance(source, bytes):
        raw_bytes = source
    elif hasattr(source, "read"):
        raw_bytes = source.read()
    else:
        raise DocumentLoadError(f"Unsupported source type: {type(source)}")

    for enc in (encoding, "utf-8-sig", "latin-1", "cp1252", "ascii"):
        try:
            return raw_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue

    # Fallback with error replacement
    return raw_bytes.decode("utf-8", errors="replace")


def extract_text_from_pdf(source: Union[str, bytes, BinaryIO]) -> tuple[str, int]:
    """
    Extract text and page count from a PDF using PyMuPDF (fitz).
    Returns (extracted_text, page_count).
    """
    try:
        if isinstance(source, str):
            doc = fitz.open(source)
        elif isinstance(source, bytes):
            doc = fitz.open(stream=source, filetype="pdf")
        elif hasattr(source, "read"):
            data = source.read()
            doc = fitz.open(stream=data, filetype="pdf")
        else:
            raise DocumentLoadError(f"Unsupported PDF source type: {type(source)}")

        page_count = len(doc)
        if page_count == 0:
            raise EmptyDocumentError("The PDF contains 0 pages.")

        extracted_pages = []
        for page_idx in range(page_count):
            page = doc[page_idx]
            page_text = page.get_text("text")
            if page_text.strip():
                extracted_pages.append(page_text.strip())

        doc.close()

        full_text = "\n\n".join(extracted_pages).strip()
        if not full_text:
            raise EmptyDocumentError(
                "PDF file was parsed, but contains no extractable text. "
                "The file may be a scanned image or password protected."
            )

        return full_text, page_count

    except EmptyDocumentError:
        raise
    except fitz.FileDataError as e:
        raise DocumentLoadError(f"Corrupt or invalid PDF file: {str(e)}") from e
    except Exception as e:
        raise DocumentLoadError(f"Failed to extract text from PDF: {str(e)}") from e


def load_document_from_source(
    source: Union[str, bytes, BinaryIO],
    filename: str,
    preprocessor: Optional[Any] = None,
    file_type: Optional[str] = None,
) -> Document:
    """
    Unified document ingestion helper.
    Extracts text, calculates statistics, runs preprocessor, and returns a Document object.
    """
    clean_filename = os.path.basename(filename)
    lower_ext = os.path.splitext(clean_filename)[1].lower()

    if file_type is None:
        if lower_ext == ".pdf":
            file_type = "pdf"
        elif lower_ext in (".txt", ".md", ".csv", ".json"):
            file_type = "txt"
        else:
            file_type = "txt"

    if file_type == "pdf":
        raw_text, page_count = extract_text_from_pdf(source)
    else:
        raw_text = extract_text_from_txt(source)
        page_count = 1

    clean_text = raw_text.strip()
    if not clean_text:
        raise EmptyDocumentError(f"Document '{clean_filename}' is empty or contains only whitespace.")

    content_hash = compute_content_hash(clean_text)
    doc_id = f"doc_{content_hash[:12]}"
    title = extract_title_from_text(clean_text, clean_filename)

    # Process tokens
    if preprocessor is not None:
        processed_tokens = preprocessor.preprocess(clean_text)
    else:
        # Fallback simple split
        processed_tokens = [w.lower() for w in re.findall(r"\b\w+\b", clean_text)]

    processed_text = " ".join(processed_tokens)
    words = re.findall(r"\b\S+\b", clean_text)

    return Document(
        doc_id=doc_id,
        title=title,
        filename=clean_filename,
        file_type=file_type,
        raw_text=clean_text,
        processed_text=processed_text,
        processed_tokens=processed_tokens,
        content_hash=content_hash,
        char_count=len(clean_text),
        word_count=len(words),
        token_count=len(processed_tokens),
        page_count=page_count,
        added_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
