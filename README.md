---
title: NLP Document Search Engine
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: Classical NLP and Information Retrieval Vector Space Search Engine
---

# NLP Document Search Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio%206-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/Tests-27%20Passed-brightgreen.svg)]()

An academic, production-grade **Information Retrieval (IR) and Natural Language Processing (NLP)** document search engine built for a college NLP curriculum showcase. The application indexes document collections, processes text through classical linguistic pipelines, constructs a high-dimensional Vector Space Model using **TF-IDF**, and executes ranked retrieval via **Cosine Similarity**.

---

## 📋 Table of Contents
1. [Core Features](#-core-features)
2. [NLP Pipeline & Theoretical Foundations](#-nlp-pipeline--theoretical-foundations)
3. [Method Comparison: TF-IDF vs. Lexical Matching](#-method-comparison-tf-idf-vs-lexical-matching)
4. [Theoretical Limitations of Lexical Search](#-theoretical-limitations-of-lexical-search)
5. [Future Enhancements: Dense Embeddings & Hybrid Search](#-future-enhancements-dense-embeddings--hybrid-search)
6. [Repository Architecture](#-repository-architecture)
7. [Installation & Local Execution](#-installation--local-execution)
8. [Automated Testing Suite](#-automated-testing-suite)
9. [Hugging Face Spaces Deployment](#-hugging-face-spaces-deployment)
10. [Git Setup & Remote Push Instructions](#-git-setup--remote-push-instructions)

---

## 🚀 Core Features

- **Ad-Hoc Ranked Search:** Query document collections with top-$K$ configurable precision.
- **Pre-Loaded Sample Academic Corpus:** 8 foundational documents (TXT and PDF) covering NLP, IR, transformers, machine learning, and corpus linguistics for instant out-of-the-box readiness.
- **Multi-Format Document Ingestion:** Upload plain text (`.txt`) and PDF (`.pdf`) documents parsed via **PyMuPDF (`fitz`)**.
- **Interactive Preprocessing Pipeline:** NLTK tokenization, normalization, stopword filtering, and Porter stemming with a real-time step-by-step pipeline inspector.
- **Vector Space Model (TF-IDF):** Strict mathematical implementation using scikit-learn primitives, smoothed IDF weights, and $L_2$ unit normalization.
- **Dynamic Snippet Extraction & Query Highlighting:** Automatically isolates highest-density query windows and applies `<mark>` badges matching preprocessed stems.
- **Term-Level Explainability:** Exposes exact coordinate-wise contributions showing which terms drove document rankings.
- **Pedagogical Comparison Mode:** Side-by-side evaluation of TF-IDF Cosine ranking against baseline lexical keyword matching.
- **Corpus Analytics Dashboard:** Key corpus metrics (vocabulary size, total words, average length), frequency distributions, and extreme IDF term tables.
- **Corpus Lifecycle Management:** Add documents, deduplicate via SHA-256 fingerprints, delete items, or reset to defaults.
- **Audit History Log:** Session-level query logging with timestamp, result counts, top document, and single-click history purging.
- **Robust Error Handling:** Graceful handling of empty queries, corrupt/scanned PDFs, duplicate uploads, and out-of-vocabulary inputs.

---

## 🔬 NLP Pipeline & Theoretical Foundations

```
Raw Text Stream 
      │
      ▼
1. Text Ingestion & Parsing (TXT / PyMuPDF)
      │
      ▼
2. Normalization & Case-Folding (lowercase, typographic strip)
      │
      ▼
3. Sentence Segmentation & Word Tokenization (NLTK Punkt)
      │
      ▼
4. Non-Alphanumeric & Noise Filtering
      │
      ▼
5. Stopword Elimination (NLTK English Lexicon: 179 closed-class words)
      │
      ▼
6. Morphological Stemming (Porter Stemmer: e.g., 'searching' ➔ 'search')
      │
      ▼
7. Vector Space Modeling (TF-IDF Matrix & Query Projection)
      │
      ▼
8. L2 Normalization & Cosine Similarity Ranking (cos θ = q̂ · d̂)
      │
      ▼
9. Dynamic Snippet Extraction & Query-Term Highlighting
```

### Mathematical Formulation

#### 1. Term Frequency ($TF$)
Quantifies local topical density within a document:
$$\text{TF}(t, d) = \text{count}(t, d)$$

#### 2. Inverse Document Frequency ($IDF$)
Penalizes ubiquitous corpus words and elevates rare, discriminating terms across corpus $D$:
$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + \text{df}(t)}\right) + 1$$
where $|D|$ is the total number of documents and $\text{df}(t)$ is the document frequency of term $t$.

#### 3. Composite TF-IDF Weight
$$w(t, d) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

#### 4. Euclidean $L_2$ Normalization
Prevents verbose documents from scoring higher purely due to length:
$$\hat{\mathbf{v}}_d = \frac{\mathbf{v}_d}{\|\mathbf{v}_d\|_2} = \frac{\mathbf{v}_d}{\sqrt{\sum_{i=1}^{|V|} w(t_i, d)^2}}$$

#### 5. Cosine Similarity Ranking
Computes the cosine of the angle between query vector $\mathbf{q}$ and document vector $\mathbf{d}$ in $|V|$-dimensional space:
$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \hat{\mathbf{q}} \cdot \hat{\mathbf{d}} = \sum_{t \in \mathbf{q} \cap \mathbf{d}} \hat{w}(t, q) \times \hat{w}(t, d)$$

---

## ⚖️ Method Comparison: TF-IDF vs. Lexical Matching

| Feature / Dimension | TF-IDF Vector Space Model | Basic Lexical Match (Baseline) |
|---|---|---|
| **Weighting Philosophy** | Information-theoretic (local TF × global IDF) | Unweighted raw occurrence count |
| **Document Length Bias** | **Neutralized** via Euclidean $L_2$ normalization | **Severe bias** favoring verbose documents |
| **Common vs. Rare Words** | Downweights ubiquitous words, prizes rare terms | Treats all matching words identically |
| **Geometric Model** | Continuous angle in $|V|$-dimensional metric space | Discrete set overlap / term frequency sums |
| **Explainability** | Coordinate-wise dot product contributions | Raw term occurrence frequencies |

---

## ⚠️ Theoretical Limitations of Lexical Search

While TF-IDF is computationally efficient and mathematically elegant, it operates under the **Bag-of-Words (BoW)** assumption and exhibits fundamental lexical limitations:

1. **The Synonymy Bottleneck (Vocabulary Mismatch):** Two terms with identical semantic intent (e.g., *"doctor"* and *"physician"*) inhabit orthogonal vector axes. A query for *"physician"* yields exactly $0.0$ cosine similarity with a document discussing *"doctors"*.
2. **Polysemy:** Words with disparate meanings depending on context (e.g., *"bank"* as a financial institution vs. a river embankment) collapse into a single vector dimension.
3. **Word Order and Syntactic Oblivion:** The unordered bag-of-words assumption treats *"not good, very bad"* identically to *"not bad, very good"*.
4. **Out-of-Vocabulary (OOV) Terms:** Morphological variants not recognized by the stemmer or misspelled tokens fail completely.

---

## 🚀 Future Enhancements: Dense Embeddings & Hybrid Search

To overcome lexical barriers without sacrificing speed, modern search architectures employ:

1. **Dense Semantic Embeddings (Bi-Encoders):**
   - **Sentence-BERT (SBERT):** Maps entire passages and queries into continuous 384- or 768-dimensional dense spaces where distance correlates directly with semantic meaning.
2. **Hybrid Retrieval (Lexical + Dense):**
   - Combining classical inverted indices (BM25 / TF-IDF) for keyword precision with dense embeddings for conceptual recall.
   - Merging ranked candidate lists via **Reciprocal Rank Fusion (RRF)**:
     $$\text{RRF Score}(d) = \sum_{m \in \{\text{TF-IDF}, \text{Dense}\}} \frac{1}{k + r_m(d)}$$

---

## 📁 Repository Architecture

```
NLP-Based-Search/
├── .gitignore                      # Python, virtualenv, and test cache exclusions
├── LICENSE                         # MIT License
├── README.md                       # Documentation & Hugging Face Spaces metadata
├── requirements.txt                # Locked project dependencies
├── app.py                          # Hugging Face Spaces & local web application entry
│
├── data/
│   └── sample_docs/                # Academic default corpus (immediate deployment readiness)
│       ├── 01_intro_to_nlp.txt
│       ├── 02_vector_space_models.txt
│       ├── 03_information_retrieval.txt
│       ├── 04_machine_learning_nlp.txt
│       ├── 05_deep_learning_language.txt
│       ├── 06_transformers_attention.txt
│       ├── 07_preprocessing_techniques.txt
│       └── 08_corpus_linguistics.pdf # Binary PDF compiled with PyMuPDF
│
├── src/
│   ├── __init__.py
│   ├── preprocessor.py             # NLTK tokenization, normalization, stopwords, Porter stemming
│   ├── document_loader.py          # TXT & PyMuPDF PDF parser, deduplication, metadata
│   ├── tfidf_engine.py             # Vector space model, IDF computation, cosine similarity
│   ├── keyword_engine.py           # Lexical term frequency baseline engine
│   ├── corpus_manager.py           # Collection state, Pandas analytics, engine re-indexing
│   ├── search_service.py           # Snippet generator, query highlighter, audit history
│   └── ui/
│       ├── __init__.py
│       ├── theme.py                # Academic design tokens, responsive CSS, card renderers
│       └── views.py                # Gradio Blocks multi-tab interface & event listeners
│
└── tests/
    ├── __init__.py
    ├── test_preprocessor.py        # Tests for normalization, tokenization, stemming
    ├── test_document_loader.py     # Tests for TXT, PDF extraction, empty & corrupt files
    ├── test_tfidf_engine.py        # Tests for matrix math, cosine bounds, OOV queries
    ├── test_keyword_engine.py      # Tests for lexical overlap baseline
    ├── test_search_service.py      # Tests for snippets, HTML highlighting, search history
    ├── test_corpus_manager.py      # Tests for Pandas statistics, deduplication, removal
    └── test_integration_flows.py   # End-to-end integration flows across all UI operations
```

---

## 💻 Installation & Local Execution

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/SahajIVVIX-1/NLP-Based-Search.git
cd NLP-Based-Search
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:7860/`.

---

## 🧪 Automated Testing Suite

The project includes 27 comprehensive automated tests covering unit logic, mathematical guarantees, edge cases, and end-to-end user flows.

To run the full test suite:
```bash
pytest -v
```

Expected output:
```
tests/test_corpus_manager.py::test_add_and_deduplicate PASSED
tests/test_corpus_manager.py::test_corpus_summary_statistics PASSED
tests/test_corpus_manager.py::test_remove_document PASSED
tests/test_corpus_manager.py::test_load_sample_corpus PASSED
tests/test_document_loader.py::test_compute_content_hash PASSED
tests/test_document_loader.py::test_title_extraction PASSED
tests/test_document_loader.py::test_load_text_document PASSED
tests/test_document_loader.py::test_empty_document_error PASSED
tests/test_document_loader.py::test_load_sample_pdf PASSED
tests/test_integration_flows.py::test_complete_user_flows PASSED
tests/test_keyword_engine.py::test_keyword_search PASSED
tests/test_keyword_engine.py::test_keyword_search_no_match PASSED
tests/test_preprocessor.py::test_normalization PASSED
tests/test_preprocessor.py::test_tokenization PASSED
tests/test_preprocessor.py::test_stopword_removal PASSED
tests/test_preprocessor.py::test_stemming PASSED
tests/test_preprocessor.py::test_preprocess_end_to_end PASSED
tests/test_preprocessor.py::test_empty_and_noise_handling PASSED
tests/test_preprocessor.py::test_pipeline_inspector PASSED
tests/test_search_service.py::test_snippet_extraction_and_highlighting PASSED
tests/test_search_service.py::test_search_tfidf_service PASSED
tests/test_search_service.py::test_search_history_manager PASSED
tests/test_search_service.py::test_compare_search PASSED
tests/test_tfidf_engine.py::test_fit_and_vocab_size PASSED
tests/test_tfidf_engine.py::test_cosine_similarity_search PASSED
tests/test_tfidf_engine.py::test_empty_or_oov_queries PASSED
tests/test_tfidf_engine.py::test_idf_rankings PASSED
============================= 27 passed =============================
```

---

## 🤗 Hugging Face Spaces Deployment

The repository is pre-configured for one-click deployment on **Hugging Face Spaces** using the Gradio SDK:

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Select **Gradio** as the SDK.
3. Link your GitHub repository or push directly to the Hugging Face Space git remote.
4. Hugging Face Spaces automatically reads the YAML frontmatter in `README.md`, installs `requirements.txt`, and executes `app.py`.

---

## 📌 Git Setup & Remote Push Instructions

> **Note on Push Approval:** Per project safety guidelines, no commits are pushed to any remote repository without explicit user review and approval.

### Inspecting Local Changes
```bash
git status
```

### Staging and Committing Locally
```bash
git add .
git commit -m "feat: complete NLP document search engine with TF-IDF, PyMuPDF, UI, and test suite"
```

### Pushing to GitHub (Upon User Approval)
```bash
git push origin initial
```
