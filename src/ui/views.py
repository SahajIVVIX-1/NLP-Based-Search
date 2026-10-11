"""
Gradio UI Layout & Event Handlers for the NLP Document Search Engine.

Implements all six academic tabs:
1. Document Search (ranked results, highlighted snippets, explainability)
2. Search Method Comparison (TF-IDF vs Keyword matching)
3. Corpus Analytics (summary metrics, frequency & IDF tables)
4. Corpus Management (upload PDF/TXT, inspect library, deduplication, reset)
5. Search History (audit log, clear history)
6. NLP Theory & Preprocessing Playground (interactive pipeline inspector)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd

from src.corpus_manager import CorpusManager
from src.document_loader import Document
from src.keyword_engine import KeywordSearchEngine
from src.preprocessor import TextPreprocessor
from src.search_service import SearchService
from src.tfidf_engine import TfidfSearchEngine
from src.ui.theme import (
    CUSTOM_CSS,
    render_comparison_html,
    render_hero_header,
    render_metric_cards_html,
    render_search_results_html,
)


class AppState:
    """Encapsulates active application services and state."""

    def __init__(self, sample_dir: str = os.path.join("data", "sample_docs")) -> None:
        self.preprocessor = TextPreprocessor()
        self.tfidf_engine = TfidfSearchEngine(preprocessor=self.preprocessor)
        self.keyword_engine = KeywordSearchEngine(preprocessor=self.preprocessor)
        self.corpus_manager = CorpusManager(
            tfidf_engine=self.tfidf_engine,
            keyword_engine=self.keyword_engine,
            preprocessor=self.preprocessor,
        )
        self.search_service = SearchService(
            tfidf_engine=self.tfidf_engine,
            keyword_engine=self.keyword_engine,
            preprocessor=self.preprocessor,
        )
        self.sample_dir = sample_dir

        # Auto-load initial sample corpus
        if os.path.exists(self.sample_dir):
            self.corpus_manager.load_sample_corpus(self.sample_dir)


def create_search_app(app_state: AppState) -> gr.Blocks:
    """Build the complete Gradio Blocks application."""

    with gr.Blocks(
        title="NLP Document Search Engine",
    ) as demo:

        # Header Banner
        gr.HTML(render_hero_header())

        with gr.Tabs() as tabs:

            # ============================================================
            # TAB 1: DOCUMENT SEARCH
            # ============================================================
            with gr.TabItem("🔍 Document Search", id="tab_search"):
                with gr.Row():
                    with gr.Column(scale=8):
                        search_input = gr.Textbox(
                            label="Enter Search Query",
                            placeholder="e.g. transformer self attention mechanisms or vector space cosine similarity...",
                            lines=1,
                            max_lines=2,
                            show_label=True,
                            elem_id="search_query_box",
                        )
                    with gr.Column(scale=2):
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Top-K Results",
                            elem_id="search_top_k",
                        )

                # Quick query suggestion buttons
                with gr.Row():
                    gr.Markdown("**Quick Syllabus Queries:**")
                    q_btn1 = gr.Button("Transformer Attention", size="sm", variant="secondary")
                    q_btn2 = gr.Button("Vector Space Cosine", size="sm", variant="secondary")
                    q_btn3 = gr.Button("Zipf's Law Corpus", size="sm", variant="secondary")
                    q_btn4 = gr.Button("Naive Bayes Classification", size="sm", variant="secondary")
                    q_btn5 = gr.Button("Stemming & Stopwords", size="sm", variant="secondary")

                with gr.Row():
                    search_btn = gr.Button("🔍 Search Documents", variant="primary", scale=2)
                    clear_search_btn = gr.Button("Clear", variant="secondary", scale=1)

                # Results container
                search_output_html = gr.HTML(
                    value=render_search_results_html(app_state.search_service.search_tfidf("")),
                    label="Search Results",
                    elem_id="search_results_display",
                )

                # Event handlers for Search
                def handle_search(query: str, top_k: int) -> str:
                    resp = app_state.search_service.search_tfidf(query, top_k=int(top_k))
                    return render_search_results_html(resp)

                search_btn.click(
                    fn=handle_search,
                    inputs=[search_input, top_k_slider],
                    outputs=[search_output_html],
                )
                search_input.submit(
                    fn=handle_search,
                    inputs=[search_input, top_k_slider],
                    outputs=[search_output_html],
                )

                def clear_search() -> Tuple[str, str]:
                    empty_resp = app_state.search_service.search_tfidf("")
                    return "", render_search_results_html(empty_resp)

                clear_search_btn.click(
                    fn=clear_search,
                    inputs=[],
                    outputs=[search_input, search_output_html],
                )

                # Preset query handlers
                for btn, preset in [
                    (q_btn1, "transformer self attention"),
                    (q_btn2, "vector space cosine similarity"),
                    (q_btn3, "zipf's law quantitative corpus linguistics"),
                    (q_btn4, "naive bayes classifier text"),
                    (q_btn5, "stemming normalization stopwords"),
                ]:
                    btn.click(
                        fn=lambda q=preset: (q, render_search_results_html(app_state.search_service.search_tfidf(q, 5))),
                        inputs=[],
                        outputs=[search_input, search_output_html],
                    )

            # ============================================================
            # TAB 2: SEARCH METHOD COMPARISON
            # ============================================================
            with gr.TabItem("⚖️ Method Comparison", id="tab_compare"):
                gr.Markdown(
                    "### Pedagogical Comparison: TF-IDF Cosine Ranking vs. Basic Keyword Matching\n"
                    "Compare how classical Information Retrieval weighting (TF-IDF with L2 normalization) "
                    "diverges from raw keyword occurrence matching on the same documents."
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        compare_input = gr.Textbox(
                            label="Comparison Query",
                            placeholder="e.g. language processing or neural representation...",
                            lines=1,
                            value="natural language processing computational",
                        )
                    with gr.Column(scale=2):
                        compare_top_k = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Top-K for Both",
                        )

                compare_btn = gr.Button("⚖️ Run Comparative Analysis", variant="primary")
                compare_output_html = gr.HTML(
                    value=render_comparison_html(
                        app_state.search_service.compare_search("natural language processing computational", top_k=5)
                    ),
                    label="Comparison Output",
                )

                def handle_compare(query: str, top_k: int) -> str:
                    comp_data = app_state.search_service.compare_search(query, top_k=int(top_k))
                    return render_comparison_html(comp_data)

                compare_btn.click(
                    fn=handle_compare,
                    inputs=[compare_input, compare_top_k],
                    outputs=[compare_output_html],
                )

            # ============================================================
            # TAB 3: CORPUS ANALYTICS
            # ============================================================
            with gr.TabItem("📊 Corpus Analytics", id="tab_analytics"):
                gr.Markdown("### Quantitative Corpus Statistics & Term Distributions")

                analytics_refresh_btn = gr.Button("🔄 Refresh Corpus Analytics", variant="secondary", size="sm")

                stats_metrics_html = gr.HTML(
                    value=render_metric_cards_html(app_state.corpus_manager.get_summary_statistics()),
                    label="Summary Cards",
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Most Frequent Processed Terms (Corpus-wide)")
                        top_freq_df = gr.Dataframe(
                            value=app_state.corpus_manager.get_top_frequent_terms(15),
                            interactive=False,
                            label="Top Terms by Frequency",
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("#### Most Discriminative Terms (Highest IDF Weights)")
                        high_idf_df = gr.Dataframe(
                            value=app_state.corpus_manager.get_idf_rankings_dataframe(15)[0],
                            interactive=False,
                            label="Highest IDF (Rarest) Terms",
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Lowest IDF Terms (Ubiquitous Across Documents)")
                        low_idf_df = gr.Dataframe(
                            value=app_state.corpus_manager.get_idf_rankings_dataframe(15)[1],
                            interactive=False,
                            label="Lowest IDF Terms",
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("#### Document Length Breakdown")
                        doc_summary_df = gr.Dataframe(
                            value=app_state.corpus_manager.get_documents_dataframe()[[
                                "Document Title", "Type", "Words", "Stems"
                            ]],
                            interactive=False,
                            label="Document Lengths",
                        )

                def refresh_analytics() -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                    stats = app_state.corpus_manager.get_summary_statistics()
                    m_html = render_metric_cards_html(stats)
                    freq = app_state.corpus_manager.get_top_frequent_terms(15)
                    high_idf, low_idf = app_state.corpus_manager.get_idf_rankings_dataframe(15)
                    docs_df = app_state.corpus_manager.get_documents_dataframe()
                    doc_sub = docs_df[["Document Title", "Type", "Words", "Stems"]] if not docs_df.empty else pd.DataFrame()
                    return m_html, freq, high_idf, low_idf, doc_sub

                analytics_refresh_btn.click(
                    fn=refresh_analytics,
                    inputs=[],
                    outputs=[stats_metrics_html, top_freq_df, high_idf_df, low_idf_df, doc_summary_df],
                )

            # ============================================================
            # TAB 4: CORPUS MANAGEMENT
            # ============================================================
            with gr.TabItem("📁 Document Management", id="tab_corpus"):
                gr.Markdown(
                    "### Ingest Documents and Manage Corpus Collection\n"
                    "Upload custom plain text (`.txt`) or PDF (`.pdf`) documents. "
                    "The system automatically validates contents, detects duplicates, "
                    "extracts text via PyMuPDF, and retrains the TF-IDF search engine."
                )

                with gr.Row():
                    with gr.Column(scale=6):
                        file_uploader = gr.File(
                            label="Upload Documents (.txt, .pdf)",
                            file_count="multiple",
                            file_types=[".txt", ".pdf", ".md"],
                            type="filepath",
                        )
                        upload_btn = gr.Button("📥 Ingest Uploaded Documents", variant="primary")
                        upload_status = gr.Markdown("Ready to ingest files.")

                    with gr.Column(scale=4):
                        gr.Markdown("#### Corpus Controls")
                        reset_sample_btn = gr.Button("♻️ Reset to Default Sample Corpus", variant="secondary")
                        clear_corpus_btn = gr.Button("🗑️ Clear Entire Corpus", variant="stop")
                        corpus_action_status = gr.Markdown("")

                gr.Markdown("#### Current Indexed Document Library")
                doc_library_df = gr.Dataframe(
                    value=app_state.corpus_manager.get_documents_dataframe(),
                    interactive=False,
                    label="All Documents",
                )

                # Document deletion
                def get_doc_choices() -> List[str]:
                    return [f"{d.doc_id}: {d.title}" for d in app_state.corpus_manager.documents]

                with gr.Row():
                    doc_select_dropdown = gr.Dropdown(
                        label="Select Document to Delete",
                        choices=get_doc_choices(),
                        value=get_doc_choices()[0] if get_doc_choices() else None,
                        scale=8,
                    )
                    delete_doc_btn = gr.Button("❌ Delete Document", variant="stop", scale=2)

                # Ingestion handler
                def handle_upload(files: List[str]) -> Tuple[str, pd.DataFrame, gr.Dropdown]:
                    if not files:
                        return "No files selected.", app_state.corpus_manager.get_documents_dataframe(), gr.Dropdown(choices=get_doc_choices())

                    messages = []
                    for f_path in files:
                        fname = os.path.basename(f_path)
                        ok, msg, _ = app_state.corpus_manager.add_document_from_source(f_path, fname)
                        icon = "✅" if ok else "⚠️"
                        messages.append(f"{icon} {msg}")

                    status_report = "\n\n".join(messages)
                    choices = get_doc_choices()
                    return (
                        status_report,
                        app_state.corpus_manager.get_documents_dataframe(),
                        gr.Dropdown(choices=choices, value=choices[0] if choices else None),
                    )

                upload_btn.click(
                    fn=handle_upload,
                    inputs=[file_uploader],
                    outputs=[upload_status, doc_library_df, doc_select_dropdown],
                )

                # Reset to sample corpus handler
                def handle_reset_sample() -> Tuple[str, pd.DataFrame, gr.Dropdown]:
                    app_state.corpus_manager.clear_corpus()
                    count, msgs = app_state.corpus_manager.load_sample_corpus(app_state.sample_dir)
                    choices = get_doc_choices()
                    return (
                        f"✅ Reset complete: Loaded {count} default academic documents.",
                        app_state.corpus_manager.get_documents_dataframe(),
                        gr.Dropdown(choices=choices, value=choices[0] if choices else None),
                    )

                reset_sample_btn.click(
                    fn=handle_reset_sample,
                    inputs=[],
                    outputs=[corpus_action_status, doc_library_df, doc_select_dropdown],
                )

                # Clear corpus handler
                def handle_clear_corpus() -> Tuple[str, pd.DataFrame, gr.Dropdown]:
                    app_state.corpus_manager.clear_corpus()
                    return (
                        "🗑️ Corpus cleared completely. Search engine is empty.",
                        app_state.corpus_manager.get_documents_dataframe(),
                        gr.Dropdown(choices=[], value=None),
                    )

                clear_corpus_btn.click(
                    fn=handle_clear_corpus,
                    inputs=[],
                    outputs=[corpus_action_status, doc_library_df, doc_select_dropdown],
                )

                # Delete single document handler
                def handle_delete_doc(selected: str) -> Tuple[str, pd.DataFrame, gr.Dropdown]:
                    if not selected:
                        return "No document selected.", app_state.corpus_manager.get_documents_dataframe(), gr.Dropdown(choices=get_doc_choices())

                    doc_id = selected.split(":")[0].strip()
                    ok, msg = app_state.corpus_manager.remove_document(doc_id)
                    choices = get_doc_choices()
                    return (
                        f"{'✅' if ok else '⚠️'} {msg}",
                        app_state.corpus_manager.get_documents_dataframe(),
                        gr.Dropdown(choices=choices, value=choices[0] if choices else None),
                    )

                delete_doc_btn.click(
                    fn=handle_delete_doc,
                    inputs=[doc_select_dropdown],
                    outputs=[corpus_action_status, doc_library_df, doc_select_dropdown],
                )

            # ============================================================
            # TAB 5: SEARCH AUDIT HISTORY
            # ============================================================
            with gr.TabItem("📜 Search History", id="tab_history"):
                gr.Markdown("### Session Search Audit Log\nTrack executed queries, ranking results, and scores.")

                with gr.Row():
                    refresh_history_btn = gr.Button("🔄 Refresh History", size="sm", variant="secondary")
                    clear_history_btn = gr.Button("🗑️ Clear History", size="sm", variant="stop")

                def get_history_df() -> pd.DataFrame:
                    entries = app_state.search_service.history.get_history()
                    if not entries:
                        return pd.DataFrame(
                            columns=["Timestamp", "Query", "Engine", "Results Found", "Top Document", "Top Score"]
                        )
                    return pd.DataFrame(entries)

                history_df = gr.Dataframe(
                    value=get_history_df(),
                    interactive=False,
                    label="Search Activity Log",
                )

                refresh_history_btn.click(
                    fn=get_history_df,
                    inputs=[],
                    outputs=[history_df],
                )

                def clear_history() -> pd.DataFrame:
                    app_state.search_service.history.clear()
                    return get_history_df()

                clear_history_btn.click(
                    fn=clear_history,
                    inputs=[],
                    outputs=[history_df],
                )

            # ============================================================
            # TAB 6: NLP THEORY & PREPROCESSING PLAYGROUND
            # ============================================================
            with gr.TabItem("🎓 NLP Theory & Pipeline", id="tab_theory"):
                gr.Markdown("### Interactive NLP Preprocessing Playground")
                gr.Markdown(
                    "Inspect how classical computational linguistics transforms a raw natural language sentence "
                    "through every discrete pipeline stage."
                )

                with gr.Row():
                    playground_input = gr.Textbox(
                        label="Input Text for Pipeline Inspection",
                        placeholder="Type any arbitrary English sentence to observe tokenization, stopword removal, and stemming...",
                        value="The computational linguistics algorithms are searching through multiple documents and optimizing queries!",
                        lines=2,
                    )
                inspect_btn = gr.Button("🔬 Inspect Pipeline Stages", variant="primary")

                playground_output_json = gr.JSON(
                    value=app_state.preprocessor.inspect_pipeline_stages(
                        "The computational linguistics algorithms are searching through multiple documents and optimizing queries!"
                    ),
                    label="Pipeline Stage Inspection Output",
                )

                inspect_btn.click(
                    fn=lambda text: app_state.preprocessor.inspect_pipeline_stages(text),
                    inputs=[playground_input],
                    outputs=[playground_output_json],
                )

                gr.Markdown("---")
                gr.Markdown("""
### Academic Foundations of Information Retrieval

#### 1. Text Preprocessing Pipeline
- **Word Tokenization:** Deconstructing unbroken strings into semantic token units using the NLTK Punkt rule-based tokenizer.
- **Normalization:** Lowercasing and typographic stripping to unify orthographic variations (e.g., *'Retrieval'* vs *'retrieval'*).
- **Stopword Elimination:** Filtering frequent closed-class grammatical tokens (*'the'*, *'is'*, *'which'*) that carry minimal topical specificity.
- **Morphological Stemming:** Reducing derived forms to base stems via algorithmic suffix-stripping rules (Porter Stemmer: *'searching'* $\\to$ *'search'*).

#### 2. Vector Space Model & TF-IDF
Documents and queries are modeled as vectors in a high-dimensional vocabulary space $\\mathbb{R}^{|V|}$:
- **Term Frequency:** $\\text{TF}(t, d) = \\text{count}(t, d)$ represents local topical density.
- **Inverse Document Frequency:**
  $$\\text{IDF}(t, D) = \\ln\\left(\\frac{1 + |D|}{1 + \\text{df}(t)}\\right) + 1$$
  Penalizes ubiquitous terms appearing across all documents and boosts rare, highly discriminative keywords.
- **TF-IDF Weighting:**
  $$w(t, d) = \\text{TF}(t, d) \\times \\text{IDF}(t, D)$$
- **Euclidean $L_2$ Normalization:**
  $$\\hat{v}_d = \\frac{v_d}{\\|v_d\\|_2}$$
  Prevents longer documents from dominating search results purely due to higher word counts.
- **Cosine Similarity Ranking:**
  $$\\cos(\\theta) = \\frac{\\mathbf{q} \\cdot \\mathbf{d}}{\\|\\mathbf{q}\\|_2 \\|\\mathbf{d}\\|_2} = \\hat{\\mathbf{q}} \\cdot \\hat{\\mathbf{d}}$$

#### 3. Fundamental Limitations of Lexical Search
Classical TF-IDF and bag-of-words search have well-documented theoretical boundaries:
- **Synonymy (Vocabulary Mismatch):** Queries searching for *'physician'* yield zero similarity with documents using *'doctor'*, because the terms correspond to orthogonal vector dimensions.
- **Polysemy:** Words with multiple meanings (e.g., *'bank'* as a river bank vs financial institution) share the same vector coordinate.
- **Word Order & Context Oblivion:** The bag-of-words assumption treats *'not good, very bad'* identically to *'not bad, very good'*.
- **Out-of-Vocabulary (OOV) Terms:** Misspelled words or unseen terms produce zero vectors.

#### 4. Future Enhancements: Dense Neural Embeddings & Hybrid Search
- **Dense Vector Search:** Modern architectures use Transformer bi-encoders (e.g., **Sentence-BERT**) to project queries and passages into low-dimensional continuous embedding spaces (e.g., 384 or 768 dimensions), matching documents on underlying semantic intent rather than literal token overlap.
- **Hybrid Search:** Industrial systems combine classical lexical inverted indices (BM25 / TF-IDF) with dense vector embeddings via **Reciprocal Rank Fusion (RRF)** for optimal precision and recall.
                """)

    return demo
