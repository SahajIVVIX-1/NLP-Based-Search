"""
End-to-End User Flow Simulation Tests.
Verifies all 12 core requirements, edge cases, and UI workflows.
"""

import os
from src.ui.views import AppState


def test_complete_user_flows(tmp_path):
    # Initialize application state with default sample documents
    app = AppState()
    assert app.corpus_manager.document_count == 8

    # 1. Search Flow: TF-IDF ranked retrieval & term highlighting
    res = app.search_service.search_tfidf("transformer self attention mechanisms", top_k=3)
    assert res["status"] == "success"
    assert len(res["results"]) > 0
    top_doc = res["results"][0]
    assert "Attention" in top_doc["title"] or "Transformer" in top_doc["title"]
    assert top_doc["score"] > 0.0
    assert '<mark class="nlp-highlight">' in top_doc["highlighted_snippet"]

    # 2. Search Method Comparison Flow: Side-by-side TF-IDF vs Keyword
    comp = app.search_service.compare_search("natural language processing computational", top_k=3)
    assert len(comp["tfidf_results"]) > 0
    assert len(comp["keyword_results"]) > 0
    assert len(comp["analysis"]) > 50

    # 3. Corpus Analytics Flow: Aggregate statistics & Pandas tables
    stats = app.corpus_manager.get_summary_statistics()
    assert stats["total_documents"] == 8
    assert stats["vocabulary_size"] > 100
    assert stats["total_raw_words"] > 500

    freq_df = app.corpus_manager.get_top_frequent_terms(5)
    assert len(freq_df) == 5
    high_idf, low_idf = app.corpus_manager.get_idf_rankings_dataframe(5)
    assert len(high_idf) == 5
    assert len(low_idf) == 5

    # 4. Ingestion, Deduplication, and Deletion Flow
    test_file = tmp_path / "custom_nlp_note.txt"
    test_file.write_text("Reinforcement learning from human feedback aligns neural models.", encoding="utf-8")

    ok, msg, doc = app.corpus_manager.add_document_from_source(str(test_file), "custom_nlp_note.txt")
    assert ok
    assert app.corpus_manager.document_count == 9

    # Duplicate rejection test
    dup_file = tmp_path / "duplicate_note.txt"
    dup_file.write_text("Reinforcement learning from human feedback aligns neural models.", encoding="utf-8")
    ok_dup, msg_dup, _ = app.corpus_manager.add_document_from_source(str(dup_file), "duplicate_note.txt")
    assert not ok_dup
    assert "Duplicate detected" in msg_dup
    assert app.corpus_manager.document_count == 9

    # Document deletion test
    ok_del, msg_del = app.corpus_manager.remove_document(doc.doc_id)
    assert ok_del
    assert app.corpus_manager.document_count == 8

    # 5. Search Audit History Flow
    hist = app.search_service.history.get_history()
    assert len(hist) >= 2
    app.search_service.history.clear()
    assert len(app.search_service.history.get_history()) == 0

    # 6. NLP Theory & Pipeline Inspection Flow
    inspection = app.preprocessor.inspect_pipeline_stages("The linguistic models are retrieving documents!")
    assert "linguist" in inspection["stemmed_tokens"]
    assert "model" in inspection["stemmed_tokens"]
    assert len(inspection["stemming_transformations"]) > 0
