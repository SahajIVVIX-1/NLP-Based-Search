"""
UI Design System & Custom CSS for the NLP Document Search Engine.

Implements an academic, modern, responsive visual theme with tailored colors,
glassmorphism, score badges, term highlights, and typography.
"""

CUSTOM_CSS = """
/* === Root Theme Variables & Fonts === */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --primary: #4f46e5;
    --primary-hover: #4338ca;
    --primary-light: #eef2ff;
    --secondary: #0ea5e9;
    --surface-bg: #ffffff;
    --page-bg: #f8fafc;
    --card-border: #e2e8f0;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --accent-success: #10b981;
    --accent-warning: #f59e0b;
    --highlight-bg: #fef08a;
    --highlight-text: #854d0e;
}

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--page-bg) !important;
    color: var(--text-main) !important;
}

/* === Academic Banner Header === */
.nlp-hero-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #1e293b 100%);
    border-radius: 16px;
    padding: 32px 36px;
    color: #ffffff;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.nlp-hero-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
    color: #ffffff !important;
}

.nlp-hero-header p {
    font-size: 1.05rem;
    color: #cbd5e1;
    max-width: 820px;
    line-height: 1.5;
    margin-bottom: 16px;
}

.nlp-badge-group {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}

.nlp-pill {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(8px);
    color: #f8fafc;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.nlp-pill-accent {
    background: rgba(14, 165, 233, 0.3);
    color: #7dd3fc;
    border-color: rgba(56, 189, 248, 0.4);
}

/* === Result Cards & Snippets === */
.nlp-result-card {
    background: #ffffff;
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
}

.nlp-result-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 6px 16px -2px rgba(15, 23, 42, 0.06);
    transform: translateY(-1px);
}

.nlp-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    flex-wrap: wrap;
    gap: 8px;
}

.nlp-card-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #0f172a;
    display: flex;
    align-items: center;
    gap: 8px;
}

.nlp-score-badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

.score-high {
    background-color: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
}

.score-mid {
    background-color: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
}

.score-low {
    background-color: #f8fafc;
    color: #64748b;
    border: 1px solid #e2e8f0;
}

.nlp-snippet-box {
    background: #f8fafc;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #334155;
}

/* === Query Term Highlighting === */
mark.nlp-highlight {
    background-color: #fef08a !important;
    color: #713f12 !important;
    font-weight: 600;
    padding: 1px 5px;
    border-radius: 4px;
    border-bottom: 2px solid #eab308;
}

/* === Contribution Pills === */
.nlp-contrib-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin-top: 8px;
    font-size: 0.82rem;
}

.nlp-contrib-label {
    color: #64748b;
    font-weight: 500;
}

.nlp-contrib-pill {
    background: #f1f5f9;
    color: #1e293b;
    padding: 2px 8px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    border: 1px solid #e2e8f0;
}

/* === Statistics Metric Cards === */
.nlp-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}

.nlp-metric-card {
    background: #ffffff;
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
}

.nlp-metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary);
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.2;
}

.nlp-metric-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 6px;
}

/* === Comparison View === */
.nlp-compare-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-top: 16px;
}

@media (max-width: 768px) {
    .nlp-compare-container {
        grid-template-columns: 1fr;
    }
}

.nlp-compare-column {
    background: #ffffff;
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 18px;
}

.nlp-compare-header {
    font-size: 1.1rem;
    font-weight: 600;
    padding-bottom: 10px;
    margin-bottom: 14px;
    border-bottom: 2px solid #e2e8f0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* === Concept Card styling === */
.nlp-concept-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}

.nlp-concept-card h3 {
    margin-top: 0;
    color: #1e1b4b;
    font-size: 1.3rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
}
"""


def render_hero_header() -> str:
    """Render the academic project header banner and embed styles."""
    return f"""
    <style>
    {CUSTOM_CSS}
    </style>
    <div class="nlp-hero-header">
        <h1>NLP Document Search Engine</h1>
        <p>
            An academic Information Retrieval system implementing classical Natural Language Processing pipelines:
            lexical normalization, morphological stemming, TF-IDF vector space modeling, and cosine similarity ranking.
        </p>
        <div class="nlp-badge-group">
            <span class="nlp-pill">College NLP Course Project</span>
            <span class="nlp-pill nlp-pill-accent">NLTK Preprocessing</span>
            <span class="nlp-pill nlp-pill-accent">Scikit-Learn TF-IDF</span>
            <span class="nlp-pill nlp-pill-accent">PyMuPDF Document Parser</span>
            <span class="nlp-pill">Classical IR Syllabus</span>
        </div>
    </div>
    """


def render_metric_cards_html(stats: dict) -> str:
    """Render the analytics summary metric cards."""
    return f"""
    <div class="nlp-metric-grid">
        <div class="nlp-metric-card">
            <div class="nlp-metric-value">{stats.get('total_documents', 0)}</div>
            <div class="nlp-metric-label">Indexed Documents</div>
        </div>
        <div class="nlp-metric-card">
            <div class="nlp-metric-value">{stats.get('total_raw_words', 0):,}</div>
            <div class="nlp-metric-label">Total Corpus Words</div>
        </div>
        <div class="nlp-metric-card">
            <div class="nlp-metric-value">{stats.get('vocabulary_size', 0):,}</div>
            <div class="nlp-metric-label">Unique Vocabulary (Stems)</div>
        </div>
        <div class="nlp-metric-card">
            <div class="nlp-metric-value">{stats.get('avg_words_per_doc', 0)}</div>
            <div class="nlp-metric-label">Avg Words / Document</div>
        </div>
    </div>
    """


def render_search_results_html(response: dict) -> str:
    """Render formatted search results cards."""
    status = response.get("status")
    results = response.get("results", [])

    if status == "empty_query":
        return """
        <div style="padding: 24px; text-align: center; color: #64748b; background: #ffffff; border-radius: 12px; border: 1px dashed #cbd5e1;">
            <p style="font-size: 1.1rem; margin: 0;">🔍 <strong>Enter a search query above</strong> to retrieve relevant documents.</p>
            <p style="font-size: 0.88rem; color: #94a3b8; margin-top: 6px;">Try terms like <em>"transformer self attention"</em>, <em>"vector space cosine"</em>, or <em>"zipf's law"</em>.</p>
        </div>
        """

    if status == "no_valid_tokens":
        return f"""
        <div style="padding: 20px; background: #fffbeb; border-radius: 12px; border: 1px solid #fef3c7; color: #92400e;">
            ⚠️ <strong>Zero informative tokens found:</strong> The query <em>"{response.get('query')}"</em> consists entirely of stopwords or non-alphanumeric noise.
        </div>
        """

    if not results:
        return f"""
        <div style="padding: 24px; text-align: center; background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; color: #64748b;">
            <p style="font-size: 1.1rem; font-weight: 600; color: #334155;">No matching documents found</p>
            <p style="font-size: 0.9rem; margin-top: 6px;">No indexed document in the corpus contains the query terms: <code>{response.get('query_tokens', [])}</code>.</p>
        </div>
        """

    cards_html = []
    for r in results:
        score = r["score"]
        if score >= 0.35:
            score_class = "score-high"
        elif score >= 0.15:
            score_class = "score-mid"
        else:
            score_class = "score-low"

        # Format contributions
        contrib_html_parts = []
        for term, val in r.get("top_contributions", []):
            contrib_html_parts.append(f'<span class="nlp-contrib-pill">{term}: +{val:.3f}</span>')
        contrib_str = " ".join(contrib_html_parts) if contrib_html_parts else "None"

        card = f"""
        <div class="nlp-result-card">
            <div class="nlp-card-header">
                <div class="nlp-card-title">
                    <span style="color: #6366f1; font-weight: 700;">#{r['rank']}</span>
                    {r['title']}
                    <span class="nlp-pill" style="font-size: 0.7rem; color: #475569; background: #f1f5f9; border-color: #e2e8f0;">{r['file_type'].upper()}</span>
                </div>
                <div>
                    <span class="nlp-score-badge {score_class}">
                        Cosine Sim: {r['score_percent']} ({score:.4f})
                    </span>
                </div>
            </div>

            <div class="nlp-snippet-box">
                {r['highlighted_snippet']}
            </div>

            <div class="nlp-contrib-container">
                <span class="nlp-contrib-label">Top Term Weights:</span>
                {contrib_str}
            </div>
        </div>
        """
        cards_html.append(card)

    header_summary = f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; color: #475569; font-size: 0.92rem;">
        <span>Found <strong>{len(results)}</strong> matching document(s) for query: <code>{response.get('query')}</code></span>
        <span>Preprocessed Stems: <code>{response.get('query_tokens', [])}</code></span>
    </div>
    """

    return header_summary + "\n".join(cards_html)


def render_comparison_html(comp_data: dict) -> str:
    """Render side-by-side comparison between TF-IDF and Keyword Search."""
    tfidf_results = comp_data.get("tfidf_results", [])
    kw_results = comp_data.get("keyword_results", [])
    analysis = comp_data.get("analysis", "")

    # TF-IDF Column
    tfidf_cards = []
    for r in tfidf_results:
        tfidf_cards.append(f"""
        <div style="padding: 12px; margin-bottom: 10px; background: #f8fafc; border-radius: 8px; border-left: 3px solid #4f46e5;">
            <div style="font-weight: 600; font-size: 0.95rem; display: flex; justify-content: space-between;">
                <span>#{r['rank']} {r['title']}</span>
                <span style="font-family: monospace; color: #4f46e5; font-weight: 700;">{r['score_percent']}</span>
            </div>
            <div style="font-size: 0.85rem; color: #475569; margin-top: 6px;">
                {r['highlighted_snippet']}
            </div>
        </div>
        """)

    # Keyword Column
    kw_cards = []
    for r in kw_results:
        kw_cards.append(f"""
        <div style="padding: 12px; margin-bottom: 10px; background: #f8fafc; border-radius: 8px; border-left: 3px solid #0ea5e9;">
            <div style="font-weight: 600; font-size: 0.95rem; display: flex; justify-content: space-between;">
                <span>#{r['rank']} {r['title']}</span>
                <span style="font-family: monospace; color: #0284c7; font-weight: 700;">Matches: {r.get('raw_match_count', 0)} ({r.get('query_coverage', '0%')})</span>
            </div>
            <div style="font-size: 0.85rem; color: #475569; margin-top: 6px;">
                {r['highlighted_snippet']}
            </div>
        </div>
        """)

    tfidf_content = "".join(tfidf_cards) if tfidf_cards else "<p style='color: #94a3b8;'>No matches found.</p>"
    kw_content = "".join(kw_cards) if kw_cards else "<p style='color: #94a3b8;'>No matches found.</p>"

    return f"""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
        <h3 style="margin-top: 0; color: #1e1b4b; font-size: 1.15rem;">📐 Comparative Analysis</h3>
        <div style="font-size: 0.95rem; line-height: 1.6; color: #334155; background: #f8fafc; padding: 14px 18px; border-radius: 8px; border: 1px solid #e2e8f0;">
            {analysis.replace(chr(10), '<br>')}
        </div>
    </div>

    <div class="nlp-compare-container">
        <div class="nlp-compare-column">
            <div class="nlp-compare-header" style="color: #4f46e5;">
                <span>TF-IDF Vector Space Model</span>
                <span class="nlp-pill" style="background: #eef2ff; color: #4f46e5; border-color: #c7d2fe;">Cosine Sim</span>
            </div>
            {tfidf_content}
        </div>

        <div class="nlp-compare-column">
            <div class="nlp-compare-header" style="color: #0284c7;">
                <span>Basic Lexical Match (Baseline)</span>
                <span class="nlp-pill" style="background: #f0f9ff; color: #0284c7; border-color: #bae6fd;">Term Overlap</span>
            </div>
            {kw_content}
        </div>
    </div>
    """
