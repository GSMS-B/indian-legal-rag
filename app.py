"""
app.py — Streamlit UI for the Indian Criminal Law Assistant
------------------------------------------------------------
Run with:  streamlit run app.py
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Criminal Law Assistant",
    page_icon="scales",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium look ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}
.main-header h1 {
    font-size: 2.6rem;
    font-weight: 700;
    color: #2e7d32;
    margin-bottom: 0.2rem;
}
.main-header .subtitle {
    font-size: 1rem;
    color: #888;
    letter-spacing: 0.1em;
    font-weight: 400;
}

.source-card {
    background: rgba(102,126,234,0.06);
    border: 1px solid rgba(102,126,234,0.15);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s ease;
}
.source-card:hover {
    border-color: rgba(102,126,234,0.4);
}
.source-card .label {
    font-weight: 600;
    font-size: 0.95rem;
    color: #667eea;
}
.source-card .meta {
    font-size: 0.8rem;
    color: #888;
    margin-top: 0.2rem;
}
.source-card .preview {
    font-size: 0.82rem;
    color: #aaa;
    margin-top: 0.5rem;
    line-height: 1.5;
}

.answer-box {
    background: rgba(102,126,234,0.04);
    border-left: 4px solid #667eea;
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    line-height: 1.7;
}

.disclaimer {
    background: rgba(255,193,7,0.08);
    border: 1px solid rgba(255,193,7,0.2);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: #e0a800;
    text-align: center;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Lazy-load heavy modules with caching ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the RAG pipeline (models + ChromaDB) once, cached across reruns."""
    from rag.pipeline import answer as rag_answer
    from rag.retriever import get_collection_count
    return rag_answer, get_collection_count


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### System Info")

    try:
        _, get_count = load_pipeline()
        chunk_count = get_count()
        if chunk_count > 0:
            st.success(f"**Database**: {chunk_count} sections indexed")
        else:
            st.error("**Database**: Not initialized")
    except Exception:
        st.warning("**Database**: Loading...")
        chunk_count = 0

    st.markdown("---")

    st.markdown("#### Models")
    st.markdown(
        "- **Embedding**: `bge-base-en-v1.5`\n"
        "- **Reranker**: `ms-marco-MiniLM-L-6-v2`\n"
        "- **LLM**: Groq → OpenRouter (auto fallback)"
    )

    # Show which provider answered the last query
    if "provider_used" in st.session_state:
        prov = st.session_state["provider_used"]
        st.caption(f"Last answer via: **{prov}**")

    st.markdown("---")

    st.markdown("#### The Three Acts")
    st.markdown(
        "**BNS 2023** -- *Bharatiya Nyaya Sanhita*\n"
        "Defines crimes and punishments. Replaces IPC 1860. 358 sections.\n\n"
        "**BNSS 2023** -- *Bharatiya Nagarik Suraksha Sanhita*\n"
        "Defines criminal procedures -- FIRs, arrests, bail, trials. Replaces CrPC 1973. 531 sections.\n\n"
        "**BSA 2023** -- *Bharatiya Sakshya Adhiniyam*\n"
        "Defines rules of evidence -- oral, documentary, electronic. Replaces Evidence Act 1872. 170 sections."
    )

    st.markdown("---")
    st.caption("Built with Streamlit | ChromaDB | Groq | OpenRouter")

# ── Main content ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">'
    "<h1>Indian Criminal Law Assistant</h1>"
    '<div class="subtitle">Powered by BNS 2023 | BNSS 2023 | BSA 2023</div>'
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "Ask any question about Indian criminal law in plain English. "
    "The system searches all three new criminal law codes simultaneously "
    "and provides precise answers with exact section citations."
)

# Disclaimer
st.warning(
    "**Educational tool only.** Not legal advice. "
    "Always consult a qualified lawyer for legal matters."
)

# ── Example question buttons ────────────────────────────────────────────────
EXAMPLES = [
    "What is the punishment for murder?",
    "Can police arrest without a warrant?",
    "Are WhatsApp messages valid as evidence?",
    "What are the rights of an arrested person?",
]

cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    with col:
        if st.button(example, key=f"ex_{example}", use_container_width=True):
            st.session_state["query_input"] = example

# ── Query input ──────────────────────────────────────────────────────────────
st.markdown("**Ask your legal question in plain English**")
col_input, col_btn = st.columns([6, 1])

with col_input:
    user_input = st.text_input(
        "Ask your legal question in plain English",
        value=st.session_state.get("query_input", ""),
        placeholder="e.g., What is the punishment for robbery?",
        key="main_query",
        label_visibility="collapsed"
    )

with col_btn:
    send_clicked = st.button("Send", use_container_width=True)

# Determine the actual query to run
query_to_run = ""

if "query_input" in st.session_state and st.session_state["query_input"]:
    # Triggered by example button
    query_to_run = st.session_state["query_input"]
    # We don't pop it here immediately so the text input keeps the value during the run
elif send_clicked and user_input:
    # Triggered by Send button
    query_to_run = user_input
elif user_input and st.session_state.get("last_query") != user_input:
    # Triggered by pressing Enter (value changed)
    query_to_run = user_input

# Save the last executed query to avoid infinite loops on re-renders
if query_to_run:
    st.session_state["last_query"] = query_to_run
    # Clear the example trigger now that we are running it
    if "query_input" in st.session_state:
        st.session_state.pop("query_input", None)

# ── Run pipeline ─────────────────────────────────────────────────────────────
if query_to_run:
    with st.spinner(f'Searching across BNS, BNSS, and BSA for: "{query_to_run}" ...'):
        try:
            rag_answer, _ = load_pipeline()
            result = rag_answer(query_to_run)
        except Exception as e:
            result = {
                "answer": f"Error running pipeline: {str(e)}",
                "sources": [],
                "expanded_query": "",
                "provider_used": "none",
            }

    # Store provider for sidebar display
    if result.get("provider_used"):
        st.session_state["provider_used"] = result["provider_used"]

    # ── Display answer ───────────────────────────────────────────────────
    st.markdown("### Answer")
    st.markdown(
        f'<div class="answer-box">{result["answer"]}</div>',
        unsafe_allow_html=True,
    )

    # ── Display sources ──────────────────────────────────────────────────
    if result["sources"]:
        with st.expander("Sources Retrieved", expanded=False):
            for src in result["sources"]:
                score_val = src.get("rerank_score")
                score = f"{score_val:.3f}" if score_val else "--"
                st.markdown(
                    f'<div class="source-card">'
                    f'<div class="label">{src["source_label"]}</div>'
                    f'<div class="meta">'
                    f'Chapter: {src["chapter"]} | Rerank Score: {score}'
                    f"</div>"
                    f'<div class="preview">{src["text_preview"]}...</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Debug info in expander ───────────────────────────────────────────
    if result.get("expanded_query"):
        with st.expander("Debug: Expanded Query", expanded=False):
            st.code(result["expanded_query"])
