import streamlit as st
import os
import time
from utils.pdf_processor import extract_text_from_pdf, split_text_into_chunks
from utils.embeddings import get_embedding
from utils.pinecone_client import (
    init_pinecone,
    upsert_chunks,
    search_query,
    get_index_stats,
)

st.set_page_config(
    page_title="DocFinder AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
    background-color: #0d0a1a !important;
    color: #e8d4f0 !important;
}

/* ── TOP BANNER ── */
.top-banner {
    background: #160d2a;
    border-left: 5px solid #a855f7;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    border-top: 1px solid #2d1a4a;
    border-right: 1px solid #2d1a4a;
    border-bottom: 1px solid #2d1a4a;
}
.banner-icon { font-size: 3.2rem; line-height: 1; }
.banner-text h1 {
    color: #a855f7;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0 0 0.2rem 0;
    letter-spacing: -0.5px;
}
.banner-text p { color: #9a7ab0; font-size: 0.9rem; margin: 0; }
.banner-text .badge {
    display: inline-block;
    margin-top: 0.5rem;
    background: rgba(168,85,247,0.1);
    border: 1px solid rgba(168,85,247,0.3);
    color: #a855f7;
    padding: 0.18rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ── RESULT CARDS ── */
.rcard {
    background: #160d2a;
    border: 1px solid #2d1a4a;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    border-left: 4px solid #a855f7;
}
.rcard-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.7rem;
}
.rcard-doc { color: #a855f7; font-weight: 600; font-size: 0.92rem; }
.rcard-score {
    background: #a855f7;
    color: #0d0a1a;
    font-weight: 700;
    font-size: 0.8rem;
    padding: 0.2rem 0.7rem;
    border-radius: 6px;
    white-space: nowrap;
}
.rcard-body { color: #c4a8d4; font-size: 0.88rem; line-height: 1.75; }
.rcard-foot { color: #3a2a5a; font-size: 0.75rem; margin-top: 0.5rem; }

/* ── STAT BOXES ── */
.sbox {
    background: #160d2a;
    border: 1px solid #2d1a4a;
    border-radius: 10px;
    padding: 0.9rem;
    text-align: center;
    margin-bottom: 0.7rem;
}
.sbox-label { color: #6a4a8a; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; }
.sbox-val { color: #a855f7; font-size: 1.6rem; font-weight: 700; }

/* ── BUTTONS ── */
.stButton > button {
    background: #a855f7 !important;
    color: #0d0a1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-family: 'Poppins', sans-serif !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.82 !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab"] {
    color: #9a7ab0 !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    color: #a855f7 !important;
    border-bottom: 2px solid #a855f7 !important;
}

/* ── INPUTS ── */
.stTextInput input {
    background: #160d2a !important;
    border: 1px solid #2d1a4a !important;
    color: #e8d4f0 !important;
    border-radius: 8px !important;
}
.stTextInput input:focus {
    border-color: #a855f7 !important;
    box-shadow: 0 0 0 2px rgba(168,85,247,0.15) !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #110920 !important;
    border-right: 1px solid #2d1a4a !important;
}
section[data-testid="stSidebar"] label { color: #9a7ab0 !important; }
section[data-testid="stSidebar"] h2 { color: #a855f7 !important; }

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    background: #160d2a !important;
    color: #a855f7 !important;
    border: 1px solid #2d1a4a !important;
    border-radius: 8px !important;
}

/* ── PROGRESS ── */
.stProgress > div > div { background: #a855f7 !important; }

/* ── ALERTS ── */
.stWarning { background: #1a0d2e !important; border-color: #a855f7 !important; }
</style>
""", unsafe_allow_html=True)

# ── BANNER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
    <div class="banner-icon">📚</div>
    <div class="banner-text">
        <h1>DocFinder AI</h1>
        <p>Upload your PDFs, ask questions in plain language — get precise answers from your own documents.</p>
        <span class="badge">🧑‍💻 Built by Sohaib</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔑 API Setup")
    pinecone_api_key = st.text_input(
        "Pinecone API Key",
        type="password",
        placeholder="pcsk_xxxxxxxxxx",
        help="Free key at app.pinecone.io"
    )
    pinecone_index_name = st.text_input("Index Name", value="mini-search-engine")

    st.markdown("---")
    st.markdown("## 📈 Index Stats")
    stats_placeholder = st.empty()

    st.markdown("---")
    st.markdown("## 🎛️ Search Options")
    top_k = st.slider("Results to show", 1, 10, 5)
    score_threshold = st.slider("Min match score", 0.0, 1.0, 0.0, 0.05)

    st.markdown("---")
    st.markdown("""
    <small style='color:#2a4a46;font-size:0.78rem;'>
    Local SentenceTransformer model.<br>No OpenAI key needed.<br><br>
    <b style='color:#a855f7'>Sohaib</b> — DocFinder AI
    </small>""", unsafe_allow_html=True)


@st.cache_resource
def get_pinecone_index(api_key, index_name):
    return init_pinecone(api_key, index_name)


def refresh_stats(index):
    try:
        stats = get_index_stats(index)
        total = stats.get("total_vector_count", 0)
        dim   = stats.get("dimension", "—")
        stats_placeholder.markdown(f"""
        <div class="sbox"><div class="sbox-label">Vectors Stored</div><div class="sbox-val">{total}</div></div>
        <div class="sbox"><div class="sbox-label">Dimensions</div><div class="sbox-val">{dim}</div></div>
        """, unsafe_allow_html=True)
    except Exception:
        stats_placeholder.caption("Connect Pinecone to view stats.")


# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📂 Add Documents", "🔎 Find Answers"])

# ═══════════════════════════════ TAB 1 ═══════════════════════════════════════
with tab1:
    st.markdown("### Add Your PDF Files")
    st.caption("Upload PDFs — the system will read, chunk and store them so you can search later.")

    if not pinecone_api_key:
        st.warning("⚠️ Enter your Pinecone API key in the sidebar to get started.")
    else:
        uploaded_files = st.file_uploader(
            "Drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file(s) ready to process:**")
            for f in uploaded_files:
                st.markdown(f"- 📄 `{f.name}` — {f.size/1024:.1f} KB")

            if st.button("⚡ Start Indexing", type="primary"):
                try:
                    index = get_pinecone_index(pinecone_api_key, pinecone_index_name)
                except Exception as e:
                    st.error(f"Could not connect to Pinecone: {e}")
                    st.stop()

                bar    = st.progress(0)
                status = st.empty()
                total_chunks = 0

                for i, pdf_file in enumerate(uploaded_files):
                    status.markdown(f"Working on `{pdf_file.name}` …")
                    with st.expander(f"📄 {pdf_file.name}", expanded=True):
                        try:
                            with st.spinner("Reading text …"):
                                text = extract_text_from_pdf(pdf_file)
                            st.success(f"Read {len(text):,} characters")

                            with st.spinner("Splitting into chunks …"):
                                chunks = split_text_into_chunks(text)
                            st.success(f"Created {len(chunks)} chunks")

                            with st.spinner("Saving to Pinecone …"):
                                chunk_dicts = []
                                for j, chunk_text in enumerate(chunks):
                                    emb = get_embedding(chunk_text)
                                    chunk_dicts.append({
                                        "id": f"{pdf_file.name}__chunk_{j}",
                                        "embedding": emb,
                                        "text": chunk_text,
                                        "doc_name": pdf_file.name,
                                        "chunk_idx": j,
                                    })
                                upsert_chunks(index, chunk_dicts)
                                total_chunks += len(chunk_dicts)
                            st.success("✅ Saved!")
                        except Exception as e:
                            st.error(f"Error: {e}")

                    bar.progress((i + 1) / len(uploaded_files))

                status.success(f"Done! {total_chunks} chunks saved from {len(uploaded_files)} file(s).")
                refresh_stats(index)

# ═══════════════════════════════ TAB 2 ═══════════════════════════════════════
with tab2:
    st.markdown("### Ask Anything About Your Documents")
    st.caption("Type a question in plain language — the AI finds the most relevant sections from your PDFs.")

    if not pinecone_api_key:
        st.warning("⚠️ Enter your Pinecone API key in the sidebar to get started.")
    else:
        query = st.text_input("Your question", placeholder="e.g. What causes global warming?", label_visibility="collapsed")
        search_btn = st.button("🔎 Search Documents", type="primary")

        if search_btn and query.strip():
            try:
                index = get_pinecone_index(pinecone_api_key, pinecone_index_name)
            except Exception as e:
                st.error(f"Pinecone error: {e}")
                st.stop()

            with st.spinner("Searching through your documents …"):
                qe      = get_embedding(query)
                results = search_query(index, qe, top_k=top_k)

            results = [r for r in results if r["score"] >= score_threshold]

            if not results:
                st.info("No matching results found. Try a different question or lower the min score.")
            else:
                st.markdown(f"**{len(results)} match(es) found for:** *{query}*")
                st.markdown("---")
                for rank, r in enumerate(results, 1):
                    score     = r["score"]
                    doc_name  = r.get("doc_name", "Unknown")
                    text      = r.get("text", "")
                    chunk_idx = r.get("chunk_idx", "?")

                    st.markdown(f"""
                    <div class="rcard">
                        <div class="rcard-top">
                            <span class="rcard-doc">#{rank} &nbsp;📄 {doc_name}</span>
                            <span class="rcard-score">Match: {score:.3f}</span>
                        </div>
                        <div class="rcard-body">{text}</div>
                        <div class="rcard-foot">Chunk {chunk_idx} &nbsp;·&nbsp; {len(text)} chars</div>
                    </div>
                    """, unsafe_allow_html=True)

            refresh_stats(index)

        elif search_btn:
            st.warning("Please type a question first.")

if pinecone_api_key:
    try:
        refresh_stats(get_pinecone_index(pinecone_api_key, pinecone_index_name))
    except Exception:
        pass
