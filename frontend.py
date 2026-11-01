"""
EvidenceLens FRONTEND: Streamlit user interface
================================================
This file contains only the user interface. It performs no retrieval, grading,
generation or verification itself. Every piece of information shown on screen
(answers, citations, evidence status, grading results, retrieval trace, document
list, statistics) is produced by the FastAPI backend in `backend.py` and fetched
over HTTP.

How the frontend connects to the backend:
    API_BASE_URL  ->  address of the FastAPI service (default http://127.0.0.1:8000)
    api_get(...)  ->  GET  requests  (/stats, /documents, /health)
    api_post(...) ->  POST requests  (/query, /ingest, /ingest/sample, /feedback)
    DELETE        ->  /documents/{document_id}

Pages:
    Ask EvidenceLens  - chat, cited answers, evidence status, trace, feedback
    Document Library  - upload files, index URLs, sample corpus, delete documents
    System Overview   - architecture summary, backend /health and /stats

Start the backend FIRST (in one terminal):
    python -m uvicorn backend:api --host 127.0.0.1 --port 8000

Then start the UI (in another terminal):
    python -m streamlit run frontend.py

Set API_BASE_URL in .env if the backend runs on a different host or port.
"""
from __future__ import annotations

import os
import uuid

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (backend connection + display flags for the sidebar)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


# ---------------------------------------------------------------------------
# Streamlit user interface
# ---------------------------------------------------------------------------
def run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="EvidenceLens | Documentation Intelligence",
                       page_icon="🔎", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
    .stApp {background:linear-gradient(135deg,#0b1020 0%,#10182b 55%,#111827 100%); color:#e8edf7;}
    [data-testid="stSidebar"] {background:#0b1220;border-right:1px solid #26334a;}
    [data-testid="stHeader"] {background:rgba(11,16,32,.8);}
    .hero {padding:28px 30px;border:1px solid #263750;border-radius:20px;
      background:linear-gradient(120deg,rgba(26,45,75,.92),rgba(18,30,51,.92));margin-bottom:22px;}
    .brand {font-family:Manrope,sans-serif;font-size:31px;font-weight:800;letter-spacing:-1px;color:#f4f7ff;}
    .sub {color:#9fb0ca;font-size:14px;margin-top:4px;}
    .metric {background:#121e32;border:1px solid #293b56;border-radius:14px;padding:15px 17px;}
    .metric-label {font-size:12px;color:#91a4c0;text-transform:uppercase;letter-spacing:1px;}
    .metric-value {font-size:25px;font-weight:700;color:#edf4ff;margin-top:4px;}
    .answer {background:#111d30;border:1px solid #2b405e;border-radius:16px;padding:22px 24px;line-height:1.75;}
    .small-muted {color:#9aabc4;font-size:12px;}
    div.stButton>button {border-radius:10px;border:1px solid #3a5275;}
    div[data-testid="stChatInput"] textarea {border-radius:14px;}
    .stTabs [data-baseweb="tab"] {color:#b9c8dc;}
    </style>
    """, unsafe_allow_html=True)

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "chat" not in st.session_state:
        st.session_state.chat = []

    with st.sidebar:
        st.markdown("## 🔎 EvidenceLens")
        st.caption("Self-Corrective Documentation Intelligence")
        if st.button("＋ New conversation", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat = []
            st.rerun()
        st.divider()
        st.markdown("**Workspace**")
        page = st.radio("Navigate", ["Ask EvidenceLens", "Document Library", "System Overview"],
                        label_visibility="collapsed")
        st.divider()
        st.markdown("**Configuration**")
        st.caption(f"LLM: {'Connected' if GROQ_API_KEY else 'API key required'}")
        st.caption(f"Web fallback: {'Enabled' if ENABLE_WEB_SEARCH and TAVILY_API_KEY else 'Disabled'}")
        st.caption(f"Session: `{st.session_state.session_id[:8]}`")

    # -----------------------------------------------------------------------
    # Backend connection helpers
    # -----------------------------------------------------------------------
    def api_get(path: str):
        return requests.get(f"{API_BASE_URL}{path}", timeout=10)

    def api_post(path: str, **kwargs):
        return requests.post(f"{API_BASE_URL}{path}", timeout=180, **kwargs)

    if page == "Ask EvidenceLens":
        st.markdown("""
        <div class="hero">
          <div class="brand">EvidenceLens</div>
          <div class="sub">Ask technical questions. Inspect the evidence. Trust what can be verified.</div>
        </div>
        """, unsafe_allow_html=True)
        try:
            stats = api_get("/stats").json()
            cols = st.columns(4)
            for col, label, value in zip(cols, ["Indexed documents", "Knowledge chunks", "Feedback entries", "Active sessions"],
                                         [stats["documents"], stats["chunks"], stats["feedback_entries"], stats["sessions"]]):
                col.markdown(f'<div class="metric"><div class="metric-label">{label}</div>'
                             f'<div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
        except Exception:
            st.info("Start the FastAPI backend to connect the workspace.")
        st.write("")
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("result"):
                    result = msg["result"]
                    st.caption(f"Evidence: {result.get('evidence_status')} · Verified: {result.get('verification', {}).get('verified', False)}")
                    with st.expander("Inspect sources and retrieval trace"):
                        st.json({"citations": result.get("citations"), "trace": result.get("trace"),
                                 "retrieved_evidence": result.get("retrieved_evidence")})
        question = st.chat_input("Ask a question about your indexed technical documentation...")
        if question:
            st.session_state.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing question, retrieving evidence and checking the answer..."):
                    try:
                        response = api_post("/query", json={
                            "question": question, "session_id": st.session_state.session_id
                        })
                        response.raise_for_status()
                        result = response.json()
                        st.markdown(result["answer"])
                        status = result.get("evidence_status", "UNKNOWN")
                        verified = result.get("verification", {}).get("verified", False)
                        st.caption(f"Evidence: **{status}** · Grounding verified: **{'Yes' if verified else 'No / unavailable'}**")
                        if result.get("citations"):
                            with st.expander("Sources and evidence"):
                                for source in result["citations"]:
                                    st.markdown(f"**[{source['citation_id']}] {source['document_name']}**")
                                    st.caption(f"{source.get('section')} · {source.get('chunk_id')} · {source.get('source_type')}")
                                    if source.get("url"):
                                        st.markdown(f"[Open source]({source['url']})")
                                    st.write(source.get("excerpt", ""))
                        with st.expander("Retrieval trace"):
                            st.json(result.get("trace", {}))
                            st.json(result.get("verification", {}))
                        st.session_state.chat.append({"role": "assistant", "content": result["answer"], "result": result})
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            comment = st.text_input("Optional feedback comment", key=f"comment_{len(st.session_state.chat)}")
                        with c2:
                            rating = st.radio("Rate this answer", ["👍 Helpful", "👎 Not helpful"],
                                              horizontal=True, key=f"rating_{len(st.session_state.chat)}")
                        if st.button("Submit feedback", key=f"feedback_{len(st.session_state.chat)}"):
                            fb = api_post("/feedback", json={
                                "question": question, "answer": result["answer"],
                                "session_id": st.session_state.session_id,
                                "rating": "up" if rating.startswith("👍") else "down",
                                "comment": comment or None,
                            })
                            if fb.ok:
                                st.success("Feedback saved. Thank you.")
                            else:
                                st.error("Feedback could not be saved.")
                    except requests.ConnectionError as exc:
                        st.error(f"Cannot connect to FastAPI at {API_BASE_URL}. Start the backend terminal first. Details: {exc}")
                    except requests.HTTPError as exc:
                        detail = exc.response.text[:1000] if exc.response is not None else str(exc)
                        st.error(f"The API returned an error: {detail}")
                    except requests.RequestException as exc:
                        st.error(f"Request/network error: {exc}")
                    except Exception as exc:
                        st.error(f"Request failed: {exc}")
        if not st.session_state.chat:
            st.markdown("#### Try a question")
            examples = [
                "How does dependency injection work in FastAPI?",
                "How should request validation errors be handled?",
                "What is the difference between a Pydantic model and a Python dataclass?",
            ]
            cols = st.columns(len(examples))
            for col, example in zip(cols, examples):
                if col.button(example, use_container_width=True):
                    st.session_state.pending_question = example
                    st.rerun()
            if st.session_state.get("pending_question"):
                st.info(f"Suggested question: {st.session_state.pending_question}. Submit it in the chat box.")

    elif page == "Document Library":
        st.markdown("## Document Library")
        st.caption("Upload documentation or index a public documentation URL.")
        if st.button("Index included sample corpus", type="secondary"):
            try:
                sample_response = api_post("/ingest/sample")
                sample_response.raise_for_status()
                st.success(sample_response.json())
                st.rerun()
            except Exception as exc:
                st.error(f"Sample corpus indexing failed: {exc}")
        uploaded = st.file_uploader("Choose documents", type=["pdf", "txt", "md", "markdown", "html", "htm"],
                                    accept_multiple_files=True)
        if uploaded and st.button("Upload and index", type="primary"):
            progress = st.progress(0)
            for i, file in enumerate(uploaded):
                try:
                    response = api_post("/ingest", files={"file": (file.name, file.getvalue(), file.type)})
                    response.raise_for_status()
                    data = response.json()
                    st.success(f"{data['name']}: {data['chunks']} chunks ({data['status']})")
                except Exception as exc:
                    st.error(f"{file.name}: {exc}")
                progress.progress((i + 1) / len(uploaded))
        url = st.text_input("Or ingest a public documentation URL")
        if st.button("Index URL"):
            if not url.strip():
                st.warning("Enter a URL first.")
            else:
                try:
                    response = api_post("/ingest", data={"url": url.strip()})
                    response.raise_for_status()
                    st.success(response.json())
                except Exception as exc:
                    st.error(f"URL ingestion failed: {exc}")
        st.divider()
        try:
            documents = api_get("/documents").json().get("documents", [])
            if not documents:
                st.info("No documents indexed yet.")
            for doc in documents:
                with st.container(border=True):
                    left, right = st.columns([5, 1])
                    left.markdown(f"**{doc['name']}**")
                    left.caption(f"{doc['source_type']} · {doc['chunk_count']} chunks · Updated {doc['updated_at'][:19]}")
                    with right:
                        if st.button("Delete", key=f"del_{doc['document_id']}"):
                            deleted = requests.delete(f"{API_BASE_URL}/documents/{doc['document_id']}", timeout=30)
                            if deleted.ok:
                                st.success("Document removed.")
                                st.rerun()
                            else:
                                st.error("Could not remove document.")
        except Exception:
            st.warning("FastAPI backend is not running.")

    else:
        st.markdown("## System Overview")
        st.markdown("""
        EvidenceLens uses a bounded LangGraph workflow to analyze a question, retrieve relevant
        chunks, grade evidence, rewrite failed searches, generate a cited answer, and verify
        the answer against the retrieved context.

        **Core safeguards**
        - Relevance grading before generation.
        - Bounded corrective retrieval attempts.
        - Explicit insufficient-evidence responses.
        - Citation mapping to indexed chunks.
        - Separate grounding verification.
        - Persistent session history and feedback.
        - Incremental document indexing using content hashes.
        """)
        st.code("Backend:  python -m uvicorn backend:api --host 127.0.0.1 --port 8000\nFrontend: python -m streamlit run frontend.py", language="bash")
        try:
            st.json(api_get("/health").json())
            st.json(api_get("/stats").json())
        except Exception:
            st.warning("Start the API server to inspect system status.")


if __name__ == "__main__":
    run_streamlit()
