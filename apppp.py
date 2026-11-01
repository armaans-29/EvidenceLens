
"""
EvidenceLens: Self-Corrective RAG Documentation Intelligence
Single-file application: FastAPI backend + Streamlit frontend + LangGraph workflow.

Run API:
    python -m uvicorn apppp:api --host 127.0.0.1 --port 8000

Run UI (in another terminal):
    python -m streamlit run apppp.py

Configure GROQ_API_KEY in .env or environment variables.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TypedDict
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration and logging
# ---------------------------------------------------------------------------
APP_NAME = "EvidenceLens"
PROJECT_DIR = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_DIR / "docs"
BASE_DIR = Path(os.getenv("EVIDENCELENS_DATA_DIR", str(PROJECT_DIR / "data"))).resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma"
DB_PATH = BASE_DIR / "evidencelens.db"
for directory in (BASE_DIR, UPLOAD_DIR, CHROMA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("evidencelens")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "evidencelens")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.20"))
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    session_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=12000)
    session_id: Optional[str] = None
    rating: str = Field(pattern="^(up|down)$")
    comment: Optional[str] = Field(default=None, max_length=2000)


class IngestURLRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2000)


class SourceCitation(BaseModel):
    citation_id: str
    document_name: str
    section: str = "Document"
    chunk_id: str
    source_type: str = "upload"
    url: Optional[str] = None
    excerpt: Optional[str] = None


# ---------------------------------------------------------------------------
# Persistent SQLite storage: documents, sessions, messages, feedback
# ---------------------------------------------------------------------------
def db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            source TEXT NOT NULL,
            source_type TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            session_id TEXT,
            rating TEXT NOT NULL,
            comment TEXT,
            created_at TEXT NOT NULL
        );
        """)
init_db()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_history(session_id: str, limit: int = 8) -> list[dict[str, str]]:
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def save_message(session_id: str, role: str, content: str) -> None:
    now = utc_now()
    with db_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions(session_id, created_at, updated_at) VALUES(?,?,?)",
            (session_id, now, now),
        )
        conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) VALUES(?,?,?,?)",
            (session_id, role, content, now),
        )
        conn.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (now, session_id))


def save_feedback(item: FeedbackRequest) -> int:
    with db_connection() as conn:
        cur = conn.execute(
            """INSERT INTO feedback(question, answer, session_id, rating, comment, created_at)
               VALUES(?,?,?,?,?,?)""",
            (item.question, item.answer, item.session_id, item.rating, item.comment, utc_now()),
        )
        return int(cur.lastrowid)


# ---------------------------------------------------------------------------
# Lazy model and vector-store initialization
# ---------------------------------------------------------------------------
_embeddings = None
_vector_store = None
_llm = None
_graph = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
    return _embeddings


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        from langchain_chroma import Chroma
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=get_embeddings(),
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _vector_store


def get_llm():
    global _llm
    if _llm is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Add it to your .env file before asking questions."
            )
        from langchain_groq import ChatGroq
        _llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.1,
            max_retries=1,
            timeout=60,
            max_tokens=2500,
        )
    return _llm


def invoke_text(prompt: str) -> str:
    response = get_llm().invoke(prompt)
    return str(response.content).strip()


def extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("Model did not return a JSON object.")
    return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# Document ingestion and safe URL loading
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".html", ".htm"}


def safe_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only valid HTTP/HTTPS URLs are accepted.")
    host = parsed.hostname.lower()
    if host in {"localhost", "metadata.google.internal"} or host.endswith(".local"):
        raise ValueError("Local and internal hostnames are not allowed.")
    try:
        addresses = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
        for result in addresses:
            ip = ipaddress.ip_address(result[4][0])
            if not ip.is_global:
                raise ValueError("Private, loopback, and reserved network addresses are blocked.")
    except socket.gaierror as exc:
        raise ValueError("The supplied hostname could not be resolved.") from exc
    return url


def extract_document_text(filename: str, raw: bytes) -> tuple[str, str]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")
    if suffix == ".pdf":
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"\n\n[Page {page_num}]\n{text}")
        content = "\n".join(pages)
    elif suffix in {".html", ".htm"}:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw.decode("utf-8", errors="replace"), "html.parser")
        for element in soup(["script", "style", "nav", "footer", "noscript"]):
            element.decompose()
        content = soup.get_text("\n", strip=True)
    else:
        content = raw.decode("utf-8", errors="replace")
    content = content.replace("\x00", "")
    content = re.sub(r"[ \t]+", " ", content)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    if len(content) < 30:
        raise ValueError("The document contains too little extractable text.")
    return content, suffix.lstrip(".")


def fetch_url_text(url: str) -> tuple[str, str]:
    safe_public_url(url)
    response = requests.get(
        url,
        timeout=(5, 20),
        headers={"User-Agent": "EvidenceLens/1.0 documentation-ingestion"},
        allow_redirects=False,
        stream=True,
    )
    if response.is_redirect:
        raise ValueError("Redirects are disabled for URL ingestion. Submit the final public URL.")
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if not any(x in content_type for x in ("text/html", "text/plain", "text/markdown", "application/xhtml")):
        raise ValueError(f"Unsupported URL content type: {content_type or 'unknown'}")
    chunks = []
    total = 0
    for block in response.iter_content(65536):
        total += len(block)
        if total > 8 * 1024 * 1024:
            raise ValueError("URL content exceeds the 8 MB ingestion limit.")
        chunks.append(block)
    raw = b"".join(chunks)
    text, _ = extract_document_text("url.html" if "html" in content_type else "url.txt", raw)
    return text, "html" if "html" in content_type else "txt"


def split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 130) -> list[tuple[str, str]]:
    """Heading-aware character chunker that avoids empty chunks."""
    if chunk_size < 200 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size/overlap configuration.")
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    heading = "Document"
    buffer: list[str] = []
    for line in lines:
        stripped = line.strip()
        is_heading = bool(re.match(r"^(#{1,6}\s+|[A-Z][A-Z0-9 /&:()_-]{4,}$)", stripped))
        if is_heading and buffer:
            sections.append((heading, "\n".join(buffer).strip()))
            buffer = []
        if is_heading:
            heading = re.sub(r"^#{1,6}\s*", "", stripped).strip()
        if stripped:
            buffer.append(line)
    if buffer:
        sections.append((heading, "\n".join(buffer).strip()))

    result: list[tuple[str, str]] = []
    for section, body in sections or [("Document", text)]:
        if len(body) <= chunk_size:
            if body.strip():
                result.append((section, body.strip()))
            continue
        start = 0
        while start < len(body):
            end = min(start + chunk_size, len(body))
            if end < len(body):
                boundary = max(body.rfind("\n", start, end), body.rfind(". ", start, end))
                if boundary > start + int(chunk_size * 0.60):
                    end = boundary + 1
            piece = body[start:end].strip()
            if piece:
                result.append((section, piece))
            if end >= len(body):
                break
            start = max(start + 1, end - overlap)
    return result


def ingest_text(name: str, text: str, source: str, source_type: str) -> dict[str, Any]:
    started = time.perf_counter()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    with db_connection() as conn:
        existing = conn.execute(
            "SELECT document_id, content_hash, chunk_count FROM documents WHERE source=?",
            (source,),
        ).fetchone()
    if existing and existing["content_hash"] == digest:
        return {
            "status": "unchanged", "document_id": existing["document_id"],
            "name": name, "chunks": existing["chunk_count"], "duration_seconds": 0.0,
        }

    document_id = existing["document_id"] if existing else str(uuid.uuid4())
    pieces = split_into_chunks(text)
    if not pieces:
        raise ValueError("No indexable text was found.")
    from langchain_core.documents import Document
    docs = []
    ids = []
    for index, (section, content) in enumerate(pieces):
        chunk_id = f"{document_id}:{index}"
        ids.append(chunk_id)
        docs.append(Document(
            page_content=content,
            metadata={
                "document_id": document_id,
                "document_name": name,
                "source": source,
                "source_type": source_type,
                "section": section,
                "chunk_id": chunk_id,
                "chunk_index": index,
                "content_hash": digest,
            },
        ))
    store = get_vector_store()
    if existing:
        store.delete(where={"document_id": document_id})
    # Chroma add_documents batches internally; batch explicitly to control memory.
    batch_size = 64
    for start in range(0, len(docs), batch_size):
        store.add_documents(docs[start:start + batch_size], ids=ids[start:start + batch_size])

    now = utc_now()
    with db_connection() as conn:
        conn.execute(
            """INSERT INTO documents(document_id,name,source,source_type,content_hash,chunk_count,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(document_id) DO UPDATE SET name=excluded.name, source=excluded.source,
               source_type=excluded.source_type, content_hash=excluded.content_hash,
               chunk_count=excluded.chunk_count, updated_at=excluded.updated_at""",
            (document_id, name, source, source_type, digest, len(docs), now, now),
        )
    return {
        "status": "indexed", "document_id": document_id, "name": name,
        "chunks": len(docs), "duration_seconds": round(time.perf_counter() - started, 3),
    }


# ---------------------------------------------------------------------------
# Retrieval, grading, query rewriting, generation, and verification
# ---------------------------------------------------------------------------
def retrieve(query: str, k: int = TOP_K) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    results = get_vector_store().similarity_search_with_relevance_scores(query, k=k)
    chunks = []
    for doc, score in results:
        chunks.append({
            "text": doc.page_content,
            "metadata": dict(doc.metadata),
            "score": float(score),
        })
    return chunks, round(time.perf_counter() - started, 3)


def analyze_query(question: str, history: list[dict[str, str]]) -> dict[str, str]:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    recent = "\n".join(f'{m["role"]}: {m["content"][:500]}' for m in history[-4:])
    prompt = f"""You analyze questions for a technical documentation search engine.
Return ONLY valid JSON with keys: rewritten_query, query_type, needs_context.
query_type must be one of conceptual, how-to, troubleshooting, api-reference, comparison, other.
Preserve the original intent. Expand only when useful. Resolve pronouns from conversation history.
Original question: {question}
Recent conversation:
{recent or "None"}
"""
    try:
        data = extract_json(invoke_text(prompt))
        return {
            "rewritten_query": str(data.get("rewritten_query") or question).strip(),
            "query_type": str(data.get("query_type") or "other"),
        }
    except Exception as exc:
        logger.warning("Query analysis fallback: %s", exc)
        return {"rewritten_query": question, "query_type": "other"}


def grade_chunks(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not chunks:
        return []
    # One structured LLM grading call for all candidates reduces latency.
    payload = [
        {"id": i, "section": c["metadata"].get("section", "Document"),
         "text": c["text"][:1800], "retrieval_score": round(c["score"], 3)}
        for i, c in enumerate(chunks)
    ]
    prompt = f"""You are a strict technical-document relevance evaluator.
For each chunk, decide if it contains useful evidence for answering the question.
Return ONLY JSON: {{"grades":[{{"id":0,"relevant":true,"support":"direct|indirect|none","reason":"brief factual reason"}}]}}
Do not mark a chunk relevant merely because it shares keywords.
Question: {question}
Chunks: {json.dumps(payload, ensure_ascii=False)}
"""
    try:
        data = extract_json(invoke_text(prompt))
        grades = {int(g["id"]): g for g in data.get("grades", [])}
    except Exception as exc:
        logger.warning("Chunk grading failed; conservative retrieval-score fallback: %s", exc)
        grades = {}
    output = []
    for i, chunk in enumerate(chunks):
        grade = grades.get(i, {})
        relevant = bool(grade.get("relevant", False))
        # A modest retrieval threshold is a secondary safeguard, not the relevance judge.
        relevant = relevant and chunk["score"] >= SIMILARITY_THRESHOLD
        output.append({**chunk, "relevant": relevant,
                       "support": grade.get("support", "none"),
                       "grade_reason": grade.get("reason", "No valid relevance grade.")})
    return output


def rewrite_query(question: str, previous_query: str, attempt: int) -> str:
    prompt = f"""Rewrite this technical documentation search query to retrieve different, more useful evidence.
Do not answer the question. Change terminology or focus, avoid repeating the prior query.
Return only the new query, no quotation marks.
Question: {question}
Previous query: {previous_query}
Attempt: {attempt}
"""
    try:
        rewritten = invoke_text(prompt).strip().strip('"')
    except Exception as exc:
        logger.warning("Query rewrite failed; using deterministic expansion: %s", type(exc).__name__)
        rewritten = ""
    if not rewritten or rewritten.lower() == previous_query.lower():
        rewritten = f"{question} implementation configuration examples API behavior"
    return rewritten[:1000]


def web_search(query: str) -> list[dict[str, str]]:
    if not ENABLE_WEB_SEARCH or not TAVILY_API_KEY:
        return []
    response = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": TAVILY_API_KEY, "query": query, "search_depth": "basic",
              "max_results": 4, "include_answer": False},
        timeout=20,
    )
    response.raise_for_status()
    results = []
    for item in response.json().get("results", []):
        results.append({
            "title": str(item.get("title", "Web result")),
            "url": str(item.get("url", "")),
            "content": str(item.get("content", ""))[:3500],
            "source_type": "web",
        })
    return results


def generate_answer(question: str, evidence: list[dict[str, Any]],
                    web_results: list[dict[str, str]]) -> tuple[str, str]:
    context_parts = []
    for i, item in enumerate(evidence, start=1):
        meta = item["metadata"]
        context_parts.append(
            f"[S{i}] Document: {meta.get('document_name')} | Section: {meta.get('section')} "
            f"| Chunk: {meta.get('chunk_id')}\n{item['text']}"
        )
    next_id = len(context_parts) + 1
    for item in web_results:
        context_parts.append(
            f"[W{next_id}] WEB: {item['title']} | URL: {item['url']}\n{item['content']}"
        )
        next_id += 1
    if not context_parts:
        return ("I could not find sufficient supporting information in the indexed documentation. "
                "Please add relevant documentation or enable web search and try again.",
                "INSUFFICIENT_EVIDENCE")
    prompt = f"""You are EvidenceLens, a careful technical documentation assistant.
Answer the user's question using ONLY the evidence below.
Rules:
- Cite factual claims inline using [S1], [S2], or [W3] identifiers exactly as provided.
- Never invent APIs, parameters, behavior, filenames, URLs, or citations.
- If only part is supported, answer that part and state what is missing.
- If evidence conflicts, describe the conflict and cite both sources.
- If evidence is insufficient, say so clearly.
- Be concise but technically useful. Include examples only when supported or clearly labelled illustrative.
Question: {question}
Evidence:
{chr(10).join(context_parts)}
"""
    answer = invoke_text(prompt)
    # A separate verifier assigns the evidence status. Avoid a redundant LLM call here.
    return answer, "PENDING_VERIFICATION"


def verify_answer(answer: str, evidence: list[dict[str, Any]],
                  web_results: list[dict[str, str]]) -> dict[str, Any]:
    context = []
    for i, item in enumerate(evidence, start=1):
        context.append(f"[S{i}] {item['text'][:1800]}")
    for i, item in enumerate(web_results, start=len(context) + 1):
        context.append(f"[W{i}] {item['content'][:1800]}")
    if not context:
        return {"verified": False, "evidence_status": "INSUFFICIENT_EVIDENCE",
                "unsupported_claims": ["No evidence available."],
                "summary": "No evidence was available for verification."}
    prompt = f"""You are a strict answer-grounding evaluator.
Compare every material factual claim in the answer with the supplied evidence.
Return ONLY valid JSON with this exact schema:
{{"verified":true,"evidence_status":"DIRECTLY_SUPPORTED|PARTIALLY_SUPPORTED|INSUFFICIENT_EVIDENCE|CONTRADICTORY_EVIDENCE","unsupported_claims":[],"summary":"brief explanation"}}
Rules:
- verified=true only if every material factual claim is supported and every citation ID exists in the evidence.
- If some claims are supported but important details are missing, use PARTIALLY_SUPPORTED.
- If evidence does not answer the question, use INSUFFICIENT_EVIDENCE.
- If sources disagree, use CONTRADICTORY_EVIDENCE.
Answer: {answer}
Evidence:
{chr(10).join(context)}
"""
    try:
        data = extract_json(invoke_text(prompt))
        allowed = {"DIRECTLY_SUPPORTED", "PARTIALLY_SUPPORTED",
                   "INSUFFICIENT_EVIDENCE", "CONTRADICTORY_EVIDENCE"}
        status = str(data.get("evidence_status", "PARTIALLY_SUPPORTED"))
        if status not in allowed:
            status = "PARTIALLY_SUPPORTED"
        unsupported = data.get("unsupported_claims", [])
        if not isinstance(unsupported, list):
            unsupported = [str(unsupported)]
        return {
            "verified": bool(data.get("verified", False)),
            "evidence_status": status,
            "unsupported_claims": [str(x)[:400] for x in unsupported[:8]],
            "summary": str(data.get("summary", "Verification completed."))[:500],
        }
    except Exception as exc:
        logger.warning("Grounding verification unavailable: %s", type(exc).__name__)
        return {"verified": False, "evidence_status": "PARTIALLY_SUPPORTED",
                "unsupported_claims": [],
                "summary": f"Verification unavailable: {type(exc).__name__}"}


# ---------------------------------------------------------------------------
# LangGraph StateGraph workflow
# ---------------------------------------------------------------------------
class RAGState(TypedDict, total=False):
    question: str
    session_id: str
    history: list[dict[str, str]]
    rewritten_query: str
    query_type: str
    retrieved: list[dict[str, Any]]
    graded: list[dict[str, Any]]
    relevant: list[dict[str, Any]]
    retry_count: int
    retrieval_seconds: float
    web_results: list[dict[str, str]]
    answer: str
    evidence_status: str
    verification: dict[str, Any]
    revision_count: int
    trace: dict[str, Any]
    final_status: str


def build_graph():
    from langgraph.graph import StateGraph, START, END

    def query_node(state: RAGState) -> dict[str, Any]:
        analysis = analyze_query(state["question"], state.get("history", []))
        return {"rewritten_query": analysis["rewritten_query"],
                "query_type": analysis["query_type"]}

    def retrieval_node(state: RAGState) -> dict[str, Any]:
        chunks, elapsed = retrieve(state.get("rewritten_query", state["question"]))
        return {"retrieved": chunks,
                "retrieval_seconds": state.get("retrieval_seconds", 0.0) + elapsed}

    def grading_node(state: RAGState) -> dict[str, Any]:
        graded = grade_chunks(state["question"], state.get("retrieved", []))
        return {"graded": graded, "relevant": [x for x in graded if x["relevant"]]}

    def route_after_grading(state: RAGState) -> str:
        if state.get("relevant"):
            return "generate"
        if state.get("retry_count", 0) < MAX_RETRIES:
            return "rewrite"
        return "web"

    def rewrite_node(state: RAGState) -> dict[str, Any]:
        next_attempt = state.get("retry_count", 0) + 1
        new_query = rewrite_query(
            state["question"], state.get("rewritten_query", state["question"]), next_attempt
        )
        return {"rewritten_query": new_query, "retry_count": next_attempt}

    def route_after_rewrite(state: RAGState) -> str:
        return "retrieve"

    def web_node(state: RAGState) -> dict[str, Any]:
        try:
            results = web_search(state.get("rewritten_query", state["question"]))
        except Exception as exc:
            logger.warning("Web fallback unavailable: %s", exc)
            results = []
        return {"web_results": results}

    def route_after_web(state: RAGState) -> str:
        return "generate" if state.get("web_results") else "abstain"

    def generate_node(state: RAGState) -> dict[str, Any]:
        answer, status = generate_answer(
            state["question"], state.get("relevant", []), state.get("web_results", [])
        )
        return {"answer": answer, "evidence_status": status}

    def verify_node(state: RAGState) -> dict[str, Any]:
        result = verify_answer(
            state.get("answer", ""), state.get("relevant", []), state.get("web_results", [])
        )
        return {"verification": result,
                "evidence_status": result.get("evidence_status", "PARTIALLY_SUPPORTED")}

    def route_after_verify(state: RAGState) -> str:
        if state.get("verification", {}).get("verified") or state.get("revision_count", 0) >= 1:
            return "finish"
        return "revise"

    def revise_node(state: RAGState) -> dict[str, Any]:
        revision_evidence = []
        for i, item in enumerate(state.get("relevant", []), start=1):
            meta = item.get("metadata", {})
            revision_evidence.append(
                f"[S{i}] {meta.get('document_name', 'Document')} / "
                f"{meta.get('section', 'Section')}\n{item['text']}"
            )
        web_offset = len(revision_evidence) + 1
        for i, item in enumerate(state.get("web_results", []), start=web_offset):
            revision_evidence.append(f"[W{i}] {item['title']} | {item['url']}\n{item['content']}")
        prompt = f"""Revise the answer using only the evidence below. Remove unsupported claims.
Use only citation IDs that appear in the evidence. If evidence cannot support an answer, explicitly abstain.
Question: {state['question']}
Previous answer: {state.get('answer', '')}
Unsupported claims: {state.get('verification', {}).get('unsupported_claims', [])}
Evidence:
{chr(10).join(revision_evidence)}
"""
        try:
            answer = invoke_text(prompt)
        except Exception:
            answer = ("The available evidence could not be verified sufficiently. "
                      "Please consult the cited documentation directly.")
        return {"answer": answer, "revision_count": state.get("revision_count", 0) + 1}

    def abstain_node(state: RAGState) -> dict[str, Any]:
        return {
            "answer": ("I could not find sufficient relevant evidence in the indexed documents"
                       + (" or the configured web sources" if ENABLE_WEB_SEARCH else "")
                       + ". I don't want to guess. Try adding the relevant technical documentation "
                         "or asking a narrower question."),
            "evidence_status": "INSUFFICIENT_EVIDENCE",
            "verification": {"verified": False, "unsupported_claims": [],
                             "summary": "Abstained because no sufficient evidence was found."},
        }

    def finish_node(state: RAGState) -> dict[str, Any]:
        return {"final_status": "completed"}

    graph = StateGraph(RAGState)
    graph.add_node("analyze", query_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("grade", grading_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("web_search", web_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("revise", revise_node)
    graph.add_node("abstain", abstain_node)
    graph.add_node("finish", finish_node)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grading, {
        "generate": "generate", "rewrite": "rewrite", "web": "web_search"
    })
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges("web_search", route_after_web, {
        "generate": "generate", "abstain": "abstain"
    })
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {
        "finish": "finish", "revise": "revise"
    })
    graph.add_edge("revise", "verify")
    graph.add_edge("abstain", "finish")
    graph.add_edge("finish", END)
    return graph.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def make_citations(relevant: list[dict[str, Any]],
                   web_results: list[dict[str, str]]) -> list[dict[str, Any]]:
    citations = []
    for i, item in enumerate(relevant, start=1):
        meta = item["metadata"]
        citations.append({
            "citation_id": f"S{i}",
            "document_name": meta.get("document_name", "Unknown document"),
            "section": meta.get("section", "Document"),
            "chunk_id": meta.get("chunk_id", "unknown"),
            "source_type": meta.get("source_type", "upload"),
            "url": meta.get("source") if str(meta.get("source", "")).startswith("http") else None,
            "excerpt": item["text"][:500],
            "retrieval_score": round(item["score"], 4),
        })
    offset = len(citations) + 1
    for i, item in enumerate(web_results, start=offset):
        citations.append({
            "citation_id": f"W{i}", "document_name": item["title"],
            "section": "External web result", "chunk_id": f"web:{i}",
            "source_type": "web", "url": item["url"], "excerpt": item["content"][:500],
        })
    return citations


def run_question(question: str, session_id: Optional[str] = None) -> dict[str, Any]:
    started = time.perf_counter()
    session_id = session_id or str(uuid.uuid4())
    history = get_history(session_id)
    with db_connection() as conn:
        indexed_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    if indexed_count == 0:
        answer = ("No documents have been indexed yet. Open Document Library, upload a PDF/MD/TXT/HTML file, "
                  "or index the included sample corpus, then ask your question again.")
        save_message(session_id, "user", question)
        save_message(session_id, "assistant", answer)
        return {
            "question": question, "answer": answer,
            "evidence_status": "INSUFFICIENT_EVIDENCE", "citations": [],
            "verification": {"verified": False, "evidence_status": "INSUFFICIENT_EVIDENCE",
                             "unsupported_claims": [], "summary": "The document library is empty."},
            "session_id": session_id,
            "trace": {"retrieved_chunks": 0, "relevant_chunks": 0, "retry_attempts": 0,
                      "retrieval_seconds": 0.0, "total_seconds": round(time.perf_counter() - started, 3)},
            "retrieved_evidence": [],
        }
    initial: RAGState = {
        "question": question, "session_id": session_id, "history": history,
        "rewritten_query": question, "retry_count": 0, "revision_count": 0,
        "retrieval_seconds": 0.0, "retrieved": [], "graded": [], "relevant": [],
        "web_results": [],
    }
    state = get_graph().invoke(initial, config={"recursion_limit": 30})
    citations = make_citations(state.get("relevant", []), state.get("web_results", []))
    answer = state.get("answer", "The workflow did not produce an answer.")
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", answer)
    graded = state.get("graded", [])
    trace = {
        "rewritten_query": state.get("rewritten_query", question),
        "query_type": state.get("query_type", "other"),
        "retrieved_chunks": len(state.get("retrieved", [])),
        "relevant_chunks": len(state.get("relevant", [])),
        "rejected_chunks": sum(1 for x in graded if not x.get("relevant")),
        "retry_attempts": state.get("retry_count", 0),
        "web_fallback_used": bool(state.get("web_results")),
        "retrieval_seconds": state.get("retrieval_seconds", 0.0),
        "total_seconds": round(time.perf_counter() - started, 3),
        "revision_attempts": state.get("revision_count", 0),
    }
    return {
        "question": question, "answer": answer,
        "evidence_status": state.get("evidence_status", "INSUFFICIENT_EVIDENCE"),
        "citations": citations,
        "verification": state.get("verification", {"verified": False}),
        "session_id": session_id, "trace": trace,
        "retrieved_evidence": [
            {"text": x["text"], "metadata": x["metadata"],
             "score": x["score"], "relevant": x.get("relevant", False),
             "grade_reason": x.get("grade_reason", "")}
            for x in graded
        ],
    }


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
api = FastAPI(
    title="EvidenceLens API",
    description="Self-corrective, evidence-grounded technical documentation assistant.",
    version="1.1.0",
)
api.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8501").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@api.get("/health")
def health():
    return {
        "status": "ok",
        "application": APP_NAME,
        "llm_configured": bool(GROQ_API_KEY),
        "web_search_enabled": bool(ENABLE_WEB_SEARCH and TAVILY_API_KEY),
        "vector_store_initialized": _vector_store is not None,
    }


@api.post("/query")
def query_endpoint(item: QueryRequest):
    try:
        return run_question(item.question.strip(), item.session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=502, detail=f"Query processing failed ({type(exc).__name__}). Check the API terminal for details.")


@api.post("/ingest")
async def ingest_endpoint(
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
):
    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide either a document file or a URL.")
    try:
        if file:
            filename = Path(file.filename or "uploaded_document").name
            if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise HTTPException(status_code=415, detail="Unsupported file type.")
            raw = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
            if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB.")
            text, kind = extract_document_text(filename, raw)
            result = ingest_text(filename, text, f"upload:{filename}", kind)
        else:
            safe_url = safe_public_url(url or "")
            text, kind = fetch_url_text(safe_url)
            name = Path(urlparse(safe_url).path).name or urlparse(safe_url).netloc
            result = ingest_text(name, text, safe_url, kind)
        return {"success": True, **result}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {str(exc)[:300]}")


@api.post("/ingest/sample")
def ingest_sample_corpus():
    """Index the three small technical documents shipped with the repository."""
    if not DOCS_DIR.exists():
        raise HTTPException(status_code=404, detail="The docs/ sample corpus folder is missing.")
    results = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        try:
            content = path.read_text(encoding="utf-8")
            results.append(ingest_text(path.name, content, f"sample:{path.name}", "markdown"))
        except Exception as exc:
            logger.exception("Could not index sample document %s", path.name)
            results.append({"name": path.name, "status": "failed", "error": type(exc).__name__})
    if not results:
        raise HTTPException(status_code=404, detail="No Markdown sample documents were found in docs/.")
    return {"success": all(x.get("status") != "failed" for x in results), "documents": results}


@api.get("/documents")
def documents_endpoint():
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT document_id,name,source,source_type,chunk_count,created_at,updated_at "
            "FROM documents ORDER BY updated_at DESC"
        ).fetchall()
    return {"documents": [dict(row) for row in rows], "count": len(rows)}


@api.delete("/documents/{document_id}")
def delete_document_endpoint(document_id: str):
    with db_connection() as conn:
        row = conn.execute("SELECT document_id FROM documents WHERE document_id=?", (document_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        get_vector_store().delete(where={"document_id": document_id})
        with db_connection() as conn:
            conn.execute("DELETE FROM documents WHERE document_id=?", (document_id,))
        return {"success": True, "document_id": document_id}
    except Exception as exc:
        logger.exception("Document deletion failed")
        raise HTTPException(status_code=500, detail="Document deletion failed.")


@api.post("/feedback")
def feedback_endpoint(item: FeedbackRequest):
    try:
        feedback_id = save_feedback(item)
        return {"success": True, "feedback_id": feedback_id}
    except Exception:
        logger.exception("Feedback persistence failed")
        raise HTTPException(status_code=500, detail="Could not save feedback.")


@api.get("/stats")
def stats_endpoint():
    with db_connection() as conn:
        docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = conn.execute("SELECT COALESCE(SUM(chunk_count),0) FROM documents").fetchone()[0]
        feedback = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    return {"documents": docs, "chunks": chunks, "feedback_entries": feedback,
            "sessions": sessions, "embedding_model": EMBEDDING_MODEL, "llm_model": GROQ_MODEL}


@api.get("/sessions/{session_id}")
def session_endpoint(session_id: str):
    return {"session_id": session_id, "messages": get_history(session_id, limit=50)}


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

        uploaded = st.file_uploader(
            "Choose documents",
            type=["pdf", "txt", "md", "markdown", "html", "htm"],
            accept_multiple_files=True
        )

        if uploaded and st.button("Upload and index", type="primary"):
            progress = st.progress(0)

            for i, file in enumerate(uploaded):
                try:
                    response = api_post(
                        "/ingest",
                        files={"file": (file.name, file.getvalue(), file.type)}
                    )
                    response.raise_for_status()
                    data = response.json()
                    st.success(
                        f"{data['name']}: {data['chunks']} chunks ({data['status']})"
                    )
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
                    left.caption(
                        f"{doc['source_type']} · {doc['chunk_count']} chunks · "
                        f"Updated {doc['updated_at'][:19]}"
                    )

                    with right:
                        if st.button("Delete", key=f"del_{doc['document_id']}"):
                            deleted = requests.delete(
                                f"{API_BASE_URL}/documents/{doc['document_id']}",
                                timeout=30
                            )
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
        st.code("Browser UI: python -m streamlit run apppp.py\nAPI: python -m uvicorn apppp:api --host 127.0.0.1 --port 8000", language="bash")
        try:
            st.json(api_get("/health").json())
            st.json(api_get("/stats").json())
        except Exception:
            st.warning("Start the API server to inspect system status.")


# ---------------------------------------------------------------------------
# CLI helpers and entry point
# ---------------------------------------------------------------------------
def export_feedback_csv(output_path: str = "feedback_export.csv") -> str:
    import csv
    with db_connection() as conn:
        rows = conn.execute("SELECT * FROM feedback ORDER BY id").fetchall()
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(rows[0].keys() if rows else
                        ["id", "question", "answer", "session_id", "rating", "comment", "created_at"])
        for row in rows:
            writer.writerow(list(row))
    return output_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "export-feedback":
        print(export_feedback_csv())
    else:
        run_streamlit()

# For Fast API
# python -m uvicorn apppp:api --host 127.0.0.1 --port 8000

# For Streamlit
# python -m streamlit run apppp.py
