import os
import json
import time
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception:
    chromadb = None
    embedding_functions = None

logger = logging.getLogger("chroma_store")
logger.setLevel(logging.INFO)

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_client = None
_openai_ef = None

def _init_client():
    global _client, _openai_ef
    if _client is not None:
        return _client

    try:
        _openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY, model_name="text-embedding-3-large"
        )
    except Exception:
        _openai_ef = None

    try:
        from chromadb.config import Settings
        _client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=CHROMA_DIR
        ))
        logger.info("Chroma client created with Settings()")
    except Exception:
        _client = chromadb.Client()
        logger.info("Chroma client created with default Client()")
    return _client

def _get_client():
    global _client
    return _client or _init_client()

def _get_embedding_function():
    global _openai_ef
    if _openai_ef is None:
        try:
            _openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY, model_name="text-embedding-3-large"
            )
        except Exception:
            _openai_ef = None
    return _openai_ef

transcripts_col = None
sessions_col = None
checkpoints_col = None

def _ensure_collections():
    global transcripts_col, sessions_col, checkpoints_col
    client = _get_client()

    if transcripts_col is None:
        try:
            transcripts_col = client.get_collection("transcripts")
        except Exception:
            try:
                transcripts_col = client.create_collection(
                    "transcripts", embedding_function=_get_embedding_function()
                )
            except Exception:
                transcripts_col = client.create_collection("transcripts")

    if sessions_col is None:
        try:
            sessions_col = client.get_collection("sessions")
        except Exception:
            sessions_col = client.create_collection("sessions")

    if checkpoints_col is None:
        try:
            checkpoints_col = client.get_collection("checkpoints")
        except Exception:
            checkpoints_col = client.create_collection("checkpoints")

def _sanitize_metadata_values(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        else:
            try:
                out[k] = json.dumps(v, default=str, ensure_ascii=False)
            except Exception:
                out[k] = str(v)
    return out

def persist_message(session_id: str, role: str, content: str,
                    job_id: Optional[str] = None,
                    citations: Optional[List[Dict]] = None,
                    msg_index: Optional[int] = None):
    _ensure_collections()
    ts = datetime.utcnow().isoformat()
    safe_citations = None
    if citations:
        try:
            safe_citations = json.dumps(citations, default=str, ensure_ascii=False)
        except Exception:
            safe_citations = str(citations)

    metadata = {"session_id": session_id, "role": role, "timestamp": ts, "job_id": job_id}
    if msg_index is None:
        try:
            existing = sessions_col.get(where={"session_id": session_id}, include=["metadatas"])
            idxs = [int(m.get("msg_index", 0) or 0) for m in (existing.get("metadatas") or []) if isinstance(m, dict)]
            next_index = max(idxs) + 1 if idxs else 1
        except Exception:
            next_index = 1
    else:
        next_index = int(msg_index)
    metadata["msg_index"] = next_index
    if safe_citations is not None:
        metadata["citations"] = safe_citations

    metadata = _sanitize_metadata_values(metadata)
    doc_id = f"{session_id}::msg::{next_index}::{int(time.time()*1000)}"
    sessions_col.add(ids=[doc_id], metadatas=[metadata], documents=[content])

    client = _get_client()
    persist_fn = getattr(client, "persist", None)
    if callable(persist_fn):
        try:
            persist_fn()
        except Exception:
            logger.debug("client.persist() failed (non-fatal).")
    return metadata

def index_transcript_chunks(job_id: str, chunks: List[Dict[str, Any]]):
    _ensure_collections()
    ids, documents, metadatas = [], [], []
    for i, c in enumerate(chunks):
        doc_id = f"{job_id}::chunk::{i}::{int(time.time()*1000)}"
        ids.append(doc_id)
        documents.append(c["page_content"])
        md = c.get("metadata", {}).copy()
        md["job_id"] = job_id
        metadatas.append(_sanitize_metadata_values(md))
    transcripts_col.add(ids=ids, documents=documents, metadatas=metadatas)

    client = _get_client()
    persist_fn = getattr(client, "persist", None)
    if callable(persist_fn):
        try:
            persist_fn()
        except Exception:
            logger.debug("client.persist() failed after indexing (non-fatal).")
    logger.info("Indexed %d chunks for job %s", len(ids), job_id)
    return len(ids)

def query_transcript_similar(question: str, k: int = 4, job_filter: Optional[str] = None):
    _ensure_collections()
    where = {"job_id": job_filter} if job_filter else {}
    try:
        results = transcripts_col.query(query_texts=[question], n_results=k, where=where)
    except Exception as e:
        logger.exception("Chroma query failed: %s", e)
        return []
    if not results or not results.get("documents"):
        return []
    docs_list = results.get("documents", [[]])
    metas_list = results.get("metadatas", [[]])
    dists_list = results.get("distances", [[]]) if results.get("distances") else [None]*len(docs_list[0])

    docs = docs_list[0] if isinstance(docs_list[0], (list, tuple)) else docs_list
    metas = metas_list[0] if metas_list and isinstance(metas_list[0], (list, tuple)) else (metas_list or [])
    dists = dists_list[0] if dists_list and isinstance(dists_list[0], (list, tuple)) else (dists_list or [None]*len(docs))

    hits = []
    for doc, meta, score in zip(docs, metas, dists):
        hits.append({"text": doc, "metadata": meta, "score": score})
    return hits

def get_top_chunks_for_job(job_id: str, k: int = 4) -> List[Dict[str, Any]]:
    _ensure_collections()
    res = transcripts_col.get(where={"job_id": job_id}, include=["documents", "metadatas"])
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    ids = res.get("ids") or []

    out = []
    for i, (doc, md) in enumerate(zip(docs, metas)):
        out.append({
            "id": ids[i] if i < len(ids) else None,
            "text": doc,
            "metadata": md,
            "score": None
        })
    return out[:k]

_ID_INDEX_RE = re.compile(r"::chunk::(\d+)::")

def get_all_chunks_for_job(job_id: str) -> List[Dict[str, Any]]:
    _ensure_collections()
    res = transcripts_col.get(where={"job_id": job_id}, include=["documents", "metadatas"])
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    ids = res.get("ids") or []

    items = []
    for i, (doc, md) in enumerate(zip(docs, metas)):
        cid = ids[i] if i < len(ids) else ""
        m = _ID_INDEX_RE.search(cid or "")
        idx = int(m.group(1)) if m else 10**9 + i
        items.append({"id": cid, "idx": idx, "text": doc, "metadata": md})
    items.sort(key=lambda x: x["idx"])
    return items

def build_compilation_text(job_id: str, with_speakers: bool = True) -> str:
    items = get_all_chunks_for_job(job_id)
    lines = []
    for it in items:
        md = it.get("metadata") or {}
        speaker = md.get("speaker", "Speaker_0")
        start = md.get("start", 0)
        end = md.get("end", 0)
        if with_speakers:
            lines.append(f"{speaker} ({start}-{end}): {it.get('text','')}".strip())
        else:
            lines.append(it.get("text","").strip())
    return "\n".join([ln for ln in lines if ln])

def save_checkpoint(job_id: str, node_name: str, payload: Dict[str,Any], node_index: int = 0):
    _ensure_collections()
    ts = datetime.utcnow().isoformat()
    metadata = {"job_id": job_id, "node_name": node_name, "timestamp": ts, "node_index": node_index}
    content = json.dumps(payload, default=str, ensure_ascii=False)
    doc_id = f"{job_id}::checkpoint::{node_index}::{int(time.time()*1000)}"
    checkpoints_col.add(ids=[doc_id], metadatas=[_sanitize_metadata_values(metadata)], documents=[content])

    client = _get_client()
    persist_fn = getattr(client, "persist", None)
    if callable(persist_fn):
        try:
            persist_fn()
        except Exception:
            logger.debug("client.persist() failed in save_checkpoint (non-fatal).")
    return metadata
