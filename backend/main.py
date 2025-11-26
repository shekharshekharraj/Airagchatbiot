import os
import uuid
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bson import ObjectId
from dotenv import load_dotenv

# ----- Load .env from project root (one level above backend/) -----
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
root_env = ROOT / ".env"
if root_env.exists():
    load_dotenv(dotenv_path=root_env, override=False)
# also load local backend/.env if present (non-overriding)
local_env = HERE.parent / ".env"
if local_env.exists():
    load_dotenv(dotenv_path=local_env, override=False)

from utils import save_upload_file_tmp
from orchestrator import build_and_run_graph
from db import jobs as jobs_collection, get_sync_collection
from graph import build_chat_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Audio-RAG Chatbot Backend (LangGraph)")
executor = ThreadPoolExecutor(max_workers=2)

# Build the LangGraph chat graph once at startup
_graph_chat = build_chat_graph()

# Log presence of critical env vars (without printing secrets)
logger.info("Env checks: OPENAI_API_KEY=%s, TAVILY_API_KEY=%s, SENDGRID_API_KEY=%s, MONGODB_URI=%s",
            "set" if os.getenv("OPENAI_API_KEY") else "missing",
            "set" if os.getenv("TAVILY_API_KEY") else "missing",
            "set" if os.getenv("SENDGRID_API_KEY") else "missing",
            "set" if os.getenv("MONGODB_URI") else "missing")

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ------------------ Sanitizers ------------------
def sanitize_value(v):
    if isinstance(v, ObjectId):
        return str(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", errors="ignore")
        except Exception:
            return str(v)
    return v

def sanitize_mongo_doc(doc: dict) -> dict:
    if not isinstance(doc, dict):
        return doc
    out = {}
    for k, v in doc.items():
        if k == "_id":
            out[k] = sanitize_value(v); continue
        if isinstance(v, dict):
            out[k] = sanitize_mongo_doc(v)
        elif isinstance(v, list):
            out[k] = [sanitize_mongo_doc(i) if isinstance(i, dict) else sanitize_value(i) for i in v]
        else:
            out[k] = sanitize_value(v)
    return out

# ------------------ Upload Helpers ------------------
MIME_TO_EXT = {
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/x-m4a": ".m4a",
    "audio/mp4": ".m4a",
    "audio/x-ms-wma": ".wma",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
}

def safe_uploaded_filename(upload_file: UploadFile, uid: str) -> str:
    orig = (upload_file.filename or "").strip()
    orig_name = Path(orig).name
    base = orig_name if orig_name else "upload"
    ext = Path(base).suffix
    if not ext:
        guess = MIME_TO_EXT.get(upload_file.content_type or "", "") or ".bin"
        base = base + guess
    import re as _re
    base = _re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return f"{uid}_{base}"

# ------------------ Jobs ------------------
@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = await jobs_collection.find_one({"job_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(sanitize_mongo_doc(job))

# ------------------ Upload Audio ------------------
@app.post("/upload_audio")
async def upload_audio(request: Request, file: UploadFile = File(...)):
    logger.info("Incoming upload: filename=%r content_type=%r", file.filename, file.content_type)
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file received. In Postman: Body=form-data, key 'file' (type: File), select a real audio file.",
        )

    uid = str(uuid.uuid4())
    final_name = safe_uploaded_filename(file, uid)
    raw_path = os.path.join(UPLOAD_DIR, final_name)
    job_id = uid

    await save_upload_file_tmp(file, raw_path)
    await jobs_collection.insert_one({"job_id": job_id, "status": "queued", "raw_path": raw_path})

    def _bg_wrapper():
        jobs_sync = get_sync_collection("jobs")
        try:
            res = build_and_run_graph(raw_path, job_id, False, None, False)
            summary = res.get("summary", "") if isinstance(res, dict) else ""
            segments = res.get("segments", []) if isinstance(res, dict) else []
            jobs_sync.update_one(
                {"job_id": job_id},
                {"$set": {"status": "done", "summary": summary, "segments": segments,
                          "updated_at": datetime.utcnow()}}, upsert=True)
        except Exception as e:
            jobs_sync.update_one(
                {"job_id": job_id},
                {"$set": {"status": "failed", "error": str(e),
                          "updated_at": datetime.utcnow()}}, upsert=True)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _bg_wrapper)
    return {"job_id": job_id}

# ------------------ Chat DTOs ------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    job_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []

# ------------------ Chat (non-streaming) ------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or "default"
    user_msg = (req.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message")

    state_in = {"session_id": session_id, "job_id": req.job_id, "user_msg": user_msg}
    out = await _graph_chat.ainvoke(state_in)
    return ChatResponse(answer=out.get("answer", "") or "", citations=out.get("citations", []))

# ------------------ Chat (streaming via LangGraph) ------------------
@app.post("/chat_stream")
async def chat_stream(req: ChatRequest):
    session_id = req.session_id or "default"
    user_msg = (req.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message")

    state_in = {"session_id": session_id, "job_id": req.job_id, "user_msg": user_msg}

    async def _gen():
        final_chunks: List[str] = []
        final_answer: Optional[str] = None

        async for ev in _graph_chat.astream_events(state_in, version="v1"):
            etype = ev.get("event")
            data = ev.get("data") or {}

            # Stream tokens from LLM nodes
            if etype == "on_chat_model_stream":
                chunk_obj = data.get("chunk")
                content = getattr(chunk_obj, "content", None)
                if isinstance(content, list):
                    text = "".join([c.get("text", "") for c in content if isinstance(c, dict)])
                else:
                    text = content or ""
                if text:
                    final_chunks.append(text)
                    yield text
                continue

            if etype == "on_llm_stream":
                text = data.get("chunk") or ""
                if text:
                    final_chunks.append(text)
                    yield text
                continue

            # Capture final output from non-LLM nodes
            if etype in ("on_chain_end", "on_graph_end"):
                out = data.get("output") or {}
                if isinstance(out, dict):
                    ans = out.get("answer") or ""
                    if ans:
                        final_answer = ans

        # If nothing was streamed but we got a final answer from a tool, yield it
        if not final_chunks and final_answer:
            yield final_answer

        # Absolute fallback
        if not final_chunks and not final_answer:
            out = await _graph_chat.ainvoke(state_in)
            answer = out.get("answer", "") or ""
            if answer:
                yield answer

    return StreamingResponse(_gen(), media_type="text/plain; charset=utf-8")

# ------------------ Tiny debug endpoint for Tavily ------------------
@app.get("/debug/tavily")
async def debug_tavily():
    from agent_tools import web_search_tavily_tool
    res = await web_search_tavily_tool("latest AI news", 3)
    return res
