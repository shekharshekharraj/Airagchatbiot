import os
import re
import logging
from typing import Dict, Any, List, Optional, Tuple

from chroma_store import (
    query_transcript_similar,
    get_top_chunks_for_job,
    build_compilation_text,
)
from utils import send_email_via_sendgrid
from db import jobs as jobs_collection

logger = logging.getLogger("agent_tools")
logger.setLevel(logging.INFO)

# Tavily is optional
try:
    from tavily import TavilyClient  # type: ignore
except Exception:  # pragma: no cover
    TavilyClient = None

# ----------------------------- RAG ---------------------------------
async def rag_search_tool(
    question: str = "",
    job_id: Optional[str] = None,
    k: int = 4,
    return_all: bool = False,
) -> Dict[str, Any]:
    """
    If return_all=True -> return a compiled full transcript (best-effort).
    Otherwise -> return top-k RAG hits (Chroma first, then Mongo fallback).
    """
    if not job_id:
        return {"hits": [], "compilation": "", "error": "job_id missing"}

    if return_all:
        # 1) Try prebuilt compiled text from vector store ordering
        compilation = build_compilation_text(job_id, with_speakers=True)
        if compilation:
            return {"hits": [], "compilation": compilation}

        # 2) Try MongoDB segments -> compile
        job = await jobs_collection.find_one({"job_id": job_id})
        if job:
            if job.get("compiled_transcript"):
                return {"hits": [], "compilation": job["compiled_transcript"]}
            segs = job.get("segments") or []
            if segs:
                lines: List[str] = []
                for seg_idx, s in enumerate(segs):
                    sp = s.get("speaker", "Speaker_0")
                    st = s.get("start", 0) or 0.0
                    en = s.get("end", 0) or 0.0
                    tx = (s.get("text") or "").strip()
                    if not tx:
                        continue
                    try:
                        lines.append(f"{sp} ({float(st):.2f}-{float(en):.2f}): {tx}")
                    except Exception:
                        lines.append(f"{sp} ({st}-{en}): {tx}")
                return {"hits": [], "compilation": "\n".join(lines)}
        return {"hits": [], "compilation": "", "error": "no_chunks_indexed"}

    # RAG: vector query first
    hits = query_transcript_similar(question or "", k=k, job_filter=job_id) or []
    if not hits:
        # Fallback: just return top chunks we indexed for this job
        top = get_top_chunks_for_job(job_id, k=k) or []
        if not top:
            # Last-resort from Mongo segments
            job = await jobs_collection.find_one({"job_id": job_id})
            if job and job.get("segments"):
                out: List[Dict[str, Any]] = []
                for seg_idx, s in enumerate(job["segments"][:k]):
                    out.append({
                        "text": (s.get("text") or "")[:500],
                        "metadata": {
                            "speaker": s.get("speaker", "Speaker"),
                            "start": s.get("start"),
                            "end": s.get("end"),
                            "segment_id": s.get("segment_id", seg_idx),
                            "job_id": job_id,
                        },
                        "score": None,
                    })
                return {"hits": out}
            return {"hits": [], "error": "no_chunks_indexed"}
        hits = top

    trimmed: List[Dict[str, Any]] = []
    for h in hits:
        md = h.get("metadata") or {}
        trimmed.append({
            "text": (h.get("text") or "")[:500],
            "metadata": {
                "speaker": md.get("speaker"),
                "start": md.get("start"),
                "end": md.get("end"),
                "segment_id": md.get("segment_id"),
                "job_id": md.get("job_id") or job_id,
            },
            "score": h.get("score"),
        })
    return {"hits": trimmed}

# -------------------------- Summary fetch --------------------------
async def get_summary_tool(job_id: str) -> Dict[str, Any]:
    if not job_id:
        return {"summary": "", "compilation": "", "error": "job_id missing"}
    job = await jobs_collection.find_one({"job_id": job_id})
    if not job or job.get("status") not in ("indexed", "done"):
        return {"summary": "", "compilation": "", "error": "summary not ready"}
    summary = job.get("summary", "") or ""
    compilation = job.get("compiled_transcript", "")
    return {"summary": summary, "compilation": compilation}

# --------------------------- Email sending -------------------------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

async def send_email_tool(job_id: str, to_email: str) -> Dict[str, Any]:
    """
    Sends transcript/summary via SendGrid using utils.send_email_via_sendgrid.
    Returns: {"status": "sent"} on success, else {"status":"error","error": "..."}.
    """
    if not job_id:
        return {"status": "error", "error": "job_id missing"}
    if not to_email or not EMAIL_RE.match(to_email):
        return {"status": "error", "error": "invalid to_email"}

    job = await jobs_collection.find_one({"job_id": job_id})
    if not job or job.get("status") not in ("indexed", "done"):
        return {"status": "error", "error": "job not ready"}

    # Prefer stored summary; else build a short one from chunks/compiled text
    summary = (job.get("summary") or "").strip()
    if not summary:
        comp = build_compilation_text(job_id, with_speakers=True)
        if comp:
            summary = comp[:2000]
        else:
            hits = get_top_chunks_for_job(job_id, k=5) or []
            if hits:
                summary = "\n\n".join((h.get("text") or "")[:400] for h in hits if h.get("text"))

    subject = f"Transcript Summary - {job_id}"
    html = f"<h2>Transcript Summary</h2><pre>{(summary or '(no summary available)')}</pre>"

    try:
        resp = send_email_via_sendgrid(to_email, subject, html)
        try:
            await jobs_collection.update_one(
                {"job_id": job_id},
                {"$set": {"email_sent_to": to_email}},
            )
        except Exception:
            logger.debug("Could not persist email_sent_to for job %s", job_id)

        status_code = getattr(resp, "status_code", None)
        if isinstance(status_code, int) and 200 <= status_code < 300:
            return {"status": "sent"}
        if status_code is None and resp:
            return {"status": "sent"}

        return {"status": "error", "error": f"sendgrid_response={status_code or str(resp)}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ---------------------------- Tavily search ------------------------
def _normalize_items(raw: Dict[str, Any], cap: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in (raw.get("results") or [])[:cap]:
        title = r.get("title") or "Untitled"
        url_ = r.get("url") or ""
        snippet = (r.get("content") or r.get("snippet") or r.get("raw_content") or "").strip()
        score = r.get("score")
        if not snippet:
            snippet = title
        else:
            words = snippet.split()
            if len(words) > 80:
                snippet = " ".join(words[:80]) + " ..."
        out.append({"title": title, "url": url_, "snippet": snippet, "score": score})
    return out

async def web_search_tavily_tool(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Tavily search that returns snippets + links. Requires:
      - env TAVILY_API_KEY (in project root .env)
      - pip install tavily-python
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"answer": None, "results": [], "error": "TAVILY_API_KEY missing"}
    if TavilyClient is None:
        return {"answer": None, "results": [], "error": "tavily client not installed"}

    tv = TavilyClient(api_key=api_key)

    try:
        res: Dict[str, Any] = tv.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            timeout=30,  # seconds
        )
        items = _normalize_items(res, max_results)

        # Light fan-out if too few results
        if len(items) <= 1:
            seeds: Tuple[str, ...] = ("latest", "today", "breaking")
            seen = {it["url"] for it in items if it.get("url")}
            for s in seeds:
                sub: Dict[str, Any] = tv.search(
                    query=f"{s} {query}",
                    max_results=max_results,
                    search_depth="advanced",
                    include_answer=False,
                    include_raw_content=False,
                    timeout=30,
                )
                for it in _normalize_items(sub, max_results):
                    url = it.get("url")
                    if url and url not in seen:
                        seen.add(url)
                        items.append(it)
                if len(items) >= max(8, max_results * 2):
                    break

        return {"answer": res.get("answer"), "results": items}
    except Exception as e:  # pragma: no cover
        logger.exception("Tavily request failed: %s", e)
        return {"answer": None, "results": [], "error": str(e)}
