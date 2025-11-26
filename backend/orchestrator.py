import os
import logging
from typing import Dict, Any, List, Optional, TypedDict

from utils import (
    run_ffmpeg_to_wav,
    get_audio_duration_sec,
    transcribe_with_openai,
    transcribe_with_openai_chunked,
    diarize_with_pyannote,
    diarize_with_pyannote_chunked,
    merge_transcript_and_diarization,
    create_documents_from_segments,
    send_email_via_sendgrid,
)
from chroma_store import index_transcript_chunks, save_checkpoint
from db import jobs as jobs_collection

from langgraph.graph import StateGraph, END

logger = logging.getLogger("orchestrator")
logger.setLevel(logging.INFO)

CHUNK_THRESHOLD_SEC = int(os.getenv("CHUNK_THRESHOLD_SEC", "900"))
CHUNK_SECONDS       = int(os.getenv("CHUNK_SECONDS", "600"))
CHUNK_OVERLAP_SEC   = float(os.getenv("CHUNK_OVERLAP_SEC", "1.0"))


class PipelineState(TypedDict, total=False):
    job_id: str
    raw_path: str
    wav_path: str
    transcription: Dict[str, Any]
    diarization: List[Dict[str, Any]]
    segments: List[Dict[str, Any]]
    docs_count: int
    summary: str
    send_email: bool
    to_email: Optional[str]
    auto_enrich: bool
    error: Optional[str]


def _ensure_nonempty_file(path: str, job_id: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{job_id}] Uploaded file not found: {path}")
    size = os.path.getsize(path)
    if size == 0:
        raise ValueError(f"[{job_id}] Uploaded file is empty (0 bytes): {path}")


def _summarize_with_llm(text: str) -> str:
    if not text.strip():
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        messages = [
            {"role": "system", "content": (
                "You are a sharp meeting summarizer. Return:\n"
                "- 5–10 bullet points\n"
                "- Action items (who/what/when)\n"
                "- One-line executive summary"
            )},
            {"role": "user", "content": f"Summarize this transcript:\n\n{text}"}
        ]
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.2, max_tokens=700
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("LLM summary failed: %s", e)
        return ""


def node_transcribe(state: PipelineState) -> PipelineState:
    job_id = state["job_id"]; raw_path = state["raw_path"]
    _ensure_nonempty_file(raw_path, job_id)

    wav_path = raw_path + ".wav"
    logger.info("[%s] ffmpeg -> %s", job_id, wav_path)
    run_ffmpeg_to_wav(raw_path, wav_path)

    dur = get_audio_duration_sec(wav_path)
    logger.info("[%s] wav duration: %.2fs", job_id, dur if dur > 0 else -1)

    if dur > 0 and dur > CHUNK_THRESHOLD_SEC:
        logger.info("[%s] Using CHUNKED transcription (chunk=%ss overlap=%.2fs)", job_id, CHUNK_SECONDS, CHUNK_OVERLAP_SEC)
        resp = transcribe_with_openai_chunked(wav_path, chunk_seconds=CHUNK_SECONDS, overlap_seconds=CHUNK_OVERLAP_SEC)
    else:
        logger.info("[%s] Using SINGLE-SHOT transcription", job_id)
        resp = transcribe_with_openai(wav_path)

    seg_count = len(resp.get("segments") or [])
    save_checkpoint(job_id, "transcribe", {"segments": seg_count, "duration": dur}, node_index=1)
    state.update({"wav_path": wav_path, "transcription": resp})
    return state


def node_diarize(state: PipelineState) -> PipelineState:
    job_id = state["job_id"]; wav_path = state["wav_path"]
    dur = get_audio_duration_sec(wav_path)
    if dur > 0 and dur > CHUNK_THRESHOLD_SEC:
        logger.info("[%s] Using CHUNKED diarization (chunk=%ss overlap=%.2fs)", job_id, CHUNK_SECONDS, CHUNK_OVERLAP_SEC)
        diar = diarize_with_pyannote_chunked(wav_path, chunk_seconds=CHUNK_SECONDS, overlap_seconds=CHUNK_OVERLAP_SEC)
    else:
        logger.info("[%s] Using SINGLE-SHOT diarization", job_id)
        diar = diarize_with_pyannote(wav_path)

    save_checkpoint(job_id, "diarize", {"count": len(diar)}, node_index=2)
    state["diarization"] = diar
    return state


def node_merge(state: PipelineState) -> PipelineState:
    job_id = state["job_id"]
    transcription = state.get("transcription") or {"text": "", "segments": []}
    diar = state.get("diarization") or []
    segments = merge_transcript_and_diarization(transcription, diar)
    save_checkpoint(job_id, "merge", {"segments": len(segments)}, node_index=3)
    state["segments"] = segments
    return state


def node_chunk_index(state: PipelineState) -> PipelineState:
    job_id = state["job_id"]; segments = state.get("segments") or []
    docs = create_documents_from_segments(segments)
    num = index_transcript_chunks(job_id, docs)
    save_checkpoint(job_id, "chunk_index", {"indexed": num}, node_index=4)
    state["docs_count"] = num
    return state


def node_summarize(state: PipelineState) -> PipelineState:
    job_id = state["job_id"]; segments = state.get("segments") or []
    if not segments:
        state["summary"] = ""
        return state
    lines = []
    for s in segments:
        sp = s.get("speaker", "Speaker")
        st = s.get("start", 0.0); en = s.get("end", st)
        tx = (s.get("text") or "").strip()
        lines.append(f"{sp} ({float(st):.2f}-{float(en):.2f}): {tx}")
    compiled = "\n".join(lines)
    summary = _summarize_with_llm(compiled) or ((segments[0]["text"][:200] + "...") if segments else "")
    save_checkpoint(job_id, "summary", {"len": len(summary)}, node_index=5)
    state["summary"] = summary
    return state


def node_email_if_requested(state: PipelineState) -> PipelineState:
    if not state.get("send_email"):
        return state
    job_id = state["job_id"]
    to_email = state.get("to_email")
    summary = state.get("summary", "") or ""
    sample = ""
    segs = state.get("segments") or []
    if segs:
        sample_lines = []
        for s in segs[:10]:
            sp = s.get("speaker", "Speaker")
            st = s.get("start", 0.0); en = s.get("end", st)
            tx = (s.get("text") or "").strip()
            sample_lines.append(f"{sp} ({float(st):.2f}-{float(en):.2f}): {tx}")
        sample = "\n".join(sample_lines)

    html = f"<h2>Summary</h2><pre>{summary or '(empty)'}</pre>"
    if sample:
        html += f"<h2>Transcript (sample)</h2><pre>{sample}</pre>"

    try:
        if to_email:
            send_email_via_sendgrid(to_email, f"Transcript & Summary — {job_id}", html)
            save_checkpoint(job_id, "email", {"to": to_email}, node_index=6)
    except Exception as e:
        logger.warning("Email send failed: %s", e)
    return state


def _build_graph():
    g = StateGraph(PipelineState)
    g.add_node("transcribe", node_transcribe)
    g.add_node("diarize", node_diarize)
    g.add_node("merge", node_merge)
    g.add_node("chunk_index", node_chunk_index)
    g.add_node("summarize", node_summarize)
    g.add_node("email_if_requested", node_email_if_requested)

    g.set_entry_point("transcribe")
    g.add_edge("transcribe", "diarize")
    g.add_edge("diarize", "merge")
    g.add_edge("merge", "chunk_index")
    g.add_edge("chunk_index", "summarize")
    g.add_edge("summarize", "email_if_requested")
    g.add_edge("email_if_requested", END)
    return g.compile()


_graph = _build_graph()


def build_and_run_graph(
    raw_path: str,
    job_id: str,
    send_email: bool = False,
    to_email: Optional[str] = None,
    auto_enrich: bool = False,
) -> Dict[str, Any]:
    logger.info("LangGraph pipeline starting for %s", job_id)
    try:
        init: PipelineState = {
            "job_id": job_id,
            "raw_path": raw_path,
            "send_email": send_email,
            "to_email": to_email,
            "auto_enrich": auto_enrich,
        }
        final_state: PipelineState = _graph.invoke(init)

        summary = final_state.get("summary") or ""
        segments = final_state.get("segments") or []

        jobs_collection.update_one(
            {"job_id": job_id},
            {"$set": {"status": "done", "summary": summary, "segments": segments}},
            upsert=True
        )
        logger.info("[%s] job completed (LangGraph).", job_id)
        return {"summary": summary, "segments": segments}
    except Exception as e:
        logger.exception("LangGraph run failed for %s: %s", job_id, e)
        jobs_collection.update_one(
            {"job_id": job_id},
            {"$set": {"status": "failed", "error": str(e)}},
            upsert=True
        )
        return {"error": str(e)}
