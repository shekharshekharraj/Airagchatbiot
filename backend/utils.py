import os
import re
import subprocess
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import aiofiles

logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---- Upload helper ----
async def save_upload_file_tmp(upload_file, destination: str) -> None:
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    async with aiofiles.open(destination, "wb") as out:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            await out.write(chunk)
    await upload_file.close()
    size = os.path.getsize(destination)
    logger.info("Saved upload to %s (%d bytes)", destination, size)
    if size == 0:
        raise ValueError("Upload appears empty (0 bytes).")

# ---- FFmpeg helpers ----
def run_ffmpeg_to_wav(in_path: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", "-vn", out_path]
    logger.info("Running ffmpeg: %s", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        logger.error("FFmpeg failed.\nSTDOUT:\n%s\nSTDERR:\n%s", proc.stdout, proc.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)

def get_audio_duration_sec(path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        logger.warning("ffprobe failed, assuming unknown duration: %s", proc.stderr)
        return -1.0
    try:
        info = json.loads(proc.stdout)
        dur = float(info.get("format", {}).get("duration", "0") or 0)
        if dur <= 0:
            for st in info.get("streams", []) or []:
                if "duration" in st:
                    dur = max(dur, float(st["duration"]))
        return float(dur) if dur > 0 else -1.0
    except Exception:
        return -1.0

def split_wav_into_chunks(
    wav_path: str,
    chunk_seconds: int = 600,
    overlap_seconds: float = 1.0,
    workdir: Optional[str] = None,
) -> List[Tuple[str, float]]:
    from math import isfinite
    duration = get_audio_duration_sec(wav_path)
    if duration < 0 or not isfinite(duration) or duration <= chunk_seconds:
        return [(wav_path, 0.0)]

    base_dir = workdir or os.path.join(os.path.dirname(wav_path) or ".", ".chunks")
    os.makedirs(base_dir, exist_ok=True)

    chunks: List[Tuple[str, float]] = []
    start = 0.0
    idx = 0
    logger.info("Splitting wav into %ds chunks (overlap %.2fs). Duration=%.2fs", chunk_seconds, overlap_seconds, duration)

    while start < duration:
        end = min(start + chunk_seconds, duration)
        out_chunk = os.path.join(base_dir, f"{Path(wav_path).stem}_part_{idx:04d}.wav")
        actual_start = max(0.0, start - (overlap_seconds if idx > 0 else 0.0))
        length = end - actual_start

        cmd = [
            "ffmpeg", "-y", "-i", wav_path,
            "-ac", "1", "-ar", "16000",
            "-ss", f"{actual_start:.3f}",
            "-t", f"{length:.3f}",
            out_chunk
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            logger.error("Chunk ffmpeg failed idx=%d.\nSTDERR:\n%s", idx, proc.stderr)
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)

        chunks.append((out_chunk, actual_start))
        idx += 1
        start += chunk_seconds

    logger.info("Created %d chunks", len(chunks))
    return chunks

# ---- OpenAI Transcription (whisper-1) ----
try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None

def _openai_whisper_transcribe_single(audio_path: str) -> Dict[str, Any]:
    if OpenAIClient is None:
        raise RuntimeError("OpenAI SDK not installed. Install `openai>=1.0`.")
    client = OpenAIClient(api_key=OPENAI_API_KEY)
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )

    text = getattr(resp, "text", "") or ""
    segments = None
    if hasattr(resp, "segments") and resp.segments:
        segments = resp.segments
    else:
        try:
            segments = resp.model_dump().get("segments")
        except Exception:
            segments = None

    norm: List[Dict[str, Any]] = []
    if segments:
        for s in segments:
            if hasattr(s, "start") or hasattr(s, "end") or hasattr(s, "text"):
                start = float(getattr(s, "start", 0.0) or 0.0)
                end = float(getattr(s, "end", start) or start)
                text_s = (getattr(s, "text", "") or "").strip()
            else:
                start = float(s.get("start", 0.0))
                end = float(s.get("end", start))
                text_s = (s.get("text", "") or "").strip()
            if end <= start:
                end = start + 0.01
            norm.append({"start": start, "end": end, "text": text_s})
    return {"text": text, "segments": norm}

def transcribe_with_openai(audio_path: str) -> Dict[str, Any]:
    return _openai_whisper_transcribe_single(audio_path)

def transcribe_with_openai_chunked(
    wav_path: str,
    chunk_seconds: int = 600,
    overlap_seconds: float = 1.0,
    cleanup: bool = True
) -> Dict[str, Any]:
    chunks = split_wav_into_chunks(wav_path, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds)
    if len(chunks) == 1:
        return _openai_whisper_transcribe_single(wav_path)

    full_text_parts: List[str] = []
    all_segments: List[Dict[str, Any]] = []

    try:
        for idx, (chunk_path, offset) in enumerate(chunks):
            logger.info("Transcribing chunk %d/%d (offset=%.2fs): %s", idx + 1, len(chunks), offset, chunk_path)
            res = _openai_whisper_transcribe_single(chunk_path)
            tx = (res.get("text") or "").strip()
            if tx:
                full_text_parts.append(tx)
            for s in (res.get("segments") or []):
                all_segments.append({
                    "start": s["start"] + offset,
                    "end": s["end"] + offset,
                    "text": s["text"],
                })
    finally:
        if cleanup and len(chunks) > 1:
            for p, _ in chunks:
                if os.path.abspath(p) != os.path.abspath(wav_path) and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            default_dir = os.path.join(os.path.dirname(wav_path) or ".", ".chunks")
            try:
                if os.path.isdir(default_dir) and not os.listdir(default_dir):
                    os.rmdir(default_dir)
            except Exception:
                pass

    return {"text": "\n".join(full_text_parts).strip(), "segments": all_segments}

# ---- Diarization (optional) ----
def _pyannote_load_pipeline():
    from pyannote.audio import Pipeline
    hf_token = os.getenv("HF_AUTH_TOKEN")  # optional
    try:
        return Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
    except TypeError:
        return Pipeline.from_pretrained("pyannote/speaker-diarization", token=hf_token)

def diarize_with_pyannote(audio_path: str) -> List[Dict]:
    try:
        pipeline = _pyannote_load_pipeline()
        diarization = pipeline(audio_path)
        segments: List[Dict[str, Any]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": float(getattr(turn, "start", 0.0)),
                "end": float(getattr(turn, "end", 0.0)),
                "speaker": str(speaker),
            })
        return segments
    except Exception as e:
        logger.error("Diarization failed: %s", e)
        return []

def diarize_with_pyannote_chunked(
    wav_path: str,
    chunk_seconds: int = 600,
    overlap_seconds: float = 1.0,
    cleanup: bool = True
) -> List[Dict]:
    chunks = split_wav_into_chunks(wav_path, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds)
    if len(chunks) == 1:
        return diarize_with_pyannote(wav_path)

    try:
        pipeline = _pyannote_load_pipeline()
    except Exception as e:
        logger.error("Failed to load pyannote pipeline: %s", e)
        return []

    all_segments: List[Dict[str, Any]] = []
    try:
        for idx, (chunk_path, offset) in enumerate(chunks):
            try:
                logger.info("Diarizing chunk %d/%d (offset=%.2fs): %s", idx + 1, len(chunks), offset, chunk_path)
                diarization = pipeline(chunk_path)
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    st = float(getattr(turn, "start", 0.0)) + offset
                    en = float(getattr(turn, "end", 0.0)) + offset
                    all_segments.append({"start": st, "end": en, "speaker": str(speaker)})
            except Exception as e:
                logger.warning("Chunk diarization failed idx=%d: %s", idx, e)
                continue
    finally:
        if cleanup and len(chunks) > 1:
            for p, _ in chunks:
                if os.path.abspath(p) != os.path.abspath(wav_path) and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            default_dir = os.path.join(os.path.dirname(wav_path) or ".", ".chunks")
            try:
                if os.path.isdir(default_dir) and not os.listdir(default_dir):
                    os.rmdir(default_dir)
            except Exception:
                pass

    all_segments.sort(key=lambda x: (x["speaker"], x["start"], x["end"]))
    merged: List[Dict[str, Any]] = []
    for seg in all_segments:
        if merged and merged[-1]["speaker"] == seg["speaker"] and seg["start"] <= merged[-1]["end"] + 0.5:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg)
    return merged

# ---- Merge + Chunker + Email ----
def merge_transcript_and_diarization(transcription: Dict[str, Any], diarization_segments: List[Dict]) -> List[Dict]:
    out_segments = []
    t_segs = transcription.get("segments", []) if isinstance(transcription, dict) else []
    full_text = transcription.get("text", "") if isinstance(transcription, dict) else ""
    if not t_segs and full_text:
        return [{"start": 0.0, "end": 0.01, "speaker": "Speaker_0", "text": full_text.strip()}]

    diar_sorted = sorted(diarization_segments or [], key=lambda d: d.get("start", 0.0))

    def find_speaker(mid: float) -> str:
        for d in diar_sorted:
            st = d.get("start", 0.0); en = d.get("end", 0.0)
            if st <= mid <= en:
                return d.get("speaker", "Speaker_0")
        return "Speaker_0"

    for t in t_segs:
        start = float(t.get("start", 0))
        end = float(t.get("end", start))
        if end <= start:
            end = start + 0.01
        text = (t.get("text", "") or "").strip()
        mid = (start + end) / 2.0
        out_segments.append({"start": start, "end": end, "speaker": find_speaker(mid), "text": text})
    return out_segments

def create_documents_from_segments(segments, min_chars: int = 200, overlap: int = 40):
    docs = []
    for seg_idx, s in enumerate(segments):
        text = (s.get("text") or "").strip()
        if not text:
            continue
        if len(text) <= min_chars:
            docs.append({
                "page_content": text,
                "metadata": {"speaker": s.get("speaker"), "start": s.get("start"), "end": s.get("end"), "segment_id": seg_idx}
            })
            continue
        start_char = 0
        while start_char < len(text):
            end_char = start_char + min_chars
            chunk_text = text[start_char:end_char]
            docs.append({
                "page_content": chunk_text,
                "metadata": {"speaker": s.get("speaker"), "start": s.get("start"), "end": s.get("end"), "segment_id": seg_idx}
            })
            start_char = max(0, end_char - overlap)
    return docs

def send_email_via_sendgrid(to_email: str, subject: str, content: str):
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
    key = os.getenv("SENDGRID_API_KEY")
    if not key:
        raise RuntimeError("SENDGRID_API_KEY not set")
    from_email = os.getenv("FROM_EMAIL", "no-reply@example.com")
    message = Mail(from_email=from_email, to_emails=to_email, subject=subject, html_content=content)
    sg = SendGridAPIClient(key)
    return sg.send(message)
