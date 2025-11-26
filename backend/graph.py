import re
import logging
from typing import TypedDict, Optional, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from db import messages as messages_collection, jobs as jobs_collection
from agent_tools import rag_search_tool, web_search_tavily_tool, send_email_tool
from chroma_store import persist_message

logger = logging.getLogger("graph")
logger.setLevel(logging.INFO)

# ---------------- Helpers (routing regex) ----------------
def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    for bad, good in {
        "curent": "current",
        "cureent": "current",
        "currnet": "current",
        "todays": "today's",
    }.items():
        t = t.replace(bad, good)
    return t

_FULL_TX_PATTERNS = [
    r"\b(full|complete)\s+transcript\b",
    r"\bgive me (the )?transcript\b",
    r"\bshow (me )?(the )?entire transcript\b",
    r"\ball text\b",
]
_SUMMARY_PATTERNS = [
    r"\bmeeting summary\b",
    r"\bsummar(?:y|ise|ize)\b",
    r"\baction items\b",
    r"\bminutes\b",
    r"\bkey points\b",
]
_AUDIO_PATTERNS = [
    r"\baudio\b",
    r"\btranscript\b",
    r"\bmeeting\b",
    r"\bspeakers?\b",
    r"\bwhat did (the )?speaker\b",
    r"\bwhat did .* say\b",
]
_EMAIL_INTENT_PATTERNS = [
    r"\bsend (an )?email\b",
    r"\bsend mail\b",
    r"\bemail (it|this|the (summary|transcript))\b",
    r"\bsend (the )?(summary|transcript)\b",
]
EMAIL_RE = re.compile(r"(?P<email>[^@\s]+@[^@\s]+\.[^@\s]+)", re.I)
_REALTIME_PATTERNS = [
    r"\b(current|today|now|latest|live|breaking|real[- ]?time)\b",
    r"\b(what('?s| is)? the (current )?time|time\s+now|time\s+in\s+[a-zA-Z ,.-]+|current\s+date|today('?s)?\s+date)\b",
    r"\b(utc|gmt|ist|pst|cst|est)\b.*\btime\b",
    r"\b(weather|forecast|temperature|rain|humidity)\b",
    r"\b(current|today|now)\s+(weather|forecast)\b",
    r"\b(stock|share|price|quote|ticker|market|index|nifty|sensex|nasdaq|dow|banknifty)\b",
    r"\b(usd|inr|eur|gbp|jpy|aud|cad|chf|cny)\b.*\b(rate|fx|exchange|price)\b",
    r"\b(bitcoin|btc|eth|crypto)\b.*\b(price|rate|today|now)\b",
    r"\b(current|live)\s+(price|rate|quote)\b",
    r"\b(score|scores|result|results|fixture|fixtures|schedule|match|live\s+score)\b",
    r"\b(release date|release|\bupdated?\b|\bupdate\b|\bversion\b)\b",
    r"\b(flight|train|bus)\b.*\b(status|schedule|arrival|departure)\b",
    r"\bnews\b",
]

def _wants_full_transcript(t: str) -> bool: return any(re.search(p, _normalize(t)) for p in _FULL_TX_PATTERNS)
def _wants_summary(t: str) -> bool:        return any(re.search(p, _normalize(t)) for p in _SUMMARY_PATTERNS)
def _is_audio_related(t: str) -> bool:     return any(re.search(p, _normalize(t)) for p in _AUDIO_PATTERNS)
def _wants_email(t: str) -> bool:          return any(re.search(p, _normalize(t)) for p in _EMAIL_INTENT_PATTERNS)
def _is_realtime_query(t: str) -> bool:    return any(re.search(p, _normalize(t)) for p in _REALTIME_PATTERNS)

_AWAIT_EMAIL_MARKER = "[Awaiting email address]"

# ---------------- Chat state ----------------
class ChatState(TypedDict, total=False):
    session_id: str
    job_id: Optional[str]
    user_msg: str
    answer: str
    citations: List[Dict[str, Any]]
    route: str
    rag_hits: List[Dict[str, Any]]
    web_items: List[Dict[str, Any]]

# ---------------- Persistence helpers ----------------
async def _persist_user_message(session_id: str, content: str) -> None:
    await messages_collection.insert_one({"session_id": session_id, "role": "user", "content": content})

async def _persist_bot_message(session_id: str, content: str) -> None:
    await messages_collection.insert_one({"session_id": session_id, "role": "bot", "content": content})
    persist_message(session_id, "assistant", content)

async def _compile_transcript_from_db(job_id: str) -> str:
    job = await jobs_collection.find_one({"job_id": job_id}, projection={"segments": 1})
    segs = (job or {}).get("segments", []) or []
    if not segs:
        return ""
    def _start(s): return s.get("start") if isinstance(s.get("start"), (int, float)) else s.get("start_time", 0.0)
    segs = sorted(segs, key=_start)
    lines = []
    for s in segs:
        sp = s.get("speaker") or s.get("speaker_label") or "Speaker"
        st = s.get("start", s.get("start_time", 0.0))
        en = s.get("end", s.get("end_time", st))
        tx = s.get("text") or s.get("content") or ""
        try:
            lines.append(f"{sp} ({float(st):.2f}-{float(en):.2f}): {tx}")
        except Exception:
            lines.append(f"{sp}: {tx}")
    return "\n".join(lines)

async def _session_history_messages(session_id: str, limit: int = 12) -> List:
    cursor = messages_collection.find({"session_id": session_id}).sort([("_id", -1)]).limit(limit)
    msgs = [m async for m in cursor]
    msgs.reverse()
    out = []
    for m in msgs:
        role = "assistant" if m.get("role") == "bot" else "user"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            out.append(SystemMessage(content="(assistant) " + content))
        else:
            out.append(HumanMessage(content=content))
    return out

# ---------------- LLM Models ----------------
_llm_default = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)
_llm_research = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, streaming=True)
_llm_summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)

# ---------------- Router Node ----------------
async def node_router(state: ChatState) -> ChatState:
    await _persist_user_message(state["session_id"], state["user_msg"])
    msg = (state.get("user_msg") or "").strip()
    session_id = state["session_id"]

    last_bot = await messages_collection.find_one({"session_id": session_id, "role": "bot"}, sort=[("_id", -1)])
    last_bot_content = (last_bot or {}).get("content", "")
    email_match = EMAIL_RE.search(msg)

    if _wants_full_transcript(msg):
        state["route"] = "full_tx"
    elif _wants_summary(msg):
        state["route"] = "summary"
    elif _wants_email(msg) and not email_match:
        state["route"] = "email_request"
    elif email_match and (_wants_email(msg) or _AWAIT_EMAIL_MARKER in (last_bot_content or "")):
        state["route"] = "email_send"
    elif _is_audio_related(msg):
        state["route"] = "audio_rag"
    elif _is_realtime_query(msg):
        state["route"] = "web"
    else:
        state["route"] = "chat"
    return state

# ---------------- Branch Nodes ----------------
async def node_full_tx(state: ChatState) -> ChatState:
    if not state.get("job_id"):
        state["answer"] = "job_id required."
        return state
    compiled = await _compile_transcript_from_db(state["job_id"])
    if not compiled:
        tool_result = await rag_search_tool("", state["job_id"], 1, True)
        compiled = tool_result.get("compilation") or "Transcript not ready yet."
    state["answer"] = compiled
    return state

async def node_summary(state: ChatState) -> ChatState:
    if not state.get("job_id"):
        state["answer"] = "job_id required."
        return state
    tool_result = await rag_search_tool("", state["job_id"], 1, True)
    compiled = tool_result.get("compilation") or ""
    if not compiled:
        state["answer"] = "Transcript not ready yet."
        return state

    sys = SystemMessage(content="You are a sharp meeting summarizer. Return concise bullets, action items, and a one-line summary.")
    user = HumanMessage(content=f"Summarize this transcript:\n\n{compiled}")
    result = await _llm_summarizer.ainvoke([sys, user])
    state["answer"] = result.content or ""
    return state

async def node_email_request(state: ChatState) -> ChatState:
    state["answer"] = "Sure — to whom should I send the email? " + _AWAIT_EMAIL_MARKER
    return state

async def node_email_send(state: ChatState) -> ChatState:
    if not state.get("job_id"):
        state["answer"] = "job_id required."
        return state
    match = EMAIL_RE.search(state["user_msg"])
    if not match:
        state["answer"] = "Please provide a valid email address."
        return state
    res = await send_email_tool(state["job_id"], match.group("email"))
    if res.get("status") == "sent":
        state["answer"] = f"Done. Sent transcript + summary to **{match.group('email')}**."
    else:
        state["answer"] = f"Email failed: {res.get('error','unknown error')}."
    return state

async def node_audio_rag(state: ChatState) -> ChatState:
    if not state.get("job_id"):
        state["answer"] = "job_id required."
        return state
    res = await rag_search_tool(state["user_msg"], state["job_id"], 6, False)
    hits = res.get("hits", [])
    if not hits:
        state["answer"] = "I couldn’t find anything relevant in the transcript."
        return state
    parts = [
        f"{h['metadata'].get('speaker','Speaker')} "
        f"[{h['metadata'].get('start','?')}-{h['metadata'].get('end','?')}]: {h['text']}"
        for h in hits[:6]
    ]
    state["answer"] = "Here’s what I found:\n\n" + "\n\n".join(parts)
    state["rag_hits"] = hits
    return state

async def node_web(state: ChatState) -> ChatState:
    q = state["user_msg"]
    res = await web_search_tavily_tool(q, 5)

    # Surface the error so you know why it didn't search
    err = res.get("error")
    if err:
        state["answer"] = (
            "I tried to use live web search but couldn’t.\n\n"
            f"- Reason: {err}\n"
            "- Fix: install `tavily-python`, set `TAVILY_API_KEY` in your project root .env, and restart the server."
        )
        return state

    items = res.get("results", [])
    if not items:
        return await node_chat(state)

    notes = "\n\n".join([
        f"- {it.get('title','Untitled')}\n  {(it.get('snippet') or it.get('content') or '')}\n  URL: {it.get('url','')}"
        for it in items[:5]
    ])
    sys = SystemMessage(content="You are a careful researcher. Use ONLY the provided notes to answer.")
    user = HumanMessage(content=f"Question: {q}\n\nNotes:\n{notes}")
    result = await _llm_research.ainvoke([sys, user])
    state["answer"] = result.content or ""
    state["web_items"] = items
    return state

async def node_chat(state: ChatState) -> ChatState:
    history = await _session_history_messages(state["session_id"], 12)
    sys = SystemMessage(content="You are a helpful assistant. Keep answers concise unless asked otherwise.")
    msgs = [sys] + history + [HumanMessage(content=state["user_msg"])]
    result = await _llm_default.ainvoke(msgs)
    state["answer"] = result.content or ""
    return state

async def node_persist_exit(state: ChatState) -> ChatState:
    if state.get("answer"):
        await _persist_bot_message(state["session_id"], state["answer"])
    return state

# ---------------- Graph builder ----------------
def build_chat_graph():
    g = StateGraph(ChatState)

    g.add_node("router", node_router)
    g.add_node("full_tx", node_full_tx)
    g.add_node("summary", node_summary)
    g.add_node("email_request", node_email_request)
    g.add_node("email_send", node_email_send)
    g.add_node("audio_rag", node_audio_rag)
    g.add_node("web", node_web)
    g.add_node("chat", node_chat)
    g.add_node("persist_exit", node_persist_exit)

    g.set_entry_point("router")

    def _branch_selector(s: ChatState):
        return s.get("route", "chat")

    g.add_conditional_edges(
        "router",
        _branch_selector,
        {
            "full_tx": "full_tx",
            "summary": "summary",
            "email_request": "email_request",
            "email_send": "email_send",
            "audio_rag": "audio_rag",
            "web": "web",
            "chat": "chat",
        },
    )

    for leaf in ["full_tx", "summary", "email_request", "email_send", "audio_rag", "web", "chat"]:
        g.add_edge(leaf, "persist_exit")
    g.add_edge("persist_exit", END)

    return g.compile()
