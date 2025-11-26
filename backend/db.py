import os
import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger("db")
logger.setLevel(logging.INFO)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "chatbot")

# ---------- Async client ----------
_async_client = AsyncIOMotorClient(MONGO_URI)
db = _async_client[MONGO_DB]

jobs = db.jobs
messages = db.messages
sessions = db.sessions
checkpoints = db.checkpoints
notifications = db.notifications

# ---------- Sync helpers ----------
def get_sync_client(server_selection_timeout_ms: int = 3000):
    from pymongo import MongoClient
    if getattr(get_sync_client, "_client", None) is None:
        get_sync_client._client = MongoClient(
            MONGO_URI, serverSelectionTimeoutMS=server_selection_timeout_ms
        )
        try:
            get_sync_client._client.admin.command("ping")
            logger.info("Sync MongoClient connected.")
        except Exception as e:
            logger.warning("Sync MongoClient ping failed (continuing): %s", e)
    return get_sync_client._client  # type: ignore[attr-defined]

def get_sync_db():
    return get_sync_client()[MONGO_DB]

def get_sync_collection(name: str):
    return get_sync_db()[name]
