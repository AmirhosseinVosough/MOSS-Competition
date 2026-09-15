"""The process-wide Moss client.

Moss serves queries from the index held in this process's own memory, so the
index has to be downloaded once before anything can be searched. Doing that
per conversation cost ~2.5s of silence at the start of every call, which on a
project about latency was the wrong place to leave two and a half seconds.
"""

import asyncio
import logging
import os
import time

from moss import MossClient

from config import KNOWLEDGE_INDEX

logger = logging.getLogger("agent.moss")

# One Moss client per worker process, with its knowledge index already loaded.
# Building it per conversation cost ~2.5s of silence at the start of every call,
# because each Assistant created its own client and re-downloaded the index.
_shared_client: MossClient | None = None
_shared_client_lock = asyncio.Lock()


async def get_shared_client() -> MossClient:
    """Return the process-wide Moss client, loading the knowledge index once.

    The lock matters: without it two callers arriving together would both see an
    empty slot and both pay the load.
    """
    global _shared_client
    async with _shared_client_lock:
        if _shared_client is None:
            client = MossClient(
                os.getenv("MOSS_PROJECT_ID"), os.getenv("MOSS_PROJECT_KEY")
            )
            started = time.perf_counter()
            await client.load_index(KNOWLEDGE_INDEX)
            logger.info(
                "Loaded Moss knowledge index '%s' in %.0fms (once per worker)",
                KNOWLEDGE_INDEX,
                (time.perf_counter() - started) * 1000,
            )
            _shared_client = client
    return _shared_client
