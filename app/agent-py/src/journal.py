"""Crash insurance for facts the agent has just learned.

Facts are written to the local Moss session in ~4.5ms and only pushed to the
cloud when a conversation ends, because push_index triggers a cloud rebuild
that returns in ~2s and takes ~74s to complete. That trade is what makes
remembering fast, and it means everything learned sits somewhere fragile
until the very end -- with an agent that aborts on shutdown routinely
(exit code -6, "mutex lock failed").

So each fact is also appended here the instant it is learned. Appending costs
about a tenth of a millisecond and survives the process dying; the index
itself cannot be saved cheaply, because saving it means rebuilding it.
"""

import json
import logging
import os
import pathlib
import time

from moss import DocumentInfo

logger = logging.getLogger("agent.journal")

# Crash insurance for remembered facts.
#
# Facts are written to the local session in ~4.5ms and only pushed to the cloud
# at the end of a conversation, because push_index triggers a cloud rebuild that
# returns in ~2s and takes ~74s to complete. That trade is what makes remembering
# fast, but it means a crash between the two loses everything learned -- and this
# agent aborts on shutdown often enough (exit code -6, "mutex lock failed") that
# the loss is routine rather than theoretical.
#
# So every fact is also appended to a plain file the instant it is learned.
# Appending is O(1) and survives the process dying; the index cannot be saved
# cheaply because saving it means rebuilding it. On startup anything still in the
# file is replayed into the session, and the file is cleared once a push succeeds.
JOURNAL_PATH = pathlib.Path(
    os.getenv("MEMORY_JOURNAL_PATH")
    or pathlib.Path(__file__).resolve().parent.parent / ".memory-journal.jsonl"
)


def journal_append(doc: DocumentInfo) -> None:
    """Append one fact. Never raises -- the conversation matters more."""
    try:
        JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with JOURNAL_PATH.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "id": doc.id,
                        "text": doc.text,
                        "metadata": doc.metadata or {},
                        "ts": time.time(),
                    }
                )
                + "\n"
            )
            fh.flush()
            # Without fsync the line can sit in an OS buffer and die with the
            # process, which would defeat the entire point of writing it.
            os.fsync(fh.fileno())
    except Exception:
        logger.exception("could not journal fact %s", getattr(doc, "id", "?"))


def journal_read() -> list[DocumentInfo]:
    """Facts from a previous run that never reached the cloud."""
    if not JOURNAL_PATH.exists():
        return []
    docs: list[DocumentInfo] = []
    try:
        for line in JOURNAL_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # A half-written final line is expected after a crash. Skip it
                # rather than discarding every fact that came before.
                logger.warning("skipping malformed journal line")
                continue
            if entry.get("id") and entry.get("text"):
                docs.append(
                    DocumentInfo(
                        id=entry["id"],
                        text=entry["text"],
                        metadata=entry.get("metadata") or {},
                    )
                )
    except Exception:
        logger.exception("could not read memory journal")
    return docs


def journal_clear() -> None:
    """Called only after a push succeeds; the cloud now holds these facts."""
    try:
        JOURNAL_PATH.unlink(missing_ok=True)
    except Exception:
        logger.exception("could not clear memory journal")
