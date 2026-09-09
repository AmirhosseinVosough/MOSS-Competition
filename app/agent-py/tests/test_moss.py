"""Unit tests for the agent's Moss-backed tools.

Unlike the LLM-judged evals in `test_agent.py`, these are deterministic unit
tests that exercise the tool methods directly. They stub the Moss client and
session so they run with no Moss credentials and no network access — the live,
credentialed behavior is validated separately.

Note the shape these tests assume: `Assistant` no longer builds its own
`MossClient` (that is now a process-wide, pre-loaded client assigned in
`on_enter`), and all memory reads and writes go through a local `SessionIndex`
rather than the cloud client.
"""

import json

import pytest

import agent as agent_module
from agent import Assistant

PATIENT_ID = agent_module.PATIENT_ID


class _FakeDoc:
    """Stand-in for a Moss query-result document (`.text/.score/.metadata`)."""

    def __init__(self, text: str, score=None, metadata=None) -> None:
        self.text = text
        self.score = score
        self.metadata = metadata


class _FakeSearchResult:
    """Stand-in for a Moss `SearchResult` (`.docs/.time_taken_ms`)."""

    def __init__(self, docs, time_taken_ms: float = 12.5) -> None:
        self.docs = docs
        self.time_taken_ms = time_taken_ms


class _FakeMossClient:
    """Records calls instead of contacting Moss. Assigned to `assistant._moss`."""

    def __init__(self, *args, **kwargs) -> None:
        self.load_index_calls: list[str] = []
        self.query_calls: list[tuple] = []
        self.query_result = _FakeSearchResult([])

    async def load_index(self, name, *args, **kwargs):
        self.load_index_calls.append(name)

    async def query(self, index, query, options=None):
        self.query_calls.append((index, query, options))
        return self.query_result


class _FakeSession:
    """Stand-in for a Moss `SessionIndex` — the local, writable memory view."""

    def __init__(self) -> None:
        self.add_docs_calls: list[list] = []
        self.query_calls: list[tuple] = []
        self.pushed = 0
        self.query_result = _FakeSearchResult([])

    async def add_docs(self, docs, options=None):
        self.add_docs_calls.append(docs)
        return (len(docs), 0)

    async def query(self, query, options=None):
        self.query_calls.append((query, options))
        return self.query_result

    async def push_index(self):
        self.pushed += 1
        return type("PushResult", (), {"doc_count": 1, "job_id": "job-1"})()


class _FakePublisher:
    def __init__(self) -> None:
        self.published: list[tuple] = []

    async def publish_data(self, payload, reliable=None):
        self.published.append((payload, reliable))


class _FakeRoom:
    def __init__(self) -> None:
        self.local_participant = _FakePublisher()


def _wire(assistant, *, session=True):
    """Attach fakes in place of the shared client and local memory session."""
    assistant._moss = _FakeMossClient()
    assistant._memory_session = _FakeSession() if session else None
    return assistant


@pytest.fixture
def assistant_with_room():
    room = _FakeRoom()
    return _wire(Assistant(room=room)), room


async def test_search_care_notes_returns_joined_text_and_publishes_context(
    assistant_with_room,
) -> None:
    """search_care_notes joins snippets and publishes a well-formed payload."""
    assistant, room = assistant_with_room
    assistant._moss.query_result = _FakeSearchResult(
        [
            _FakeDoc("First snippet.", score=0.9, metadata={"source": "notes"}),
            _FakeDoc("Second snippet.", score=0.8),
        ],
        time_taken_ms=7.0,
    )

    result = await assistant.search_care_notes(None, "when is my heart appointment?")

    assert result == "First snippet.\n\nSecond snippet."

    # Queried the knowledge index, semantically, with a small top_k.
    assert len(assistant._moss.query_calls) == 1
    index, query, options = assistant._moss.query_calls[0]
    assert index == agent_module.KNOWLEDGE_INDEX
    assert query == "when is my heart appointment?"
    assert options.top_k == 3
    assert options.alpha == agent_module.SEARCH_ALPHA

    # Published exactly one moss_context message, reliably.
    assert len(room.local_participant.published) == 1
    payload_bytes, reliable = room.local_participant.published[0]
    assert reliable is True

    payload = json.loads(payload_bytes.decode("utf-8"))
    assert payload["type"] == "moss_context"
    data = payload["data"]
    # Contractual keys consumed by the frontend parser.
    assert set(data) == {"query", "matches", "time_taken_ms", "timestamp"}
    assert data["query"] == "when is my heart appointment?"
    assert data["time_taken_ms"] == 7.0
    assert isinstance(data["timestamp"], (int, float))

    matches = data["matches"]
    assert len(matches) == 2
    assert matches[0]["text"] == "First snippet."
    assert matches[0]["score"] == 0.9
    assert matches[0]["metadata"] == {"source": "notes"}


async def test_list_care_category_returns_whole_category_severe_first() -> None:
    """A category sweep returns every note, unranked, with severe ones first."""
    assistant = _wire(Assistant(room=_FakeRoom()))
    assistant._moss.query_result = _FakeSearchResult(
        [
            _FakeDoc("Mild allergy to shellfish.", metadata={"severity": "mild"}),
            _FakeDoc("Serious allergy to penicillin.", metadata={"severity": "severe"}),
        ]
    )

    result = await assistant.list_care_category(None, "allergy")

    # Severe is listed first regardless of the order Moss returned.
    assert result.splitlines()[0] == "Serious allergy to penicillin."
    assert "shellfish" in result

    # Filtered to the category, with a top_k far above the real count so that
    # nothing can be silently truncated.
    _index, _query, options = assistant._moss.query_calls[0]
    assert options.filter == {
        "field": "category",
        "condition": {"$eq": "allergy"},
    }
    assert options.top_k >= 25


async def test_list_care_category_rejects_unknown_category() -> None:
    """An unknown category is refused rather than silently returning nothing."""
    assistant = _wire(Assistant())
    result = await assistant.list_care_category(None, "biscuits")
    assert "not a category" in result
    # Nothing was queried.
    assert assistant._moss.query_calls == []


async def test_remember_fact_writes_locally_and_tags_the_patient() -> None:
    """remember_fact writes through the local session, scoped to the patient."""
    assistant = _wire(Assistant())

    fact = "His knee ached after gardening."
    result = await assistant.remember_fact(None, fact)
    assert isinstance(result, str) and result

    # Written to the local session, never to the cloud client.
    assert len(assistant._memory_session.add_docs_calls) == 1
    docs = assistant._memory_session.add_docs_calls[0]
    assert len(docs) == 1

    doc = docs[0]
    assert doc.text == fact
    # Memory belongs to the person being cared for, not to a browser session.
    assert doc.metadata == {"patient_id": PATIENT_ID}
    assert doc.id.startswith(f"{PATIENT_ID}-")

    # The session is marked dirty so on_exit knows there is something to push.
    assert assistant._memory_dirty is True


async def test_recall_facts_reads_the_session_and_filters_by_patient() -> None:
    """recall_facts reads through the session, scoped to the patient."""
    assistant = _wire(Assistant(room=_FakeRoom()))
    assistant._memory_session.query_result = _FakeSearchResult(
        [_FakeDoc("His knee ached after gardening."), _FakeDoc("Ellie rang on Sunday.")]
    )

    result = await assistant.recall_facts(None, "how has his knee been?")

    assert result == "His knee ached after gardening.\nEllie rang on Sunday."

    # Read from the session, not the cloud client: facts written this turn are
    # only in the local copy, and the cloud cannot answer text queries against a
    # pushed index.
    assert assistant._moss.query_calls == []
    assert len(assistant._memory_session.query_calls) == 1
    query, options = assistant._memory_session.query_calls[0]
    assert query == "how has his knee been?"
    assert options.top_k == 5
    assert options.filter == {
        "field": "patient_id",
        "condition": {"$eq": PATIENT_ID},
    }


async def test_memory_degrades_honestly_without_a_session() -> None:
    """With no session, memory says so plainly instead of crashing or lying.

    There is deliberately no cloud fallback: push_index marks the index as using
    custom embeddings, after which a cloud text query raises. A write we cannot
    read back is worse than an honest refusal.
    """
    assistant = _wire(Assistant(), session=False)

    said = await assistant.remember_fact(None, "His knee ached.")
    assert "can't hold on to that" in said
    assert assistant._memory_dirty is False

    recalled = await assistant.recall_facts(None, "how is his knee?")
    assert "can't call anything to mind" in recalled

    # Nothing was written to the cloud client behind our back.
    assert assistant._moss.query_calls == []


async def test_on_exit_pushes_only_when_there_is_something_to_push() -> None:
    """The cloud push happens at teardown, and only if a fact was written."""
    assistant = _wire(Assistant())

    # Nothing written yet -> nothing pushed.
    await assistant.on_exit()
    assert assistant._memory_session.pushed == 0

    await assistant.remember_fact(None, "His knee ached.")
    await assistant.on_exit()
    assert assistant._memory_session.pushed == 1
    assert assistant._memory_dirty is False
