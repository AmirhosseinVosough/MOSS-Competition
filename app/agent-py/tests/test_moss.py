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

import asyncio
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


class _FakeTurnCtx:
    """Stand-in for llm.ChatContext, recording what gets injected."""

    def __init__(self) -> None:
        self.added: list[tuple] = []

    def add_message(self, role, content):
        self.added.append((role, content))


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.text_content = text


async def test_turn_end_injects_notes_into_the_prompt() -> None:
    """The notes reach the LLM with the question, not via a tool call."""
    assistant = _wire(Assistant(room=_FakeRoom()))
    assistant._moss.query_result = _FakeSearchResult(
        [_FakeDoc("Metformin 500mg twice daily."), _FakeDoc("Aspirin 75mg after lunch.")]
    )
    ctx = _FakeTurnCtx()

    await assistant.on_user_turn_completed(ctx, _FakeMessage("what pills do I take"))

    assert len(ctx.added) == 1
    role, content = ctx.added[0]
    assert role == "system"
    assert "Metformin 500mg twice daily." in content
    assert "Aspirin 75mg after lunch." in content


async def test_turn_end_prefers_notes_gathered_while_speaking() -> None:
    """If a partial already retrieved notes, do not search again at turn end."""
    assistant = _wire(Assistant(room=_FakeRoom()))
    assistant._spec_notes = ["Cardiology appointment Tuesday at half past two."]
    ctx = _FakeTurnCtx()

    await assistant.on_user_turn_completed(ctx, _FakeMessage("when is my appointment"))

    assert "Cardiology appointment" in ctx.added[0][1]
    # No fresh query: the speculative result was reused.
    assert assistant._moss.query_calls == []


async def test_turn_notes_are_cleared_between_turns() -> None:
    """One question's notes must not leak into the next answer."""
    assistant = _wire(Assistant(room=_FakeRoom()))
    assistant._spec_notes = ["Something from the previous question."]

    await assistant.on_user_turn_completed(_FakeTurnCtx(), _FakeMessage("anything"))

    assert assistant._spec_notes == []
    assert assistant._spec_best_seq == -1


async def test_a_stale_speculative_result_cannot_overwrite_a_newer_one() -> None:
    """A search on an earlier, shorter partial may finish last. It must lose."""
    assistant = _wire(Assistant(room=_FakeRoom()))

    assistant._moss.query_result = _FakeSearchResult([_FakeDoc("Newer, better notes.")])
    await assistant._speculative_search("what pills do I take", seq=5)
    assert assistant._spec_notes == ["Newer, better notes."]

    # Sequence 2 started earlier but finishes now; it must be discarded.
    assistant._moss.query_result = _FakeSearchResult([_FakeDoc("Older, vaguer notes.")])
    await assistant._speculative_search("what pills", seq=2)
    assert assistant._spec_notes == ["Newer, better notes."]


async def test_speculation_skips_partials_that_are_too_short() -> None:
    """The first interim of an utterance is routinely empty or one word."""
    assistant = _wire(Assistant(room=_FakeRoom()))

    assistant._maybe_speculate("what")
    assert assistant._spec_tasks == set()

    assistant._maybe_speculate("what pills do I take in the morning")
    assert len(assistant._spec_tasks) == 1
    for task in list(assistant._spec_tasks):
        await task


async def test_injection_failure_does_not_raise() -> None:
    """If retrieval fails at turn end the agent still answers, using its tools."""
    assistant = _wire(Assistant(room=_FakeRoom()))

    async def boom(*args, **kwargs):
        raise RuntimeError("index not loaded")

    assistant._moss.query = boom
    ctx = _FakeTurnCtx()

    await assistant.on_user_turn_completed(ctx, _FakeMessage("what pills do I take"))

    assert ctx.added == []
    assert assistant._spec_notes == []


class _FakeAlt:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeSpeechEvent:
    def __init__(self, etype, text: str = "") -> None:
        self.type = etype
        self.alternatives = [_FakeAlt(text)]


async def _drive_stt_node(assistant, events):
    """Push fake speech events through stt_node, returning what came out."""
    from livekit.agents import Agent

    async def fake_default(_self, _audio, _settings):
        for ev in events:
            yield ev

    original = Agent.default.stt_node
    Agent.default.stt_node = fake_default
    try:
        return [ev async for ev in assistant.stt_node(None, None)]
    finally:
        Agent.default.stt_node = original


async def test_stt_node_passes_every_event_through_and_speculates(monkeypatch) -> None:
    """The hook must be transparent, and must actually call the search.

    This exists because the speculative call was once defined but never wired in:
    the edit that was meant to add it silently did not apply, and nothing failed.
    """
    from livekit.agents import stt as stt_module

    assistant = _wire(Assistant(room=_FakeRoom()))
    fired: list[str] = []
    monkeypatch.setattr(assistant, "_maybe_speculate", lambda text: fired.append(text))

    events = [
        _FakeSpeechEvent(stt_module.SpeechEventType.START_OF_SPEECH),
        _FakeSpeechEvent(stt_module.SpeechEventType.INTERIM_TRANSCRIPT, "what pills do I take"),
        _FakeSpeechEvent(stt_module.SpeechEventType.FINAL_TRANSCRIPT, "what pills do I take?"),
    ]
    out = await _drive_stt_node(assistant, events)

    # Transparent: everything passed through, in order, unchanged.
    assert out == events
    # And the partial actually reached the speculative path.
    assert fired == ["what pills do I take"]


async def test_stt_node_keeps_streaming_when_our_own_code_raises(monkeypatch) -> None:
    """A failure in the observer must not stop transcription."""
    from livekit.agents import stt as stt_module

    assistant = _wire(Assistant(room=_FakeRoom()))

    def boom(_text):
        raise RuntimeError("index not loaded")

    monkeypatch.setattr(assistant, "_maybe_speculate", boom)

    events = [
        _FakeSpeechEvent(stt_module.SpeechEventType.INTERIM_TRANSCRIPT, "what pills do I take"),
        _FakeSpeechEvent(stt_module.SpeechEventType.FINAL_TRANSCRIPT, "what pills do I take?"),
    ]
    out = await _drive_stt_node(assistant, events)

    # The agent stays hearing even though the observer blew up on every event.
    assert out == events


async def test_speech_start_does_not_discard_notes_gathered_mid_sentence() -> None:
    """Voice detection fires constantly; it must not wipe the buffer.

    Regression: clearing on START_OF_SPEECH threw away notes retrieved while the
    user was speaking, so every injection fell back to a turn-end search.
    """
    from livekit.agents import stt as stt_module

    assistant = _wire(Assistant(room=_FakeRoom()))
    assistant._spec_notes = ["Metformin 500mg twice daily."]

    await _drive_stt_node(
        assistant, [_FakeSpeechEvent(stt_module.SpeechEventType.START_OF_SPEECH)]
    )

    assert assistant._spec_notes == ["Metformin 500mg twice daily."]


# --------------------------------------------------------------------------
# Grounding gate
# --------------------------------------------------------------------------


async def _collect(agen):
    return [chunk async for chunk in agen]


async def _stream(*chunks):
    for c in chunks:
        yield c


def test_numbers_are_read_as_phrases_not_words() -> None:
    """"five hundred" is 500, not 5 and 100.

    Regression: reading number words individually made the gate reject a correct
    500mg dose, because the notes contain 500 and never 5 or 100. A gate that
    blocks correct information is worse than no gate.
    """
    n = Assistant._numbers_in
    assert n("five hundred milligrams") == {"500"}
    assert n("one thousand milligrams") == {"1000"}
    assert n("twenty two") == {"22"}
    assert n("seventy five") == {"75"}
    assert n("500mg") == {"500"}
    # Ordinals, because dates are spoken as "the fifteenth".
    assert n("the fifteenth") == {"15"}


def test_only_specific_claims_are_checked() -> None:
    """Ordinary conversation must pass without a lookup."""
    checkable = Assistant._is_checkable
    assert checkable("You take metformin, five hundred milligrams, twice daily.")
    assert checkable("Your appointment is on the fifteenth.")
    # No number to be wrong about.
    assert not checkable("You are allergic to penicillin.")
    assert not checkable("Good morning Bill, how did you sleep?")
    assert not checkable("The dahlias are looking lovely today.")


async def test_gate_speaks_a_supported_claim() -> None:
    assistant = _wire(Assistant(room=_FakeRoom()))
    assistant._moss.query_result = _FakeSearchResult(
        [_FakeDoc("Metformin 500mg. Take one tablet twice daily.")]
    )
    out = await _collect(
        assistant._gate(_stream("You take metformin, five hundred milligrams. "))
    )
    assert "five hundred milligrams" in "".join(out)


async def test_gate_replaces_an_unsupported_claim_with_a_hedge() -> None:
    """The wrong dose must not reach the speaker."""
    from agent import GATE_HEDGE

    assistant = _wire(Assistant(room=_FakeRoom()))
    assistant._moss.query_result = _FakeSearchResult(
        [_FakeDoc("Metformin 500mg. Take one tablet twice daily.")]
    )
    spoken = "".join(
        await _collect(
            assistant._gate(_stream("You take metformin, one thousand milligrams. "))
        )
    )
    assert "one thousand" not in spoken
    assert GATE_HEDGE.split(".")[0] in spoken


async def test_gate_leaves_ordinary_conversation_alone() -> None:
    """No claim, no lookup, no interference."""
    assistant = _wire(Assistant(room=_FakeRoom()))
    spoken = "".join(
        await _collect(assistant._gate(_stream("Good morning Bill. ", "How did you sleep? ")))
    )
    assert spoken == "Good morning Bill. How did you sleep? "
    assert assistant._moss.query_calls == []


async def test_gate_fails_open_when_retrieval_breaks() -> None:
    """A broken gate must not silence the agent.

    Staying quiet is a worse failure than speaking an unverified sentence: the
    model is already instructed to ground its answers, and a mute assistant is
    no use to anyone.
    """
    assistant = _wire(Assistant(room=_FakeRoom()))

    async def boom(*args, **kwargs):
        raise RuntimeError("index not loaded")

    assistant._moss.query = boom
    spoken = "".join(
        await _collect(
            assistant._gate(_stream("You take metformin, five hundred milligrams. "))
        )
    )
    assert "five hundred milligrams" in spoken


async def test_gate_fails_open_when_retrieval_hangs() -> None:
    """A hung lookup must not stop the agent speaking either."""
    import agent as agent_module

    assistant = _wire(Assistant(room=_FakeRoom()))

    async def hang(*args, **kwargs):
        await asyncio.sleep(10)

    assistant._moss.query = hang
    original = agent_module.GATE_TIMEOUT_S
    agent_module.GATE_TIMEOUT_S = 0.05
    try:
        spoken = "".join(
            await _collect(
                assistant._gate(_stream("You take metformin, five hundred milligrams. "))
            )
        )
    finally:
        agent_module.GATE_TIMEOUT_S = original
    assert "five hundred milligrams" in spoken


# --- the memory journal -------------------------------------------------
# Facts live only in the local session until on_exit pushes them, and the agent
# aborts on shutdown often enough that relying on that push loses real data. The
# journal is the crash insurance: one appended line per fact, replayed on the
# next start.


@pytest.fixture
def journal(tmp_path, monkeypatch):
    path = tmp_path / "journal.jsonl"
    monkeypatch.setattr(agent_module, "JOURNAL_PATH", path)
    return path


def test_journal_round_trips_a_fact(journal) -> None:
    from moss import DocumentInfo

    agent_module.journal_append(
        DocumentInfo(id="bill-1", text="His knee ached.", metadata={"patient_id": "bill"})
    )

    docs = agent_module.journal_read()
    assert len(docs) == 1
    assert docs[0].id == "bill-1"
    assert docs[0].text == "His knee ached."
    assert docs[0].metadata == {"patient_id": "bill"}


def test_journal_skips_a_half_written_line(journal) -> None:
    """A crash mid-write leaves a truncated final line. Earlier facts survive."""
    journal.write_text(
        '{"id":"bill-1","text":"First fact."}\n'
        '{"id":"bill-2","text":"Second fact."}\n'
        '{"id":"bill-3","text":"Third fa'  # killed mid-write
    )

    docs = agent_module.journal_read()
    assert [d.id for d in docs] == ["bill-1", "bill-2"]


def test_journal_read_is_empty_when_there_is_no_file(journal) -> None:
    assert not journal.exists()
    assert agent_module.journal_read() == []


async def test_remember_fact_journals_what_it_writes(journal) -> None:
    assistant = _wire(Assistant())

    await assistant.remember_fact(None, "His knee ached after gardening.")

    docs = agent_module.journal_read()
    assert len(docs) == 1
    assert docs[0].text == "His knee ached after gardening."
    # Same document that went into the session.
    assert docs[0].id == assistant._memory_session.add_docs_calls[0][0].id


async def test_on_exit_clears_the_journal_only_after_a_successful_push(journal) -> None:
    assistant = _wire(Assistant())
    await assistant.remember_fact(None, "His knee ached.")
    assert journal.exists()

    await assistant.on_exit()

    assert assistant._memory_session.pushed == 1
    assert not journal.exists()


async def test_a_failed_push_keeps_the_journal_for_the_next_run(journal) -> None:
    """If the cloud push fails the facts are not safe yet, so the file stays."""
    assistant = _wire(Assistant())
    await assistant.remember_fact(None, "His knee ached.")

    async def boom():
        raise RuntimeError("cloud unreachable")

    assistant._memory_session.push_index = boom

    await assistant.on_exit()

    assert journal.exists()
    assert agent_module.journal_read()[0].text == "His knee ached."


async def test_a_broken_journal_does_not_break_remembering(journal, monkeypatch) -> None:
    """Journalling is insurance. It must never take down the conversation."""

    def boom(_doc):
        raise OSError("disk full")

    monkeypatch.setattr(agent_module, "journal_append", boom)
    assistant = _wire(Assistant())

    # remember_fact calls the real journal_append, so patch where it is looked up.
    result = await assistant.remember_fact(None, "His knee ached.")

    assert "remember" in result.lower()
    assert len(assistant._memory_session.add_docs_calls) == 1
