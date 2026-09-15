import asyncio
import contextlib
import json
import logging
import re
import textwrap
import time
import uuid
from datetime import datetime, timezone

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.agents import stt as stt_module
from livekit.plugins import ai_coustics, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from moss import DocumentInfo, MossClient, QueryOptions

from config import (
    DEFAULT_USER_ID,
    GATE_HEDGE,
    GATE_TIMEOUT_S,
    KNOWLEDGE_INDEX,
    MEMORY_INDEX,
    PATIENT_ID,
    SEARCH_ALPHA,
    SIMULATED_CLOUD_LATENCY_S,
    SPECULATIVE_MIN_CHARS,
    SPECULATIVE_MIN_GAP_MS,
    SWEEPABLE_CATEGORIES,
)
from grounding import is_checkable, numbers_in
from journal import journal_append, journal_clear, journal_read
from moss_runtime import get_shared_client

logger = logging.getLogger("agent")


class Assistant(Agent):
    """Voice agent that wires Moss retrieval + per-user memory into LiveKit."""

    def __init__(self, *, room=None, user_id: str = DEFAULT_USER_ID) -> None:
        super().__init__(
            # The LLM (the agent's brain) runs on LiveKit Inference — no
            # provider API key required. STT/TTS are configured on the
            # AgentSession below. See https://docs.livekit.io/agents/models/llm/
            llm=inference.LLM(model="openai/gpt-5.2-chat-latest"),
            instructions=textwrap.dedent(
                """\
                You are a calm, warm companion for Bill, an older gentleman
                living on his own. You help him keep track of his medicines,
                his appointments, and the people in his life. You are good
                company, not a nurse and not a machine reading out a list.

                # Grounding (this is the important part)

                - Before saying ANYTHING about a medicine, a dose, a time, a
                  date, an appointment, an allergy, or a phone number, you MUST
                  look it up and use what comes back.
                - If he asks about one specific thing, use `search_care_notes`.
                - If he asks about a whole group -- what he is allergic to, what
                  medicines he takes, what appointments are coming up, who he can
                  call -- use `list_care_category` instead. It returns every note
                  in that group. Searching would return only the closest few and
                  could leave out something that matters.
                - When you read out allergies, always say the serious ones
                  first.
                - Never state a dose, a time, a date, or a number that did not
                  appear in the notes you retrieved. Not a guess, not a
                  reasonable assumption, not something you recall from earlier.
                - If the notes do not answer the question, say plainly that you
                  do not have it written down and offer to help him check with
                  Sarah or the surgery. That is always the right answer, and it
                  is never a failure.
                - If he tells you something that contradicts the notes, do not
                  argue. Say what the notes say, and suggest checking with
                  Sarah.

                # Remembering

                - When he shares something worth keeping - how he slept, that
                  his knee hurts, that Ellie called - use `remember_fact`.
                - When something he asks depends on an earlier conversation,
                  use `recall_facts` first.

                # How to speak

                Your words are spoken aloud, so they must sound like speech:

                - Plain sentences only. No lists, no markdown, no symbols, no
                  emoji, no formatting of any kind.
                - Two or three sentences at most. Ask one question at a time.
                - Say numbers as words. Half past two, not 14:30. Ten milligrams,
                  not 10mg.
                - Do not speak slowly or over-explain. He is hard of hearing,
                  not slow, and he dislikes being fussed over.
                - Call him Bill.

                # Care

                - If he mentions chest pain, a fall, or sounds confused or
                  frightened, say clearly that he should ring Sarah or 999, and
                  stay with him.
                - Never give medical advice of your own. You read what is
                  written down; you do not decide anything.
                """
            ),
        )
        self._room = room
        self._user_id = user_id
        # Assigned in on_enter from the process-wide client, which already has
        # the knowledge index loaded.
        self._moss: MossClient | None = None
        # Local, mutable view of the memory index. Writes land in process memory
        # instead of triggering a cloud rebuild + full re-download, which measured
        # 4257ms per fact (bench/bench_memory.py). Local writes measure ~4.5ms.
        # Falls back to the cloud path if the session cannot be created.
        self._memory_session = None
        self._memory_dirty = False
        # Notes retrieved while the user was still speaking, used by
        # on_user_turn_completed. Sequence numbers guard against a slow search on
        # an early, shorter partial landing after a later, better one.
        self._spec_notes: list[str] = []
        self._spec_seq = 0
        self._spec_best_seq = -1
        self._spec_last_text = ""
        self._spec_last_at = 0.0
        self._spec_tasks: set = set()
        # Toggled from the browser. Off unless someone deliberately turns it on.
        self._simulate_cloud_latency = False

    async def on_enter(self) -> None:
        # Preload the knowledge index and open a writable session over memory so
        # the first query is fast. Guarded: log and continue on failure so the
        # tools can still retry on use.
        #
        # Note: the spoken greeting is intentionally triggered from the
        # entrypoint (after `session.start`/`ctx.connect`) rather than here, per
        # the documented LiveKit pattern. Keeping `on_enter` side-effect-free for
        # speech keeps `session.start(Assistant())` deterministic for the evals
        # in tests/test_agent.py (a single turn yields a single reply).
        if self._moss is None:
            try:
                self._moss = await get_shared_client()
            except Exception:
                logger.exception("Could not obtain shared Moss client")
                return

        # Open a writable local session over the memory index. session() adopts the
        # stored model, so model_id is deliberately omitted -- passing one that
        # disagrees with the index raises.
        if self._memory_session is None:
            try:
                self._memory_session = await self._moss.session(MEMORY_INDEX)
                logger.info("Opened local Moss session for '%s'", MEMORY_INDEX)

                # Anything in the journal belongs to a run that died before it
                # could push. Upserts are keyed on id, so replaying a fact that
                # did make it to the cloud is harmless.
                pending = journal_read()
                if pending:
                    try:
                        await self._memory_session.add_docs(pending)
                        self._memory_dirty = True
                        logger.info(
                            "recovered %d fact(s) from the journal after an "
                            "unclean shutdown",
                            len(pending),
                        )
                    except Exception:
                        logger.exception("could not replay journalled facts")
            except Exception:
                logger.exception(
                    "Could not open memory session; falling back to cloud writes"
                )
                try:
                    await self._moss.load_index(MEMORY_INDEX)
                except Exception:
                    logger.exception("Failed to load memory index for fallback reads")

    def _reset_turn_notes(self) -> None:
        """Drop anything gathered for the previous turn.

        Without this the previous question's notes would still be in the buffer
        when the next one is answered -- a correctness bug, not untidiness.
        """
        self._spec_notes = []
        self._spec_best_seq = -1
        self._spec_last_text = ""
        self._spec_last_at = 0.0

    async def _speculative_search(self, text: str, seq: int) -> None:
        """Search on a partial transcript. Never awaited by the audio path."""
        try:
            result, elapsed_ms = await self._timed_query(
                KNOWLEDGE_INDEX, text, QueryOptions(top_k=3, alpha=SEARCH_ALPHA)
            )
            # A search started earlier can finish later. Only a newer sequence
            # number may replace the buffer.
            if seq < self._spec_best_seq:
                return
            self._spec_best_seq = seq
            self._spec_notes = [
                (getattr(d, "text", "") or "").strip()
                for d in (getattr(result, "docs", None) or [])
            ]
            self._spec_notes = [n for n in self._spec_notes if n]
            logger.info(
                "speculative search #%d buffered %d note(s) for %r",
                seq,
                len(self._spec_notes),
                text[:45],
            )
            await self._publish_moss_context(
                f"(mid-sentence) {text}", result, elapsed_ms
            )
        except Exception:
            # Speculative retrieval is an optimisation; its failure must never
            # surface to the caller or the audio pipeline.
            logger.exception("speculative search failed for %r", text[:40])

    def _maybe_speculate(self, text: str) -> None:
        """Fire a search on a partial if it is worth one. Returns immediately."""
        now = time.perf_counter()
        if len(text) < SPECULATIVE_MIN_CHARS:
            return
        if text == self._spec_last_text:
            return
        if (now - self._spec_last_at) * 1000 < SPECULATIVE_MIN_GAP_MS:
            return
        if self._moss is None:
            return

        self._spec_last_text = text
        self._spec_last_at = now
        self._spec_seq += 1
        # create_task, not await: blocking here would stall transcription.
        task = asyncio.create_task(self._speculative_search(text, self._spec_seq))
        self._spec_tasks.add(task)
        task.add_done_callback(self._spec_tasks.discard)

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        """Put the relevant care notes in front of the LLM with the question.

        This is the change that actually pays. Without it the LLM has to notice it
        needs a fact, call a tool, wait, then think again -- two round trips. With
        the notes already in context it answers in one. The tools remain for
        follow-ups the injected notes do not cover.

        Uses notes gathered while the user was speaking when there are any;
        otherwise searches now, which at ~5ms is still far cheaper than the round
        trip it removes.
        """
        try:
            text = ""
            content = getattr(new_message, "text_content", None) or getattr(
                new_message, "content", None
            )
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join(c for c in content if isinstance(c, str))
            text = (text or "").strip()
            if not text:
                return

            notes, source = self._spec_notes, "while speaking"
            if not notes:
                result, turn_ms = await self._timed_query(
                    KNOWLEDGE_INDEX, text, QueryOptions(top_k=3, alpha=SEARCH_ALPHA)
                )
                notes = [
                    (getattr(d, "text", "") or "").strip()
                    for d in (getattr(result, "docs", None) or [])
                ]
                notes = [n for n in notes if n]
                source = "at turn end"
                await self._publish_moss_context(text, result, turn_ms)

            if notes:
                turn_ctx.add_message(
                    role="system",
                    content=(
                        "Notes from Bill's care records that may be relevant to what "
                        "he just said. Use them if they answer his question. If they "
                        "do not, look further before answering:\n\n"
                        + "\n".join(f"- {n}" for n in notes)
                    ),
                )
                logger.info(
                    "injected %d note(s) retrieved %s for %r",
                    len(notes),
                    source,
                    text[:50],
                )
        except Exception:
            # Falling back to the tool path is slower but still correct.
            logger.exception("context injection failed; tools remain available")
        finally:
            self._reset_turn_notes()

    async def stt_node(self, audio, model_settings):
        """Wrap the default speech-to-text node to watch partial transcripts.

        Speech-to-text emits INTERIM_TRANSCRIPT events as the user is still
        speaking -- "what", "what pills", "what pills do I take". The default
        pipeline ignores them and acts only on the final one. Retrieval is cheap
        enough (about 5ms) to run against every partial instead, so the notes are
        already assembled by the time the user stops talking.

        Each partial long enough to be worth one triggers a search whose result
        is buffered for on_user_turn_completed. Timings are logged alongside so
        the cadence stays visible. Every event is passed through untouched.
        """
        turn_started = time.perf_counter()
        last_at = turn_started
        last_text = ""
        seen = 0

        async for event in Agent.default.stt_node(self, audio, model_settings):
            # Everything this node does beyond passing the event on is
            # observation. It sits between the microphone and the rest of the
            # pipeline, so a failure here would stop transcription entirely and
            # the agent would simply go deaf. Swallow anything our own code
            # raises, and yield the event regardless.
            try:
                etype = getattr(event, "type", None)

                if etype == stt_module.SpeechEventType.INTERIM_TRANSCRIPT:
                    alts = getattr(event, "alternatives", None) or []
                    text = (alts[0].text if alts else "") or ""
                    now = time.perf_counter()
                    seen += 1
                    logger.info(
                        "PARTIAL #%d  +%.0fms since last  %.0fms into turn  "
                        "chars=%d (+%d)  %r",
                        seen,
                        (now - last_at) * 1000,
                        (now - turn_started) * 1000,
                        len(text),
                        len(text) - len(last_text),
                        text[-60:],
                    )
                    last_at, last_text = now, text
                    self._maybe_speculate(text)

                elif etype == stt_module.SpeechEventType.PREFLIGHT_TRANSCRIPT:
                    alts = getattr(event, "alternatives", None) or []
                    text = (alts[0].text if alts else "") or ""
                    now = time.perf_counter()
                    seen += 1
                    logger.info(
                        "PREFLIGHT #%d  +%.0fms since last  %.0fms into turn  "
                        "chars=%d (+%d)  %r",
                        seen,
                        (now - last_at) * 1000,
                        (now - turn_started) * 1000,
                        len(text),
                        len(text) - len(last_text),
                        text[-60:],
                    )
                    last_at, last_text = now, text
                    self._maybe_speculate(text)

                elif etype == stt_module.SpeechEventType.FINAL_TRANSCRIPT:
                    alts = getattr(event, "alternatives", None) or []
                    text = (alts[0].text if alts else "") or ""
                    logger.info(
                        "FINAL after %d partials, %.0fms into turn: %r",
                        seen,
                        (time.perf_counter() - turn_started) * 1000,
                        text,
                    )
                    turn_started = time.perf_counter()
                    last_at, last_text, seen = turn_started, "", 0

                elif etype == stt_module.SpeechEventType.START_OF_SPEECH:
                    turn_started = last_at = time.perf_counter()
                    last_text, seen = "", 0
                    # Deliberately NOT clearing the speculative notes here.
                    # Voice activity detection fires constantly on silence and on
                    # the agent's own speech -- most of these events are followed
                    # by an empty transcript. Clearing on each one wiped the notes
                    # gathered mid-sentence before on_user_turn_completed could use
                    # them, so every injection fell back to a turn-end search.
                    # on_user_turn_completed clears in its `finally`, which is the
                    # only point a turn is genuinely over.
                    logger.info("--- START OF SPEECH ---")

            except Exception:
                logger.exception("stt_node observer failed; passing event through")

            yield event

    async def _is_supported(self, sentence: str) -> bool:
        """Is every number in this sentence present in Bill's source records?

        Deliberately not a similarity threshold. Similarity cannot tell 500mg
        from 1000mg -- they are near-identical as text and opposite as facts.
        Comparing the numbers themselves can.
        """
        result, _ = await self._timed_query(
            KNOWLEDGE_INDEX,
            sentence,
            QueryOptions(
                top_k=5,
                alpha=SEARCH_ALPHA,
                filter={
                    "field": "doc_type",
                    "condition": {"$eq": "source_of_truth"},
                },
            ),
        )
        evidence = " ".join(
            (getattr(d, "text", "") or "") for d in (getattr(result, "docs", None) or [])
        )
        if not evidence.strip():
            return False
        supported = numbers_in(evidence)
        claimed = numbers_in(sentence)
        return claimed.issubset(supported)

    async def _gate(self, text_stream):
        """Yield the agent's words, holding back unsupported factual claims.

        Fails open throughout: any error here lets the sentence through. A silent
        assistant is a worse outcome than an unverified sentence, and the model is
        already instructed to ground its answers.
        """
        buffer = ""
        async for chunk in text_stream:
            buffer += chunk
            # Emit sentence by sentence so a claim is checked before it is heard.
            while True:
                match = re.search(r"[.!?]+[\s]", buffer)
                if not match:
                    break
                sentence, buffer = buffer[: match.end()], buffer[match.end() :]
                async for out in self._gate_one(sentence):
                    yield out
        if buffer.strip():
            async for out in self._gate_one(buffer):
                yield out

    async def _timed_query(self, index: str, query: str, options: QueryOptions):
        """Run a Moss query and report how long it actually took.

        Every lookup goes through here so that (a) the timing shown in the UI is
        measured the same way everywhere, and (b) the cloud-latency simulation
        has exactly one place to live rather than four.

        The simulated delay is inside the measurement on purpose: when it is on,
        the UI should show what the user is really waiting for.
        """
        started = time.perf_counter()
        if self._simulate_cloud_latency:
            await asyncio.sleep(SIMULATED_CLOUD_LATENCY_S)
        result = await self._moss.query(index, query, options)
        return result, (time.perf_counter() - started) * 1000

    async def _publish_gate_event(self, sentence: str, verdict: str, ms: float) -> None:
        """Tell the frontend a sentence was checked before it was spoken.

        Retrieval is already visible in the panel; verification was not, and it
        is the part that only works because a lookup costs milliseconds.
        """
        if self._room is None:
            return
        try:
            payload = {
                "type": "moss_gate",
                "data": {
                    "sentence": sentence.strip()[:160],
                    "verdict": verdict,  # pass | block | skip | timeout
                    "time_taken_ms": round(ms, 2),
                    "timestamp": datetime.now(timezone.utc).timestamp(),
                },
            }
            await self._room.local_participant.publish_data(
                payload=json.dumps(payload, default=str).encode("utf-8"), reliable=True
            )
        except Exception:
            logger.exception("failed to publish moss_gate event")

    async def _gate_one(self, sentence: str):
        """Check a single sentence and yield it, or a hedge in its place."""
        try:
            if self._moss is None or not is_checkable(sentence):
                yield sentence
                return

            started = time.perf_counter()
            supported = await asyncio.wait_for(
                self._is_supported(sentence), timeout=GATE_TIMEOUT_S
            )
            elapsed_ms = (time.perf_counter() - started) * 1000

            if supported:
                logger.info("gate PASS %.1fms %r", elapsed_ms, sentence.strip()[:60])
                await self._publish_gate_event(sentence, "pass", elapsed_ms)
                yield sentence
            else:
                logger.warning(
                    "gate BLOCK %.1fms unsupported claim: %r",
                    elapsed_ms,
                    sentence.strip()[:60],
                )
                await self._publish_gate_event(sentence, "block", elapsed_ms)
                yield GATE_HEDGE + " "
        except TimeoutError:
            logger.warning("gate timed out; letting sentence through")
            yield sentence
        except Exception:
            logger.exception("gate failed; letting sentence through")
            yield sentence

    async def tts_node(self, text, model_settings):
        """Speak the agent's reply, with the grounding gate in front of it.

        This is the point where Moss's latency stops being a nicety. One
        retrieval per spoken sentence is only affordable at single-digit
        milliseconds; the same design against a hosted vector database would add
        seconds to every reply.
        """
        async for frame in Agent.default.tts_node(
            self, self._gate(text), model_settings
        ):
            yield frame

    async def on_exit(self) -> None:
        # Persist locally-written facts so they survive into the next session.
        # Deliberately deferred to teardown: push_index triggers a cloud rebuild,
        # which is exactly the multi-second stall we removed from remember_fact.
        if self._memory_session is not None and self._memory_dirty:
            try:
                result = await self._memory_session.push_index()
                logger.info(
                    "Pushed memory session to cloud (docs=%s)",
                    getattr(result, "doc_count", "?"),
                )
                self._memory_dirty = False
                # Only now is it safe to drop the journal. If the push had raised
                # we would keep the file so the next run can replay it.
                journal_clear()
            except Exception:
                logger.exception("Failed to push memory session to cloud")

    async def _publish_moss_context(
        self, query: str, result, elapsed_ms: float | None = None
    ) -> None:
        """Publish a `moss_context` data message for the frontend panel.

        The payload shape is contractual — the frontend parser
        (agent-react/hooks/useMossContextEvents.ts) depends on these exact
        keys. `timestamp` is epoch SECONDS (the frontend multiplies by 1000).
        """
        if self._room is None:
            return
        try:
            matches: list[dict] = []
            for doc in getattr(result, "docs", None) or []:
                entry: dict = {"text": (getattr(doc, "text", "") or "").strip()}
                score = getattr(doc, "score", None)
                if score is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        entry["score"] = float(score)
                metadata = getattr(doc, "metadata", None)
                if metadata:
                    entry["metadata"] = metadata
                matches.append(entry)

            payload = {
                "type": "moss_context",
                "data": {
                    "query": query,
                    "matches": matches,
                    # Moss reports time_taken_ms as a whole number covering only
                    # the vector comparison, which is sub-millisecond and so
                    # rounds to 0. Prefer our own wall-clock measurement: it
                    # includes embedding generation, which is most of the cost
                    # and is time the user actually waits.
                    "time_taken_ms": (
                        round(elapsed_ms, 2)
                        if elapsed_ms is not None
                        else getattr(result, "time_taken_ms", None)
                    ),
                    "timestamp": datetime.now(timezone.utc).timestamp(),
                },
            }
            encoded = json.dumps(payload, default=str).encode("utf-8")
            await self._room.local_participant.publish_data(
                payload=encoded, reliable=True
            )
        except Exception:
            logger.exception("Failed to publish moss_context data")

    @function_tool()
    async def search_care_notes(self, context: RunContext, query: str) -> str:
        """Look up Bill's care notes: medicines, appointments, allergies, contacts.

        You MUST call this before saying anything about a medicine, a dose, a
        time, a date, an appointment, an allergy, or a phone number. It also
        covers his family, his routines and what he likes, so use it whenever
        the answer depends on something about him rather than the world.

        Returns the matching notes as plain text. If nothing comes back, say
        you do not have it written down rather than guessing.

        Args:
            query: What to look up, in the user's own words.
        """
        result, elapsed_ms = await self._timed_query(
            KNOWLEDGE_INDEX, query, QueryOptions(top_k=3, alpha=SEARCH_ALPHA)
        )
        await self._publish_moss_context(query, result, elapsed_ms)

        docs = getattr(result, "docs", None) or []
        snippets = [(getattr(d, "text", "") or "").strip() for d in docs]
        snippets = [s for s in snippets if s]
        if not snippets:
            return "No relevant documentation was found for that question."
        return "\n\n".join(snippets)

    @function_tool()
    async def list_care_category(self, context: RunContext, category: str) -> str:
        """List every note in one category of Bill's records, leaving none out.

        Use this instead of `search_care_notes` whenever the question is about a
        whole group rather than one item -- "what am I allergic to", "what
        medicines do I take", "what appointments have I got coming up", "who can
        I call". Searching ranks notes and returns only the closest few, which
        can leave out something important; this returns all of them.

        Args:
            category: One of allergy, medication, appointment, contact,
                      medical_alert.
        """
        category = (category or "").strip().lower()
        if category not in SWEEPABLE_CATEGORIES:
            return (
                f"'{category}' is not a category I can list. "
                f"Valid categories: {', '.join(SWEEPABLE_CATEGORIES)}."
            )

        # top_k is deliberately far above the real count so nothing is truncated.
        result, elapsed_ms = await self._timed_query(
            KNOWLEDGE_INDEX,
            category,
            QueryOptions(
                top_k=25,
                filter={"field": "category", "condition": {"$eq": category}},
            ),
        )
        await self._publish_moss_context(
            f"all {category} notes", result, elapsed_ms
        )

        docs = getattr(result, "docs", None) or []
        # Severe first, so the most important note is never buried at the bottom.
        docs = sorted(
            docs,
            key=lambda d: 0 if (d.metadata or {}).get("severity") == "severe" else 1,
        )
        notes = [(getattr(d, "text", "") or "").strip() for d in docs]
        notes = [n for n in notes if n]
        if not notes:
            return f"There are no {category} notes on file."
        return "\n".join(notes)

    @function_tool()
    async def remember_fact(self, context: RunContext, fact: str) -> str:
        """Persist a durable fact the user shares about themselves.

        Use for the user's name, role, what they're building, or preferences,
        so you can recall it in future turns and sessions.

        Args:
            fact: A short, self-contained statement of the fact to remember.
        """
        doc = DocumentInfo(
            id=f"{PATIENT_ID}-{uuid.uuid4()}",
            text=fact,
            metadata={"patient_id": PATIENT_ID},
        )
        if self._memory_session is None:
            # There is deliberately no cloud fallback here. push_index() marks the
            # index as using custom embeddings, after which plain-text cloud
            # queries against it raise -- so a cloud write would succeed and then
            # be unreadable. Failing honestly beats writing somewhere we cannot
            # read back.
            logger.error("remember_fact: no memory session; fact not stored")
            return (
                "I'm sorry, I can't hold on to that at the moment. "
                "It would be worth telling Sarah instead."
            )

        started = time.perf_counter()
        await self._memory_session.add_docs([doc])
        # Guarded at the call site as well as inside journal_append. The fact is
        # already in the session by this point; losing the crash insurance is bad,
        # but failing to remember at all because the disk is full is worse.
        try:
            journal_append(doc)
        except Exception:
            logger.exception("journalling failed; fact is in memory but not on disk")
        self._memory_dirty = True
        logger.info(
            "remember_fact: local write in %.2fms",
            (time.perf_counter() - started) * 1000,
        )
        return "Got it, I'll remember that."

    @function_tool()
    async def recall_facts(self, context: RunContext, query: str) -> str:
        """Recall facts this user shared earlier, scoped to them.

        Use when answering depends on something the user told you before
        (their name, role, project, or preferences).

        Args:
            query: What you want to recall about the user.
        """
        options = QueryOptions(
            top_k=5,
            filter={
                "field": "patient_id",
                "condition": {"$eq": PATIENT_ID},
            },
        )
        # Reads go through the session for two reasons: facts written this turn
        # exist only in the local copy until on_exit pushes them, and the session
        # holds the embedding model, so it can answer text queries against an
        # index that push_index() has marked as using custom embeddings. The
        # cloud client cannot.
        if self._memory_session is None:
            logger.error("recall_facts: no memory session; cannot read memory")
            return "I'm sorry, I can't call anything to mind just now."

        result = await self._memory_session.query(query, options)
        await self._publish_moss_context(query, result)

        docs = getattr(result, "docs", None) or []
        facts = [(getattr(d, "text", "") or "").strip() for d in docs]
        facts = [f for f in facts if f]
        if not facts:
            return "I don't have anything remembered for you yet."
        return "\n".join(facts)


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    # Pull the knowledge index in before any caller arrives. Best-effort: if this
    # fails, get_shared_client() still loads it lazily on the first conversation.
    try:
        asyncio.run(get_shared_client())
    except Exception:
        logger.exception("Prewarm of Moss index failed; will load on first use")


server.setup_fnc = prewarm


# Keep the registered dispatch name as "agent-py": the frontend (Task 6) sets
# AGENT_NAME=agent-py to dispatch explicitly to this worker. Do not rename.
@server.rtc_session(agent_name="agent-py")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Identify the user from agent dispatch metadata. The frontend packs
    # {"user_id": ...} into ctx.job.metadata; console mode has none, so we fall
    # back to DEFAULT_USER_ID. Parsed before ctx.connect() to stay off the
    # connection critical path.
    user_id = DEFAULT_USER_ID
    if ctx.job.metadata:
        try:
            meta = json.loads(ctx.job.metadata)
            user_id = meta.get("user_id", DEFAULT_USER_ID)
        except json.JSONDecodeError:
            logger.warning("ctx.job.metadata was not valid JSON; using default user_id")

    # Set up a voice AI pipeline using LiveKit Inference and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        # nova-3-medical over plain nova-3: this agent talks about drug names,
        # and the general model mis-heard them ("Take any medication today",
        # "allergic to vitamin" / "D?" split across two utterances). Mis-hearing a
        # medicine matters more here than a few milliseconds of latency.
        #
        # interim_results is off by default; without it Deepgram emits only a
        # final transcript. Measured with it on: interims arrive on a fixed
        # ~1030ms tick against a median utterance of 526ms, so most questions end
        # before the first useful one. Speculative search is therefore a bonus for
        # long utterances, not the main mechanism -- see on_user_turn_completed.
        #
        # Deepgram Flux was trialled and dropped: its eager end-of-turn signal is
        # promising but we could not get a session to connect, and nova-3 has a
        # working track record here.
        # AssemblyAI rather than Deepgram: LiveKit Inference started returning
        # 429 Too Many Requests for the Deepgram models on this account, which
        # killed every voice session ("failed to recognize speech after 3
        # attempts"). Different provider, different quota. Its streaming model
        # emits interim transcripts by default, so the speculative path still
        # works; the Deepgram-specific extra_kwargs do not apply here and have
        # been dropped.
        stt=inference.STT(model="assemblyai/universal-streaming"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        vad=ctx.proc.userdata["vad"],
        # Endpointing is left at its defaults deliberately.
        #
        # Bill pauses mid-sentence, and the agent splits his questions in two --
        # "what medications should I" and "take in the morning with my breakfast"
        # arrive as separate turns. Raising endpointing min_delay to 1.2s was
        # tried and reverted: it did not merge the fragments (his pauses are
        # longer than that) and it added roughly three seconds of silence before
        # every reply, which is far more damaging than the fragmentation. One
        # measured turn waited 3960ms after the speaker finished at ~845ms.
        #
        # The semantic turn detector (MultilingualModel, above) should be holding
        # grammatically incomplete turns open on its own; understanding why it is
        # not is the next thing to look at, rather than fighting it with timeouts.
        #
        # preemptive_generation moved in here from its own argument, which is
        # deprecated in favour of turn_handling.
        # turn_detection MUST live inside turn_handling. When turn_handling is
        # given at all, AgentSession reads turn detection from it and ignores the
        # deprecated top-level argument outright:
        #
        #     raw_turn_detection = turn_handling.get("turn_detection", None)
        #
        # Passing turn_handling for preemptive_generation while leaving
        # turn_detection outside silently disabled the semantic detector, and
        # that is what was cutting Bill off mid-sentence -- nothing was left to
        # judge whether he had finished a thought, so any pause ended his turn.
        # It fails quietly: no warning, no error, just a worse conversation.
        turn_handling={
            "turn_detection": MultilingualModel(),
            "preemptive_generation": {"enabled": True},
        },
    )

    assistant = Assistant(room=ctx.room, user_id=user_id)

    # Let the browser turn the cloud-latency simulation on and off mid-call.
    # Doing it live is the whole value: the same agent, the same question, the
    # only change being what a lookup costs. Describing that difference is easy
    # to disbelieve; hearing it is not.
    def _on_data(packet) -> None:
        try:
            raw = getattr(packet, "data", None)
            if not raw:
                return
            message = json.loads(bytes(raw).decode("utf-8"))
            if message.get("type") != "moss_simulate_cloud":
                return
            enabled = bool(message.get("enabled"))
            assistant._simulate_cloud_latency = enabled
            logger.info(
                "cloud latency simulation %s (%.0fms per lookup)",
                "ON" if enabled else "OFF",
                SIMULATED_CLOUD_LATENCY_S * 1000 if enabled else 0,
            )
        except Exception:
            logger.exception("could not handle data message from the browser")

    ctx.room.on("data_received", _on_data)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=assistant,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()

    # Greet the user once connected. Triggered here (not in Agent.on_enter) per
    # the documented LiveKit pattern so the greeting runs against a connected
    # room and on_enter stays deterministic for the test suite.
    await session.generate_reply(
        instructions=(
            "Greet Bill warmly by name in one short sentence, as someone who "
            "already knows him would. Do not introduce yourself or explain what "
            "you can do. Then ask how he is doing today."
        )
    )


if __name__ == "__main__":
    cli.run_app(server)
