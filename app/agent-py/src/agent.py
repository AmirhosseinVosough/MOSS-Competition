import contextlib
import json
import logging
import os
import textwrap
import time
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
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
from livekit.plugins import ai_coustics, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from moss import DocumentInfo, MossClient, QueryOptions

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Moss index names (overridable via env so create_index.py and the agent
# stay in sync). `knowledge` backs RAG; `memory` is the per-user agentic
# memory store. See agent-py/src/create_index.py.
KNOWLEDGE_INDEX = os.getenv("MOSS_INDEX_NAME", "knowledge")
MEMORY_INDEX = os.getenv("MOSS_MEMORY_INDEX_NAME", "memory")

# Fallback identity used only when ctx.job.metadata is absent (e.g. when
# running `uv run src/agent.py console`). The frontend provides a real
# per-browser user_id via agent dispatch metadata.
DEFAULT_USER_ID = "user_1"

# Categories that are small, bounded, and unsafe to sample. Ranking exists to trim
# large result sets; these have nothing to trim, and picking a "best" allergy is how
# a severe one goes unmentioned. list_care_category returns every document in one.
SWEEPABLE_CATEGORIES = ("allergy", "medication", "appointment", "contact", "medical_alert")

# Measured on the 41-document care index: pure semantic beat every hybrid setting
# (6/6 correct vs 5/6), and keyword-leaning pulled in unrelated documents.
SEARCH_ALPHA = 1.0


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
        self._moss = MossClient(
            os.getenv("MOSS_PROJECT_ID"), os.getenv("MOSS_PROJECT_KEY")
        )
        self._indexes_loaded = False
        # Local, mutable view of the memory index. Writes land in process memory
        # instead of triggering a cloud rebuild + full re-download, which measured
        # 4257ms per fact (bench/bench_memory.py). Local writes measure ~4.5ms.
        # Falls back to the cloud path if the session cannot be created.
        self._memory_session = None
        self._memory_dirty = False

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
        if not self._indexes_loaded:
            try:
                await self._moss.load_index(KNOWLEDGE_INDEX)
                self._indexes_loaded = True
                logger.info("Loaded Moss knowledge index '%s'", KNOWLEDGE_INDEX)
            except Exception:
                logger.exception("Failed to preload knowledge index; will retry on use")

        # Open a writable local session over the memory index. session() adopts the
        # stored model, so model_id is deliberately omitted -- passing one that
        # disagrees with the index raises.
        if self._memory_session is None:
            try:
                self._memory_session = await self._moss.session(MEMORY_INDEX)
                logger.info("Opened local Moss session for '%s'", MEMORY_INDEX)
            except Exception:
                logger.exception(
                    "Could not open memory session; falling back to cloud writes"
                )
                try:
                    await self._moss.load_index(MEMORY_INDEX)
                except Exception:
                    logger.exception("Failed to load memory index for fallback reads")

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
            except Exception:
                logger.exception("Failed to push memory session to cloud")

    async def _publish_moss_context(self, query: str, result) -> None:
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
                    "time_taken_ms": getattr(result, "time_taken_ms", None),
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
        result = await self._moss.query(
            KNOWLEDGE_INDEX, query, QueryOptions(top_k=3, alpha=SEARCH_ALPHA)
        )
        await self._publish_moss_context(query, result)

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
        result = await self._moss.query(
            KNOWLEDGE_INDEX,
            category,
            QueryOptions(
                top_k=25,
                filter={"field": "category", "condition": {"$eq": category}},
            ),
        )
        await self._publish_moss_context(f"all {category} notes", result)

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
            id=f"{self._user_id}-{uuid.uuid4()}",
            text=fact,
            metadata={"user_id": self._user_id},
        )
        started = time.perf_counter()

        if self._memory_session is not None:
            # Local write: searchable immediately, no rebuild, no re-download.
            await self._memory_session.add_docs([doc])
            self._memory_dirty = True
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info("remember_fact: local write in %.2fms", elapsed_ms)
        else:
            # Fallback path: the original cloud write, kept so a session failure
            # degrades the agent's latency rather than breaking its memory.
            await self._moss.add_docs(MEMORY_INDEX, [doc])
            try:
                await self._moss.load_index(MEMORY_INDEX)
            except Exception:
                logger.exception("Failed to reload memory index after write")
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.warning("remember_fact: cloud fallback write in %.0fms", elapsed_ms)

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
                "field": "user_id",
                "condition": {"$eq": self._user_id},
            },
        )
        # Read through the session when we have one: facts written this turn are
        # only present in the local copy until on_exit pushes them.
        if self._memory_session is not None:
            result = await self._memory_session.query(query, options)
        else:
            result = await self._moss.query(MEMORY_INDEX, query, options)
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
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(room=ctx.room, user_id=user_id),
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
