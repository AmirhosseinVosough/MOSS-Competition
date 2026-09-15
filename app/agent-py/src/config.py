"""Tuning values for the care agent, with the reasoning behind each.

Everything here was chosen from a measurement or a failure, not a guess. The
numbers are kept together so a change can be made deliberately rather than
discovered in the middle of eight hundred lines of behaviour.
"""

import os
import pathlib

from dotenv import load_dotenv

# Loaded here rather than in agent.py so that importing config is sufficient
# to have the values. Otherwise every module that reads a setting has to be
# imported after a load_dotenv() call sitting somewhere else, which is both
# fragile and the reason the imports in agent.py were out of order.
load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env.local")

# Moss index names (overridable via env so create_index.py and the agent stay in
# sync). `knowledge` holds Bill's care notes; `memory` is what the agent learns
# while talking. See src/create_index.py.
KNOWLEDGE_INDEX = os.getenv("MOSS_INDEX_NAME", "knowledge")
MEMORY_INDEX = os.getenv("MOSS_MEMORY_INDEX_NAME", "memory")

# Whose care records these are. Memory is scoped to the person being cared for,
# not to the browser that happens to be connected: Bill may speak from more than
# one device, and Sarah must see the same history he does. The starter's
# per-browser user_id was right for a multi-tenant docs bot and wrong here.
PATIENT_ID = os.getenv("PATIENT_ID", "bill")

# Fallback identity used only when ctx.job.metadata is absent (e.g. when running
# `uv run src/agent.py console`). Used for logging and dispatch; it no longer
# scopes memory.
DEFAULT_USER_ID = "user_1"

# Categories that are small, bounded, and unsafe to sample. Ranking exists to
# trim large result sets; these have nothing to trim, and picking a "best"
# allergy is how a severe one goes unmentioned. list_care_category returns every
# document in one of these.
SWEEPABLE_CATEGORIES = (
    "allergy",
    "medication",
    "appointment",
    "contact",
    "medical_alert",
)

# Measured on the 41-document care index: pure semantic beat every hybrid
# setting (6/6 correct vs 5/6), and keyword-leaning pulled in unrelated
# documents.
SEARCH_ALPHA = 1.0

# A partial shorter than this carries no searchable content -- the first interim
# of an utterance routinely arrives empty or one word long.
SPECULATIVE_MIN_CHARS = 10

# Interims measured ~1030ms apart, so this rarely binds; it exists so a chattier
# STT model cannot flood the retrieval path.
SPECULATIVE_MIN_GAP_MS = 350

# Demo switch: pretend every lookup crosses a network to a hosted vector
# database. 300ms is the middle of the 200-500ms range Moss cites for a remote
# round trip. Measured locally we are at 5-9ms including embedding generation,
# so this is roughly a 50x handicap -- which is the point. The claim that fast
# retrieval changes what is possible is easy to assert and hard to believe; this
# lets someone hear the difference instead of reading it.
SIMULATED_CLOUD_LATENCY_S = 0.300

# How long the gate may spend verifying one sentence before giving up and
# letting it through. Retrieval measures ~5ms; this is a hang guard, not a
# budget.
GATE_TIMEOUT_S = 0.5

# Said instead of an unsupported claim.
GATE_HEDGE = (
    "I am not certain about that one, so I would rather not say. "
    "It is worth checking with Sarah."
)
