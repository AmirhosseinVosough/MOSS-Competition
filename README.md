# Moss latency benchmark

Measures whether Moss is fast enough locally to retrieve **many times per turn** —
the premise the whole project rests on.

## Setup

```bash
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env      # add MOSS_PROJECT_ID / MOSS_PROJECT_KEY from portal.usemoss.dev
```

## Run

```bash
./.venv/bin/python bench/bench.py
```

Tuning:

```bash
DOCS=100000 ./.venv/bin/python bench/bench.py    # match Moss's published 100K claim
MODEL=moss-mediumlm ./.venv/bin/python bench/bench.py   # quality/latency tradeoff
```

## What it measures

| Stage | Question it answers |
|---|---|
| index build | one-time cloud cost |
| `load_index()` cold | your **deploy cold-start budget** |
| query p50/p95/p99 | steady-state latency, embedding generated per query |
| precomputed embedding | how much of a query is embedding generation |
| burst x6 | cost of firing a query on **every ASR partial transcript** |
| session write | local `add_docs` latency (no cloud job polling) — the agent-memory path |

Ends with a **GO / PARTIAL / RETHINK** verdict against a 700ms voice turn budget.
Results are written to `results/*.json` — keep them, they are the PRD evidence.

> Python 3.12 is used deliberately: the system Python here is 3.15.0a5, an alpha,
> and `moss` ships a native Rust core.

A JavaScript port of the same benchmark lives at `bench/bench.js` (`npm run bench`).
The JS SDK additionally exposes `saveToDisk()` / `loadFromDisk()` on sessions, which
the Python SDK does not.









Function	What it does
create_index()	Upload your documents; Moss turns them into a searchable index
load_index()	Download that index into your app's memory — this is what makes queries sub-10ms
query()	Ask a question in plain language, get back the most relevant documents
session()	A local index you can add to instantly, without a cloud round-trip
add_docs() / delete_docs()	Change what's in the index
query_multi_index()	Search several indexes in one call
And some options you pass into those functions:

top_k — how many results to return
filter — narrow by metadata (e.g. only this patient's records)
alpha — blend semantic vs. keyword matching
embedding — supply your own vector to skip a step and go faster
auto_refresh — automatically pull in updates others have made






bench/bench.py     ← the speed test (Python, the one to run)
bench/bench.js     ← same test in JavaScript, backup
README.md          ← how to run it
.env.example       ← where your Moss keys go




What we're making
A voice assistant for elderly people. They talk to it out loud. It answers questions about their medications, appointments, and family. Their kids can add notes to it from their phone, and the assistant knows about them right away.

Why we use Moss for it
When an AI needs to look something up, that lookup normally takes about a third of a second. Moss does it in about 5 thousandths of a second. Roughly 60 times faster.

That speed lets us do two things nobody else can:

Start looking up the answer while the person is still talking. By the time they finish their sentence, the answer is already sitting there waiting. No pause.
Fact-check every sentence before the assistant says it out loud. If the assistant is about to say "take two pills," we check that against the real medical notes first. If it's not in there, we stop it. AI making up a wrong medication dose is dangerous — this prevents it.
Both of these only work because lookups are nearly free. If each one cost a third of a second, you could only afford to do it once. That's the whole reason this project needs Moss and not something else. That's also what the judges will be checking for.

The steps, start to finish
Step 1 — Get your keys (you, 5 minutes) ← we are here

Go to portal.usemoss.dev. Make an account. Make a project. Copy the two codes it gives you. Paste them into the .env file. Tell me when it's done.

Step 2 — Speed test (me, 5 minutes)

I run the test I already wrote. It tells us if Moss is actually as fast as they claim on your laptop. If yes, we build the voice assistant. If no, we switch to a simpler backup plan. This is why we do it first — before wasting days building the wrong thing.

Step 3 — Get it talking (me, ~3 days)

You talk into your laptop mic, it talks back. Ugly, but working.

Step 4 — Add the two clever bits (me, ~3 days)

The "search while they're still talking" part, and the "fact-check before speaking" part. This is the part that wins or loses the competition, so it gets the most time.

Step 5 — Make the speed visible (me, ~2 days)

A screen showing the lookups happening live, with a switch that slows it down to normal speed so judges can watch it get worse. Judges need to see the advantage, not be told about it.

Step 6 — Put it online (me, ~1 day)

So judges can click a link and use it themselves.

Step 7 — The paperwork (both of us, ~2 days)

The competition wants four things besides the code: a diagram of how it works, a document explaining what it does, a video demo, and the GitHub link. I write drafts, you review.

Step 8 — Submit. Deadline is September 20th. That plan finishes around the 18th, leaving two days of slack for when something breaks. Something always breaks.

What I need from you, ever
Honestly, not much:

Now: the two codes from the Moss website
Later: your voice for the demo video, and a yes/no on decisions when I ask



----------------


first we used 200 synthetic documents to evaluater the bencmarks and the speed of moss which we found out it is super fast which took 3ms for a single sneetce to retribve it while vector db takes longer,
Three, plus the demo one you can ignore.
------------------
then we gotta mek indexs:
1. Care facts — the source of truth


"Metformin 500mg, twice daily, with meals"
"Cardiology appointment: Tuesday Sept 15, 2:30pm, Dr. Patel"
"Allergic to penicillin"
Changes rarely. This is the only one the fact-checker is allowed to check against.

2. Life context — who they are


"Daughter Sarah calls on Sunday evenings"
"Prefers to be called Bill, not William"
"Walks in the garden after breakfast"
Makes the assistant feel like it knows them. Not used for verification.

3. Conversation memory — built live while talking


"Sept 8: said his knee was hurting again"
"Sept 8: asked twice about Tuesday's appointment"

---------
search_knowledge is a function tool, so the AI has to decide to call it, wait, then generate. That's two AI round-trips before it speaks.
remember_fact uses the cloud write path then reloads the whole index — slow, exactly the weakness I flagged earlier. SessionIndex fixes it.
_publish_moss_context already sends time_taken_ms to the frontend — our latency display is half-built already.