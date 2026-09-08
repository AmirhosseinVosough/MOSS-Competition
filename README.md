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
