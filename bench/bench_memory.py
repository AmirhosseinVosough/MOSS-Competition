"""Compare the two ways an agent can save a fact to memory.

BEFORE - what the starter does today:
    client.add_docs(index, [doc])   # cloud rebuild, polled until done
    client.load_index(index)        # re-download the whole index

AFTER - the local session path:
    session.add_docs([doc])         # written in memory, on device

Test docs are tagged user_id="__bench__" so they never surface for a real user.

Usage:  ./.venv/bin/python bench/bench_memory.py
"""

import asyncio
import json
import os
import re
import statistics
import sys
import time
import uuid
from pathlib import Path

from moss import DocumentInfo, MossClient, QueryOptions

ROOT = Path(__file__).resolve().parent.parent
for line in (ROOT / ".env").read_text().splitlines():
    m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
    if m and m.group(1) not in os.environ:
        os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")

PROJECT_ID = os.environ.get("MOSS_PROJECT_ID")
PROJECT_KEY = os.environ.get("MOSS_PROJECT_KEY")
INDEX = os.environ.get("MOSS_MEMORY_INDEX_NAME", "memory")
MODEL = os.environ.get("MOSS_MODEL_ID", "moss-minilm")
N_CLOUD = int(os.environ.get("N_CLOUD", 3))    # cloud writes are slow + cost credits
N_LOCAL = int(os.environ.get("N_LOCAL", 25))   # local writes are free
SKIP_CLOUD = os.environ.get("SKIP_CLOUD") == "1"
# Measured on this machine 2026-09-08 (3 cloud writes against the `memory` index).
PRIOR_CLOUD = {"add": 3314.0, "reload": 1006.0, "total": 4257.0}

if not PROJECT_ID or not PROJECT_KEY:
    sys.exit("Missing Moss credentials in .env")

FACTS = [
    "Takes metformin 500mg twice daily with meals.",
    "Cardiology appointment Tuesday at 2:30pm with Dr. Patel.",
    "Allergic to penicillin.",
    "Daughter Sarah calls on Sunday evenings.",
    "Prefers to be called Bill, not William.",
]


def stats(xs):
    s = sorted(xs)
    return {
        "n": len(xs),
        "min": round(min(xs), 1),
        "p50": round(s[len(s) // 2], 1),
        "max": round(max(xs), 1),
        "mean": round(statistics.fmean(xs), 1),
    }


def doc(i):
    return DocumentInfo(
        id=f"__bench__-{uuid.uuid4()}",
        text=f"{FACTS[i % len(FACTS)]} (bench {i})",
        metadata={"user_id": "__bench__"},
    )


async def main():
    client = MossClient(PROJECT_ID, PROJECT_KEY)
    results = {"index": INDEX, "model": MODEL, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    print(f"\n=== memory write benchmark ===\nindex={INDEX}  model={MODEL}\n")

    # ---- BEFORE: cloud add_docs + full reload (what the starter does) ----
    if SKIP_CLOUD:
        print("[1/2] BEFORE - skipped, using previously measured numbers:")
        print(f"      add={PRIOR_CLOUD['add']:.0f}ms  reload={PRIOR_CLOUD['reload']:.0f}ms  total={PRIOR_CLOUD['total']:.0f}ms")
        results["before_add"] = {"p50": PRIOR_CLOUD["add"], "min": PRIOR_CLOUD["add"], "max": PRIOR_CLOUD["add"], "n": 0}
        results["before_reload"] = {"p50": PRIOR_CLOUD["reload"], "min": PRIOR_CLOUD["reload"], "max": PRIOR_CLOUD["reload"], "n": 0}
        results["before_total"] = {"p50": PRIOR_CLOUD["total"], "min": 4074.0, "max": 5591.0, "n": 3}
        results["before_source"] = "prior measurement 2026-09-08"
    print(f"[1/2] BEFORE - cloud add_docs + load_index, {N_CLOUD} writes...") if not SKIP_CLOUD else None
    print("      (each one rebuilds the index server-side, so this is slow on purpose)")
    add_ms, reload_ms, total_ms = [], [], []
    if not SKIP_CLOUD:
        await client.load_index(INDEX)
    for i in range(0 if SKIP_CLOUD else N_CLOUD):
        t0 = time.perf_counter()
        await client.add_docs(INDEX, [doc(i)])
        t1 = time.perf_counter()
        await client.load_index(INDEX)
        t2 = time.perf_counter()
        add_ms.append((t1 - t0) * 1000)
        reload_ms.append((t2 - t1) * 1000)
        total_ms.append((t2 - t0) * 1000)
        print(f"      write {i + 1}/{N_CLOUD}: add={add_ms[-1]:.0f}ms  reload={reload_ms[-1]:.0f}ms  total={total_ms[-1]:.0f}ms")
    if not SKIP_CLOUD:
        results["before_add"] = stats(add_ms)
        results["before_reload"] = stats(reload_ms)
        results["before_total"] = stats(total_ms)

    # ---- AFTER: local session write ----
    print(f"\n[2/2] AFTER - SessionIndex.add_docs (local), {N_LOCAL} writes...")
    # Omit model_id so the session adopts whatever model the stored index uses.
    session = await client.session(INDEX)
    local_ms, query_ms = [], []
    for i in range(N_LOCAL):
        t0 = time.perf_counter()
        await session.add_docs([doc(1000 + i)])
        t1 = time.perf_counter()
        local_ms.append((t1 - t0) * 1000)
        # Confirm the fact is searchable immediately, with no reload.
        t2 = time.perf_counter()
        await session.query(
            "what medication do they take",
            QueryOptions(top_k=3, filter={"field": "user_id", "condition": {"$eq": "__bench__"}}),
        )
        query_ms.append((time.perf_counter() - t2) * 1000)
    results["after_write"] = stats(local_ms)
    results["after_query"] = stats(query_ms)

    # ---- report ----
    b, a = results["before_total"], results["after_write"]
    bar = "=" * 78
    print(f"\n{bar}\nRESULT - saving one fact to memory\n{bar}")
    print(f"  BEFORE  cloud add_docs        p50 {results['before_add']['p50']:>9}ms")
    print(f"          + full reload         p50 {results['before_reload']['p50']:>9}ms")
    print(f"          TOTAL                 p50 {b['p50']:>9}ms   (range {b['min']}-{b['max']}ms)")
    print()
    print(f"  AFTER   local session write   p50 {a['p50']:>9}ms   (range {a['min']}-{a['max']}ms)")
    print(f"          searchable instantly  p50 {results['after_query']['p50']:>9}ms")
    print()
    speedup = b["p50"] / a["p50"] if a["p50"] > 0 else float("inf")
    saved = b["p50"] - a["p50"]
    print(f"  {speedup:,.0f}x faster - {saved:,.0f}ms saved every time the agent remembers something.")
    print(f"  That delay happened mid-conversation, while the user waited.")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    f = out / f"memory-{int(time.time())}.json"
    f.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved -> {f.relative_to(ROOT)}\n")


if __name__ == "__main__":
    asyncio.run(main())
