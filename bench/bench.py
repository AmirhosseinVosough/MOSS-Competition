"""
Moss latency benchmark.

Answers the only question that matters before we build:
is local query latency low enough to retrieve MANY times per turn?

Usage:  cp .env.example .env   (fill in creds)   &&   ./.venv/bin/python bench/bench.py
Env:    DOCS=2000  TOPK=5  ITERS=200  MODEL=moss-minilm
"""

import asyncio
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

from moss import DocumentInfo, MossClient, QueryOptions

# ---------- config ----------
ROOT = Path(__file__).resolve().parent.parent
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        m = re.match(r"^\s*([A-Z_]+)\s*=\s*(.*)\s*$", line)
        if m and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2).strip()

PROJECT_ID = os.environ.get("MOSS_PROJECT_ID")
PROJECT_KEY = os.environ.get("MOSS_PROJECT_KEY")
DOCS = int(os.environ.get("DOCS", 2000))
TOPK = int(os.environ.get("TOPK", 5))
ITERS = int(os.environ.get("ITERS", 200))
MODEL = os.environ.get("MODEL", "moss-minilm")
# Point at an existing index (e.g. the Moss starter) to benchmark for free.
INDEX = os.environ.get("INDEX") or f"bench-{MODEL}-{DOCS}"
OWN_INDEX = INDEX.startswith("bench-")
# Never write into an index we did not create unless explicitly allowed.
ALLOW_WRITES = OWN_INDEX or os.environ.get("ALLOW_SESSION_WRITES") == "1"

if not PROJECT_ID or not PROJECT_KEY:
    sys.exit(
        "Missing MOSS_PROJECT_ID / MOSS_PROJECT_KEY.\n"
        "Sign up at https://portal.usemoss.dev, then: cp .env.example .env"
    )


# ---------- helpers ----------
def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int((p / 100) * len(s)))]


def stats(values):
    return {
        "n": len(values),
        "min": round(min(values), 2),
        "p50": round(pct(values, 50), 2),
        "p95": round(pct(values, 95), 2),
        "p99": round(pct(values, 99), 2),
        "max": round(max(values), 2),
        "mean": round(statistics.fmean(values), 2),
    }


async def timed(coro):
    t = time.perf_counter()
    out = await coro
    return (time.perf_counter() - t) * 1000, out


def row(label, s):
    return (
        f"  {label:<30} p50 {s['p50']:>7}ms   p95 {s['p95']:>7}ms   "
        f"p99 {s['p99']:>7}ms   max {s['max']:>8}ms"
    )


# Synthetic corpus shaped like a field-service / support knowledge base.
SUBJECTS = ["pressure valve", "coolant pump", "relay board", "intake manifold",
            "hydraulic line", "thermal sensor", "drive belt", "control module",
            "fuel injector", "brake caliper"]
ACTIONS = ["torque specification is", "requires inspection every", "must be replaced after",
           "operates at a nominal", "fails most often due to", "should be calibrated to"]
REGIONS = ["north", "south", "east", "west"]


def make_docs(n):
    return [
        DocumentInfo(
            id=f"doc-{i}",
            text=(
                f"Unit {1000 + i}: the {SUBJECTS[i % len(SUBJECTS)]} "
                f"{ACTIONS[i % len(ACTIONS)]} {20 + (i % 180)} units under standard "
                f"operating load. Refer to service bulletin SB-{2000 + (i % 400)} for "
                f"the full procedure and safety interlocks."
            ),
            metadata={"region": REGIONS[i % 4], "sb": f"SB-{2000 + (i % 400)}"},
        )
        for i in range(n)
    ]


QUERIES = [
    "what is the torque spec on the pressure valve",
    "how often should the coolant pump be inspected",
    "why does the relay board fail",
    "calibration procedure for the thermal sensor",
    "when do I replace the drive belt",
    "nominal operating load for the hydraulic line",
]
# Growing prefixes of a spoken sentence — simulates ASR partial transcripts.
PARTIALS = [
    "what", "what is the", "what is the torque", "what is the torque spec",
    "what is the torque spec on", "what is the torque spec on the pressure valve",
]


def fake_vec(seed, dim):
    """Deterministic unit vector. Content is irrelevant to search latency."""
    import math, random
    r = random.Random(seed)
    v = [r.gauss(0, 1) for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


async def probe_text_queries(client):
    """Some indexes (dashboard-created) refuse text queries and demand vectors."""
    try:
        await client.query(INDEX, "test", QueryOptions(top_k=1))
        return True
    except Exception as e:
        if "explicit query embeddings" in str(e):
            return False
        raise


async def detect_embedding_dim(client):
    """Results don't expose embeddings, so probe for the expected vector width."""
    for dim in (384, 768, 512, 1024, 256):
        try:
            await client.query(INDEX, "", QueryOptions(top_k=1, embedding=[0.01] * dim))
            return dim
        except Exception as e:
            found = re.findall(r"\b(\d{3,4})\b", str(e))
            for f in found:
                if 128 <= int(f) <= 4096 and int(f) != dim:
                    try:
                        await client.query(
                            INDEX, "", QueryOptions(top_k=1, embedding=[0.01] * int(f))
                        )
                        return int(f)
                    except Exception:
                        pass
    return None


async def main():
    client = MossClient(PROJECT_ID, PROJECT_KEY)
    results = {
        "config": {"docs": DOCS, "top_k": TOPK, "iters": ITERS, "model": MODEL, "index": INDEX},
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    print("\n=== Moss latency benchmark ===")
    print(f"index={INDEX}  docs={DOCS}  topK={TOPK}  iters={ITERS}  model={MODEL}\n")

    # --- 1. build (one-time, cloud) ---
    existing = []
    try:
        existing = await client.list_indexes()
    except Exception:
        pass
    if not any(getattr(i, "name", None) == INDEX for i in existing):
        print(f"[1/6] building index ({DOCS} docs, cloud-side, one-time)...")
        ms, _ = await timed(client.create_index(INDEX, make_docs(DOCS), MODEL))
        print(f"      built in {ms / 1000:.1f}s")
        results["build_ms"] = round(ms)
    else:
        print("[1/6] index already exists, skipping build")

    # --- 2. cold load (this is your deploy cold-start cost) ---
    print("[2/6] load_index() cold...")
    load_ms, _ = await timed(client.load_index(INDEX))
    print(f"      loaded in {load_ms:.0f}ms  <-- cold-start budget on deploy")
    results["load_ms"] = round(load_ms)

    text_ok = await probe_text_queries(client)
    dim = None
    if not text_ok:
        dim = await detect_embedding_dim(client)
        print(f"      NOTE: this index requires pre-computed vectors (dim={dim}).")
        print(f"      Measuring pure search speed, excluding embedding generation.")
        results["mode"] = "embedding_only"
        results["embedding_dim"] = dim

    # --- 3. steady-state query latency ---
    print(f"[3/6] query latency, {ITERS} iters...")
    def q(i):
        if text_ok:
            return client.query(INDEX, QUERIES[i % len(QUERIES)], QueryOptions(top_k=TOPK))
        return client.query(INDEX, "", QueryOptions(top_k=TOPK, embedding=fake_vec(i % 20, dim)))

    for i in range(20):  # warm
        await q(i)
    q_lat, q_self = [], []
    for i in range(ITERS):
        ms, r = await timed(q(i))
        q_lat.append(ms)
        t = getattr(r, "time_taken_ms", None)
        if isinstance(t, (int, float)):
            q_self.append(float(t))
    results["query"] = stats(q_lat)
    if q_self:
        results["query_self_reported"] = stats(q_self)

    # --- 4. precomputed-embedding path (the speculative-retrieval hot path) ---
    # If embedding generation dominates, reusing a vector makes repeat queries far cheaper.
    print("[4/6] precomputed-embedding query path...")
    emb_stats = None
    if not text_ok:
        print("      same as stage 3 in this mode; skipping.")
        dim = dim
    elif (dim := await detect_embedding_dim(client)):
        print(f"      embedding dim = {dim}")
        vec = [0.01] * dim
        arr = []
        for _ in range(ITERS):
            ms, _ = await timed(
                client.query(INDEX, "", QueryOptions(top_k=TOPK, embedding=vec))
            )
            arr.append(ms)
        emb_stats = stats(arr)
        results["query_precomputed_embedding"] = emb_stats
        results["embedding_dim"] = dim
    else:
        print("      could not determine embedding dim; skipping")

    # --- 5. burst: speculative retrieval on ASR partials ---
    # The whole thesis: can we fire a query on EVERY partial transcript?
    print(f"[5/6] burst — {len(PARTIALS)} queries per utterance, sequential + concurrent...")
    seq, par = [], []
    for _ in range(30):
        def bq(j):
            if text_ok:
                return client.query(INDEX, PARTIALS[j], QueryOptions(top_k=TOPK))
            return client.query(INDEX, "", QueryOptions(top_k=TOPK, embedding=fake_vec(j, dim)))

        t = time.perf_counter()
        for j in range(len(PARTIALS)):
            await bq(j)
        seq.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        await asyncio.gather(*[bq(j) for j in range(len(PARTIALS))])
        par.append((time.perf_counter() - t) * 1000)
    results["burst_sequential"] = stats(seq)
    results["burst_concurrent"] = stats(par)

    # --- 6. local mutable session: write latency ---
    # SessionIndex.add_docs is local (no cloud job polling) — the agent-memory path.
    print("[6/6] SessionIndex local write + query...")
    if not ALLOW_WRITES:
        print("      skipped — not our index, refusing to write into it.")
        print("      (local-only, but set ALLOW_SESSION_WRITES=1 if you want it)")
        results["session_skipped"] = "foreign index"
    try:
        if not ALLOW_WRITES:
            raise RuntimeError("skipped")
        session = await client.session(INDEX, MODEL)
        w_lat, sq_lat = [], []
        for i in range(50):
            ms, _ = await timed(
                session.add_docs([
                    DocumentInfo(
                        id=f"live-{i}",
                        text=f"Live note {i}: technician reported intermittent fault on unit {3000 + i}.",
                    )
                ])
            )
            w_lat.append(ms)
            qms, _ = await timed(
                session.query("intermittent fault report", QueryOptions(top_k=TOPK))
            )
            sq_lat.append(qms)
        results["session_write"] = stats(w_lat)
        results["session_query_after_write"] = stats(sq_lat)
    except Exception as e:
        if str(e) != "skipped":
            print(f"      session path unavailable: {e}")
            results["session_error"] = str(e)

    # ---------- report ----------
    bar = "=" * 96
    print(f"\n{bar}\nRESULTS  ({DOCS} docs, {MODEL})\n{bar}")
    build = f"{results['build_ms'] / 1000:.1f}s" if "build_ms" in results else "(cached)"
    print(f"  index build                    {build}")
    print(f"  load_index() cold              {results['load_ms']}ms   <-- deploy cold-start")
    print()
    print(row("query (with embedding)", results["query"]))
    if "query_self_reported" in results:
        print(row("  ^ SDK self-reported", results["query_self_reported"]))
    if emb_stats:
        print(row("query (precomputed vec)", emb_stats))
    print()
    print(row(f"burst x{len(PARTIALS)} sequential", results["burst_sequential"]))
    print(row(f"burst x{len(PARTIALS)} concurrent", results["burst_concurrent"]))
    if "session_write" in results:
        print()
        print(row("session add_docs (local)", results["session_write"]))
        print(row("session query after write", results["session_query_after_write"]))

    # ---------- verdict ----------
    p99 = results["query"]["p99"]
    burst = results["burst_sequential"]["p99"]
    budget = 700
    print(f"\n{'-' * 96}\nVERDICT\n{'-' * 96}")
    print(f"  Single query p99:            {p99}ms")
    print(f"  Full utterance ({len(PARTIALS)} partials): {burst}ms sequential / "
          f"{results['burst_concurrent']['p99']}ms concurrent")
    print(f"  Share of a {budget}ms voice turn: {(burst / budget) * 100:.1f}%")
    print()
    if p99 < 15 and burst < 100:
        print("  GO. Speculative retrieval is viable — you can query on every ASR partial")
        print("  and still spend <15% of the turn budget. Build the voice agent.")
    elif p99 < 50:
        print("  PARTIAL. Fast, but not 'free'. Retrieve every 2nd-3rd partial, not every one.")
        print("  Consider the precomputed-embedding path, or moss-minilm over mediumlm.")
    else:
        print(f"  RETHINK. p99 {p99}ms is not in 'retrieve many times per turn' territory.")
        print("  Verify load_index() succeeded — cloud fallback shows as 100-500ms.")
        print("  Pivot toward the local-first/browser idea where offline is the story.")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"bench-{MODEL}-{DOCS}-{int(time.time())}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved -> {out.relative_to(ROOT)}   (keep these numbers, they are your PRD evidence)\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.exit(f"\nFAILED: {e}")
