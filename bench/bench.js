/**
 * Moss latency benchmark.
 *
 * Answers the only question that matters before we build:
 * is local query latency actually low enough to retrieve MANY times per turn?
 *
 * Usage:  cp .env.example .env  (fill in creds)  &&  npm run bench
 * Env:    DOCS=2000  TOPK=5  ITERS=200  MODEL=moss-minilm
 */
import { MossClient } from "@moss-dev/moss";
import { existsSync, readFileSync, mkdirSync, writeFileSync } from "node:fs";

// ---------- config ----------
if (existsSync(".env")) {
  for (const line of readFileSync(".env", "utf8").split("\n")) {
    const m = line.match(/^\s*([A-Z_]+)\s*=\s*(.*)\s*$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2].trim();
  }
}
const PROJECT_ID = process.env.MOSS_PROJECT_ID;
const PROJECT_KEY = process.env.MOSS_PROJECT_KEY;
const DOCS = Number(process.env.DOCS ?? 2000);
const TOPK = Number(process.env.TOPK ?? 5);
const ITERS = Number(process.env.ITERS ?? 200);
const MODEL = process.env.MODEL ?? "moss-minilm";
const INDEX = `bench-${MODEL}-${DOCS}`;

if (!PROJECT_ID || !PROJECT_KEY) {
  console.error("Missing MOSS_PROJECT_ID / MOSS_PROJECT_KEY.");
  console.error("Sign up at https://portal.usemoss.dev, then: cp .env.example .env");
  process.exit(1);
}

// ---------- helpers ----------
const pct = (arr, p) => {
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
};
const stats = (arr) => ({
  n: arr.length,
  min: +Math.min(...arr).toFixed(2),
  p50: +pct(arr, 50).toFixed(2),
  p95: +pct(arr, 95).toFixed(2),
  p99: +pct(arr, 99).toFixed(2),
  max: +Math.max(...arr).toFixed(2),
  mean: +(arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(2),
});
const time = async (fn) => {
  const t = performance.now();
  const out = await fn();
  return [performance.now() - t, out];
};
const row = (label, s) =>
  `  ${label.padEnd(30)} p50 ${String(s.p50).padStart(7)}ms   p95 ${String(s.p95).padStart(7)}ms   p99 ${String(s.p99).padStart(7)}ms   max ${String(s.max).padStart(8)}ms`;

// Synthetic corpus in the shape of a field-service / support knowledge base.
const SUBJECTS = ["pressure valve", "coolant pump", "relay board", "intake manifold",
  "hydraulic line", "thermal sensor", "drive belt", "control module", "fuel injector", "brake caliper"];
const ACTIONS = ["torque specification is", "requires inspection every", "must be replaced after",
  "operates at a nominal", "fails most often due to", "should be calibrated to"];
const makeDocs = (n) =>
  Array.from({ length: n }, (_, i) => ({
    id: `doc-${i}`,
    text: `Unit ${1000 + i}: the ${SUBJECTS[i % SUBJECTS.length]} ${ACTIONS[i % ACTIONS.length]} ${20 + (i % 180)} units under standard operating load. Refer to service bulletin SB-${2000 + (i % 400)} for the full procedure and safety interlocks.`,
    metadata: { region: ["north", "south", "east", "west"][i % 4], sb: `SB-${2000 + (i % 400)}` },
  }));

const QUERIES = [
  "what is the torque spec on the pressure valve",
  "how often should the coolant pump be inspected",
  "why does the relay board fail",
  "calibration procedure for the thermal sensor",
  "when do I replace the drive belt",
  "nominal operating load for the hydraulic line",
];
// Prefixes of a spoken sentence — simulates ASR partial transcripts.
const PARTIALS = [
  "what", "what is the", "what is the torque", "what is the torque spec",
  "what is the torque spec on", "what is the torque spec on the pressure valve",
];

// ---------- main ----------
const client = new MossClient(PROJECT_ID, PROJECT_KEY);
const results = { config: { DOCS, TOPK, ITERS, MODEL, index: INDEX }, at: new Date().toISOString() };

console.log(`\n=== Moss latency benchmark ===`);
console.log(`index=${INDEX}  docs=${DOCS}  topK=${TOPK}  iters=${ITERS}  model=${MODEL}\n`);

try {
  // --- 1. build (one-time, cloud) ---
  const existing = await client.listIndexes().catch(() => []);
  if (!existing.some((i) => i.name === INDEX)) {
    console.log(`[1/6] building index (${DOCS} docs, cloud-side, one-time)...`);
    const [ms] = await time(() =>
      client.createIndex(INDEX, makeDocs(DOCS), {
        modelId: MODEL,
        onProgress: (p) => process.stdout.write(`\r      ${p.currentPhase ?? p.status} ${p.progress ?? 0}%   `),
      })
    );
    console.log(`\n      built in ${(ms / 1000).toFixed(1)}s`);
    results.buildMs = Math.round(ms);
  } else {
    console.log(`[1/6] index already exists, skipping build`);
  }

  // --- 2. cold load (this is your deploy cold-start cost) ---
  console.log(`[2/6] loadIndex() cold...`);
  const [loadMs] = await time(() => client.loadIndex(INDEX));
  console.log(`      loaded in ${loadMs.toFixed(0)}ms  <-- cold-start budget on deploy`);
  results.loadMs = Math.round(loadMs);

  // --- 3. steady-state query latency (includes embedding generation) ---
  console.log(`[3/6] query latency, ${ITERS} iters (embedding generated per query)...`);
  for (let i = 0; i < 20; i++) await client.query(INDEX, QUERIES[i % QUERIES.length], { topK: TOPK }); // warm
  const qLat = [], qSelf = [];
  for (let i = 0; i < ITERS; i++) {
    const [ms, r] = await time(() => client.query(INDEX, QUERIES[i % QUERIES.length], { topK: TOPK }));
    qLat.push(ms);
    if (typeof r?.timeTakenInMs === "number") qSelf.push(r.timeTakenInMs);
  }
  results.query = stats(qLat);
  if (qSelf.length) results.querySelfReported = stats(qSelf);

  // --- 4. precomputed-embedding path (the speculative-retrieval hot path) ---
  // If embedding generation dominates, reusing a vector makes repeat queries far cheaper.
  console.log(`[4/6] precomputed-embedding query path...`);
  let embLat = null;
  const probe = await client.query(INDEX, QUERIES[0], { topK: 1 });
  const dim = probe?.docs?.[0]?.embedding?.length;
  if (dim) {
    const vec = probe.docs[0].embedding;
    const arr = [];
    for (let i = 0; i < ITERS; i++) {
      const [ms] = await time(() => client.query(INDEX, "", { topK: TOPK, embedding: vec }));
      arr.push(ms);
    }
    embLat = stats(arr);
    results.queryPrecomputedEmbedding = embLat;
  } else {
    console.log(`      (embeddings not returned in results; skipping — measure via your own encoder)`);
  }

  // --- 5. burst: speculative retrieval on ASR partials ---
  // The whole thesis: can we fire a query on EVERY partial transcript?
  console.log(`[5/6] burst — ${PARTIALS.length} queries per utterance, sequential + parallel...`);
  const seq = [], par = [];
  for (let i = 0; i < 30; i++) {
    const [ms] = await time(async () => {
      for (const p of PARTIALS) await client.query(INDEX, p, { topK: TOPK });
    });
    seq.push(ms);
    const [pms] = await time(() => Promise.all(PARTIALS.map((p) => client.query(INDEX, p, { topK: TOPK }))));
    par.push(pms);
  }
  results.burstSequential = stats(seq);
  results.burstParallel = stats(par);

  // --- 6. local mutable session: write latency ---
  // SessionIndex.addDocs is local (no cloud job polling) — this is the agent-memory path.
  console.log(`[6/6] SessionIndex local write + query...`);
  try {
    const session = await client.session(INDEX, MODEL);
    await session.loadIndex(INDEX);
    const wLat = [], sqLat = [];
    for (let i = 0; i < 50; i++) {
      const [ms] = await time(() =>
        session.addDocs([{ id: `live-${i}`, text: `Live note ${i}: technician reported intermittent fault on unit ${3000 + i}.` }])
      );
      wLat.push(ms);
      const [qms] = await time(() => session.query("intermittent fault report", { topK: TOPK }));
      sqLat.push(qms);
    }
    results.sessionWrite = stats(wLat);
    results.sessionQueryAfterWrite = stats(sqLat);
    await session.close();
  } catch (e) {
    console.log(`      session path unavailable: ${e.message}`);
    results.sessionError = e.message;
  }

  // ---------- report ----------
  console.log(`\n${"=".repeat(96)}\nRESULTS  (${DOCS} docs, ${MODEL})\n${"=".repeat(96)}`);
  console.log(`  index build                    ${results.buildMs ? (results.buildMs / 1000).toFixed(1) + "s" : "(cached)"}`);
  console.log(`  loadIndex() cold               ${results.loadMs}ms   <-- deploy cold-start`);
  console.log("");
  console.log(row("query (with embedding)", results.query));
  if (results.querySelfReported) console.log(row("  ^ SDK self-reported", results.querySelfReported));
  if (embLat) console.log(row("query (precomputed vec)", embLat));
  console.log("");
  console.log(row(`burst x${PARTIALS.length} sequential`, results.burstSequential));
  console.log(row(`burst x${PARTIALS.length} parallel`, results.burstParallel));
  if (results.sessionWrite) {
    console.log("");
    console.log(row("session addDocs (local)", results.sessionWrite));
    console.log(row("session query after write", results.sessionQueryAfterWrite));
  }

  // ---------- verdict ----------
  const p99 = results.query.p99;
  const burst = results.burstSequential.p99;
  console.log(`\n${"-".repeat(96)}\nVERDICT\n${"-".repeat(96)}`);
  console.log(`  Single query p99:            ${p99}ms`);
  console.log(`  Full utterance (${PARTIALS.length} partials): ${burst}ms sequential / ${results.burstParallel.p99}ms parallel`);
  const budget = 700;
  console.log(`  Share of a ${budget}ms voice turn: ${((burst / budget) * 100).toFixed(1)}%`);
  console.log("");
  if (p99 < 15 && burst < 100) {
    console.log(`  GO. Speculative retrieval is viable — you can query on every ASR partial`);
    console.log(`  and still spend <15% of the turn budget. Build the voice agent.`);
  } else if (p99 < 50) {
    console.log(`  PARTIAL. Fast, but not "free". Retrieve on every 2nd-3rd partial, not every one.`);
    console.log(`  Consider the precomputed-embedding path, or moss-minilm over mediumlm.`);
  } else {
    console.log(`  RETHINK. p99 ${p99}ms is not in "retrieve many times per turn" territory.`);
    console.log(`  Verify loadIndex() succeeded — if queries are hitting cloud fallback you'd see 100-500ms.`);
    console.log(`  Pivot toward the local-first/browser idea where offline capability is the story.`);
  }

  mkdirSync("results", { recursive: true });
  const out = `results/bench-${MODEL}-${DOCS}-${Date.now()}.json`;
  writeFileSync(out, JSON.stringify(results, null, 2));
  console.log(`\n  Saved -> ${out}   (keep these numbers, they are your PRD evidence)\n`);
} catch (err) {
  console.error(`\nFAILED: ${err?.message ?? err}`);
  if (err?.stack) console.error(err.stack.split("\n").slice(1, 4).join("\n"));
  process.exit(1);
} finally {
  await client.close().catch(() => {});
}
