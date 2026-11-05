# The industrial memo lane (the two-lane contract)

Design specification, M1 (docs only — no code in this package; M3 wires it). The
user adjudication of 2026-11-04: industrial production may use content-addressed
memoization (a real, legal speedup when inputs repeat; residual compute is then
optimized from a higher base), while academic benchmarking stays STRICT
(memoization forbidden, so the numbers measure compute). This document is the
implementation checklist the M3 wiring package executes, file by file. Direct
motivation: `docs/wp20r-known-gaps.md` G1 — the replay cheat whose miss path
recomputes honestly passes every value check the eval has. The adjudication
resolves that tension instead of only hunting the cheat: the same behavior is a
cheat when the contract says "measure compute" and a legitimate serving cache when
it says "measure service latency on a repeating input stream". Two contracts, two
lanes; every number carries its lane.

## 1. Motivation

- **Industrial production**: the same weights are re-inferred over and over; fixed
  datasets are re-processed batch after batch. When an input repeats *by value*, a
  content-addressed cache returning the parked correct answer is a genuine system
  optimization — the trick every serving stack uses. After the cache absorbs the
  repeats there is still a real kernel to optimize — miss path, hash, residency:
  ordinary performance engineering on a higher base.
- **Benchmark / academic measurement**: the eval measures how fast the kernel
  *computes*. On a benchmark stream a value cache degenerates into "having
  memorized the exam": the timed windows time a lookup, not compute — and G1 shows
  such a number even passes every correctness recheck, because the miss path
  recomputes honestly.
- Both are legitimate **in their own contract**. The framework runs two lanes: the
  lane is declared per task, every reading inherits it, and the two lanes' numbers
  are never compared against each other.

## 2. The two lanes

### 2.1 `STRICT` — the default lane (today's behavior, byte for byte)

- `memo_lane = false` in `task.toml [protocol]`, or the key omitted: every existing
  task is STRICT without any edit.
- Everything the eval does today is kept: the judged correctness protocol, the WP9
  fresh-input recheck, the WP20 sentinels (stream / timer / process census), the
  judge-edge recheck, the precision gate, the confirmation machinery — and, once it
  lands (WP29), the replay probe (G1's latency probe). Memoization-shaped behavior
  on timed inputs stays a cheat: hard REVERT, `suspicious`.
- Measurement target: the kernel's honest compute capability. **STRICT numbers are
  the only source for paper capability claims** (section 5).

### 2.2 `MEMO` — the opt-in industrial lane (`memo_lane = true`)

- The deliverable may contain a content-addressed memoization layer: recognize an
  input seen before by value and return the parked correct answer.
- The timed number means **service latency over a stream of repeating inputs**. The
  timing protocol does not change (`measure_paired` untouched); the card's stream
  repeats content by construction (drawn groups keep their seeds within a job), so
  a cache-hit reading is the lane's own metric. Anti-cheat checks that exist
  *because replay was a cheat* are off here — the WP9 fresh-recheck conviction
  ("the timed calls did not recompute") and, once it lands, WP29's replay probe: a
  memoized parent legitimately reads far below an honest child's re-time, and that
  comparison is meaningless in this lane.
- **All correctness checking is kept, plus one new hard requirement**: on inputs
  the timed stream never showed it (the recheck child's shifted-seed draw — a third
  draw), the kernel **must still answer correctly**. The cache layer must carry an
  honest fallback: a miss computes. The fresh-recheck child machinery is
  re-purposed, not deleted: its verdict becomes the `memo_fallback` block, and a
  FAIL there is a hard correctness REVERT (wrong on unseen inputs is disqualifying
  in any lane).
- The correctness contract is otherwise identical in both lanes: same judge, same
  tolerance, same precision gate. The lane changes *where an answer may come from
  on repeated content*; it never changes what counts as correct, and it is not a
  precision budget (the C1 rejection principle stands).

### 2.3 What neither lane may do

- No run changes lanes mid-run; the task snapshot pins the lane (section 3).
- The lane does not touch: `measure_paired` and the window protocol, the precision
  gate, judging and tolerance, lever pricing / SWEEP / gate judgment, budget and
  stopping governance, plateau / restart / graveyard machinery — those read noise
  and correctness, and both are lane-independent.

## 3. Lane attribute semantics

- **The lane is a task-level attribute.** One task is born into one lane; a run
  snapshots the task; every record of that run is of that lane — no record stream
  can contain mixed-lane rows, so no consumer can accidentally average across
  lanes.
- **Cross-lane numbers are forbidden in the same comparison column.** Any table,
  digest, chart or paper row comparing numbers is either single-lane or carries an
  explicit lane grouping. A MEMO reading is not "a better STRICT reading" — it
  measures a different contract; the two are not on one scale.
- **Every report and paper table carries a lane column (or an explicit per-section
  lane label).** The trajectory rows and the digest carry the lane (section 6), so
  every renderer has the fact available; rendering it is mandatory. Comparison
  *within* a lane keeps today's rules unchanged (same-machine, same-clock-band,
  same-gate-window discipline — none of it is lane-aware).

## 4. What "keep optimizing" means inside the MEMO lane

Memoization landing does not end the optimization work; it moves the frontier:

1. **Cache management** — the cost the cache layer itself adds: keying overhead
   (device-side digest vs host-side readback; sampled vs full-content hash), hit
   policy (exact-value keying vs coarser reuse, eviction, table size and memory
   footprint), and the hit path's latency — which *is* the lane's timed metric,
   priced like any other lever.
2. **Residual compute** — the miss path: the kernel serving inputs the cache has
   never seen. The familiar problem; every existing lever family still applies.

New lever-family candidates (M3+ material; each is an ordinary lever: it lands as
a decision, is priced by measured deltas in the lane's own gate currency, and can
be falsified or swept — through the ordinary lever pricing, SWEEP_REQUEST, gate
judgment and falsified ledger. MEMO bypasses none of the priced-search governance;
it changes what a reading means, and the lane label travels with every reading so
no price is ever reused across lanes):

- **Host-side content-keyed wrapper** — digest the inputs, look up a dict of
  parked outputs, return on hit, call the kernel on miss. Priced on: per-call
  digest cost, dict overhead, park memory, eviction cost.
- **Device-side persistent cache** — outputs parked in device memory behind a
  device-computed content hash; the call prologue probes the table and early-exits
  on hit. Priced on: probe latency inside the timed window, the resident table's
  occupancy/memory cost (the resource-topology contract still holds — the lane
  does not relax M0).
- **Partial memoization** — cache sub-results inside one call (a stage's output, a
  transformed weight layout) so the timed call still computes, from a higher base.
  Priced on: stage-hit rate under the task's stream, invalidation cost.

## 5. Honesty boundaries (explicit)

- **MEMO numbers mean something only when production inputs actually repeat by
  value.** For workloads whose inputs are unique by construction (most serving
  traffic), a memo layer adds only overhead — digest cost, table residency, no
  hits. A MEMO number must never be quoted as a general speedup claim; it is a
  claim about *this task's input stream as supplied*.
- **A MEMO number is never "the kernel is N times faster than X".** The lane's
  reading is service latency over a repeating stream; any phrasing that turns it
  into a compute-speedup claim (vs a reference kernel, a baseline, another
  framework) is a misrepresentation, banned in reports, digests, papers and commit
  messages. Compute-speed claims come from STRICT only.
- **STRICT is the only source of paper capability numbers.** MEMO may appear in the
  paper only as an industrial-deployment study, clearly lane-labeled, with the
  repeating-input precondition stated beside every number.
- **The fallback check is a correctness check, not a formality.** Its known
  limitation is inherited from WP9 and recorded in G1: on file-based input cards
  (recheck mode `process`, no re-draw) the child cannot draw different content, so
  the unseen-input guarantee degenerates to fresh-process fresh-allocation
  correctness; on such cards the lane's honesty rests on the production stream's
  repeat claim, and the report must say so.
- **Recheck-child env failure never convicts** (the WP9 posture, kept verbatim):
  `pass = None` is a no-reading, reported as `memo_fallback = unknown` — it never
  silently looks verified.

## 6. Wiring specification for M3 (per file, with acceptance assertions)

Env var `RKT_MEMO_LANE` (`1` when MEMO; absent = STRICT — the house pattern: absent
means off). Task key `[protocol] memo_lane`. Card stamp: problem.json top-level
`"memo_lane": true`. Eval-JSON additions exist **only** on MEMO cards (the STRICT
eval JSON stays byte-identical).

### 6.1 `src/rsi_kernel_tune/task.py`

- `_KEYS["protocol"]`: add `"memo_lane"`. `Protocol`: add `memo_lane: bool = False`
  (absent → False → STRICT). `_read_protocol`: parse as bool; non-bool is a problem
  (the `_read_agent` `prior_survey` pattern). Nothing else — `dump_task_toml`
  already serializes the `protocol` table, so run snapshots and `self_evolve` child
  generation round-trip the key for free.

### 6.2 `src/rsi_kernel_tune/taskcard.py`

- `card_problem`: stamp `problem["memo_lane"] = task.protocol.memo_lane`
  (problem.json is generated, not a frozen surface). `write_submit`: when MEMO, one
  added `[env]` line `RKT_MEMO_LANE = "1"`; when STRICT, submit.toml byte-identical
  to today.

### 6.3 `src/rsi_kernel_tune/card_template/eval/benchkit.py`

- Add `MEMO_LANE_ENV = "RKT_MEMO_LANE"` next to `FRESH_RECHECK_ENV`, read before the
  kernel imports (the fence refuses a kernel that reads the environment — the flag
  is harness-side only).
- `_fresh_recheck_block` / `_maybe_fresh_recheck`: branch on the lane. STRICT:
  existing path untouched (byte-identical behavior). MEMO: never emit
  `fresh_recheck`; run the same fresh child instead (layer 1 fresh process, layer 2
  shifted seed — machinery reused as-is, including the child-never-nests guard and
  the timeout), attaching the verdict as top-level `memo_fallback`:
  `{"mode": <regen|process>, "pass": <bool|None>}`. On a child FAIL keep the WP20
  nondeterminism fingerprint (self-disagreement on identical content is a different
  fail form than a broken fallback); print `[bench] memo fallback:
  PASS/FAIL/no reading` in the same posture. The bench never flips
  `correctness.pass` itself: the timed-path correctness did pass; the record must
  say which check failed — the driver REVERTs on `memo_fallback` (6.5).
- Eval JSON lane stamp: MEMO cards only, `result["lane"] = {"memo": true}`; STRICT
  cards emit nothing (absent = strict).
- Sentinels unchanged in both lanes: stream / timer / process-census police timing
  attacks, not memoization; a cache hit still launches its work on the timing
  stream and reads honestly. `--mode correctness/ref/calibrate`, the precision
  gate, the fence: unchanged.
- WP29 (when it lands): the replay probe is STRICT-side ammunition for G1; it must
  read the same gate and skip on MEMO cards (a memoized parent legitimately sits
  far below the child's honest re-time — the lane's own metric, not an attack
  signature). Conditional touchpoint; M3 states the gate check at the probe's
  spawn site.

### 6.4 `src/rsi_kernel_tune/loop/parse.py`

- `EvalResult`: add `lane: dict | None = None` and `memo_fallback: dict | None =
  None`; `parse_eval` reads them off the eval JSON top level (the `fresh_recheck` /
  `gpu_state` pattern: absent on STRICT → None).

### 6.5 `src/rsi_kernel_tune/loop/driver.py`

- `LoopContext`: add `memo_lane: bool = False`, set where the context is built from
  the loaded task snapshot (`<run>/task/`, re-read by launch like every other task
  key). Every `IterationRecord` the driver writes carries `memo_lane=ctx.memo_lane`.
- New MEMO REVERT path, parallel to the WP9 fresh-recheck FAIL path (STRICT-only):
  `ev.memo_fallback.get("pass") is False` → hard REVERT, `correctness=False`,
  error text `"memo_fallback FAIL (mode=<mode>): the cache's miss path answered a
  fresh draw incorrectly — the honest fallback is broken"`. Not marked
  `suspicious` (replaying timed content is legal here; what failed is plain
  correctness on unseen inputs). `pass is None` → no conviction (WP9 posture); the
  block rides the row and the digest flags the unverified fallback.
- Brief surface: when `ctx.memo_lane`, inject one short contract paragraph (≤ 5
  lines, the injection-block discipline) into the student prompt: memoization on
  repeated content is legal in this lane; the miss path must compute honestly (a
  fresh draw will be fed to it and must be answered correctly); the gate currency
  is service latency over the repeating stream. All other discipline text stays
  untouched.

### 6.6 `src/rsi_kernel_tune/inner/trajectory.py` and the report lane column

- `IterationRecord`: add `memo_lane: bool = False` (the append-only rule already
  handles older files: `load_records` drops unknown keys; a row without the field
  is a STRICT-era row and reads False). `build_digest`: emit top-level
  `"lane": "memo" | "strict"` from the newest row (all rows of a run share it —
  section 3).
- Report / paper surfaces: any renderer laying rows into a comparison table groups
  or column-labels by lane. For M3: the digest header line, the run-end summary
  and the `rsi-kernel-tune export` provenance note (`launch.py cmd_export`) carry the
  lane string; cross-run table tools (paper layer, WP26) read the digest's `lane`
  key; M3 adds a render-time assertion that no table row set mixes lanes.

### 6.7 `scripts/self_evolve.py`

- Inheritance: the child generation copies the parent task's `task.toml`
  (`read_task_toml` → `dump_task_toml`), so once 6.1 lands the child inherits the
  lane automatically. Add the explicit guard anyway: after building the child, its
  `protocol.memo_lane` must equal the parent run's lane (regression tripwire — a
  mismatch is a bug; fail loudly).
- Carry stamp: `_write_carry` adds the parent's lane as a machine-readable stamp
  (top-level `"memo_lane"` key in `levers.json`). Lever prices and ARCH_PROBE
  readings are priors over a measurement stream; across lanes the stream differs,
  so the stamp is what makes the carry auditable.
- Carry consumption: the PRIOR-INPUT injection into the child's distill/recon
  briefs compares the stamp against the child task's lane and drops a mismatching
  carry with a log line (`carry lane memo != task lane strict — carried prices not
  valid here; starting without them`). Through `self_evolve` the lanes always
  match (inheritance + guard); the check defends the hand-edited case.

### 6.8 Acceptance assertions

- Cross-cutting: the full existing suite passes unmodified (every pin green — the
  STRICT default path is behavior-identical); new tests for every row below; the
  branding gate green; zero CJK.

| File | Acceptance assertions |
|---|---|
| `task.py` | `memo_lane = true` loads to `Task.protocol.memo_lane is True`; `memo_lane = "yes"` raises TaskError; existing fixtures load with False; `dump_task_toml` round-trips; existing task tests pass unmodified |
| `taskcard.py` | a STRICT card's submit.toml is byte-identical to pre-M1 (fixture pin); a MEMO card carries the env line; both cards' problem.json carry the stamp |
| `benchkit.py` | STRICT rehearsal (`RKT_FRESH_RECHECK_DEVICE=cpu`) eval JSON field-identical to pre-M1; MEMO rehearsal JSON carries `lane` and `memo_fallback`, never `fresh_recheck`; the fallback child answers correctly for an honest kernel on both recheck modes |
| `parse.py` | a STRICT JSON parses with `lane`/`memo_fallback` None; a MEMO JSON populates them; no existing parsed field moves |
| `driver.py` | STRICT dry-run records `memo_lane=False` with zero pins modified; MEMO dry-run (fixture card, CPU rehearsal device) records `memo_lane=True`; `memo_fallback.pass=False` produces the REVERT with the specified error text; `pass=None` produces no conviction plus the unverified-fallback flag |
| `trajectory.py` | a pre-M1 trajectory file loads and digests as strict; a MEMO run's digest carries `"lane": "memo"` |
| `self_evolve.py` | a generation chain keeps one lane end to end; a hand-flipped child ignores its carry with the log line; same-lane carry reads exactly as today |

## 7. Red lines

- **`measure_paired` and the window protocol are byte-frozen** — in both lanes, in
  this package and in M3. The lane never touches timing code.
- **The STRICT path is byte-identical.** `memo_lane` defaults to false; every new
  branch is guarded so the default path executes exactly today's code; the STRICT
  eval JSON and submit.toml are byte-identical; no existing test pin may be
  modified to accommodate the lane.
- **Anti-fraud discipline text is untouched.** The frozen discipline sentences (the
  driver's injected discipline lines, `meta/` prompts, the WP9 failure-posture
  wording) stay byte-identical; the only new injected text is 6.5's MEMO contract
  paragraph, additive and lane-gated.
- **Correctness is never negotiable in either lane.** The judge, tolerance and
  precision contract are identical in both lanes; MEMO relaxes only where an
  answer may come from on repeated content, and the unseen-input fallback check is
  a hard gate (a MEMO deliverable failing a fresh draw is reverted like any
  incorrect kernel).
- **No cross-lane contamination.** The two lanes' numbers never share a comparison
  column; lane labeling is mandatory in trajectory, digest, reports and paper;
  carry never crosses lanes (6.7). A MEMO number phrased as a compute-speedup
  claim (including "kernel N times faster than X") violates this document, not
  just style.
- **This document is a design, not an implementation license.** M1 adds no code;
  M3 implements exactly this spec and routes any deviation through adjudication.

<!-- SPDX-FileCopyrightText: Copyright 2026 The RSI-Kernel-Tune Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
