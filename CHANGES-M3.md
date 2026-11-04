# CHANGES-M3 — the two-lane contract wired

Branch `feat/m3-memo-lane-wiring`, cut from origin/main `ac24e59` (WP29
merged). Package brief: handoff/142 (the lane wiring), executed against the
M1 design's file-by-file checklist, `docs/industrial-lane.md` §6.

## 1. What and why

The 2026-11-04 adjudication resolved the G1 tension instead of only hunting
the cheat: industrial production may use content-addressed memoization (a
real speedup when inputs repeat by value), while academic benchmarking stays
STRICT (memoization forbidden; numbers measure compute). M1 wrote that
contract down as the two-lane design; M2 delivered the reference shape of a
MEMO deliverable (`tests/fixtures/memo_demo.py`, the `MemoKernel`); M3 — this
package — wires the contract through the whole chain: task key → card →
benchkit gates → eval JSON → parse → driver records/verdicts → digest/report
surfaces → self_evolve inheritance and carry. STRICT (the default, absent
key) is today's behavior byte for byte; MEMO is an explicit opt-in whose
numbers are service latency over the task's repeating input stream and are
never comparable with STRICT numbers.

## 2. Per-file changes (the §6 checklist)

### 6.1 `src/peregrine_kernel/task.py`

- `_KEYS["protocol"]` gains `memo_lane`; `Protocol` gains
  `memo_lane: bool = False` (absent → False → STRICT, so every existing task
  is STRICT without any edit).
- `_read_protocol` parses it with the `prior_survey` pattern: a non-bool
  (`memo_lane = "yes"`) is a `TaskError`, not a truthy spelling.
- `dump_task_toml` already serializes the `[protocol]` table, so run
  snapshots and `self_evolve` child generation round-trip the key for free
  (verified by test).

### 6.2 `src/peregrine_kernel/taskcard.py`

- `card_problem` stamps `problem["memo_lane"] = task.protocol.memo_lane`
  (both lanes carry the stamp; problem.json is generated, not a frozen
  surface).
- `write_submit`: a MEMO card adds exactly one `[env]` line
  `PK_MEMO_LANE = "1"`; a STRICT card's submit.toml is byte-identical to
  pre-M3 (pinned byte-for-byte in the test suite).

### 6.3 `src/peregrine_kernel/card_template/eval/benchkit.py`

- `MEMO_LANE_ENV = "PK_MEMO_LANE"` beside `FRESH_RECHECK_ENV`, plus
  `_memo_lane()` (absent / `0` / `off` = STRICT — the house pattern: absent
  means off). Harness-side only: the fence refuses a kernel that reads the
  environment.
- `_maybe_fresh_recheck` branches on the lane. STRICT: today's path
  untouched. MEMO: never emits `fresh_recheck`; the SAME fresh child runs
  (layer 1 fresh process, layer 2 shifted seed — `_fresh_recheck_block`
  reused as-is, child-never-nests guard and timeout included), its verdict
  attached as top-level `memo_fallback`
  (`{"mode": "regen"|"process", "pass": bool|None, ...}`, with the WP20
  nondeterminism fingerprint kept on a child FAIL), printed as
  `[bench] memo fallback: PASS/FAIL/no reading`.
- `_emit` stamps `result["lane"] = {"memo": True}` on MEMO cards only; a
  STRICT eval JSON emits nothing (absent = strict) and stays byte-identical.
- `_maybe_replay_probe` gains the §6.3 gate at the probe's spawn site: on a
  MEMO card the probe skips before anything is spawned (a memoized parent
  legitimately sits far below the child's honest re-time — the lane's own
  metric, not an attack signature). The WP29 probe body, thresholds and
  switch are untouched.
- Sentinels, `--mode correctness/ref/calibrate`, the precision gate and the
  fence: unchanged. `timing.py`: byte-identical (untouched file).

### 6.4 `src/peregrine_kernel/loop/parse.py`

- `EvalResult` gains `lane: dict | None = None` and
  `memo_fallback: dict | None = None`; `parse_eval` reads them off the eval
  JSON's top level (the `fresh_recheck`/`gpu_state` pattern: absent on
  STRICT → None). No existing parsed field moves (pinned by test).

### 6.5 `src/peregrine_kernel/loop/driver.py`

- `LoopContext` gains `memo_lane: bool = False`, set at construction from the
  task snapshot the launcher re-read (see §4.2 below on the wiring seam).
  Every `IterationRecord` the driver writes now carries
  `memo_lane=ctx.memo_lane` (all 12 construction sites).
- New MEMO REVERT path, parallel to the WP9 fresh-recheck FAIL path:
  `ev.memo_fallback.get("pass") is False` → hard REVERT, `correctness=False`,
  error exactly `"memo_fallback FAIL (mode=<mode>): the cache's miss path
  answered a fresh draw incorrectly — the honest fallback is broken"`, NOT
  marked `suspicious` (replaying timed content is legal in this lane; what
  failed is plain correctness on unseen inputs), the block riding the row as
  the evidence face. `pass is None` → no conviction (the WP9 posture,
  verbatim): the block rides the KEEP row and the digest flags the
  unverified fallback.
- KEEP/REVERT/gate judgment logic is otherwise unchanged — the lane is a
  measurement-currency fact, not a judgment difference.
- Brief surface: `ctx.memo_lane` injects one contract paragraph (a header +
  three bullets) into the student prompt — memoization on repeated content
  is legal here; the miss path must compute honestly (a fresh draw will be
  fed to it and must be answered correctly); the gate currency is service
  latency over the repeating stream, never comparable with STRICT. All other
  discipline text untouched; STRICT prompts are byte-identical (pinned: the
  memo prompt minus the note equals the strict prompt).
- Carry consumption (§6.7's driver side): `_carry_binding` compares the
  carry's `levers.json` top-level `memo_lane` stamp against `ctx.memo_lane`;
  a mismatch drops the carry outright with the log line
  `carry lane <X> != task lane <Y> — carried prices not valid here; starting
  without them` (logged once per directory). No stamp (a pre-M3 carry) reads
  as STRICT — its parent was necessarily STRICT.
- Cross-lane contamination guard (additive beyond §6's letter — see §4.3):
  an eval JSON stamped MEMO delivered into a STRICT run is refused (the
  `workload_scope` posture: the reading is invalid, `correctness=False`, the
  reason is the error). Dead code on every lane-consistent run; it gives
  `EvalResult.lane` its designed consumer and backs §3's "no consumer can
  accidentally average across lanes".

### 6.6 `src/peregrine_kernel/inner/trajectory.py` and the report surfaces

- `IterationRecord` gains `memo_lane: bool = False` and
  `memo_fallback: dict | None = None` (append-only rule handles older files:
  `load_records` drops unknown keys; a row without the field is a STRICT-era
  row and reads False — pinned by test).
- `build_digest` emits top-level `"lane": "memo" | "strict"` from the newest
  row (all rows of a run share it) and asserts at render time that no row set
  mixes lanes — a mixed trajectory raises instead of silently juxtaposing
  cross-lane numbers. It also flags rows whose `memo_fallback` produced no
  reading (`memo_fallback_unknown`, the WP9 no-reading posture made loud).
  The lane key sits in the header block after the `sol`/`clock` headers — an
  existing pin holds `sol` as the digest's first field and M1 forbids
  modifying existing pins (§4.4 below).
- Report surfaces carry the lane string:
  - the run-end summary (`loop/summary` writer in `loop/cli.py`) gains
    `"lane": "memo" | "strict"`;
  - `launch.write_report` gains a `- lane: <lane>` line, and for MEMO runs it
    states beside it that the speedup below is service latency over the
    task's repeating input stream — not a compute-speedup claim, never
    comparable with STRICT (the §5 honesty boundary, enforced at the render);
  - `launch.cmd_export` prints a provenance note with the lane on stderr
    (stdout stays the kernel source, byte-clean) and names the lane in the
    `--out` confirmation line.

### 6.7 `scripts/self_evolve.py`

- Inheritance: the child copies the parent's `task.toml`
  (`read_task_toml` → `dump_task_toml`), so once 6.1 landed the child
  inherits the lane automatically; the explicit guard is added anyway — after
  building the child, its `protocol.memo_lane` must equal the parent's, a
  mismatch raises loudly (regression tripwire).
- Carry stamp: `_write_carry` stamps the parent's lane as the top-level
  `memo_lane` key of `carry/levers.json` (both lanes stamped — strict reads
  `false` — so the carry is auditable in both directions). The table content
  is otherwise unchanged; unparseable lever text is left as-is.
- Carry consumption: the PRIOR-INPUT injection (`driver._carry_binding`)
  drops a mismatching carry with the log line (§6.5 above). Through
  `self_evolve` the lanes always match (inheritance + guard); the check
  defends the hand-edited case.

## 3. Acceptance evidence (§6.8)

Full suite: **2692 passed** on the branch (baseline at `ac24e59`: 2655
passed; the delta is exactly the new file's 37 tests), with
`tests/dataflow_audit.py`, `tests/test_branding.py` (zero CJK) and the repo
hygiene pins green. No existing test pin's semantics modified; the one pin
the wiring touched procedurally is recorded in §5.

The `tests/test_memo_lane.py` pins, mapped to the §6.8 table:

| File | Pinned |
|---|---|
| `task.py` | `memo_lane = true` loads `True`; `"yes"` raises `TaskError`; existing fixtures load `False`; `dump_task_toml` round-trips |
| `taskcard.py` | a STRICT card's submit.toml is byte-identical to pre-M3 (embedded golden); a MEMO card carries exactly the one env line; both cards' problem.json carry the stamp |
| `benchkit.py` | the env switch (absent/0/off = strict); MEMO never emits `fresh_recheck` and attaches `memo_fallback` with the PASS/FAIL/no-reading wording; STRICT keeps today's wording and key; the replay probe skips on MEMO (no spawn) and still engages without the env; the lane stamp rides MEMO eval JSON only |
| `parse.py` | a STRICT JSON parses `lane`/`memo_fallback` None; a MEMO JSON populates them; no existing parsed field moves |
| `driver.py` | STRICT rows record `memo_lane=False`; MEMO rows record `True`; `memo_fallback.pass=False` produces the REVERT with the specified error text and no `suspicious`; `pass=None` produces no conviction, rides the row and is flagged by the digest; the memo contract paragraph is the student prompt's only delta; carry drops a mismatching lane with the log line and reads same-lane/unstamped carries exactly as today |
| `trajectory.py` | a pre-M1 trajectory file loads and digests as strict; MEMO rows digest as `"lane": "memo"`; a mixed row set raises at render time |
| `self_evolve.py` | a generation chain keeps one lane end to end (both directions); a hand-flipped child fails loudly; the carry's levers.json carries the parent's lane stamp in both lanes |

### GPU rehearsals (RTX 3060 Laptop, torch 2.14.1+cu130, `with-lock gpu`;
the same interpreter the WP29 GPU verification used; artifacts in
`/tmp/m3-golden/E*.txt` during the session):

- **E1 — STRICT byte golden (correctness path)**: the honest-kernel fixture
  card built from origin/main's `benchkit.py` vs the wired `benchkit.py`,
  same judge/kernel/problem/interpreter: parent stdout **byte-identical**
  (3885 B both), judged PASS both.
- **E2 — STRICT field golden (full mode)**: correctness + timing +
  fresh-recheck + replay probe on GPU: the eval JSON's recursive key
  structure is **identical** pre/post (no field added, moved or dropped);
  every stdout line with digits masked is identical; judged PASS,
  `fresh_recheck: PASS`, `replay_probe: pass`, and no `lane`/`memo_fallback`
  keys.
- **E3 — MEMO e2e**: the M2 `MemoKernel` as the delivered kernel
  (content-addressed cache over an honest fallback), `--mode all` on GPU
  with `PK_MEMO_LANE=1`: judged **PASS**; `memo_fallback: PASS` (the
  shifted-seed unseen-input draw answered correctly); `"lane": {"memo":
  true}` on the eval JSON; **no** `fresh_recheck`, **no** `replay_probe`;
  timing ran — device median 0.3021 ms (the cache-hit service latency the
  lane prices).
- **E4 — STRICT catch (same delivery, no env)**: judged **PASS** and
  `fresh_recheck: PASS` — the G1 value-blindness demonstrated live — and
  `replay_probe: SUSPECTED`: honest-compute probe 23.7688 ms vs judged
  0.2726 ms, **x87.19 > 3.0**. The same behavior that is a clean MEMO
  delivery in its lane is a caught cheat in STRICT; the lanes differ only in
  the declared contract.

## 4. Adjudication notes (deviations and interpretations, for the record)

1. **Scope fence vs the §6 checklist.** The task sheet's file list named
   task.py, taskcard.py, benchkit.py, driver.py, trajectory.py, tests/,
   CHANGES-M3.md; the §6 checklist additionally requires `loop/parse.py`
   (§6.4), `scripts/self_evolve.py` (§6.7), and the LoopContext/report seams
   that live in `loop/cli.py` and `launch.py` (§6.5/§6.6). The sheet's own
   rule — "conflicts: the spec governs; deviations recorded and reported" —
   was applied: every §6 file is touched exactly as the checklist says, each
   touch is additive and lane-gated, and the frozen zones (timing.py,
   precision, gate judgment semantics, meta/ discipline text, hw_kb/,
   kernels/) are untouched.
2. **The LoopContext wiring seam.** §6.5 sets `memo_lane` "where the context
   is built from the loaded task snapshot (<run>/task/, re-read by launch
   like every other task key)". The context is constructed in `loop/cli.py`,
   so the key travels the way the neighboring task keys (`--hours`,
   `--max-meta`, `--target-ms`) do: `launch.run_loop` derives `--memo-lane`
   from the `load_task(<run>/task/)` it already re-reads and the loop passes
   it into the context. Absent flag = STRICT, byte-identical.
3. **The cross-lane contamination guard** (driver.py, additive beyond §6's
   letter): §3 makes the lane a run-level fact and §7 bans cross-lane
   contamination; the sheet's task 3 asks for "cross-lane mixing errors out
   rather than silently juxtaposing". The render-time assertion covers the
   digest; the driver-side guard covers the other seam — an eval JSON stamped
   MEMO surfacing in a STRICT run is refused instead of judged. It is dead
   code on lane-consistent runs (a STRICT run can never receive a stamped
   eval in production) and is the designed consumer of `EvalResult.lane`.
   The reverse direction (a MEMO run receiving unstamped evals — a pre-M3
   card resumed under the new harness) degrades provenance rather than
   contaminating and is logged-not-refused, per the comment at the guard.
4. **The digest lane key's position.** §6.6's "digest header line" is
   satisfied by the top-level `lane` key inside the digest's header block —
   not as the literal first field, because an existing pin
   (`test_digest_sol_header_is_the_digests_first_field`) holds `sol` first
   and §7 forbids modifying existing pins.
5. **Correctness draws in the MEMO lane.** The brief's "three correctness
   draws" note is executed as §2.2/§6.3 specify: the judged inputs
   (identical protocol) and the fresh child's shifted-seed draw
   (`memo_fallback` — the unseen-input hard gate) both do correctness in the
   MEMO lane. The WP29 probe's third draw is a LATENCY check in every lane
   (it judges no outputs even on STRICT), and §6.3 skips the probe on MEMO
   cards entirely; the unseen-input purpose that third draw serves is covered
   by the `memo_fallback` child, whose draw is equally unseen content (+7919
   vs +104729 — disjoint seeds, same "never shown in the timed stream"
   property). Recorded here in case the chief engineer reads "the probe
   inputs do correctness" as a literal third correctness check; implementing
   that would be a §6.3 deviation and was not done.
6. **M2 asset reuse.** The strict-side catch and the MEMO e2e both use the
   M2 `MemoKernel` as the delivered kernel (the sheet's task 4), with the
   honest fallback made deliberately expensive (the same fixture trick the
   WP29 GPU verification used) so the honest miss path's cost is measurable;
   the CPU suite pins the child-level correctness of the same card.

## 5. Freeze compliance

- `timing.py`: untouched (byte-identical; `measure_paired` and the window
  protocol frozen in both lanes).
- The WP29 replay probe body: untouched; one lane gate at the spawn site
  (`_maybe_replay_probe`), per §6.3's "M3 states the gate check at the
  probe's spawn site".
- STRICT default path: byte-identical — E1/E2 above; submit.toml golden
  pinned in-suite; the STRICT eval JSON emits no new keys; no existing test
  pin modified (the suite is green unmodified). One nuance, recorded for
  review: `tests/test_wp19_throughput.py`'s cross-generation golden pins the
  records' merge-base fields field by field, and its own documented
  extension mechanism (used by WP11/14/15/20/29 for exactly this case) is
  the `_STRIP_FIELDS` list — M3's two spec-mandated dataclass defaults
  (`memo_lane=False`, `memo_fallback=None`, §6.6) are registered there. The
  `_GOLDEN_*` constants are byte-untouched and every merge-base field is
  still compared value-for-value, so the pin now proves the STRICT rows'
  recorded values are exactly the baseline's (the new keys hold the
  baseline-default values False/None on the default path).
- Anti-fraud discipline text: untouched; the only new injected text is the
  MEMO contract paragraph, additive and lane-gated.
- Gate judgment semantics: untouched (KEEP/REVERT/gate read noise and
  correctness, both lane-independent); the two new REVERT paths fire only on
  a broken unseen-input gate and on cross-lane contamination — correctness
  facts, not lane-dependent judgment.
- All new/modified text is English (the branding gate is green).

## 6. Package state

Branch `feat/m3-memo-lane-wiring`, working tree clean; not pushed, not
merged (the chief engineer collects). Baseline `2655 passed` at `ac24e59`;
branch tip `2692 passed` (the delta is exactly the new test file's 37
tests).

<!-- SPDX-FileCopyrightText: Copyright 2026 The Peregrine-Kernel Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
