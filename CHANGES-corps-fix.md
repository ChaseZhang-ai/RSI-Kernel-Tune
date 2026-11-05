# CHANGES — corps fix wave R2 (2026-11-05)

Branch `feat/corps-fix-r2`, base `8977b637` (origin/main). 67 commits: the P0
fix, 8 per-cluster fix branches merged, integration fixups. **No push, no
merge to main** — the chief collects from this clone.

Full regression: **2805 passed, 0 failed** (baseline at `8977b63`: 2692
passed; +113 new regression tests, all red→green per fix).

## Scope and method

Per the chief's correction order (2026-11-05): every finding in the corps
fix ledger (61 entries, filed against the old upstream snapshot; it lives in
the workspace `handoff/` directory under a Chinese filename) was
re-adjudicated **on this tree, with evidence** — A: already
fixed here (evidence cited, skipped); B: alive (re-located, minimally
reproduced, fixed one commit per bug); C: newly introduced here (found while
triaging; fixed). The upstream snapshot tree was not touched.

- Triage: 9 read-only agents, clustered by module; every B/C verdict carries
  latest file:line + a reproduction run on this tree.
- Fixes: 8 engineer agents in separate clones (`feat/corps-fix-r2-{driver,
  parse,cli,launch,guard,monitor,submit,misc}`), file ownership staggered,
  merged zero-conflict, then integrated and re-verified.
- Archive test porting: the five `test_*_fixes.py` files from the preserved
  fix-wave archive were ported where their pinned mechanisms exist here
  (adapted, renames applied); tests pinning mechanisms that do not exist here
  or frozen semantics were dropped with reasons below.

## Counts (order's closing line)

- A (already fixed in latest, skipped with evidence): **6**
  (SEED-30 main body, SEED-37, SEED-39, C5-01, C1-01, TST-01)
- B (alive → fixed): **52**
- C (newly introduced → fixed): **1** (C-T2-01)
- Frozen-surface, report-only leftovers: **3**
  (SEED-05, SEED-06, SEED-04 direction two)

## Fixed (B, one commit each — ledger IDs)

P0: SEED-01 (gap=None re-anchor crash; `e9dc089`).
P1: SEED-02 (`82c990b`), C1-02 (`4072f9a`), SEED-04 pool direction
(`fa00e6e`), SEED-07 (`003dd69`), SEED-08 (`76b6840`), SEED-09 (`f6b7eb1`),
C3-01 (`1d21faf`), SEED-10 (`3e1b3cd`), SEED-11 (`4879de1`), SEED-12
(`fdd77ef`), SEED-13 (`9e6ef61`), SEED-14 (`d7155ee`), SEED-15 (`4dbac83`),
SEED-16 (`fb276a0`), SEED-17 (`9d57ccd`), C3-02 (`6518f53`).
P2/P3: SEED-03 (`0dbf1bf`), SEED-18 (`773ae8a`), SEED-19 (`1b89450`),
SEED-20 (`5162b5e`), SEED-21 (`4585825`), SEED-22 (`6d4d455`), SEED-23
(`6de8d72`), SEED-24 (`9e94c1e`), SEED-25 (`df89cdd`), SEED-26 (`2f0a45f`),
SEED-27 (`918ba11`), SEED-28 (`589fde1`), SEED-29 (`3c200f3`), SEED-31
(`fd42a23`), SEED-32 (`1f4481c`), SEED-33 (`3e602e2`), SEED-34 (`cabf87b`),
SEED-35 (`b69dc16`), SEED-36 (`5ad0e6b`), SEED-38 (`8a07f8c`), SEED-40
(`1e2612f`), SEED-41 (`d5fae2a`), SEED-42 (`3e50a20`), SEED-43 (`e5c9287`),
SEED-44 (`d69812e`), SEED-45 (`288b22e`), SEED-46 (`bf3bebb`), SEED-47
(`a99a047`), SEED-48 (`70c21bd`), SEED-49 (`6760b33`), SEED-50 (`7cf30eb`),
C1-03 (`688b653`), C5-02 (`47a25ac`), C5-03 (`4083da8`), C11-01 (`5235d59`).
Bonus from triage: Tee close + torn status line (`a612f9d`, SEED-39
hardening + the launch P2), SEED-30 residual clock header (`e7c28fb`).

## Frozen surfaces — report only (unchanged code)

- **SEED-05** (`driver.py:14093-14098`): `_judge_patch`'s absolute-ms branch
  prices a cross-job gain with `resolvable_diff` (0.5–1.5%, same-job pairing
  semantics) instead of the project's own 6% cross-job band
  (`_GATE_REGRESSION_TOLERANCE_MS`, used by the sibling `_gate_regression`).
  A ~1.5% node-lucky draw can be judged KEEP_PATCH, and the comment admits
  KEEP_PATCH has no later rollback. Fix = a judging-bandwidth change → frozen.
- **SEED-06** (`driver.py:2444`): the watermark confirmation agrees only
  between its two draws (`abs(ms_n - ms) <= margin * ms`), never against the
  old best; a confirmation slower than the best still records a median gain.
  Fix = a confirmation-criterion change → frozen.
- **SEED-04 direction two**: `resolvable_diff` v2's leave-one-out residual
  still reads a monotone improvement ramp as spread (3σ pins the cap), so a
  real sub-margin patch on a steady climb can be REVERTed; the docstring
  claims the ramp bias is gone. Fix = estimator semantics (detrend) → frozen.

## Archive test porting (five `test_*_fixes.py`)

- `test_driver_measurement_fixes.py` → `tests/test_driver_measurement_fixes.py`:
  fix 1 (SEED-01) at P0 time; fixes 2/6/7 (SEED-02/29/31) with their fixes;
  **dropped**: fix 3 (the own-job-sample band guard does not exist here — WP15
  replaced it with the A1 confirm-every-best routine), fix 4 (6% absolute
  band never landed in this lineage; `_judge_patch` was reworked to ratio
  currency), fix 5 (the confirmation-vs-old-best mechanism does not exist
  here and is frozen semantics anyway).
- `test_guard_fixes.py` → `tests/test_guard_fixes.py` (guard cluster, ported
  with its fixes; `BASIS_CUPTI` imported from `peregrine_kernel.baseline` —
  the old tree's wrong-module import was ledger BUG-COORD-1); host_path_note
  cases live with SEED-23's owner (`tests/test_host_path_note_directions.py`).
- `test_cli_submit_fixes.py` → split by owner: submit halves in
  `tests/test_submit_direct_verdicts.py`, cli halves in
  `tests/test_cli_corps_fixes.py`; two pins rewritten to this tree's
  verdict channels (documented in commit bodies).
- `test_monitor_resume_fixes.py` → monitor/trajectory halves in
  `tests/test_monitor_fix_r2.py` + `tests/test_inner_trajectory_fix_r2.py`;
  the companion ⊗ cases went to the launch cluster's file (resume owner).
- `test_lifecycle_fixes.py` → launch cluster (`tests/test_corps_launch_fixes.py`);
  the malformed-board cases were already green here (SEED-37 = A).

## Integration fixups (corps level)

- Four shared pins updated to the landed semantics (quarantine name, REVERT
  row correctness None, digest head keys) — `46a7526`.
- Ported-file wording for the WP25 branding gate and deboard check —
  `a32d5e0`.

## Notes for the chief

- `tests/test_loop_driver.py:5433` and `tests/test_timing_audit.py:219`
  pinned SEED-23's inverted wording itself; each flipped one word (disclosed
  by the misc engineer).
- SEED-19 keeps fail-open (behavior unchanged) but now leaves a structured
  trace and no longer mis-announces "the tree's copies match the card".
- SEED-11/12: a queue-budget expiry or a dropped ssh connection with the job
  in flight is now **Lost** (re-queued, never charges an iteration); a
  relayed remote exit code is still a charged Failed. The rc=255-without-
  marker path keeps the transport-retry channel — the archive test and a
  frozen shared test pinned opposite semantics there; the chosen split is
  documented in `fdd77ef`.
- During this wave, other sessions appended 34 new findings to the shared
  ledger (BUG-CH-A..L, BUG-ZC-*, BUG-T5C-02/03). They are outside this
  order's triage scope and remain open for their owners' queues.
