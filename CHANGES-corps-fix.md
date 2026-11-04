# CHANGES — corps fix wave, worker `zc` (branch `feat/corps-fix-zc`)

Base: `8977b63` (origin/main). All findings were re-adjudicated against the latest tree per
correction order #173; the old-snapshot reference repo is frozen and was not touched. Every fix
below was reproduced before the fix and the repro was confirmed gone after it, with a targeted
regression test. Findings and claims are tracked in the shared corps fix ledger under `handoff/`.

## Fixes delivered on this branch (10 commits, 19 ledger items)

| Commit | Items | Summary |
|---|---|---|
| `cd68414` | BUG-SEED-01 / BUG-ZC-01 (P0) | `_paired_reanchor` logs tolerate a None absolute gap (resume + workloads-mismatch path crashed after buying 2 GPU jobs; the watchdog would re-crash at the same point). 2 regression tests on the real call chain. |
| `707999d` | SEED-22, SEED-50, T6C-02, P3 (parse) | parse_eval treats NaN as unavailable (was poisoning `_best_ms` order-dependently and the band); PROBE_REQUEST/EXHAUSTED markers anchor to line starts (a record citing a marker no longer buys a GPU probe or kills a phase); the PROBE-MANIFEST comment matches the code; the "PROBE PASSED" regex keeps max_diff. |
| `823536c` | SEED-24 | precision `zip(..., strict=True)` — a length mismatch surfaces as an unavailable row instead of silently truncating. No constants touched. |
| `fea2433` | SEED-33, SEED-49, SEED-35, SEED-32, SEED-34, T5C-01 | seeding IO is locale-proof (`encoding="utf-8"` everywhere incl. the struct table); the kb quarantine keeps `<arch>/` and never overwrites; the meta verdict probe (and its manifest batch path) degrade locally on BackendError; ref role-exposure FULL tables get the dossier-sized guard and student roles split best/head names; `part 0 of n` is malformed in both readers (no more scaffold-free-pass vs rollback divergence); the struct-table log names the destination. |
| `e4a2af1` | SEED-27, SEED-46 | self_evolve `noise_floor` drops non-positive ratios (was ZeroDivisionError after the whole generation); the "and runs it" test now actually asserts the run step fires. |
| `31a729c` | C11-01 | the WP1 digest "medians stay" assertion compares against the source record (was self-comparing, always true). |
| `6086504` | ZC-63 | the eval-export import-timeout env knob is parsed defensively (a bad value falls back to 300 instead of a bare traceback past the exit-2 contract). |
| `1e445c2` | CORPS-C1 | a non-dict workload/definition in the analytic floor degrades to "no analytic floor" instead of escaping as AttributeError and failing the run. |
| `cca5e85` | ZC-50 + P3 (meta) | the meta trajectory rewrites atomically (tmp + os.replace), reads tolerate a torn trailing line, and outcomes are validated against the module enum on write. |
| `e7b79de` | P3 (window contract) | a malformed prep shape (non-list / non-int elements) is refused structurally instead of escaping as TypeError and stopping the run. |

## Verification

- Full suite on this branch: `PYTHONPATH=src <audit venv python> -m pytest tests/ -q` —
  2717 passed with only the three branding tests failing on an early draft of this very file
  (CJK path, legacy venv name, old brand word); after the rewrite below the tree is expected
  fully green (2720 collected). Per-fix targeted regressions all pass.
- New test files: `tests/test_seeding_and_bank_guards.py` (9 cases) plus additions to
  `test_loop_parse`, `test_precision_measure`, `test_self_evolve`, `test_wp1_measurement`,
  `test_eval_export`, `test_analytic_floor`, `test_meta_trajectory`, `test_window_contract`.
  Two stale assertions that pinned the old flat quarantine path were updated with that fix
  (`test_kb_arch.py`, `test_loop_driver.py`).

## Triage wave (this worker's share, all evidenced on main@8977b63)

Ten partition reviewers re-adjudicated the 60 seeded findings plus hunted new bugs: A-class
(already fixed upstream) 6 — SEED-30 (WP15 telemetry consumption), C1-01 (WP15 A1 replaced the
mechanism), SEED-47 (WP15 A-G6), C5-01 (WP20 precision re-attach), SEED-37 (task loader
rewrite), SEED-39 (refuted: the `_Tee` fd accounting is fine under refcounting); B-class (alive,
new coordinates filed to the ledger) ~45; C-class new 5 filed by this worker (BUG-ZC-44, ZC-50*,
ZC-58, ZC-62, ZC-63*; * fixed here). Cross-checked against the parallel corps triage lines with
zero divergences (one clarification: SEED-34 is unified for k>=1 but the k=0 malformed case
diverged — reproduced and fixed above).

## Not fixed here (deliberately)

- Frozen surfaces, report-only: SEED-04 residual (resolvable pool duplicates/ramp — WP15-family
  semantics), BUG-ZC-58 (timing `workloads_performance` geomean raises on unavailable host
  readings), BUG-ZC-62 (timing `SPRT_WIN_MARGIN` env knob raises at import inside the job).
  Coordinates and repros are in the ledger for the owner to rule on.
- Claimed by other corps workers (fixa/fixb/fixc/fixd/f9 lines in the ledger): the monitor hook
  family (SEED-14/15/41/42), the submit ssh family (SEED-11/12/20), the lifecycle/policy family
  (SEED-13/17/36/38/44/45), the driver measurement family (SEED-05/06/31 — already landed on the
  shared corps branch), the guard family (SEED-07/08/09, C3-01 — already landed), C5-02/C5-03,
  T6C-01, T2C-02, T7C-01, CORPS-C2.
- Archived-test porting (the five untracked legacy-wave test files under `audit-r2-2026-11-05/`):
  the port map is in the ledger (adapt/drop per file). On this branch only the tests whose root
  cause this branch fixes could land (a test pinning another worker's fix would be red here);
  the rest must land together with their fixes on the collector's assembled branch.
