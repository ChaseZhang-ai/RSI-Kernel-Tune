# CHANGES — audit-corps R3 fix wave (2026-11-05)

Base: `feat/corps-fix` @ `623baa4` (closeout of the corps-fix wave, 2915 passed).
Integration branch: `feat/corps-fix-r3` — merges the eight R3 fixer branches
(`feat/corps-fix-r3a` … `feat/corps-fix-r3h`), file-disjoint by construction, no
conflicts. Not pushed, not merged to main (commander picks up).

## Scope

50 ledger entries were still open after the closeout wave: the 31 findings of
the R3 recon groups (BUG-CH-*) and the 19 new C-class findings of the RECON-R2
audit (BUG-CORPS-C-*). All 50 are now dispositioned:

- **49 fixed** — each with a red-before/green-after regression test and one
  `fix(audit-corps): …` commit (merge commits group them per fixer branch).
- **1 verified already fixed by the previous wave** — BUG-CORPS-C-13 (ssh
  sync timeout) is covered by the closeout wave's `eed4844`; recorded in the
  ledger, not re-fixed.

## Fixes by branch

- **r3a — loop/driver.py (9)**: WP18 PRICED_NULL stamps no longer overwrite
  each other or swallow a suspended re-priced ruling (CH-C-01); non-dict
  teacher lever entries are dropped, not fatal (CH-B-01); first-green M0-audit
  budget gate reads the lifted `meta_cap` (CH-D-01); mistyped sweep `rounds`
  refuses instead of crashing (CH-B-02 — also fixes an ImportError that made
  the per-shape sweep branch unreachable since WP14); the background-lane
  temperature guard tolerates non-numeric readings (CH-A-02); the Stouffer
  advisor reads live rows only (CH-C-03); `_GATE_ROW` uses anchored matching
  (CH-A-01); recon launch facts keep the recorded clustered reading on ties
  (CH-C-02); `_num` clamps float knobs from below (CH-D-02).
- **r3b — loop/cli.py, loop/parse.py (7)**: dry-run/harvest-only no longer
  persist a hand-written arch baseline (CH-E-01); `_correctness_fail_detail`
  ranks non-numeric error strings safely (CH-E-03); harness revert consumes
  `-z` diffs so quotepath-escaped names cannot dodge it (CORPS-C-11); a dead
  resumed session is dropped after one hard failure instead of burning all
  retries (CORPS-C-03); the launch-admission sentinel now sits behind every
  refusal gate (CH-E-02); `[loop]` numeric knobs refuse readably (CH-E-05) and
  config reads pin UTF-8 (CH-E-04).
- **r3c — guards/adversarial (7)**: the goal-ref spool's harness side is
  read-only to the session (CORPS-C-09); the session CLI runs with `-P -s` so
  a cwd shadow package cannot hijack it (CORPS-C-10); `--model $VAR` resolves
  from every readable assignment and raises instead of silently dropping legs
  (CORPS-C-06, with per-statement unset/redeclare tracking — CORPS-C-12);
  wrapper detection covers .py/extensionless entries, nested scripts,
  `command cd` (CORPS-C-07); wrapped `bash -c 'ssh …'` is denied for online
  students (CORPS-C-08); `--no-git` refuses to boot unguarded unless
  `--allow-unguarded-harness` is passed explicitly (CORPS-C-05).
- **r3d — submit/ (8)**: `[env]` keys are validated and quoted (CORPS-C-14);
  entry argv is re-quoted before the remote bash -c rejoin (CORPS-C-15);
  restore streams in 1MB chunks with edge hashing (CORPS-C-17); infra faults
  record LOST instead of FAILED and usage errors are named as such
  (CORPS-C-16); the manifest memo is bound to the op-tree identity
  (CORPS-C-18); job pids carry a /proc starttime identity checked before
  kill (CORPS-C-19); input sync no longer sweeps `.exec` extraction trees
  into manifests and restore refuses to rebuild them (CH-I-01). C-13 verified
  already fixed upstream.
- **r3e — launch/localenv/task (4)**: drive() finalization lands a terminal
  status even when it throws (CORPS-C-02); `harvest` submits zero GPU jobs
  (CORPS-C-01); local.env strips inline comments outside quotes (CORPS-C-04);
  non-UTF-8 task files are collected as problems, not tracebacks (CH-I-02).
- **r3f — benchkit/cardkit/monitor/kbmenu/analytic_floor (7)**: kernel_fence
  survives non-UTF-8 kernels (CH-G-01); kbmenu zero readings fire their rules
  (CH-F-01); the WP29 replay probe runs under the WP11 workload scope
  (CH-G-03); monitor env knobs validate readably and hook-install failures
  surface (CH-H-01); seed-change detection survives fully-written params via
  the default reseed channel (CH-H-02); the LB refuse list covers `allncu`
  (CH-G-02); analytic-floor formulas guard wrong-typed axes down to a bytes
  roofline instead of crashing (CH-F-02).
- **r3g — scripts/skills (4)**: the successive-halving backtest actually
  halves (budget fixed, cuts instrumented, vacuous gates FAIL) (CH-J-01);
  eval-export preflight degrades to SKIP on runtime faults (CH-J-02) and
  length-checks outputs instead of truncating silently (CH-J-03); the
  timeline probe keeps the computed table when a waits re-run fails
  (CH-J-04).
- **r3h — tests/kernel (4)**: three placeholder tests now assert real
  behavior (T7 sha-mismatch via captured driver log — CH-K-01; SWEEP/MC
  delegation wording with a restored sink — CH-K-02; de-board surface with
  live assertions, `cmd_export` verified legitimate and kept — CH-K-03); the
  gen0 attention kernel pins entry shapes/dtype and keys its compile cache
  (CH-L-01, static verification only).

## Verification

- Full suite on the integration tip: see ledger closeout entry (baseline
  2915 + 166 new tests from the eight branches; expected 3081 passed).
- Every fix was written red-first; r3c additionally ran revert-mutation
  checks on all seven fixes.

## Discipline

- Frozen surfaces untouched: `timing.py` `measure_paired`, precision
  constants, `meta/` discipline text, gate semantics, LICENSE/NOTICE,
  resolvable_diff estimator semantics (SEED-04 direction 2 stays report-only).
- All new/modified files are English; no pushes, no merges to main.
- The commander's earlier standalone branches `feat/corps-fix-p0` (SEED-01
  gap=None) and `feat/corps-fix-port` (12 ported pins) are **kept for
  reference, not merged**: SEED-01 is already fixed on the shared line
  (`0226a7f`, `cd68414`) and the five test files were already ported by the
  closeout wave — pick one of each when collecting.
