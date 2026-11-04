# CHANGES-corps-fix (core / reconciliation record)

This branch carries **no code fixes**. It is the reconciliation and verification
record of the 2026-11-05 audit-corps fix round, delivered so the collector can
merge the parallel fix lines without losing work. All code fixes live on the
sibling branches listed below; the repair ledger under `handoff/` (section
`[RECON-R2]`) is the authoritative disposition of record.

## Scope of this round

- Target repository: `origin/main` @ `8977b63` (package `peregrine_kernel`).
- Input: the ledger's 61 original entries (BUG-SEED-01..50, BUG-COORD-1,
  BUG-C1-01..03, BUG-C3-01..02, BUG-C5-01..03, BUG-C11-01, BUG-TST-01), each
  re-tried against the current tree with fresh file:line evidence and, where
  feasible, a minimal repro (reports `findings/triage-t1..t5-*.md`).
- Baseline verified before any work: 2692 passed, 0 failed.

## Verdicts

- **A (already fixed in the current tree, skipped): 9** — SEED-01, SEED-20,
  SEED-30, SEED-37, SEED-47, COORD-1 (carrier file no longer exists), C5-01,
  C1-01, TST-01 (current semantics already match the assertion).
  Note on SEED-01: the round's standing order claimed the crash still lived at
  driver.py:13274->13520; direct inspection shows all four `gap` uses inside
  `_paired_reanchor` (13457/13469/13506/13533) are None-guarded, and commit
  `0226a7f` + `tests/test_paired_reanchor_none_gap.py` are the fix. The order's
  coordinate was stale; the triage agent's initial B verdict was wrong and was
  overturned by direct code inspection plus blame.
- **B (alive): 52** — 51 fixed on at least one branch; SEED-04 is the single
  unfixed item (verdict split, see Leftovers).

## Branch map (collector's checklist)

- `feat/corps-fix` (shared line): SEED-02, 03, 05, 06, 07, 08, 09, 10, 11, 12,
  13, 14, 15, 16, 17, 18, 19, 21, 24, 29, 31, 41, 42, 45, 48, C1-02, C1-03,
  C3-01, C3-02, C5-02, C5-03 (31 commits referenced in the ledger RECON-R2).
- `feat/corps-fix-a`: SEED-11, SEED-12 (queue-time = Lost, marker after the
  lock, session file moved outside the synced tree). Full suite 2699 passed.
- `feat/corps-fix-b`: SEED-07, 08, 09, C3-01, C5-02, C5-03. Full suite
  2712 passed.
- `feat/corps-fix-c`: SEED-10, 13 (with the five test_launch fixtures re-pinned
  to the correct contract), 17, 18, 19, 21, 23, 36, 38, 39, 40, 44, 45, 46, 48,
  C3-02, C11-01. Full suite 2713 passed.
- `feat/corps-fix-d`: SEED-14, 15, 16, 25, 26, 28, 41, 42, 43. Full suite
  2715 passed.
- `feat/corps-fix-zc`: SEED-22, 24, 27, 32, 33, 34, 35, 46, 49, 50, C11-01 and
  its own C-class finds. Full suite 2720 passed.
- `feat/corps-fix-f9`: SEED-49, 50 plus three archived-test ports.

**Fixes present on exactly one branch (do not lose these at merge time):**
36/38/39/40/44 only on `-c`; 43/25/26/28 only on `-d`; 22/27/32/33/34/35 only
on `-zc`; 49/50 only on `-zc`/`-f9`. SEED-11/12 exist on both the shared line
and `-a`; the `-a` variant additionally moves the session file out of the
`rm -rf` surface and adds queue telemetry.

## Archived-test ports (all five delivered somewhere)

`test_cli_submit_fixes` -> `-a` (7 passed); `test_lifecycle_fixes` -> `-c`
(9 passed); `test_guard_fixes` -> `-b` (18 passed) and f9;
`test_monitor_resume_fixes` -> `-d` (20 passed) and f9;
`test_driver_measurement_fixes` -> f9. Spot-verified at the tips by this
session: 7/18/9/20 passed respectively.

## New findings ledgered this round (C-class, queued)

19 entries `BUG-CORPS-C-01..19` (harvest budgeting, lifecycle wrap-up, stale
resume-session retries, env parsing, guard degradations, agent-leg visibility,
ssh delivery hardening) — sources `findings/a06/a08/a10`, each verified on the
current tree with local experiments before entry. Dedup against other groups'
sections is recorded in the ledger.

## Leftovers (for the collector)

1. SEED-04/T1C-01 (resolvable_diff pool pollution): split verdict — one triage
   group says the WP15 rewrite covers it, two groups measured both pollution
   directions still live (0.0050 vs honest 0.0150). Suggested: dedupe reused
   readings from the sample pool; de-trending optional.
2. Reference-timing pollution residue (same-process timing; the WP20 sentinel
   does not watch compute ops).
3. Timing-aggregation regression pins archived with `-b` notes (upstream fixes
   never rode along; the precision-truncation case is covered by the shared
   line's length-mismatch fix).
4. SEED-20 residue: manually cloned same-name trees can still collide (low).
