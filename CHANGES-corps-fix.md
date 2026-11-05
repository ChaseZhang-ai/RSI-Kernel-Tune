# CHANGES — audit corps fix line f9

Line f9 of the audit corps (session registry and per-bug verdicts live in the corps'
shared fix ledger under `handoff/` in the parent workspace — the ledger file keeps its
original non-English filename, so it is referenced here by location only).
Base: `feat/corps-fix` @ `04d9baa`, which already carries the nine
commits the parallel corps lines landed first. Branch: `feat/corps-fix-f9`.
Sub-line workspaces: `wk6-corps-fix-f9{d,p,s,b,t}` (branches `feat/corps-fix-f9-{d,p,s,b,t}`).

## Scope

Twelve unclaimed P2 ledger entries plus four test ports from the audit snapshot's
untracked suite. The fifth source file (`test_lifecycle_fixes.py`) belongs to the fixc
line and was not touched.

## Fixes — one commit per bug, each with a targeted regression test

| Ledger entry | Commit | What changed | Test |
|---|---|---|---|
| SEED-22 | `a440a8b` | `parse_eval` treats a non-finite timing reading as an absent one (same bar as the probe path), so NaN can no longer hijack `_best_ms` ordering, plateau checks, or the band pool | `tests/test_loop_parse.py` |
| SEED-50 | `c86bf7a` | `PROBE_REQUEST`/`EXHAUSTED` markers anchor to their own line; prose citing a marker no longer routes as a request | `tests/test_loop_parse.py` |
| SEED-20 | `caddec3` | `remote_tree` keys the remote job tree by the clone (op path hash), not by two directory basenames; same-named clones stop erasing each other | `tests/test_direct_delivery.py` |
| SEED-27 | `99e62df` | `self_evolve` guards the noise-floor denominator (a zero-ratio generation reports "nothing to optimize") and refuses a doomed chain start before any job runs | `tests/test_self_evolve.py` |
| T5C-02 | `4feb8d1` | eval-export `best_keep` falls back on a missing `cluster` key exactly like its `sha` sibling; a mixed-shape trajectory no longer dies in a bare `sorted` | `tests/test_eval_export.py` |
| SEED-24 | `d16b05e` | the precision measure zip is `strict`: unequal-length outputs fail closed into unavailable rows instead of truncating to a silent pass | `tests/test_precision_strict_zip.py` |
| T5C-03 | `1eb606a` | a junk replay-probe timeout env value falls back to the default instead of crashing the bench after every timing window closed | `tests/test_replay_probe_timeout_env.py` |
| SEED-34 | `9704673` | a malformed `part-k-of-n` declaration gets one semantics at both sites: consistent refusal, no free-eval channel | `tests/test_pkg_decl_validity.py` |
| SEED-49 | `3185996` | a quarantined KB entry keeps its tree's name; same-named entries from different subtrees stop overwriting each other | `tests/test_kb_quarantine_names.py` |
| SEED-33 | `928f204` | seeding reads and writes pin utf-8 (plus the same-family unguarded write), so a non-UTF-8 locale cannot crash recon at startup | `tests/test_seeding_utf8.py` |
| SEED-32 | `77a29d6` | the role exposure table gets the dossier's guard: FULL writes check size/identity, and a head tree never overwrites the best tree's uniquely named file | `tests/test_corps_fix_driver3.py` |
| SEED-35 | `ca81fe0` | the meta verdict probe and the manifest probe degrade locally on a backend error, matching their sibling probe paths; one GPU flake no longer kills the run | `tests/test_corps_fix_driver4.py` |

T5C-02/T5C-03 were discovered by this line's C-class sweep of the post-WP25 code
(verdicts and repro scripts recorded in the ledger and `handoff/findings/triage-T5.md`).

## Test ports (from the audit snapshot's untracked suite)

- `ff52c8d` `test_monitor_resume_fixes`: 1 ported (digest clock telemetry, pinned in its
  shipped WP15 shape), 16 deferred — each deferred test travels with the fix line that
  owns its defect (monitor/resume line).
- `b15cf39` `test_guard_fixes`: 1 ported (precision stress row measures the task
  distribution, WP20 semantics), 10 deferred (guard/reference purity defects owned by
  the fixb line).
- `0bd17ff` `test_driver_measurement_fixes`: 10 ported (they pin the nine fixes the
  corps landed first: re-anchor gap guard, never-evaluated REVERT neutrality, cross-job
  band pricing, old-best dissent, free-pass comment, safety-net traceback, drift
  baseline cluster), 3 dropped — reasons in the commit body (e.g. the data-poor
  free-pass assertion is false under the WP15 confirmation redesign).
- `test_cli_submit_fixes`: 0 ported / 12 deferred — every pinned defect is still alive
  at this base and owned by the submit/loop-cli lines; the verdict is recorded in that
  file's commit on `feat/corps-fix-f9-t` and deliberately not re-committed here as an
  empty marker.

Deferred tests port together with their owning lines' fixes; per-test verdicts live in
the sub-line commit bodies.

## Frozen surface

Untouched: `timing.py` `measure_paired`, precision constants (`K_RATIO`/`FLOOR`), the
`meta/` discipline text, gate adjudication semantics, `LICENSE`/`NOTICE`. The strict-zip
and timeout-guard fixes are structural, not constant or adjudication changes. The
`resolvable_diff` residual (SEED-04) stays report-only per the ledger's frozen list.

## Validation

Full suite green on each sub-line before each commit (2717–2732 passed as lines grew).
Integration checkpoint at ten commits (through `1eb606a`): **2741 passed, 0 failed,
0 errors** in 11m13s. Final regression over the fifteen-commit tip recorded in the
ledger. Worktrees clean; nothing pushed.
