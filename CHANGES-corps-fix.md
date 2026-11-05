# CHANGES-corps-fix

Audit-corps correction order (2026-11-05): every review and fix in this wave targets the
**latest** repository, peregrine-kernel (`src/peregrine_kernel/`), based on `origin/main`
= `8977b63`. Branch `feat/corps-fix`, not pushed, not merged. The legacy upstream
snapshot tree was out of bounds for this entire wave; the corps' earlier findings made
there were re-convicted on this codebase one by one before any fix landed.

## Triage outcome (61 legacy findings re-convicted on latest)

| Verdict | Count | Meaning |
|---|---|---|
| A — already fixed here | 5 | `BUG-SEED-04` (WP15 layered-σ estimator rewrite), `BUG-SEED-30` (clock/gpu telemetry now consumed), `BUG-SEED-47` (WP15 operator-agnostic correctness detail), `BUG-TST-01` (test_cluster_aware pins current semantics), `BUG-C5-01` (task-suite precision block re-attached after WP24's rename drop, comment in situ) |
| N/A — mechanism removed | 3 | `BUG-C1-01` (old band guard removed by WP15), `BUG-COORD-1` (tests/test_guard_fixes.py does not exist here), `BUG-SEED-37` (problem.json no longer parsed by task.py) |
| B — survived, fixed here | 52 | re-located to fresh file:line, empirically reproduced where feasible, one commit each |
| C — new-generation bugs, fixed here | 8 | found while re-convicting (`BUG-T1C-01`, `T2C-01`, `T2C-02`, `T4C-01`, `T5C-01`, `T6C-01`, `T6C-02`, `T7C-01`) |

Every B/C verdict was verified on this codebase (symbol re-location + minimal
reproduction); no old line numbers or conclusions were transplanted. Evidence trail:
`handoff/triage-wk6/triage-T1..T7.md`.

## Fixes (60 bugs, one commit each, `fix(audit-corps): ...`)

Measurement & adjudication (loop/driver.py, loop/parse.py):
- P0 `BUG-SEED-01`: `_paired_reanchor` no longer formats `gap=None` (resume +
  workloads-mismatch path); regression test pins both None and float cases.
- `BUG-SEED-02`: the four never-evaluated REVERT paths stop claiming
  `correctness=False` (now `None`), so a never-evaluated workspace no longer drops
  into scaffold NEUTRAL, across restarts too.
- `BUG-SEED-03`: the stale "free pass" comment on the WP1 band branch is honest now.
- `BUG-SEED-05`: `_judge_patch`'s absolute-ms branch prices past the cross-job 6% band
  (`_GATE_REGRESSION_TOLERANCE_MS`), aligned with `_gate_regression`; two tests that
  pinned the old same-job band were re-anchored (-3.8% → REVERT, -15.4% → KEEP).
- `BUG-SEED-06`: a confirmation draw landing past the old best dissents instead of
  confirming; the band returns to the suspect draw.
- `BUG-SEED-22` / `BUG-T4C-01`: eval readings (list and scalar) drop non-finite values
  and reject bool/str, so a NaN or `"device_median_ms": true` cannot poison
  plateau/gate accounting.
- `BUG-SEED-29`: the unattended-run safety net logs the traceback, not just the repr.
- `BUG-SEED-31`: the SM-clock drift baseline only reads the sitting best of the same
  cluster.
- `BUG-C1-02`: a bad byte in a student probe degrades the probe instead of stopping
  the run (utf-8 + `(OSError, UnicodeError)`, matching the `_timeline_*` precedent).
- `BUG-C1-03`: cluster anchors are consumed only for the gate window they were read in.
- `BUG-T1C-01` (minimal): `resolvable_diff` excludes reused readings
  (`reading_of is None`) from its sample pool. Deliberately NOT done here (reported):
  between-pool sha dedup / de-sloping, `combined_workload_test` filtering, v1 parity.

Guards & kernel loading (the cheat-surface):
- `BUG-SEED-07`: tree-root shadow modules (`sitecustomize.py`, `json.py`, stdlib and
  judge-critical third-party names) are harness; delivery/hygiene/overlay-drop/
  preflight all route through the widened pathspecs. Kernel siblings stay legal
  (load_entry documents that intent).
- `BUG-T2C-01`: gpu_entry keeps the monitor-hook directory ahead of `$WORKSPACE` on
  PYTHONPATH, so a student shadow module can no longer unseat the hook.
- `BUG-SEED-08`: the library import scan reads every name on the line, plus
  `import_module`/`__import__` and parenthesized continuations.
- `BUG-C3-01`: the library scan names the torch C++ API spellings (`torch::`, `at::`,
  `c10::`), generated from the same name table as the Python patterns.
- `BUG-SEED-09`: `load_entry` appends (no longer prepends) the kernel directory to
  sys.path; a sibling cannot shadow the judge's imports.

Process, ssh channel, cli:
- `BUG-SEED-10`: an interrupted agent turn ends its whole process group.
- `BUG-SEED-11`: the ssh job deadline starts when the job begins (BEGIN marker moves
  after the host lock), so queue time cannot fake a FAILED.
- `BUG-SEED-12`: an ssh drop after begin (rc=255) is `STATUS_LOST` + `_stop_remote`,
  not a charged Failed with an orphan holding the lock.
- `BUG-SEED-20`: remote job trees carry the op tree's identity (sha digest), not a
  colliding basename.
- `BUG-T2C-02`: the ssh extract step has a sized timeout and rides the transport
  channel.
- `BUG-SEED-18` / `BUG-SEED-48`: retry budgets say what they spend and stop sleeping
  after the final failure; a hung submit is no longer retried 3x600s.
- `BUG-SEED-21`: a poll that never reaches a terminal state counts toward giving up.
- `BUG-SEED-19`: a failed harness-overlay build refuses the reading (M3-lane
  contaminated-record precedent) instead of silently re-judging.
- `BUG-C3-02`: reference label keys (`ref_name`/`ref_shape`/...) no longer read as a
  declared reference; target-ms-only runs get "n/a", not `EXIT_WINDOW_UNDECLARED`.

Lifecycle (launch/monitor/resume/agent_policy):
- `BUG-SEED-13`: a run is DONE only when its final confirmation met the target — the
  stop word alone cannot bypass a correctness alarm.
- `BUG-SEED-14`: a monitor whose hook never injected reports the injection failure
  instead of an "ok" misattributed to the bench.
- `BUG-SEED-15`: a final marker without its JSON line falls back to the per-profile
  rebuild instead of dropping every finding.
- `BUG-SEED-16`: the reopen closure pulls in companions named by a reopened decision
  (symmetric closure) and one-sided ⊗ declarations are warned.
- `BUG-SEED-17`: command findings dedup by command text, so the teacher leg's policy
  findings survive.
- `BUG-SEED-41` / `BUG-SEED-42`: a hook dump that cannot write leaves a stderr trace;
  merged hook calls carry pid + cross-process index.
- `BUG-SEED-45` / `BUG-T6C-01`: the agent-policy hook blocks on unreadable or
  non-object events/policies instead of failing open.

Bench/eval (card_template/eval):
- `BUG-C5-02`: a Python kernel's import-time failure lands a structured
  `IMPORT_ERROR` verdict (same shape as CUDA's `COMPILE_ERROR`) instead of a bare
  crash with no JSON.
- `BUG-C5-03`: an empty `input_groups` workload fails with `INPUT_ERROR` (never a
  0-check fake PASS); ncu_window refuses empty pairs instead of dividing by zero.
- `BUG-T7C-01`: the fresh-recheck child prints its `pass=null` verdict even when the
  kernel fails to re-import (WP29 replay-probe precedent).
- `BUG-SEED-24`: the precision measure zips with `strict=True`.
- `BUG-SEED-23`: `host_path_note` picks thinner/thicker by the actual direction.

Misc (driver evidence, seeders, scripts, tests):
- `BUG-SEED-32`: role-exposure tables get the dossier's overwrite guard; head≠best
  tables stop colliding on one filename.
- `BUG-SEED-33`: toolkit seeders pin `encoding="utf-8"` (C locale cannot kill the
  loop's first step).
- `BUG-SEED-34`: malformed `part k of n` declarations share one well-formedness bar on
  both the scaffold and HOLD-pricing paths.
- `BUG-SEED-35`: a verdict probe whose backend commit/push/submit fails degrades to
  NO VERDICT like its sibling probes.
- `BUG-SEED-49`: quarantine collisions flatten the relative path instead of silently
  replacing the occupant. `BUG-T5C-01`: that OSError log carries dst and error
  separately.
- `BUG-SEED-26` / `BUG-SEED-27` / `BUG-SEED-36` / `BUG-SEED-38` / `BUG-SEED-39` /
  `BUG-SEED-40` / `BUG-SEED-44`: milestone validate anchors at line start; self_evolve
  reports ratio=0 and chain collisions readably; task names are validated against the
  load-side rules; git status/birth failures are loud; `_Tee` closes its handle;
  `_submit_cluster` catches narrowly and warns; a hung deliverable check ends as a
  readable rc=124 with the session saved resumable.
- `BUG-SEED-46` / `BUG-C11-01`: two tests that asserted nothing (a stub never called;
  a key compared with itself) now assert the real thing.

## Legacy regression tests ported (5 archived files → tests/test_corps_port_*.py)

Per the order, every assertion was first checked against current semantics:
**9 ported, 2 rewritten, 60 dropped** — drops are either covered by this wave's fresh
tests (`tests/test_corps_fix_*.py`), or their premise was removed (WP24 de-boarded
leaderboard cards) or inverted by a fix above (e.g. queue-timeout→Lost no longer
exists; overlay failure is now fail-closed). Commits `d26a311`, `b56a311`, `c1a269c`,
`97b8743`; full per-test rationale in those commit messages.

## Frozen surfaces — untouched, reported only

`timing.py measure_paired`, precision constants, `meta/` discipline text, gate
semantics (`_gate_regression` / gate windows), LICENSE/NOTICE: zero changes. The
closest approach is `BUG-SEED-05`, which touches `_judge_patch` (patch pricing, not
the gate) and only aligns it to the existing cross-job constant; its two pinned tests
were re-anchored rather than deleted. Statistical-threshold defaults unchanged.

## Residuals reported, not fixed (need a chief decision or a later lane)

1. `BUG-T1C-01` remainder: WP15 v2 between-pool dedup / de-sloping,
   `combined_workload_test` filtering, `_resolvable_diff_v1` parity — estimator
   semantics, deliberately left.
2. `workloads_performance` (card_template/eval/timing.py) raises TypeError
   (`round(None)`) on non-positive host/device readings — the legacy "degrade to None"
   aggregation is missing on this line (found while porting).
3. `launch.status_text` still bare-`json.loads`es status.jsonl — torn lines crash,
   asymmetric with `load_records`' new tolerance (found while porting).
4. Encoding audit beyond the seeding family: driver evidence IO, launch.py,
   taskcard.py still have unfixed unpinned-encoding spots.
5. `_run_probe_manifest_batch` and one bare commit/push near driver.py:12121 lack the
   sibling probes' local degradation.
6. Adjudication notes: SEED-36's name rule deliberately matches the load side (dots
   allowed inside names); `_submit_cluster` KeyError passthrough is the "visible
   exception" option of the triage.

## Verification

- Baseline at start: `2692 passed` (`@8977b63`).
- Cumulative full-suite checkpoints during the wave: 2701 → 2714 → 2757 → 2781 →
  2822 passed (independent detached-worktree runs).
- Every fix carries targeted regression tests (≈200 new, all English) and a
  mutation check (revert the fix → new tests fail → restore); 60/60 fixed bugs
  verified this way.
- Final full suite on this branch tip: see the closing report (must be green, 2692 +
  additions, zero failures).
