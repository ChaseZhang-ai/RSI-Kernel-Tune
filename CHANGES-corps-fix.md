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

<!-- ---- r2 line changelog (parallel wave, unioned at merge) ---- -->

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
## W1 corps wave (parallel fix line, dedup-merged into main)

A second corps line (W1) ran the same correction order in parallel: an eight-agent
triage of the legacy findings plus its own wave-1 discoveries against this
repository, then nine file-partitioned fix agents. Its raw commits were deduplicated
against the fixes above — for every finding this branch already carries, the W1
line surrendered its own implementation and kept this one — and only the faces this
branch lacked were re-landed here (41 cherry-picks):

- **kernel deliverables** (`kernels/`, untouched by the wave above): fp8_gemm cache
  keys to full tensor identity with held references (the fp4 `_tensor_identity`
  discipline), conv3d_fwd no longer pip-installs + `os.execve`s at import, from_agent
  / fp8 two-slot result rotation, dsa_sparse_bwd k2 key carries `sm_scale`,
  empty-sequence NaN guards in the two paged-decode kernels, b200 shape validation,
  dsa_score_recompute epilogue predicates, gen0 key carries device/dtype/strides.
- **the black-box guard vocabulary** (`scripts/blackbox-guard.sh`, untouched): a bad
  `BLACKBOX_EXTRA` regex now fails closed at source time, `--set full` /
  `--section SourceCounters` / `--page source` and the attribute-chain introspection
  spellings are denied, the guard's own file is `realpath`ed, and `\bstrings\b` no
  longer bites "f-strings".
- **scripts the wave above does not reach**: `agent-claude.sh` goal branch carries
  `PK_GOAL_TIMEOUT`, `claude-goal.py` refusal exits honor the stdout contract and
  lose the dead `met_seen_at`, `timeout-run.py` maps signal deaths to 128+n, the
  two skills recipes fail with a verdict line instead of an IndexError traceback,
  `self_evolve` distinguishes infra failure (rc 3) from a natural chain end,
  `baseline_cli --from-json` reports a broken archive cleanly.
- **monitor CLI**: `--kernel-file` resolves from the card, and the monitor hook's
  `_chain` never executes a sitecustomize planted in the workspace (defense in
  depth on top of the gpu_entry PYTHONPATH fix), tmp dump names are thread-scoped.
- **cli/launch seams**: a config-phase `SystemExit` lands FAILED in the run record;
  `peregrine status` survives a torn status.jsonl line; `hours` is documented as the
  whole loop wall clock in both places.
- **parse/inner/judge/misc**: `_correctness_error` reads tensor names from the bench
  JSON (SEED-47's dq/dk/dv hardcode was still present here), `_gate_ratio` rejects
  non-finite ratios, "PROBE PASSED" keeps its max_diff capture, `meta_trajectory`
  validates outcomes at the update entry, `judge` spans every shape's axes,
  `window_contract` reports a malformed shape instead of crashing, `goal_ref`'s
  accepted-reference declarations land under their own window's key.
- **driver** (W1-discovered, not in the legacy ledger): teacher `ablation_rounds`
  prose degrades instead of killing a 20h run; the in-place reopen checkpoint is the
  cluster-aware best; the paired re-anchor memo expires like the absolute channel's;
  a paid-for parity hit rebuilds into the paired memo across restarts (one pinned
  test re-anchored, see below).
- **tests**: the archived fix-wave batch's W1-relevant sections were ported and
  rewritten per the current tree; seven new `test_w1_*_fixes.py` files carry the
  faces above.

Surrendered to this branch (kept yours): SEED-07/08/09/10/11/12/13/14/15/16/17/
19/20/21/22/23/24/25/26/27/28/31/32/33/35/36/38/39/40/41/42/43/44/45/46/48/50 and
C1-02/C1-03/C3-01/C3-02/C5-02/C5-03/T2C-01/T2C-02/T5C-01/T6C-01/T7C-01. Two policy
notes for the reviewer: SEED-18 — this branch documents the clock-lock budget as
attempts-total and declares the lost_retries asymmetry intentional; W1's alternative
(counter unification, `>=` → `>`, one more resubmit) is preserved on branch
`feat/corps-fix-w1-driver` for adjudication. SEED-45 — this branch makes the
agent-policy hook fail closed on unreadable events; W1 had judged fail-open correct
and only documented it. Reported, not fixed: SEED-04's trend de-sloping half (the
reuse-pool half landed as T1C-01), and the frozen-surface items W1-DRV-03
(median-vs-min residual) and W1-DRV-05 (voided trees still set the absolute-branch
bar) await adjudication.

Full suite on this branch: **2976 passed**, 0 failed.

Lifecycle / cli cluster, B-class fixes from the R2 audit triage (base 8977b63).
Each fix: reproduced red, fixed, repro green, one commit. Regression numbers in
the final report; branch left unmerged for the chief engineer.

## P1

### BUG-SEED-10 — an interrupted agent turn leaves the agent's whole session alive
`loop/cli.py` `_agent_turn`: the Popen now starts the agent in its own POSIX
session (`start_new_session`), and the interrupt branch hands the process to
`_end_process_group` (SIGTERM the group, SIGKILL the rest) instead of
`proc.kill()` — which only reached the shell and left the model process burning
API usage. Test: `test_loop_cli.py::test_an_interrupted_agent_turn_ends_the_whole_session`.

### BUG-SEED-13 — stop=target bypassed the correctness alarm / the confirmation
`launch.py` `drive()`: DONE now follows `met` (the confirmed-at-target check,
which already carries the alarm veto) alone; the loop's own `stop == STOP_TARGET`
may no longer vouch for the target. Note: the original ledger suggestion
`stop == STOP_TARGET and met` was NOT applied — it would break the legitimate
"budget ran out after the target was met" DONE, which
`test_launch.py::test_a_confirmed_best_at_the_target_completes_the_run` pins.
The five launch fixtures that pinned the buggy behaviour (stop=target, no
confirmation, exit 0) now supply a meeting confirmation. Report line no longer
withholds the confirmation sentence from a confirmed stop=target run. Tests:
`test_lifecycle_fixes.py::test_stop_at_target_*` (3).

### BUG-SEED-17 — the teacher's policy findings were dropped wholesale
`loop/agent_policy.py` `command_findings`: dedup moved from the fixed
"agent command" source string to a (source, text) pair, and the caller names
each leg (`student agent command` / `teacher agent command`) so a refusal says
which leg is at fault. A duplicated command is still reported once; the second
leg is always examined. Tests: `test_agent_policy.py::test_every_command_is_examined_not_just_the_first`.

## P2

### BUG-SEED-18 — clock-lock retry budget docstring
Aligned the prose to the pinned behaviour: `clock_lock_retries` counts submits
IN TOTAL (1 first + 2 resubmissions for the default 3), unlike `lost_retries`
which names resubmissions after the first job. Chose the docstring side because
the total-submits behaviour is pinned by
`test_clock_lock_refusal.py::test_a_gpu_that_keeps_refusing_stops_the_submit_without_a_verdict`.

### BUG-SEED-19 — overlay build failure was fail-open
`_with_job_env` re-raises now: a job whose harness/job-env overlay cannot be
built is refused (fail-closed) instead of silently running the tree's own judge.
The timeline PROBE overlay's documented local degradation (driver.py) is a
different path and is unchanged. Test:
`test_harness_guard.py::test_an_overlay_that_cannot_be_built_refuses_the_job`.

### BUG-SEED-21 — a poll timeout never counted toward giving the job up
Both poll paths (foreground `_submit_gpu_once`, lane `submit_gpu_finish`) feed
the same give-up counter as an unreadable log; after `_INFLIGHT_LOG_ATTEMPTS`
misses the in-flight record is dropped. One aborted poll still leaves the job
adoptable (existing guard unchanged). Test:
`test_loop_cli.py::test_a_poll_timeout_counts_toward_giving_the_job_up`.

### BUG-SEED-48 — a hung submit cost ~32 minutes of wall clock
Three behaviour changes, all inside the retry plumbing:
1. `_retry_call` catches only `(OSError, subprocess.SubprocessError)` — other
   exceptions are call-site bugs and propagate (was bare `except Exception`).
2. No backoff sleep after the final failed attempt (was sleep-then-raise).
3. Submits run under a dedicated `_SUBMIT_TIMEOUT_S = 120.0` (was the 600 s
   `_forge` default), bounding the retry budget at roughly 6 minutes instead of
   ~32 (3 x 600 s + tail sleep). A submit is a local hand-off; the job's own
   clock is the `[job]` timeout. Tests: `test_loop_cli.py::test_retry_call_*`,
   `test_loop_cli.py::test_a_hung_submit_retries_within_minutes_not_half_an_hour`.

### BUG-SEED-36 — task names were not validated at creation
New `scaffold.task_name_problems` (one plain path component, `[A-Za-z0-9]` lead,
no path separators, no leading `.`/`_`, not a CLI command name) guards
`cmd_init`, the `--from-kernels` path, `scaffold.main`, and
`scaffold.new_operator` (ValueError on the API path). Refusals are English, one
line per problem, exit 2 before anything is written. Test:
`test_lifecycle_fixes.py::test_init_refuses_an_illegal_or_reserved_task_name`.

### BUG-SEED-38 — a broken git read as "nothing to commit"
`taskcard._commit` checks `git status --porcelain`'s returncode and raises
`CalledProcessError` (stderr attached) instead of silently skipping the commit.
Test: `test_lifecycle_fixes.py::test_commit_loudly_refuses_when_git_status_fails`.
(The `launch.kb_changes` half of the triage entry belongs to the fixd session
and is untouched here.)

### BUG-SEED-39 — run-log tee handles leaked per drive
`_Tee.close()` closes only the run.log handle (never the wrapped stream —
`__getattr__` would have delegated it), and `_entered` closes both tees on the
way out. Test: `test_lifecycle_fixes.py::test_entered_closes_the_run_log_handles`.

### status query crashed on a torn status line
`launch.status_text` skips unparseable `status.jsonl` lines (and reads
errors=replace) — `peregrine status` is a query, not an audit of the loop's
last words. Test: `test_lifecycle_fixes.py::test_status_skips_a_half_written_line`.

### BUG-SEED-40 — the baseline cluster stamp degraded on anything, in silence
`baseline_cli._submit_cluster` catches only `(OSError, ValueError, KeyError)`
(the config-reading classes) and prints a `[baseline] note:` naming the op and
the consequence (cross-machine guard off); anything else propagates. Tests:
`test_lifecycle_fixes.py::test_submit_cluster_*` (2).

### BUG-SEED-44 — a hung deliverable check killed the goal leg
`scripts/claude-goal.py`: watch() reads the check through
`deliverable_or_note()` (a hung check is "not confirmed" plus a note, the
branches carry on), and run() keeps a belt that ends the leg as `check-hung`
(rc 124) so save_session always runs; a check-hung session is saved resumable
(the turn itself likely finished — only its check wedged). Tests:
`test_claude_goal.py::test_a_hung_deliverable_check_*` (2).

### BUG-SEED-45 — the policy hook was fail-open on an unparseable event
`scripts/agent_policy_hook.py`: an event that does not json-parse now blocks
(exit 2, BLOCKED on stderr), symmetric with the unreadable-policy branch two
lines up. Empty stdin is still "no event, no opinion" (exit 0). Test:
`test_agent_policy.py::test_the_hook_blocks_when_the_event_does_not_parse`.

### BUG-SEED-46 — the self-evolve "runs it" test verified nothing
`tests/test_self_evolve.py`: the fixture records `cmd_run`'s argv next to
`cmd_new`'s; the generation tests assert one run per generation (and that a
chain ending without a best still ran its first generation).

### BUG-SEED-23 — the host-path note said "thinner" for a thicker wrapper
`timing_audit.host_path_note` picks thinner/thicker from the ratio's direction
(the docstring always promised the thicker direction to be visible). The
the tests that had pinned the backwards word are corrected (in
`test_timing_audit.py` and the end-to-end
`test_loop_driver.py::test_a_span_basis_round_is_not_audited_as_host_bound`,
whose fixture's enqueue path is the thick one); a both-directions case added
(`test_timing_audit.py::test_the_host_path_note_names_both_directions`).

### BUG-C3-02 — label-only reference keys declared a reference
`_ref_rides_job` rides only a plain `ref_cmd`; `_arm_has_reference` counts only
the delivery keys (`ref_cmd` / `ref_mode` / `submit_op` routing) besides the
architecture-reference channels. The label keys (`ref_name` / `ref_shape` /
`ref_hardware` / `ref_ncu_cmd`) describe a reference without delivering one, so
a label-only loop.toml no longer sends a target-ms-only run into
EXIT_WINDOW_UNDECLARED or the driver's in-job-ratio path. Residual, out of
scope: a label-only spec still passes the truthiness checks at the two
`_measure`-adjacent call sites (`cli.py` reanchor / confirm re-anchor); those
fire only when a measurement is actually attempted. Tests:
`test_loop_cli.py::test_label_only_ref_keys_declare_no_reference`,
`test_ref_rides_job_judgment` extended fixture key (`ref_cmd`).

### BUG-C11-01 — digest median asserted against itself
`tests/test_wp1_measurement.py`: the "medians stay" assertion compares the
digest's per-workload median against the SOURCE record's (mutation-checked:
nulling the digest median fails now; the old self-comparison passed).

## Archived test batch (test_lifecycle_fixes.py) disposition

Ported, green (9): stop-at-target x3, tee handles, torn status line, init name
validation, commit-loudly, submit-cluster x2.

Dropped — board lane removed from peregrine_kernel (no
`--leaderboard` creation channel, no board card reader, no "unexpected
structure" text anywhere): `test_board_init_refuses_an_illegal_or_reserved_task_name`,
`test_a_malformed_board_card_is_a_task_error` (x4 param),
`test_new_board_task_refuses_a_card_that_does_not_read`.

Not ported here — fixes owned by another session (fixd), noted so they are not
lost: `test_kb_changes_warn_instead_of_staying_silent` (launch.kb_changes),
`test_baseline_show_reports_a_bad_archive` (x2, rc-4 wrapper),
`test_baseline_from_json_reports_parse_failures` (x2),
milestone line-start anchor tests (x4).
<!-- ---- f9 line changelog (parallel wave, unioned at merge) ---- -->

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
# CHANGES-corps-fix — the audit-corps fix-d batch (monitor / resume / trajectory)

Branch `feat/corps-fix-d`, cut from origin/main `8977b63`. Nine B-class bugs
from the R2 audit (the shared handoff ledger, triage T2/T3/T4), fixed one per commit,
each reproduced first via the ported regression pins (red on this base), then
turned green.

## 1. The bugs and the fixes

| Bug | Verdict | Commit | What was wrong | What changed |
|---|---|---|---|---|
| BUG-SEED-14 (P1) | fixed | `83aecda` | `monitor.classify` had no consumer for run.py's per-profile `hook.processes` count; a failed sitecustomize injection read as info "the bench does not import the kernel as a module" (the wrong cause) and info does not raise severity, so a passing card read `status: "ok"` while the NaN / input-shift checks had never executed | classify adds its own evidence: `processes == 0` in any profile that ran yields one warn (profile `"-"`) naming the profiles and the injection chain; the misleading info only fires when the hook itself was alive in the base run; the card's own verdict still dominates (a failed base stays alarm) |
| BUG-SEED-15 (P1) | fixed | `32a3df5` | run.py printed the final MARKER and the result JSON as two prints; a kill between them left MARKER over no payload, `parse_log` returned None and the per-profile rebuild (written exactly for cut-short jobs) was unreachable — every earlier profile's alarm-level finding was lost | both halves: run.py writes marker + JSON in one `write()` + flush; `parse_log` falls through to the per-profile rebuild whenever `_json_after` comes back empty (rebuilt partial keeps `complete: False` and the "cut short after N profile(s)" warn; a complete final JSON still wins) |
| BUG-SEED-16 (P1) | fixed | `33d19f8` | `compute_reopen` only pulled companions named by the candidates themselves (`d.companions & reopen`): with A declaring ⊗ B and B silent, reopening B kept A — the pair split across the reopen, the kept half's evidence describing the old tree; `dependency_warnings` (moved to milestone.py) checked only ⇒/⇐ echoes, so a one-sided ⊗ passed silently | the fixpoint also admits a decision NAMED by any reopened member (symmetric closure, chains close from any entry point; ⇒/⇐ untouched); `dependency_warnings` flags a one-sided ⊗ with the same message shape as the ⇒ echo; the contradictory Decision/ReopenPlan docstrings aligned (companions widen the reopen closure symmetrically; the `compute_resume` prefix cut still ignores them) |
| BUG-SEED-41 (P2) | fixed | `b2c045d` | `pk_monitor_hook._dump` wrapped the whole write path in `except Exception: pass` — an unwritable PK_MON_OUT lost every summary with zero trace, bypassing the module's own `_err` convention (which depends on `_dump` succeeding) | a failed dump is recorded via `_err`, said on stderr once (module-level flag), and the payload gets a last-resort write to the temp dir (`pk-mon-dump-fallback-<pid>.json`) so the data is not simply gone |
| BUG-SEED-42 (P2) | fixed | `6980a65` | the hook counts each call's index `i` from zero per process but call records carried no pid, and `_calls` deduplicated on `(fn, i)` — merged multi-process records (DataLoader workers / mp) let the processes overwrite each other | the hook writes `pid` into every call record; `_calls` keys on `(pid, fn, i)` (pre-pid records share the None key and read exactly as before); cross-run joins use a pid-insensitive view (`_base_calls` / `_call_key`, `_in_sig` keyed `(fn, i)`) — two profiles are different processes by construction, so comparisons match calls by what they were, not who held them |
| BUG-SEED-43 (P2) | fixed | `ce142b6` | `load_records` fed every non-empty line of status.jsonl to a bare `json.loads`; append_record is a plain append, so one torn tail line made the whole trajectory unreadable and every resume impossible | a torn FINAL line is dropped with a stderr warning ("torn"); a broken MIDDLE line still raises (a hole mid-history is corruption; silently resuming over it would build the prefix-reuse plan on a trajectory that never existed); the file is read utf-8 with `errors="replace"` so a cut multibyte character cannot raise before the parser sees it |
| BUG-SEED-25 (P2) | fixed | `f888397` | `_DIFF_SIG` matched only decimal fractions, so scientific-notation diffs lost their exponent: `1.5e-05` and `1.5e-07` shared the signature "1.5" and three genuinely different failing attempts fired `constant_output_flag` (driver.py production consumer) | the numeric branch admits an optional exponent (and bare integers like `max_diff=0`); inf/nan unchanged; bit-identical repeats still flag |
| BUG-SEED-26 (P2) | fixed | `0eb84f6` | `validate_milestone` located REQUIRED_SECTIONS markers with `text.find`: "### 4. M2 pipeline" contains the substring "## 4.", so a milestone missing the real "## 4. Known dead ends" passed on the strength of a numbered subheading (the same trap `_section_body` already fixed for the decisions cut) | the presence check uses the same line-anchored `_heading_at`; presence and order are judged on real heading lines |
| BUG-SEED-28 (P2) | fixed | `2709bd6` | `parse_correctness` read any "N/M FAILED" marker as a fail (including "0/5 FAILED" — zero failures); `_first_json_with_key` hand-counted braces without knowing JSON strings, so a `}` inside a string value broke the depth and quietly returned None for the whole output | the count decides (`passed = fails == 0`); the scan goes through `JSONDecoder.raw_decode` (the idiom run.py already uses); `parse_perf`'s empty result carries a `note` naming which diagnosis it was (marker present but unparseable vs no marker) |

Two further commits carry the ported regression suite:

- `65dc63b` — `tests/test_monitor_resume_fixes.py`: the archived pre-rename fix
  batch's 20 pins, rebranded to this package. 15 were red on `8977b63`
  (reproduction evidence for the table above), 5 were green and pin the
  behavior the fixes must not disturb (complete-JSON precedence, live-hook ok,
  old records without a `processes` verdict, middle-line raise, pid-ignorant
  joins).
- `06b2c84` — the ported file's docstring reworded to stay outside the WP25
  branding gate's forbidden-pattern surface (it named the archive paths
  verbatim).

One commit carries the digest telemetry half of the ported pins:

- `988aa34` — `build_digest` stamps each curve point with its row's
  `clock_drift` flag and adds the best row's `sm_clock_mhz` to the achieved
  block (omitted when no gpu_state). Report-only; the archived batch surfaced
  both, this tree carried only the `clock` header. Report-only telemetry, no
  gate reads it.
- `bcd7674` — retires the `IterationRecord.clock_drift` line in
  `tests/dataflow_audit_baseline.json`: the curve stamping is the field's
  first read point, and the dataflow audit's own rule ("registered entries are
  stale — the consumer has landed, delete the line") enforces the deletion;
  `test_annotations_are_current_and_substantiated` failed on it until the line
  went. `gate_spread_pct` / `ref_host_enqueue_ms` / `variant_timings` stay
  registered (still readerless).

## 2. Deliberate decisions

- **SEED-43 middle-line semantics**: the corps fix-d brief's prose (a failing
  middle line is dropped with an alarm, without blowing up the whole run) and
  the archived pin
  (`test_load_records_still_raises_on_a_broken_middle_line`, endorsed by
  triage-t5: pin that the previous fix did not loosen the middle segment)
  disagree. The pin wins: a middle line failing to parse still raises. A
  silently skipped middle row would make resume decisions on a trajectory with
  a hole; tolerance is exactly the torn tail.
- **test_dependency_warnings_flag_one_sided_downstream_claims extended**: the
  existing test asserted the exact warning list, and the `_ARROW_MS` fixture
  genuinely carries a one-sided ⊗ D0.3 — which is now (correctly) warned. The
  ⇒ assertion is kept verbatim as a membership check; a dedicated one-sided ⊗
  pin was added.
- **Digest clock telemetry** is not one of the nine ledger bugs, but the
  ported pin (`test_digest_curve_carries_clock_drift_and_best_sm_clock`)
  requires it; implemented as report-only additions, no gate semantic touched.

## 3. Territory and freeze compliance

- Touched files: `monitor/__init__.py`, `monitor/runtime/run.py`,
  `monitor/runtime/hook/pk_monitor_hook.py`, `resume/invalidation.py`,
  `milestone.py`, `inner/trajectory.py`, `inner/results.py`, plus tests.
- No forbidden file touched (loop/driver.py, loop/parse.py, sitecustomize.py,
  loop/cli.py, launch.py, loop/agent_policy.py, taskcard.py, baseline_cli.py,
  timing_audit.py, scripts/ all byte-identical to `8977b63`).
- Frozen surfaces untouched: `timing.py measure_paired`, precision constants,
  meta/ discipline text, gate semantics, LICENSE/NOTICE.

## 4. Regression

Full suite (all of `tests/`, venv python 3.14.4, this tree's `src/` on
PYTHONPATH): **2715 passed, 0 failed** (614.86 s). The pre-batch base collected
2692; +20 ported pins, +2 new milestone pins = 2715 exactly.
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
# CHANGES — corps fix batch B (guards / adversarial surface)

Branch `feat/corps-fix-b`, base 8977b63. Six B-class bugs from the audit-r2
triage, each reproduced against the pre-fix code in /tmp before the fix and
re-run clean after; regression tests ported from the archived upstream fix
batch as `tests/test_guard_fixes.py`.

## Fixes (one commit per bug)

### BUG-SEED-07 (P0) — tree-root shadow modules hijack the harness — b2bfdff

A student delivery consisting only of `sitecustomize.py` / `json.py` at the
tree root passed all three guard layers (the delivery diff, the per-iteration
hygiene, the job overlay), rode every job tree, and was imported by the bench
itself: `gpu_entry.sh` put `$WORKSPACE` FIRST on PYTHONPATH, and the monitor
hook's `_chain()` chained whatever `sitecustomize` came next — the student's.

* `loop/harness_guard.py`: `ROOT_PY_PATHSPEC = "*.py"` joins
  `HARNESS_PATHSPECS` for the git calls (diff / ls-files take the wildcard;
  only the depth is narrowed, by `is_harness_path` — a root-level module is
  harness, the kernel file and `notes/` stay exempt, deep modules stay the
  student's). `tree_harness_files` lists the whole tree and narrows in Python:
  `git ls-tree` takes no wildcard pathspec.
* `cardkit.py`: the generated entry appends `$WORKSPACE/eval` and `$WORKSPACE`
  behind the inherited PYTHONPATH, so the monitor's hook directory (prepended
  by the runner) stays in front and a surviving root sitecustomize cannot
  overtake it. `-I` is NOT usable here — it would kill the sitecustomize
  injection mechanism itself.
* `benchkit.py`: the fresh-recheck / replay-probe children inherit the same
  order.
* `monitor/runtime/hook/sitecustomize.py`: `_chain()` drops student-owned
  sys.path entries (the deployed `<ws>/.pk_monitor/runtime/hook` layout derives
  the workspace) before chaining — only an image-shipped sitecustomize is
  handed over.

Reproduction: `/tmp/seedexp/exp_seed07.py` (root shadows passed all three
layers and rode the job tree; after the fix the delivery diff and the drift
name them and the overlay drops them), `/tmp/seedexp/chaindemo` (with the
workspace first on PYTHONPATH a bare `python3` ran the student's
sitecustomize; with the hook dir first and the fixed `_chain`, the hook
installed and an image-shipped sitecustomize was chained instead).

### BUG-SEED-08 (P1) — library scan escapes — 013d5a1

The scan matched one name after a line-anchored `(?:from|import)`;
`import os, flashinfer`, `a = 1;import flashinfer`, `importlib.import_module`,
`__import__`, `exec('import flashinfer')` and a `load_inline` cuda_sources
string all screened as library-free — a library-call kernel could become the
oracle. The import rules now walk an AST over the source as-is (every alias,
dynamic imports with literal targets, exec/eval recursed one level,
source-carrying keyword arguments scanned); an unparseable source falls back
to a statement-splitting regex (semicolons, continuations, parenthesized
from-imports) that checks every name. A variable target stays invisible.

Reproduction: `/tmp/seedexp/exp_seed08_09_c3.py` (all five escapes returned
`[]` pre-fix; all flagged post-fix; the innocent kernel's finding unchanged).

### BUG-C3-01 (P1) — torch C++ API forms unscreened — aee7adf

`kernel.cu` candidates computing through `torch::matmul` / `at::mm` /
`at::softmax` / `torch::ops::x` screened as library-free. The torch compute
and nn op lists moved to shared constants; `torch::/at::/c10::` and
`torch::ops::` rules join `_LIB_PATTERNS`, mirroring the python ones.

Reproduction: same script (all four C++ shapes returned `[]` pre-fix; flagged
post-fix; an innocent CUDA kernel stays clean).

### BUG-SEED-09 (P1) — load_entry shadowed the reference — c05a3f4

`load_entry` put the kernel's directory at `sys.path[0]`; the reference loads
first and runs later in the same process, so a kernel-directory `json.py`
resolved ahead of the stdlib for the reference's own lazy imports (measured:
a forged `helper` module resolved to `'kernel-forged'` instead of
`'reference'`). The directory is now appended — siblings still resolve, the
stdlib and site-packages stay in front.

Reproduction: `/tmp/seedexp/exp_seed08_09_c3.py` (shadowed True → False,
`ref_dir` now precedes `k_dir` on sys.path).

### BUG-C5-02 (P2) — python load failure: bare traceback, no JSON — e3d5a53

Only a `.cu` suffix failure was structured; a python kernel's module-level
syntax error / missing entry / broken sibling was re-raised and the generated
bench.py has no catch — the job died with a traceback and no `=== JSON ===`
block. `_impl(problem, "ref")` sat outside the try, so a broken reference
killed every mode the same way. Both load sites now land in the structured
channel: `.cu` keeps COMPILE_ERROR, python becomes IMPORT_ERROR, the
reference becomes REFERENCE_ERROR (pass=False, why truncated, rc 0).

Reproduction: `/tmp/seedexp/exp_c5_02_03.py` (pre-fix `RAW SyntaxError … NO
JSON AT ALL`; post-fix rc=0, json emitted, status IMPORT_ERROR; a reference
without its `make_inputs` yields REFERENCE_ERROR).

### BUG-C5-03 (P2) — empty input_groups judged PASSED — d46c156

`task_correctness` never entered its loop for an empty `input_groups` and
fell through to pass=True with 0 checks; `--mode all` handed timing an empty
set list (ValueError), and `ncu_window` raised ZeroDivisionError on
`calls // len(pairs)` before `max()` could mask it. Empty groups are now an
INPUT_ERROR verdict and the profiler exits rc 3. `timing.py` itself is
untouched — its explicit "needs at least one workload" ValueError is the
precedent this fix makes unreachable (`measure_paired` stays frozen).

Reproduction: `/tmp/seedexp/exp_c5_02_03.py` (pre-fix `pass=True, checks=0`;
post-fix `pass=False, INPUT_ERROR`; ncu window exits 3 on the same card).

## Test port — tests/test_guard_fixes.py — d0adef9

Ported (18 tests, all green): the four tree-root guard tests; the import-scan
tests (commas / semicolons / importlib / `__import__` / exec / unparseable
source / quiet-on-clean-and-CUDA); the C++ form and load_inline-string tests
(new, this batch); the load_entry append test; the IMPORT_ERROR /
REFERENCE_ERROR verdict-JSON tests and the empty-groups INPUT_ERROR / NCU
refusal tests (new); the gpu_entry PYTHONPATH order; the two monitor-hook
chain tests (new); the precision stress record (ported as-is).

Dropped from the archived file, with reasons:

* `test_fib_judge_refuses_a_non_finite_reference` — the board lane
  (`fib_judge`, `FIB_DEFAULTS`, `SUITE_TASK`) was deleted with the board
  (WP24); `task_error` covers the non-finite reference.
* `test_host_path_note_names_both_directions` — SEED-23, assigned to another
  engineer; porting it here would stay red. `timing_audit.py` untouched.
* The timing-aggregate trio (`_geomean` degrade, workload/ref aggregate
  degrade) — NOT ported: this tree's `_geomean` skips None entries and
  `workloads_performance` crashes on a 0.0 host reading
  (`round(None)`). That is the archived aggregate fix not carried over,
  inside the frozen `timing.py` surface (only the ncu divide point may move
  in this batch) — it needs its own task.
* `test_measure_with_mismatched_output_counts_is_unavailable` — NOT ported:
  this tree's `precision.measure` silently zip-truncates a length mismatch to
  `pass`. Same situation: the guard was never carried, precision constants
  are a frozen surface, and the fix is outside this batch's list.
* `test_task_stress_rows_draw_make_inputs_not_the_sol_heuristics` — pins
  `gen_inputs` routing the task's `make_inputs`; peregrine dropped that
  channel BY DESIGN (docstring: custom_inputs_fn is accepted and ignored; the
  task draws through `task_inputs`).

Also updated: `tests/test_harness_guard.py` — `evaluation.py` moves to the
harness side of the parametrize (a tree-root module), plus
`sitecustomize.py` / `json.py` cases.

## Intentional behavior changes to be aware of

* Students may no longer ADD tree-root `.py` files: the delivery check
  REVERTs them, the hygiene restore removes them, and the job overlay drops
  them. A card's own tracked root modules ride as before (byte-identical
  overlay). Sibling imports of the kernel still work (the kernel's directory
  is appended to sys.path by `load_entry`).
* gpu_entry.sh PYTHONPATH order: inherited front (monitor hook, image paths)
  → `$WORKSPACE/eval` → `$WORKSPACE`. A card relying on its root shadowing an
  image module would now lose that — no generated card does.
* New verdict statuses: IMPORT_ERROR (python kernel load failure),
  REFERENCE_ERROR (reference load failure) beside the existing COMPILE_ERROR.
# CHANGES — corps-fix-a (feat/corps-fix-a, base 8977b63)

Scope: the submit/ssh delivery cluster's B-class verdict bugs (BUG-SEED-11,
BUG-SEED-12), the regression pins for them, and the read-only re-verification of
BUG-SEED-20. `src/peregrine_kernel/loop/driver.py` and `loop/parse.py` were not
touched; the frozen surfaces (timing.py, precision constants, meta/ discipline
text, gate semantics, LICENSE/NOTICE) were not touched. No threshold defaults
changed.

## BUG-SEED-11 (P1) — ssh queueing time charged to the job deadline, timeout recorded Failed

Commits: `760dc8b` (fix), `baa991b` (regression pins).

- Was: `_exec_cmd` echoed `PK_JOB_BEGIN` *before* `setsid --wait flock`, so the
  flock queue wait (unbounded) rode inside the runner deadline
  (`timeout + RUN_HEADROOM_S`); a host busy with another run's job expired the
  deadline before the task ever started, and the `TimeoutExpired` branch recorded
  `Failed` — charging the kernel an iteration (loop-side: no final JSON →
  parse_eval correctness False → REVERT) for a job that never ran. The local
  delivery already took the lock before its marker (WP11 task 4) and logged queue
  telemetry; the ssh side did neither.
- Now:
  - the begin marker prints from *inside* the flock'd session script, i.e. only
    once the GPU lock is HELD (`setsid < flock < marker` in the exec command, and
    the bare pre-session `echo PK_JOB_BEGIN` is gone);
  - a deadline that expires with no marker in the log is recorded `Lost`
    ("queued on the GPU lock past Ns without the begin marker -- nothing ran"),
    which the loop resubmits without charging; a deadline spent *running*
    (marker seen) is still `Failed` ("exceeded ... killed");
  - the queue wait is logged in the driver's telemetry format
    (`[direct] GPU <gpu> acquired after <t>s wait`, measured on the host via
    `date +%s` deltas across the flock wait) — `_queue_wait_from_log`
    (loop/driver.py) now sees ssh jobs' queue waits too, mirroring the local
    delivery.
- Repro (red): fake-ssh `DirectRunner` scenarios 11a/11b — `("timeout", False)`
  returned `Failed` (should be `Lost`); the marker preceded `setsid`/`flock` in
  the exec command. Green after `760dc8b`; session-script quoting and live
  behavior (telemetry line → marker → entry, rc relay) verified with a real
  `setsid --wait sh -c` + `flock` run on this machine.
- Pins: `test_a_deadline_spent_queuing_on_the_gpu_lock_is_lost_not_failed`,
  `test_the_remote_begin_marker_is_printed_by_the_lock_holder`,
  `test_a_real_run_timeout_is_still_failed` (contract pair).

## BUG-SEED-12 (P1, aggravated) — ssh drop recorded Failed; unstoppable orphan; sid destroyed by the next sync

Commits: `253525f` (fix + two existing-test updates), `b5a67c8` (regression pins).

- Was: `rc != 0` with the marker in the log went straight to `Failed` — no
  survival check, no `_stop_remote`. But `rc 255` is the ssh *client's own* exit
  code for a dropped/failed connection (`setsid --wait` can only relay the remote
  chain's own rc otherwise), so the outcome was unknowable while the detached
  session kept holding the GPU lock. The session's one stop handle,
  `.pk_job.sid`, was written *inside* the job's tree, which every sync `rm -rf`s
  before extracting: the orphan could never be stopped, and every later job
  queued behind its lock into BUG-SEED-11's false-charge deadline (the cascade).
- Now:
  - `rc == 255` → best-effort `_stop_remote` (the sid file is reachable again,
    see below) and `Lost` ("the job's outcome on the host is unknown"), for
    began and never-began alike; the loop resubmits, nothing is charged;
  - `rc 1-254` + marker → still `Failed` (a relayed remote exit is trustworthy
    and a broken kernel must still cost an iteration); no stop attempt (the
    session is already gone — the chain relayed its rc);
  - `rc != 0` without a marker (and not 255) still walks the transport retry
    into the TRANSPORT marker channel (unchanged contract);
  - the session id lives *beside* the tree, one file per op per run:
    `<workdir>/<run>/.pk_jobs/<op>.sid` (`SESSION_DIR = ".pk_jobs"`), outside
    the `rm -rf`'d tree — `_stop_cmd` and the session script both use it, so an
    orphan stays stoppable for `submit stop` and for every later timeout path.
- Existing tests updated (`tests/test_direct_delivery.py`): the two never-began
  retry tests moved off rc 255 — `test_ssh_that_never_delivers_is_a_transport_error_after_retries`
  now drives rc 1 (`255, False` would now be `Lost` outright, so rc 1 keeps the
  retry → TRANSPORT-marker contract it exists to pin);
  `test_ssh_transport_blip_then_success` likewise `[(1, False), (0, True)]`.
  `test_ssh_payload_failure_is_charged_without_retry` (rc 1 + marker → Failed)
  is untouched and still green.
- Repro (red): scenarios 12c/12e — `(255, True)` returned `Failed` with zero
  stop attempts; `_stop_cmd` read the sid from
  `<tree>/.pk_job.sid`, a path the next sync wipes. Green after `253525f`
  (status `Lost`, 1 stop attempt; sid at
  `<workdir>/<run>/.pk_jobs/<op>.sid`).
- Pins: `test_a_dropped_ssh_connection_is_lost_whatever_the_log_shows`,
  `test_a_relayed_remote_exit_code_is_still_a_charged_failure`,
  `test_the_runner_records_a_lost_verdict_that_nothing_charges`.

## BUG-SEED-20 (P2) — re-verified read-only, NOT fixed (per assignment)

`remote_tree` (`submit/direct.py`) is still `<workdir>/<op_dir.parents[1].name>/<op_dir.name>`
— a basename-only run key with no uuid/hash. `runs.new_run` (`src/peregrine_kernel/runs.py:105-113`)
does make loop-created run dirs unique (`<stamp>-<token_hex(2)>`, collision-retried
mkdir), so under loop-driven runs the run segment is unique and the T3 "already
fixed" verdict holds for that path. But nothing in the submit package hashes the
local path, and the demo below shows the manual/CI surface a10 P2-9 /
TRIAGE-MONSUB-05 reported is still live: two same-named clones (`.../alice/peregrine-kernel/operators/rope`
and `.../bob/peregrine-kernel/operators/rope`) both map to
`/srv/pgk/peregrine-kernel/rope` — one run's `rm -rf` reaches the other's live
tree, and their `.jobs/lock` files are different files so nothing serializes
them. Verdict: A for loop-driven runs, B (same root cause) for manual same-named
trees. Report-only per the assignment; the fix (hash the op_dir's absolute path
into the run key, as the archived fix wave did) is left to the owner of that
item. Note: the new sid file inherits the tree's path (it is keyed by
`<workdir>/<run>/<op>`), so it collides exactly where the tree collides — same
class as the old in-tree sid, no new surface.

## Test porting (`tests/test_cli_submit_fixes.py`, from the archived fix-wave file)

Ported, rebranded (peregrine_kernel imports, `PeregrineKernel` clone layout,
PK markers; brand-clean per `test_branding.py`):

1. `test_a_deadline_spent_queuing_on_the_gpu_lock_is_lost_not_failed` — SEED-11 pin
2. `test_a_real_run_timeout_is_still_failed` — contract pair (directly green)
3. `test_the_remote_begin_marker_is_printed_by_the_lock_holder` — SEED-11 pin
4. `test_a_dropped_ssh_connection_is_lost_whatever_the_log_shows` — SEED-12 pin
5. `test_a_relayed_remote_exit_code_is_still_a_charged_failure` — contract pair (directly green)
6. `test_the_runner_records_a_lost_verdict_that_nothing_charges` — directly green
7. `test_clock_lock_retry_counts_total_submits_and_says_so` — rewritten against
   the current wording before porting: the archived text asserted a progress
   line ("attempt 2 of 3 in total") and a stdout final message this tree never
   had; the current code prints `resubmitting (n/3)` and puts "3 refusal(s) in
   a row" in the raised `ClockLockRefusedError`, which the rewritten test
   asserts via `excinfo`.

Dropped (with reasons; all five are loop/cli items outside this cluster's
submit/ssh scope — porting them red would fail the suite, fixing them is the
loop/cli cluster's work):

- `test_two_same_named_clones_get_different_remote_trees`: pins the archived
  wave's hash-based `remote_tree`; that fix is deliberately not ported here —
  BUG-SEED-20 is a report-only re-verification item for this assignment, and
  the t5-rewritten variant would pin the current basename behavior (same-named
  clones still collide) as if it were the contract. Conclusion recorded above
  and in findings/fix-fixa.md.
- `test_a_poll_timeout_counts_toward_giving_the_job_up`: verified experimentally
  red on this tree — a poll timeout raises `BackendError` at
  loop/cli.py `_submit_gpu_once`'s `poll_until_terminal` without ever calling
  `_note_inflight_log_failure` (only `_fetch_log` failures count), so
  `log_failures` stays 0 and the record is re-adopted forever; the fix is not
  landed and belongs to the loop/cli cluster.
- `test_an_overlay_build_failure_is_recorded_as_a_structured_trace`: the
  structured `overlay_fallback_file` ledger does not exist in this tree and the
  misleading success line is still printed after a fail-open (t5's finding
  re-confirmed); unlanded fix, loop/cli cluster.
- `test_retry_call_announces_retries_and_sleeps_only_between_them`: `_retry_call`
  has no per-attempt announcement and still sleeps after the last failure
  (t5's finding re-confirmed); unlanded fix, loop/cli cluster.
- `test_an_interrupted_agent_turn_ends_the_whole_session`: SEED-10 (T2) is alive
  in this tree — `_agent_turn`'s `Popen` has no `start_new_session=True` and the
  interrupt branch only `proc.kill()`s — so the port would be a permanent red
  pinning an unlanded loop/cli fix.

## Commit list (base 8977b63)

- `760dc8b` fix(audit-corps): print the ssh job's begin marker only once the GPU lock is held
- `baa991b` test(audit-corps): pin the lock-queue verdict semantics for the ssh delivery
- `253525f` fix(audit-corps): record a dropped ssh connection as Lost and keep the session file outside the synced tree
- `b5a67c8` test(audit-corps): pin the dropped-connection verdict semantics for the ssh delivery
- `dd338ec` test(audit-corps): pin the clock-lock retry budget's total-submits counting
- (tip) reword of the ported file's docstring (brand-gate: the archived wave's
  project name is whitelisted nowhere outside attribution files) + this file.

Full regression at the tip: **2699 passed, 0 failed** (563 s; baseline before any
change was 2692 passed, 0 failed, 493 s — the +7 are the ported tests; the two
test_direct_delivery updates are rewrites, not additions).
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
