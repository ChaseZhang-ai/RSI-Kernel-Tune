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
