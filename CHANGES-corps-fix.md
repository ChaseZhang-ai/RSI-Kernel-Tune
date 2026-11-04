# CHANGES — audit corps fix-c (feat/corps-fix-c)

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
