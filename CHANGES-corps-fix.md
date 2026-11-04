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

Full regression at the tip: see findings/fix-fixa.md (recorded after the final
run; baseline before any change was 2692 passed, 0 failed, 493 s).
