# Throughput governor

`rsi_kernel_tune.loop.governor` wraps a loop backend (`DriverBackend`) and governs exactly
two things that decide kernel quality. It is not a cost control: **there are no token
limits**, by explicit user constraint.

1. **Iteration throughput.** A teacher phase must not eat the run while the student has
   barely iterated.
2. **Architecture escape.** When a run is far from its target after several iterations
   and the last rounds bought nothing, tell the teacher to change the structure instead
   of tuning the current one.

Motivation (issue #8, an external user reproducing the RMSNorm example on an H100):
10.6 hours, 136 M tokens, **92.5 % of the wall clock in teacher turns, 7 effective
student iterations** (2 of them did not change the computation), three meta steps burned
on "resource-budget addendum / performance attribution / measurement verification", final
result 0.547x (decode 0.289x). The tokens were not the problem; the spent hours bought
no iterations, and the teacher never questioned the starting structure.

## Where it sits

```
launch.run_loop  --FK_BUDGET_POLICY(JSON)-->  loop/cli.main
                                               backend      = RealBackend(...)     \
                                               recon_backend= RealBackend(...)      > _govern_backend()
                                                                                    /
                                                   loop/governor.GovernedBackend
                                                       └── inner.run_agent(prompt, role=...)
```

`loop/cli.py` wraps both backends it constructs (the student's loop backend and the
reconnaissance backend) with the same policy and the same ledger, so reconnaissance
counts as teacher time. `loop/driver.py` is unchanged: every teacher turn already goes
through `_TeacherView`, which passes `role="teacher"`; student turns arrive without a
role and are recognised by their prompt title.

The judge turn is outside this: `launch.write_judge` runs before the loop with its own
unwrapped `RealBackend`, so it neither counts against the teacher share nor appears in
the ledger. `judge` is still in the phase table because that prompt can reach a governed
backend in other launch paths (it is what the title table is tested against).

## Turning it off

- `FK_GOVERNOR=off` (also `0`, `false`, `no`, `disabled`): `loop/cli.py` returns the
  real backend unwrapped. No ledger, no brakes, no escape text. `forge status` and
  `report.md` then say the governor has no ledger.
- Removing the `[budget]` keys is not an off switch — they have defaults.

## Configuration

`task.toml` `[budget]` (defaults in `global_config/loop.toml`; validated in
`rsi_kernel_tune/task.py`):

| key | default | meaning |
|---|---|---|
| `teacher_share` | `0.5` | while the student has fewer than `min_student_iters` iterations, teacher cumulative wall clock may not exceed `hours × teacher_share` without a soft brake |
| `min_student_iters` | `5` | the student iteration count that lifts the share rule |
| `max_teacher_turns_per_phase` | `12` | consecutive teacher turns in one phase (no student turn between) before a soft brake |

There is no token key; a `[budget] max_tokens` is an unknown-key error on purpose.

`launch.run_loop` serialises the four keys (the three above plus `hours`) into
`FK_BUDGET_POLICY` for the loop call and restores the previous value afterwards.
`loop/cli.py` validates the payload; a malformed payload, an unknown key, or an
out-of-range value stops the launch with exit code 2 rather than running under a policy
nobody can read.

## The rules

**Soft brake** = a fixed block prepended to that turn's prompt. The turn still runs.

```
[THROUGHPUT] This phase has used most of the time the run can spend on it without the kernel
getting any further. Finish now: write the files / verdict this step requires, list what remains
unresolved in one short block, and send no further requests. An unresolved question is an
acceptable outcome; an unbounded investigation is not.
```

**Hard stop** = raise `GovernorStop` (an `AgentError` subclass) *before* the turn runs;
the message names the rule, the used amount and the limit.

| rule | soft brake | hard stop |
|---|---|---|
| teacher wall clock, while `student_iterations < min_student_iters` | `used ≥ hours × teacher_share` | `used ≥ 1.25 × hours × teacher_share` |
| consecutive teacher turns in one phase | `count ≥ max_teacher_turns_per_phase` | `count ≥ max_teacher_turns_per_phase + 4` |

- Teacher wall clock is the sum of the `elapsed_s` of completed teacher turns on the
  ledger — reconnaissance included (same ledger, both backends). Student turns cost
  time but never count against the teacher.
- A student turn resets the consecutive-turn counter and allows the escape rule to fire
  again in the next meta round. A different teacher phase starts its own count.
- **Student turns are never soft-braked or stopped** — not by the share rule, not by
  the turn counter, not even when an explicit `role="student"` arrives on a prompt whose
  title looks like a teacher's.
- Both brakes can apply to one turn (one prefixed block per rule; the fixed texts are
  not duplicated).
- Why soft first: a hard stop ends the run. The soft block is usually enough — the
  teacher can still land the step's deliverable and name what stays unresolved — and it
  costs nothing when ignored (the next turn past 1.25x is refused).

### What a hard stop does to the run (verified against the code)

`loop/cli.main` wraps the whole two-layer loop (`run_two_layer`, which includes
reconnaissance, distillation and the meta layer) in
`except (AgentError, BackendError)`; a `GovernorStop` is an `AgentError`, so it is caught
there:

- the log line is `[loop] meta-layer agent_failed — stopping cleanly: <reason>` and the
  summary's `stop` is `agent_failed` (the cli's existing label; changing it would mean
  editing that catch block, which this task does not own);
- the best-sha confirmation still runs and `best_confirmation.json` is written;
- the gap queue and the harvest report still run (the finish-time KB write-up is
  skipped for `agent_failed`, as for any agent failure);
- `summary.json` and `report.md` are written, and `launch.drive` returns **2** (the
  run did not reach its target).

So: best preserved, report written, exit code 2. The one cosmetic mismatch is the
`agent_failed` stop label instead of a budget-specific one; it is recorded here as a
limitation because the fix needs the cli's catch block (integration note).

A `GovernorStop` raised inside the finish-time KB harvest is caught by that function's
own total catch (it never raises); it cannot end an already-finished loop.

## The architecture escape rule

Read from the run's own records (`student/status.jsonl`) before a `diagnose` or `patch`
teacher turn. Three conditions together:

1. at least **3** student iterations since the last `M0-gate` row that KEEPed;
2. the best reading in the run's gate window is still **more than 2×** the loop's
   target (`--target-ms`, whichever window the loop gates on);
3. neither of the last **2** evaluated iterations was a KEEP that improved the running
   best by at least the **1.5 %** noise floor (the same floor `detect_plateau` uses;
   "improved" is a strict improvement beyond the floor).

Then the turn is prefixed with:

```
[ESCAPE RULE] After N student iterations the best reading is still X times the target, and the last
two iterations did not improve it beyond the noise floor. A gap this large is rarely closed by
tuning inside the current structure. Treat this as R1 (a foundation ceiling) unless the measured
evidence in front of you shows the current structure is within a few percent of its own ceiling:
prefer a structural patch of the earliest milestone over more parameter tuning.
```

`N` and `X` are the actual numbers. It is advice only — the teacher decides — and it
fires **at most once per meta round** (a student turn starts the next round). No target
(dry runs) or no M0 gate means it never fires. A recent KEEP that improved less than
1.5 % does not count as progress.

## The ledger

`<run>/operators/<op>/student/governor_ledger.jsonl` (next to `student/status.jsonl`),
append-only, one JSON object per line:

| `event` | written when | fields |
|---|---|---|
| `policy` | each governor is constructed | `hours`, `teacher_share`, `min_student_iters`, `max_teacher_turns_per_phase`, `target_ms`, `gate_window` |
| `turn` | every agent turn (ran, failed, or student) | `role`, `phase`, `elapsed_s`, `student_iters`, `brakes` (rules applied), `escape`, `turns_in_phase`, `error` |
| `soft_brake` | before a braked turn | `rule`, `phase`, `student_iters`, `teacher_s`, `share_s`, `turns_in_phase`, `limit` |
| `hard_stop` | instead of a refused turn | `rules`, `detail`, `phase`, `student_iters` |
| `escape_fired` | with the escaped turn | `phase`, `iters`, `best_ms`, `target_ms`, `x` |
| `over_share` | the first time teacher time crosses the share while the student is short | `teacher_s`, `share_s`, `student_iters` |

Teacher wall clock is the sum of `turn` rows' `elapsed_s` for `role=teacher`; it spans
resumes (the ledger lives in the run's tree) and both backends.

Phases (`detect_phase`, from the prompt's first non-empty line): `judge`, `recon`,
`distill` (both variants), `m0_audit`, `diagnose`, `patch`, `kb` (recipe + harvest),
`student`; anything else is `other` (KLIB one-turn prompts such as seed-adapt and
baseline-rewrite, and appended fragments like the verdict-probe brief). The title table
is tested against every file under `meta/`, language variants included.

A turn's role is the explicit `role=` when the driver passes one (`_TeacherView` does
for every teacher turn); otherwise it comes from the phase. An `other` title resolves to
student: a turn this module cannot identify is left alone rather than refused — the hard
guarantee that a student turn is never stopped wins over a hypothetical unlabelled
teacher turn.

## Reading it

`forge status <run>` adds two lines when a ledger exists:

```
  throughput: teacher 14 / 18340.0 s; student 9 / 5200.0 s
  throughput rules: teacher_over_share, budget_starved (soft brakes 3, hard stops 0)
```

`report.md` gains a `## Throughput` section: the policy, student iterations (the
`status.jsonl` row count), the three flags, turns and seconds by role and by phase,
brake counts, and — **only when `<run>/agent_usage.jsonl` exists** (written by the
backends task) — tokens and seconds by role/phase. Without a ledger the section says
so.

Flags:

| flag | true when |
|---|---|
| `teacher_over_share` | an `over_share` event exists (teacher time crossed the share while the student was short) |
| `budget_starved` | `teacher_over_share` **and** the run ended with fewer than `min_student_iters` student iterations |
| `escape_fired` | an `escape_fired` event exists |

## Design choices and limitations

- **A decorator backend, not a driver change.** `loop/driver.py` is untouched; the two
  construction sites in `loop/cli.py` wrap the backend. Tests that drive the driver
  with `FakeBackend` are unaffected.
- **Transparent by default.** Below every limit the prompt reaches the inner backend
  byte-for-byte. Attributes and methods are delegated; public attribute writes
  (`backend.kernel_file = ...`, `job_env_files`) are forwarded to the inner backend so
  post-construction configuration still lands.
- **Student iterations = `status.jsonl` rows.** That is the number the rest of the
  harness calls iterations (`forge status`); driver-archived gate rows count too.
- **The escape rule's thresholds are fixed** (3 / 2× / 1.5 % / 2 iterations), not
  configurable: they are the rule as specified. Only the target comes from the run.
- **A hard stop ends the run.** That is the specified outcome (graceful stop, best kept,
  exit 2). If a future task wants "brake forever instead of stopping", it belongs in
  this module's `_precheck`.
- **Unverified under a real model** (see the report): whether the soft text actually
  makes a teacher wrap up, whether the escape text changes a diagnosis, and how often
  the rules fire on a real run. Everything here is exercised with fakes and a
  controlled clock; there is no GPU/agent path in this repository's test suite.
