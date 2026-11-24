# Evaluation integrity (task guard)

The loop is rewarded by a reading, and a reading can be gamed. This document says what
`forgekernel.integrity` and the card's own `integrity` block check, where each check sits
in an evaluation, what a failure costs, how to tune it, and what is deliberately **not**
covered.

Two places need this:

| where | what runs | why it has to live there |
|---|---|---|
| the card (`eval/leaderboard.py`, copied into a task at `forge init`) | the **runtime** checks A–D, after the timed window | the card runs in a job image where the `forgekernel` package may not exist; it is a standalone module by design (like `eval/timing.py`) |
| `forge export` / `campaign.compose.export` | the **static** checks E (lint), the licence gate, targethw's target rules, the cold-start gate | the file about to be uploaded is on the host, and this is the last place to stop it |

The package is the host-side door to the same rules (`snapshot.py`, `timers.py` load the
card module), so there is one implementation and the tests pin it.

---

## A. Fresh-input recheck

**What it defends against.** A candidate that is right on the inputs it has already seen
and cheap afterwards: a value cache filled during correctness, a CUDA graph whose replay
skips the work (the AKO4X archive has that case), a branch on "call number". The board's
correctness and timing phases both run on inputs the candidate has seen by then, so only a
draw *after* the timed window proves the kernel computes the function.

**Where.** Immediately after the timed window, on the same card, never inside it.

**How.** The card redraws inputs **the way the card draws them** (`gen_inputs` for the
two board suites, `task_inputs` with a bumped group seed for the task suite) — no second
generator. The seeds are seeded per job and restored afterwards (`random`, `torch`,
`torch.cuda`), so the suite's own seed-200 sequence never moves. The reference is computed
first, on the fresh inputs, with the **same judge the correctness phase used** (the
definition's own `run` / the task's judge — *not* the in-job timing reference), then the
candidate, and the two are compared with **this card's own tolerance** (`sol_error_stats` /
`fib_judge` / `task_error`).

**Workloads.** `FK_INTEGRITY_FRESH_N` (default **2**): workload 0, plus one chosen by the
job's pid so repeated jobs rotate through the group. `0` disables it.

**Skipped, with the reason in the block** when the values cannot be redrawn: a
safetensors input (a fixed file), a definition whose every input is a heuristic (the
redraw produces the identical tensors), a run with no timed window (`--mode correctness`),
a redraw that raises. A skip is never a pass, but it is also never a failure: the job must
not die because a check could not run.

**On failure.** `correctness.pass = false`, `status = "INTEGRITY_ERROR"`, `error` starts
with `INTEGRITY: fresh-input recheck mismatch on <workload>: <why> (max relative error
<err>)`. The loop treats it as a correctness failure (REVERT) — a kernel that only looked
right is never KEPT. The performance block is still emitted untouched, so the reading can
be inspected.

## B. Global state and reference stability

**What it defends against.** The candidate and the scorer share one process: raising
`torch.backends.cuda.matmul.allow_tf32`, changing `torch.get_float32_matmul_precision()`,
or setting an environment variable (`CUBLAS_*`, `CUDA_*`, `TF32`, `TORCH_*`, `OMP_*`,
`NCCL_*`) can make the *reference* slower or less precise so the candidate's ratio looks
better. The official reward-hack code captures the identity of `elapsed_time` and the
thread count for the same reason.

**Snapshots at three moments:** before the candidate's module is imported, after the
correctness phase, after the timed window. Flags:
`torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32`,
`torch.get_float32_matmul_precision()`, the fp16/bf16 reduced-precision-reduction flags,
`torch.backends.cudnn.benchmark`; the matching environment variables; `threading.active_count()`.
The harness's own switches (`FK_INTEGRITY_*`, `FK_LB_*`, `LB_*`, `FK_TARGET_*`) are
ignored — a check may not read its own configuration as evidence about the candidate.

* **A flag or environment change** → `INTEGRITY: global state changed after the candidate
  ran: …` → correctness fails.
* **A thread-count change** → a warning only (`global_state.threads`), warned and printed.
  The card already fails the timed window on thread injection above the correctness
  phase's snapshot (the official check); this is the slower-moving case (a thread started
  and still alive).
* A flag that was already changed before the candidate was imported is *not* attributed to
  it; the baseline is taken before the import.

**Reference stability.** The correctness phase keeps the reference's output on the input
it judged (a copy of the input is taken before the reference runs, so an in-place
definition is handled). After every candidate call, the reference is recomputed on that
input and compared with what it produced before — with the card's tolerance. A mismatch →
`INTEGRITY: reference output changed after the candidate ran` → correctness fails
(`reference_stability.status = "fail"`, `max_rel_err`).

## C. Timer consistency (warning by default)

**What it defends against.** Work that leaves the measured interval: a kernel on another
stream a per-call event window does not bracket, an empty CUDA-event path, a timing
function that was replaced. The suites' own timer is CUPTI (all streams), so this is a
*detector*, not the gate.

After the timed window, one call is measured three ways: CUDA events around it, a CUPTI
span of the activities it launched, and the wall clock. A per-call span measurement has a
fixed shape — `wall ≥ events ≥ span` — and the check is a factor of 2 away from it:

| reading | status | why |
|---|---|---|
| `span < 0.5 × events` | suspicious | the work escaped the event window (a hidden stream, a skipped body) |
| `events < 0.5 × span` | suspicious | the events did not see the work CUPTI did |
| `wall < 0.5 × span` | suspicious | impossible for one call |
| `wall > 20 × max(events, span)` **and** `wall > 1 ms` | suspicious | the reading is host-bound, not device time |

Factors of 2 are loose enough for a call whose kernels queue behind the L2 flush, tight
enough that an escape shows; the `> 1 ms` floor keeps ordinary micro-kernels (20 µs of host
time around 2 µs of work) out of it. `FK_INTEGRITY_STRICT=1` turns `suspicious` into a
failure (`INTEGRITY: timer consistency suspicious: …`); the default is a warning
(`timer_consistency.status = "suspicious"`, printed).

`unavailable` with a reason when there is no CUDA device, or when the profiler cannot see
the call.

## D. Implausible speedup (warning)

A reading below `0.6 × T_SOL` is above the speed of light: only a measurement defect or a
cheat can sit there. `implausible_speedup = true` in the block and a printed warning. It
never fails the run — a plausible-on-its-face reading is still checked by A–C and at the
board.

## E. Static review (`forgekernel.integrity.lint`)

Input: a submission's source (`lint_source`, `lint_texts` for several files, `lint_paths`
for a tree). Output: `Finding{rule, severity, file, line, snippet, detail}` with
`severity ∈ {block, warn, info}`.

| severity | rules | why block vs warn |
|---|---|---|
| **block** | `env-assign` (writing `os.environ`, `putenv`, `unsetenv`, `update`/`pop`), `torch-flag-assign` (a `torch.backends.*` or `set_float32_matmul_precision` write with no restore in the same function), `timing-function-patch`, `monkeypatch` (writing an attribute of an imported *module*), `thread-or-process` (`threading`/`multiprocessing`/`concurrent.futures`/`os.fork`/`subprocess.Popen`), `eval-path-read`, `absolute-path`, `path-traversal`, `credential-access`, `pointer-keyed-cache`, `call-count-branch`, `proxy-return`, `no-top-level-run`, `syntax-error`, `too-large` (10 MB) | no legitimate reading in a kernel submission; each is on the board's prohibited list or a shape only a cheat takes |
| **warn** | `cuda-graph`, `non-default-stream`, `event-construct`, `wall-clock`, `ctypes-or-cupti` | legal techniques; the graph case is what A exists for, and a non-default stream is caught by CUPTI (the official evaluator relies on exactly that — see `tests/.../test_timing.py::TestStreamHidingDetection`) |
| **info** | `inline-cuda`, `torch-compile`, `env-read`, `torch-flag-restored` | inlined CUDA is a first-class submission form; its arch/SM issues are targethw's rules |

Deliberate calibrations (each has a negative test):

* `from m import n` then `n = …` is a **local rebind**, not a monkey patch on `m`; the
  repo's own kernels do it (`_T = _extras.types`). Only `module.attr = …` written on an
  imported module is a patch.
* `torch.cuda.current_stream().cuda_stream` (reading the handle to pass to a driver API)
  is not a hidden stream; only creating/entering one is a warning.
* `data_ptr()` for alignment is fine; only a container **keyed** by `data_ptr()`/`id()` is
  a cache.
* Absolute paths are blocked except under `/usr/`, `/opt/`, `/usr/local/` — an inlined CUDA
  source legitimately includes `/usr/local/cutlass`.
* Docstrings may name anything (they document; they do not read); eval-path, traversal and
  absolute-path rules skip them.
* An embedded multi-line kernel source is skipped by the literal rules (its own comments,
  includes and strings are the kernel's, not the harness's).
* `lint_paths` lints `*.py` only — a `.cu` is not Python.
* `submission=False` (a loop `kernel.py`) skips the shape rules (top-level `run`, size) and
  the path rules.

## F. The eval JSON's `integrity` block

```json
"integrity": {
  "fresh_input_recheck": {"status": "pass | fail | skipped", "workloads": ["wl5"],
                          "reason": "", "max_rel_err": 1e-06},
  "global_state":        {"status": "clean | changed", "diff": [],
                          "threads": ["threads: 3 -> 4"]},
  "reference_stability": {"status": "pass | fail | skipped", "max_rel_err": 0.0,
                          "reason": ""},
  "timer_consistency":   {"status": "ok | suspicious | unavailable",
                          "events_ms": 0.11, "span_ms": 0.10, "wall_ms": 0.12,
                          "reason": ""},
  "implausible_speedup": false
}
```

* A card that ran no timed window (`--mode correctness`) adds `"timed_window": false` and
  A/C read `skipped` with that reason.
* `loop/parse.py` carries the block as `EvalResult.integrity` (verbatim, no gate reading
  touched). A card without the block parses to `None`.
* `forgekernel.integrity.report.validate` is the schema both sides are held to;
  `report.failures` reproduces the `INTEGRITY:` lines the card writes.

## Export gates

`forge export <run>` and `campaign.compose.export` both review the file that is about to
leave the harness:

1. `forgekernel.integrity.export_check(source)` — E's lint rules, plus
   `forgekernel.targethw.rules.check_submission` (a **soft dependency**: imported if
   installed, skipped with an `info` finding otherwise), plus the licence gate when
   `origins` are given. `forge export` prints every finding and **refuses** (exit 2) on any
   `block` unless `--allow-integrity-findings` is passed; then the leaderboard solution's
   `description` gains `integrity: N block-level finding(s) overridden`. A non-board export
   that is allowed writes the kernel source to stdout with no other stdout output
   (diagnostics go to stderr).
2. **Licence gate** (`license.license_check`): every origin's `license` must be
   `Apache-2.0` / `MIT` / `BSD*`, else `license-not-allowed` (block); every origin's
   `notice` must appear in the file, else `license-notice-missing` (block) with the text to
   add. `notices.banner(origins)` renders the block; `campaign.compose.export` already
   calls both slots.
3. **Cold start** (`cold_start_gate`): reads `cold_start_s` from a benchx result
   (`--bench-result`, or `bench.json` / `bench_result.json` / `bench/last.json` /
   `bench/result.json` beside the run) and refuses over
   `forgekernel.benchx.api.COLD_START_BUDGET_S` (100 s). No result → `unmeasured`, printed
   as "not measured", not treated as fast. `--allow-slow-cold-start` overrides, like
   campaign's.
4. `fallback.wrap_with_fallback(kernel_src, reference_src)` — the optional final wrapper:
   a single file whose `run(*args)` calls the fast implementation and, on **any**
   exception (including one raised while the fast module's own top level runs), falls back
   to the reference and copies its result into the preallocated outputs. The kernel's
   leading comment block (the NOTICE banner) is kept at the top of the new file. This
   hides a broken kernel — it scores 0.5 instead of 0 — so it is applied only at the final
   export, never during development, and the first fallback prints the exception.

## Multi-seed (`forgekernel.integrity.multiseed`)

A kernel that lowers precision (TF32, bf16 accumulation, FP8) can pass one draw and fail
the next. `multiseed.check(card_dir, candidate)` runs the card once per seed with
`FK_INTEGRITY_SEED` set — 5 seeds by default `(0, 1, 7, 42, 1234)` — and requires every
job's `correct` **and** a passing fresh-input recheck. A job with no recheck in its JSON is
reported as `unverified`, not as a pass. It needs a GPU (it drives a real `Bench`); the CPU
tests cover the orchestration with `FakeBench`.

## Tuning

| variable | default | effect |
|---|---|---|
| `FK_INTEGRITY` | on | `0`/`off` turns the card's whole block off (the export gates are unaffected) |
| `FK_INTEGRITY_FRESH_N` | 2 | how many workloads the fresh recheck covers (`0` = none) |
| `FK_INTEGRITY_SEED` | `90000 + job` | base seed of the redraw (per workload: `base + 101·index`; multiseed drives it) |
| `FK_INTEGRITY_STRICT` | off | `1` makes a `suspicious` timer reading a failure |
| `--allow-integrity-findings` | off | `forge export` writes despite block-level findings |
| `--allow-slow-cold-start` | off | `forge export` writes despite an over-budget cold start |

## What this does not cover

* **The card is frozen per task.** `forge init` copies `eval/leaderboard.py` into the
  task; the checks affect **tasks created after this change**. An existing task keeps the
  card it was created with — re-create it (or copy the new card in and accept a harness
  drift) to get the checks. `loop/harness_guard.py` treats the card as harness: a student
  cannot edit it away.
* **Kernels that set global state at import time are refused.** Some published kernels
  (two entries in `kernel_library/` — Hyra's reconstructed solutions) write
  `torch.backends.cuda.matmul.allow_tf32 = True`, `torch.set_float32_matmul_precision` and
  `os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", …)` at module level. Those are
  exactly what checks B and E exist for: imported into the card's process they change what
  the reference computes. A candidate derived from such a kernel must have those lines
  removed (or moved into a `run()` try/finally that restores them) before export —
  `forge export` names the lines.
* **Abusing a loose tolerance** (a definition whose tolerance is so wide that an
  approximation passes) and **semantic cheating** that is statistically correct (a
  different algorithm that passes every draw) are not detectable by any of this; the
  board's LLM review is the reviewer for those. The precision record
  (`eval/precision.py`) and `multiseed` cover the numeric half.
* **The board's own protocol.** Nothing here changes tolerance, input groups, the
  10 + 50-call protocol, the L2 flush, the pointer shifting or the official four checks. A
  new check is a new failure category (`INTEGRITY: …`) or a warning.
* **A real GPU.** The C check's CUDA path, the cost of A–C on a B200/B300 card and whether
  a real CUDA-graph stale output is caught by A are unverified here (no GPU work in this
  round); see the report's "unverified" section for how to verify each.
* **The pre-export LLM review** (prompt `meta/export_review_prompt.md`) is not implemented
  (left as a follow-up; it needs a real teacher turn).
* **campaign's compose does not refuse** on `export_check`'s blocks today: its slot records
  the verdict in the result but the write has already happened. `forge export` refuses;
  wiring compose's refusal is noted as an integration item.
