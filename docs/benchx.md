# `forge bench` — one candidate, one card, the card's own reading

`forge bench` answers one question for every consumer that needs a reading of a kernel:

> On this GPU, how does *this candidate* do on *this board problem*, judged and timed
> the way the board judges and times it?

It is the real implementation behind `forgekernel.benchx.api.Bench`; the kernel library,
the campaign, evolve and the per-workload routing (`shapes`) all go through it, so there
is one scoring path and one reading format instead of four.

## Why its reading is the loop's reading

The bench does not re-implement the evaluation. It installs the frozen card with
`taskcard.install_board_trees` (what `forge run` does for a board task), clones the
workspace with `taskcard.clone_workspace`, commits the candidate and pushes it into the
card's branch, and submits the card's own `gpu_entry.sh all` through
`RealBackend.submit_gpu` — the same call the loop's iterations make. The log is then read
the way the loop reads it: `forgekernel.loop.parse.parse_eval` for the verdict, and the
same JSON's `performance.leaderboard.sol_board` block for T_b / T_SOL / the SOL score.

So the number `forge bench` prints and the number an iteration of `forge run` would
record come from the same tree, the same script, the same clock handling and the same
arithmetic. `--repeat N` is the one place the bench derives something itself: the median
of N jobs' readings (a single job's score is always the card's own).

## Usage

```bash
# a board task (`tasks/<op>`, the frozen card under leaderboard/card) or a bare card dir
forge bench tasks/<op> --candidate kernel.py
forge bench tasks/<op> --candidate solution.json --workloads 0,3,5-9 --repeat 3 --json r.json
forge bench tasks/<op> --candidate kernel.py --mode target          # compile check only
forge bench tasks/<op> --candidate kernel.py --keep                  # keep .artifacts/benchx_*

forge bench compare a.json b.json [c.json ...]        # per-workload table + summary
forge bench best-of a.json b.json [c.json ...]        # per-workload routing table
```

* `--candidate` — a `.py` file with a top-level `run(*args)` (positional, the board's
  calling convention), or an official solution JSON (its `spec.entry_point` file and the
  other `sources` are written beside it). A solution whose
  `spec.destination_passing_style` differs from the card's `dps` gets a thin adapter at
  the card's kernel file: the protocol stays the card's, the computation stays the
  solution's (`integrity.candidate.dps_adapter` says which adapter ran).
* `--workloads` — `all` (default) or indices / ranges / names. A subset **trims the temp
  card's `problem.json`** to those workloads, so the job really judges and times only
  them (a full-card reading needs `all`). The card's own scope switch
  (`FK_LB_WORKLOAD_SCOPE`) is deliberately not used: a scoped job is refused by
  `parse_eval` as a meta reading, and a subset bench should still produce a real verdict
  over what it ran. `integrity.workload_subset` records it.
* `--repeat` — eval jobs to run; each workload's reading is their median, with the spread
  under `integrity.repeat`. The cold start is not re-probed (see below).
* `--mode dev` — run GPU jobs on the machine `local.env` names (local or ssh).
  `--mode target` — run nothing; only the target compile check.
* `--json OUT` — the `BenchResult` as JSON (the input of `compare` / `best-of`).
* `--keep` — keep the temporary run directory (default: deleted, even on failure).

Exit codes: `0` the candidate is correct (or, in `--mode target`, the check passed);
`2` a usage error or a verdict that is not "correct" — the reason is printed.

## The result: `BenchResult`

| field | meaning |
|---|---|
| `correct` | the card's verdict (`correctness.pass`) |
| `workloads[]` | per workload: `name`, `axes`, `t_ms` (the card's stopwatch: `device_median_ms`, the reading the gate uses), `t_b`, `t_sol`, `s`, `correct`, `error` |
| `mean_s` | the mean of the per-workload SOL scores — the track score's per-problem term. Taken from the card's `sol_score_mean` for a single job; recomputed with `leaderboard.sol_score` from the median reading only under `--repeat` |
| `fast1` | how many workloads read faster than their T_b (`fast1_ratio()` = fast1 / timed) |
| `job` | the GPU job id(s), comma separated (probe job not included) |
| `elapsed_s` | wall clock of the whole bench, probe and jobs included |
| `error` | the card's own failure sentence; empty when correct |
| `cold_start_s` | see below |
| `integrity` | machine facts that qualify the reading: `cold_start`, `clock`, `clock_note`, `reading_basis`, `repeat`, `workload_subset`, `target_compile`, `failure_kind` |
| `confidence` | how far a dev-machine reading can be trusted as a B200 reading: always `"dev-only"`, plus hints from the target profile module |

## Cold start (`cold_start_s`)

The board re-compiles every candidate on every evaluation: an inline-CUDA `load_inline`, a
Triton / CuTe DSL JIT and a `torch.compile` all pay it inside the evaluation job. The
official open evaluator defaults to a **120 s compile timeout** and a 600 s job budget; a
candidate slower than that scores zero however fast its kernel. So the bench measures it:

* a probe job (`cmd:python3 benchx_coldstart.py`, written into the tree at commit time)
  starts a fresh process on the same GPU, with the card's `gpu_entry.sh` environment and
  clocks, takes `time.perf_counter()` **before importing torch or the candidate**, imports
  the implementation exactly as `eval/bench.py` does (`leaderboard._impl`), draws the
  card's first input set and makes one successful call — then prints
  `[benchx] COLD_START_S <seconds>`;
* it runs **before** the eval job(s), so the extension/JIT caches are cold and the number
  is what a fresh container pays. The eval jobs that follow run with warm build caches;
  that changes their start-up, not the timed kernel window (which is measured after
  warm-up, inside the card).

`COLD_START_BUDGET_S = 100.0` (`forgekernel/benchx/api.py`): above it,
`integrity["cold_start"] = "over-budget"` and the campaign's export refuses the candidate.
Unmeasurable is recorded as `None` with `integrity["cold_start"] = "unmeasured"` and a
reason — never as 0. Under `--repeat`, the cold start is the first probe's reading only:
later jobs' caches do not describe the board.

## `confidence`

`"dev-only"` is always there: the reading was taken on the development machine (B300),
not on the evaluation machine (B200) — the B300 differs in dense NVFP4 throughput (~1.5×)
and SFU rate (~2×), fewer/more SMs per SKU, and its clock lock state. The numbers are
comparable for *screening* and for A/B of candidates on the same machine; the board is
the only source of truth for a score.

When the `targethw` task's module (`forgekernel.targethw.rules`) is importable and
exposes `confidence_hints(source, kernel_file=, language=, arch=) -> list[str]`, those
`info`-level hints are appended (`"fp4-bound"`, `"sfu-bound"`,
`"sm-count-sensitive"`, …). benchx never depends on it; absent means `["dev-only"]`.

## Target compile check (`--mode target`, and after every dev bench)

`integrity["target_compile"]` always contains a builtin check: a syntax compile of the
entry file plus the feature flags a target build must cover (`inline_cuda`, `triton`,
`cutedsl`, `torch_compile`). When the `targethw` module exists and exposes
`target_compile(source, kernel_file=, language=, arch=)`, its result is attached under
`external` (or `{"status": "error", ...}` when it raises); otherwise `external` is
`"unavailable"`. In `--mode target` nothing runs: `correct` is not a verdict there and
`integrity["verdict"]` says so.

## The temporary run

`.artifacts/benchx_<stamp>-<hex>/` in the `runs` layout (`run.json` with
`"kind": "benchx"`, `task/`, `operators/`, `ws/`), so `forge status` lists it. It is
deleted when the bench ends — success, failure or a usage error (`try/finally` in the
runner) — unless `--keep`. A kept run is a bench run: it has the card, the candidate
commit and the job logs under `operators/<op>/.jobs/`, but no loop configuration (a loop
run would need `loop.toml`).

Everything that runs a GPU job goes through ForgeKernel's own submission path, so the
external GPU lock (`FK_JOB_LOCK_CMD`) applies as usual. Nothing is uploaded, the board is
never contacted (T_b / T_SOL were frozen into the card at `forge init`), and no agent is
started.

## compare / best-of

Both are pure functions over `BenchResult` JSON files (`forgekernel/benchx/compare.py`):

* `compare` — per workload: every candidate's reading, SOL score and delta against the
  fastest, who was fastest; summary per candidate: mean S, `fast1` ratio, geometric mean
  reading. Multi-workload cards should be compared per workload: a geomean over workloads
  mixes shapes.
* `best-of` — the routing table the `shapes` task consumes: per workload the fastest
  *correct* reading, with candidates within **1.5 %** of it counted as a tie and the tie
  going to the earlier file (pass the simpler implementations first); a candidate that is
  incorrect on a workload never wins it; workloads nobody got right have no winner. The
  fallback is the candidate with the highest mean S — what to run when a workload has no
  winner at all. Output: `{"routes": [{workload, winner, t_ms, s, within_noise}],
  "fallback", "fallback_mean_s"}`.

## Limitations

* **Board tasks only.** A non-leaderboard task is refused with a message: ordinary tasks
  have no frozen card, so there is nothing to reproduce the loop's reading with.
* **Readings are the development machine's** (`confidence: dev-only`). A B300 reading can
  be optimistic for FP4-heavy or SFU-heavy kernels and pessimistic when the SM count
  differs; only the board's own evaluation decides.
* **The card's clock lock governs.** An SOL card refuses to time when the board's lock
  (SM 1500 / memory 3996 MHz) does not hold on the machine; the bench then reports the
  refusal as a job failure (nothing was judged). On a machine that cannot lock, the card's
  documented escape is `LB_REQUIRE_CLOCK_LOCK=0` in the job environment (local delivery:
  the environment `forge` runs in; ssh: `REMOTE_ENV_FILE`) — the readings are then not
  comparable to the board's, and the card's `clock`/`clock_note` say so.
* **A subset is a subset.** A `--workloads` reading proves correctness/timing on what it
  ran; the full card (and the track score) needs `all`.
* **The cold-start probe is a second job.** It costs one job's start-up and it is what
  warms the build caches the eval jobs then reuse.
* **`fast1` is the count**; the ratio is `fast1 / len(timed)`.

## Upstream suggestions

* `runs.new_run(..., prefix=)` — a run directory whose name is not `<task>_<id>` (bench
  runs are `benchx_*`) currently needs a copy of `new_run`'s record writer.
* `loop.parse.parse_eval` could return the raw eval JSON (or expose the last-object
  scanner); the reader re-scans for `performance.leaderboard.sol_board` because the
  `EvalResult` does not carry it.
