# `forge evolve` — multi-lineage self-evolution on one task

`scripts/self_evolve.py` walks one chain: each generation's best kernel becomes the next
generation's reference, and the chain stops when a generation cannot beat its own reference.
That is a single trajectory, serial, and (in the old script) fixed to `cutedsl`.

`forge evolve` keeps **N lineages on the same problem at once**. Each lineage has its own
start (reference / library seed), language, optional architecture hypothesis and optional
effort. After each generation the best lineages are kept, a laggard that stopped improving
is handed the global best as its next start (migration), and a *refuge* lineage is kept from
migrating so the population never collapses onto one architecture. It is the same
"elite + selection pressure + crossing" idea the top teams describe, expressed with the
public `forge` commands only.

The loop internals are untouched: a lineage is a plain `forge run` of a generated task
directory. `scripts/self_evolve.py` is not modified and keeps working.

---

## Commands

```
forge evolve tasks/<op> [--lineages N] [--generations G] [--step 1.05] [--parallel P]
                       [--langs cutedsl,cuda] [--hypotheses <dir|file> …]
                       [--lessons <file>] [--elite K] [--refuge 1]
                       [--migration-from 2] [--s-floor 0.005]
                       [--max-hours H] [--max-fails 2]
                       [--from-lib] [--sealed-ref] [--efforts high,max]
                       [--target-profile FILE] [--population DIR]
                       [--new-population] [--dry-run]
forge evolve status <population dir>
forge evolve show   <population dir>
```

While the step-0 extension dispatcher is not installed in a build, the same command runs
as `python -m rsi_kernel_tune.ext.evolve …` (`PYTHONPATH=<repo>/src`). The module defines
`COMMAND = "evolve"` and `main(argv)`, which is what the dispatcher routes to.

| Option | Default | Meaning |
|---|---|---|
| `--lineages N` | 2 (or one per `--from-lib` seed) | active lineages |
| `--generations G` | no cap | stop after G generations |
| `--step` | 1.05 | every generated task's target ratio |
| `--parallel P` | 2 | at most P `forge run` subprocesses in flight |
| `--langs` | the task's language | rotated over lineages (`langs[i % len(langs)]`) |
| `--hypotheses` | none | a directory of `lineage-<i>.md` or one file for all |
| `--lessons` | none | passed to every lineage as `FK_LESSONS_PATH` (only if the file exists) |
| `--elite K` | `max(1, ceil(N/3))` | elites per generation |
| `--refuge` | 1 | non-elite lineages that never migrate |
| `--migration-from` | 2 | migration is allowed only after this generation |
| `--s-floor` | 0.005 | board-task noise floor (SOL score points) |
| `--max-fails` | 2 | retire after this many consecutive generations without a best |
| `--from-lib` | off | seed lineages from `forge lib find` (task seedlib) |
| `--sealed-ref` | off | board handoffs use `--sealed-ref` (black box) |
| `--efforts` | none | rotated `[agent.student] effort` values (pilot A/B) |
| `--target-profile` | `FK_TARGET_PROFILE_PATH` or `target/B200.json` | target profile passed to every run |
| `--population DIR` | latest for the problem | population directory to continue |
| `--new-population` | off | start a fresh population |

`--lineages 1` reproduces `scripts/self_evolve.py`: the same `tasks/<op>` (generation 1),
`tasks/<op>-g<k>` naming, the same `gain <= 1 + noise floor` stop, and exit 1 when the
first generation produces no confirmed best.

---

## Concepts

```
generation k:   l0 ──forge run──> reading ─┐
                l1 ──forge run──> reading ─┼─> elites  = top --elite by Metric.order
                l2 ──forge run──> reading ─┘    best   = rank 0
                                                │
   after generation k (k >= --migration-from):  │
     ├─ non-elite, stalled, and best is ahead by > floor ──> MIGRATE (next start = best)
     ├─ refuge (>= --refuge, different language preferred) ─> keep its own architecture;
     │                                                        a stalled refuge RETIRES
     ├─ stalled AFTER adopting the global best ──────────────> RETIRE
     └─ no confirmed best for --max-fails generations ───────> RETIRE
```

* **population** — one directory, one problem, N lineages, a generation counter.
* **lineage** — a language, a start (its own previous best, another lineage's best, a
  library seed, or the base task's reference), an optional hypothesis file, an optional effort.
* **generation** — every active lineage runs once; all bookkeeping happens before the next
  generation is allowed to start.
* **migration** — the next generation's start becomes the global best's archived kernel.
* **refuge** — at least `--refuge` non-elite lineages do not migrate, so a different
  architecture survives. Among the non-elite, one with a *different language* from the
  global best is preferred; ties go to the better metric, then the lower lineage index.
* **retirement** — a lineage leaves the population for good. The population stops when
  every lineage retired, when a generation passes with every active lineage stalled and
  nothing scheduled to migrate (`stalled`), at `--generations`, or at `--max-hours`.

"Stalled" is the lineage's **own gain** this generation, at or below its noise floor:

* ordinary task with its own previous best as reference: `speedup - 1` (exactly the old
  chain's rule);
* board task: the SOL-score difference (score points, floor `--s-floor`);
* an adopted/foreign reference: `previous_ms / current_ms - 1`, because the reference changed.

The **global best / elites** are ranked by `Metric.order`: the SOL score when there is one,
otherwise the absolute median latency (`-ms`). A lineage's `speedup` alone is relative to
its own reference and is *not* comparable across lineages, so it is never the ranking key
when an absolute median is known.

---

## Metrics

`metric.py` reads a generation's evidence; it never times anything itself.

| Kind | Where it comes from | Floor |
|---|---|---|
| `sol` | mean SOL score: `rsi_kernel_tune.benchx.reader.read_run(run_dir, task_dir)` (task benchx) → `rsi_kernel_tune.campaign.solscore.score_kernel(eval_json)` (task campaign) → the card's own `performance.leaderboard.sol_board` in the confirmation jobs' eval JSON (`sol_score_mean`, or per-workload `sol_score`) | `--s-floor` |
| `speedup` | `launch.best_of()["speedup"]` — the confirmed kernel's same-job ratio | ordinary: the confirmation ratio spread, min 1.5 % (identical to `scripts/self_evolve.py`); board fallback: `--s-floor` |
| `ms` | the confirmed absolute median, only when no ratio exists | 1.5 % |

Every board reading carries `confidence = ("dev-only", …)` (and `sol-clock-warning` when
the card says the clock was not locked); the archive marks those rows **待榜单确认** —
a development-machine reading must still be confirmed on B200.

The three hooks are soft dependencies: absent package, wrong shape or a raised exception
falls through to the next source, never fails a generation.

---

## Task directories and handoff

Each generation writes `tasks/<op>.e<lineage>.g<k>/` (`tasks/<op>` for generation 1 and
`tasks/<op>-g<k>` for later generations when `--lineages 1`). The directory copies
`description.md`, `shapes.json` and `inputs/` from the base task, overrides
`[task].language`, sets `[target].ratio = --step` (except generation 1 of a one-lineage
population, which keeps the base target, exactly like the old chain), optionally sets
`[agent.student].effort`, and is validated with `load_task` immediately. Directories are
working products: their paths are appended to the repository's **local**
`.git/info/exclude` (never `.gitignore`) and are never committed.

### Ordinary tasks (source reference)

* A Python donor becomes `reference/kernel.py` with
  `[reference] run = "reference/kernel.py:run"`, `source = "reference"` — the same source
  mode `scripts/self_evolve.py` uses. The donor is also copied into `prior/`.
* A `.cu` donor **cannot** become an ordinary task's source reference (the `[reference]`
  entry must be a `file.py:fn` Python entry). When a lineage's own best is a `.cu`, the next
  generation retries from the same reference and logs that limitation; the `.cu` version
  stays in the archive for campaign.
* With no donor, the base task's own `reference/` is copied verbatim.

### Board tasks (no `[reference]` allowed)

* With a donor, the next generation is a fresh `forge init --leaderboard …`:
  * readable default: `--init-kernel <donor kernel>` (task warmstart); the donor also lands
    in `prior/`;
  * `--sealed-ref <solution.json>` when the user asked for a black box
    (`--sealed-ref`/`--blackbox`), when warmstart is not installed, or when the donor is a
    `.cu` / the lineage's language is `cuda` (a sealed reference is language-agnostic).
    The solution JSON comes from `launch.cmd_export` of the donor's run.
  * a `--from-lib` seed has no run to export: with no readable handoff available
    (warmstart absent, `--sealed-ref`, or a cuda lineage) the seed is **dropped** with a
    logged reason and the lineage starts from the task's own reference — the 2nd-version
    fallback, never a failed generation.
* The re-init reconstructs `--suite/--dataset/--definition/--workload` from the card's
  frozen `problem.json` (`source.dataset_root`, `source.problem`, the group indices / uuids).
  The dataset tree must exist on this machine; otherwise the lineage fails with that reason.
* The board's T_b / T_SOL are carried over when the card covers every workload of its
  problem (indices `0..n-1`): a record is rebuilt and passed as `--sol-board`, so descendant
  cards still score SOL. A sub-range card gets `--no-sol-board`; that lineage's metric then
  falls back to the in-job reference ratio.
* The card's `kernel_file` and the lineage's language must agree (`kernel.cu` ↔ `cuda`);
  a mismatch re-initialises the card with the right `--kernel-file` and is checked after
  every generation.
* With no donor, a board task is copied wholesale (no dataset, no network) and only the
  target/language/effort are patched.

---

## Diversity

* **Language** — `--langs` rotates over lineages. A language is accepted only when
  `rsi_kernel_tune.task.LANGUAGES` accepts it at runtime; otherwise the command exits 2 listing
  the accepted languages. (The `triton` task may extend that tuple later.)
* **Architecture hypotheses** — `--hypotheses dir/` reads `lineage-<i>.md` per lineage, or
  `--hypotheses file.md` for all. The path is injected through
  `rsi_kernel_tune.runcontext.env_for({"hypotheses": …})` (`FK_HYPOTHESES_PATH`) into every
  `forge run` subprocess. The harness does not constrain the file's content beyond the
  standing rules: it is **runtime data**, and it must not contain the reference kernel's
  implementation body (the operator-agnostic prompt invariant). This task never asks an LLM
  to write hypotheses.
* **Library seeds** — `--from-lib` calls `rsi_kernel_tune.klib.find/find_kernels/search(problem, k)`
  (task seedlib, soft dependency), keeps only merge-allowed licenses
  (Apache-2.0 / MIT / BSD) and existing files, and gives each lineage its own seed. With no
  library available it logs the fallback and starts every lineage from the base reference,
  exactly like `--from-lib` absent.
* **Effort** — `--efforts high,max` sets `[agent.student].effort` per lineage, so a pilot can
  A/B settings on the same problem.
* **Target profile** — always passed when available (`--target-profile`, else
  `FK_TARGET_PROFILE_PATH`, else `target/B200.json`). With no profile the command warns; it
  does not invent one.

---

## Concurrency

`--parallel P` caps how many `forge run` child processes are in flight. Task creation and
`forge new` stay serial (they write into `.artifacts/`); only the runs fan out. The GPU work
*inside* a run is serialised by RSI-Kernel-Tune's own per-GPU lock (`FK_JOB_LOCK_CMD` included),
so parallel lineages mean parallel agent thinking, not concurrent benchmarking. `forge run`
itself is never wrapped in `with-lock`.

---

## Resume and state

```
.artifacts/evolve/<problem>-pop_<id>/
  population.json   config, per-generation state, decisions, stop reason
  ledger.jsonl      append-only events (no rewriting)
  variants/         archived kernels (see below)
  solutions/        exported solution JSONs and rebuilt board records
  archive.md        the human-readable table
```

* Rerunning the same command resumes the latest population for that problem (or
  `--population DIR`). If the stored state says `interrupted`, the marker is cleared and the
  in-flight generation continues; finished generations are never run again.
* A lineage whose run was recorded but not finished is handed to `forge run` again, which
  resumes from its own records.
* A subprocess failure records one failure for that lineage and does not kill the
  population; `--max-fails` retires a lineage that keeps failing.
* `forge evolve status <dir>` prints the generation/lineage table and decisions;
  `forge evolve show <dir>` prints `archive.md`.

---

## Archive

Every generation's confirmed best is copied to `variants/<lineage>-g<gen>/kernel.<ext>` with
`parent.txt` (the variant it started from, `init` for generation 1) and `result.json`
(metric, floor, run, sha, language, reference, time). **Every correctness-passing version**
of that generation (`status.jsonl` `KEEP` + `correctness=True`) is kept under
`history/<sha12>.<ext>` and listed in `versions.json` — campaign's per-workload best-of picks
from those. `archive.md` is rebuilt mechanically: one table row per variant plus a comment
block (`# Variant:`, `# Lineage / generation:`, `# Language:`, `# Reference:`, `# Measured:`,
`# Run:`) and the first lines of each kernel. No LLM writes into the archive.

---

## Soft dependencies

| Package / task | Used for | Without it |
|---|---|---|
| `rsi_kernel_tune.runcontext` (step 0) | `env_for()` for target / hypotheses / lessons | the documented env names (`FK_<NAME>_PATH`) are set directly |
| `rsi_kernel_tune.klib` (seedlib) | `--from-lib` seeds | log + language rotation from the base reference |
| `rsi_kernel_tune.warmstart` (warmstart) | board `--init-kernel` (readable handoff) | sealed `--sealed-ref` handoff (black box) |
| `rsi_kernel_tune.benchx.reader` (benchx) | mean SOL score of a run | campaign hook, then the card's own `sol_board`, then `speedup` |
| `rsi_kernel_tune.campaign.solscore` (campaign) | `score_kernel(eval_json)` | card's own `sol_board`, then `speedup` |
| `rsi_kernel_tune.targethw` | not called directly; confidence hints come with benchx readings | only `dev-only` |

The `forge evolve` dispatch itself belongs to step 0's extension dispatcher
(`rsi_kernel_tune/__main__.py`); this task adds `src/rsi_kernel_tune/ext/evolve.py` with the
`COMMAND`/`main` contract and an `ext/__init__.py` so the module imports as
`rsi_kernel_tune.ext.evolve` (`python -m rsi_kernel_tune.ext.evolve --help` works today).

---

## Relation to `scripts/self_evolve.py`

* `--lineages 1` is the regression mode: `tasks/<op>` for generation 1, `tasks/<op>-g<k>`
  after it, `gain = speedup - 1` compared with `max(1.5 %, ratio spread)`, stop on a stall
  or on no confirmed best, exit 1 only when generation 1 has no best. `self_evolve.noise_floor`
  is reproduced verbatim and compared in the tests.
* With N > 1 the same chain lives *inside* the population: a stalled lineage can be rescued
  by migration instead of ending the run, and the population stops when nothing can improve.
* The old script is untouched and remains usable.

---

## Limitations

* A `.cu` donor cannot seed an ordinary task's source reference (documented above); such a
  lineage retries from the same reference, so it does not inherit its own previous best.
* Hypotheses are user/tool supplied; `forge evolve` does not generate them and only checks
  that the paths exist.
* The campaign/benchx metric hooks are interface guesses against not-yet-merged tasks
  (`reader.read_run(run_dir, task_dir)`, `solscore.score_kernel(eval_json)`); when those
  tasks land with different signatures, the native card reader is the fallback and the
  adapter in `metric.py` is the one place to adjust.
* The child interpreter is the PATH's `python3` (with `PYTHONPATH=<repo>/src`), not
  `sys.executable`: a path-valued executable plus a path argument is what command-injection
  scanners flag. Any Python 3.11+ on PATH runs the harness; a deployment that needs a
  specific interpreter must put it first on PATH.
* Per-lineage `speedup` values are not comparable across lineages by construction; ranking
  uses the absolute median when it is known, the SOL score for board tasks.
* The `--efforts` axis is wired through task.toml but has no dedicated test beyond the
  lineage patch; the pilot still has to judge the A/B on real runs.

---

## Files

```
src/rsi_kernel_tune/ext/evolve.py        the command (COMMAND = "evolve")
src/rsi_kernel_tune/ext/__init__.py      makes the extension package importable
src/rsi_kernel_tune/evolve/population.py state machine, resume, scheduling, archive writing
src/rsi_kernel_tune/evolve/lineage.py    task directories, handoff, library seeds
src/rsi_kernel_tune/evolve/migration.py  elites/stall/migration/refuge/retirement (pure)
src/rsi_kernel_tune/evolve/metric.py     a generation's reading from its run's evidence
src/rsi_kernel_tune/evolve/archive.py    variants, per-version history, archive.md
tests/test_evolve_migration.py       pure selection rules + the local-optimum scenario
tests/test_evolve_lineage.py         ordinary/board task generation, argv contracts
tests/test_evolve_population.py      run, resume, concurrency, archive, metric, CLI
```
