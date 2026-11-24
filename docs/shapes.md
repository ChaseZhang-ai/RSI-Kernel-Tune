# shapes: one kernel per shape, and one entry point that dispatches between them

`forge shapes` turns "one kernel serves every workload of a problem" into "one kernel per
size class, and one `submission.py` that routes each input shape to the implementation that
was written for it". It does not touch the loop: each cluster is an ordinary board task
driven with the ordinary `forge init` / `new` / `run` / `export`, and the composition is
finished by a dispatcher this tool generates.

- commands: `forge shapes plan | run | compose` (`src/forgekernel/ext/shapes.py`)
- modules: `src/forgekernel/shapes/` (`cluster`, `plan`, `run`, `evalx`, `matrix`, `compose`)
- products: `tasks/<op>/shapes/` (a working directory, kept out of git via
  `.git/info/exclude`, never `.gitignore`)
- tests: `tests/test_shapes_{cluster,matrix,compose,run}.py`, fixture card
  `tests/fixtures/shapes_card/`

## 1. Why per-workload dispatch

Three reasons, in the order they matter:

1. **The score is per workload.** A board workload's score is
   `S = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))` and a problem's score is the arithmetic
   mean of its workloads' `S` (background §3.3). Every workload running its own fastest
   correct implementation is therefore *the* optimum of that formula, not an approximation
   of it.
2. **One kernel for every size is the known failure.** ForgeKernel's own README lists
   "Shape clustering: group the shapes of a task into clusters and optimize per cluster, so
   one run can cover a wide range of shapes" as the plan for v1.5, and the reported evidence
   is stark: the same kernel reached 1.037x on Prefill and 0.289x on Decode, 0.547x overall
   (issue #8). Those are two size classes of one problem; nothing about them wants the same
   tile.
3. **The dispatch is not on the clock.** The card times a CUPTI *device* window around the
   kernel launches (`device_window_basis`, see `docs/leaderboards.md`), and the routing is a
   tuple-of-shapes dict lookup in Python that happens before the first launch. So a
   dispatcher that always picks the right kernel costs nothing measurable.

Caveat for host-bound kernels: the argument above is about the board's stopwatch, not about
wall-clock throughput in a serving loop. If a judging protocol ever measured end-to-end
host time, a dict lookup plus one extra Python frame would be ~1 µs per call and mattered
only for kernels in that range.

## 2. Clustering (`cluster.py`, pure)

Input: the card's `problem.json` (`group` entries with `axes` and `sol_board`). Each
workload becomes `Workload(index, name, axes, t_b, t_sol, elements)`, where `elements` is
the definition's tensor element count at those axes (the same helper `meta_workload_names`
uses, imported rather than re-derived).

The rule, in order:

1. **Order** the workloads by `T_SOL` (the board's analytic floor for that shape: the size
   of the *problem*, not of one implementation). If any workload has no `T_SOL` -- a card
   built with `--no-sol-board`, or a group entry the board did not cover -- **every**
   workload is ordered by element count instead and a note says so: mixing two metrics in
   one ratio would compare unrelated numbers.
2. **Split** between neighbours whose size ratio exceeds `--split-ratio` (default 2.0).
3. **Fold** any cluster smaller than `--min-size` (default 2) into the neighbour with the
   closest geometric mean (log-space distance; a tie goes to the smaller neighbour).
4. **Merge** the adjacent pair with the smallest ratio until at most `--max-clusters`
   (default 4) remain.

Determinism and order-independence are properties of the implementation, not hopes: sorting
uses `(size, elements, name, index)` -- content, never the position in the input list -- so
shuffling the card's workload order gives the same clusters, and two `plan` runs of the same
input write the same bytes (apart from the `created` stamp, which `build_plan(now=...)` can
pin).

Each cluster's `[target] ms` is `geomean_i(time_for_score(sol_goal, T_b_i, T_SOL_i))` -- the
geometric mean over the cluster's workloads, because that is how the card aggregates a
group's readings. `--sol-goal` defaults to 0.90, the same number campaign uses. A cluster
with no usable `T_b > T_SOL` gets no target (and the plan warns), rather than a made-up one.

Edge cases, deliberately: a card with **one** timed workload is not a clustering problem,
so `plan` says so and exits 0 (a directory that is not a board card at all is a usage error,
exit 2). `--split-ratio <= 1` is refused.

## 3. `forge shapes plan`

```
forge shapes plan tasks/<op> [--max-clusters K] [--split-ratio R] [--min-size N]
                            [--init-args "..."] [--sol-goal S] [--out DIR]
                            [--dry-run] [--json FILE]
```

Writes `tasks/<op>/shapes/plan.json` and lists the cluster task directories in the
repository's `.git/info/exclude` (found with `git rev-parse --git-path`, so a worktree writes
the shared file, not a stray one). `--dry-run` prints the plan and writes nothing.

### `plan.json`

```json
{
  "schema": 1,
  "created": "2026-11-24T12:00:00+0000",
  "op": "069_rms_norm",
  "task": "/abs/path/tasks/069_rms_norm",
  "card": "/abs/path/tasks/069_rms_norm/leaderboard/card",
  "product_dir": "/abs/path/tasks/069_rms_norm/shapes",
  "suite": "sol-execbench",
  "definition": "069_rms_norm",
  "sol_goal": 0.9,
  "params": {"max_clusters": 4, "split_ratio": 2.0, "min_size": 2},
  "init_argv_base": ["--leaderboard", "sol-execbench", "--dataset", "...",
                     "--definition", "L1/069_rms_norm", "--no-sol-board"],
  "workloads": [{"index": 0, "name": "wl0", "axes": {"B": 4}, "t_b": 0.032,
                 "t_sol": 0.004, "elements": 4096}],
  "clusters": [
    {"name": "069_rms_norm-c0", "workloads": [0, 1], "metric": "t_sol",
     "t_sol_range": [0.004, 0.006], "elements_range": null, "size": 2,
     "target_ms": 0.0087,
     "forge_init_argv": ["--leaderboard", "sol-execbench", "...", "--workload", "0,1"],
     "reason": "2 workload(s), T_SOL 0.004..0.006 ms; the next cluster starts 12.1x larger"}
  ],
  "warnings": ["..."]
}
```

The cluster's `forge_init_argv` is the original problem's arguments with the workload
selector replaced: `--workload <this cluster's indexes>`. `--init-args` is the authority on
what "the original arguments" are; it is the only way to carry what the card does not retain
(a saved `--sol-board` record, `--init-kernel`, `--kernel-file`). Without it the arguments are
derived from the card's provenance (`suite`, `source.dataset_root`, `source.problem`) and
pass `--no-sol-board`, because the board's `T_b` / `T_SOL` block frozen in the card is *not*
a `kernels/<id>` record that `forge init --sol-board` re-validates -- a warning says so, and
the cluster's target is still written from the card's table. Every other shapes product is
keyed by this file: `run` executes the argv it records, `compose` scores against the card and
workload list it names.

## 4. `forge shapes run`

```
forge shapes run <plan.json | tasks/<op>> [--root DIR] [--python P] [--max-hours H]
                 [--cluster-hours H] [--sol-goal S] [--dry-run] [--json FILE]
```

Per cluster, in order, each step a child process of this interpreter:

```
forge init <cluster> <the plan's argv>
(write [target] ms and, with --cluster-hours, [budget] hours into tasks/<cluster>/task.toml)
forge new tasks/<cluster>            # prints "run directory .artifacts/<cluster>_<id>"
forge run .artifacts/<cluster>_<id>
forge export <run dir> --out tasks/<op>/shapes/clusters/<cluster>/solution.json
```

`forge export` writes the board solution JSON; `compose` reads the kernel source back out of
its `sources[0]`. Nothing uploads and nothing starts an agent.

`runs.json` is the state:

```json
{"schema": 1, "plan": ".../plan.json", "sol_goal": 0.9,
 "clusters": {"069_rms_norm-c0": {
    "status": "planned|running|solved|failed", "task": "tasks/069_rms_norm-c0",
    "run": ".artifacts/069_rms_norm-c0_261124-101010-abcd",
    "solution": "tasks/069_rms_norm/shapes/clusters/069_rms_norm-c0/solution.json",
    "target_ms": 0.0087, "hours": null,
    "steps": {"init": 0, "new": 0, "run": 0, "export": 0},
    "started": "...", "finished": "...", "error": "", "notes": []}}}
```

Resume semantics: `solved` clusters are skipped; a cluster left `running` whose run
directory exists resumes with `forge run <that dir>` (the loop continues where it stopped);
`failed` retries from the step it reached; a cluster whose task directory already exists is
not re-`init`ed. `--max-hours` caps *this invocation*: unstarted clusters stay `planned`, the
exit code is 3, and running the same command again continues. `--cluster-hours` sets the
loop's own `[budget] hours` per cluster.

GPU exclusion is ForgeKernel's own (its per-GPU job lock and `FK_JOB_LOCK_CMD`, read from
`local.env` by the job path). `run` deliberately does **not** wrap `forge run` in an outer
`with-lock`: an hours-long outer lock would be held through every acquisition the loop
itself makes.

Exit codes: 0 all clusters solved (or `--dry-run`), 2 a usage error, 3 something failed or
the budget ran out (both resumable).

## 5. `forge shapes compose`

```
forge shapes compose <plan.json | tasks/<op>> [--out PATH] [--also FILE ...]
                     [--overall FILE] [--allow-slower] [--no-reuse] [--no-versions]
                     [--no-lib] [--repeat N] [--keep] [--json FILE]
```

### 5.1 Candidates

| kind | where it comes from |
|---|---|
| `cluster` | `clusters/<name>/solution.json`, the export `run` wrote (a specialist) |
| `version` | every KEEP with `correctness != False` in the cluster runs' `status.jsonl`, read from the run workspace's git (`git show <sha>:kernel.py`); evolve populations contribute their `variants/*/kernel.*` |
| `overall` | `--overall`: one run's whole-card export; it also becomes the fallback |
| `extra` | `--also` (repeatable): another agent's solution, a hand-written variant |
| `library` | `kernel_library/index.json` entries whose `problem_keys` name this problem and whose licence is submittable (`forgekernel.klib.schema.submittable`) |
| `baseline` | `forge lib baselines`' mechanical torch candidates (soft dependency on seedlib) |
| `eager` | the definition's own reference, wrapped as a DPS kernel |

A library entry that only matches a *family* keyword is deliberately not taken: turning a
family reference into a kernel for this problem is `forge lib adapt`'s job, and its output
reaches `compose` as an `--also` candidate.

A candidate that does not parse, has no top-level `run`, or is a `.cu` is reported and left
out (`cuda_to_inline`, seedlib, is what turns CUDA C++ into a single-file candidate; until
then a `.cu` cannot enter a `submission.py`).

### 5.2 Readings: benchx, and only benchx

Every candidate is written to `candidates/<label>.py` and scored **once, on the problem's
own card, over every workload** by `forgekernel.benchx` (`Bench.bench(card, path)`). The
reading, the verdict and the per-workload `S` are the card's own. `compose` never times or
judges anything itself: without benchx it says "needs benchx" and stops.

Readings are cached in `matrix.json` by the candidate's source hash and the card's hash, so
re-composing after adding one candidate only benches that candidate (`--no-reuse` forces a
re-bench).

### 5.3 The routing rule

Per workload:

1. **the fastest correct reading wins** -- a cell the card judged incorrect (or that has no
   reading) never competes;
2. **a tie is not a win**: readings within the noise floor (1.5 %, the number
   `benchx.compare` uses) are a tie, and the tie goes to the earlier kind -- cluster
   specialist, that run's other correct versions, overall, extra, library, baseline;
3. **the eager reference is the floor, not a competitor**: a workload whose best specialist
   is slower than the reference beyond the noise floor routes to the **reference**, and the
   route says why. `--allow-slower` keeps the specialist instead and records the workload as
   above the floor;
4. a workload with **no correct candidate** routes to the fallback (`--overall` when it has a
   reading there), else to the reference. The generated dispatcher also keeps a `try/except`
   around every call.

So by default every routed reading is at or below the reference (within the noise floor),
and `shapes_compose.json` records where the guarantee was checked and where (if anywhere) it
was waived.

### 5.4 The dispatcher

`submission.py` is one file:

* each routed candidate's source is inlined with **every module-level binding renamed** by a
  private prefix (`_fk_0_c0_run`, `_fk_0_c0_BLOCK`, ...). Imports are aliased the same way
  (`import torch as _fk_0_c0_torch`); plain `import a.b` is left alone; string literals are
  untouched, so an inlined CUDA source keeps its text. `from __future__` imports are stripped
  from each candidate and the union is hoisted to the top of the file (a future import is
  only legal at the top of a module);
* the definition's reference is pasted prefixed (`_fk_ref_*`, entry `_fk_reference_run`) and
  becomes `_fk_eager` (DPS: call it, write its return value into the pre-allocated outputs;
  value-returning card: return it);
* the entry is `def run(*args)`. The routing key is the shape tuple of every **input tensor**
  (inputs the definition gives no shape to -- scalars such as `eps` or a scale -- are not
  keys, and the pre-allocated outputs at the end are not inputs). The lookup is exact; a
  shape the table does not name runs `_FK_FALLBACK`, and any exception runs `_fk_eager`.

Why an AST rewrite rather than string surgery: Triton's `@triton.jit` and CuTe DSL's
decorators read their function's *source* at compile time. The file we write is a real file,
so renaming a name inside it keeps that property; moving a kernel body into a string and
`exec()`-ing it would not (there would be no source to read). That is also why **only Python
module code is inlined** -- torch, Triton, CuTe DSL and *inlined CUDA strings* all are, while
a `.cu` file is not.

Why `def run(*args)` and not the definition's parameter names: the board's contract is
argument order and count, and names are not enforced (background §3.2). The local validator
accepts both, and requires the names to be exactly the definition's when they are given.

### 5.5 Products

| file | what |
|---|---|
| `submission.py` | the deliverable: one file, top-level `run(*args)` |
| `solution.json` | the same source as an official single-problem solution (`sources` has one entry) |
| `matrix.json` | every reading of every candidate on every workload (also the cache) |
| `shapes_compose.json` | the routing table: who won each workload, its reading, `S`, mean `S`, the guarantee's verdict, the candidates, both local checks |
| `report.md` | the per-workload table a human reads |

`--out` may name a directory (then `submission.py` and its `solution.json` go there) or the
`.py` file itself; the rest stays in `tasks/<op>/shapes/`.

Local checks (run on every compose; their findings land in `shapes_compose.json`):

* `submission.py` compiles (`py_compile`'s check, no bytecode left behind), has a top-level
  `run`, and its signature is either `*args` or exactly the definition's names in DPS order;
* `solution.json` has the required fields, known languages (no C++/Python mix), a known
  `entry_point` whose file is one of `sources`, relative non-duplicated source paths, a DPS
  flag that agrees with the card's, and a submitted file identical to its entry source. The
  multi-source form is checked too -- it is legal for a single-problem submission, just not
  the collection route.

### 5.6 What campaign sees

campaign calls `forge shapes compose <task dir> --out <dir> --also …` (its soft dependency,
`campaign/tools.py`). It works without a saved plan (the default clustering is used and noted)
and leaves `<dir>/submission.py`, which campaign then treats as one more candidate to export.

## 6. Not covered / deliberate limits

* **CUDA C++ candidates** are not composable into one file by this task: they need
  `klib.normalize.cuda_to_inline` (seedlib). They are reported, never silently inlined.
* **Multi-file solution JSON**: supported by the official schema for single-problem
  submissions, and the validator checks it, but it is not what a collection submission takes,
  so `compose` writes one file.
* **No GPU here.** The dev machine is a 6 GB sm_86 card; every reading in tests comes from
  `FakeBench`. The real path (`CardBench`) is exercised by `forge bench`, whose own tests
  cover it.
* **Dispatch cost** is argued from the card's device-window clock, not measured on the board
  (see §1).
* **`element`-based clustering** (no `sol_board`) is exercised only on the fixture; no real
  `--no-sol-board` card has been clustered yet.
