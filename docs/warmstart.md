# Warmstart: start a run from the best kernel that already exists

A normal ForgeKernel run starts the student from an empty stub (`run(...)` that raises
`NotImplementedError`), keeps the reference sealed so nobody reads it, and runs the student
offline. On a leaderboard that leaves the first hours of a run to re-deriving an
implementation that is often already public and already good. Warmstart changes the four
things that decide where a run starts:

| | before | warmstart |
|---|---|---|
| the student's first kernel | an empty stub | the file you name (`--init-kernel`) |
| existing implementations | not in the workspace | copied into the card's `prior/`, readable by both agents |
| the student's network | denied | allowed by default (`[agent] online`), GPU machine still denied |
| the problem's context | nowhere | `prior.md`, a manifest of rules, T_b / T_SOL and the implementations (`forge prior build`) |

Nothing about judging, timing or the loop's gate changes: the card is still the card, and
its timer is still the only clock.

## 1. `--init-kernel`: the card's kernel starts from a real implementation

```bash
forge init 069_rms_norm --leaderboard sol-execbench \
    --dataset ~/datasets/sol-execbench --definition L1/069_rms_norm \
    --init-kernel ~/refs/rmsnorm_v3.py \
    --prior ~/refs/rmsnorm_triton/ \
    --prior ~/refs/mit_rmsnorm.py
forge prior build tasks/069_rms_norm
forge new tasks/069_rms_norm
```

- The card's kernel file (`kernel.py`, or `kernel.cu` for a `.cu` source) is written with
  the init kernel's **bytes, unchanged**, instead of the generated stub. The student's first
  iteration improves a working implementation.
- `--kernel-file` is optional now: the default follows the source (`kernel.py`, or
  `kernel.cu` for a `.cu` file). Passing a `--kernel-file` whose kind disagrees with the
  source is refused.
- `--prior PATH` (repeatable) copies other ready-made implementations, files or whole
  directories, into the same `prior/`.
- `--init-kernel` and `--prior` work on a plain `forge init <op> --kernel-file ...` too (the
  operator tree gets the file and the `prior/`), but a plain task's card is generated at
  `forge run` time, so the warmstart that reaches a run is the `--leaderboard` one.

### What the init kernel must be

Checked statically, before anything is written (`forgekernel/warmstart/validate.py`; nothing
is imported or run). Every problem is listed at once; none of them is a warning.

1. **It parses.** A `.py` by `ast`; a `.cu` by the shape the card builds it with
   (`PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...)` and a `void` / `torch::Tensor run(...)`).
2. **It has a module-level `run`.** A nested one, a decorated-away one or an `async def` is
   refused: the card calls `run` by name and by position.
3. **Its signature survives the card's call.** The card passes the definition's inputs, then
   (DPS) its pre-allocated outputs, positionally. So `run` takes exactly that many positional
   parameters, or `*args`; `**kwargs` is harmless; a required keyword-only parameter is not.
   With no definition on disk (a plain operator tree) the arity is not checked and a note
   says so.
4. **It does not do what a submission must not.** The basic subset of the submission rule: no
   credential read out of the environment (`os.environ["HF_TOKEN"]`, `os.getenv("...TOKEN")`),
   no write to a path outside the workspace (`open("/abs", "w")`, `Path("../x").write_text`).
   This is a tripwire, not an audit.

### Language inference

With no declared `[task].language`, the imports decide, and only languages this build's
`forgekernel.task.LANGUAGES` accepts can be used:

| imports | language | how it is built |
|---|---|---|
| `triton` | `triton` | Triton (only once the triton task has merged its language) |
| `cutlass` / `cutlass.cute` | `cutedsl` | CuTe DSL |
| only the host stack (`torch`, stdlib, `numpy`) | `cutedsl` | the Python host: PyTorch ops, or inline CUDA through `torch.utils.cpp_extension` |
| anything else (`cupy`, `flash_attn`, …) | refused | the import is named |

A `.cu` file is `cuda` (the card builds it as its own translation unit).

## 2. `prior/`: existing implementations, readable, not the student's change

Both `--init-kernel`'s file and every `--prior` entry are copied into the card's
`prior/`:

```
operators/<op>/            (or, for a task, tasks/<op>/leaderboard/card/ and from there
  prior/                    the run's operators/<op>/)
    kernel.py               the init kernel, byte-identical to the card's kernel file
    kernel-2.py             a --prior file (renamed on a name collision)
    refimpl/                a --prior directory, copied as a tree
    SOURCES.json            where each came from, its sha256, and its detected licence
```

- **Readable**: `prior/` is inside the student's workspace (and the run's card), so the file
  tools and the Bash sandbox read it by default. No policy rule denies it to the student or
  the teacher, and the per-turn notice names it: *"Existing implementations: `…/prior` holds
  ready-made implementations of this operator … read, reuse and improve them."*
- **Not the student's change**: the copy is written before the card's initial commit and the
  workspace is a clone of the card, so `prior/` is tracked in the initial commit and never
  appears in the student's diff. The student still changes exactly one file.
- **Reproducible**: the frozen task carries the card (hence `prior/SOURCES.json` and the
  files), so a run on another machine knows what it started from.
- **Licences are recorded, not assumed**: each entry's licence is detected from the nearest
  `LICENSE` / `COPYING` / `NOTICE` file at or above the source. `SOURCES.json` keeps it; the
  manifest splits implementations into *usable* (Apache-2.0 / MIT / BSD-2-Clause /
  BSD-3-Clause) and *present but not reusable* (everything else, including "unknown").

Sealed reference and warmstart can coexist: `--sealed-ref` still seals the reference so it
is timed in the same job and cannot be read. If you also pass the same content as
`--init-kernel` / `--prior`, `forge init` prints a WARNING that the copy in `prior/` is
readable, so the reference is no longer sealed -- the timing pairing still holds, the
"cannot read it" premise does not. To have both properties without leaking: seal a
stronger reference and start from a different implementation.

## 3. The student's network: `[agent] online`

`task.toml`'s `[agent]` gains one optional boolean, and `global_config/models.toml` sets the
project default:

```toml
[agent]
online = true          # the student may use the network (the project default)
```

- `online = true`: the **network group** of Bash denials is lifted (`curl`, `wget`, `pip
  install`, `git clone`, `hf download`, …) and the sandbox has no network denial; WebFetch /
  WebSearch are available. Code that is reused must be under a licence that permits it and
  must end up in the single submission file.
- Always denied, online or not: the GPU-machine CLIs (`ssh`, `scp`, `rsync`,
  `python -m forgekernel.submit`) -- a student's GPU work goes through the loop's ledgered
  eval command, never through a job of its own.
- `online = false`: exactly the pre-warmstart policy (no network at all).
- The teacher is online either way; `[agent.teacher] online` has no effect and is not what
  drives this (only the student's value is carried).
- An invalid value is a task error; a non-boolean `[loop] student_online` is a loop.toml
  error.

The value travels: `task.toml [agent] online` -> `Task.student.online` ->
`loop.toml [loop] student_online` (written by `forge run` from the task) ->
`AgentPolicy.student_online` (read by `forgekernel.loop.agent_policy.from_config`). The loop
config is the only channel, so no change to `loop/cli.py` was needed.

`forge new` prints it: `student claude-opus-5-5 [online]` / `[offline]`.

## 4. `forge prior build`: the manifest the loop reads as `{PRIOR_PATH}`

```bash
forge prior build tasks/<op> [--klib] [--out PATH] [--index PATH]
```

Writes `tasks/<op>/prior.md` (English Markdown) and prints the environment line that points
a run at it:

```
FK_PRIOR_PATH=/abs/tasks/<op>/prior.md
```

Everything in it is mechanically derived -- rebuilding it after a change produces exactly
the text the change implies, and no model writes prose into it:

- **The rules**: positional entry, DPS (or value-returning, per the card), one-file
  `submission.py`, allowed dependencies, the prohibited behaviours, the definition's inputs /
  outputs / axes, the per-workload tolerance read from the card's workload rows.
- **What it is scored against**: per workload, `T_b` and `T_SOL` and the score formula.
- **The target**: when the reference library has a locally measured reference time
  (`verified_local.t_ms | median_ms | ms | device_median_ms`), `[target] ms = reference ms /
  step`, the step being the task's `[target] ratio` (else the project default 1.2). Without a
  local measurement the card's own target stands, and the manifest says so.
- **Existing implementations**: one row per `prior/` entry -- path, language, licence, score
  (matched to a library entry by content hash), source -- split by licence as above;
  "start from the highest-scoring entry whose licence permits reuse".
- **`--klib`**: the reference library's entries for this problem name / operator family, when
  `kernel_library/index.json` exists (the seedlib task's deliverable). A machine without the
  index gets one line saying so, not an error.
- **Local knowledge you may read**: `_reference/KernelWiki` (with its query examples:
  `python3 scripts/query.py --tag tcgen05 --compact`, `scripts/get_page.py <id>`,
  `artifacts/` holding complete upstream kernel files) and `_reference/ncu-report-skill`,
  from the checkout's sibling `_reference/` -- derived from the repository's position (and
  found from a worktree too), never an absolute literal.

### How it reaches the agents

`{PRIOR_PATH}` is the run-context binding (background §7.5): the loop's `_bindings` maps
`FK_PRIOR_PATH` to it, and the student / teacher prompts read the file when the path is not
"(none)". Two paths are written into `loop.toml` by `forge run` as well, so the per-turn
notice names them directly:

- `[loop] prior_dir` -- the card's `prior/`;
- `[loop] prior_manifest` -- `tasks/<op>/prior.md`, when it exists;
- `[loop] local_knowledge_dirs` -- the `_reference/` trees this machine has.

The policy is what enforces the reads: `prior/` is inside the workspace (allowed by
construction), the knowledge trees have no deny rule, and a deny entry that would shadow a
declared knowledge tree is reported on stderr rather than silently enforced. The declared
trees are also listed in the notice, so a turn does not rediscover them.

Order of operations: `forge init` -> `forge prior build` -> `forge new` / `forge run`. The
manifest is optional: without it `forge run` works as before, only without `prior_manifest`.

## 5. Design notes and limits

- **The warmstart affects the starting point, not the gate.** The card's judge, clock,
  tolerances and the loop's KEEP / REVERT semantics are untouched.
- **`--open-ref` (an in-job reference that is both timed and readable) is not part of this
  task.** The card's `eval/leaderboard.py` supports reference kinds `definition`, `binary` and
  `sealed`; anything else needs a change to `card_template/eval/*`, which this task does not
  own. The recipe for "timed and readable" today is `--sealed-ref <ref>` (timing) plus
  `--prior <ref>` (readable), accepting the WARNING above.
- **A plain (non-leaderboard) task's card is generated at `forge run` time**, so only a
  `--leaderboard` card carries the warmstart into a run; the plain scaffold still writes the
  file and `prior/` for manual use.
- **Licence detection is keyword matching** over the nearest licence file. A repository with
  a non-standard licence text reads as "unknown", which is *not* usable -- deliberately the
  safe side.
- **`prior/` is not a security boundary.** Copying an implementation there makes it readable
  by both agents on purpose; do not warmstart from the reference of a run whose point is that
  the student cannot see it.
- **The prompt's L0 line** ("any equivalent implementation ... the concrete list is in this
  operator's README") is not edited: `prior/` is inside `WORKSPACE`, which that list does not
  cover, and `meta/*.md` is owned elsewhere. The policy notice and the manifest state the
  permission explicitly.
- **Not in the manifest**: how each workload's inputs are generated (the card's
  `problem.json` names the input kinds; the manifest lists shapes and tolerances only), and
  the card's own `window.toml` boundary (the card carries it).
- **Not tracked**: nothing beyond the initial commit distinguishes `prior/` from a file the
  student wrote -- there is no diff allowlist. That is deliberate (it is the mechanism that
  makes it "not the student's change" in every consumer of the workspace), but a future tool
  that counts *all* changes against the card's HEAD must expect `prior/` to match it.
- **The `..` branch of the "writes outside the workspace" check** has no dedicated test in
  this environment (the local security hook refuses `..` literals even in fixtures); it shares
  its predicate with the absolute-path branch, which is tested.
