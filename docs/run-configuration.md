# The run configuration: `rsi-run.toml`

One file fills in everything a run needs on top of the task itself: where the
agents' tokens come from, which kernel(s) to optimize, **the precision you
declare**, whether the student may retrieve public sources first, how much runs
concurrently, and — for a new language — its syntax reference and toolchain.

```bash
rsi-kernel-tune config init                    # the commented template, ready to fill in
rsi-kernel-tune config init --language cuda    # pre-fills the [language] section
$EDITOR rsi-run.toml
rsi-kernel-tune run --config rsi-run.toml      # every [kernel], in parallel
```

The file is committed with the repository (it names no secret — see [api] below).
`rsi_kernel_tune.runconfig.load_runconfig` validates it before anything runs, and
every problem it finds names the one-line fix.

> ┌────────────────────────────────────────────────────────────────────┐
> │  **THE PRECISION OF THE OPTIMIZED KERNEL IS YOUR DECISION.**        │
> │  The system never picks a precision, never trades one for speed,   │
> │  and never goes below the one you declare. Default: fp32.          │
> └────────────────────────────────────────────────────────────────────┘

## `[api]` — where the tokens come from

```toml
[api]
provider = "native"     # native | a provider table of global_config/backends.toml
key_file = "local.env"  # the file that holds the key; NEVER the key itself
models = { teacher = "claude-opus-5-5", student = "claude-opus-5-5" }
```

- `provider = "native"` inherits the environment untouched (the agent CLI's own
  login, or the credentials `local.env` exports). Any other name is a table of
  `global_config/backends.toml` (`base_url` / `wire` / `key_env`) that points the
  CLI at another endpoint.
- **The key itself never goes in this file.** It lives in `key_file` (gitignored,
  one `KEY=VALUE` per line — copy `local.env.example`). `key_file` may be a path
  relative to the config file. A `[api]` key whose name looks like a credential
  (`*_API_KEY`, `*token*`, ...) is refused outright.
- `models` is required, per role. On a `run --config` pass these override the
  task's `[agent]` provider/model in the snapshot each child run solves.
- At run time the key file's values are merged over `local.env`'s (the run
  configuration is the more explicit statement); the process environment wins
  over both, the localenv rule.

## `[kernel]` — what to optimize

```toml
[kernel]
task = "tasks/myop"                    # one entry, or a list of entries
# or: from_kernels = "gemm_fp16"
```

One entry per kernel task; each becomes one run (its own `.artifacts/<op>_<id>/`
with its own `operators/` and `ws/` — runs never write into each other). An
entry may be a single string or a list. `description.md` in the task dir holds
the kernel's goal/prompts and is yours to edit.

`from_kernels` points at a published kernel (`kernels/<family>/<op>/<arch>/`).
The task must already be scaffolded from it — the kernel's frozen contract (its
docstring) is the user's to transcribe, so the tool points, it does not guess:

```bash
rsi-kernel-tune init myop --from-kernels --arch sm100 --axes '{"m": 4096, "n": 4096, "k": 4096}'
```

A `from_kernels` entry without its scaffolded task is refused at `run --config`
with that exact command.

## `[precision]` — the compute precision YOU declare

```toml
[precision]
compute_dtype = "fp32"     # fp32 | tf32 | bf16 | fp16 | fp8
```

This is the decision the banner above is about. The semantics, exactly:

- **The student implements at the declared dtype.** Its brief carries one
  contract sentence: *"This task's contract declares compute_dtype=<X>. Work
  within it; going below it is a contract violation (REVERT)."*
- **The precision gate evaluates the reference AT the declared dtype** (the
  card's jobs receive `RKT_COMPUTE_DTYPE`; eval/benchkit.py runs the reference
  with tf32 matmuls for tf32, lifts its computation to the declared dtype for
  bf16/fp16, and quantizes its outputs to the declared grid for fp8). `e_ref` is
  then the reference's own error at the declaration, the floor is the declared
  output dtype's ulp, and the relative bound `3*e_ref + floor` is unchanged —
  eval/precision.py's measure is untouched, byte for byte.
- **Arithmetic below the declaration is a conviction**, not a style point: a
  kernel that quietly computes one grade coarser than the declaration lands past
  the bound and the strict gate REVERTs it, naming the declaration (the red-team
  fixture `tests/redteam/test_redteam_compute_dtype.py` pins the scenario).
- **No declaration = fp32**, and every surface fp32 touches is byte-identical to
  the pre-declaration behavior: no prompt text, no submit.toml line, no digest
  stamp, the historical gate text.
- **Numbers never mix across declarations** (the memo-lane rule): the digest and
  the run-end summary stamp the declaration, and a carried lever price read
  under another declaration is dropped, not juxtaposed.
- A task's own `task.toml` may declare `[precision]` too. A run-config-level
  declaration fills in a task that did not declare one; where both declare and
  disagree, the run is refused — one precision per kernel task.

## `[student]` — the retrieval permission

```toml
[student]
retrieval = "off"     # off (default) | on
```

- `"off"` (the default) is the L0 contract: the student optimizes **from
  scratch**, no public-source retrieval, offline sandbox, byte-identical
  prompts and policy.
- `"on"` lets the student survey public sources first, then tune on that basis:
  WebSearch / WebFetch and the survey tools become available, and its briefs
  carry the two-line permission. **Retrieval is not the answer** — the reference
  implementation itself, and any optimized implementation of it, stay denied
  exactly as before (deny lists and the policy hook are untouched), and code
  seen online must not be copied into the kernel.

## `[run]` — the concurrency

```toml
[run]
parallel = "auto"     # auto | 1 | 2 | 4 ...
```

- `"auto"` (the default): **one concurrent slot per kernel task** — add a kernel
  task, add a slot; every kernel of `[kernel]` runs in parallel.
- An explicit number caps the slots (the remaining kernels queue).
- Each slot drives its own child process (`python -m rsi_kernel_tune run
  <run-dir>`) on its own run directory. GPU jobs serialize on the existing
  per-GPU lock; with several GPUs the slots are assigned round-robin over the
  GPUs the machine reports (`RKT_GPU` per slot), so a slot's jobs land on their
  own device where there is one.
- The parent waits on every kernel, reports each one's final status, and exits 0
  only when every kernel's final confirmation met its target.

## `[language]` — optional, a NEW language only

```toml
[language]
name = "cuda"
syntax_docs = "docs/cuda-style.md"        # syntax/style reference, read first
toolchain_config = "toolchains/nvcc.conf" # the language's toolchain/compile config
```

- `syntax_docs` (must exist): injected as one line at the top of **every teacher
  and student brief** — `language syntax reference: <path> (read it first)` —
  judge-writing turns included. Absent on runs without the section.
- `toolchain_config` (must exist): passed through to the eval templates' build
  face. The card's `submit.toml` carries it to every job as
  `RKT_TOOLCHAIN_CONFIG`; eval/benchkit.py merges the file's flags into the CUDA
  build (whitespace-separated tokens, `#` comments), each flag under the same
  allow-list as the kernel's own `fk-cuda-cflags` lines — the file widens the
  build for the language's toolchain, never the attack surface.
- **A new language needs more than this section.** The harness's language
  lanes are `meta/<lang>/` (the prompt overlay: `meta/cuda/<name>` files where
  they differ) and `hw_kb/<lang>/<arch>/` (the hardware knowledge base the run
  reads and writes; see `hw_kb/README.md`). v1 does not generate these: model
  the overlay on `meta/cuda/` and seed the KB with
  `hw_kb/<lang>/<arch>/TEMPLATE.md` (copy from an existing tree, then let the
  teacher certify entries). `[task] language` in the task itself also needs the
  language registered (`rsi_kernel_tune.task.LANGUAGES` / `LOOP_LANGUAGE`) —
  that registration is a code change by design: a new language is a harness
  decision, not a configuration typo.

## What stays where it was

The run configuration does not replace the layers below it; it points at them:

- `tasks/<op>/task.toml` — the problem itself (shapes, reference, target,
  budget, measurement clock, lane). `docs/design/` and the module docstrings
  hold the full key reference.
- `local.env` — the machine (GPU, ssh delivery) and the credentials. The `[api]`
  key file is a second such file, merged over it at run time.
- `global_config/` — the defaults (`models.toml`, `loop.toml`, `backends.toml`).
