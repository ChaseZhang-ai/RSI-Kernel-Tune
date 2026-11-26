# Extension points: the base the bestKernel work is built on

> This is the design record of what step 0 ("基座准备") put into the repository on `bk/base`: the
> places where the work that follows plugs in, what each one's contract is, and what is an
> interface waiting for an implementation. §0–§7 are the extension points, §8 the development
> environment, §9 what a task that builds on them must keep, and §10 the trade-offs made in
> implementation and where the result departs from the plan.
>
> The plan itself, the facts about the board and the rules of the project are not repeated here;
> they are in the project's prompt files (`prompts/00-背景与约定.md`, `prompts/01-总控.md` §2).

---

## 0. Principles

1. **An interface first, its one implementation later.** Where several tasks depend on the same
   thing (evaluating a candidate, the reference library's index), step 0 puts down the interface
   and a fake, and nothing real: the tasks code against one contract instead of each inventing one.
2. **A prompt stays operator-agnostic; what is specific to a run is a file.** The GPU a result is
   graded on, the roofline, the lessons of an operator class, the kernels that already exist —
   none of it is written into `meta/`; each reaches an agent as a file the prompt names.
3. **Nothing here can break what was there.** A run that uses none of the extension points
   behaves exactly as it did on `b2c8ae8`; every existing test still passes unchanged.
4. **A broken extension breaks only itself.** A command extension that does not import is
   reported when it is called and nowhere else; a context file that is not set is `(none)`.
5. **Only what was verified is written down as fact.** What has not been measured is marked so.

---

## 1. Overview

| Extension point | Where | What is here | What is a task's |
|---|---|---|---|
| A. Command extensions | `src/rsi_kernel_tune/ext/` | the registry, the dispatch in `__main__.py`, reserved task names | every command (bench, lib, target, prior, campaign, evolve, board) |
| B. Run context files | `src/rsi_kernel_tune/runcontext.py`, `loop/driver.py`, `meta/*.md` | four names and variables, four prompt bindings, the prompt sections | writing the files (targethw, roofline, opkb, warmstart, seedlib) |
| C. Operator families | `src/rsi_kernel_tune/family.py`, `families.toml` | the rule table, `family_key`, `is_backward` | corrections, reported rather than worked around |
| D. External GPU lock | `submit/direct.py`, `FK_JOB_LOCK_CMD` | complete | — |
| The target profile | `target/B200.json`, `src/rsi_kernel_tune/target_profile.py` | a draft of verified facts; the Markdown rendering | the final profile and how it reaches a run (targethw) |
| The `benchx` stub | `src/rsi_kernel_tune/benchx/` | the protocol, the data classes, `FakeBench` | `CardBench`, the reader, compare (benchx) |
| The `klib` stub | `src/rsi_kernel_tune/klib/` | the index schema, its checks | building, finding and scoring entries (seedlib) |

---

## 2. Extension point A: command extensions

```
src/rsi_kernel_tune/ext/
  __init__.py      the registry (adds no command)
  <name>.py        one command each: COMMAND = "<name>" and main(argv) -> int
```

`forge <name> ...` runs `main` with what follows the command and makes its return value the exit
status (`None` is 0). Nothing is registered anywhere.

- **Found by listing the directory, imported only when called.** `ext.commands()` names every
  module under `ext/` whose name does not start with `_` (`pkgutil.iter_modules`) and imports
  nothing. So an extension that is broken -- it does not import, does not parse, has no callable
  `main`, or its `COMMAND` is not its name -- cannot break `forge --help` or any built-in
  command. Called, it prints `[forge] extension command '<name>' failed to load: <what is
  wrong>` and answers 2. What a loaded extension's `main` raises is its own, as for a built-in.
- **The command name is the module's name.** `COMMAND` must say the same. A name that could only
  be read by running the module could route nothing when the module is the thing that is broken,
  and could not be listed in the usage text without importing it.
- **Built-in commands come first.** `__main__` looks the table up after every built-in command
  and before the operator commands (`forge <op> submit ...`). So an extension named `init`,
  `new`, `run`, `status`, `harvest`, `export` or `help` is never reached, and is left out of the
  usage text. The second word matters as well: after any first word, `loop`, `baseline`,
  `window`, `leaderboard`, `monitor`, `ncu-trace` and `ncu-report` are built-in operator
  commands -- **an extension must not have a subcommand of one of those names.**
- **`forge --help`** ends with the extension commands found, by name:
  `Extension commands (src/rsi_kernel_tune/ext/, one module each): ...`.
- **A task may not take an extension's name** (`task.load_task`): `forge <name>` would never reach
  that task's operator commands. The names refused are whatever is in the directory when the task
  is read.

Tests: `tests/test_ext.py`.

---

## 3. Extension point B: run context files

| Name | Environment variable | Prompt placeholder | What the file holds | Written by |
|---|---|---|---|---|
| `target` | `FK_TARGET_PROFILE_PATH` | `{TARGET_PROFILE_PATH}` | the GPU the result is graded on, and the rules a submission must satisfy | targethw |
| `roofline` | `FK_ROOFLINE_PATH` | `{ROOFLINE_PATH}` | the GPU's measured peaks, and per workload the distance to SOL | roofline |
| `lessons` | `FK_LESSONS_PATH` | `{LESSONS_PATH}` | what was measured before on other operators of the same class | opkb |
| `prior` | `FK_PRIOR_PATH` | `{PRIOR_PATH}` | kernels that already exist for this operator or its family | warmstart, seedlib |

`rsi_kernel_tune.runcontext`:

```python
path_for(name, env=None) -> Path | None     # absolute, ~ expanded; None when not set or blank
paths(env=None) -> dict[str, Path | None]
env_for(context: dict) -> dict[str, str]    # the inverse: {"target": p} -> {"FK_TARGET_PROFILE_PATH": ...}
binding(name, env=None) -> str              # the path, or "(none)"
```

`loop/driver.py::_bindings(ctx)` binds the four placeholders -- the absolute path, or `(none)` when
the run sets none -- so every prompt the driver renders has them: `__main__`'s lookup order puts
no hand-built binding dict in front of it (`tests/test_runcontext.py` reads the driver's source
to hold every `render_prompt` call site to that).

The variables reach the process through the environment or `local.env` (which `forge run` exports
while it drives a run); the environment wins for a key both set.

**The prompts.** Each of the six teacher prompts ends with the same section, which says what the
four files are and to proceed exactly as without them when a path is `(none)`:
`recon_prompt.md`, `distill_prompt.md`, `distill_blackbox_prompt.md`, `diagnose_prompt.md`,
`patch_prompt.md`, `m0_audit_prompt.md`. The student prompt (`student_prompt.md`) ends with a
shorter one naming the target and the existing implementations. The CUDA overlays
(`meta/cuda/<same name>`) carry the same sections, so an overlay still has exactly its shared
prompt's placeholders and `## ` sections (an existing test holds the overlays to that).

The two sections name no operator and no operator structure (invariant L1); they say that the
files are measured or mechanically extracted data for this run, not advice, and that reading,
reusing and improving the kernels the prior file lists is allowed.

---

## 4. The target device profile

`target/B200.json` is the profile of the GPU every submission is graded on: its facts (CC 10.0 /
`sm_100a`, 148 SMs, 192 GB HBM3e at 8 TB/s, clocks locked at 1500 / 3996 MHz, FP8 5 | 10 and dense
NVFP4 10 PFLOPS, SFU 5 TeraExp/s, TMEM 256 KB per SM, the software of the evaluation image), how
the board evaluates and scores, what a submission must be, how the development machine (a B300)
may differ and what to do about each difference, and the rules a kernel must satisfy. What was not
measured when it was written -- the B300's SM count and clock lock, the board's live time limits
and image, whether the evaluation image's stack runs on CC 10.3 -- is marked so and given no
number.

`rsi_kernel_tune.target_profile.render_markdown(profile)` renders it to the Markdown an agent reads,
data-driven: the target and its facts first, then how the board evaluates, what a submission must
be, how the development machine differs, the rules. A key added to the profile appears with no
change to the code. `load_profile(path)` and `render_file(path)` read the repository's file by
default.

This is a **draft**: task targethw completes the profile with what it measures and owns how it
reaches a run (the plan is that `FK_TARGET_PROFILE_PATH` names the rendered Markdown).

---

## 5. Extension point C: operator families

`families.toml` (repository root) declares the families and holds an **ordered** rule table: the
first rule whose `match` (a regular expression) is found in a problem's name decides its family;
a name no rule matches is `other`. Rules see the name in lower case, without the board's `NNN_`
prefix and without a trailing `backward` -- a backward kernel is the same family as its forward,
and `is_backward(name)` says which it is.

```python
from rsi_kernel_tune.family import family_key, is_backward, families
family_key("069_rms_norm")                                            # "rmsnorm"
family_key("060_fused_attention_qk_matmul_scale_mask_softmax_backward")  # "attention"
is_backward("060_fused_attention_qk_matmul_scale_mask_softmax_backward")  # True

python -m rsi_kernel_tune.family 069_rms_norm 001_fp8_attention_output_projection
python -m rsi_kernel_tune.family --counts < leaderboards.json     # a board snapshot, or names
```

The rules were induced from the 235 problem names of the board snapshot of 2026-11-24, where they
give:

| Family | Problems | What is in it |
|---|---|---|
| `attention` | 48 | attention cores and their pieces: score matmul, softmax, masks, P·V, GQA repeat, KV cache writes, paged and ragged decode / prefill |
| `moe` | 30 | routing, dispatch, expert computation, combine, load balancing |
| `gemm` | 29 | dense GEMMs and linear projections at full precision: QKV, out and value projections, LM head, patch embedding |
| `quant_gemm` | 26 | quantized (fp8, nvfp4) dense GEMMs and projections, whatever model part they feed |
| `transformer_block` | 15 | whole transformer layers and residual chains |
| `rope` | 14 | rotary position embeddings: frequencies, cos / sin, application |
| `rmsnorm` | 13 | RMSNorm in every form |
| `ssm` | 13 | Mamba, Mamba2, Hyena, gated delta rule: scans, segsum, discretization |
| `vision_block` | 12 | whole blocks of image, video and diffusion models: VAE, SAM, ConvNeXt, UNet, ResNet |
| `mlp` | 10 | feed-forward blocks: gate-up with SwiGLU / GEGLU, GELU feedforward, gated MLP |
| `other` | 8 | names no rule files (AltUp, a time-decay stabilization, concatenation, fusion and loss kernels) |
| `norm` | 6 | LayerNorm, GroupNorm, adaptive and instance norm, AdaLN modulation |
| `conv` | 6 | conv1d / 2d, depthwise, strided, causal |
| `activation` | 5 | standalone activations and small gated elementwise fusions |

The order of the rules is what decides the names that could go two ways, and §10.3 lists the
calls that were made. `tests/test_family.py` pins the family of every one of the 235 problems: a
rule change that moves a problem is a visible diff. **A rule that files a problem wrongly is a bug
in `families.toml`**: report it (in the task report); do not work around it in code that reads
families.

---

## 6. The two stubs

### 6.1 `benchx`: the one way to evaluate a candidate (`src/rsi_kernel_tune/benchx/api.py`)

```python
bench.bench(card_dir, candidate, workloads=None, mode="dev") -> BenchResult
```

`card_dir` is a frozen leaderboard card; `candidate` a Python file with a top-level `run` or an
official JSON solution; `workloads` names the workloads to run (all when `None`); `mode="dev"` runs
them, `mode="target"` only checks that the candidate compiles for the board's GPU.

- `WorkloadReading`: `name, axes, t_ms, t_b, t_sol, s, correct, error` -- one per workload, every
  time from the card's own clock and every verdict from its own judge.
- `BenchResult`: `correct, workloads, mean_s, fast1, integrity, job, elapsed_s, error,
  cold_start_s, confidence`. `cold_start_s` is process start to the first `run()` returned, all
  compiling included; `COLD_START_BUDGET_S` (100 s; the open-source evaluator's compile timeout
  defaults to 120) and `cold_start_over_budget()` say when it is too long. `confidence` holds the
  `CONF_*` tags, `dev-only` always.
- `FakeBench(script)` runs nothing: `script` is one `BenchResult` (the answer to every call), a
  sequence of them (in turn, the last repeating), a callable `f(call) -> BenchResult`, or nothing
  (a passing result with no workload). `calls` records every `BenchCall`.
- `Bench` is the protocol; anything with a `bench` of that shape is one.

**No evaluation is in the stub**, and the module says so: do not write a second evaluator, wait
for benchx. The reference library's scoring, campaign's choice of a starting point, evolve's
selection and the per-workload dispatch of shapes all go through `Bench.bench`.

### 6.2 `klib`: the reference library's index (`src/rsi_kernel_tune/klib/schema.py`)

```python
validate(entry) -> list[str]               # every reason the entry is off the schema; [] when it is not
load_index(path) -> list[KernelEntry]      # raises SchemaError naming every problem
save_index(entries, path) -> Path          # refuses (nothing written) what load_index would refuse
submittable(entry) -> bool                 # may this kernel be merged into a submission
```

An index is `{"version": 1, "entries": [...]}`. An entry (`KernelEntry`) has `id`, `problem_keys`
(exact problem names and family keys), `family`, `path` (relative to `kernel_library/`, a Python
file), `language` (`triton` / `cutedsl` / `pytorch` / `cuda-inline`), `entry` (always `run`),
`source` (`repo`, `commit`, `path`), `license`, `reported`, `verified_local`, and
`usable_in_submission`.

Two checks are the project's rules and not the schema's taste: `family` must be one
`rsi_kernel_tune.family` knows and be among the entry's `problem_keys` (an entry is found through its
family); and **a license other than Apache-2.0, MIT or BSD may stay in the library only with
`usable_in_submission: false`** -- such an entry can be read and adapted from, and
`submittable()` says it cannot be merged.

Building the index from sources, finding and scoring entries are task seedlib's.

---

## 7. Extension point D: an external GPU lock (`FK_JOB_LOCK_CMD`)

Local delivery can put a lock shared with other work on the machine in front of every job. Set in
`local.env`:

```
FK_JOB_LOCK_CMD=/home/<user>/tools/bin/with-lock gpu 1800 --
```

The value is split like a shell would (`shlex.split`) and ends with `--`; the job then runs
`<lock command> -- <entry command>` inside this project's own GPU lock. It is for `local` delivery
only; over ssh the host's lock is used, as before. A run that sets no lock command behaves exactly
as it did.

- **The entry is started from inside the lock** by a shell that prints the begin marker
  (`FK_JOB_BEGIN`) first and then becomes the entry. So the marker means "the lock was acquired
  and the entry is starting"; a job still waiting has printed none.
- **A job that never got the lock is "not delivered".** When the lock command ends without the
  entry having begun -- it gave up waiting (`with-lock` exits 3), it was not found, it answered
  without starting anything -- the runner writes `TRANSPORT_ERROR_MARKER` into the log, the line
  the ssh transport writes, which the loop already reports as a transport problem
  (`JobTransportError`) and never charges to the kernel. A lock that does not start the entry
  within the job's own time limit is given up on the same way.
- **An entry that failed after it began is the failure it always was.** A card's `gpu_entry.sh`
  exits 3 when the board's clock lock does not hold; through `with-lock`, which passes a command's
  status on, that must not read as a lock that gave up -- the begin marker is what tells them
  apart, and the loop's clock-lock handling sees the refusal as before.
- **The wait for the lock is not taken out of the entry's time.** The entry's deadline begins at
  its begin marker; the wait before it is bounded by the same time limit on its own.

Known limits (§10.5): `with-lock` is not crash-safe, and the loop's message for a transport
failure still speaks of ssh.

---

## 8. The development environment

Python 3.11 in the repository's own virtual environment; the package is not installed (the tests
import it from the worktree they run in, `pythonpath = ["src"]` in `pyproject.toml`):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python pytest torch safetensors \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

Do not `pip install -e .`. To run `forge` from a checkout without installing it:

```bash
PYTHONPATH=src .venv/bin/python -m rsi_kernel_tune --help
```

**Tests** are run in the root of the worktree, with the one environment:

```bash
.venv/bin/python -m pytest -q
```

The suite touches the GPU: `tests/test_task_suite.py` also runs on the `cuda` device when torch
finds one, which it does on the machine this was written on. So that a run of the suite does not
distort someone's measurement, the full suite is run as one job under the machine's GPU lock:

```bash
~/tools/bin/with-lock gpu 1800 -- .venv/bin/python -m pytest -q
```

A single file that does not reach the GPU needs no lock. The baseline of `b2c8ae8` -- its
counts, and anything in them the environment explains -- is in the project's
`reports/baseline.md`, beside the repository. The environment matters for it: the recipe above
brings no `numpy`, and without it three tests fail and one is skipped (that file says which).
What it records as the baseline's own failures are the only failures a task may leave standing.

The development machine is reached over ssh (`FK_DELIVERY=ssh` in `local.env`); its environment
is the project's to align with the evaluation image (the prompt files say how).

---

## 9. What a task that builds on this must keep

- **A new command** is `src/rsi_kernel_tune/ext/<name>.py` with `COMMAND = "<name>"` and
  `main(argv) -> int`; its subcommands are not named after a built-in operator command (§2). It
  has `--help` and exits cleanly from it.
- **A new prompt overlay directory** (`meta/triton/`, say) gets the two sections of §3 at the end
  of each of its overlays, word for word, or the overlay loses the run context files its shared
  prompt has.
- **A context file is data.** It is measured or mechanically extracted; an LLM is not asked to
  write a narrative of what might work into it. A path that does not exist is not an error.
- **Nobody evaluates a candidate a second way.** Through `Bench.bench`, with `FakeBench` in tests.
- **`benchx/api.py`** takes the real `CardBench` and keeps the protocol and `FakeBench` as they
  are; **`klib/schema.py`** may be completed but its public function signatures stay.
- **`families.toml`** is corrected where it is wrong and reported, never worked around.

---

## 10. Implementation record

Trade-offs made in implementation, and where the result departs from the plan.

1. **The command name is the module's name.** The plan has each extension define `COMMAND` and
   the registry collect it. Collecting it means importing the module, and a module that is broken
   is exactly the one whose command cannot be found that way -- nor listed in the usage text
   without risk. So discovery lists the directory only, the name is the file's, and `COMMAND` is
   checked to agree when the command is called. Every extension the first wave plans has the two
   equal. The cost: an extension cannot be renamed independently of its file.
2. **No extension was added.** The registry, the dispatch and the reserved-name check are
   complete, and `src/rsi_kernel_tune/ext/` holds only its `__init__.py`, as the plan asks. The
   second-word rule (§2) is a consequence of the plan's lookup order, not a new restriction.
3. **Context files are bound as absolute paths.** The agents run with the student's workspace as
   their working directory, so a relative path would be read from the wrong place. A fourth
   function, `binding()`, was added to keep `(none)` in one place. The sections were appended at
   the very end of each prompt, after its output rules, as the plan says.
4. **The profile is rendered by a generic renderer.** A hand-written template would have to be
   edited for every key the final profile gains; this one shows any JSON. The draft renders to
   about 155 lines, where the targethw task asks for a page and a half: that is the task's to
   settle, by what it keeps in the profile. The draft also states twelve rules; their final
   wording is the task's.
5. **Families: the calls that were made.** (a) A quantized projection is `quant_gemm` whoever
   eats its output -- its kernel is scaled-matmul and block-scale work that every other one
   shares -- while routing, attention cores, RoPE and SSM steps of the Quantization track stay
   with their operator. (b) A name says its leading operator: `kv_cache_update_with_rope` is
   `attention`, `fused_rope_with_qk_norm_and_kv_cache_update` is `rope`. (c) A projection of
   attention tensors is `gemm`, a score matmul or a softmax `attention`. (d) `fused_add_rmsnorm`
   is `rmsnorm` with the plain ones (nine on the FlashInfer track, not six). (e) Hyena is `ssm`,
   a VAE's attention block is `vision_block`, a VAE's convolution is `conv`. (f) Two names whose
   words mislead every rule are filed by exact name: `patch_embed_to_joint_attention_input` is a
   concatenation, and AltUp's "magnitude normalization" is not a norm layer. (g) `other` holds
   what a name alone cannot place; `064_latent_kv_expansion_with_split` was filed as `attention`
   on the reading that a latent KV is MLA's, which the name alone does not confirm.
6. **`klib.validate` returns a list and does not raise.** A caller that wants an exception gets
   one from `load_index` / `save_index`, which list every problem at once; one that wants to
   collect problems across entries can. `submittable()` answers for anything it is handed, since
   it is a gate.
7. **`FakeBench` takes a script, not keyword readings.** Its interface is frozen once other tasks
   code against it, so it takes the three forms a test of a caller needs (one answer, a sequence,
   an answer computed from the call) rather than one way of describing readings. It refuses a
   `mode` it does not know and records no call for it.
8. **The job lock: the begin marker moved inside the lock.** The plan asks for the prefix and for
   a lock that timed out to be "not run". Writing the marker before the process starts, as the
   runner did, would make "the entry began" unknowable -- and `rc = 3` is not the lock's alone: a
   card's `gpu_entry.sh` exits 3 for a clock lock that does not hold, and `with-lock` passes it
   on. So the marker is printed by the shell that starts the entry, under the lock, and every end
   of the lock command without a marker is "not delivered". Two things the plan does not mention
   were needed for that to be true: the entry's deadline starts at the marker (a long wait would
   otherwise be taken out of the entry's time and its killing charged to the kernel), and a lock
   that never starts the entry is given up on within the time limit.
9. **`with-lock` is not crash-safe.** It holds its lock by creating a file and removes it when
   its command ends. A job killed at its deadline, or stopped, kills the lock command too, and
   the file stays until another `with-lock` takes it over two hours later. A trap in `with-lock`
   would close this; it is the tool's to change, not the repository's.
10. **The loop's message for a transport failure speaks of ssh.** `JobTransportError` says "check
    the ssh alias and the host"; for a lock that was not acquired the detail in the same message
    says what happened, but the first half misleads. The wording is in `loop/cli.py`, which step
    0 had no reason to touch; it is left for the integration.
11. **Measured with the machine's own lock.** One local job was run through `rsi_kernel_tune.submit`
    under the real `with-lock`: the lock was taken, the begin marker printed, the entry ran
    inside the lock, the lock released. The paths a lock contention takes were measured with a
    stand-in lock in the tests, not with the real one: holding the machine's lock to make it
    refuse would have made other work on the machine wait.
12. **The first wave started before this was committed.** The eight task worktrees were created
    from `b2c8ae8`, before the first commit of step 0, so their branches began without any of
    this. What each task made for itself until it picked `bk/base` up is the integration's to
    reconcile; the record of what was found is in the step 0 report, not here.
