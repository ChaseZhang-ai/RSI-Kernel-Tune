# Triton mode (loop language `triton`)

Triton is a first-class language for ForgeKernel's loop: a `[task].language = "triton"` task runs a
student that writes a **single Python `kernel.py` of `@triton.jit` kernels**, reads its own prompt
set, its own hardware knowledge base (`hw_kb/triton/`) and files its provenance under
`docs/kb-provenance/triton/`. This document covers why the mode exists, how it is wired, what the
harness can and cannot read out of a Triton kernel, how the leaderboard path works, how to
cross-compile a kernel for the target **without a GPU**, and how to smoke the whole loop on a small
card.

## 1. Why Triton

- **The track submission format is Python-only.** A collection submission is one directory per
  kernel with a `submission.py` in it; CUDA C++ has to be embedded and built at run time, CuTe DSL
  is Python too — but a Triton solution is already exactly one Python file with a `run` entry.
  Nothing has to be transformed on the way out.
- **It compiles in seconds.** CuTe DSL and CUDA builds cost tens of seconds to minutes per variant;
  a Triton kernel is compiled by the Triton compiler in seconds, so the same wall-clock budget buys
  many more iterations per student round — and "the student iterates too few times" is the loop's
  headline defect.
- **It is good enough to win.** Public results beat hand-tuned expert implementations on inference
  kernels with plain Triton, and the leaderboard's Triton solution language (`spec.languages ==
  ["triton"]`) is accepted with `dependencies: ["triton >= 2.3"]`.
- **It works on the development card.** A Triton kernel is compiled for the device it runs on, so a
  kernel that names no architecture runs on an RTX 3060 (sm_86), a B300 (sm_103) and the target
  B200 (sm_100) alike. That is what makes an end-to-end smoke possible on cheap hardware.

## 2. How the mode is wired

| what | where | Triton value |
|---|---|---|
| task language | `task.py` `LANGUAGES` / `LOOP_LANGUAGE` | `"triton"` → loop language `"triton"` |
| mode | `loop/lang_mode.py` `mode_of` / `TRITON_LANGUAGES` / `is_triton` | `MODE_TRITON` |
| KB root | `lang_mode.KB_ROOT` | `hw_kb/triton` |
| provenance | `lang_mode.PROVENANCE_SUBDIR` | `docs/kb-provenance/triton` |
| prompt overlay | `lang_mode.PROMPT_OVERLAY_DIR` | `meta/triton/<name>` (falls back to `meta/<name>`) |
| kernel file | `taskcard.KERNEL_FILE` | `kernel.py` (the file-suffix guard keeps `triton` in the `.py` camp) |
| off-line API source | `loop/dsl_source.py` `resolve(language="triton")` | the installed `triton` package's roots |
| job env preflight | `loop/job_env.py` `import_check_flag` | `--triton` (the job's `python3` must `import triton`) |
| self-evolution | `scripts/self_evolve.py` | a Triton kernel can seed the next generation (it is Python with a `run`) |
| target-compile check | `src/forgekernel/triton_check.py` | `triton.compile` + an explicit `GPUTarget`, no launch |

A task validates `hw_kb/triton/<arch>/` the same way the other languages validate theirs: the tree
must exist with an `index.md`, and a run refuses to start (exit 14) when it does not.

### The prompt set

Seven overlays live in `meta/triton/`: `student_prompt.md`, `recon_prompt.md`, `distill_prompt.md`,
`distill_blackbox_prompt.md`, `patch_prompt.md`, `kb_recipe_prompt.md`, `kb_harvest_prompt.md`.
Each is a copy of the shared prompt with **only** the CuTe-DSL-specific passages replaced (the
language constraints, the layout-probe examples, the crash-triage recipe, the two rungs of the
expressibility ladder, the KB and provenance paths). Bindings (`{LANGUAGE}`, `{KB_ARCH}`, …) and
section headings are identical to the shared prompt, so the driver renders and parses them the same
way; `tests/test_triton_mode.py` enforces that parity. Prompts that name no language
(`diagnose_prompt.md`, `m0_audit_prompt.md`, `verdict_probe_brief*.md`) have no overlay.

## 3. The knowledge base (`hw_kb/triton/`)

Four trees: `sm100/` (the target, the full set), `sm90/`, `sm86/` and `sm103/` (the two development
GPUs' trees; `sm86` is explicitly a **development / smoke** target). This release starts the tree
the way the CUDA tree started: from **methodology entries**. Every entry opens with a
`Certification source` line:

- `✅ Verified` — a GPU probe of this repository certified the claim, and the line names the probe;
- `[PENDING-CERT]` — written from public Triton documentation / release notes and a public
  MIT-licensed Triton guide (AKO4X), **not** yet earned by a probe here. Treat the numbers as a
  starting point to verify, never as a reading.

`docs/kb-provenance/triton/` records where each entry's borrowed material came from (it is not in
any agent's read set). Certifying an entry means walking the ladder in
`hw_kb/triton/sm100/gpu_probe_ladder.md` and replacing the line. The tree is fill-on-demand: each
`index.md` keeps its own to-be-filled list.

Every runnable example in the tree was compiled for CC 10.0 (`python -m forgekernel.triton_check …`,
no launch) before it was shipped; `docs/kb-provenance/triton/README.md` says what that validated.

## 4. What the harness can read out of a Triton kernel

The full table (a tool per row, the three languages across):

| tool | cutedsl | cuda (`cuda-sol`) | triton | in Triton mode |
|---|---|---|---|---|
| **NCU profiling** (`ncu`, the digest, PC sampling) | yes | yes | **yes** | the kernel name is the `@triton.jit` function's (mangled: argument types + constexpr specializations) |
| **stall pricing** (`stall_price_run`, PC-sampling ledger) | yes | yes | **yes** | language-agnostic: it reads an `.ncu-rep` |
| **SASS text + `sass_tools` / `sched_live_tools` / `struct_tools` / `orch_tools`** | yes | yes | **yes** | `ncu --import <rep> --page source --print-source sass` on the profiled kernel; the parsers do not care who compiled the cubin. It is the run's own probe, not an automatic step |
| **lever pricing** (`lever_pricing_run`: source-text ablations + the card's bench) | yes | yes (nvcc prebuild) | **yes** | text edits on `kernel.py`, then the card's own bench — no build step |
| **job-env preflight** (`job_env_check`) | torch | torch + `--cuda` toolchain | **torch + `--triton`** | asserts the job's `python3` imports `triton` |
| **timeline probe** (`_seg` / `FK_SEG` markers + device clock) | yes | yes | **disabled, explicitly** | no marker-stamping channel for `@triton.jit`: the driver logs `tool timeline: unavailable in triton mode — …` once, `_timeline_on` is false whatever `loop.toml` says, and the student's prompt says the probe is unavailable instead of inviting a round for it |
| **orchestration census** (per-iteration SASS digest) | yes (`CUTE_DSL_KEEP_CUBIN`) | yes (`--kernel kernel.cu`) | **disabled, explicitly** | it reads the cubin the tree's build dumps; a Triton kernel's cubin goes to the Triton cache. The driver logs `tool census: unavailable …` once and profiles with plain NCU |
| **`FK_CUDA_LINEINFO` / `// fk-cuda-cflags:`** | n/a | yes | n/a | Triton's own knobs (`TRITON_DISABLE_LINE_INFO`, a pinned `TRITON_CACHE_DIR`) replace them |

Nothing is silently wrong: a disabled tool says so in the run's log and in the student's prompt, and
a milestone written against a census axis it cannot produce is refused by the overlays' own
instruction (the meta agent is told not to anchor a gate on a `[sass-derived]` axis in this mode).

The readable artefact a Triton student *does* get is the compiler's own output: with
`TRITON_ALWAYS_COMPILE=1` and a pinned `TRITON_CACHE_DIR`, the job keeps `*.ptx` / `*.ttgir` /
`*.cubin` next to the kernel, which answers "which MMA did it pick / is the accumulator in TMEM".

## 5. The leaderboard path (`forge init --leaderboard … --language triton`)

```bash
forge init myproj --leaderboard sol-execbench --dataset <tree> --definition L1/001_rms \
    --workload 0 --language triton            # a Triton card: kernel.py, loop.toml.example = triton
# then edit tasks/myproj/task.toml -> [task] language = "triton" (init writes cutedsl for a .py card)
forge new tasks/myproj && forge run .artifacts/myproj_<id>
```

- The card's stub is `kernel.py` with the definition's signature (`def run(hidden_states, weight,
  output)` for a DPS card); `forge <op> leaderboard show` checks the signature, and `export` writes
  `"languages": ["triton"]`, `"dependencies": ["triton >= 2.3"]`,
  `"entry_point": "kernel.py::run"`.
- `--language` is validated against the kernel file by `lang_mode.language_problem`: a
  CUDA-mode language over a `kernel.py` is refused at scaffold time.
- A leaderboard **run** takes the task's language rather than the card's suggestion
  (`taskcard.write_board_loop_toml` sets `language` from `task.loop_language`); without that, a card
  scaffolded before the task named its language would silently run a Triton task in the DSL mode.
- `forge init` itself has no `--language`: `launch.new_board_task` writes `[task] language = "cuda"`
  for a `.cu` card and `"cutedsl"` otherwise, and the printed message tells the user to adjust
  `task.toml`. For a Triton card, change that one line (an integration remark in the task report).

## 6. Target-compile check (provisional)

`src/forgekernel/triton_check.py` answers "does this kernel lower for the target?" without a GPU:

```bash
python -m forgekernel.triton_check kernel.py --target 10.0
# -> TRITON-COMPILE OK|PARTIAL|FAIL|UNAVAILABLE target=10.0 kernels=N compiled=N unverified=N
```

- It always reads the **static contract**: the file parses, has a top-level `run` and at least one
  `@triton.jit` kernel, names no development-GPU architecture (`sm_103` / `compute_*` /
  `TRITON_OVERRIDE_ARCH`; comments are ignored, code is not), and warns about `@triton.autotune`
  (the board's ten warmups and its L2 flush make in-run tuning the wrong measurement).
- Then it hands each kernel to `triton.compile` with an explicit
  `GPUTarget("cuda", <arch>, 32)` and never launches anything. This needs the target's `ptxas`
  (the Triton wheel carries one) but no device: it runs on a CPU-only box.
- What it cannot know: a kernel's **launch signature** (pointer dtypes, integer widths, constexpr
  values) lives in `run`, not in the kernel's definition. Without `--signature` / `--constexprs` it
  uses placeholder types and reports a kernel it cannot read as *unverified* (never as failed). For
  a faithful reading name the kernel and its launch signature:
  `--kernel _scale --signature '{"x_ptr": "*bf16", "n": "i32"}' --constexprs '{"BLOCK": 1024}'`.
  A TMA descriptor argument needs its own type spelling: `{"desc": "tensordesc<fp32[64,64]>"}`.
- Exit codes: `0` passed (incl. partial), `1` failed, `2` unavailable (no usable Triton here).
- **Provisional**: task targethw's `compile-check` owns the general target-compilation gate (CuTe
  DSL and CUDA included) and supersedes this module; until it lands, this is what an export can
  call.

## 7. fbtriton 3.7.1 — what a student may and may not rely on

The evaluation image installs `fbtriton==3.7.1` (a Triton distribution; the import name is still
`triton`), not upstream Triton. Anything below "works here" was verified only against upstream
Triton 3.9 on the development machine, so **measure on the image before building a round on it**:

```bash
python3 - <<'PY'
import triton, triton.language as tl
print("version", triton.__version__)
for name in ("make_tensor_descriptor", "dot_scaled", "inline_asm_elementwise", "assume",
             "range", "join", "permute", "reshape", "atomic_add", "device_print"):
    print(f"tl.{name}", hasattr(tl, name))
import importlib.util as u
for mod in ("triton.tools.tensor_descriptor", "triton.experimental.gluon",
            "triton.language.extra.cuda"):
    print(mod, u.find_spec(mod) is not None)
import triton.backends.compiler as bc, triton.compiler.compiler as cc
print("GPUTarget", hasattr(bc, "GPUTarget"), "ASTSource", hasattr(cc, "ASTSource"))
PY
```

| surface | status in this release | why |
|---|---|---|
| `@triton.jit`, `tl.program_id`, `tl.arange`, `tl.load/tl.store` (mask/`other`/`cache_modifier`/`eviction_policy`), `tl.sum`, `tl.dot` (+`acc=`, `out_dtype`, `input_precision`), `tl.join`/`tl.permute`/`tl.reshape`, `tl.atomic_add` (`sem`/`scope`), `num_warps`/`num_stages`/`num_ctas`, `triton.cdiv` | **use** | the stable surface every release has carried; all of it is exercised by the shipped KB examples and compiled for CC 10.0 |
| `tl.make_tensor_descriptor` / `triton.tools.tensor_descriptor.TensorDescriptor` | **probe first** | the host/kernel split and the type spelling have moved between releases; `tensordesc<dtype[BM,BN]>` is the current argument type |
| `tl.range(..., warp_specialize=True)` | **probe first** (compile-only) | restricted to pure MMA-accumulator loops; a softmax interlude aborts in `ttgir` |
| `tl.dot_scaled`, `tl.inline_asm_elementwise`, `tl.assume`, `tl.device_print` | **probe first** | present in recent upstream releases; presence in the fork is a measurement |
| `triton.language.extra.cuda` PDL intrinsics (`gdc_launch_dependents` / `gdc_wait`, `launch_pdl=True`) | **probe first** | version- and backend-gated |
| `triton.experimental.gluon` | **do not use** | an experimental surface; a fork may not carry it, and a kernel written against it is not a Triton solution on the board |
| `triton.compile` / `ASTSource` / `GPUTarget` internals beyond an explicit target + options | **do not use in a kernel** | a student kernel must not depend on the compiler's Python internals; the compile check is harness-side only |
| `@triton.autotune`, in-kernel timing of candidate configs | **do not use** | the board warms up ten calls and flushes L2 before every timed call; ship a shape-keyed table |
| `torch.compile`, `torch.matmul` (or any torch compute op) *as the kernel* | **forbidden** | the operator tree's fence: the compute must be `@triton.jit` code |
| reading/writing the Triton cache, `os.environ` juggling, subprocess | **forbidden** | the sealed-reference fence (`SEALED_FENCE` in the card's `eval/leaderboard.py`) |

The target picture gets one line from this section, for task targethw to fold in: *"Triton solutions
run on `fbtriton==3.7.1`; the API surface a kernel may rely on is the stable `tl.*` core, and any
newer lever (`make_tensor_descriptor`, `warp_specialize`, `dot_scaled`, PDL) must be probed on the
image before a round is built on it."*

## 8. Smoking the whole loop on a small card (no agent)

The goal: generate an operator tree, put a **hand-written** Triton kernel where the student's
submission goes, and run the card's own correctness + timing job. Nothing here calls an agent or the
board.

Prerequisite: a `python3` with a CUDA build of torch **and** `triton`. On the development machine:

```bash
/home/zcz/bestKernel/ForgeKernel/.venv/bin/python -c \
  'import torch, triton; print(torch.__version__, torch.cuda.is_available(), triton.__version__)'
# 2.15.0+cu132 True 3.9.0          (an RTX 3060 Laptop, 6 GB, is enough)
```

1. **The task.** `tasks/triton_rmsnorm/` is the Triton twin of `tasks/rmsnorm/`. Its `arch` is the
   *target* (`sm100`); the card's clock and correctness do not depend on it, so a local smoke needs
   no edit. If you want the KB tree a local run reads to be the local one, set `arch = "sm86"`.
2. **The card** (an operator tree, no agent):

   ```python
   # write_check.py — the judge the card times against (no agent): the task's own semantics.
   import json, sys
   from pathlib import Path
   from forgekernel import judge as judge_mod, taskcard
   from forgekernel.localenv import load_machine
   from forgekernel.runs import new_run
   from forgekernel.task import load_task
   src = load_task(Path("tasks/triton_rmsnorm"))
   run = new_run(src, machine=load_machine(Path("."), environ={}))
   d = run.dir / judge_mod.CHECK_DIR
   d.mkdir(parents=True)
   (d / judge_mod.DEFINITION).write_text(json.dumps({
       "inputs": {"x": {"shape": ["m", "n"], "dtype": "bfloat16"},
                  "w": {"shape": ["n"], "dtype": "bfloat16"}},
       "outputs": {"y": {"shape": ["m", "n"], "dtype": "bfloat16"}}}))
   (d / judge_mod.JUDGE).write_text(
       "import torch\n\n"
       "def run(x, w):\n"
       "    xf = x.double()\n"
       "    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.double()\n"
       "    return y.to(torch.bfloat16)\n\n"
       "def make_inputs(axes, group, seed, device):\n"
       "    g = torch.Generator().manual_seed(seed)\n"
       "    x = torch.randn(axes['m'], axes['n'], generator=g).to(device)\n"
       "    return {'x': x.to(torch.bfloat16), 'w': torch.ones(axes['n'], device=device).to(torch.bfloat16)}\n")
   groups = {s.name: [{"name": "plain", "intent": "randn", "seed": 1},
                      {"name": "spiky", "intent": "outliers", "seed": 2},
                      {"name": "tiny", "intent": "near zero", "seed": 3}] for s in src.shapes}
   (d / judge_mod.GROUPS).write_text(json.dumps(groups))
   task = load_task(run.task_dir, name=run.task_name)
   check = judge_mod.load_check(d)
   taskcard.write_card(run, task, check, load_machine(Path("."), environ={}))
   print(run.operators_dir / "triton_rmsnorm")
   ```
   `python write_check.py` prints the card's path. (The run's judge is an agent step in a real
   launch; this writes the same files by hand so the smoke involves no model.)
3. **The student's submission** — a hand-written Triton kernel, dropped into the card's `kernel.py`:

   ```python
   import torch
   import triton
   import triton.language as tl

   @triton.jit
   def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n, EPS: tl.constexpr, BLOCK: tl.constexpr):
       row = tl.program_id(0)
       cols = tl.arange(0, BLOCK)
       mask = cols < n
       x = tl.load(x_ptr + row * n + cols, mask=mask, other=0.0).to(tl.float32)
       w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
       rms = tl.rsqrt(tl.sum(x * x, axis=0) / n + EPS)
       tl.store(out_ptr + row * n + cols, (x * rms * w).to(tl.bfloat16), mask=mask)

   _KNOBS = {"prefill": (4, 4096), "decode": (4, 4096)}

   def run(x, w):
       y = torch.empty_like(x)
       m, n = x.shape
       warps, block = _KNOBS["prefill" if m >= 1024 else "decode"]
       _rmsnorm_kernel[(m,)](x, w, y, n, EPS=1e-6, BLOCK=block, num_warps=warps, num_stages=1)
       return y
   ```
   Before running anything expensive, the target-compile check reads it without a GPU:
   `python -m forgekernel.triton_check kernel.py --target 10.0 --kernel _rmsnorm_kernel
   --signature '{"x_ptr": "*bf16", "w_ptr": "*bf16", "out_ptr": "*bf16", "n": "i32"}'
   --constexprs '{"EPS": 1e-6, "BLOCK": 4096}'`.
4. **The card's own job** — correctness + timing, on the local GPU (the GPU lock is this machine's
   convention for real measurements):

   ```bash
   cd <card>
   PATH=/home/zcz/bestKernel/ForgeKernel/.venv/bin:$PATH \
     ~/tools/bin/with-lock gpu 1800 -- bash gpu_entry.sh perf
   ```
   `perf` (alias `all`) runs `eval/bench.py --mode all`: every input group of every shape is judged
   (correctness must pass) and timed on the card's clock; the summary line carries
   `device_median_ms` and the per-workload readings. `bash gpu_entry.sh correctness` runs only the
   judge; `bash gpu_entry.sh help` lists the other tasks.
5. What the smoke proves: the task loads on its Triton tree, the card was generated for a Triton
   kernel, the hand-written kernel is judged and timed by the card itself, and the compile check
   says it lowers for CC 10.0. What it does not prove: any number about the target (see below).

The optional acceptance item in the task report is run exactly this way; its commands and output are
in the report.

## 9. Self-evolution

`scripts/self_evolve.py` chains generations by sealing each run's best kernel as the next task's
reference. A Triton kernel is a Python file with a `run` entry, so it can be sealed exactly like a
CuTe DSL one; `next_generation` accepts `language in ("cutedsl", "triton")` and still refuses a
CUDA task (its kernel is not something the reference loader seals as a Python module).

## 10. Limitations

- **No timeline probe.** There is no `_seg`-style marker channel for `@triton.jit`; the driver says
  so and never schedules the probe. A student's diagnosis has NCU, the SASS text and the compiled
  artefacts instead.
- **No automatic per-iteration census.** The SASS contract axes (`cache_ops`, `commit`, `supply`,
  NamedBarrier verdicts) are not re-read by the driver in this mode, so a milestone gate must not
  anchor on a `[sass-derived]` axis. The recon's own SASS transcription is still available and
  belongs in the report as evidence.
- **CuTe-DSL-specific course content does not apply.** The TMEM allocator handles, the warpgroup
  register zones, `make_tiled_copy` layouts and the `ptx_escape` ladder are DSL mechanisms; a Triton
  milestone is written in terms of tile shapes, launch knobs, `tl.dot` dtypes/accumulation and the
  memory-access form.
- **The dev GPU is not the target.** A kernel developed on the B300 or smoked on the 3060 has to be
  read on the target before its numbers mean anything; the softmax-heavy and dense-low-precision
  families are the ones expected to read optimistically on the B300.
- **`forge init --leaderboard` cannot say "triton" yet.** It writes the task's language from the
  kernel file's suffix (`.cu` → cuda, else cutedsl); a Triton card needs that one line edited in
  `task.toml` (or `--language triton` on `forge init`, which sets the card's suggested `loop.toml`).
- **The target-compile check is provisional and placeholder-typed by default**; a faithful reading
  needs the kernel's launch signature supplied. Its dynamic half is what task targethw supersedes.
