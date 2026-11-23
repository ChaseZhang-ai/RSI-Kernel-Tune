# Target hardware: develop on one Blackwell, be graded on another

The leaderboard grades kernels on an **NVIDIA B200** (compute capability 10.0, `sm_100a`).
This campaign develops them on a **B300** (10.3, `sm_103a`). The two are not binary compatible
in either direction, and they are not the same machine in ways that change which kernel is
fastest. This document is the reference for everything that follows from that: the facts, the
rules that came out of them, and the tooling that keeps a development result from being read
as a board result.

- `forge target probe` measures the machine you are on → `target/dev.json` (not committed)
- `forge target check` grades the differences → blocker / risk / info, with an action each
- `forge target compile-check PATH` builds a submission for the target without a target GPU
- `forge target show` prints the profile the agents read (`target/B200.json` rendered)

The profile is injected into every run: `target_profile.materialize(root)` writes the
Markdown and `runcontext.env_for({"target": path})` puts it in the environment, where
`{TARGET_PROFILE_PATH}` in a prompt receives its **text** (the agents run on another machine,
so a path would not resolve there).

---

## 1. The two GPUs

| | B200 (graded) | B300 (development) | Why it matters |
|---|---|---|---|
| Compute capability | 10.0, `sm_100a` | 10.3, `sm_103a` | `sm_100a` cubins do not run on 10.3 and `sm_103a` cubins do not run on 10.0. `sm_100f` is the family-conditional target that runs on both |
| SMs | 148 | "up to 160, model dependent" — measure it | persistent kernels, grid sizes and wave counts tuned on one do not transfer |
| Memory | 192 GiB HBM3e, 8 TB/s | 288 GiB HBM3e, 8 TB/s | bandwidth the same; a workspace that fits here may not fit there |
| Dense FP8 / FP4 | 5 / 10 PFLOPS | same / **15 PFLOPS** | FP4-bound kernels read **1.5x optimistic** on the development machine |
| SFU (`EX2`) | 5 TeraExp/s | **10.7 TeraExp/s** | softmax-heavy kernels read ~**2x optimistic** |
| Clocks while timing | locked: SM 1500 MHz, memory 3996 MHz | whatever it is; lock it yourself | an unlocked clock changes the compute:bandwidth balance |
| TMEM | 256 KB per SM | the same | — |
| L2 | flushed before every timed call | the same protocol | no cache residency across calls |

Sources: SOL-ExecBench's board card and Dockerfile (the stack, the clock lock), NVIDIA's
*Inside NVIDIA Blackwell Ultra* (memory, dense throughput, SFU, FP4 ratio) and *Blackwell and
CUDA 12.9 family-specific architecture features* (`sm_100f`, the `a`/`f` suffixes).

Board stack, for `forge target check` to compare against (SOL-ExecBench `docker/Dockerfile`):
CUDA 13.1.1, Python 3.12, `torch==2.9.0`, `fbtriton==3.7.1`, `nvidia-cutlass-dsl==4.4.2`,
`cupti-python>=13.0.1`, CUTLASS headers under `/usr/local/cutlass`, driver ≥ 580.

`sources: target/B200.json` — every number above is there, with where it came from.

---

## 2. The rules, and why each one exists

These are the twelve rules the profile shows an agent, grouped by the fact behind them.

1. **Graded on B200: CC 10.0, `sm_100a`, 148 SMs, 8 TB/s, locked clocks, cold L2.** → Write for
   that machine, not for the one you are sitting at.
2. **No CC 10.3 feature; never name an architecture — derive it from the device.** →
   `sm_103a` cubins do not load on the board. A literal architecture in the code is either
   wrong on the board or wrong here; `torch.cuda.get_device_capability()` is right on both.
3. **Never hard-code the SM count; query it (`FK_TARGET_SMS` overrides when set).** → 148 vs
   the development count differ; `FK_TARGET_SMS=148` lets a grid be sized for the target,
   and because the variable does not exist on the board, the code must fall back to the query.
4. **Dense FP4 and SFU throughput are higher here.** → a local reading of an FP4-bound or
   `exp`-heavy kernel is a lower bound; report it as one.
5. **One `submission.py`, top-level `run(*args)`, positional, outputs last.** → the track
   submission format (only `.py`; a separate `.cu` cannot be submitted).
6. **Allowed libraries only; CUDA C++ inline.** → FlashInfer / FlashAttention / FlashMLA /
   DeepGEMM may not be imported; embed the source if you need it.
7. **Compilation happens on every evaluation: keep it well inside a minute.** → the board's
   compile timeout is 120 s and every solution is compiled from scratch in its own process.
8. **Leave global state as you found it.** → `torch.backends` flags are process-wide; the
   timing and the reference share the process.
9. **Cold L2, inputs cloned to a new address.** → caches and pointer identity do not survive a
   call; minimise bytes moved.
10. **The stopwatch spans the first to the last kernel of one call.** → host-side gaps count;
    keep the Python between launches minimal.
11. **Outputs are pre-allocated (DPS).** → allocating a result and copying costs a kernel.
12. **Every evaluation is a cold start.** → prefer per-shape constants over run-time
    autotuning; autotuning time is charged to the cold start.

### SM-count simulation (`FK_TARGET_SMS`)

A kernel may read `FK_TARGET_SMS` and size its grid for the target's SM count. Documented
demo shapes:

```python
# Python side
n_sms = int(os.environ.get("FK_TARGET_SMS") or 0) or \
    torch.cuda.get_device_properties(0).multi_processor_count
```

```cpp
// CUDA side
int n_sms = 0;
const char* env = std::getenv("FK_TARGET_SMS");
if (env && *env) n_sms = std::atoi(env);
if (n_sms <= 0) cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, 0);
```

It is an approximation: it changes the grid, not what one SM does. A kernel whose per-SM
throughput is bound by something else (FP4 pipes, SFU, DRAM) reads exactly as fast with 148
or 160 SMs; use it to check that a persistent kernel still covers the machine, not to predict
a score.

---

## 3. What the card compiles for (`dev_arch`)

`task.toml` has two architectures:

```toml
[task]
language = "cutedsl"
arch     = "sm100"     # graded: picks hw_kb/<lang>/sm100 and the target profile
dev_arch = "sm103"     # development: what the card compiles and runs   (optional)
```

- `arch` — the graded GPU. Chooses the hardware knowledge base. Unchanged.
- `dev_arch` — the GPU the card runs on. Optional; defaults to `arch`. **No knowledge base is
  required for it** (a development architecture may be one this repository has never seen).

What follows `dev_arch`:

| place | without `dev_arch` | with `dev_arch = "sm103"` |
|---|---|---|
| `gpu_entry.sh` `CUTE_DSL_ARCH` | `sm_100a` | `sm_103a` |
| `gpu_entry.sh` `FK_CUDA_ARCH` | absent | `sm_103a` |
| `submit.toml` `[env]` | `CUTE_DSL_ARCH = "sm_100a"` | + `FK_CUDA_ARCH = "sm_103a"` |
| the card's CUDA build (`eval/leaderboard.py device_gencode`) | `sm_100a` | `sm_103a` |
| the recon tree's reference build (`binref`) | `sm_100a` | `sm_103a` |
| `loop.toml` `kb_arch` | `sm100` | `sm100` (the graded one) |
| `forge <op> leaderboard export` | the board's `sm_100a` | **also** the board's `sm_100a` |

The last row is the important one: **a development machine's architecture never reaches an
exported solution.** Export passes the board's gencode explicitly (`CUDA_GENCODE`), and a
kernel file that names its own `-gencode` / `-arch` keeps it (the official rule).

With no `dev_arch`, every one of those outputs is byte-for-byte what it was before this
feature existed (the `FK_CUDA_ARCH` lines are simply not written).

**A submission never names an architecture.** `submission.py` derives its gencode from
`torch.cuda.get_device_capability()` inside `run()` / at build time; the rules in §5 refuse a
literal.

---

## 4. The probe

`forge target probe` runs one program — `targethw/probe_program.py`, a real module, not text
built at run time — on the machine being described:

- GPU: `nvidia-smi --query-gpu` (name, CC, memory, driver, current / application / max clocks)
  and `nvidia-smi -q` (clocks, applications clocks, max clocks, throttle reasons). The clock
  lock is judged from three numbers: locked when an application clock is set, the current clock
  matches it within 50 MHz, and it is below the maximum. Three numbers that do not decide it
  are reported as *unknown*, not guessed.
- Stack: python, `torch` (+ its CUDA), Triton and **which distribution** it is (`fbtriton` or
  stock), `nvidia-cutlass-dsl`, CUPTI, `nvcc` (`--version`), and the CUTLASS tree if one is
  found.
- Three smokes, each reporting compile / run / reading: a masked vector add (`@triton.jit`), a
  1-D scale-and-store (`@cute.kernel` + `@cute.jit`), and a TF32 `mma.sync` built with
  `torch.utils.cpp_extension.load_inline` — the last one derives its `-gencode` from
  `torch.cuda.get_device_capability()`, so it is also a working example of rule 2.

How it runs:

- **local**: `probe_program.collect()` is called in this process, under the same per-GPU lock a
  GPU job takes;
- **`--remote`**: the same file is written into a one-task job tree (git-initialised) and run
  through `submit.direct.DirectRunner` — the delivery `forge run` uses: `git archive` over ssh
  to `REMOTE_WORKDIR`, the host's GPU lock, the job timeout and session handling, a log. The
  probe is a task in that machinery, not a second way to reach the host.

Limits worth knowing:

- the three smokes run **in process** (no generated program is executed): a smoke that
  segfaults takes the probe with it. `--no-smokes` covers the machine-only case.
- the local probe cannot enforce `--timeout` (there is no subprocess to kill); the remote one
  can, through the job's timeout.
- the probe does **not** source `REMOTE_ENV_FILE`: sourcing is shell work, and nothing here
  builds a shell command. If the board environment is only enabled by that file, run the probe
  on that host (`forge target probe`) or make the interpreter reachable as `python3`.

`target/dev.json` is not committed (`.gitignore`), for the same reason a measurement of one
machine is not a fact about another.

---

## 5. The static rules (`targethw/rules.py`)

`check_submission(source, language)` returns `Finding(rule, severity, line, snippet, detail)`.

| rule | severity | fires on | does **not** fire on |
|---|---|---|---|
| `arch-literal` | block | an architecture literal in code position: `sm_100a`, `sm_103a`, `compute_100`, `sm_100f`, … | the same literal in a comment or a docstring; a CUDA `//` comment inside an embedded source; a gencode built from `get_device_capability()` |
| `b300-only` | block | `__CUDA_ARCH_FEAT_SM103*`, `compute_103a`, `sm_103a`, or behaviour keyed to `== (10, 3)` / `major == 10 and minor == 3` | a capability *ladder* (`{(9,0), (10,0), (10,3)}`), which is the correct pattern |
| `hardcoded-sms` | warn | `148` / `160` on a line that also talks about a grid, blocks, a wave or `num_sms` | the same number in a tolerance, an index, a shape |
| `fk-target-sms-no-fallback` | block | `os.environ["FK_TARGET_SMS"]` with no fallback anywhere in the surrounding lines | `.get(...)`, `getenv(..., default)`, `in os.environ`, an `else` branch, an `except` |
| `gencode-dev-only` | block | arch flags naming only `103a` | gencode derived from the device; several gencodes; a source that names none (the board appends its own) |
| `fp4-bound` | info | an FP4 MMA kind (`mxf4`, `e2m1`, `block_scale`, …) | anything else |
| `sfu-bound` | info | `exp` / `ex2` inside a loop, or three or more sites | one `exp` in a straight-line kernel |

Three normal styles must produce zero findings, and do (tests):
a device-derived gencode with an `FK_TARGET_SMS` fallback, a `@triton.jit` kernel, a CuTe DSL
kernel.

The `info` findings are also the reading-confidence labels: `target_profile.confidence_hints()`
returns `["dev-only", "fp4-bound", "sfu-bound"]` for a source that has them, which `benchx`
puts in `BenchResult.confidence`.

---

## 6. The target compile check

`forge target compile-check <submission.py | task dir>` answers "would this build on the
board?" without a board:

- **cuda-inline**: every string in the submission that holds `__global__` is extracted and
  compiled with the board's flags — `nvcc -c -gencode arch=compute_100a,code=sm_100a -O3
  --use_fast_math -std=c++20`, torch's `-D` flags, torch's include paths (plus the
  interpreter's, because `torch/extension.h` includes `Python.h`) and CUTLASS's if the source
  needs them. It is **not run**: no B200 is needed to know whether it builds for `sm_100a`.
- **triton**: each `@triton.jit` kernel is extracted (with its decorator), imported from a
  temporary module, given an inferred signature (pointers `*fp32`, integers `i32`, constexpr
  parameters a placeholder) and cross-compiled with
  `triton.compile(ASTSource(...), target=GPUTarget("cuda", 100, 32))`. A cubin for CC 10.0 is
  `ok`; a failure that names the target (`ptxas`, `compute_100`, `sm_100`, unsupported) is
  `fail`; anything else is `unknown` — it is a statement about our guessed call, not about the
  target.
- **cutedsl**: a CuTe DSL kernel compiles by loading it on a device of the target
  architecture, and this machine has none. The check reports `unknown` with that reason, plus
  whether `nvidia-cutlass-dsl` imports at all; the source is still checked statically.
- **static**: the rules above; a block finding is a failure.

Blind spots, stated rather than hidden:

- compiled with `-std=c++20` and this machine's torch headers; the board's torch 2.9 may use
  `c++17`, so C++20-only syntax in a submission would pass here and fail there.
- the Triton check uses the installed Triton (3.9 here), not the board's `fbtriton` 3.7.1: a
  construct that exists in one and not the other is missed. `forge target check` reports the
  distribution difference as a risk.
- a Triton kernel that depends on module-level names cannot be imported on its own →
  `unknown` ("checked by the card's own compile instead"), not a failure.
- CuTe DSL is not compiled for the target at all (see above).
- the check compiles, it does not link: a missing `-l` flag or a driver-API symbol shows up on
  the board, not here.

### Why the flags are the board's

`export_solution` writes `compile_options` with the board's gencode (`CUDA_GENCODE`) because
the exported solution is built on the board. The *card* uses the development architecture
(§3). The two must not be conflated: a card that builds `sm_103a` locally is right, and a
solution JSON that says `sm_103a` is a zero.

---

## 7. Reading confidence

A local reading is always `dev-only`. On top of that:

| label | where it comes from | what it means |
|---|---|---|
| `dev-only` | every reading | taken on the development machine; the board may differ |
| `clocks-unlocked` | `probe.confidence_hints()` | the clocks drift between runs; lock them (`nvidia-smi -lgc`/`-lmc`) before comparing kernels |
| `fp4-bound` | rule `fp4-bound` | dense FP4 is 1.5x higher here: the local number is a lower bound |
| `sfu-bound` | rule `sfu-bound` | SFU is ~2x higher here: the local number is a lower bound |
| `sm-count-sensitive` | not derived yet | a kernel whose grid is sized by SM count can be *slower* on the target; `FK_TARGET_SMS=148` is the local stand-in |

The campaign record keeps "local estimate / board measurement" per problem for exactly this
reason: the ratio is the calibration, and it is not 1.

---

## 8. `forge target check`: the three levels

- **blocker** — the board's stack cannot compile or run on this machine: no CUDA device, an
  older compute capability, a driver below 580, a missing torch / Triton / CuTe DSL, a smoke
  that fails, or no GPU at all. Nothing measured here is worth anything until it is fixed.
- **risk** — the stack works, but a reading would be optimistic or not transfer: a different SM
  count, memory, FP4 / SFU throughput, unlocked or different clocks, other torch / Triton /
  CUDA versions, a stock Triton instead of `fbtriton`, no CUPTI.
- **info** — expected: the compute capability differs (that is why `dev_arch` exists and a
  submission derives its arch), the python version, smokes that were skipped with
  `--no-smokes`.

`forge target check` exits 2 when anything blocks, 0 otherwise, so a script can gate on it.

---

## 9. What this task does not do

- It does not measure the board. Everything about B200 comes from the board's published card,
  the Dockerfile and NVIDIA's documents; `target/B200.json` records which.
- It does not make the board's `fbtriton` 3.7.1 available here, so the Triton cross-compile is
  indicative, not authoritative (§6).
- It does not compile CuTe DSL for the target (§6).
- It does not update the *prompts* to carry `{TARGET_PROFILE_PATH}`: the prompt templates are
  not this task's files (invariant: `meta/*.md` is untouched). `runcontext` provides the
  mechanism and `target_profile.materialize` the file; wiring `_bindings` in
  `loop/driver.py` is an integration step (see the report's integration notes).
- `FK_TARGET_SMS` simulation changes the grid only (§2).
