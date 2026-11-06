# Reproduction: the rented-B200 day-1 runbook

> **Anti-fabrication preamble** (`docs/paper/README.md` binds this file too): every number the
> paper publishes comes from measured runs recorded on disk. This runbook contains **commands and
> gates only**; the few figures it states are labeled **ESTIMATE** (planning assumptions to be
> replaced by the day-1 measurements they name) or are pointers to files. No benchmark number
> lives here.

Purpose: when the rented B200 arrives, testing starts immediately, in this order, with these
gates. A failed gate is a stop — do not improvise around it; record the failure in the day's run
slot and fix forward.

Companion documents: `docs/paper/experiment-matrix.md` (arms, taxonomy, decision rules,
statistics); `docs/energize-protocol.md` (the shared-box discipline this file adapts to an
exclusive rental); `docs/external-eval.md` (the leased-card measurement discipline, section
"Self-evaluation on a leased card").

## 0. Ground rules on a rental

- The box is **exclusive**: the `~/tools/bin/with-lock gpu` discipline of
  `docs/energize-protocol.md` (rule 5) exists for the shared home lab and is **not needed**
  here — but the framework's own per-GPU serialization still applies (`RKT_GPU`, one GPU):
  its jobs take turns by itself.
- Because no other tenant can move the clocks, the honesty burden moves to **recording**:
  every job's `gpu_state` telemetry (SM/mem clocks, temperature, power via NVML) rides every
  reading, and the framework's **clock-drift guard** (two readings > 2% apart in SM clock or
  > 5 °C apart are not one measurement condition; a flagged reading produces no verdict) is the
  protection that matters. A quiet-rental claim without per-job telemetry is not evidence.
- Locking clocks is the card's `[protocol]` discipline (lock, read back within ±50 MHz, refuse
  to time when the lock does not hold, reset on exit). Use it if the rental exposes
  `nvidia-smi -lgc`; a rental that forbids clock locks is acceptable **only** with the drift
  guard on and the fact recorded in the run slot.
- Nothing about the rental is trusted: the day-1 calibration job (section 2, step 6) measures
  this box's own noise floor before any verdict is classified against it.

## 1. Environment gate (all green before any setup)

Run each check; a failing line is a stop until resolved. Record all outputs into
`docs/paper/runs/<yyyymmdd>-envgate/` on day 1.

| # | gate | command | pass condition |
|---|---|---|---|
| 1 | the GPU is a B200 and idle | `nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv` | name B200, compute cap 10.0; `nvidia-smi --query-compute-apps=pid --format=csv` lists nothing |
| 2 | driver/runtime for `sm_100a` | from (1) + `nvcc --version` | B200 needs a CUDA **runtime ≥ 12.8**; upstream's environment table (`kernels/README.md`) pins **CUDA 13.0** for B200 — treat driver reporting CUDA 13 capability as the target, ≥ 12.8 as the hard floor; `nvcc` must accept `-gencode arch=compute_100a,code=sm_100a` (the CUDA C++ DSA kernel in the corpus JIT-compiles with it) |
| 3 | torch build | `python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` | a cu128+ build (upstream pins PyTorch 2.13 + CUDA 13.0 for B200); `torch.cuda.get_device_capability() == (10, 0)` |
| 4 | CuTe DSL | `python3 -c "import cutlass; print(cutlass.__version__)"` | `nvidia-cutlass-dsl` 4.6.0 or 4.6.2 (upstream's B200 pin); 24 of the corpus's 25 kernels import it (all but the CUDA C++ DSA kernel) |
| 5 | Nsight Compute | `ncu --version` | present and on PATH; the framework's NCU jobs run with `--cache-control none --clock-control none` (the timing-regime anchor), which needs a reasonably recent ncu |
| 6 | NVML readable | `python3 -c "import pynvml; pynvml.nvmlInit(); h=pynvml.nvmlDeviceGetHandleByIndex(0); print(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM), pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM), pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU), pynvml.nvmlDeviceGetPowerUsage(h))"` | prints four sane numbers. **The whole honest stack depends on this gate**: `gpu_state` telemetry and the clock-drift guard cannot exist without NVML |
| 7 | python | `python3 --version` | ≥ 3.11 (`pyproject.toml` requires-python) |
| 8 | disk | `df -h .` | ≥ 60 GB free on the repo volume (ESTIMATE: torch + CUTLASS + NCU + `.artifacts/` run trees + JIT build caches; measured again after section 2, step 5) |
| 9 | clocks lockable (optional) | `sudo nvidia-smi -lgc <sm>,<sm>` then `nvidia-smi --query-gpu=clocks.current.graphics --format=csv`, then `sudo nvidia-smi -rgc` | read-back within ±50 MHz of the lock, and reset works. If the rental forbids it: record that fact; the drift guard stays mandatory |
| 10 | thermal/clock baseline | `nvidia-smi -q -d CLOCK,TEMPERATURE,POWER > docs/paper/runs/<slot>/env/smi-q-start.txt` | recorded, for comparison against every job's `gpu_state` |

## 2. Setup sequence

In order; each step depends on the previous one being green.

1. **Sync the repo.**
   ```bash
   git clone git@github.com:ChaseZhang-ai/RSI-Kernel-Tune.git   # or: git pull --ff-only
   cd RSI-Kernel-Tune
   git rev-parse HEAD                      # record into docs/paper/runs/<slot>/env/
   git cat-file -t 5030503d3ff7            # must print: commit  (Arm B's upstream object)
   ```
2. **Install** (environments into `~/venvs/`, tools into `~/tools/`, per the workspace's
   placement rules; never into the repo or the job tree).
   - **Reuse first**: check `ls ~/venvs` — an existing venv that passes the section 1 gate
     (python ≥ 3.11, the CUDA torch build, `nvidia-cutlass-dsl`) is the driver env; do not
     create a second one.
   - Driver env (editable install of this repo) and job env (the section 1 CUDA torch +
     `nvidia-cutlass-dsl` pins) may be the same venv on day 1; keep them separate only when
     the job env needs pins the driver must not get.
   ```bash
   python3 -m venv ~/venvs/rkt-b200 && ~/venvs/rkt-b200/bin/pip install -e .
   # + the CUDA torch / nvidia-cutlass-dsl pins from the section 1 gate, into the same venv
   ~/venvs/rkt-b200/bin/rsi-kernel-tune --version
   git clone --depth 1 https://github.com/NVIDIA/cutlass ~/tools/cutlass   # header-only
   ```
3. **`rsi-kernel-tune config init`** — writes the commented `rsi-run.toml` template (unified
   run configuration: `[api]`, `[kernel]`, `[precision]`, `[student]`, `[run]`, `[language]`).
   Fill per arm before each arm's runs (Arms B/C); Arm A needs no API section at all.
4. **API keys via `local.env` only.** Copy `local.env.example` → `local.env` (gitignored);
   set `RKT_DELIVERY=local`, `RKT_GPU=0`, the provider key its `global_config/backends.toml`
   `key_env` names (or the `native` credentials), and the teacher/student model selection goes
   in `rsi-run.toml [api] models`. **No secret, key, or token may appear in any file under
   `docs/`** — the docs get the provider/model *names* and the key-file pointer, never the key.
   Arms B and C spend tokens; Arm A runs with no API at all (the released kernel bytes are the
   student file — no agent turns).
5. **Preflight smoke** — the whole run wrap on a stub agent, no API, no performance conclusion
   (the same red line the energize protocol draws for the 3060):
   ```bash
   bash tasks/wp21mini/run_smoke.sh
   ```
   Green = judge → card → seal → calibrate → NCU probe → loop.toml → workspace → baseline →
   loop all work end-to-end on this box. The smoke's wall-clock composition line is also the
   first `t_job` datum. **Adapting the smoke to a fresh rental**: the script pins
   `~/venvs/forgekernel` (driver python) and `~/venvs/ml/bin` on PATH (the job-side CUDA-torch
   python3), and `CUTLASS_DIR=~/tools/cutlass` for the job-env precheck — recreate those two
   venv names (or export the same variables before invoking it). The `LIBRARY_PATH` line in
   the script is a WSL-only `libcuda.so` dev-symlink workaround and does not apply on a native
   Linux rental; ignore it (the script appends, so leaving it is harmless).
6. **Noise-floor calibration job — FIRST, before any verdict is produced.** The framework
   calibrates its resolution per-run (`resolvable_diff` from each run's own paired ratios), so
   *make calibration its own explicit day-1 step* and every later verdict inherits a floor
   measured on this box instead of a constant:
   - re-time the smoke card's kernel unchanged-tree, R = 6 repeat jobs (the estimator's sample
     count), `profile = full`, seeds 0–2;
   - compute the box floor from the paired ratios with the shared arithmetic
     (`scripts/retime_corpus.py --collect` computes floors from repeat jobs the same way);
   - write `docs/paper/runs/<slot>/floor.json` (floor, per-seed ratios, `t_job` measured);
   - every Arm's classification and every budget estimate now cites this floor and this
     `t_job`, replacing the ESTIMATE ranges in `experiment-matrix.md`.
7. **Arm A** — plan and run the corpus re-timing:
   ```bash
   python3 scripts/retime_corpus.py --plan --arch b200 > docs/paper/runs/<slot>/retime-plan.json
   # run the plan's jobs (floor repeats, then honest + upstream_style timing jobs)
   python3 scripts/retime_corpus.py --collect --retime-root <retime-root> \
       > docs/paper/runs/<slot>/retime-rows.json
   ```
   (`--plan` is offline and safe to run before the box arrives.)
8. **Arm C** — the task set itself has landed (`tasks/b200-set/`: ten operator cards,
   `tasks/b200-*`, dry-run validated); the runs start only after the `[profile baseline]`
   off-switch bundle lands and `python3 -m pytest tests/test_baseline_profile.py -q` is green.
   Then run the same task set under both profiles, same seeds, audit both.
9. **Arm B** — the long API arm; start after A and C so the harness and the floor are warm:
   ```bash
   git worktree add ../forge-5030503 5030503d3ff7
   ```
   configure its runs per upstream's own surface, one run per (operator × seed); per run:
   `rsi-kernel-tune audit <run>` (the read-only provenance report + false-condemnation
   ledger) and one honest-stack re-timing job of the run's confirmed-best kernel
   (classified per `experiment-matrix.md` §4).
10. **Arm D** — offline; after Arms A/B/C have produced the collected-kernel corpus; run the
    external judges per `experiment-matrix.md` §3 (Arm D), outside this framework.

## 3. Bookkeeping (every run, every arm)

- **Run slots**: `docs/paper/runs/<yyyymmdd>-<arm>-<n>/` — commit the run's `rsi-run.toml`
  (paths + seeds; never keys), the plan/collect JSONs, and a one-line RESULT note per run
  pointing at its `.artifacts/` archive (archives stay out of git, per the energize protocol).
- **Env snapshot** per slot: `pip freeze`, `nvidia-smi -q`, driver + `nvcc` + `ncu` versions,
  the section 1 gate outputs, the git HEAD — into `<slot>/env/`.
- **Cost log** per slot, appended per job; this feeds table T5:

  | date | arm | job | profile | seed | wall-min (measured) | GPU-h | API tokens |
  |---|---|---|---|---|---|---|---|

  (GPU-h = wall-min/60 on this single-GPU box; API tokens from the provider's usage view —
  recorded per arm, never estimated in the paper.)
- **Failures are data**: a gate failure, a `sentinel_flagged` or `unverifiable` row goes into
  the slot's notes with its reason string — they are reported in the paper, not cleaned away.

## 4. Order and why

calibration (6) → **A** (7, no API: exercises the audit lane and the corpus plumbing cheaply) →
**C** (8, one-variable stack comparison) → **B** (9, the expensive arm, warm floor, warm
harness) → **D** (10, offline, needs the corpus). Arm order also fixes attribution: any
harness defect Arm A flushes out cannot contaminate B/C's verdicts.

<!-- SPDX-FileCopyrightText: Copyright 2026 The RSI-Kernel-Tune Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
