# Options reference — every knob in RSI-Kernel-Tune

Three layers: **task contract** (you declare per operator, in the task's `task.toml`),
**loop governance** (per run, in the run config / `loop.toml`), **infrastructure**
(per machine, in `local.env` / env). Every knob has a default and a rollback switch.

## Layer 1 — task contract (you declare)

| Knob | Where | Meaning | Default | Effect when changed |
|---|---|---|---|---|
| `compute_dtype` | run config `[precision]` | The precision the optimized kernel computes at: `fp32` (default) / `tf32` / `bf16` / `fp16` / `fp8`. **This is your decision** — the system optimizes at exactly the declared precision and never goes below it. The fp64 error-ratio gate still judges every candidate. | `fp32` | Lower dtype = faster kernels, larger numeric error. The gate bounds the error; if a candidate misses the bound it is reverted regardless of speed. |
| `memo_lane` | task.toml `[protocol]` | Industrial memoization lane: the kernel keeps a content-keyed cache — judged (repeating) inputs become cache hits; unseen inputs compute honestly through the fallback door. | `false` | `true` = production serving latency on repeating inputs (much faster on hits). Never mixed with STRICT-lane numbers. |
| clock | task.toml `[protocol]` `sm_clock_mhz`/`mem_clock_mhz` | Declared measurement clock; the harness verifies the lock and refuses to time unlocked. | unlocked | A declared, verified clock makes timing stable across runs. |
| tolerance / shapes / inputs | task.toml `[task]`/`[shapes]` | What the operator computes, on what inputs, and what "correct" means. | — | The task itself — not a knob. |
| `prior_survey` / `prior_art` | task.toml `[agent.teacher]` | Teacher-side prior supply: public-source survey and the prior-art library. | `false` | `true` = the teacher starts from public-state-of-the-art context before distilling the curriculum. Re-priced by measurement in this run regardless. |
| `stagnation_k` (+ override) | task.toml `[loop]` | After N non-improving rounds the loop forces an architecture change instead of parameter grinding (the teacher may override once). | 3 | Higher = more patience; lower = earlier structural escape. |
| `max_meta` / `plateau_window` / `gate_window` / confirm budget | task.toml `[loop]`/`[budget]` | GPU-time spend per experiment: teacher revision rounds, the plateau judgment window, the gate comparison window, per-round confirmation buys. | per task | The budget dials — prevents airtime waste. |
| `floor_stop` / `floor_margin` | task.toml `[loop]` | Analytic-floor stop: when the measured kernel reaches the computed hardware floor (within margin), stop. | `false` (defaults on for tasks with a computable floor after the 2026-11-05 default flip; rollback: set `false`) | Direct time savings — no more runs once the wall is reached. |
| `sprt_early_stop` | loop config | Sequential early stop: an experiment that is clearly losing ends early instead of sampling to the full window. | on (evidence-backed) | Saves 50%+ GPU time on losing experiments. |
| `merge_eval_ncu` | loop config | Evaluation and profiling in one job (one build+launch instead of two). | on | Direct speedup per iteration. |
| `ncu_on_revert` | loop config | Whether reverted kernels are also profiled. | on | `off` saves time; `on` keeps the teaching value of failures. |
| build cache | env `RKT_BUILD_CACHE=1` | Content-keyed build cache: identical sources are never rebuilt (~40% of job wall clock). Artifact hashes are re-verified on every hit; anything suspicious falls back to a full build. | off pending final real-machine evidence; rollback: unset | Direct speedup per iteration. |
| background lane | env `RKT_BACKGROUND_LANE=1` | While the agent thinks, the GPU queues non-gate measurements (confirmations, KB certification) instead of idling. Gate readings are never served from the lane. | off (rollback: unset) | Direct speedup — the idle window works for you. |

## Layer 2 — run config (`rsi-run.toml`, `rsi-kernel-tune config init`)

| Section | Fields | Meaning |
|---|---|---|
| `[api]` | `provider`, `key_file`, `models` | Where the teacher/student LLM tokens come from (the key itself lives in `key_file`, gitignored — never in this file) and which models the agents may use. |
| `[kernel]` | `task` or `from_kernels` | What to optimize: an existing task dir, or a published kernel's frozen contract from `kernels/`. |
| `[precision]` | `compute_dtype` | Your declared compute precision (see Layer 1). The banner in the generated config says it explicitly: YOUR decision. |
| `[student]` | `retrieval` | `"off"` (default) = the student optimizes from scratch (it never sees the reference's source or any optimized implementation — this is the anti-cheat contract and has no off switch); `"on"` = the student may first survey public sources, then tune on that basis (still never the reference itself). |
| `[run]` | `parallel` | `"auto"` (default) = one concurrent slot per kernel task — kernel tasks run in parallel, each with its own artifacts and worktree. A number = fixed slot count. |
| `[language]` | `name`, `syntax_docs`, `toolchain_config` | Only when optimizing in a NEW language: point at the syntax/style reference (teacher and student read it first) and the toolchain config. |

## Layer 3 — infrastructure (`local.env` + env)

| Knob | Meaning |
|---|---|
| `RKT_DELIVERY` / `RKT_GPU` | Where the GPU is (local / ssh host) and which device index. |
| `ANTHROPIC_API_KEY` / provider key env | The agents' LLM credentials (from your `key_file`). |
| `RKT_PRECISION_LANE` / `RKT_COMPUTE_DTYPE` | Emitted from the task contract — not set by hand. |

## The two things that are NOT knobs

- **Anti-cheat (L0)**: the student never sees the reference's source or any optimized
  implementation. No switch exists. (The MEMO lane's cache is a production feature with a
  correctness door — not a way to see answers.)
- **The measurement math**: paired window timing, the fp64 error-ratio gate, the stress
  profiles. No switch exists — they are why a win on this framework is trustworthy.

## Rollback discipline

Every default flipped for speed keeps its rollback switch (set the env/key back).
Details per knob: [docs/run-configuration.md](run-configuration.md),
[docs/industrial-lane.md](industrial-lane.md), [docs/compute-dtype.md](compute-dtype.md).
