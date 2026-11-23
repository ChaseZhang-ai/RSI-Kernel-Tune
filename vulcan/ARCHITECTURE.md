# Reproduction of the ForgeKernel architecture

**Upstream:** OpenBMB/ForgeKernel — `https://github.com/OpenBMB/ForgeKernel`
**Pin:** `b2c8ae8510e02c28576808910a6b8827721ab7dc` (2026-11-11), Apache-2.0
**Paper:** *ForgeKernel: Self-Evolving Kernels Catching and Surpassing Experts and Agents*,
arXiv:2611.04998.
**Vendored at:** `vulcan/vendor/ForgeKernel/` (gitignored; created by `scripts/bootstrap.sh`).

We reproduce the architecture by **vendoring the upstream at a fixed commit and supervising it**,
rather than re-implementing 135k lines. This section documents what the architecture actually is,
where each piece lives in the code, and how the VULCAN layer attaches.

---

## 1. The three pillars of ForgeKernel

### Pillar 1 — separation of architectural design from code implementation

A dedicated **architecture agent** owns design decisions and plan revisions; a separate
**coding agent** realizes them. The interface is *a design plan going down, measurement evidence
coming up*; the coding agent never sees the reference implementation's source.

* **Milestone plans.** Optimization is a curriculum `M0 → M1 → … → Mk`. `M0` fixes the
  architecture in one shot — kernel split, dataflow and stationarity, intermediate-tensor ABI,
  precision contract, tile shapes, warp roles, register budget, smem layout, MMA atom layout — and
  is **frozen** once its gate passes (correctness + architecture-conformance + mechanism lines;
  no absolute performance line). Later milestones only tune inside that structure. When
  measurement shows only a new structure can lift the limit, the architecture agent revises `M0`.
* **Decision five-tuple.** Every decision is written as `Action / Purpose / Trigger condition /
  Cost / Depends on`, with a machine-read dependency graph (`⇐` = derived-from, `⊗` = must be
  co-committed atomically). A patch's `changed_entries` propagate prefix invalidation along `⇐`.

Code: `src/forgekernel/milestone.py` (838 lines — DAG parse, prefix invalidation, validators),
`meta_milestone.md` (operator-agnostic distillation policy), `docs/milestones-template.md`
(the normative template), `src/forgekernel/loop/driver.py` (the two-layer loop, 12k lines),
`src/forgekernel/loop/agent_policy.py`, `src/forgekernel/agents.py` (per-role agent command),
`src/forgekernel/task.py` + `taskcard.py` (roles, models, effort, budgets).

### Pillar 2 — self-evolution: within-run revision + cross-run reference promotion

* **Within-run:** when the executable decisions in the current plan are exhausted or a milestone
  cannot close, the trajectory and measurements are returned to the architecture agent, which
  revises decisions and the plan. Structural changes restart from the earliest affected decision
  (up to `M0`). A change with no attributable gain is reverted and its failure conditions recorded.
* **Cross-run:** an **accepted kernel is sealed and becomes the next run's reference** — a
  measurable black box whose source is *not* available to the next run's agents. The architecture
  agent re-analyzes it and issues a fresh plan, so an existing optimized artifact becomes the
  starting point for new architecture analysis instead of a frozen constraint. The chain stops
  when a generation's gain does not exceed the noise threshold.

Code: `driver.run_two_layer` (plan revision; stagnation triggers), `src/forgekernel/sealref.py` and
`binref.py` (sealing a source-hidden / binary reference), `resume/invalidation.py` (prefix
invalidation on a plan patch), `runs.py`, `baseline.py`, `loop/harness_guard.py`,
`scripts/self_evolve.py` (the paper's self-evolution experiment).

### Pillar 3 — a measurement-validated hardware knowledge base

Mechanisms discovered during optimization are validated by **independent probes on the target
GPU** before being stored; entries are organized per GPU architecture and per kernel language, and
are cited in later milestone plans. Each entry records mechanism, applicability conditions,
trigger symptoms, a minimal runnable example and known limitations.

Code: `hw_kb/{cuda,cutedsl}/{sm90,sm100}/` (+ `TEMPLATE.md`, `README.md`),
`src/forgekernel/loop/kb_lint.py` (linting), `loop/harvest.py` (write-up of a stopped run),
`docs/kb-harvest.md`, `docs/kb-provenance/`, `skills/` (the agent's tools: `sass`, `stall`,
`timeline`, `orchestration`, `lever-pricing`, `ncu-trace`, `job-env-check`).

---

## 2. Supporting machinery

| concern | where |
|---|---|
| execution harness (correctness, paired timing, mechanism validation) | `loop/driver.py`, `loop/precision_gate.py`, `judge.py` |
| correctness monitor under stressed inputs | `src/forgekernel/monitor/` |
| GPU-job submission (ssh/local) | `src/forgekernel/submit/` |
| leaderboard task cards (both boards' judges + stopwatches, pinned upstream) | `src/forgekernel/card_template/eval/{leaderboard,closed_ref,precision,timing,ncu_probe}.py` |
| timed-window contract + audit | `window_contract.py`, `timing_audit.py` |
| NCU / SASS / trajectory collection | `src/forgekernel/inner/{ncu,results,trajectory,workspace}.py` |
| config | `global_config/{models,loop,backends}.toml`, `tasks/<op>/task.toml` |
| CLI | `src/forgekernel/__main__.py` (`forge init/new/run/status/harvest/export`), `launch.py`, `loop/cli.py` |
| tests (fake backend; no GPU) | `tests/` (`test_self_evolve.py`, `test_template_paired_timing.py`, `test_false_win_guard.py`, `test_resume_invalidation.py`, …) |

The loop's side-effecting seam is a protocol, which is why it is testable without a GPU:

```python
class DriverBackend(Protocol):
    def run_agent(self, prompt: str) -> str: ...
    def submit_gpu(self, branch: str, mode: str) -> str: ...
    def commit(self, message: str) -> str: ...
    def push_branch(self, branch: str) -> None: ...
    def reset(self, ref: str) -> None: ...
    def head(self) -> str: ...
```

## 3. Leaderboard fidelity (why the task's verdict equals the board's verdict)

Upstream replicates each board's evaluation exactly, pinned to an upstream commit
(`flashinfer-bench@15618a67`, `SOL-ExecBench@a9fa0804`): the tolerance, the input generation, the
anti-cheating checks and the stopwatch. For SOL-ExecBench the card does **not** time the eager
PyTorch reference (host-bound noise) but ranks against the board's published `T_b` (scoring
baseline) and `T_SOL` (speed-of-light floor), frozen into `problem.json` at `forge init` time.
It locks clocks (SM 1500 / memory 3996 MHz on the B200 preset) and passes the board's own SOL
score into the eval JSON (`sol_board.{sol_score, vs_t_b}`). See
`vendor/ForgeKernel/docs/leaderboards.md` for the authoritative description.

This is the single most important thing the reproduction buys us: **our local measurements and the
board's judgement use the same judge.**

## 4. Reproduce it

```bash
# 1. clone + install the pinned upstream
vulcan/scripts/bootstrap.sh                 # needs Python >= 3.11 for upstream

# 2. datasets (see upstream docs/leaderboards.md for URLs)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace
#    SOL-ExecBench: run scripts/download_solexecbench.py in that repository

# 3. create and run one board task (smoke test)
cd vulcan/vendor/ForgeKernel
python -m forgekernel init rmsnorm --leaderboard sol-execbench \
    --dataset ~/SOL-ExecBench/data/benchmark \
    --definition L1/... --workload 0        # authoritatively: use --list-workloads first
python -m forgekernel new tasks/rmsnorm
python -m forgekernel run .artifacts/rmsnorm_<id>
python -m forgekernel export .artifacts/rmsnorm_<id> --out solution.json --name vulcan --author zixinzenith
```

The `<definition>` ids and workload indices come from the board; use
`--list-workloads` to enumerate them. `vulcan/scripts/fetch_board.py` collects the same
per-workload `T_b`/`T_SOL` into `vulcan/data/` for planning.

## 5. How VULCAN attaches

VULCAN never edits upstream. It (a) plans with the board's objective, (b) issues upstream
`forge` commands through `backends/forge.py`, and (c) gates promotion with
`stats.accept_gain` and `precision_guard`. Concretely:

```
vulcan.plan          -> per-kernel hours + per-workload target_ms   (objective, allocator)
vulcan.curriculum    -> which kernel next, Fast1 stragglers first    (curriculum)
vulcan.portfolio     -> which M0 hypotheses to try, at what budgets  (portfolio)
vulcan.backends.forge-> forge init/new/run/export                    (adapter)
vulcan.precision_guard / stats -> accept or revert the candidate     (gate)
vulcan.kb_graph      -> record certified mechanisms + refuted ideas  (memory)
```

The one bridge not yet written is `kb_graph` → upstream `hw_kb/<lang>/<arch>/` markdown (so that
mechanisms VULCAN certifies appear in the upstream knowledge base that the agents read). Until
that adapter exists, run with upstream's own `hw_kb` and treat `kb_graph` as the planner's memory.

## 6. Known gaps / caveats

* The open-source release does **not** include the paper's LLM static review and the quantisation /
  sampling special evaluators; upstream documents which definitions it refuses (`sampling`,
  DSA top-k indexer).
* Two boards publish `T_b`/`T_SOL` measured on *their* machine; the residual between their B200 and
  ours remains in the score. This is a property of the board, not of the reproduction.
* A host-bound multi-kernel call is unstable under any single-call stopwatch; hence SOL tasks rank
  against published `T_b`, not a timed PyTorch reference.
