# VULCAN

**V**alue-of-information **U**tilization for **L**eaderboard-**C**onditioned **A**rchitecture
search — a **N**etwork of score-aware optimizers over a reproduced ForgeKernel.

> `vulcan/` is one person's track of RSI-Kernel-Tune (see the root [`README.md`](../README.md)).
> The other contributor's work lives in a sibling top-level folder. The name is only a label so
> the two tracks stay separable.

---

## 1. Why VULCAN exists

The leaderboard we are attacking — **SOL-ExecBench, FlashInfer-Bench collection (B200)** — does
not score raw speedup. It scores a **SOL Score**, and the collection score is computed in a very
specific way. We recovered it exactly from the public API (see
[`docs/LEADERBOARD_ANALYSIS.md`](docs/LEADERBOARD_ANALYSIS.md)):

```
per workload  S_w = 1 / (1 + (T_w − T_SOL,w) / (T_b,w − T_SOL,w))
per kernel    S_k = geomean over the kernel's workloads of S_w
collection    S   = UNWEIGHTED arithmetic mean over the 26 kernels of S_k
```

Two consequences decide everything:

1. **Every kernel counts equally, no matter how many workloads it has.** A GEMM at `S=0.53`
   (`009_gemm_n5120_k2048`) carries the same weight as an attention kernel at `S=0.99`, but has
   ~40× the headroom. Optimizing the attention kernels further is nearly worthless; the GEMM and
   small-`h` RMSNorm kernels are where the score is.
2. **`S` is convex in latency**, so the value of a fixed *fractional* latency cut grows as the
   kernel leaves the floor. Effort (GPU-hours) therefore has a non-obvious optimum, which
   [`allocator.py`](src/vulcan/allocator.py) computes directly.

A system that targets "beat the reference by 20% per kernel" optimizes the wrong thing. VULCAN
targets the published objective.

## 2. State of the board (snapshot 2026-11, B200, stack v1.1)

| entry | SOL Score | Fast₁ |
|---|---|---|
| SOL Bound (unreachable ceiling) | 1.000000 | 660/660 |
| **Golden Bridge** (rank 1) | **0.815321** | 654/660 |
| **ForgeKernel** (rank 2, the reproduction baseline) | 0.815204 | 647/660 |
| Scoring Baseline | 0.500000 | 0/660 |

**Mean of the best public kernel per kernel = 0.819011 → already above rank 1.** Matching the
best-known implementation on every kernel would take first place. VULCAN's orchestrator is built
to do exactly that and then push past it.

## 3. What is reproduced, and what is added

**Reproduced** (vendored, not forked): the full ForgeKernel architecture — architecture agent /
coding agent split, milestone plans with a frozen `M0`, self-evolution over sealed references,
and the probe-validated hardware knowledge base. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the
paper-to-code map and the reproduction procedure.

**Added** (`src/vulcan/`, standard library only, no dependency on upstream at import time):

| module | what it does |
|---|---|
| `objective.py` | the exact board algebra; `latency_for_score` turns a score target into the per-workload `[target] ms` the upstream loop needs |
| `allocator.py` | score-aware budget allocation: exact DP (multiple-choice knapsack) over integer hour-slots, plus a marginal-greedy fast path; explicit, bounded improvement model |
| `portfolio.py` | successive halving + UCB over architecturally distinct M0 hypotheses (attacks ForgeKernel's stated local-optimum limitation) |
| `curriculum.py` | kernel ordering (value × family knowledge transfer) and Fast₁ straggler triage |
| `kb_graph.py` | knowledge graph with **executable applicability predicates**, first-class **negative** (refuted) entries, and **cross-architecture analog** lifting (candidate-only) |
| `precision_guard.py` | precision contract enforcement, adversarial probes, static reward-hack scan — makes accuracy a hard constraint |
| `stats.py` | paired-ratio bootstrap CI and a promotion gate that refuses gains below the measured noise floor |
| `cluster.py` | exact 1-D k-means (Jenks DP) shape clustering — the upstream roadmap's "shape clustering" item |
| `roofline.py` | counters → bottleneck class → lever list; also produces the features `kb_graph` matches on |
| `calibration.py` | cross-device transfer: fit `t_ref ≈ a·t_dev + b` (or a ratio) from paired samples, predict B200 scores from B300 measurements **and** emit B300 development targets for a B200 score goal |
| `kb_hw_adapter.py` | render certified entries as upstream `hw_kb/<lang>/<arch>/*.md` (with the banned-token lint), so the agents actually read VULCAN's mechanisms |
| `supervisor.py` | the closed loop gluing all of the above over a `CampaignBackend` |
| `backends/mock.py` | deterministic synthetic GPU (CPU validation + falsifiable A/B) |
| `backends/forge.py` | production adapter to the upstream `forge` CLI |

## 4. Quickstart

```bash
cd vulcan
export PYTHONPATH=src

python3 -m vulcan selftest                   # fast internal consistency checks
python3 -m vulcan stats                      # headroom ranking of the 26 kernels
python3 -m vulcan plan --hours 200           # allocation + per-kernel latency targets
python3 -m vulcan plan --hours 200 --json-out plan.json   # machine-readable plan
python3 -m vulcan transfer --target best --ratio 1.8      # B300 dev targets / predicted scores
python3 -m vulcan demo --hours 200 --seed 7  # mock A/B ablation

python3 -m unittest discover -s tests        # 94 tests
bash scripts/local_validate.sh               # all of the above
```

No GPU, no torch, no third-party packages: Python ≥ 3.9 and the standard library.

## 5. Local validation status

* **94 unit tests pass** (`tests/`): objective round-trips, aggregation identities, exact-DP ≥
  greedy ≥ uniform allocator on random instances, successive-halving best-arm recovery, knowledge
  predicate/negative/analog behaviour, precision-contract flags, exact 1-D k-means recovery,
  dataset integrity (26 kernels / 660 workloads, the three published scores to 6 dp, and Fast₁
  fidelity: the reconstruction reproduces ForgeKernel's 647/660), and an end-to-end mock campaign
  (budget respected, VULCAN beats uniform on average, determinism, invalid-kernel revert).
* **The production adapter is integration-tested**: `tests/fake_forge/` is a stub `forgekernel`
  package that mimics the upstream CLI; `tests/test_forge_backend.py` runs `ForgeBackend` end-to-end
  against it (init → new → run → parse `device_median_ms` → export), so the subprocess and JSON
  parsing paths are exercised on CPU, not just dry-run.
* **Cross-device transfer is unit-tested** (`tests/test_calibration.py`): affine and ratio fits
  recover known maps, score round-trips through the calibration, and intervals behave.
* **A/B ablation** (synthetic ground truth, `--seed` reproducible): portfolio contributes the
  large lift, allocation a smaller one; reported separately.
* **Not locally verifiable** (needs GPU): actual kernel latency, the upstream loop under real
  agents, and the board's clock-locked measurement.

## 6. Layout

```
vulcan/
  README.md                  this file
  ARCHITECTURE.md            reproduction of ForgeKernel: paper <-> code map + procedure
  docs/
    LEADERBOARD_ANALYSIS.md  exact scoring rule recovery + per-kernel targets
    IMPROVEMENTS.md          ranked improvement designs, evidence, risks
    B300_RUNBOOK.md          step-by-step GPU execution plan and checklist
  data/                      board snapshot (per-workload T_b / T_SOL, published scores)
  configs/                   vulcan.yaml, forge_loop.toml.example
  src/vulcan/                the code (see table above)
  tests/                     94 CPU tests (incl. fake_forge/ integration stub)
  scripts/                   bootstrap.sh, local_validate.sh, fetch_board.py
  vendor/UPSTREAM.lock       upstream repo + pinned commit
```

## 7. How VULCAN attaches to upstream

`backends/forge.py` drives the documented upstream CLI per kernel:
`forge init ... --leaderboard sol-execbench` → `forge new` → `forge run` →
`forge export`. VULCAN decides the kernel order, the hour budget, the architecture hypotheses,
the `[target] ms`, and whether to promote the result. The upstream loop is unchanged, so an
upstream update only needs a re-pin, not a re-write.

`kb_hw_adapter.py` closes the loop the other way: certified VULCAN mechanisms are rendered into
upstream's `hw_kb/<lang>/<arch>/` markdown (matching its `TEMPLATE.md`, provenance routed out of
the agent read set), so the upstream agents can cite them in milestone plans.
