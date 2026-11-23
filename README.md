# RSI-Kernel-Tune

Research repository for **recursive self-improving GPU kernel generation**, targeting the
**SOL-ExecBench / FlashInfer-Bench** leaderboard.

This repository is developed by two people. To keep the work separable, each track lives in
its own top-level folder and is labelled with one name:

| folder | owner | track |
|---|---|---|
| [`vulcan/`](vulcan/README.md) | **zixinzenith** | reproduce the ForgeKernel architecture + score-aware orchestration research (VULCAN) |
| *(to be added)* | other contributor | separate track |

Nothing outside `vulcan/` belongs to the VULCAN track. Do not edit `vulcan/` from the other
track, and do not edit the other track's folder from here.

---

## TL;DR (board snapshot: 2026-11, B200, evaluation stack v1.1)

* The reference system, **ForgeKernel** (OpenBMB/Tsinghua, arXiv:2611.04998), is **rank 2** on
  the FlashInfer-Bench collection: **SOL Score 0.815204**, Fast₁ **647/660**.
* **Rank 1 is "Golden Bridge"** with **0.815321** and Fast₁ **654/660**. The paper's "#1 as of
  2026-10-30" has since been overtaken.
* We reverse-engineered the scoring rule exactly: the collection score is the **unweighted
  arithmetic mean of the 26 per-kernel SOL scores**, and each per-kernel score is the
  **geometric mean over its workloads** of
  `S = 1 / (1 + (T − T_SOL) / (T_b − T_SOL))`. Reproduced to six decimals.
* **Consequence:** with equal weight per kernel, headroom lives in the mid/low kernels (GEMM,
  small-`h` RMSNorm), not in the attention kernels that already sit at `S ≈ 0.99`.
* **Matching the current best public kernel on every kernel already yields 0.819011 — above
  rank 1.** So #1 is reachable by better *orchestration* before any new kernel idea.
* The remaining Fast₁ gap is concentrated: 13 workloads, all in four GEMM definitions
  (`007`: 3, `009`: 8, `010`: 1, `011`: 1).

Details: [`vulcan/docs/LEADERBOARD_ANALYSIS.md`](vulcan/docs/LEADERBOARD_ANALYSIS.md).

---

## What VULCAN is

A **score-aware orchestration layer** over a faithful reproduction of ForgeKernel. It does not
fork the upstream loop; it supervises it, decides *which* kernel gets *how many* GPU-hours,
*which architecture hypotheses* to try, and *when a gain is real*. Everything is CPU-validated;
the kernels themselves run on the GPU.

| capability | module | locally validated |
|---|---|---|
| exact leaderboard objective + latency targets | `vulcan/objective.py` | ✅ |
| score-aware budget allocation (exact DP + greedy) | `vulcan/allocator.py` | ✅ |
| architecture portfolio / successive halving | `vulcan/portfolio.py` | ✅ |
| Fast₁ straggler triage + curriculum | `vulcan/curriculum.py` | ✅ |
| executable-predicate knowledge graph + negatives + cross-arch analogs | `vulcan/kb_graph.py` | ✅ |
| precision-contract guard + adversarial probes + reward-hack scan | `vulcan/precision_guard.py` | ✅ |
| statistical promotion gate (paired bootstrap CI) | `vulcan/stats.py` | ✅ |
| shape clustering (exact 1-D k-means) | `vulcan/cluster.py` | ✅ |
| roofline → bottleneck → lever diagnosis | `vulcan/roofline.py` | ✅ |
| closed-loop supervisor + mock GPU backend | `vulcan/supervisor.py`, `backends/` | ✅ |
| production adapter to upstream `forge` CLI | `vulcan/backends/forge.py` | dry-run only |

Ranked improvement designs and their evidence: [`vulcan/docs/IMPROVEMENTS.md`](vulcan/docs/IMPROVEMENTS.md).

---

## Quickstart (no GPU)

```bash
cd vulcan
export PYTHONPATH=src
python3 -m vulcan selftest                 # internal consistency
python3 -m vulcan stats                    # headroom table from the live board snapshot
python3 -m vulcan plan --hours 200         # where the GPU-hours should go
python3 -m vulcan demo --hours 200 --seed 7  # mock A/B: uniform vs headroom vs VULCAN
bash scripts/local_validate.sh             # everything above + 74 unit tests
```

Local A/B result (synthetic ground truth, 200 GPU-hours, reproducible via `--seed`):

| strategy | S_after | gain vs uniform |
|---|---|---|
| uniform (equal hours, one architecture) | 0.8815 | — |
| headroom (hours ∝ 1−S) | 0.8956 | +0.0140 |
| allocation only (exact DP) | 0.8932 | +0.0117 |
| portfolio only (successive halving) | 0.9190 | +0.0375 |
| **VULCAN (allocation + portfolio + curriculum)** | **0.9271** | **+0.0456** |

The portfolio term dominates; the allocation term is real but second-order. Both are reported
separately so the claim can be checked rather than believed.

---

## Path to a real leaderboard run

Read [`vulcan/docs/B300_RUNBOOK.md`](vulcan/docs/B300_RUNBOOK.md). Two facts up front:

1. **The board scores on B200 at a locked clock.** B300 is for generation and iteration; the
   final submission must be measured on the board's B200 clock to be comparable. Do not score
   B300 timings against B200 `T_b`/`T_SOL`.
2. **The reproduction vendors upstream at a fixed commit** (`vulcan/vendor/UPSTREAM.lock`,
   `b2c8ae8`, 2026-11-11). `scripts/bootstrap.sh` clones and installs it.
