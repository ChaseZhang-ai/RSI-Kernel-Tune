# ARGUS — Adversarial Reward-hack Guard with Unified Search

A **verifier-first, attribution-driven agentic GPU-kernel optimizer** for the
**SOL-ExecBench** leaderboard (NVIDIA, B200). ARGUS is one track of `RSI-Kernel-Tune`; the
sibling top-level folder (`vulcan/`) is a different track and is not touched here.

> **Base system reproduced:** **CudaForge** (`OptimAI-Lab/CudaForge`, MIT, pin
> `eaba693f`) — a training-free two-agent loop (Coder + Judge) that reads Nsight-Compute
> counters and iterates. It is a *different* system from ForgeKernel, which is the base of
> the sibling track.

ARGUS keeps CudaForge's two-agent loop and adds the three things that decide whether a
kernel is *scored* or *zeroed* on the board:

| # | Contribution | Module | Local status |
|---|---|---|---|
| **C1** | **Board-faithful, hack-resistant verification** — the SOL element rule over *all* workloads, differential probes, a float64 precision contract, static scan, and a counter fingerprint. | `verifier.py`, `probes.py`, `fingerprint.py` | ✅ validated |
| **C2** | **Causal, counter-grounded attribution + model-based pre-screening** — turns the raw counter dump into a bottleneck, a *delta attribution* with an unexplained-speedup flag, and a bounded lever list; pre-screens mechanisms before a GPU eval is spent. | `attribution.py`, `roofline.py` | ✅ validated |
| **C3** | **Population search with regression-proof lineage** — K architecture bets, UCB over mechanisms, prerequisite-aware arm filtering, multi-fidelity verification-gated promotion; the objective is the board score, not raw speedup. | `search.py` | ✅ validated |

Supporting (not claimed as novel, see `docs/IMPROVEMENTS.md` for honest scope): the exact
board objective (`objective.py`, shared substrate), cross-task mechanism memory
(`memory.py`), a lightweight eval-budget planner (`schedule.py`).

---

## Why verification first

SOL-ExecBench scores a kernel with a **mean of per-kernel SOL scores**, and a kernel that
fails correctness is **zeroed**. CudaForge certifies with a single `torch.allclose` at 1e-3
on one shape. That gate is blind to the four things most likely to zero a submission —
tf32/precision downgrading, shape specialisation, memoisation, and faked timings — and
CudaForge's objective (`ref_latency / test_latency`) actively *rewards* them.

Local, falsifiable result (`python -m argus.cli hacklab`):

```
kernel                CudaForge    ARGUS   why
honest                     True     True
tf32_accum                 True    False   tf32 accumulation (SOL's #1 flagged hack class)
shape_specialised          True    False   correct on the tuned shape, zeros elsewhere
memoised                   True    False   computes once, returns that answer for every input
fake_fast                  True    False   reports a 3x-faster time with unchanged counters
hacks passing CudaForge: 4/4      hacks passing ARGUS: 0/4
```

Held at equal budget, that folds straight into the leaderboard metric
(`python -m argus.cli ab`):

```
(i)  verification: same greedy loop; effective board score  0.0000 (gate off)  ->  0.6472 (gate on)
(ii) search:       verification on for both; landscape optimum 0.9412
                   greedy chain 0.7942   ARGUS population 0.9412  (every seed)
```

The greedy chain loses ceiling permanently because it applies mechanisms in a fixed order
and destroys prerequisites; ARGUS refuses those orderings *in model space* and keeps a
population so a bad bet is never the parent of the next step.

---

## Quickstart (no GPU, no LLM)

```bash
cd argus
export PYTHONPATH=src
python3 -m argus.cli selftest                 # consistency
python3 -m argus.cli hacklab --seed 1         # CudaForge gate vs ARGUS, on real kernels
python3 -m argus.cli attrib                   # attribution + pre-screen consistency
python3 -m argus.cli ab --seeds 6 --evals 40  # verification + search ablation
python3 -m argus.cli dryrun                   # production adapter, no GPU
bash scripts/local_validate.sh                # all of the above + 53 unit tests
```

No GPU, no torch, no third-party packages beyond `numpy`; Python ≥ 3.9.

---

## Layout

```
argus/
  README.md                 this file
  ARCHITECTURE.md           CudaForge reproduction map + the ARGUS architecture
  docs/
    LEADERBOARD_ANALYSIS.md the SOL objective, the live board snapshot, and what it implies
    IMPROVEMENTS.md         ranked improvements, evidence, and honest scope notes
    B300_RUNBOOK.md         the candidate GPU plans, canary protocol, checklists
  data/board_snapshot.json  live public snapshot (fetch_board.py)
  src/argus/                the code (see the table above)
  experiments/              the three local experiments
  tests/                    53 CPU tests
  scripts/                  bootstrap.sh, local_validate.sh, fetch_board.py
  vendor/                   upstreams, cloned at pinned commits (gitignored)
```

## What is validated where

* **Validated on CPU** — everything above: the verifier against real hacked kernels, the
  attribution and pre-screening logic, the search A/B on a synthetic-but-rugged landscape,
  and the whole reasoning stack's unit tests.
* **GPU-pending** — real kernel latency, the live CudaForge loop under real agents, and the
  board's clock-locked B200 measurement. `backends/cudaforge.py` is exercised in `--dry-run`
  only; `docs/B300_RUNBOOK.md` lists the plans to run on hardware, ordered by
  risk-adjusted value, *before* any resource is requested.
