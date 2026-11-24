# Design: the reproduced loop and the three additions

**Base system:** CudaForge — `https://github.com/OptimAI-Lab/CudaForge`, pin
`eaba693fa01bdafc9848334cada2d789e240ce3f` (MIT, © 2025 Zijian Zhang).
**Paper:** *CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel
Optimization*, arXiv:2511.01884.
**Vendored at:** `vendor/CudaForge/` (gitignored; created by `scripts/bootstrap.sh`).

ARGUS does **not** fork CudaForge. It reproduces its loop faithfully behind a driver seam,
so the same reasoning stack can run against a deterministic CPU mock (local validation) or
against the real CLI (GPU). This document is the paper→code map, the reproduction procedure,
and the ARGUS architecture on top.

---

## 1. What CudaForge is (reproduced)

A **training-free two-agent loop** for KernelBench-style tasks:

* a **Coder** writes CUDA C++ for the module;
* a **Judge** reads the Nsight-Compute metrics and prescribes the single highest-impact
  optimisation;
* the two iterate, with a repair path when a candidate does not run.

### The per-round state machine (upstream `main.py`)

```
round 0            seed prompt -> Coder -> compile -> correctness -> timing
round i, invalid   problem-identify prompt -> Coder(repair) -> ... -> timing
round i, runnable  NCU profile -> Judge(one optimisation, JSON) -> Coder(apply) -> timing
```

### What is reproduced, and where

| CudaForge behaviour | upstream | ARGUS reproduction |
|---|---|---|
| two-agent loop, per-round state machine | `main.py::_run_single_task` | `src/argus/cudaforge.py::run_cudaforge_loop` (+ `search.greedy_chain`) |
| **single greedy chain** (parent = the *latest* candidate) | `main.py` (`current_kernel = ind`) | `search.greedy_chain` — deliberately un-improved, used as the control |
| objective = raw speedup `ref/test` | `main.py` (`metrics["score"] = speedup`) | `EvalResult.score` in CudaForge mode |
| **one-shape `allclose`** correctness | `utils/compile_and_run.py` (`torch.allclose(ref,test,atol=tol,rtol=tol)`, default `tol=1e-3`) | `backends/mock.py::_allclose_gate` (reproduces it exactly for the A/B) |
| 24 Nsight-Compute metrics | `run_ncu.py::METRICS` | `cudaforge.CUDAFORGE_METRICS` (byte-for-byte list) |
| counters → prompt (raw JSON dump) | `run_ncu.py::metrics_to_prompt` + `prompts/judger_optimization.py` | CudaForge mode builds the raw dump; ARGUS mode builds the causal block (`attribution.build_judge_block`) |
| Judge returns one optimisation as JSON | `prompts/judger_optimization.py` | `cudaforge.JUDGE_SYSTEM` (contract preserved) |
| repair path (problem-identify → rewrite) | `prompts/error.py` | `cudaforge.build_repair_prompts` |
| candidate bookkeeping | `scripts/individual.py::KernelIndividual` | `search.Candidate` |
| code/JSON extraction | `utils/kernel_io.py` | `cudaforge.extract_code_block/extract_json/extract_cuda_kernel_names` |

**Rewritten, not copied:** prompt *text* (the *contracts* are preserved and asserted in
`tests/test_cudaforge_repro.py`). `run_ncu.py` wrapping (subprocess plumbing) is reproduced
in `backends/cudaforge.py`. Nothing outside `argus/` is touched.

### Reproduction procedure (GPU-free parts)

```bash
argus/scripts/bootstrap.sh                  # clones CudaForge @ eaba693f into vendor/
python3 -m argus.cli dryrun                 # exercises the adapter's prompts/parsers
```

The GPU parts (conda env, `ncu` running with elevated counters, one real task) are in
`docs/experiments.md`.

---

## 2. What ARGUS adds, on the same seam

Everything side-effecting is behind `backends/base.py::CampaignBackend`
(`seed` / `propose` / `eval_kernel` / `profile` / `peak` / `flops`). ARGUS swaps the two
places where CudaForge is weak, and leaves the rest:

```
                 CudaForge                              ARGUS
                 ---------                              -----
seed        one architecture bet            K lineages, distinct architecture bets
propose     Coder applies the Judge's JSON  same, but the Judge block is CAUSAL (C2)
select      round-robin from the LATEST     UCB over (lineage, mechanism), arm-filtered by the
                                            model's prerequisites (C3); parent = lineage HEAD
evaluate    one-shape allclose + timer      full verification stack (C1) at two fidelities
objective   ref/test speedup                board SOL score; rejected candidates score 0
accept      best-so-far checkpoint          verified + fingerprint-clean + head promotion
```

### C1 — verification (`verifier.py`, `probes.py`, `fingerprint.py`)

A layered gate. **Hard** checks gate acceptance; **soft** checks only flag.

1. static scan for known cheat classes (`verifier.HACK_PATTERNS`);
2. SOL element rule `|o-r| <= atol + rtol*|r|` with `matched_ratio >= 0.99` over **all**
   workloads (`verifier.check_tolerance`);
3. differential probes: determinism, **memoisation** (perturbed inputs), identity,
   degenerate variance, shape/dtype/finiteness;
4. **precision contract** vs a float64 oracle: a candidate whose output passes tolerance but
   whose error against fp64 is far above the reference's own error is an *internal-precision
   downgrade*;
5. **metamorphic probes** (`probes.py`): scale-invariant RMSNorm, shift-invariant softmax,
   homogeneous GEMM, row-permutation equivariance;
6. **counter fingerprint** (`fingerprint.py`) over the run trajectory: a latency drop that no
   physical counter can explain, achieved BW above peak, vanished stores, or a tensor-pipe
   jump with a shrinking accuracy margin.

### C2 — attribution (`attribution.py`, `roofline.py`)

The Judge no longer reads a wall of percentages:

* counters → an achieved (bytes/s, flops/s) point and a **bottleneck class**;
* **delta attribution** between consecutive candidates: the resource change that explains
  the latency change, plus a `consistent` flag and an **`unexplained_speedup`** flag when the
  latency moved but no counter did;
* a **bounded lever list** with an explicit `<= hi x` ceiling per mechanism;
* **model-based pre-screening**: mechanisms whose bound cannot pay off are refused *before* a
  GPU evaluation is spent. Prerequisite table (`roofline.PREREQS`) feeds the search's arm
  filter.

### C3 — search (`search.py`)

`PopulationSearch`: K lineages; UCB1 over `(lineage, mechanism)`; arms filtered by
`roofline.prereq_satisfied`; successive-halving over lineages (`prune_margin`,
`diversity_floor`); multi-fidelity (cheap screen, then a board-protocol confirm only for a
candidate that verifies and beats its parent). **The parent is always the lineage head** —
a regression can never become the parent, so the derail bug is structurally impossible.

`greedy_chain` reproduces the upstream chain next to it, for the A/B.

---

## 3. Module map

```
src/argus/
  objective.py     SOL algebra (shared substrate)
  roofline.py      peak presets, achieved point, bottleneck, mechanism bounds, PREREQS, prescreen
  attribution.py   stall profile, delta attribution, lever list, judge block
  verifier.py      tolerance rule, adversarial probes, precision contract, static scan, orchestrator
  probes.py        metamorphic laws per operator class
  fingerprint.py   counter-fingerprint audit + hack-risk score
  search.py        PopulationSearch (UCB, multi-fidelity, regression-proof) + greedy_chain control
  memory.py        cross-task mechanism memory (outcomes + refuted), keyed by executable conditions
  schedule.py      lightweight eval-budget planner
  cudaforge.py     reproduction: metrics, prompt contracts, parsers, both loops
  backends/
    base.py        CampaignBackend protocol
    mock.py        synthetic GPU + synthetic counters + REAL verification (the local testbed)
    cudaforge.py   production adapter to the upstream CLI (dry-run validated)
```

## 4. Known gaps / caveats

* The mock's *magnitude* is synthetic; it validates the *logic* (verification wins,
  prereq-aware population ≥ chain) and nothing about real kernel speed. Real lifts must be
  measured under the board's protocol.
* `flops` and peak constants are priors; replace with measured peaks (`GpuPeak.from_file`).
* The production adapter's live paths (agent call, `compare_and_bench`, NCU) are written but
  only dry-run-validated; `docs/experiments.md` carries the verification checklist.
* The board scores on **B200 at a locked clock**; B300 is for generation/iteration. Do not
  score B300 timings against B200 `T_b`/`T_SOL`.
