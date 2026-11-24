# Improvement designs over CudaForge, ranked

Status tags: **[VALIDATED]** implemented and checked locally (tests + experiment);
**[IMPLEMENTED, GPU-PENDING]** implemented and unit-tested, real effect needs a GPU;
**[DESIGNED]** specified, not coded.

Ranking is by expected leaderboard impact per unit of effort, given
[`LEADERBOARD_ANALYSIS.md`](LEADERBOARD_ANALYSIS.md). The guiding fact: on the board a kernel
that fails verification scores **0**, so a generator that optimises raw speedup is optimising a
proxy that the board does not pay for.

---

## 1. [VALIDATED] Board-faithful, hack-resistant verification (C1)

**Mechanism.** Replace CudaForge's single `allclose(atol=rtol=1e-3)` on one shape with the SOL
element rule over *all* workloads, plus static scan, differential probes (memoisation,
identity, degenerate variance, determinism), a float64 precision contract, metamorphic probes,
and a counter fingerprint.

**Why it matters.** CudaForge's objective is raw speedup and its gate is blind to the four
hack families most likely to zero a submission. Local evidence (`argus.cli hacklab`): all 4
hacks pass CudaForge's gate, 0 pass ARGUS; the honest kernel passes both. Folding the gate
into the loop at equal budget (`argus.cli ab`): effective board score **0.0000 → 0.6472**
(mean over 6 seeds).

**Effort / risk.** Medium / low. The static scan is heuristic (advisory); the destructive parts
(tolerance, probes, precision) are numeric and deterministic.

---

## 2. [VALIDATED] Prerequisite-aware population search (C3)

**Mechanism.** Replace the round-robin chain (parent = the latest candidate) with K lineages,
UCB over `(lineage, mechanism)`, an arm filter that refuses mechanisms a lineage is not yet
ordered to apply (`roofline.PREREQS`), multi-fidelity verification-gated promotion, and a
regression-proof parent (the lineage *head*).

**Why it matters.** Two independent failures of the chain: it derails (a bad step becomes the
parent) and it can permanently destroy ceiling by applying a mechanism before its prerequisite.
Local A/B on a rugged landscape, equal budget, verification on for both, mean over 6 seeds:
greedy **0.7942** vs ARGUS **0.9412** (= the landscape optimum, every seed).

**Effort / risk.** Medium / needs the mock's prerequisite structure to be a faithful analogue
of real ISA/layout ordering (it is declared, not fitted). On real hardware the prereq table must
be reviewed per architecture.

---

## 3. [VALIDATED] Causal attribution + model-based pre-screening (C2)

**Mechanism.** Turn counters into a bottleneck, a delta attribution with an
`unexplained_speedup` flag, and a bounded lever list; refuse mechanisms the model says cannot
pay off before spending a GPU eval.

**Why it matters.** CudaForge's Judge reads a raw 24-counter dump with no causal statement. The
model both (a) gives the Judge the next lever *and its ceiling*, and (b) saves evaluations:
on a DRAM-bound kernel `argus.cli attrib` refuses 2/7 mechanisms (`tensor_core`, `split_k`)
and the attribution classifier agrees with the constructed ground truth 3/3.

**Effort / risk.** Low-medium / the peak constants are priors; the *ranking* is robust, the
absolute bound is not. Fit peaks per device.

---

## 4. [IMPLEMENTED, GPU-PENDING] Board objective as the promotion signal

**Mechanism.** Score candidates by the SOL score (from the anchors), not `ref/test`.
**_Why._** It is the graded metric; raw speedup is not.
**_Effort / risk._** Trivial / needs anchors from the dataset (see the analysis doc).

## 5. [IMPLEMENTED, GPU-PENDING] Cross-task mechanism memory

**Mechanism.** Persist per-task mechanism outcomes keyed by executable conditions
(bottleneck class + counters) and feed the UCB selector a prior; record refuted mechanisms.
**_Why._** A later task should not re-discover that `tensor_core` regressed under a memory
ceiling. **Scope note:** this is *lighter* than the sibling track's certified knowledge graph —
it stores outcome statistics, not certified entries. Keep them separate.
**_Effort / risk._** Low / the conditioning features must be computed identically every run.

## 6. [IMPLEMENTED] Lightweight eval-budget planner

**Mechanism.** Marginal-greedy allocation of a global eval budget by `(1 - S) x feasibility`.
**Scope note:** the sibling track owns the exact score-aware allocation (its stated contribution)
— ARGUS keeps only enough to not waste the scarce resource.
**_Effort / risk._** Low.

## 7. [DESIGNED] Measured-peak fit

Replace `GpuPeak` priors with per-device measured DRAM/Tensor peaks (a microbenchmark kernel
pair), so the pre-screen bounds become trustworthy and transfer across B200/B300.

## 8. [DESIGNED] Fusion-decision search (graph level)

The real ceiling beyond mechanism tuning: search over *which neighbouring ops to absorb*, with a
fusion-correctness checker. Highest ceiling, highest risk; needs the graph IR.

## 9. [DESIGNED] Canary submission protocol

Submit one already-strong kernel first to de-risk the path (clock lock, tolerance, screening,
B200-vs-B300), before committing a campaign. In [`B300_RUNBOOK.md`](B300_RUNBOOK.md).

---

## Threats to validity

* **Mock magnitude != real magnitude.** The synthetic landscape validates the *logic*
  (verification wins; prereq-aware population >= chain) and nothing about real kernel speed.
* **Prereq table is declared, not learned.** It encodes known architecture ordering; wrong
  entries would mis-filter arms.
* **The verification win is conditional on the hack families being realistic.** They are drawn
  from SOL's own flagged classes and the ForgeKernel ablation, but a new hack class would need
  a new check.
* **Board numbers move.** Refresh the snapshot; re-derive anchors from the dataset.
* **Copying vs generating.** The claim to hold is "generate then verify then exceed", not
  re-submitting a competitor's kernel.
