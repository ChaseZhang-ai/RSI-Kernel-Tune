# Improvement designs, ranked, with evidence

Status tags:

* **[VALIDATED]** — implemented and checked locally (unit tests and/or the dataset).
* **[MOCK]** — implemented; its benefit is demonstrated under the synthetic GPU (falsifiable,
  reproducible via `--seed`), but the magnitude must be re-measured on real hardware.
* **[IMPLEMENTED, GPU-PENDING]** — implemented and unit-tested; the real effect needs a GPU run.
* **[DESIGNED]** — specified here, not yet coded.

The ranking is by expected leaderboard impact per unit of effort, given the analysis in
[`LEADERBOARD_ANALYSIS.md`](LEADERBOARD_ANALYSIS.md).

---

## 1. [VALIDATED] Leaderboard-exact objective and per-kernel latency targets

**Mechanism.** Optimize the board's actual scalar — the mean of per-kernel SOL scores — and turn a
score target into the `[target] ms` the upstream loop gates on (`objective.latency_for_score`).

**Why it matters.** Every other kernel-agent optimizes speedup or percentage-of-baseline. The board
does not. Targeting the wrong scalar wastes the budget by construction.

**Evidence.** The formula reproduces the published collection scores to six decimals
(`tests/test_dataset.py`). Target inversion round-trips (`tests/test_objective.py`).

**Effort/risk.** Trivial / none.

---

## 2. [VALIDATED] Best-of-breed per-kernel orchestration

**Mechanism.** Set each kernel's target to the best public score for that kernel
(`plan --target best`), i.e. reproduce the best-known architecture per kernel rather than a single
uniform recipe.

**Why it matters.** The published per-kernel numbers show the *oracle* of "best public kernel per
kernel" equals **0.819011**, already above rank 1's 0.815321. The work is orchestration: route each
kernel to the architecture class that other leaders used for it.

**Evidence.** `tests/test_dataset.py::test_best_beats_rank_one_competitor`;
`plan --target best` prints the per-kernel targets that realize it.

**Effort/risk.** Low / the risk is copying public solutions rather than generating them. Research
claim should be "matches then exceeds best-known", not "takes others' kernels"; use the best-known
*structure* as a target, generate our own implementation.

---

## 3. [MOCK] Architecture portfolio with successive halving

**Mechanism.** Have the architecture agent propose `K` structurally **different** `M0`s (CTA-per-tile,
persistent-CTA, warp-specialized, split-K, register-resident, fused-epilogue). Score each cheaply
(1 h), keep the better half, re-score at 3 h, then 7 h, then commit the remaining budget to the
survivor (`portfolio.successive_halving`; `portfolio.UCB1` for a pure bandit variant).

**Why it matters.** ForgeKernel's own stated limitation is settling into a local optimum because
`M0` freezes early. A single self-evolution chain is one gamble on one `M0`. A portfolio converts
that into best-arm identification, which is near-optimal and bounded.

**Evidence.** Local A/B (200 GPU-hours, seed 7): portfolio-only lifts the collection score by
**+0.0375** over uniform — the largest single contribution in the ablation.

**Effort/risk.** Medium (more backend calls: 179 vs 26 in the mock) / risk is budget fragmentation;
successive halving bounds it, but `eta` and the budget ladder must be tuned to the real cost of a
short run.

---

## 4. [VALIDATED] Fast₁ straggler triage

**Mechanism.** Rank the workloads that are not yet strictly faster than `T_b`, and repair them
first; `allocator(fast1_weight>0)` biases hours toward them and `curriculum` surfaces them.

**Why it matters.** Fast₁ is a first-class leaderboard column and ForgeKernel's 13 misses are all in
four GEMM definitions (`009`: 8, `007`: 3, `010`: 1, `011`: 1). `009_gemm_n5120_k2048` is
simultaneously the largest Fast₁ gap and the largest score headroom — one kernel, two metrics.

**Evidence.** `stats`/`plan` output and `tests/test_allocator.py::test_fast1_weight_changes_allocation`.

**Effort/risk.** Low / `009`'s low `S` (0.53) may reflect a genuinely hard small-shape GEMM (SOL
bound optimistic for small `N`), in which case Fast₁ repair is achievable before score repair — and
that alone matches rank 1's count.

---

## 5. [MOCK] Score-aware hour allocation (exact DP)

**Mechanism.** Maximise the projected collection score over integer hour-slots with an explicit,
bounded latency-improvement model; exact multiple-choice-knapsack DP, with a marginal-greedy fast
path. Baselines `uniform` and `headroom` are provided.

**Why it matters.** With equal per-kernel weight and concave `S`, the marginal value of an hour
differs sharply across kernels. The DP provably beats uniform and headroom *under the model*.

**Evidence.** `tests/test_allocator.py` (DP ≥ greedy ≥ uniform on 25 random instances; DP beats
uniform on lopsided cases and respects the budget). In the mock A/B, allocation-only adds
**+0.0117** over uniform — real but second-order to the portfolio.

**Effort/risk.** Low / the model's `rate` and `floor_frac` are priors; re-fit from logs (item 12)
before trusting absolute projections. The *ranking* it produces is robust; the *number* is not.

---

## 6. [IMPLEMENTED, GPU-PENDING] Executable-predicate knowledge graph with negative and analog memory

**Mechanism.** Replace prose knowledge entries with typed entries whose `applicability` is a
mapping `feature -> (low, high)` evaluated against measured counters/features. Only `certified`
entries are prescribed; `refuted` entries are returned by `negative_matches` so the agent is told
"this was tried and regressed under exactly these conditions". `lift_analog` offers a mechanism from
another GPU architecture as a **candidate** that must be probed before it can be prescribed.

**Why it matters.** Upstream retrieval is qualitative ("memory-bound"), so wrong mechanisms get
prescribed; and failed ideas are not first-class memory, so they get re-tried. Executable
predicates make retrieval a *filter*; negative memory makes failures reusable; analog lifting turns
an SM90 finding into a bounded SM100 hypothesis without pretending it is certified.

**Evidence.** `tests/test_kb_graph.py` (matching, unknown-feature rejection, same-arch-only
retrieval, negative separation, analog-is-candidate, JSONL round-trip).

**Effort/risk.** Medium / needs a written adapter to upstream `hw_kb/<lang>/<arch>/` markdown
(item 15) so the agents actually read these entries at run time.

---

## 7. [IMPLEMENTED, GPU-PENDING] Precision-contract guard, adversarial probes, reward-hack scan

**Mechanism.** Declare the precision contract an operator needs (from its tolerance and class);
reject any candidate whose accumulation precision is below it **even if its elementwise error
passes**; run adversarial probes that expose approximation shortcuts; statically scan for the
classic cheats (calling the reference, returning the input, identity caching, in-kernel downcast,
clock/L2 tampering).

**Why it matters.** The upstream ablation is a live warning: an attention-backward kernel without
the milestone template *passed the output check but lowered internal precision* and scored +16.2%.
On the board, that is fragile — SOL enforces per-workload tolerance and a `max_error_cap`, and one
failing workload zeroes its kernel (geometric mean) and loses its Fast₁ credit. Accuracy must be a
hard constraint, not a speed knob.

**Evidence.** `tests/test_precision_guard.py` (accum-downgrade flagged, exact-op equality, output
narrowing, probe coverage, cheats detected, gate verdicts).

**Effort/risk.** Low-medium / static scan is heuristic (advisory); the contract table needs
per-operator review.

---

## 8. [IMPLEMENTED] Statistical promotion gate

**Mechanism.** Promote a candidate reference only when the lower end of the paired-ratio bootstrap
CI exceeds `1 + margin`, with `margin` auto-set from the measured noise floor.

**Why it matters.** The self-evolution chain must never promote a reference that only *looks*
faster; a noisy promotion poisons every later generation. Pairing (candidate and reference timed in
the same job) cancels shared drift.

**Evidence.** `tests/test_stats.py` (clear gain accepted, identical rejected, sub-margin rejected,
noise-floor behaviour).

**Effort/risk.** Trivial / needs a handful of paired repeats per acceptance, which costs GPU time.

---

## 9. [IMPLEMENTED] Shape clustering (exact 1-D k-means)

**Mechanism.** Group a definition's workloads into regimes (exact Jenks DP, deterministic, no
initialisation randomness) and tune one kernel per regime.

**Why it matters.** It is on upstream's own v1.5 roadmap; a GEMM definition has 25–43 shapes, and
an architecture is usually shared within a cluster. Amortising the search is the cheapest real win.

**Evidence.** `tests/test_cluster.py` (group recovery, `k` selection, weights, representative is a
real shape).

**Effort/risk.** Low-medium / wiring per-cluster targets into multiple upstream tasks/workloads.

---

## 10. [IMPLEMENTED] Roofline → bottleneck → lever bridge

**Mechanism.** Convert counters (`dram_bytes`, `flops`, time) into achieved bandwidth/flops,
arithmetic intensity and a bottleneck class, and emit the lever list for that class — and the
features (`dram_pct`, `tensor_pct`, `bytes_per_flop`) that `kb_graph` matches on.

**Why it matters.** It is what makes the knowledge-predicate system *measurable* rather than
rhetorical: the features are computed the same way on every run.

**Evidence.** `tests/test_roofline.py` (memory/compute/latency classification, feature presence).

**Effort/risk.** Low / must feed real NCU counters in; peaks are approximate and should be replaced
by whatever the board's `T_SOL` implies.

---

## 11. [IMPLEMENTED] Family-seeded curriculum

**Mechanism.** Order kernels by value × family; attempt a family's easy members first so the
mechanisms they certify seed its hard members.

**Evidence.** `tests/test_curriculum.py`. **Effort/risk.** Low.

---

## 12. [DESIGNED] Online re-fit of the improvement-rate model

Replace `rate`/`floor_frac` priors with per-family rates estimated from observed
latency-vs-hours; then the allocator's absolute projection becomes trustworthy and it can re-plan
mid-campaign.

## 13. [DESIGNED] Graph-level fusion search

The real prize beyond orchestration: search over fusion *decisions* (which neighbouring ops to
absorb), not just implementations. Highest ceiling, highest risk; needs the graph IR and a fusion
correctness checker.

## 14. [DESIGNED] Canary submission protocol

Submit one or two already-good kernels first to de-risk the submission path (clock lock, tolerance,
anti-cheat, the B200-vs-B300 measurement question) before committing a full campaign.

## 15. [DESIGNED] `kb_graph` → upstream `hw_kb` adapter

Emit certified entries as `hw_kb/<lang>/<arch>/*.md` so the upstream agents read VULCAN's
mechanisms during a run. Small but necessary for item 6 to pay off end-to-end.

---

## Threats to validity

* **Mock magnitude ≠ real magnitude.** The synthetic GPU validates the *logic* (portfolio ≥
  allocation ≥ uniform) and nothing about absolute kernel speed. Real lifts must be measured on the
  board's hardware and clock.
* **Allocation is model-bound.** Its edge depends on the improvement model; the *ranking* it
  produces survives bias in the model parameters, the projected *score* does not.
* **Public numbers move.** The snapshot must be refreshed (`fetch_board.py`); rank 1 has changed
  once already during this work.
* **Copying vs generating.** Beating the board by re-submitting competitors' best structures is not
  a research result. The claim to hold is "generate then exceed", with best-known as the target —
  not the artifact.
