# Experimental protocol: planned GPU studies

The local experiments in [`../experiments/`](../experiments) falsify the two structural claims
on a synthetic landscape, where they are necessarily claims about *logic*. This note turns each
one into a measurement on hardware, states what would refute it, and orders the studies so that
one allocation of GPU time answers every question the local work raises. Nothing here has been
run yet; the studies are written down first, deliberately.

Two constraints fix the design. The metric is measured on **B200 at a locked SM clock
(1500 MHz)**, so **B300 is used for generation and iteration while the reported numbers come
from the B200 protocol** (the SOL-ExecBench harness, or the submission path). And an evaluation
is the scarce resource, so every study is costed in evaluations rather than in wall-clock hours.

The studies are nested (A ⊂ B ⊂ C ⊂ D), so stopping at any tier still yields a coherent result;
the order to spend time in is at the end.

---

## 0. Ground rules

1. **Refresh first.** `python3 scripts/fetch_board.py` (the snapshot rotates); re-derive anchors
   from the vendored dataset (`<collection>/<kernel>/workload.jsonl`) — the public API does not
   expose `T_SOL`/`T_b`.
2. **Pick one collection to start.** FlashInfer-Bench (id 4, 26 kernels) is the smallest and
   the one with published baselines; L1 (94) is larger and more varied.
3. **Verify before you score.** Every candidate goes through `verifier.verify_kernel`; a
   candidate that does not verify contributes nothing and is never promoted or submitted.
4. **Log the counters.** Every evaluation must record the counter dict the fingerprint audit
   needs (`dram__bytes_*`, `sm__inst_executed.sum`, tensor pipe, stalls).
5. **Never fit the tuned shape.** All workloads of a definition are evaluated, always.

## 1. Environment checklist (GPU box)

```bash
# 1. upstreams
argus/scripts/bootstrap.sh                       # CudaForge @ eaba693f, SOL-ExecBench @ a9fa0804
# 2. dataset (authoritative anchors)
git lfs install && GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/nvidia/SOL-ExecBench
# 3. CudaForge env (needs python>=3.11; see vendor/CudaForge/environment.yml)
# 4. prerequisites the paper names
nvidia-smi                                       # driver >= 580; lock clocks for B200 runs
nvcc --version && ncu --version                  # must match
sudo -v                                          # NCU needs elevated counter access
python3 -c "import torch,ninja; print(torch.__version__, torch.cuda.is_available())"
```

Write the live agent runner the adapter expects (`vendor/CudaForge/argus_runner.py`, a thin
wrapper over upstream `agents/query_server.py`) — the only not-yet-written glue; the dry-run
shows the exact prompt contracts it must honour (`python3 -m argus.cli dryrun`).

## 2. Cost model (estimates — refine on the first real task)

One *evaluation* = compile + correctness over all workloads + (optionally) one NCU profile +
timing (board protocol: 10 warmup + 50 iterations, L2 flushed, mean of 3 trials).

| step | est. time |
|---|---|
| compile + correctness (one candidate) | 20–60 s |
| NCU profile (one hot kernel) | 30–120 s |
| board-protocol timing (one candidate) | 10–30 s |
| one full candidate (compile+correct+time) | **~1 min** |
| one candidate with a profile | **~2–3 min** |

So a 40-eval campaign on one kernel ≈ **40–90 min**; the whole FlashInfer collection at 40
evals/kernel ≈ **25–40 GPU-hours**. Plans are costed in evals, not "$".

## 3. The studies

All plans reuse the same two agents; they differ only in the boxed component. `E` = evals per
kernel. The **falsification criterion** is what must be true to abandon the tier.

### Study A — verification alone. **Run this first.**
* **Change vs CudaForge:** verification (C1) + the board objective. The loop stays the naive
  greedy chain otherwise. This is the minimal, highest-confidence intervention.
* **Hypothesis:** most of the gap to a *scored* submission is that hacks and shape-specialised
  kernels get zeroed; gating them converts a worthless submission into a scored one.
* **Budget:** E=20–30 on 3–5 representative kernels ≈ **2–4 GPU-hours**.
* **Simulated result:** `argus.cli hacklab` (0/4 hacks pass) and `argus.cli ab` claim (i)
  (0.00 → 0.65 effective).
* **Deliverable:** a verified kernel per trial kernel + the hack families ARGUS blocks in
  practice. **Falsified if** ARGUS blocks a kernel that the real harness *accepts* (a false
  positive) — then tighten the contract before scaling.

### Study B — A + population search (C3), and the prerequisite probe
* **Change:** replace the chain with `search.PopulationSearch` (UCB, prereq-filtered arms,
  multi-fidelity, regression-proof parent).
* **Hypothesis (H2):** the chain locks itself out of the ceiling; a population of architecture
  bets with prerequisite-aware ordering reaches it. The *prerequisite probe* is the part that
  would refute the table: for each declared prerequisite, apply the mechanism out of order and
  measure whether the ceiling actually drops irreversibly.
* **Budget:** E=40 on the same kernels ≈ **4–8 GPU-hours**.
* **Simulated result:** `argus.cli ab` claim (ii) — 0.7942 vs 0.9412 at equal budget.
* **Deliverable:** per-kernel curve (verified best vs evals) for chain vs population, and a
  measured verdict per declared prerequisite (irreversible / recoverable / commutative).
  **Falsified if** the chain is never meaningfully worse on real kernels (i.e., real mechanisms are
  commutative and the prereq table is over-modelled) — then drop the arm filter and keep UCB.

### Study C — B + causal attribution for the judge (C2)
* **Change:** the Judge reads `attribution.build_judge_block` (bottleneck + delta attribution +
  bounded levers) instead of the raw counter dump; pre-screening refuses doomed mechanisms.
* **Hypothesis:** a causal, ceiling-bounded lever list yields better proposals **per eval**, so
  fewer evals for the same score.
* **Budget:** E=40 with profiles on the same kernels ≈ **6–10 GPU-hours**.
* **Simulated result:** `argus.cli attrib` (attribution 3/3; 2/7 mechanisms pre-screened out).
* **Deliverable:** evals-to-target with causal vs raw Judge context. **Falsified if** the causal
  block does not beat the dump on evals-to-target.

### Study D — C at corpus scale
* **Change:** run the whole chosen collection with ARGUS end-to-end, using
  `schedule.plan_eval_budget` and `memory.MechanismMemory` (cross-kernel priors), submitting
  per kernel.
* **Hypothesis:** the collection mean beats the published baselines once every kernel is
  verified and the near-roof kernels are not over-invested in.
* **Budget:** FlashInfer (26 kernels) × E=40 ≈ **25–40 GPU-hours**; L1 (94) ≈ **90–140**.
* **Deliverable:** a per-kernel verified solution set + the projected collection score from the
  anchors + the actual board result after submission.
* **Falsified if** the projected collection score does not exceed the published reference
  score for the same collection-and-stack, or the per-kernel measurements disagree with the
  local projection beyond the stated noise floor.

### Study E — stretch: fusion-decision search
* **Change:** search over fusion decisions (which neighbouring ops to absorb), with a
  fusion-correctness checker (`docs/design/findings.md` #8).
* **Hypothesis:** the real ceiling beyond mechanism tuning is structural.
* **Budget:** open-ended; only start if D is close and hours remain.
* **Deliverable:** a fusion-decision search prototype + correctness checker.

## 4. Calibration submission (do before Study D)

1. Take one kernel that is **already strong** (high published score).
2. Generate one ARGUS solution, verify it, and **submit it** to the board as a canary.
3. Confirm on the board's own evaluation: (a) it is accepted, (b) the SOL score is close to
   ARGUS's local projection, (c) the B200 measured latency matches expectations (clock lock,
   L2-flush, tolerance behaved).
4. Only then run Plan D. If the canary's board score disagrees with the projection, stop and
   reconcile the harness configuration **before** spending campaign hours.

## 5. Spend order (if the grant is limited)

| grant | do this |
|---|---|
| ≤ 4 GPU-h | **Plan A + canary** on 3–5 kernels. |
| ≤ 12 GPU-h | A then **B** (chain vs population curve) on the same kernels. |
| ≤ 25 GPU-h | A, B, **C** (causal Judge ablation) — a clean 3-way comparison. |
| ≤ 40 GPU-h | + **D** on FlashInfer-Bench end-to-end, submit per kernel. |
| more | **D** on L1/L2/Quant; then **E** if D is near the top. |

## 6. What to report back (so the next session can iterate)

Per kernel: the verified solution, its measured latency per workload, the projected per-kernel
`S_k`, the `Fast1` count, the mechanism lineage (which mechanisms were accepted), the
fingerprint flags, and the evals spent. Per campaign: the projected vs actual collection score
and the collection score after submission.

## 7. Risk register

| risk | mitigation |
|---|---|
| B300 timing not comparable to B200 anchors | measure the final set under the board protocol; never score B300 vs B200 anchors |
| NCU counters need elevated privileges | checklist step 4; fall back to latency-only (loses fingerprint) if unavailable |
| the live agent runner is unwritten | write `argus_runner.py` from the dry-run prompt contracts first |
| verification false positive blocks a good kernel | Plan A falsification criterion; contract is per-operator and reviewed |
| prereq table wrong for sm100 | Plan B falsification criterion; drop the arm filter if real mechanisms are commutative |
| collection score moves | refresh snapshot; compare against the live board at submission time |
