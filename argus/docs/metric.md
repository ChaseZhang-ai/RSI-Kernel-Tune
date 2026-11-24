# The objective we study: the metric, its anchors, and what they imply

This note fixes the metric that every experiment in this repository reports against.
Everything here is either (a) the published scoring algebra, or (b) a fact pulled by
`scripts/fetch_board.py` from the public board API. Numbers I could not re-verify this
session are labelled **[external]**.

---

## 1. The scoring rule

```
per workload   S_w = 1 / (1 + (T_w - T_SOL,w) / (T_b,w - T_SOL,w))      clamped at 1
per kernel     S_k = geometric mean over the kernel's workloads of S_w
collection     S   = unweighted arithmetic mean over the kernels of S_k
```

* `S = 0.5` when a workload exactly matches the **Scoring Baseline** `T_b`; `S -> 1` as it
  approaches the **Speed-of-Light** bound `T_SOL`; a workload **below `T_SOL` is clamped**, so
  a timing that appears to beat the roof is treated as a measurement artefact, not a win.
* `Fast1` counts workloads *strictly* faster than `T_b` (a separate, first-class column).
* Correctness is a hard gate: a kernel that fails tolerance scores **0** for that kernel.

`argus/objective.py` implements this exactly and is unit-tested for the anchors, the inverse
(`latency_for_score`), the aggregation identities, and convexity.

**Where the anchors live.** `T_SOL` and `T_b` are **not** exposed by the public API routes
(verified this session: `/kernels` returns only id/name/tags/submission_count;
`/kernels/{id}` returns the definition metadata but no anchors). They are published inside
the harness dataset — `NVIDIA/SOL-ExecBench`, `<collection>/<kernel>/workload.jsonl` — which
is what the board itself scores with. So the pipeline is: vendor the dataset
(`scripts/bootstrap.sh`), read the anchors per workload, and invert the formula to get a
per-workload target latency (`objective.latency_for_score`).

## 2. The live snapshot (fetched this session)

`data/board_snapshot.json` (`scripts/fetch_board.py`):

| collection | id | kernels | submissions |
|---|---:|---:|---:|
| L1 — Single Operations | 1 | 94 | 20,693 |
| L2 — Fused Operations | 2 | 82 | 14,765 |
| Quantization | 3 | 33 | 9,468 |
| FlashInfer-Bench | 4 | 26 | 10,991 |
| **total** | | **235** | **55,917** (775 users) |

Hardware: **B200** for every collection (the live board has no B300 track).

**[external]** The specific per-collection SOL scores circulating for the FlashInfer-Bench
collection (rank 1 ≈ 0.8153, the ForgeKernel reproduction ≈ 0.8152 with `Fast1` 647/660) come
from the ForgeKernel paper / the sibling track's earlier snapshot and are **not** reproduced
here; the board's SOL-score view is a different endpoint from the ones public this session
(the rank table I could fetch lists *kernels solved*, not scores). Treat any single number as
a moving target and re-derive it from the anchors.

## 3. What the objective implies for strategy

1. **Every kernel counts the same**, regardless of its workload count. The score lives in the
   mid/low kernels; further work on the near-`S≈0.99` kernels is nearly free of value.
2. **A zeroed kernel is catastrophic.** One failed workload zeroes its kernel's score (and its
   `Fast1` credit). With ~1/26th of the collection riding on each kernel, a reward-hacked
   kernel that the board rejects costs as much as never writing it. *This is why ARGUS couples
   the objective to a hard verifier rather than to raw speedup.*
3. **`S` is convex** in latency, so a fixed *additive* latency cut is worth more far from the
   roof, and a fixed *fractional* cut saturates near it — allocation of effort is non-obvious,
   and the marginal-eval heuristic in `schedule.priority_weight` uses `1 - S` as the headroom
   proxy.
4. **The canonical failure mode is measurable and common.** SOL's own screening reports a
   large flagged fraction dominated by **precision downgrades**; the local catalogue
   (`python -m argus.cli hacklab`) shows the gate CudaForge uses passes all four realistic
   hack families. Verification is not a nicety here; it is the difference between a score and
   a zero.

## 4. Risks to validity

* The snapshot rotates; refresh with `scripts/fetch_board.py` before planning.
* Per-kernel anchors must be read from the dataset; the *ranking* the model produces is robust
  to anchor error, the projected *score* is not.
* Do not compare B300 absolute timings to B200 anchors.
