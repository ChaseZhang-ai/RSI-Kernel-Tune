# Leaderboard analysis: the exact objective and where the score is

All numbers below are reproduced from the board's public API by
`scripts/fetch_board.py` into `data/sol_execbench_b200_flashinfer.json`
(26 kernels, 660 workloads, B200, evaluation stack v1.1). Re-run the script to refresh.

---

## 1. The scoring rule, recovered exactly

```
per workload   S_w = 1 / (1 + (T_w − T_SOL,w) / (T_b,w − T_SOL,w))
per kernel     S_k = geomean over the kernel's workloads of S_w
collection     S   = UNWEIGHTED arithmetic mean over the 26 kernels of S_k
```

`S = 0.5` when a kernel exactly matches the **Scoring Baseline** `T_b`; `S → 1` as it approaches
the **SOL Bound** `T_SOL`. `Fast₁` is the separate count of workloads strictly faster than `T_b`.

**Verification.** Feeding the board's own published per-kernel scores through the collection
formula reproduces the published collection scores exactly:

| aggregation over the 26 per-kernel scores | ForgeKernel | Golden Bridge | published |
|---|---|---|---|
| **arithmetic mean** | **0.815204** | **0.815321** | ✅ matches |
| geometric mean | 0.802951 | 0.802892 | ✗ |
| workload-weighted arithmetic mean | 0.819068 | (n/a) | ✗ |

So the collection score is the **plain mean of the 26 per-kernel scores**. `S = 0.5` is the
Scoring Baseline, `S = 1.0` the SOL Bound — both confirmed against the board's reference entries.

## 2. Board snapshot

| entry | SOL Score | Fast₁ |
|---|---|---|
| SOL Bound | 1.000000 | 660/660 |
| Golden Bridge (rank 1) | 0.815321 | 654/660 |
| **ForgeKernel (rank 2; our baseline)** | **0.815204** | 647/660 |
| Scoring Baseline | 0.500000 | 0/660 |

The paper's "rank 1 as of 2026-10-30" has been overtaken by "Golden Bridge". **The target is
> 0.815321 and Fast₁ ≥ 654.**

## 3. Per-kernel breakdown (`dS = GB − FK`; positive means ForgeKernel trails rank 1)

| kernel | nw | FK_S | GB_S | best_S | headroom | FK Fast₁ miss | dS |
|---|--:|--:|--:|--:|--:|--:|--:|
| 009_gemm_n5120_k2048 | 25 | 0.5327 | 0.5325 | 0.5364 | 0.4673 | **8** | −0.0002 |
| 010_gemm_n6144_k4096 | 43 | 0.5978 | 0.5989 | 0.5989 | 0.4022 | **1** | +0.0011 |
| 007_gemm_n4096_k4096 | 43 | 0.6250 | 0.6061 | 0.6250 | 0.3750 | **3** | −0.0189 |
| 006_gemm_n2048_k4096 | 29 | 0.6363 | 0.6574 | 0.6574 | 0.3637 | 0 | +0.0211 |
| 001_fused_add_rmsnorm_h2048 | 7 | 0.6840 | 0.6811 | 0.6840 | 0.3160 | 0 | −0.0029 |
| 011_gemm_n28672_k4096 | 43 | 0.6990 | 0.6703 | 0.6990 | 0.3010 | **1** | −0.0288 |
| 008_gemm_n4096_k14336 | 43 | 0.7067 | 0.7148 | 0.7148 | 0.2933 | 0 | +0.0081 |
| 004_gemm_n128_k2048 | 25 | 0.7167 | 0.7194 | 0.7194 | 0.2833 | 0 | +0.0027 |
| 005_gemm_n256_k7168 | 17 | 0.7345 | 0.7381 | 0.7381 | 0.2655 | 0 | +0.0036 |
| 003_fused_add_rmsnorm_h7168 | 8 | 0.7440 | 0.7475 | 0.7475 | 0.2560 | 0 | +0.0036 |
| 022_rmsnorm_h512 | 8 | 0.7492 | 0.7529 | 0.7566 | 0.2508 | 0 | +0.0037 |
| 021_rmsnorm_h128 | 14 | 0.7685 | 0.7653 | 0.7685 | 0.2315 | 0 | −0.0033 |
| 024_rmsnorm_h2048 | 7 | 0.8013 | 0.8042 | 0.8042 | 0.1987 | 0 | +0.0029 |
| 025_rmsnorm_h4096 | 14 | 0.8449 | 0.8434 | 0.8449 | 0.1551 | 0 | −0.0014 |
| 026_rmsnorm_h7168 | 8 | 0.8714 | 0.8715 | 0.8760 | 0.1286 | 0 | +0.0001 |
| 023_rmsnorm_h1536 | 8 | 0.8854 | 0.8914 | 0.8914 | 0.1146 | 0 | +0.0060 |
| 002_fused_add_rmsnorm_h4096 | 14 | 0.8930 | 0.8975 | 0.9035 | 0.1070 | 0 | +0.0045 |
| 013_gqa_paged_decode_h32_kv8_d128_ps1 | 48 | 0.9318 | 0.9362 | 0.9362 | 0.0682 | 0 | +0.0044 |
| 017_gqa_ragged_prefill_causal_h32_kv8_d128 | 21 | 0.9409 | 0.9420 | 0.9420 | 0.0591 | 0 | +0.0011 |
| 016_gqa_ragged_prefill_causal_h32_kv4_d128 | 15 | 0.9482 | 0.9503 | 0.9503 | 0.0518 | 0 | +0.0020 |
| 012_gqa_paged_decode_h32_kv4_d128_ps1 | 48 | 0.9537 | 0.9557 | 0.9599 | 0.0463 | 0 | +0.0020 |
| 018_mla_paged_decode_h16_ckv512_kpe64_ps1 | 47 | 0.9714 | 0.9713 | 0.9740 | 0.0286 | 0 | −0.0001 |
| 020_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 | 19 | 0.9879 | 0.9894 | 0.9914 | 0.0121 | 0 | +0.0016 |
| 014_gqa_paged_prefill_causal_h32_kv4_d128_ps1 | 30 | 0.9897 | 0.9848 | 0.9911 | 0.0103 | 0 | −0.0049 |
| 019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | 38 | 0.9900 | 0.9913 | 0.9917 | 0.0100 | 0 | +0.0012 |
| 015_gqa_paged_prefill_causal_h32_kv8_d128_ps1 | 38 | 0.9915 | 0.9852 | 0.9924 | 0.0085 | 0 | −0.0063 |

## 4. Three findings that determine strategy

**Finding 1 — the score is dominated by the mid/low kernels.** With equal weight per kernel, the
eight kernels at `S < 0.75` hold ~2.6 of the total 4.8 points of headroom, while the seven kernels
at `S > 0.98` are essentially exhausted (total headroom < 0.07). Spending hours on `014/015/019`
buys almost nothing; `009/010/007/006` are where the score changes.

**Finding 2 — matching the best public kernel everywhere already wins.** Taking, per kernel, the
best public submission yields collection **0.819011 > 0.815321**. So the #1 spot is reachable by
*selecting the best-known architecture/structure per kernel* and closing a handful of small gaps —
an orchestration problem. `vulcan plan --target best` produces exactly that plan.

**Finding 3 — Fast₁ is a small, closed set.** ForgeKernel's 13 missing workloads are **all** in
four GEMM definitions (`009`: 8, `007`: 3, `010`: 1, `011`: 1). Rank 1 misses 6. Reaching Fast₁
660/660 requires only these four kernels, and `009_gemm_n5120_k2048` (8 misses, `S=0.53`) is both
the largest Fast₁ gap and the largest score headroom — a single high-leverage target.

## 5. What this implies for the run

1. **Target per kernel, not per run.** Use `plan.json` (`kernels.<name>.target_ms_geomean` and the
   per-workload `target_ms`) to set each upstream task's `[target] ms`, aiming first at the
   best-known score per kernel (→ 0.819011) and then past it.
2. **Front-load the low-`S` kernels** in the campaign order; the allocator already does this.
3. **Repair Fast₁ in the four GEMMs**; treat `009` as the flagship.
4. **Do not over-invest in attention.** Gains there are worth ≤ 0.07 total across seven kernels.
5. **A stretch target of 0.84 collection** requires lifting the eight low kernels from ~0.66
   average to ~0.90 — hard, and the place where new mechanisms (not tuning) would be needed. This
   is the honest ceiling of "orchestration-only" versus "new architecture" work.
