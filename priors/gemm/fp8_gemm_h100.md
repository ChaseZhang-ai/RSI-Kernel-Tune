# prior: FP8 GEMM, 128-block scaling (H100)

- Kernel: `kernels/gemm/fp8_gemm/h100/` (CuTe DSL, nvidia-cutlass-dsl)
- Published: +0.8% vs DeepGEMM (Figure 3, kernels/README.md)
- Status: filled (hand-written architecture decision map; decisions and values only —
  the kernel source stays in kernels/, nothing here transcribes it)

## Persistent scheduling
Persistent: grid 132 = SM count; software loop over 256 cluster-tiles (2×1×1 clusters,
66 of them), so tail absorption is a short last iteration, not empty CTAs.

## Warp-role split
Two math warpgroups (warps 0-7) plus one producer warpgroup (warps 8-11); register split
40/232 via setmaxnreg. Warp-invariant bookkeeping (tile coordinates, stage indices) rides
the uniform register file so it does not draw on the per-lane allocation.

## Pipeline depth
Three-stage TMA async ring over the K loop; one mbarrier pair per stage (full arrive 1,
empty arrive 16 cluster-scoped). Budgeted at three blocking rendezvous per k-block per
math warpgroup — the non-blocking try + tight-spin wait face beat the blocking suspend
face on this kernel. Stage depth itself was A/B'd (2 against 3); 3 won decisively.

## Data residency
A operand and its scale factors multicast across the cluster pair; B scales read once per
k-block (tiny, smem-resident). Output tile 256×128×128 per CTA-pair; the D staging is one
swizzled smem buffer (single stage — the epilogue drains before reuse).

## Epilogue path
Accumulator → swizzled smem → STSM → one single-shot TMA store of the whole 256×128 tile.
No atomics, no split-K partial buffer, no second kernel: the reduction-in-registers path
means global memory sees each output byte exactly once.

## Shape domain
Dense FP8 GEMM with 128-deep scale blocks, square 4096-class shapes; grouped
rasterization (group 16) over the persistent loop keeps L2 locality on wide N. Tall/skinny
or heavily grouped MoE shapes need the grouped-GEMM decomposition instead (see index).
