# prior: attention forward, causal, GQA 16:1 (H100, generation 2)

- Kernel: `kernels/attention/attn_fwd_gqa16/h100/gen2/` (CuTe DSL, nvidia-cutlass-dsl)
- Published: +4.2% over gen 1, cumulative +15% vs FlashAttention-4 (Table 1, kernels/README.md)
- Status: filled (hand-written architecture decision map; decisions and values only —
  the kernel source stays in kernels/, nothing here transcribes it)

## Persistent scheduling
Persistent grid of 132 with a static zigzag tile schedule — neighboring CTAs walk
neighboring KV ranges so L2 reuse of K/V pages stays local without runtime work
scheduling. The chain's gen 1→2 gain came mostly from reshaping this schedule and the
overlap below, not from re-tiling.

## Warp-role split
Three warpgroups: WG0 hosts the single producer warp that issues every TMA; WG1/WG2 are
consumers that ping-pong the softmax+MMA chain, each blocking on its own named barrier at
the epilogue so one stores while the other computes. Register split via setmaxnreg.

## Pipeline depth
K/V run a two-stage smem ring; Q lives in a double slot so the next tile's Q is prefetched
while the current round computes, with a QK(0) shadow round overlapping the previous
tile's epilogue drain. The supply warp runs ahead until the ring's empty wait throttles it.

## Data residency
K/V pages in the smem ring; the softmax row state (m, l) stays in registers as resident
fp32 for the whole KV walk — no row-state round-trip through smem or global across tiles.
Index math (stage, phase, tile counter) rides the uniform register file.

## Epilogue path
Rows leave registers via STSM into a per-consumer-warpgroup smem slab (64 rows each), the
storing warpgroup's TMA drains it under its named barrier. Output written once, bf16, no
atomics; the LSE side-channel rides the same path.

## Shape domain
Causal prefill-length attention, GQA 16:1 (one KV head feeds 16 Q heads — Q reuse makes
the zigzag schedule pay). Decode-length or paged-KV workloads need the paged-decision maps
instead (see index: gqa_paged_decode, mla_paged_decode).
