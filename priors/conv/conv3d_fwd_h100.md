# prior: conv3d forward (H100)

- Kernel: `kernels/conv/conv3d_fwd/h100/` (CuTe DSL, nvidia-cutlass-dsl)
- Published: +7.6% vs cuDNN (Figure 3, kernels/README.md)
- Status: filled (hand-written architecture decision map; decisions and values only —
  the kernel source stays in kernels/, nothing here transcribes it)

## Persistent scheduling
Persistent grid of 132 over 392 output tiles (linear mapping, ~3 waves); no split-K and
no cluster — the win comes from supply, not decomposition.

## Warp-role split
One producer warpgroup (TMA issue) + two consumer warpgroups (WGMMA + epilogue), each
consumer owning half the tile's M extent; setmaxnreg 40/232, and min-blocks-per-SM pinned
to 1 — without that pin the compiler silently drops the register reallocation and the
kernel spills.

## Pipeline depth
One full + one empty mbarrier per stage covering BOTH operands with a merged
expected-byte count; the WGMMA issue stays one group in flight behind the descriptor
advance, so supply and math hand off with a single rendezvous per stage.

## Data residency
The activation side enters through an im2col-mode TMA descriptor — the hardware does the
sliding-window gather, so no explicit patch extraction, no smem staging of halo tiles,
and no host-side repacking (the kernel needs a DSL recent enough to expose im2col mode).
Weights enter through a plain tile-mode TMA from a transposed view — the layout change is
free because the descriptor does it. All 128 fp32 accumulators live in registers
end-to-end.

## Epilogue path
Accumulator → smem ping-pong (two 4 KiB slots) → plain tile-mode TMA store, 16 rounds.
Each output byte is written once; no atomics, no second kernel.

## Shape domain
Single-output-channel-block 3-D convolutions (the class where N fits one tile and K is
deep); the im2col-TMA supply spine transcribes to other conv shapes and to the CUDA tree
(hw_kb CANDIDATES queues exactly that transcription).
