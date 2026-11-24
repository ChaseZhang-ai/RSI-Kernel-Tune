# Provenance — `blackwell_tma_and_warp_specialize` (hw_kb/triton/sm100/)

## Where the entry came from
Ported and reworded from the public MIT-licensed Triton guide
(`templates/skills/triton/triton.md`, "Triton 3.6 advanced-load levers"):
- `tl.range(..., warp_specialize=True)` aborts in the `ttgir` pass for a loop body with non-MMA work
  between two `tl.dot`s in one iteration (`RuntimeError: PassManager::run failed`), confirmed on a
  reduce-shaped kernel and on an attention-shaped loop, working on a pure `acc += tl.dot(A, B)`
  loop;
- `tl.make_tensor_descriptor`'s two unadvertised costs: per-CTA descriptor setup (loses to
  `ld.global.b128` on a non-persistent grid) and the small-tile penalty (a bulk-DMA-then-wait per
  CTA loses below roughly 16 KB per CTA);
- TMA pays for matmul-sized tiles (≳ 64 KB) and bypasses L1.
The guide's measured comparison sketch was replaced by the mechanism statements; no number from it
was kept as a reading.

The descriptor API's *naming* varies by release (`tl.make_tensor_descriptor` inside the kernel vs
`triton.tools.tensor_descriptor.TensorDescriptor.from_tensor` on the host); the entry says to verify
it on the installed release rather than pinning one spelling.

## Status
`[PENDING-CERT]`. To certify: a compile-only probe for `warp_specialize` (does this release accept
the loop?) plus one A/B at the largest shape for the descriptor form, both on the target.
