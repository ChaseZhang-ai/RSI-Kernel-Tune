# Provenance — `grid_axis_dispatch_order` (all four trees)

## Where the entry came from
- Ported and reworded from the public MIT-licensed Triton guide
  (`templates/skills/triton/triton.md`, "Grid-axis dispatch order: `tl.program_id(0)` varies
  fastest"): the grid tuple becomes `gridDim`, the hardware dispatches `blockIdx.x`-fastest, and a
  reorder that swaps the grid tuple without re-binding `program_id` inside the kernel is a no-op.
  The guide's measured example (a flash-decode split kernel with a per-grid-axis L2 fanout table)
  was replaced by a generic mechanism statement; its numbers were not carried over.
- The underlying hardware statement is the CUDA C++ Programming Guide's thread-block scheduling.

## Transcription
The sm90 and sm86 copies carry a `Certification source` line naming sm100 and this file; the text is
identical apart from that line. The mechanism names no architecture.

## Status
`[PENDING-CERT]` in every tree. To certify: an A/B duration for one kernel under both bindings
(binding-swapped grid axes) with the L2 hit-rate reading from NCU to show the reuse actually moved.
