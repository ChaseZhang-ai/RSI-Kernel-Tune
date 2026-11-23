# Provenance — `num_warps_num_stages_table` (hw_kb/triton/sm100 + sm103)

## Where the entry came from
- "Do not tune inside the job; the protocol's warmups and cache flush make in-run timings the wrong
  measurement, and the autotune's own trials cost clock" — derived from the board protocol the card
  implements (fixed warmup count, L2 flush per timed call) plus the public MIT-licensed Triton
  guide's autotune pitfalls section ("variance from re-selection", "hardcode the best config and
  remove `@triton.autotune`").
- "`num_stages` is a shared-memory budget" — Triton documentation for the launch parameter and the
  public guide's stage-count discussion.
- "`num_warps` also changes which MMA the tile picker chooses" — the public guide's small-N fp8
  table; that detail is carried in `dot_tile_and_warp_count.md` and cross-referenced rather than
  restated with numbers here.

## What the entry does NOT claim
No per-knob table values: the example's `(4, 3, 1024)` / `(8, 4, 4096)` are illustrative shapes of
the *form*, not readings.

## Status
`[PENDING-CERT]`. To certify: for the target architecture, one job per row of a real table with the
card's own clock, and the table recorded here with the readings.
