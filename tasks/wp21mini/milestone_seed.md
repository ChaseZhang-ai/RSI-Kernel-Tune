# Milestone — _wp21mini (WP21 mini smoke seed)

## §0 target

target_ms: null          # derived from the ref baseline by the loop (--equivalence-ratio 1.0)

## M1 — single decision phase (smoke)

- D1.1: elementwise add as one coalesced CUDA kernel launch (at::add baseline).

Phase order: M1. No coverage shape beyond `tiny`. This milestone is a smoke seed:
the loop runs with --no-oracle (no distillation) and --no-m0-audit.
