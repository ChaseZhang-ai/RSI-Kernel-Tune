# Provenance archive — Triton knowledge base

> **Written for people, not for agents.** This subdirectory is in no agent's read set: a Triton run
> reads only `hw_kb/triton/<kb_arch>/`. It exists for the same reason as `docs/kb-provenance/` and
> `docs/kb-provenance/cuda/` (see the parent README): an entry's "Sources" may cite official docs,
> public-infra API symbols and this repo's probes; everything *narrated* — which public guide a
> passage was derived from, what was borrowed, what was reworded — lives here.

## What is in this release

`hw_kb/triton/` started from **methodology entries** ported from public, permissively licensed
material and public documentation; none of them has been certified by a GPU probe of this
repository. Each entry says so itself (`[PENDING-CERT]`), and each entry that carries borrowed
wording has a file here with the same name.

| origin | licence | used for |
|---|---|---|
| Triton documentation, release notes (`triton-lang.org`, `triton-lang/triton`) | public documentation | the API surfaces every entry cites: `tl.*`, `num_warps`, `num_stages`, `triton.compile`, `GPUTarget` |
| CUDA C++ Programming Guide, PTX ISA | public documentation | the hardware statements (coalescing, dispatch order, `cp.async`, `wgmma`, `cp.async.bulk.tensor`, TMEM) |
| AKO4X `templates/skills/triton` (a public Triton guide) | MIT | the empirical claims: the MMA tile picker moving with `num_warps`, the TMEM budget and its count-permanent allocator, the `return`-inside-loop restriction, the pipeliner not carrying across an outer `while`, the spin-wait SM stall, the `tl.cat` order caveat and the `tl.join` form, the TMA small-tile penalty, the `warp_specialize` restrictions, the 2D-tile reduce form, the grid-axis dispatch order |
| SOL-ExecBench `docs/solution.md` (the suite's published solution format) | Apache-2.0 | the one-file Triton solution contract |

The AKO4X passages were **reworded into operator-agnostic mechanism prose** and their numbers were
prefixed with the `[PENDING-CERT]` warning rather than copied as readings. `NOTICE` records the
attribution.

## What a person should do with this

- When a probe certifies an entry, replace that entry's `[PENDING-CERT]` line with the probe's
  verdict and add the probe's path here.
- When an entry's borrowed claim turns out to be wrong on the target, retire the entry (the tree's
  `index.md` row goes with it) and note it here — a wrong mechanism is worse than a missing one.
