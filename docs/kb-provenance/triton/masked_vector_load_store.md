# Provenance — `masked_vector_load_store` (sm100 + sm90 + sm86)

## Where the entry came from
- The mask / `other` / cache-modifier / `eviction_policy` semantics are the Triton documentation's
  `tl.load` / `tl.store` API surface; `tl.multiple_of` / `tl.max_contiguous` are the documented
  hints the compiler uses to widen an access.
- The coalescing floor (a per-row store below ~64 B fragments into sub-cache-line writes; aim for
  64 B, prefer 128 B) and the "the per-CTA store size is what matters, a bigger grid does not
  recover coalescing" statement were ported and reworded from the public MIT-licensed Triton guide
  (`templates/skills/triton/triton.md`, "Split-K reduce kernels"). The guide's measured regression
  figures were left out.
- The transaction sizes are the CUDA C++ Programming Guide's "Device Memory Accesses" material.

## Transcription
The sm90 copy adds a line saying the coalescing floor was written for the target architecture and is
a starting point there; the sm86 copy adds the same plus the "development / smoke target" caveat.

## Status
`[PENDING-CERT]` in every tree. To certify: an NCU `l1tex__t_sectors_per_request` (or equivalent
byte-efficiency) reading for one kernel under a per-row size above and below the floor.
