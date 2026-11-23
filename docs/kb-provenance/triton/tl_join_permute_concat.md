# Provenance — `tl_join_permute_concat` (hw_kb/triton/sm100/tl_join_permute_concat.md)

## Where the entry came from
Ported and reworded from the public MIT-licensed Triton guide
(`templates/skills/triton/triton.md`, "Tensor layout: deterministic hcat / vcat via `tl.join` +
permute + reshape"): `tl.cat` supports only `can_reorder=True`, so its element order is
unspecified; `tl.join` + `tl.permute` + `tl.reshape` gives the ordered hcat / vcat forms.
The `tl.cat` docstring statement is the API surface the entry cites directly.

The guide's register-budget figure (`[64, 64]` bf16 ≈ 8 KB in flight per CTA) was kept because it
is arithmetic on the dtype and the tile shape, not a measurement of a kernel; the entry states it
as an order-of-magnitude sizing aid.

## Status
`[PENDING-CERT]`. To certify: a rung-1 probe that assembles both concatenations from tiles with
distinct sentinel values and checks every element's position on the target.
