# prior: h100

- Kernel: `kernels/sparse_attention/dsa_sparse_bwd/h100/` (CuTe DSL (nvidia-cutlass-dsl))
- Published: ? vs ? (kernels/README.md)
- Status: skeleton (auto-generated; fill the sections below — decisions and values in prose, never an implementation body)

## Persistent scheduling

TODO: grid == SM count with a software tile loop, or one CTA per output tile? How are tail tiles absorbed?.

## Warp-role split

TODO: which warps produce (TMA/cp.async), which consume (MMA), which own the epilogue; the register split between them.

## Pipeline depth

TODO: stages per ring buffer, who arrives/waits on what, the in-flight depth the schedule sustains.

## Data residency

TODO: what lives in smem vs registers vs L2 across the hot loop; what is multicast/shared inside a cluster.

## Epilogue path

TODO: accumulator to smem to global spelling (R2S/STSM/TMA store), swizzle, whether a second kernel or an atomic ever touches the output.

## Shape domain

TODO: the shapes this architecture wins at (decode vs prefill, tall vs square GEMM, grouped vs dense) and where it degrades.
