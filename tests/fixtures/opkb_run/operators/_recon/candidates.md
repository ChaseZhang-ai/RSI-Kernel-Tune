# Candidate architectures -- 069_rms_norm

The first line of every candidate states the cluster shape and the CTA count.

## Candidate A

| decision | value | notes | tag |
|---|---|---|---|
| tile_M x tile_N | 128x256 | one CTA per tile | [VERIFY: ncu-derived] |
| stage count | 3 | smem budget 172032 B | [VERIFY: sass-derived] |
| register split | 40/232 | USETMAXREG.DEALLOC.CTAPOOL | [VERIFY: ncu-direct] |

## Candidate B

| decision | value | notes | tag |
|---|---|---|---|
| tile_M x tile_N | 256x128 | the transposed factorization | [VERIFY: inferred] |
| stage count | 4 | does not fit the smem budget | [VERIFY: none] |
