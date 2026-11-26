# kernel_library — the reference library

Every kernel here already existed somewhere: a public SOL-ExecBench submission, this
repository's own released kernels, an archived contest variant, an upstream pull request, or
the official examples. They are collected so that a task never starts from nothing: the
campaign can seed a problem with the best known starting point, `forge lib adapt` can turn a
same-family reference into the problem's own submission, and the torch-only baselines cover
what no kernel can be found for.

```
kernel_library/
  sources.toml            # where kernels come from (local reference trees + opt-in git sources)
  index.json              # the index: one entry per kernel (rsi_kernel_tune.klib.schema)
  kernels/<id>/kernel.py  # ONE Python file, top-level `run(*args)`, positional, outputs last
  kernels/<id>/meta.json  # origin, original files, license + where it was read, notes, signature_check
  _sources/<name>/        # `forge lib fetch` clones (git-ignored); never committed
```

## What is in it (134 entries)

| origin | entries | what it is | license |
|---|---|---|---|
| `hyra` | 10 | Tencent's public SOL-ExecBench submissions, 9 Python + 1 converted to `cuda-inline` | Apache-2.0 |
| `forgekernel` | 25 | this repository's released kernels (CuTe DSL; one CUDA C++ bundle) | Apache-2.0 |
| `ako4x` | 48 | archived variants of the FlashInfer-Bench contest kernels | MIT |
| `kernelwiki` | 32 | upstream pull-request artifacts (reference bundles for `adapt`) | per artifact (header) |
| `sol-example` | 10 | the official SOL-ExecBench examples, one per language | Apache-2.0 |
| `auto-gpu-kernel` | 9 | DSA / GDN / MoE / GEMM kernels from a contest entry | Apache-2.0 |

13 board problems have an **exact** match (`problem_keys` names them); everything else is a
same-family reference (`forge lib find <problem>` still returns it, ranked below exact hits).
The exact hits are the 10 Hyra names plus `013_gqa_paged_decode_h32_kv8_d128_ps1`,
`020_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048` and `021_rmsnorm_h128`
from AKO4X (several also have `006_gemm_n2048_k4096` and the MLA problems from both).

## Licensing

`kernel_library/index.json` carries `license` and `usable_in_submission` per entry, and
`meta.json` says **where** the license was read (`license_source`: a repository LICENSE file,
a `PROVENANCE.yaml`, or the source file's own header). Only Apache-2.0 / MIT / BSD-2-Clause /
BSD-3-Clause entries are `submittable()`; anything else stays readable reference material with
`usable_in_submission: false`. Artifacts whose headers say NVIDIA-proprietary
(`LicenseRef-NvidiaProprietary`) are not redistributed here at all. See `NOTICE` for the
per-source attribution.

## Using it

```bash
forge lib find 021_rmsnorm_h128          # exact hits first, then the same family, score desc
forge lib find ssm --k 20               # everything of one family
forge lib show hyra-018_mla_paged_decode_h16_ckv512_kpe64_ps1
forge lib baselines 021_rmsnorm_h128 --dataset ~/SOL-ExecBench/data/benchmark --out /tmp/bl
forge lib cover --dataset ~/SOL-ExecBench/data/benchmark --out /tmp/cover --verify
forge lib adapt 021_rmsnorm_h128 --seed ako4x-rmsnorm-h128-triton_tiered_rows \
      --dataset ~/SOL-ExecBench/data/benchmark --task tasks/<op> --rounds 3
forge lib score 021_rmsnorm_h128 --card tasks/<op>/leaderboard/card   # needs benchx (B300)
```

## Rebuilding

```bash
forge lib build           # reads sources.toml, rewrites kernels/** and index.json
forge lib build --prune   # also removes generated entries this build no longer produces
```

The build reads the local reference trees (`$FK_REFERENCE_ROOT`, or the nearest `_reference`
directory at or above the repository). It never clones anything: `forge lib fetch <name>` is
the opt-in clone, and only for the `kind = "git"` sources listed in `sources.toml` (they are
disabled by default). `adapt` results live in the same tree; a rebuild keeps them.

## The one rule every `kernel.py` follows

A single Python file with a top-level `run(*args)`, positional arguments; when the harness
passes the outputs they are pre-allocated and last (destination-passing style), otherwise
`run` returns them. Nothing in this tree contains a file the evaluator would have to compile
beside it — a source that could not be brought to that shape mechanically is stored as a
documented *reference bundle* (`meta.json` says `runnable: false`) and is meant to be adapted,
not submitted.
