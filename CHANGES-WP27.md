# CHANGES-WP27 — external evaluation export package (2026-11-04)

Branch `feat/wp27-eval-export` (off `origin/main` = `31b0712`). Scope per the WP27
task pack (handoff 104; handoff 80, section 11, WP27): package an optimization product
into an external evaluation suite's accepted submission format — a standalone, opt-in,
human-run export lane. The WP24 de-boarding adjudication is **not** rolled back: the
framework core keeps zero network egress and zero external-benchmark surface; this
package only adds the manual lane beside it.

## Results

- **Baseline:** 2594 passed (recorded before any change, `tests/ -q`, 322.7 s).
- **Final:** 2615 passed (2594 + 21 new), 0 failed — see "Verification" below.
- No GPU used (this machine: CPU-only torch); the correctness preflight stage and its
  SKIP path are exercised exactly as designed.

## Files added / changed

| Path | What |
|---|---|
| `scripts/eval-export/pack.py` | the packager: one entry point, three product modes, three preflight stages, hash manifest |
| `scripts/eval-export/README.md` | short usage pointer (protocol lives in the doc) |
| `docs/external-eval.md` | the external evaluation protocol document (was a WP25 placeholder; now filled) |
| `docs/README.md` | the placeholder index line updated (one line, per the scope fence) |
| `tests/test_eval_export.py` | 21 tests: packaging correctness + isolation pins |
| `CHANGES-WP27.md` | this file |

Nothing else was touched: no `src/`, no `card_template`, no `meta/`, no `skills/`,
no `hw_kb/`, no `kernels/`, no run/gate paths.

## What the packager does

`python3 scripts/eval-export/pack.py OUT_DIR ...`, single entry point:

- **Product modes** (mutually exclusive):
  - `--run-dir .artifacts/<task>_<id> --op <op>` — auto-locates the **best KEEP**:
    the minimum gate-window reading (`device_median_ms`, else `bwd_median_ms`) over
    green KEEP rows of `operators/<op>/student/status.jsonl`, then pulls the kernel
    bytes from the run's git history at that iteration's sha (`git show <sha>:kernel.py`
    in `ws/`, else the card repo). If no repository holds the sha, the working tree's
    bytes are exported **with an explicit warning recorded in the manifest**. A
    cross-cluster span among green KEEP rows is recorded as a note (absolute readings
    are same-machine comparable only).
  - `--tree <operator tree>` — the tree as it stands; contract from the card's
    `eval/problem/problem.json`.
  - `--kernel <file>` — explicit file; the signature comes from the CLI
    (`--inputs/--outputs/--entry`); the manifest states that there is no run
    provenance and the gate claim is the operator's.
- **Contract precedence:** `eval/problem/problem.json` → `window.toml` `[student]` →
  CLI. DPS (destination passing style) is decided from the kernel's actual signature
  (inputs only = value-returning; inputs+outputs = DPS) and cross-checked against the
  contract's declaration; a drifted signature is refused (as the pre-WP24 exporter
  refused).
- **Submission documents:** the suites' solution JSON, `--target sol|fib|both`
  (default both → `solution-fib.json` + `solution-sol.json`; single target → plain
  `solution.json`). Language table and spec fields per the mined format (below).
- **Preflight, three stages** (recorded in `preflight.json`):
  1. lint / manifest static checks (always; any failure refuses, nothing written):
     contract parse, signature match, language resolution, CUDA flag allowlist,
     Python syntax compile, solution well-formedness;
  2. compile check (only with the local toolchain): `kernel.cu` built via nvcc +
     torch `cpp_extension` with the exported flags (CPU-only torch / missing nvcc /
     unusable toolchain → SKIP; compiler rejection → FAIL, refused); Python kernels
     imported in a bounded subprocess (import behavior is environment-dependent, so
     PASS/SKIP/WARN only — the refusing gate for Python is stage 1);
  3. correctness on the task contract's inputs (only with torch + CUDA device):
     deterministic inputs from the contract's declared shapes/dtypes, the contract's
     own reference vs the kernel, the contract's tolerance rule; no GPU or sealed
     reference → SKIP. This stage is a smoke re-check; **the run's own full gate
     remains the correctness authority**.
- **Hash manifest** `artifact_sha256.json`: sha256 of every package file, the
  contract files, the kernel bytes, and the product's provenance (run, op, iteration,
  sha, selection rule, winning reading, cluster, bytes source,
  `exported_verbatim: true`). Written last; never self-referential.

## Honesty rules (as implemented)

- The packaged kernel is the gated product **verbatim**: bytes are copied, never
  re-formatted, re-optimized, or post-processed ("no friendly tweaks on export").
- The manifest anchors the bytes and states where they came from; every shortcut
  (missing sha, explicit file with no run provenance, multi-file kernel whose sibling
  sources do not travel) produces a visible warning in `preflight.json` / the
  manifest rather than a silent package.
- CUDA solutions export **the flags the card built with** — the packager's CUDA
  constants are byte-identical to `card_template/eval/benchkit.py`'s, and a test
  pins that equality (`test_cuda_build_constants_stay_in_sync_with_the_card`).

## Format knowledge: mined from git history, not invented

Per the task pack, the submission format was mined from the pre-WP24 tree:

- `git show 82fa28d^` — the exporter module the WP24 de-boarding removed: the
  solution JSON shape (`name / definition / author / description / spec / sources`),
  the language table (`cutedsl → python|cute_dsl+cutlass`, `triton`,
  `pytorch`, CUDA → `cuda|cuda_cpp`), FIB vs SOL spec differences
  (`language` vs `languages`, `NVIDIA B200` vs `B200`, `dependencies`,
  SOL-only `compile_options.cuda_cflags`), the signature-drift refusal, and the
  `entry_point = "<file>::<entry>"` convention;
- `git show 65d735a^` (the board eval template + its doc) — the suites' protocols the
  document records (inputs/correctness/timing/scoring/clocks), and the naming
  conventions (`kernel.py` / `kernel.cu` as the single source file; solution naming
  default `<op>_<self-brand>`, now `_peregrine` where the historical default carried
  the old brand);
- cross-checked against the **current** tree: `cardkit.py` (the card's
  `eval/problem/problem.json` contract) and `benchkit.py` (the CUDA build constants,
  including the frozen `fk-cuda-cflags` directive).

Task 3 (FIB adaptation) closes as **implemented**: the difference between the two
formats is confined to `spec` (the diff table is in `docs/external-eval.md`), both
branches are built by the same packager, and tests pin both spec shapes; there is no
format work left open for `fib`.

## Isolation evidence (the pins that hold the boundary)

- `test_src_and_loop_tooling_never_reference_the_lane` — `git grep -i -E "eval[-_]export"`
  over `src skills meta` returns nothing: the framework never references the lane, so
  no automatic loop can reach it.
- `test_packager_is_stdlib_only_and_network_free` — AST of `pack.py`: no network
  module (urllib/http/socket/ssl/requests/...), no framework import (`peregrine_kernel`);
  the import set is exactly the declared stdlib allowlist (plus the compile probe's
  guarded `torch` import).
- `test_lane_is_documented_as_manual_export` — the lane carries the **"manual
  evaluation export"** label in the packager, the doc, and the generated package
  README; `docs/external-eval.md` records the audit note verbatim: the `src/`
  zero-network-egress audit scope is unchanged, `scripts/eval-export/` sits outside
  `src/` and is the explicitly-exempt, labeled lane.
- `test_cuda_build_constants_stay_in_sync_with_the_card` — the honesty anchor above.
- The WP24 zero-residue gate and the WP25 branding gate pass with the new files: the
  code surface keeps the suites' full names out (the packager says `sol` / `fib`;
  `docs/external-eval.md`, outside the code surface, is the mapping document).

## Fixture walkthrough (acceptance item 2)

A released operator from `kernels/` (the post-paper sparse-attention kernel), exported
end to end with the explicit-file mode:

```bash
python3 scripts/eval-export/pack.py /tmp/…/dsv41-b200 \
    --kernel kernels/sparse_attention/dsv41_fused_attn_rope_cast/b200/kernel.py \
    --definition dsv41_attn_rope_cast \
    --inputs q,kv,indices,token_positions,cos_sin_cache,attn_sink,topk_length,sm_scale,rms_norm_eps,enable_q_norm \
    --outputs out,sf,ml,lse --target both
```

Output (exit 0):

```
[eval-export] wrote /tmp/…/dsv41-b200 (2 suite format(s): fib, sol; kernel kernel.py
              3066 bytes, sha256 d461f10762df)
[eval-export]   locate       PASS
[eval-export]   lint         PASS
[eval-export]   compile      WARN
[eval-export]   correctness  SKIP
[eval-export] submit by hand; the protocol is docs/external-eval.md
```

Package tree (as generated):

```
dsv41-b200/
  README.md             (1404 bytes)  what this package is; manual-submission note
  artifact_sha256.json  (1828 bytes)  hash manifest + provenance
  kernel.py             (3066 bytes)  the released kernel, byte-identical
  preflight.json        (1403 bytes)  the three-stage report
  solution-fib.json     (3679 bytes)  fib submission document
  solution-sol.json     (3707 bytes)  sol submission document
```

Manifest excerpt:

```json
{
  "targets": ["fib", "sol"],
  "product": {"kind": "explicit-kernel-file", "kernel_file": "kernel.py",
              "kernel_sha256": "d461f10762df884ddd8284c8a2b8a783e72710cf2e80453092bec43496dff032",
              "exported_verbatim": true},
  "contract": {"language": "pytorch", "entry": "run",
               "destination_passing_style": false, "files": []}
}
```

- fib spec: `{"language": "python", "target_hardware": ["NVIDIA B200"],
  "entry_point": "kernel.py::run", "dependencies": [],
  "destination_passing_style": false}`
- sol spec: `{"languages": ["pytorch"], "target_hardware": ["B200"],
  "entry_point": "kernel.py::run", "destination_passing_style": false,
  "dependencies": ["torch"]}`

The `compile WARN` is honest and informative: this kernel JIT-builds a sibling
`dsa.cu` at import, which does not travel with a single-file export — exactly the
case the doc warns about ("a kernel whose device code lives in a sibling file is not
exportable in this form"), surfaced here rather than hidden. `correctness SKIP`: no
CUDA device on this machine.

The run-directory and operator-tree modes were walked through the same way (synthetic
run with a git workspace: best KEEP selected from the trajectory — iteration, sha,
reading, cluster and cross-cluster note recorded; bytes proven equal to
`git show <sha>:kernel.py` and unequal to the working tree's leftover bytes —
`tests/test_eval_export.py::test_best_keep_selection_pulls_bytes_from_git`).

## Verification

- Full suite after the change: **2615 passed** (`python -m pytest tests/ -q` in the
  project virtualenv named in the run book), including the new 21 tests and all
  pre-existing gates (deboard regression, branding, dataflow audit).
- The new tests are pure file I/O: no network, no GPU required (the GPU-dependent
  assertions are guarded by `torch.cuda.is_available()` and were exercised here on
  the SKIP path).

## Boundary notes for the auditor

- The packager is **offline by construction**; the submission upload is a human step,
  out of scope of the repository.
- The lane's opt-in/manual nature is pinned three ways: `src/` cannot reference it
  (grep pin), it imports nothing from `src/` (AST pin), and it is documented as the
  labeled exemption ("manual evaluation export") in `docs/external-eval.md`.
- No mechanism was added to the framework core (freeze order respected): the packager
  reads framework *artifacts* (trajectory, problem.json, window.toml) as flat files,
  without importing framework code.

<!-- SPDX-FileCopyrightText: Copyright 2026 The Peregrine-Kernel Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
