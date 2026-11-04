# External evaluation export

This document specifies the manual lane that packages a Peregrine-Kernel optimization
product into the submission format of an external evaluation suite: the NVIDIA SOL
suite (SOL-ExecBench, free B200 evaluation) and FlashInfer-Bench. The tooling is
[`scripts/eval-export/pack.py`](../scripts/eval-export/pack.py); everything below the
submission-format section is a faithful record of the suites' own published protocols —
**their clock, their inputs, their stopwatch, their judging: that is their grading, not
our measurement.** Where our own measurement discipline differs, section
["Self-evaluation on a leased card"](#self-evaluation-on-a-leased-card) says so
explicitly.

## What this lane is (and is not)

- **Manual, opt-in, human-run.** The packager is run by a person, by hand. It is in no
  automatic loop, the driver never calls it, and it performs **zero network I/O** — the
  upload to a suite's submission channel is a separate human step.
- **Independent of the framework.** `scripts/eval-export/` imports nothing from
  `src/peregrine_kernel/` (a test pins this), so it cannot drag framework code onto a
  submission path. The WP24 de-boarding adjudication stands: the framework core stays
  zero-network-egress and has zero external-benchmark surface. The export lane is a
  boundary asset, documented here and in the audit note below.
- **Audit scope note.** The `src/` zero-network-egress audit scope is unchanged:
  `scripts/eval-export/` sits outside `src/`, and is listed in the audit's exemption
  notes with the label **"manual evaluation export"** (this document is the label's
  anchor). Its isolation (stdlib-only, no framework imports, no network modules) is
  pinned by `tests/test_eval_export.py`.
- **Why the code surface says `sol` / `fib`.** The WP24 zero-residue gate greps the code
  surface (`src/ skills/ scripts/ tests/`) for the suites' names and keeps them off.
  The packager therefore takes `--target sol|fib`, and this document — which is outside
  the code surface — is the mapping: `sol` = the NVIDIA SOL suite (SOL-ExecBench),
  `fib` = FlashInfer-Bench.

## The submission format

Both suites accept a **solution JSON** document. The format below is mined from the
framework's pre-WP24 export path (git history, commit `82fa28d^` — the exporter the
WP24 de-boarding removed), which replicated what the suites' own harnesses read; it is
re-implemented, shape-for-shape, by `scripts/eval-export/pack.py`.

```json
{
  "name": "<solution name>",
  "definition": "<the suite dataset's definition name>",
  "author": "<author>",
  "description": "<one line>",
  "spec": { "...suite-specific; see the diff table..." : "..." },
  "sources": [ { "path": "kernel.py", "content": "<the whole kernel file>" } ]
}
```

- `definition` **must match the suite dataset's definition name** (e.g. a
  `flashinfer-trace` definition id, or the SOL dataset's `<subset>/<problem>` id). The
  framework's own task names are not suite ids; pass `--definition` explicitly and take
  the value from the dataset you are submitting against.
- `entry_point` is `"<kernel file>::<entry>"` (the framework's frozen contract is
  `kernel.py::run` or `kernel.cu::run`).
- `destination_passing_style` reflects the kernel's actual signature: inputs only =
  value-returning (`false`); inputs then outputs = destination-passing (`true`). The
  packager refuses a signature that matches neither form.
- `sources` holds the kernel file **verbatim** — the same bytes the run's gate judged.
  One file per solution (the frozen contract: "a solution is this one file"; a kernel
  whose device code lives in a sibling file is not exportable in this form and the
  preflight report says so in its warning).

Language table (kernel file → suite-facing language; inferred from the file suffix and
its imports, overridable with `--language`):

| Kernel language | `fib` spec `language` | `sol` spec `languages` | `sol` `dependencies` |
|---|---|---|---|
| CuTe DSL (`kernel.py`, imports `cutlass`) | `python` | `cute_dsl` | `cutlass` |
| Triton | `triton` | `triton` | `triton >= 2.3` |
| PyTorch host | `python` | `pytorch` | `torch` |
| CUDA C++ (`kernel.cu`) | `cuda` | `cuda_cpp` | (none) |

A `cuda_cpp` solution additionally carries `compile_options.cuda_cflags`: **the flags
the card built it with** (defaults `-O3 --use_fast_math` + sm_100a codegen, plus any
`// fk-cuda-cflags:` line's flags, allowlist-checked) — mirroring
`src/peregrine_kernel/card_template/eval/benchkit.py`, so the bytes and flags judged on
the target machine are the bytes and flags submitted.

Package contents written by `pack.py OUT_DIR ...`:

```
OUT_DIR/
  solution.json            # one per suite ("solution-fib.json" + "solution-sol.json" for --target both)
  kernel.py                # the exported product, byte-identical to the gated artifact
  preflight.json           # the three-stage preflight report (below)
  README.md                # what this package is; generated
  artifact_sha256.json     # hash manifest: every file + the kernel's provenance
```

The manifest's `product` block anchors the honesty rule: which run, which operator,
which iteration (`sha`), which reading won (`best KEEP by <window>`), where the bytes
came from (`git show <sha>:kernel.py` or the working tree), `kernel_sha256`, and
`exported_verbatim: true`. For an explicit `--kernel` export there is no run
provenance, and the manifest says so in so many words: the hash anchors the bytes; the
gate claim is the operator's.

## Preflight (before anything leaves the machine)

Three stages, recorded in `preflight.json`:

1. **Lint / manifest static checks** (always): the task contract
   (`eval/problem/problem.json`, else `window.toml` `[student]`, else the command line)
   parses; the kernel's entry signature is exactly the contract's inputs (+ outputs
   under DPS); the language resolves; CUDA flags are inside the card's allowlist; the
   Python source compiles (syntax); the solution documents are well-formed. **Any
   failure here refuses the export** (nothing is written).
2. **Compile check** (only when this machine has the toolchain): a `kernel.cu` is built
   with nvcc + torch `cpp_extension` using the exported flags; a Python kernel is
   imported in a bounded subprocess. A toolchain gap → `SKIP` with the reason (the
   target machine has it; this one may not — a CPU-only torch, a missing nvcc, or a
   kernel that JIT-builds at import each read as `SKIP`/`WARN`, never as a verdict on
   the kernel). A real compiler rejection of the kernel source → `FAIL` (refused).
3. **Correctness on the task contract's inputs** (only with torch + a CUDA device):
   deterministic inputs drawn from the contract's declared shapes/dtypes, the
   contract's own reference against the kernel, the contract's tolerance rule
   (`max|y − judge| / max|judge|` per output, floor two units of the output dtype's
   rounding). No GPU, or a sealed/binary reference → `SKIP`+warning. **This stage is a
   convenience smoke re-check; the run's own full gate remains the correctness
   authority** — the exported artifact is gated bytes, not re-gated bytes.

## Suite protocols (as their documentation states them)

### The NVIDIA SOL suite (SOL-ExecBench)

Upstream: `NVIDIA/SOL-ExecBench` (pinned in the framework's historical templates at
commit `a9fa0804`; protocol surface: `driver/templates/eval_driver.py`,
`core/bench/{io,correctness,timing,reward_hack}.py`). Dataset:
`huggingface.co/datasets/nvidia/SOL-ExecBench`. Board and per-workload numbers:
`research.nvidia.com/benchmarks/sol-execbench/api` (public API; free B200 evaluation
via their submission channel).

- **Inputs**: one draw seed (`seed = 200`); ten rounds, each a fresh input group drawn
  with parameter-name heuristics (norm weight = 1, bias = 0, causal/binary masks, rope
  cos/sin, positive statistics, softmax outputs, `randn/√fan_in` weights) falling back
  to `randn`; the workload may also carry its own generator or safetensors inputs.
  Timing uses the last round's group only.
- **Correctness**: reference then solution in every round. Round 0 checks the output
  type is strictly `torch.Tensor`, its shape and dtype, and records the thread count.
  Every round: `inf`/`NaN` on either side is wrong (only `allow_negative_inf` lets a
  `-inf` at the same position through); an all-zero solution against a non-zero
  reference is wrong; elementwise `|d| ≤ atol + rtol·|ref|`. Tolerance comes from the
  workload, default `atol = rtol = 1e-2`, `required_matched_ratio = 0.99`, optional
  `max_error_cap`. Anti-cheat probes: lazy outputs and thread injection during timing.
- **Timing** (their protocol, recorded to anchor what their numbers mean): last round's
  group, 10 warm-up calls, 50 timed calls, L2 flushed before every call, a fresh
  allocation address every call (random offset in multiples of 256 B, seed 200), the
  **median** single-call kernel-span latency is the reading.
- **Clock**: their B200 preset locks SM 1500 MHz / memory 3996 MHz
  (their `device_config.py`); T_b and T_SOL are readings under that lock.
- **Scoring**: `S = 1 / (1 + (T_k − T_SOL) / (T_b − T_SOL))`, where T_b is the board's
  scoring baseline latency and T_SOL the SOLAR hardware floor, **both published by the
  board per workload** and measured on their machine. A kernel at T_b scores 0.5;
  halfway to the floor 0.75.

### FlashInfer-Bench

Upstream: `flashinfer-ai/flashinfer-bench` (pinned in the framework's historical
templates at commit `15618a67`; protocol surface: `bench/evaluators/default.py`,
`bench/utils.py`, `bench/config.py` + `eval_config.yaml`). Dataset:
`huggingface.co/datasets/flashinfer-ai/flashinfer-trace` (definitions, workloads,
baseline solutions, safetensors blobs).

- **Inputs**: `num_trials = 3` groups, **unseeded** — every evaluation job draws
  different random inputs (float `randn`; fp8/fp4 `randn` clipped to ±2; int `randint`;
  bool; scalars / safetensors from the workload). A kernel that fails occasionally gets
  caught in some round instead of being hidden by a fixed seed.
- **Correctness** (their default evaluator): the reference on every group first, then
  the solution group by group; shape → dtype → non-finite solution values (wrong at
  once) → elementwise rule: an element is wrong only when `|d| > atol` **and**
  `|d|/|ref| > rtol`; default `atol = rtol = 1e-2` with a 1.0 match ratio (`moe`
  op types default to 0.95). Two special evaluators exist (sampling: TVD over 500k
  samples; DSA top-k indexer: re-scoring and sorting by returned indices) — the
  framework's historical judge did not replicate them and `pack.py` does not try to
  pre-judge them; the board grades them its own way.
- **Timing**: per group, 10 warm-up calls, 50 timed calls, L2 flushed before every
  call, the median of the single-call CUPTI kernel span; **the reading is the mean
  over the 3 groups**. FlashInfer-Bench synchronizes after the L2 flush (the SOL suite
  does not — one more reason their numbers are their numbers).
- **Clock**: the board does not lock clocks.
- **Ranking**: speedup over the dataset's baseline solution (the same solution the
  board's reference run times).

### What a submission gives back

The SOL suite's free evaluation **returns verdict numbers only — no NCU trace, no
profiling artifacts**. If the paper needs microarchitectural evidence (SASS census,
per-kernel tables, timeline probes), that evidence comes from the framework's own
tooling on a *rented* B200 (the user's 2026-11-04 decision: rent a B200 for the search
loop — the same architecture as the free evaluation, so no cross-architecture caveat).
Budget one short rental segment for profiling separately from the search rental.

## The two formats, side by side (`fib` vs `sol`)

The formats are siblings; the differences are exactly these:

| | `--target fib` | `--target sol` |
|---|---|---|
| `spec.language` / `spec.languages` | `language`: single string | `languages`: list |
| `target_hardware` | `["NVIDIA B200"]` (vendored prefix) | `["B200"]` (bare name) |
| `dependencies` | always `[]` | from the kernel language (`cutlass` / `triton >= 2.3` / `torch` / none) |
| `compile_options` | absent | `{"cuda_cflags": [...]}` for `cuda_cpp` solutions |
| everything else | identical: `name`, `definition`, `author`, `description`, `entry_point`, `destination_passing_style`, `sources` | identical |

Because the difference is confined to `spec`, one package can carry both
(`--target both` → `solution-fib.json` + `solution-sol.json`); submit the file that
matches the suite you submit to. Both branches are implemented and tested; there is no
format work left open for `fib`.

## Self-evaluation on a leased card

The suites are **submission-based evaluators: they grade, they are not timing
oracles**. The optimization loop needs interactive measurement, and the free
evaluation cannot serve it (no iteration, no profiling, days of latency per data
point). The framework's own strict measurement discipline applies on the leased B200:

- **Locked clocks** (`[protocol]` in the card): lock SM/memory clocks, read them back
  (± 50 MHz), refuse to time when the lock does not hold, reset on exit. Absolute
  numbers are comparable only under a declared lock — and the SOL suite's published
  T_b / T_SOL are readings under *their* lock (SM 1500 / mem 3996), so same-lock
  readings on the rented card are what may be compared with them.
- **Paired A-B timing**: burn-in, then the mirrored order ref / kernel / kernel / ref,
  each side the mean of its two readings (`eval/timing.py` `measure_paired`) — the
  first timed window of a job is the fastest and drifts with temperature; an unpaired
  pair measures that drift, not the kernel.
- **Three-workload geomean gate**: one shape is not a claim; the gate reads the
  geomean over three representative workloads, and the reference is measured on the
  same set (single-ref contract: no stand-in library rides a job).
- **One flip at a time**: acceptance A/Bs change one default at a time (the energize
  protocol, `docs/energize-protocol.md`), so a win is attributable to a mechanism, not
  to a bundle.
- **Repetition buys the claim**: per-iteration verdicts stay single-job for
  throughput; the run's final claim is re-measured across jobs (median vs median;
  node bimodality is real at the ±6% level).

Paper-number provenance: every performance claim in the paper comes from this
framework's own measured runs under the discipline above; upstream published numbers
are quoted as upstream results, with citation (see `docs/paper/` for the integrity
rules).

## Public board submission: an optional decision

The paper needs evaluation numbers; being listed on a public board is a side effect,
not a goal. Submitting to the public board additionally means: the solution becomes
public, tied to the definition it was written against, and graded by the board's
schedule and rules. The packager makes the submission package; the decision to
actually submit is the user's, made per suite, per kernel.

## Usage

```bash
# 1) the run's best KEEP (auto-located from the trajectory; bytes pulled from the
#    run's git history at that iteration's sha)
python3 scripts/eval-export/pack.py pkg/ \
    --run-dir .artifacts/rmsnorm_1a2b3c --op rmsnorm \
    --definition <suite-dataset-definition-name> --target both

# 2) an operator tree as it stands (contract read from the card's problem.json)
python3 scripts/eval-export/pack.py pkg/ --tree operators/rmsnorm \
    --definition <suite-dataset-definition-name> --target sol

# 3) an explicit kernel file (no contract files: pass the signature on the CLI)
python3 scripts/eval-export/pack.py pkg/ \
    --kernel kernels/sparse_attention/dsv41_fused_attn_rope_cast/b200/kernel.py \
    --definition <suite-dataset-definition-name> \
    --inputs q,kv,indices,token_positions,cos_sin_cache,attn_sink,topk_length,sm_scale,rms_norm_eps,enable_q_norm \
    --outputs out,sf,ml,lse
```

Exit code 0 = package written; 2 = refused (a lint failure, a compiler rejection, or a
correctness FAIL — nothing is written on refusal). Full CLI: `python3
scripts/eval-export/pack.py --help`.

<!-- SPDX-FileCopyrightText: Copyright 2026 The Peregrine-Kernel Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
