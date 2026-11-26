# The kernel reference library (`forge lib`)

The library exists so that no problem starts from nothing. It collects every kernel we may
legally read and reuse, normalizes each to the one shape a submission must have, records where
it came from and what it scored, and offers three generators that need no kernel work at all:
the torch-only baselines of a definition, the whole-board reference fallback, and `adapt`,
which turns a same-family reference into the problem's own starting point.

Command: `forge lib <build|find|show|score|baselines|cover|fetch|adapt>`
(implementation: `src/rsi_kernel_tune/ext/lib.py`, `src/rsi_kernel_tune/klib/`). The library itself is
`kernel_library/` — see its `README.md` for the inventory and `NOTICE` for attribution.

## 1. What it holds and where it comes from

`kernel_library/sources.toml` registers every source: a local reference tree (path relative to
`$FK_REFERENCE_ROOT`, or the nearest `_reference` directory at or above the repository), or a
git repository that `forge lib fetch` may clone at a pinned commit. Each source names its
extractor, its license and where that license was read.

`forge lib build` rewrites `kernels/**` and `index.json`; `--prune` also removes generated
entries the build no longer produces (`adapt` results are never removed), `--strict-licenses`
refuses to write anything whose license is not one of the allowed ones.

| source | extractor | entries |
|---|---|---|
| `_reference/Hyra-results/AI4AI/sol_execbench` | `hyra` | 10 |
| this repository's `kernels/` | `rsi_kernel_tune` | 25 |
| `_reference/AKO4X/reference/**/variants/*` | `ako4x` | 48 |
| `_reference/KernelWiki/artifacts/**` | `kernelwiki` | 32 |
| `_reference/SOL-ExecBench/examples/**` | `sol-example` | 10 |
| `_reference/auto-gpu-kernel/archive/**` | `auto-gpu-kernel` | 9 |

Not ingested, with reasons recorded in `sources.toml`: `AKO4ALL` (a skill/bench harness — its
README reports 1.12–1.50× expert on four operators but ships no kernel source), and Cursor's
235 public solutions / `qhy991/SOL-Baseline` / `amateur` repositories without a license
(background doc §5: their *numbers* are used as data, their code is not). The larger libraries
(FlashInfer, FlashAttention, FlashMLA, DeepGEMM, FlagGems, vLLM, SGLang) are registered as
opt-in `git` sources — `forge lib fetch <name>` shallow-clones one at its pinned commit into
`kernel_library/_sources/<name>/`, registers the commit and the license it found, and stops
there. Normalization rules for those are per-library work for later; nothing is fetched by
default and `build` never touches the network.

`reported` and `source` are always copied from the source data (a manifest, a `result.json`, a
README table, a `PROVENANCE.yaml`); a number with no provenance stays out. `problem_keys`
names the board problems an entry is **known** to solve (exact names, from the source's own
metadata or the family mapping in this module) plus its family; a kernel that is merely the
same kind of operator gets the family key only.

## 2. Every entry is one file: `kernel_library/kernels/<id>/kernel.py`

`rsi_kernel_tune.klib.normalize` is pure text-in/text-out, with one rule per source shape:

1. **single-file Python** — kept byte for byte; if its entry is not called `run`, an alias
   `run = <entry>` is appended (`ensure_entry`; the last public function if the source does not
   name one).
2. **multi-file Python** — merged in dependency order. Every top-level binding of an inlined
   file is prefixed `_klib_<module>__` and *all* references to it are rewritten (AST positions
   → byte splices, so comments, formatting and license headers survive); the entry file is
   appended last with its imports of the inlined modules rewritten to the prefixed names. The
   prefix is the collision rule the task asked for. A file that loads a sibling *by path at run
   time* cannot be fixed that way and is noted as needing `adapt`.
   *Deviation worth knowing:* the first implementation isolated modules with `exec` in a
   `ModuleType`; the repository's security scanner refuses `exec` of external data, and the
   prefix rewrite is what the task text asked for anyway.
3. **Python that builds sibling CUDA files from disk** (RSI-Kernel-Tune's DeepSeek-V4.1 kernel) —
   cannot be reduced to one file mechanically. It is stored as a **reference bundle**: the
   Python entry file and its `.cu`/`.cuh` siblings embedded as string literals, `run` defined
   but refusing loudly, `meta.json` saying `runnable: false`. `adapt` is the consumer.
4. **CUDA C++** (`.cu`/`.cpp` + headers with a `PYBIND11_MODULE`) — `cuda_to_inline` generates
   a `submission.py` that embeds the sources (quoted `#include`s of files we have are inlined
   recursively), builds them once at import with `torch.utils.cpp_extension.load_inline`
   (`functions=None`, so the file's own binding is used), and derives the gencode from
   `torch.cuda.get_device_capability()` at import — a `-gencode`/`-arch` flag the source pinned
   is dropped and noted. The module name is content-addressed (plus the device capability), so
   a rebuild of the same source reuses torch's build cache. `n_outputs` controls the DPS copy:
   when the binding returns tensors they are copied into the pre-allocated outputs.
   The CUDA sources are text: nothing is compiled or run at build time.

`static_rules_check` (the soft version of task targethw's target rules) refuses an architecture
literal, a pinned code-generation target or a hardcoded SM count in generated wrapper code; it
is applied to the wrapper only, never to an embedded kernel's own text.

## 3. The index and the license gate

`kernel_library/index.json` is `{"version": 1, "entries": [...]}` and its schema is task step
0's stub (`rsi_kernel_tune/klib/schema.py`, unchanged): `id`, `problem_keys`, `family`, `path`,
`language`, `entry`, `source`, `license`, `reported`, `verified_local`,
`usable_in_submission`. `validate()` reports every problem at once; `submittable()` is the
gate: only Apache-2.0 / MIT / BSD-2-Clause / BSD-3-Clause entries may be merged into a
submission. `meta.json` beside each kernel holds what the index has no field for: origin,
original files and entry, which files were merged, **where the license was read**, the reported
score, `signature_check`, `runnable`, and mechanical notes (including a trap line quoted from a
variant's own header, which `find` prints).

Two decisions worth writing down, because the task text and the schema stub disagree on them:

* A license that is **not allowed** does not stop an entry from existing. The stub's own rule
  is that such an entry must say `usable_in_submission: false`, and the third revision of the
  task says KernelWiki artifacts whose license cannot be read are marked `unknown` and blocked
  from merging — i.e. kept, not dropped. So `build` keeps them (blocked by `submittable()`),
  and `forge lib build --strict-licenses` is available for the stricter reading ("nothing
  unusable enters `kernels/`").
* A license that says **proprietary** (`LicenseRef-NvidiaProprietary`, e.g. CUTLASS's CuTeDSL
  Python files) is not redistributed at all: those artifacts are skipped with the reason
  recorded, and the upstream source stays the place to read them.

## 4. Querying: `find`, `show`, `score`

```
forge lib find <problem name or family> [--k N] [--json]
```

Ranking: an exact `problem_keys` hit first, then entries of the query's family (via
`rsi_kernel_tune.family`), then substring hits; inside a rank, entries with a reported score first,
descending, then the rest by id, so an unscored entry is visibly last. The output carries the
license, the language, whether the entry is `runnable` (reference-only bundles are not), its
`signature_check`, any trap line, and — when the metrics CSV from the board snapshot can be
found (`$FK_CURSOR_METRICS`, or `reports/board-snapshot-2026-11-24/…` next to the repository) —
what the public Cursor solutions scored on that problem under v1.0. That last number is data
about difficulty, never a target.

**Metrics are not comparable across sources** (`sol_score` from the board, `speedup_vs_expert`
from the AKO4X archives, a percentage from RSI-Kernel-Tune's README). `find` prints the metric next
to the number and sorts by the number as recorded; do not read 1.4 (a speedup) as better than
0.95 (a SOL score).

```
forge lib show <id> [--json]              # the entry, its meta.json, the head of its kernel.py
forge lib score <problem> --dataset DIR --card DIR [--ids a,b] [--mode dev|target]
```

`score` measures through `rsi_kernel_tune.benchx` — **the only evaluator**, per the interface
contract. benchx is a soft dependency: until its `CardBench` exists, `forge lib score` exits 2
saying "benchx is not installed" rather than faking a reading. `--card` is required because the
`Bench` protocol evaluates against a frozen leaderboard card (`tasks/<op>/leaderboard/card/`);
a dataset problem directory alone is not enough (there is no card in it). Each reading is
written back into the entry's `meta.json` (`verified_local`: machine, GPU, mode, correct,
`mean_s`, `fast1`, `cold_start_s`, confidence tags, integrity, per-workload readings) and the
index is rebuilt. A candidate the card rejects as a signature mismatch records
`signature_check: "mismatch"` with the evaluator's own words — that is exactly the entry
`adapt` should be run on.

## 5. The generators: `baselines`, `cover`, `adapt`

### `baselines` — the torch-only starting points of one definition

```
forge lib baselines <problem> --dataset DIR [--out DIR] [--rewrite --task tasks/<op>|--agent-cmd CMD]
```

Eight candidates, all pure text: the definition's reference wrapped in the submission contract
(`ref`), the same through `torch.compile` in three modes (`default`, `max-autotune`,
`reduce-overhead`), and each of those four with a tensor-core precision variant that sets
`allow_tf32` + `set_float32_matmul_precision("high")` around the call and **restores both in a
`finally`** (the global-flag rule of background §5; there is a test that runs the generated
guard with a recording fake and asserts the restore, including on an exception). The compiled
candidates fall back to the eager reference on any exception, so a compilation problem cannot
become a wrong answer. A candidate whose `torch.compile` fails at *first call* still pays that
cost as cold start — the manifest says so.

`--rewrite` asks one teacher turn (`meta/baseline_rewrite_prompt.md`, operator-agnostic) for
candidates built from vendor-library calls, validates the answer (JSON only, every source must
parse, define `run`, and pass the target rules) and retries once with the rejection fed back.
Tests drive both paths with a fake backend; a real turn needs `--agent-cmd`, or `--task
tasks/<op>` whose `[agent]` section names the model.

### `cover` — every problem has a correct submission

```
forge lib cover --dataset DIR --out DIR [--only p1,p2] [--verify]
```

For every `<subset>/<problem>/definition.json` in the dataset this writes
`<out>/<problem>/submission.py` — the collection layout of the submission guide: one directory
per kernel, named exactly as the kernel, one file inside, top-level `run(*args)`. The file is
the definition's own reference (renamed `_klib_ref`, otherwise verbatim) behind a
`torch.compile` fast path; **any exception falls back to the eager reference**. Nothing else is
needed to run it, so a problem with no kernel at all still has a valid entry.

What this guarantees is a *valid* submission, not 0.5: the board scores the reference's own
latency against its T_b / T_SOL table, and the board's own data shows reference
implementations anywhere from ~0.16 to 0.5 on their problems. `--verify` re-reads the layout
and reports any directory that is not one problem / one `submission.py` / a top-level
`run(*args)` / has no try-except fallback.

### `adapt` — a reference becomes this problem's starting point

```
forge lib adapt <problem> --seed <entry id | kernel.py> --dataset DIR [--card DIR] [--rounds 3]
                 [--task tasks/<op>|--agent-cmd CMD] [--sol-board JSON] [--seed-license SPDX]
```

The signature almost never matches and the semantics often differ, so a same-family reference
cannot be used as-is. `adapt` runs one teacher turn per attempt (`meta/seed_adapt_prompt.md`,
operator-agnostic — invariant L1), with the definition, the mechanically extracted rules
(signature, input/output order and dtypes, tolerance, optional T_b/T_SOL from a saved board
record), the seed kernel, the target profile and the previous attempt's feedback all passed as
**files** the prompt names, never as prose in the prompt. The answer is JSON only; the
`submission.py` in it must parse, define `run`, and pass the target rules (architecture and SM
count derived at run time). Then, when benchx and a card are given, the candidate is judged by
the card: a rejection feeds the evaluator's own words and per-workload readings into the next
round (a fresh turn, at most `--rounds`). Without benchx the candidate is only checked and the
report says `evaluated: false` — an unchecked candidate is never presented as a verified one.
A pass writes `kernels/<id>/{kernel.py,meta.json}` with `origin: adapt`, `derived_from`, the
approach the model stated, and `verified_local` when it was measured; the license of the
derived file is the seed's when the seed allows it, else `--seed-license`, else `unknown`
(blocked from submission).

## 6. What is not covered (and why)

* **No real measurement here.** `score` and a benched `adapt` need the GPU, benchx and a card.
  On this machine only the syntax/entry/target-rule checks and the extraction tests run. The
  report lists the development-machine and board-only items.
* **`signature_check` is `pending` for everything built.** Whether an entry's call signature
  matches a problem's definition can only be decided against the dataset (on the machine that
  has it) or by a Bench run; nothing here guesses.
* **Fetched libraries are not normalized.** `fetch` registers a tree; writing extractors for
  FlashInfer/FlashMLA/DeepGEMM/FlagGems/vLLM/SGLang is later work, and the pinned commits came
  from KernelWiki's provenance records where one existed (two sources are unpinned and `fetch`
  refuses them without `--allow-unpinned`).
* **KernelWiki artifacts are reference bundles, not submissions.** They are fragments of
  upstream PRs (often C++/CUDA without a binding); `runnable: false` marks them and `adapt` is
  their intended consumer. Their family key comes from the artifact path, the wiki page's tags
  and the captured file names — good enough for ranking, not a classification.
* **The multi-file merge is syntactic.** A merged file parses and its semantics are preserved
  where the sources use normal imports; a source that loads siblings from disk at run time
  needs `adapt` (the merge says so in `notes`, and the RSI-Kernel-Tune DeepSeek-V4.1 kernel is
  bundled rather than merged).
* **`cover` cannot rescue a reference that itself raises.** The fallback is the reference; if
  the reference fails on legal input, every submission fails there.
