# The experiment matrix: auditing accepted speedup verdicts of a self-evolving kernel agent

> **Anti-fabrication preamble** (extends `docs/paper/README.md`; it binds every number in the
> paper): every number in the paper comes from this framework's own measured runs, recorded in
> run archives and rollup bundles that live on disk. Upstream ForgeKernel's published numbers
> appear only as cited upstream results, never as this framework's own measurements. Every
> GPU-hour figure in this document is an **ESTIMATE** — a planning input computed from stated
> assumptions, labeled as such, and replaced by measured day-1 values in the cost log
> (`docs/paper/repro.md`). No benchmark result may be invented, projected, or interpolated
> anywhere in the paper bundle.

## 0. The question

Upstream ForgeKernel (and this framework, its fork) accepts a kernel change only when a timed comparison
says so; every "KEEP" is an accepted speedup verdict. The paper asks: **do the accepted verdicts
survive an honest re-measurement?** — and, for the verdicts that do not, separates "the claim was
too small to measure" from "the claim was wrong", without ever condemning a verdict the evidence
cannot resolve (the **zero-false-condemnation protocol**).

The unit of analysis is the **accepted verdict**: one (task, seed, measurement) row with a claimed
speedup, a re-measured speedup, a measured noise floor, and one `audit_class` label from the
shared taxonomy (section 2). Verdicts are produced by four arms (section 3); they are classified
by one pre-registered decision rule (section 4); they roll up into the paper's tables
(section 5); their statistics are fixed in section 6.

## 1. The shared audit machinery

All four arms classify verdicts with the same stack and the same two tools:

- **The honest measurement stack** (what a verdict must survive — all of it already in the
  framework): mirrored paired A-B timing (`eval/timing.py`, ref / kernel / kernel / ref); the
  device window with its host-bound audit (`src/rsi_kernel_tune/timing_audit.py`,
  `HOST_BOUND_RATIO_WARN = 0.5`, host-free bases = CUDA-graph replay and CUPTI per-launch spans);
  per-reading GPU-state telemetry (`sm_clock_mhz`, `mem_clock_mhz`, `temperature_c`, `power_w`
  via NVML); the clock-drift guard (two readings more than 2% apart in SM clock, or 5 °C in
  temperature, are not one measurement condition — a flagged reading produces **no verdict**);
  the noise-floor estimator `resolvable_diff` (`src/rsi_kernel_tune/loop/parse.py`:
  `clamp(3 × max(σ_within, σ_between), 0.005, 0.015)` with σ = 1.4826 × MAD and a minimum of 4
  paired ratios, falling back to the 1.5% cap when the data is thin); the correctness judge with
  its GPU-calibrated tolerance; the fresh re-check; and the WP20 measurement-side sentinels
  (`eval/benchkit.py`: timer / proc / window sentinels installed before the kernel module loads).
- **`rsi-kernel-tune audit <run>`** (landed: `src/rsi_kernel_tune/audit.py`) writes the run's
  audit report at **`<run>/operators/<task>/student/audit.json`** (`schema_version = 1`; beside
  the trajectory it describes — the module docstring documents the path deviation from the
  drafted `ws/student/` location): one row per trajectory verdict with the
  measurement-integrity readings that verdict was backed by and its `audit_class`, plus the
  run's **false-condemnation ledger** (a defense fired AND that iteration's kernel nevertheless
  became the confirmed best — expected 0; a nonzero tally is a red finding, reported as data,
  never hidden). The row schema is the **shared schema**; every producer in this paper — the
  audit lane, `scripts/retime_corpus.py --collect`, the Arm D readers — emits rows with the
  same `audit_class` vocabulary, and `scripts/audit_rollup.py` merges every run's audit.json
  into one `audit-bundle.json` keyed by **(task × profile × seed)** (an unknown class is a hard
  error: silent schema drift must not poison the paper's numbers). Profile vocabulary used everywhere in the paper:
  - `full` — the honest stack as configured by default;
  - `baseline` — the measurement-stack control arm (the `[profile baseline]` off-switch bundle;
    byte-identical jobs except the stack, pinned by `tests/test_baseline_profile.py`);
  - `upstream_style` — the released-corpus re-timing condition that reproduces upstream's own
    published currency: device time, median over pipelined back-to-back launches, no paired
    interleave and no drift guard (`kernels/README.md`, "Environment").

## 2. The `audit_class` taxonomy

Six classes, one per verdict row. `sentinel_flagged` and `unverifiable` are **no-verdict**
classes (the row carries no measurement claim); `noise_level` and `unconfirmed_edge` are
**non-confirmations** (recorded, never condemned); `survives` upholds; `mismatch_suspected` is
the **only condemning** class.

| class | meaning |
|---|---|
| `survives` | The claim replicates: the re-measured speedup is resolvable above parity and its resolution interval intersects the claimed band. (The landed `audit.py` applies the same vocabulary to run verdicts: a KEEP whose margin clears the run's own `resolvable_diff`, with every defense reading clean.) |
| `noise_level` | The claim is smaller than the floor this re-measurement can resolve (either the claimed margin itself sits under the floor, or the re-measured interval straddles parity). Neither confirmed nor condemned: on this hardware/harness the verdict is indistinguishable from noise. (Run-side: a KEEP with `margin_pct ≤ resolvable_diff_pct`.) |
| `unconfirmed_edge` | The edge, not the sign, is unconfirmed: the re-timed magnitude misses the claimed band on one side, or (run-side rows) a KEEP in the judge-edge confirmation band carries no confirmation jobs. |
| `mismatch_suspected` | The evidence resolves the claim false — the re-measurement resolves the **opposite sign** beyond the floor, or a run-side defense reading suspected the recorded win (replay probe suspected / fresh re-check failed as a cache replay). The only condemning class; it additionally requires clean GPU-state telemetry and no sentinel trip on the contributing readings. |
| `sentinel_flagged` | A WP20 measurement-side sentinel tripped (or the correctness judge alarmed) during the re-timing: the reading is void, the row names the sentinel, no classification from the numbers. Re-run. (Run-side: stream-sentinel violation, timer-identity mismatch, or proc-census anomaly.) |
| `unverifiable` | No verdict possible: the kernel failed to build on the re-timing hardware, a dependency is missing, the architecture does not match, the harness refused, the mechanisms were off (a `baseline`-profile row), or no clean paired ratio exists at all (the floor-calibration phase is missing and the timing job carries no usable paired ratio). Recorded with the reason; never condemned. |

The landed `src/rsi_kernel_tune/audit.py` implements the run-side rule verbatim over a run's
trajectory (its docstring states the order); `scripts/retime_corpus.py --collect` implements
the re-timing-side rules of section 4 over Arm A's outputs — same vocabulary, one schema.

## 3. The four arms

### Arm A — artifact audit (NO API cost)

**What.** Re-time the released kernels in `kernels/` — the B200-targeted ones — under two
conditions, and compare each against the claimed provenance numbers:

1. `upstream_style`: the plain amortized device window, median over back-to-back launches — the
   currency `kernels/README.md` publishes in ("Times are device time (median over pipelined
   back-to-back launches)"). This condition asks: *does the claim hold in its own currency?*
2. `full`: the honest stack (section 1). This asks: *does the claim hold under a measurement
   that defends itself?*

**Corpus.** The five B200 kernel directories (the other sixteen released kernels are H100
`sm_90a`-targeted and out of scope on a B200 box):

| kernel dir | claimed | vs | source |
|---|---|---|---|
| `kernels/attention/attn_bwd_mha/b200` | +2.5% | FlashAttention-4 | upstream Fig. 3 |
| `kernels/sparse_attention/dsv41_fused_attn_rope_cast/b200` | 1.104–1.105× | FlashMLA | after the paper |
| `kernels/linear_attention/gdn_prefill_varlen/b200` | +5.8% | Kernel Design Agents (KDA-v0.5) | upstream Fig. 4 |
| `kernels/linear_attention/recurrent_kda_prefill/b200` | +0.4% | CAKE | upstream Fig. 4 |
| `kernels/gemm/fp4_grouped_gemm/b200` | +0.6% | DeepGEMM | upstream Fig. 3 |

All five claims are **cited upstream results** (the claimed column above quotes
`kernels/README.md`); the reference implementations they were measured against are not part of
this repository. Arm A therefore re-times the released kernel against **this framework's own
reference for the same frozen contract** (scaffolded with
`rsi-kernel-tune init <op> --from-kernels`), and the comparison "measured vs claimed" is a
comparison across harnesses: the release notes the upstream harness and baseline for each claim,
and `scripts/retime_corpus.py` records both so no cross-harness number is silently mixed.
Upstream's B200 environment is pinned in `kernels/README.md` (CUDA 13.0, PyTorch 2.13,
`nvidia-cutlass-dsl` 4.6.0/4.6.2; the DSA kernel is CUDA C++ and needs `nvcc` with
`sm_100a`); day 1 must reproduce that environment before any Arm A job (section 1 of
`repro.md`).

**Protocol.** Per kernel: scaffold the task from the released bytes, one run per kernel; per
**seed** s ∈ {0, 1, 2}: one floor-calibration phase (R = 6 unchanged-tree repeat jobs —
`resolvable_diff`'s own sample count — establishing the per-kernel floor from paired ratios),
then one honest (`full`) paired timing job and one `upstream_style` timing job. Collect with
`scripts/retime_corpus.py --collect` (offline; parses the finished re-timing outputs into
shared-schema rows). No agent runs in Arm A: the released kernel bytes are the student file,
so **no API cost**.

**Budget.** ESTIMATE. Assumptions: one timing job ≈ build + JIT + bench; on day 1 the framework
has not yet measured `t_job` on B200, so assume 15–40 min/job (CuTe DSL JIT and the CUDA C++
DSA build dominate). Jobs ≈ 5 kernels × [3 seeds × 2 timing jobs + 6 short floor repeats]
= 5 × 12 = 60 jobs ⇒ **≈ 15–40 GPU-hours**. Replaced by measured numbers in the cost log after
the day-1 calibration job.

### Arm B — upstream behavior (API cost)

**What.** Run **Upstream ForgeKernel at commit `5030503d3ff7`** (the paper's baseline commit;
the object exists locally in this clone — verified with `git cat-file -t`) on
**N = 5 operators × 3 seeds**, and audit **every accepted verdict** with the honest stack. This
is the arm that measures the artifact under audit itself: the upstream loop's own judgment,
re-examined verdict by verdict.

**Operators.** The five B200 corpus operators of Arm A (same frozen contracts), so verdicts roll
up against the same task names. If upstream's card surface cannot accept one of the five, that
operator is recorded `unverifiable` for the affected seeds — never silently dropped.

**Protocol.** Check out upstream into a worktree
(`git worktree add ../forge-5030503 5030503d3ff7`); configure its runs (its own config surface,
its own budget hours per run); one run per (operator × seed). Each run is audited two ways:
(i) **`rsi-kernel-tune audit <run>`** — the landed read-only provenance report over the run's
persisted trajectory: one row per verdict with the readings it was backed by (margins vs
`resolvable_diff`, fresh re-check, replay probe, sentinels) and the run's false-condemnation
ledger; (ii) one **honest-stack re-timing job** of each run's confirmed-best kernel on the B200,
classified by the section 4 rules with the run's confirmed margin as the claim band — so the
run's own telemetry and an independent box-level measurement both speak. Seeds are
agent-sampling seeds recorded in each run's configuration; three independent runs per operator,
pre-declared, no additions after the first result is seen.

**Budget.** ESTIMATE. A self-evolving run is the expensive unit: upstream cards budget a few
hours of GPU per run; 15 runs ⇒ **order 30–75 GPU-hours** plus agent API spend (teacher +
student turns per round). The first completed run replaces this guess in the cost log. API
budget: models and keys via `rsi-run.toml [api]` + `local.env` only (`repro.md`, section 3);
token spend per arm logged, never guessed.

### Arm C — controlled: the measurement stack is the only variable (API cost)

**What.** On the **same framework** (this clone, HEAD), the **same task set** (`tasks/b200-set`,
landed: ten operator task cards, `tasks/b200-*`), run **`profile = baseline` vs
`profile = full`**. The only variable is the measurement stack; byte-golden equivalence of
everything else is pinned by `tests/test_baseline_profile.py` (the `[profile baseline]`
off-switch bundle, still landing). Arm C isolates how many verdicts the honest stack itself
flips — the counterfactual "what would the loop have accepted anyway?" that separates "the
agent gamed the window" from "the window changed the game".

**Protocol.** Same task set, same seeds s ∈ {0, 1, 2}, one run per (task × profile × seed);
every accepted verdict from both profiles audited (`rsi-kernel-tune audit`); the rollup groups
(task × profile × seed) align so a verdict flips if its `audit_class` at `baseline` and `full`
disagree in confirmatory direction. `baseline`-profile rows classify `unverifiable` by rule
(the mechanisms were off) — that is expected, and the flip analysis reads the run-level
acceptance comparison, not those rows' classes.

**Budget.** ESTIMATE. 2 profiles × 10 tasks × 3 seeds = 60 runs × the per-run hours of the
task cards ⇒ **order 60–120 GPU-hours** plus API spend. ESTIMATE; refined from the first
completed (task × profile × seed) triple in the cost log.

### Arm D — defense baselines (external tools, no integration)

**What.** Run two external defense approaches over **all collected kernels** (Arm A's five
released kernels + every accepted student kernel from Arms B and C) and build a confusion
matrix against the honest-audit ground truth. These are **external tools**: documented, cloned,
and run in their own harnesses; nothing is integrated into this framework, and no external
harness ever produces a row in `audit.json` directly.

1. **KernelGuard-style static/physics checks** — upstream `gpu-mode/kernelguard`
   (<https://github.com/gpu-mode/kernelguard>): a rule-based kernel-hack detector built on
   expert-seeded static rules and conservative physics-floor checks (its design note is the
   OpenReview submission "We Let Agents Compete and They Tried to Cheat",
   <https://openreview.net/forum?id=K0dAC7UaXL>). Obtain: `git clone` the repo; pin the commit
   used in `docs/paper/runs/<slot>/env/`; run per its README's CLI over the collected kernel
   sources.
2. **Wafer-style fixed-threshold checks** — the Wafer published approach to benchmark
   verification: a fixed "faster than baseline by ≥ threshold, repeated N times" acceptance
   rule and the reward-hack checks described in their field guide and case study
   (<https://github.com/wafer-ai/gpu-perf-engineering-resources>,
   <https://www.wafer.ai/blog/reward-hacking>). Pin which checks are reproduced (the fixed
   threshold, the repetition count) in the run slot's README before running, since the checks
   are documented rather than shipped as a single tool.

**Ground truth.** The honest audit's classification, collapsed to binary for the confusion
matrix: a verdict **fails replication** iff `audit_class = mismatch_suspected` (the only
condemning class); every other class counts as "not condemned". Both external judges are scored
on the same collapse, and their false-positive behavior on verdicts the honest stack marks
`survives` is reported — that is the paper's statement about what fixed rules cost.

**Budget.** ESTIMATE. Static checks are CPU-side (≈ 0 GPU-h). The fixed-threshold checks
re-time kernels; they **reuse** Arm A/B honest timing jobs wherever the protocol matches, plus
≈ 10–20 short GPU-hours for threshold repeats on kernels Arm A did not time. ESTIMATE; refined
from the collected-corpus size after Arms A/B/C.

**Order of arms on day 1** (rationale in `repro.md`): calibration → **A** (no API, warms the
harness, exercises the audit lane) → **C** (needs the baseline-profile bundle) → **B** (the
long API arm) → **D** (offline, needs the corpus).

## 4. Decision rules (pre-registered)

Let, for one verdict row:

- `c` = the claimed speedup band `[c_lo, c_hi]` (fractions; a single claimed number is a
  degenerate band; Arm A's ratio claim 1.104–1.105× becomes `[0.104, 0.105]`);
- `m` = the re-measured speedup (fraction, signed; positive = kernel faster than its reference);
- `f` = the measured noise floor of the re-timing (fraction), computed by the
  `resolvable_diff` arithmetic from the floor-calibration ratios — never below 0.005, never
  above 0.015; with 1–3 clean paired ratios it falls back to the 0.015 cap (the estimator's
  data-poor fallback, never a separate mode — the shared constants and arithmetic live in
  `src/rsi_kernel_tune/loop/parse.py` and are re-implemented verbatim by
  `scripts/retime_corpus.py`);
- `I = [m − f, m + f]` = the resolution interval of the re-measurement.

Classification, in precedence order (first match wins):

1. **`sentinel_flagged`** — any WP20 sentinel block or judge alarm in any job contributing to
   the row. The numbers are not read.
2. **`unverifiable`** — build/runtime failure, missing dependency, architecture mismatch,
   harness refusal, or **no** clean paired ratio at all (the floor-calibration phase is
   missing and the timing job carries no usable paired ratio). Recorded with `why`.
3. **`mismatch_suspected`** — `m + f < 0`: the interval lies entirely below parity, and the
   row's GPU-state telemetry is clean, the fresh re-check passed, no sentinel tripped. Anything
   less clean falls back to `unverifiable`, never to condemnation.
4. **`noise_level`** — `c_hi < f` (the claim is unresolvable by construction on this floor), or
   `|m| < f` (the reading straddles parity).
5. **`unconfirmed_edge`** — `m ≥ f` but `I ∩ [c_lo, c_hi] = ∅`: the sign replicates, the
   magnitude misses the band on one side (shortfall or excess).
6. **`survives`** — `m ≥ f` and `I ∩ [c_lo, c_hi] ≠ ∅`.

**Zero-false-condemnation**: only rule 3 condemns, and it requires a resolvable sign flip
measured under a clean, defended stack; every weaker outcome is recorded as a
non-confirmation. The rule set above is fixed **before** any Arm A/B/C job runs; changing it
after results are visible is forbidden (any change restarts the affected arm from scratch with
a new seed declaration in the run slots).

## 5. Headline tables and their rollup groups

Every table reads merged bundles produced by `scripts/audit_rollup.py` (the
`audit-bundle.json` bundle; group rows carry counts AND rates per class); the group key is
(task × profile × seed). No table reports a number that no bundle contains.

| table | arm | rollup group → table content |
|---|---|---|
| **T1 — Released claims under two stopwatches** | A | (task = each of the 5 corpus kernels, profile ∈ {`upstream_style`, `full`}, seed ∈ {0,1,2}) → per kernel: claimed band vs measured (both conditions), `audit_class`, per-kernel floor. |
| **T2 — Accepted verdicts under the honest audit** | B | (task = each operator, profile = `full`, seed ∈ {0,1,2}) → per operator: verdicts accepted by upstream, `audit_class` distribution, survival rate. |
| **T3 — What the measurement stack changes** | C | (task = each `tasks/b200-set` task, profile ∈ {`baseline`, `full`}, seed ∈ {0,1,2}) → per task: verdicts accepted per profile, flips attributable to the stack. |
| **T4 — Defense baselines vs the honest audit** | D | union corpus of A/B/C rows (collapsed binary truth) × {KernelGuard, Wafer-style} → confusion matrix, FP rate of each defense on `survives` verdicts. |
| **T5 — Cost** | all | per-arm GPU-hours (from the cost log) and API spend; estimates vs measured, both shown. |

## 6. Statistical treatment

- **Per verdict**: the margin is judged against the verdict's own measured floor
  (`resolvable_diff` family), never against a universal epsilon; the floor rides every row.
- **Rates** (survival rate per operator, flip rate per task, defense FP rate): reported as
  k/n with **exact Clopper–Pearson binomial confidence intervals**; n is the row count in the
  rollup group, stated beside every rate. No normal approximations.
- **The auditor's own false-positive rate** comes from the **red-team campaign controls**: the
  red-team fixtures (`tests/redteam/`, contract C16 — each cheat names the defense expected to
  catch it, written before the attack) run through the campaign runner
  (`scripts/redteam_campaign.py --gpu --out campaign-report.json`, which measures verdicts —
  caught / missed / unknown / **false-positive** — never milliseconds). A known-good control
  flagged by any defense is recorded `false_positive`; the audit's FP rate on controls is the
  count of those over the campaign, reported with the same binomial CIs. An honest-audit
  condemnation rule whose control FP rate is not bounded by these controls is not publishable.
- **No selection**: all seeds pre-declared ({0, 1, 2}); all runs reported, including
  `unverifiable`/`sentinel_flagged` rows (counted, with reasons); no post-hoc seed exclusion.
  Node-bimodality is real at the ±6% level on this framework's history, so cross-job claims
  compare median vs median across jobs, never single jobs.

<!-- SPDX-FileCopyrightText: Copyright 2026 The RSI-Kernel-Tune Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
