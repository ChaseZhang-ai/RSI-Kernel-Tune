# Changelog

RSI-Kernel-Tune — a self-evolving GPU kernel optimization framework, developed as a fork of
OpenBMB/ForgeKernel (@5030503) across three improvement generations plus a deboarding
correction. Each entry below is an English digest of the corresponding work package's
delivery document; the original per-package documents (Chinese) were removed from the tree
by WP25's Englishization adjudication and remain permanently recoverable from git history
(`CHANGES-WP<n>.md` at the repository root for gen 2/3, `docs/wp-changes/CHANGES-WP<n>.md`
for gen 1 — see any pre-WP25 commit). The same convention applied to the post-WP25
per-package records (`CHANGES-M*`, `CHANGES-WP20R/25/27/29`, `CHANGES-corps-fix*` at the
repository root): distilled into the digests below and removed by the RSI-Kernel-Tune
rebrand commit — the originals are permanent in git history.

Full-suite regression counts quoted per entry are the package's own acceptance evidence at
its time; the tree's current state is 2692 tests green (the M3 lane-wiring merge, 2026-11-04,
re-verified on the audit branch at `e18e67a`).

## Generation 1 — WP1–WP6 (baseline: upstream ForgeKernel @5030503, 1966 tests)

### WP1 — Measurement statistics upgrade (2128 passed; +44 tests, `tests/test_wp1_measurement.py`)

Judging resolution stops being a 1.5% constant and becomes the run's own measured quantity:
`resolvable_diff(records, gate_window, *, floor=0.005, cap=0.015)` (contract C1,
`loop/parse.py`) estimates resolution as 3x the rolling MAD of in-job paired ratios over the
last K=6 evaluated records, clamped to [floor, cap]; `detect_plateau` takes
`margin = min(noise_floor, resolvable_diff(...))`, and the gate-regression tolerance and
patch-improvement epsilon consume the same estimator (constants kept as caps/fallbacks).
`EvalResult` keeps per-window raw readings (`device_readings_ms`, `ref_device_readings_ms`)
plus `device_spread_pct`; `IterationRecord.gate_spread_pct` aggregates run-level. A
suspicious-winner confirmation band triggers a budgeted confirm job for improvements inside
(resolvable_diff, watermark_margin). Clock/temperature/power telemetry lands as eval-JSON
`gpu_state` (`timing.py sample_gpu_state/record_gpu_state`, pynvml with
nvidia-smi fallback, failure-degradable) and a `clock_drift` flag forbids a watermark set
under >2% SM-clock drift from standing alone. `task.toml` gains an optional `[timing]`
protocol table (device default; a report-only secondary protocol was removed in WP24).
Config keys: `resolvable_floor`, `resolvable_cap`, `confirm_budget_per_iter` (card loop.toml).

### WP2 — Per-iteration telemetry expansion (SOL / L2 / waves / coalescing + build resources)

NCU metric set 18 -> 25: compute/memory SOL %, `lts__t_sector_hit_rate` (L2 hit rate),
`launch__waves_per_multiprocessor` (tail waves) and coalescing metrics, each annotated with
its diagnostic meaning and pass cost. `filter_metrics_for_device` adapts the collection set
per architecture (launch attributes kept unconditionally; a failed/empty query passes
through rather than silently dropping regs/smem/waves). Build-resource telemetry
(`resusage`) via `cuobjdump -res-usage` records registers/shared-memory per kernel. The
digest gains a roofline header line and a fixed diagnostic reading order. (traffic_amp was
skipped by design.)

### WP3 — Knowledge compounding (falsified-lever ledger, probe archive, KB queue)

A cross-run falsified-lever ledger (`hw_kb/falsified_lever.jsonl`) becomes a design/pricing
prior; probes and pricing recipes are archived per run; a KB to-be-certified queue
(`hw_kb/CANDIDATES.md`) is introduced.

### WP4 — Search mechanisms (LANDABLE / SWEEP_REQUEST / slow-decision ledger)

`LANDABLE` lands priced lever combinations in one delivery (teacher-triggered, student
landed, standard gate); `SWEEP_REQUEST` gives the teacher a parameter-sweep channel (the
student-side sweep ban unchanged); slow decisions get a marginal ledger with
`REMOVE_CANDIDATES` diagnostics. Driver carries a C1 shim (`try-import resolvable_diff`)
for pre-WP1-integration bases.

### WP5 — Budget and stopping governance

Format mistakes no longer burn meta budget (`meta_format_retry`); meta budget couples to
remaining wall clock (`meta_hours_per_round`); headroom enters the stopping criterion with a
one-round budget extension (`meta_ext_*`); BLOCKED runs may uncap reopen
(`reopen_blocked_uncap`); `ncu_on_revert` switch.

### WP6 — M0 pre-freeze measurement + R2 structural channel (baseline 2054 passed)

`ARCH_PROBE` measures candidate topologies before M0 freeze (`m0_probe_rounds`; the freeze
follows the winner); `R2-structural` requires teacher-proposed skeletons to carry a measured
price (the machine rejects unpriced structural proposals); institutional documents land.

## Generation 2 — WP8–WP12 (baseline: gen-1 integration @c1971f1, 2222 tests)

### WP8 — Prior supply line (2252 passed; all channels default-off, off = byte-identical)

Three teacher-side prior channels: `[agent.teacher] prior_survey` (public SOTA survey via
`skills/sota-survey/`), `prior_art` (offline archive via `docs/prior-art/`), and `carry/`
(cross-generation carry). A repository prior library `priors/` (25 operator milestone
decision graphs) plus `scripts/build_priors.py` skeleton generation;
`docs/prior-supply.md` documents the discipline. Every knowledge-injection mechanism must
answer "what does it add over the same model's goal-only baseline" (the JustGoal lesson).

### WP9 — Measurement anchoring & calibration (2276 passed; +54)

NCU protocol alignment: `--cache-control none --clock-control none` at every generation
point (the old readings described a wrong regime). `loop/analytic_floor.py` solves the
analytic floor from bytes/FLOPs machine denominators (`floor_stop`, `[target] floor_margin`,
default off). `fresh_recheck` re-checks the best kernel on fresh inputs after timing: a
non-reproducing reading hard-REVERTs and is marked `suspicious`. Reanchor memo persists
across restarts.

### WP10 — Structural search deepening (2246 passed; +24)

`STAGNATION` architecture-escape trigger: `stagnation_k` consecutive in-band attempts inject
a structural-route constraint (`stagnation_override` pressure valve; adds pressure, never
changes verdicts). `multi_candidate:` MC-DELIVERY: the student fans out structures inside one
job with a prescreen (smoke -> interleaved timing -> MC-WINNER), the winner alone passes the
full gate. Compile-flags become a legitimate decision axis (whitelisted levers).

### WP11 — Process speedup (2260 passed; zero-behavior golden pin)

eval+NCU merged into one job (`merge_eval_ncu` default on, `false` returns two jobs; the
merged path's per-segment budget = the sum, never a single segment's) with
`test_switches_off_matches_baseline_records` pinning field-by-field baseline equality when
all switches are off. Content-addressed build cache (`PK_BUILD_CACHE`, default off). Payload
input-tensor content-addressed sync for the direct backend. Five wall-clock composition
telemetry fields (agent_turn/eval_job/ncu_job/teacher/queue_wait — pure measurement, no gate
reads them). Eval failures carry an error tail. SPRT window-boundary early stop
(`sprt_early_stop`, default off; only stops early, never manufactures a KEEP).

### WP12 — Knowledge injection & KB lifecycle (2287 passed; +65)

`loop/kbmenu.py` — the single deterministic KB-MENU synthesizer (contract C7): KB-HINTS
(a 10-rule symptom match table), LEVER-MENU and CALIBRATION sections, <=30 lines, empty
sections never render, all-empty renders nothing. KB entries gain staleness and dead-end
propagation across runs. `skills/probe-recipes/` ships pre-built probe recipes (occupancy
curve / phase ablation / effective bandwidth). Re-check fixed the `occupancy_scaling`
step-direction reading (physically inverted).

## Generation 3 — WP14–WP21 + WP24 (baselines: gen-2 integration @359696b, 2437 tests; post-WP24 main @19e885b, 2332 tests)

### WP14 — Shape-regime decision channel (2457 passed; +20)

Per-shape ledger: every workload of a group card keeps its own best and vs-best ratio (the
C8 single definition). `per_shape:` / `regime:` decision-line vocabulary: regime predicates
are pure functions of the shape axes, families must be mutually exclusive and fully
covering, otherwise the freeze is invalid; `milestone.py` gains five new pure parsers
(`parse_regime_predicates` / `parse_regime_expr` / `regime_expr_matches` / `regime_coverage`
/ `parse_per_shape_declarations`). Per-regime NCU evidence capped at `regime_ncu_max` per
meta step. MC-DISPATCH: per-workload structural prescreen split across jobs.

### WP15 — Resolution, confirmation & measurement statistics, gen 3

`resolvable_diff` v2 (contract C8; rolling-MAD estimate with the consumption-side sync in
`detect_plateau` as the largest implementation trap). Confirmation becomes routine: an
independent confirmation band with per-reading budget visible in the trajectory
(`confirm_jobs` / `confirm_readings`). SPRT early-stop flips to local default on backtest
evidence (family-cell cost 1.35% < the 2% gate; the only-early-loss invariant retained and
checked line by line). Stouffer per-workload combined test (advisory). Log-domain best
confidence (standard error / low-confidence flag via `parse.best_standard_error`). Six
hardcoded constants migrated to configured values; gate-only confirm mode; numeric FAIL
feedback carries the top-k worst check rows (line-A G6).

### WP16 — Active search & population governance (2463 passed; +26; `dry_run_wp16.py` artifacts)

Shrink-grid pattern search on STAGNATION (a driver-measured readings table, same-job
ratios). Noise-gated SWEEP successive halving (spec key `"halving": true`) + range recipes.
MC near-tie recheck defense. Sub-noise compounding pool (`loop/lever_pool.py` COMBO-POOL:
null-priced levers whose combinations were never measured get one priced all-on bundle; a
null bundle closes the combination). k-best runner-up pool + STAGNATION restart targets +
architecture graveyard + hybrids. EXPIRED reprice coverage, climb streak, lever coupling
graph. `floor_progress` advisory line (this run's parse floor over its own best).

### WP17 — Screening primitives & M0 fidelity

`loop/screening.py` is the single screening primitive (contract C9: frozen-spec validation,
paired ordering, node-dirty handling, analytic-floor advisory pre-ranking; SWEEP_MAX_VARIANTS_HALVING=32).
ARCH_PROBE gains fidelity tiers, a semi-real tier and a near-difference single re-measure;
`(family, arch)` readings ledger. In-process variant switching. CANDIDATES gets a scheduled
certification quota (opt-in default off).

### WP18 — Knowledge closure & beyond-reference supply

`kb_used:` student citation rows become a per-entry ledger with KEEP/REVERT feedback;
memory-bound/BLOCKED verdicts without an `effective_bw` citation bounce once. CANDIDATES
gains a batch-certification duty block and regime fields; surveys must emit
`CANDIDATE-DRAFT:` rows for uncovered mechanisms; harvest gains a survival-weight section
(`survival: <n operators>`). Kernel artifact collector (`scripts/collect_kernel_artifacts.py`
-> `build_priors --from-artifacts`) fills the prior skeletons with measured anchors.
Null levers get a PRICED_NULL stamp; lever x workload price matrix artifact. kbmenu hit
telemetry + recon KB_DIR binding (P2).

### WP19 — Process speedup, gen 3 (zero-behavior golden against @19e885b)

Build-cache key narrowing (contract C11): key = sha256(kernel source set + the
`fk-cuda-cflags` line + recursive include closure + setup/build scripts + nvcc + torch
version), manifest per-artifact sha recheck; `PK_BUILD_CACHE_KEY_NARROW` — comments change
must HIT, header change must MISS; any anomaly falls back to the whole-tree key. Probe
batch manifest (whole-batch failure falls back to per-script). Idle-GPU background
measurement lane (contract C15; never enters a gate). Teacher/student context bounding
(contract C13: digest binding with the `student_digest_binding` kill switch, default off;
NCU tail line cap). Diagnose gate merges into a single re-prompt with a structured bounce
(contract C12). Session resume + structured outputs. Meta collection merged into one job;
cross-generation carry grows to seven artifacts; ref-reading reuse requires one cheap
confirmation job ("reading reuse is acceleration, not measurement-skipping").

### WP20 — Red team & defenses (assets all under `tests/redteam/`, contract C16)

A red-team cheat fixture corpus plus measurement-side sentinels: stream-span audit,
timer identity, post-timing thread/child census — violations convert into fresh_recheck
failures; unknowns degrade, nothing is condemned on suspicion alone (accounting only).
Gate/judge accounting (zero verdict changes): `gate_degraded` (precision-gate quietly-blind
share), `corr_margin_ratio` (judge-edge evidence), edge-band first-offense re-check with a
different seed, `fresh_recheck_form` (nondeterministic vs cache replay),
`board_span_unstable` (reference-side mirror divergence, report-only). The precision-class
fixtures become the standing defense of the user-precision contract.

### WP21 — Energize & dataflow audit (the gen-3 acceptance substrate)

`tests/test_dataflow_audit.py`: every IterationRecord / EvalResult field and telemetry key
must have a reader (grep-level, tiered; registries
`tests/dataflow_audit_baseline.json` / `dataflow_audit_whitelist.json`, self-maintaining —
stale entries fail the test). Mini smoke task `tasks/wp21mini` (chain-and-dataflow evidence
on the local 3060; red line 11: never performance conclusions). The energize and harvest
protocol (`docs/energize-protocol.md` + `scripts/energize_ab.py` skeleton): an A/B matrix
with one-default-at-a-time flipping, evidence-before-flip, separate rollback switches, and
the JustGoal comparison institutionalized as a fixed acceptance action.

### WP24 — Deboarding & repositioning (2437 -> 2332; Wave 3.0 pre-package)

The leaderboard-facing machinery is removed in one cut: the two-suite supply line, the board
API fetch (the tree's only HTTP client), the board evaluation template (the 3020-line
`leaderboard.py` becomes `benchkit.py` after stripping the board protocols — it remains the
ordinary-task judging/timing engine), leaderboard-pinned clocks, and the leaderboard docs.
Card data layout `eval/leaderboard/` -> `eval/problem/`; eval-JSON keys `leaderboard` ->
`bench`. Lock-clock refusal markers de-branded. NOTICE reduced to the retained derivations
(file-by-file mapping). The framework's positioning: a pure AI kernel-optimization framework
with zero external-benchmark association and zero network egress on the evaluation path.
Task supply closes the loop: `peregrine init <op> --from-kernels` scaffolds a frozen-contract
task from a published kernel and seals it as a black-box reference via sealref.
`tests/test_deboard_regression.py` pins the zero-residual gate.

## Integration

### WP23 — Third-generation integration (2026-11-04)

Merged WP15 -> WP19 -> WP17 -> WP16 -> WP14 -> WP18 -> WP20 -> WP21 onto the post-WP24 main
(`19e885b`), with a 13-seam check, one-default-at-a-time settlement (zero default flips in
the integration itself; SPRT stays on per its WP15 backtest, build cache stays default-off
pending its A/B evidence), a mini end-to-end run, and the dataflow audit going green on the
merged tree (baseline entries deleted one by one — "all green after the merge" was the
acceptance). Final state: **2586 tests passed**; measure_paired byte-identical across
generations; `meta/student_prompt.md` gen-1 baseline byte-identical with only the reviewed
per-generation increments. Independent final review: PASS-WITH-NOTES.

### WP25 — Rename & repositioning (this package)

Namespace migration to Peregrine-Kernel (`peregrine_kernel` package, `peregrine` CLI,
`PK_*` env prefix, `pk_*` identifiers) with zero behavior change (2586 tests green before
and after); legal exemption surface pinned by `tests/test_branding.py`; README rewritten;
docs restructured; `CHANGELOG.md` (this file) and `docs/backlog.md` created; the repository
tree made 100% English (the Chinese per-package history files were distilled here and
removed — git history keeps them). See `CHANGES-WP25.md` (git history) for the full mapping tables,
replacement counts and evidence.

### WP20R — Red-team shrink (2601 passed on the branch; baseline 2594 at the WP25 merge)

The red-team matrix's two fills plus its permanent boundary: attack 1c (shape-keyed
cache) as a caught-form fixture; attack 2b (seed-change recon, the "recomputes
honestly on a miss" replay cheat) verified empirically to escape every value layer
and recorded as **open row G1** (`docs/wp20r-known-gaps.md` — a fixture asserts the
documented behavior so the hole stays regression-visible). New
`tests/test_redteam_asset_boundary.py` pins contract C16 permanently: technique
markers never leave `tests/` + the CHANGES records, the matrix never skips, and it
never shrinks below its delivery size. `CHANGES-WP20R.md` (git history) carries the
attack → defense → status table (the paper's measurement-integrity material).

### WP27 — External evaluation export (2615 passed on the branch; +21)

The manual, opt-in, offline evaluation lane (the deboarded framework's only
external-eval path): `scripts/eval-export/pack.py` packages an optimization
deliverable + its task contract into the SOL-ExecBench / FlashInfer-Bench
submission format — zero imports from `src/peregrine_kernel`, zero network I/O,
run by a person by hand, in no automatic loop (pinned by `tests/test_eval_export.py`).
`docs/external-eval.md` records the honest cost structure: the free tier is the
*evaluation*, never the optimization loop (search needs interactive timing; a
submission judge cannot be a timing oracle).

### M1 — The industrial memo lane, design (docs only; branding green)

`docs/industrial-lane.md`: the two-lane contract from the 2026-11-04 user
adjudication. STRICT (default, byte for byte) keeps every anti-cheat check and is
the only source of paper capability numbers; MEMO (opt-in, `task.toml
[protocol] memo_lane`) is an industrial serving lane whose timed number is service
latency over a repeating input stream, with correctness on unseen inputs as a hard
gate. Honesty boundaries explicit: MEMO numbers carry their lane everywhere and are
never a compute-speedup claim. The file is the implementation checklist M3 executed.

### M2 — The memoization demo fixture (2638 passed on the branch; +16)

`tests/fixtures/memo_demo.py`: the reference shape of a MEMO-lane deliverable —
`MemoKernel`, a content-addressed cache (streamed single-pass sha256 per buffer)
whose hit path never recomputes and whose miss path computes honestly; correctness
on any input is the iron rule. `tests/test_memo_demo.py` pins hit/fallback
correctness, key stability, the latency signature and the single-threaded limit.
The module docstring states the lane contract: the same shape is cheating under
STRICT (the WP29 probe exists to catch it) and legitimate only in its declared lane.

### WP29 — The replay probe, the G1 fix (2618 passed on the branch; baseline 2601; user approved 2026-11-04)

Closes G1 (the one measurement-surface package authorized under the freeze order):
after ALL timing windows close, the bench draws an independent third input per
timed group (seed +104729; judged +0, recheck +7919 — disjoint), burns in, and
measures the honest-compute latency in the main timing's device-event protocol.
`probe_median > PK_REPLAY_PROBE_RATIO (default 3.0) x judged_median` →
`replay_probe: suspected` → a fresh_recheck-same suspicious REVERT carrying both
latency numbers. `PK_REPLAY_PROBE=0` is byte-identical to pre-WP29 (golden-pinned);
`unavailable` never convicts; `timing.py` byte-identical. GPU dual-verified on the
3060: the recon fixture caught at x169.38, an honest kernel passes at x0.94.
`docs/wp20r-known-gaps.md` G1 OPEN → CLOSED; the WP20R status table row 4 synced;
backlog C5 closed ("do not re-propose").

### M3 — The two-lane contract wired (2692 passed; baseline 2655 at the WP29 merge)

`docs/industrial-lane.md` §6 executed file by file: `[protocol] memo_lane` (bool,
default false — a non-bool is a `TaskError`); the card stamps `problem.memo_lane`
and emits `PK_MEMO_LANE=1` on MEMO cards only (a STRICT card's submit.toml is
byte-identical, golden-pinned); benchkit's MEMO branch never emits `fresh_recheck`
and attaches the same fresh child's verdict as `memo_fallback` (a FAIL there is a
hard correctness REVERT, not `suspicious`; no reading never convicts), skips the
replay probe (a memoized parent legitimately sits far below its child's honest
re-time) and stamps `lane: {"memo": true}` on the eval JSON only for MEMO cards;
every `IterationRecord` carries `memo_lane`; the digest is single-lane — a mixed
row set raises instead of juxtaposing cross-lane numbers; a MEMO-stamped eval
surfacing in a STRICT run is refused as contamination; `self_evolve` inherits and
stamps the lane, and a mismatching carry is dropped with a log line. STRICT
byte-goldens re-verified on GPU (E1/E2); the same delivery passes clean as MEMO
(E3) and is caught at x87.19 as STRICT (E4). Adjudication notes and the one
procedural `_STRIP_FIELDS` registration are recorded in `CHANGES-M3.md` §4-5
(git history after the rebrand removal).

### corps-fix — the audit-corps correction wave (closeout: 2915 passed; baseline 2692 at `8977b63`)

The 2026-11-05 audit-corps correction round against the post-M3 main: the fix ledger's 61
legacy findings (BUG-SEED-01..50, BUG-COORD-1, BUG-C1/C3/C5/C11, BUG-TST-01) were
re-adjudicated on the current tree with fresh file:line evidence and minimal reproductions
(verdicts: 9 already fixed here, 3 mechanisms removed, 52 alive), plus new-generation bugs
found while re-convicting. Fixes landed in parallel file-disjoint branches — driver/parse
measurement and adjudication, loop CLI/launch admission, guards and the adversarial surface
(the root-`sitecustomize.py` tree hijack, P0), submit/ssh delivery (the flock queue wait no
longer charged to the job deadline), the monitor/resume/trajectory batch (hook dump
traceability, pid-keyed call records, torn-tail tolerance), the parse line (NaN as an
absent reading, line-anchored probe markers), self_evolve and eval-export hardening — one
commit per bug, each with a red-before/green-after regression test. A reconciliation
branch merged the lines zero-conflict and verified the dispositions; the repair ledger's
section records remain the authoritative disposition of record. No frozen surface was
touched.

### corps-fix-r3 — the R3 fix wave (50 open entries dispositioned: 49 fixed, 1 verified already fixed)

Merged the eight file-disjoint R3 fixer branches (`feat/corps-fix-r3a` … `r3h`) closing the
31 R3 recon findings (BUG-CH-*) and the 19 new C-class findings of the RECON-R2 audit
(BUG-CORPS-C-*): driver-level fixes (WP18 PRICED_NULL stamp collisions, non-dict teacher
levers, sweep `rounds` validation, Stouffer live-row reads, anchored gate-row matching),
loop CLI/parse fixes (dry-run no longer persists a hand-written baseline, harness revert
consumes `-z` quotepath-escaped diffs, dead resumed sessions dropped after one hard
failure, UTF-8-pinned config reads), guards/adversarial hardening, submit/monitor/trajectory
fixes, and eval-export/backtest shell hardening — each with a red-before/green-after
regression test and one `fix(audit-corps)` commit (merge commits grouped per fixer branch).

### Rebrand — RSI-Kernel-Tune (this package; 3343 passed, zero behavior change)

The GitHub repository was renamed to `RSI-Kernel-Tune`; this package migrates every brand
token of the Peregrine-Kernel era to the new name — a purely mechanical rename with zero
semantic change (the full suite green before and after; golden anchors pin behavior):
`src/peregrine_kernel/` -> `src/rsi_kernel_tune/` (imports `rsi_kernel_tune.*`), pyproject
`name` -> `rsi-kernel-tune`, console script `peregrine` -> `rsi-kernel-tune`, command-form
references and the `[peregrine]` log prefix -> `rsi-kernel-tune` / `[rsi-kernel-tune]`, the
`PK_*` env prefix -> `RKT_*`, `pk_*` / `pk-*` identifiers and runtime file names
(`pk_monitor_hook.py` -> `rkt_monitor_hook.py`, `.pk-jobs` -> `.rkt-jobs`, the timeline
probe's `_pk_tl_*` CUDA symbols -> `_rkt_tl_*`) -> `rkt_*` / `rkt-*`, brand prose
Peregrine-Kernel -> RSI-Kernel-Tune, and the uppercase prompt labels
(`PEREGRINE TURN`, `PEREGRINE ...` markers) -> `RSI-KERNEL-TUNE ...`. The full-name env
vars became `RSI-KERNEL-TUNE_{OPERATORS,RUNS,LOCK}_DIR` (consumed only via `os.environ` in
the harness and its tests; no shell-export site exists). Legal exemption surface preserved
verbatim: LICENSE, NOTICE, the README fork declaration ("**RSI-Kernel-Tune** is a fork of
OpenBMB/ForgeKernel", baseline `5030503`), the upstream `kernels/` corpus, the frozen
`fk-cuda-cflags` cardkit/benchkit directive, and the `docs/design/v1-launch.md` archive;
CHANGELOG historical entries keep the names of their time. `tests/test_branding.py` now
hunts the old forms (`peregrine`, `PEREGRINE`, `PK_`, `pk_`, `pk-`) with a zero-hit gate
outside that exemption surface, plus an import smoke and a CLI smoke
(`rsi-kernel-tune --version` -> `rsi_kernel_tune <ver>`). The root `CHANGES-*.md` records
were distilled into the digests above and removed from the tree (git history keeps them).
Recorded mapping counts are in the rebrand commit message.

<!-- SPDX-FileCopyrightText: Copyright 2026 The RSI-Kernel-Tune Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
