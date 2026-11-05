# Backlog — frozen and deferred items (the single ledger)

This file is the **only** registry of frozen, deferred and rejected items. Its purpose is to
stop later sessions from re-proposing frozen items as if they were new ideas. Everything here
was deferred or frozen by an explicit adjudication, with a stated reason and a stated
re-estimation trigger.

**Re-estimation trigger for every frozen item** (user adjudication 2026-11-04): the first
round of real-run A/B numbers landing on disk (the energize protocol's one-default-at-a-time
matrix, `docs/energize-protocol.md`). Until then, no new-mechanism work packages are signed.

Sources: gen-2 overview section 8 (`handoff/20`, section 8); gen-3 merged adjudication
(`handoff/59`, sections 5/6); the four independent assessments (`handoff/96`, `97`, `99`, `100`);
the WP25 package brief (`handoff/102`). The handoff directory is outside this repository;
this file carries the substance so the repo is self-contained.

## A. Frozen by the 2026-11-04 freeze order (assessments' new directions)

No new-mechanism packages until the existing mechanism stock is energized and the first real
A/B numbers exist. These are recorded so nobody re-proposes them as new:

| # | Item | Substance | Source | Re-estimation trigger |
|---|---|---|---|---|
| A1 | Teacher decision instrumenting | Upgrade the WP18 `kb_used:` ledger to per-decision retrospection: every teacher lever pick / plateau call / reopen records input readings -> choice -> measured outcome K iterations later; quantifies teacher decision hit rate; the basis for base-model selection and drop-one-out ablation of the injection region. | 96 §7.2-1, 99 | first A/B numbers |
| A2 | Framework meta-metric | After deboarding, the framework has no external anchor: build a sealed-ref task set from the `kernels/` frozen operators via `rsi-kernel-tune init --from-kernels` and adopt **geomean speedup vs sealed ref** as the public metric — two columns ("vs gen-N ref" and "vs the original frozen ref") because the denominator drifts as references upgrade; periodic offline spot-checks against public implementations. | 96 §7.2-2, 97 §B6, 99 | first A/B numbers |
| A3 | Price ledger -> exploration budget allocation | lever_pricing / falsified / combo-pool already accumulate per-family measured prices; a teacher-side advisory ranker (UCB/Thompson style, **rank only, never price** — respects the "a ranking is not a price" red line) proposes what the next probe round measures. | 96 §7.2-3, 99 #5 | first A/B numbers |
| A4 | kbmenu data-driven updates | With hit telemetry in place, Bayesian updates over (symptom, entry, KEEP/REVERT) triples: retire zero-hit rules, promote surviving patterns. The 10 hand-written rules are the most data-replaceable part. | 96 §7.2-4 | first A/B numbers |
| A5 | Multi-fidelity unified scheduling | WP17's `loop/screening.py` exists, but probe / SWEEP / MC / ARCH_PROBE still manage budgets separately; ASHA-style rung promotion could unify them (cheap tier eliminates most candidates, expensive tier keeps survivors). | 96 §7.2-5 | first A/B numbers |
| A6 | BO re-estimation (conditional, long-term) | Gen-3's rejection of OpenTuner/BO holds at current evaluation cost; after WP19 lowers per-evaluation cost by an order of magnitude, noise-model-aware BO is the right tool for expensive noisy black boxes — as a "nomination engine" on the screening primitive, still landing through the teacher. | 96 section 7.2 item 6; 59 section 6 | first A/B numbers |
| A7 | MC variant diversity guarantee | <=3 variants from the same student model homogenize; when authorizing, require explicitly divergent skeleton dimensions (e.g. v0 persistent threads / v1 double buffering / v2 warp specialization). | 96 §7.2-7 | first A/B numbers |
| A8 | Structured NCU delta pre-extraction | Give the teacher not the full report / tail-N but "which metrics moved how much since last iteration, which thresholds crossed" — cheaper on context than windowing and loses nothing critical. | 96 §7.2-8 | first A/B numbers |
| A9 | Idle-GPU background lane expansion | recon pre-runs, KB batch certification in idle windows — **must pass the C15 contract**: background readings never enter a gate, cooldown guard; batch certification verdicts declared to use the same protocol as foreground probes. | 96 §7.2-9 | first A/B numbers |
| A10 | Mechanism-gain quantification ledger | The audit answers "is the data flowing", not "what is each mechanism worth": persist the one-flip-at-a-time A/B data as a standing per-mechanism ledger (switch state x wall clock x KEEP rate x confirmation triggers x hit telemetry); without it, every base-model change starts from zero. | 100 section 5 item 1 | first A/B numbers |
| A11 | Injection-region hard budget | The C10 contract caps each package at <=30 lines but the region grows monotonically (13 segments); only a soft "add one, trim one" red line exists. Proposal: a hard total-line cap with automatic lowest-priority-block degradation. | 100 section 5 item 2 | first A/B numbers |
| A12 | Global budget arbiter | probe / SWEEP / confirmation / pattern search each have static budget classes with no cross-channel arbitration; even a simple expected-information-gain-over-cost ordering beats static allocation. WP17's P2 CANDIDATES scheduling is the embryo. | 100 section 5 item 3 | first A/B numbers |
| A13 | Task-side headroom ordering | Tasks come from `--from-kernels` (frozen operators); self-evolve will eventually exhaust the easy ones. `analytic_floor` already computes per-task headroom; order `--from-kernels` tasks by it. | 100 section 5 item 4 | first A/B numbers |
| A14 | Cross-run price-distribution prior | `lever_prices.jsonl` dies with each run (the falsified ledger crosses runs, positive prices do not); aggregate "the same lever's price distribution over N runs" as sweep starting-point ordering — rank-only, zero discipline risk. | 100 section 5 item 5 | first A/B numbers |
| A15 | Measurement-side adversarial additions | The WP20 fixture set does not yet cover "the kernel detects it is being timed" (clock reading / warmup-pattern recognition followed by behavior switching); could use existing `clock_drift` telemetry plus a first-call vs steady-state behavioral-difference sentinel. | 100 section 5 item 8 | first A/B numbers |
| A16 | Bayesian paired statistics (meta-analysis layer) | resolvable v2 is frequentist (3-sigma band, SPRT, Stouffer); for a system producing paired readings continuously, a Bayesian paired model (hierarchical normal on log-ratios) could give narrower credible bands with fewer samples. Gen-3 forbade touching the `timing.py` estimator (cross-run regime break cost); the meta-analysis layer is the legal location — long-term. | 100 section 5 item 9 | first A/B numbers |
| A17 | Heterogeneous-base review, institutionalized | Review briefs and reports do not state whether reviewer and implementer are heterogeneous (different base model / session); if same-origin, "independent QA" is discounted. Write heterogeneous review bases into the next wave's rules. | 99 section 7 | next review wave |

## B. Deferred P2 items (carried adjudications, not yet delivered)

| # | Item | Substance | Source |
|---|---|---|---|
| B1 | `driver.py` split | ~15k-line single file + the `_wp<n>_*` helper families; the per-package comment-block discipline survives merging, not long-term evolution. A **purely mechanical** module split (wp*_apply_cfg / injection-block rendering / channel handlers) after this package's merge. No refactor may ride along with other work (WP25 freeze ledger). | 100 section 5 item 7; 97 section C8, 102 fence |
| B2 | blackbox guard known failure modes | `docs/blackbox-mode.md` self-documents loop bugs (its failure-mode section); fix them as a dedicated small package. | 20 section 8 item 3 |
| B3 | Decision-set projection | The driver projects the currently deliverable decision set as one line into the student prompt. | 20 section 8 item 5 |
| B4 | Per-shape decision interaction ledger | `regression_vs_best_pct` extended to a per-shape array with REMOVE_CANDIDATES rendered per shape. WP14's per-shape ledger partially covers it; the interaction dimension remains. | 20 section 8 item 2; 97 section C11 |
| B5 | agent/GPU overlap pipeline, multi-decision batch delivery, eval/NCU card split, CuteDSL resusage, traffic_amp | Process-shape ideas all three assessment lines agreed to keep on the P2 pile. | 59 section 6 |
| B6 | REVERT lower-resolution NCU (conservative tier) | Deferred because it conflicts with the gen-1 adjudication "negative iterations also get hardware truth"; small benefit. | 20 section 7 |
| B7 | binref compiled-reference route | Compiled (.cu) reference routing is not wired into ordinary tasks; `--from-kernels` explicitly refuses `.cu` published kernels with a P2-backlog pointer. | CHANGES-WP24 §7 (git history) |
| B8 | Milestone validator "all-labels-deleted" WARN | Downgraded, non-blocking, de-silencing only (~10 lines; the residual hole `milestone.py` self-documents). | 59 WP20 P2 |
| B9 | Hot/cold L2 dual readings (report-only) + symmetric CUDA-graph timing option | Measurement-side option for launch-bound cards, report-only, never enters a gate. | 59 WP18 P2 |
| B10 | WP21 smoke defect registry | Defects D-1..D-10 found by the mini smoke are dispatched to owning packages per the energize protocol section 4 (e.g. the `_wp5_sol_pcts` getattr bug — WP5's never-realized consumption of WP2 telemetry, closed by auditA 2026-11-04: it reads the record's `profiler` dict now; `build_digest` dead link closed by WP19). | 97 §B5, energize protocol §4 |
| B11 | Optional minimal GitHub Actions CI | ubuntu + `pip install -e .` + `pytest tests/ -q`, CPU-only. **Requires user approval before enabling** (WP25 task 3, P2). | 102 task 3 |
| B12 | Run-level multi-GPU parallelism | Zero framework code; deployment self-serve (distinct `with-lock` lock names per GPU). | 59 section 6 |

## C. Closed / rejected (recorded so they are never re-proposed)

| # | Item | Disposition | Source |
|---|---|---|---|
| C1 | WP7 precision lane (a "spendable precision budget") | **Rejected on principle by the user, 2026-11-04**: precision belongs to the user's contract — "whatever precision the user demands must be written, not one bit less." The ratio gate is the detector's tolerance, not a spendable budget. The existing two-layer structure already implements the principle; zero code changes. If ever revisited, 99's precondition applies: a third-party verification protocol is a hard prerequisite, not a parallel suggestion. | 59 WP21 final adjudication; 99 section 7 |
| C2 | Board double gate / cold-L2 axis | Voided with WP24's deboarding (the board protocol no longer exists). A generic strict cold-L2 measurement axis, if ever wanted, returns via a WP22-style review. | 59 WP21 final adjudication |
| C3 | OpenTuner/BO replacing the SWEEP searcher | Long-term deferral stands: in-principle-constrained search + successive halving already cover the bulk of its benefit. (See A6 for the conditional re-estimation.) | 59 section 6 |
| C4 | board docs / leaderboard-facing machinery | Removed by WP24; WP25 completed the removal of the Chinese change-history files (content distilled into `CHANGELOG.md`; originals live in git history). Not a backlog item — recorded to prevent "restore the board docs" proposals. | WP24; WP25 task 3b |
| C5 | G1 replay-probe proposal (the recon replay cheat's honest miss path escaped every value check) | **Implemented and closed, WP29, 2026-11-04 — approved by the user** ("fill the gap directly, add a switch"): the replay probe landed in `benchkit.py` (post-window honest-compute latency on fresh draws, `RKT_REPLAY_PROBE` default on, `RKT_REPLAY_PROBE_RATIO` default 3.0, suspected -> fresh_recheck-same REVERT). Ledger entry: `docs/wp20r-known-gaps.md` G1 (now CLOSED); evidence: `tests/redteam/test_redteam_recon.py`; record: `CHANGES-WP29.md` (git history). Do not re-propose. | WP20R decision request 1 (`CHANGES-WP20R.md` §6, git history); handoff/139 |

## D. Version policy (WP25 note, P2)

SemVer. After this package merges to main, the release manager tags `v1.0.0` (bump
`pyproject.toml` `version` in the same act). Behavior-equivalent mechanical packages bump the
patch/minor component; any contract change (C1..C18) or default flip bumps minor at least.

<!-- SPDX-FileCopyrightText: Copyright 2026 The RSI-Kernel-Tune Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
