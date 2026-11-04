# Energize and harvest protocol (WP21 / line-A G25)

> Execution belongs to **WP23** (integration and harvest): this document is the protocol and
> the fixed action list; this package (WP21) only lands the document and the parameterized
> skeleton `scripts/energize_ab.py` — **neither the batch runs nor the default flips happen
> in this package**.
>
> Background: line-A G25 (the gen-2 dormant-mechanism verification debt) — `PK_BUILD_CACHE`
> off, SPRT off, the MC-DELIVERY channel open but activated only when the teacher writes
> `multi_candidate`, the cli forwarding lines for stagnation/multi_candidate were missing
> (wired in dcfdcb5), the floor gate inert until floor_backtest's `best_ms` is backfilled,
> the survey channel opt-in default off, the board A/B for the NCU flags never run, and the
> JustGoal A/B protocol written but never run. The debt incurred by gen-1 red line 8 (board
> measurements only spot-checked end-to-end) is repaid by gen 3 through this protocol.

## 0. General rules

1. **One default at a time**: each round flips exactly one default; before the flip there is
   a baseline A, after it an experiment B — same task set, same machine, same budget; the
   next flip is allowed only after the evidence is on disk.
2. **Every default must have a rollback switch**, and the rollback switch must be separate
   from the default itself (env / cli / a loop.toml key); the rollback action is a one-line
   command — "roll back by editing code" is not allowed.
3. **Evidence before the flip**: only when both evidence classes are complete — the A/B
   comparison case and the mechanism-behavior case (the mechanism really triggered) — does a
   default-flip proposal move forward; the proposal is written in CHANGES and executed by
   the WP23 integration step.
4. **The local 3060 verifies the chain only and produces no performance conclusions**
   (red line 11): on the 3060 the A/B "measurement" only answers "did the mechanism trigger,
   is the dataflow complete, did the wall-clock composition improve"; any batch run whose
   conclusion is a percentage gain must be redone on the target hardware (sm90/sm100 boards).
5. **GPU exclusivity**: every real run is wrapped in `~/tools/bin/with-lock gpu 1800 -- <command>`.
6. **JustGoal comparison, institutionalized** (section 3): the same-job A/B before and after a
   mechanism lands, plus a re-measure on a second base, is a fixed acceptance action for every
   mechanism, not an option.

## 1. The acceptance A/B matrix

| # | Item | Measurement plan | Evidence required | Rollback switch | Flip order |
|---|---|---|---|---|---|
| A1 | **Ordinary task-set execution** (the control baseline) | A fixed mini task set (`tasks/wp21mini` + 1 formal small card) x2 jobs; record: status.jsonl row counts, eval/NCU job completion rates, the five buckets of the wall-clock composition table, digest output | run archive + status.jsonl + two run.log files; the baseline is not archived in the repo (the run directory is the evidence) | — (no switch in the baseline state) | Run A1 first; it is the A side of every A/B |
| A2 | **NCU flags A/B** (WP9's leftover `ncu_flag_ab`, direction confirmation) | `scripts/ncu_flag_ab.py` (landed in WP9; not run while the 3060 had no CUDA torch — the ml venv now has CUDA torch): sample the same kernel once with the old and once with the new regime flags (`--cache-control none --clock-control none`) and compare the direction of the SOL/L2/DRAM reading shifts | two NCU reports + the shift table on disk; the conclusion only states "direction confirmed", never a performance conclusion; the report header must note that cross-generation NCU numbers must not be mixed | every NCU flags generation point carries a switch (WP9-native); switching off returns to the old regime | any time (pure measurement, no default flip; only to ground the cross-generation reading basis) |
| A3 | **SPRT default** (`sprt_early_stop`, default off) | the A1 task set x2: off vs on; measure: the per-iteration eval job duration distribution (the share of windows where early stop fired), verdict consistency (a pass requires KEEP/REVERT to match A exactly) | per-iteration eval_job_s comparison of the two status.jsonl files + a verdict comparison table; **SPRT only stops early, it never manufactures a KEEP** must be checked line by line (the driver's SPRT section) | `--sprt-early-stop` cli / loop.toml `[loop] sprt_early_stop` (default false is the rollback) | 3rd place after WP15/WP19 merge (after build cache, before floor) |
| A4 | **build cache default** (`PK_BUILD_CACHE`, default off) | the line-B compromise first: (1) verify the "same kernel rebuilt across jobs" hypothesis from real build logs (line-C T5's annotated inference); (2) key-coverage cases: a header change must miss / a comment change must hit; (3) then the A1 task set x2 measuring the wall-clock composition table's eval_job_s bucket | (1) build-log excerpts (two nvcc calls for the same sha); (2) the key-coverage unit test output; (3) two rounds of wall-clock composition tables; the manifest's per-artifact sha recheck log | `PK_BUILD_CACHE=0` (env, one-line rollback); board cards and sealed refs do not enable it (scope guard) | **1st place** (line B argues for default-on locally; with evidence complete, a CHANGES proposal — the flip itself belongs to WP23) |
| A5 | **floor_stop / floor_margin** (the analytic floor gate, default off) | prerequisite: `scripts/backtest_floors.py --from-run <archive>` backfills floor_backtest.json's 25x nulls (WP21 smoke verified the --from-run path); after the backfill, backtest_early_stop.py backtests floor_stop's counterfactual hit rate; the A1 task set x2: off vs on, measuring "wrong-early-stops = 0, plateau-verdict drift = 0" | the floor_backtest.json backfill diff + the early-stop backtest report + the floor lines of both run.log files | `[target] floor_margin` / `[loop] floor_stop` (task-layer keys, default off) | 4th place (depends on A5's prerequisite backfill + the WP15 resolvable v2 merge) |
| A6 | **survey channel** (`teacher_prior_survey`, default off) | the A1 task set x2: off vs on; measure: survey artifacts on disk (priors/ paths), the change in the teacher's diagnose turn count, whether the SOL header line was cited by the teacher (the citation rate in the diagnose text); **mechanical presence check** (line-A F18: an obligation sentence that does nothing = fail) | both run.log files + the priors artifacts + a diagnose-turn count table; survey not done with no mechanical interception = this item fails | `[loop] teacher_prior_survey = false` (card-layer loop.toml) | 5th place |
| A7 | **MC-DELIVERY activation** (the channel is on; needs the teacher to write `multi_candidate`) | the A1 task set x2: ordinary vs a teacher milestone containing a `multi_candidate:` block; measure: variant_timings rows, the merged job's prescreen rows (the smoke verdict table), the winner alone passing the full gate (the gate row only on the winner) | both status.jsonl files (the variant_timings field) + the merged job log's MC section; **note variant_timings has no baseline reader (WP21 audit baseline #7) — this item's evidence doubles as the feed verification for WP16's pool** | the teacher not writing a `multi_candidate:` block is the natural off; the `[loop] multi_candidate_max` cli forwarding (the dcfdcb5 seam) | 6th place (joint testing after the WP16 pool merges) |
| A8 | **stagnation cli forwarding recheck** (WP10 leftover, fixed in dcfdcb5) | static + dynamic double evidence: static = a grep of the cli.py forwarding lines (`stagnation_k`/`multi_candidate_max`); dynamic = run one round with the cli passing non-default values and compare the driver-side effective values (the STAGNATION trigger threshold line in run.log) | the grep output + one dynamic run.log (the threshold matching the cli value) | the cli defaults are the rollback (pass no args) | 0th place (pure recheck, no flip; a fixed item of WP23's seam check) |

**Flip order summary**: A8 recheck (no flip) -> A2 flags evidence (no flip) -> **A4 build cache** ->
A3 SPRT -> A5 floor -> A6 survey -> A7 MC. Rationale: A4 has an independent rollback env and
cuts wall clock most directly; A3/A5/A6 depend on the gen-3 verdict surface after WP15/WP19
merge; A7 depends on WP16. After each flip, rerun the A1 task set once (against interaction
regressions); only when all green, move to the next item.

## 2. Where measurements and evidence are written

- All A/B run archives stay under `.artifacts/` (not salvaged into git); CHANGES references
  the run directory name.
- The wall-clock composition table (end of run.log, `[wp11] wall-clock composition`) is the
  primary measurement for the process-speedup flips (A3/A4): the five buckets
  (agent_turn/eval_job/ncu_job/teacher/queue_wait) compared round by round.
- Verdict consistency is the primary measurement for correctness-class mechanisms
  (A3 SPRT / A5 floor): **any verdict drift = that round fails; roll back and record**.
- Every number from the 3060 is cited only as chain/behavior evidence; when written into
  CHANGES it must carry the label "3060 chain evidence, not a performance conclusion"
  (red line 11).

## 3. JustGoal comparison, institutionalized (upgraded to an acceptance action)

Around every mechanism landing, execute all of the following (missing any one disqualifies
the mechanism from a CHANGES default-flip proposal):

1. **Same-job A/B**: same run archive, same kernel tree, one merged job each with the
   mechanism off/on; compare all EvalResult fields (both windows' readings, fresh_recheck,
   gpu_state, gate_spread_pct). Wrapped by `scripts/energize_ab.py --mode same-job`.
2. **Re-measure on a second base** (the upstream debt "a base change requires re-measuring
   with all mechanisms"): with the mechanism on, at least one end-to-end round on a second
   card / second architecture (deployment decides); the evidence is that round's run archive
   + status.jsonl. On the 3060 a mini task may stand in for a formal card — it verifies
   mechanism triggering only, never gains.
3. **Mechanism-behavior cases**: pytest-level trigger cases (one per mechanism: the SPRT
   early-stop boundary, build cache key hit/miss, the floor backfill gate, the MC prescreen,
   stagnation threshold forwarding) — existing ones are referenced by test name in CHANGES;
   missing ones are added by the owning package.
4. The three evidence items (same-job A/B, second base, behavior cases) enter the mechanism's
   CHANGES entry as a checklist; WP23 accepts against the checklist.

## 4. The seam with the WP21 audit

- `tests/test_dataflow_audit.py` is this protocol's CI-side sentinel: the fields any A/B item
  wants to consume (`gate_spread_pct`, `clock_drift`, `ref_host_enqueue_ms`,
  `variant_timings`, `build_digest`) are currently in the baseline list — before flipping the
  corresponding mechanism, confirm its readers landed with WP15/WP19/WP16 (the audit test
  automatically demands the baseline entries be deleted).
- Mechanisms found silently-not-triggering during the smoke (the defect list in
  CHANGES-WP21.md's smoke section) are dispatched to their owning packages; after the fix,
  this protocol's corresponding A/B item re-verifies.
