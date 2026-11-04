# CHANGES-WP20R — red-team shrunken package (audit confirmation, gap-fill, boundary)

Branch `feat/wp20r-redteam`, cut from origin/main `31b0712` (post-WP25 renamed
main). Package brief: handoff/105 (WP20R): minimal red-team sample set + gate
accounting — executed per the brief's incremental rule: the four attack
classes were inventoried against the WP20 matrix already integrated and
reviewed (WP23 final review, item 2.6, on record) BEFORE building anything.
Result: three of the four classes were covered; this package audit-confirms
them, fills two gaps (one new caught-form fixture, one verified open row with
a recorded proposal), audit-confirms the accounting fields as already
implemented, and pins the asset boundary as a regression test. No frozen
surface was touched. All new files are English (the 2026-11-04
internationalization ruling).

## 0. Frozen surfaces — untouched (byte-identical)

| Surface | Evidence |
|---|---|
| `src/peregrine_kernel/card_template/eval/timing.py` | sha256 `6c8fd1c578edec5a01399e4f3f5bda93b2a14bd822fbc9b5c958bc2f3a6f31bf`, identical at baseline and tip; `git diff -- src/**/timing.py` is empty |
| `src/peregrine_kernel/card_template/eval/benchkit.py` | sha256 `d27eb4339ad2c287ab467b2369b7573f7391c4752c09e100566b1fbdefe48275`, identical at baseline and tip; no benchkit sentinel was landed (the G1 proposal is recorded for adjudication, not implemented) |
| gate / judge logic | zero changes: the diff names only `tests/`, `docs/`, and this file |
| `meta/`, `hw_kb/` | zero changes (same diff evidence) |

## 1. Inventory verdict (what the integrated WP20 matrix already covered)

| 105 attack class | Integrated-matrix verdict |
|---|---|
| 1. Cached-output reuse -> fresh_recheck layer 1 | COVERED (pointer-keyed fixture, attack 1) |
| 2. Seed-change recon -> fresh_recheck layer 2 | PARTIALLY COVERED: the garbage-on-miss content cache is caught (attack 2); the honest-on-miss "recon" form is NOT — verified empirically in this package, recorded as open row G1 |
| 3. Stream injection -> check defense, assert or record gap | COVERED: the WP20 stream sentinel exists, is wired into the bench (armed before any foreign import, read after all windows close), and is asserted at unit level (a CPU box cannot inject into a real GPU stream) |
| 4. Timing-surface tampering -> eval-file integrity + timer identity | COVERED: timer identity sentinel (attack 6 fixtures) + harness guard for eval-file editing (driver-side defense test cited, not duplicated) |

Task 2 (gate/judge accounting) was likewise found ALREADY IMPLEMENTED in the
integrated tree (see §3) — audit-confirmed, zero code change, exactly the
"shrunken delivery is a compliant outcome" path the brief anticipated.

## 2. Attack -> defense -> status table (paper material, brief 80 §11.5)

| # | Attack form | Triggered defense | Status | Evidence (assertion) |
|---|---|---|---|---|
| 1 | Cached-output reuse, pointer/state-keyed (timed calls replay parked answers; anything else gets zeros) | WP9 fresh-recheck layer 1: fresh process — parked artifact survives on disk, its pointer keys miss every fresh allocation, the artifact-miss path serves garbage | CAUGHT (asserted) | `tests/redteam/test_redteam_cache.py::test_attack1_pointer_keyed_cache_misses_in_a_fresh_process` |
| 2 | Cached-output reuse, content-keyed garbage-on-miss (replays the judged instance) | fresh-recheck layer 2: the child re-draws inputs at a shifted seed — the content moves under the cache | CAUGHT (asserted) | `tests/redteam/test_redteam_cache.py::test_attack2_content_keyed_cache_misses_the_shifted_draw`; `tests/test_fresh_recheck.py::test_child_regen_shifts_the_drawn_content` |
| 3 | Cached-output reuse, shape-keyed — the shape half of the (shape, pointer) family (NEW in WP20R) | fresh-recheck layer 2; structurally, one slot per shape cannot represent the card's per-group judged answers | CAUGHT (asserted) | `tests/redteam/test_redteam_cache.py::test_attack1c_shape_keyed_cache_serves_stale_values_under_the_redraw` (new) |
| 4 | Seed-change recon: replay recognized content (the timed windows), honest compute on ANY miss — "computes honestly once the input changes" | fresh-recheck value layers: blind to the miss path (honest is correct anywhere); stream sentinel: reads a replay as `ok` (no side-stream span); proc census / timer sentinel: orthogonal | **NOT CAUGHT — OPEN ROW G1**: verified, recorded, proposal filed; adjudication requested; frozen surfaces untouched | `tests/redteam/test_redteam_recon.py` (3 assertions of the documented posture); gap record + minimal proposal: `docs/wp20r-known-gaps.md` |
| 5 | try/except wrong fast path (cheap wrong answer ships if nothing raises) | correctness judge (`INCORRECT_NUMERICAL` with per-check error/tolerance rows) | CAUGHT (asserted) | `tests/redteam/test_redteam_fallback.py::test_attack3a_wrong_fastpath_fails_correctness` |
| 6 | try/except dead fast path (honest math always runs; every defense passes) | none today — no static layer scans the fallback shape | OPEN ROW (pre-existing, unchanged) | `tests/redteam/test_redteam_fallback.py::test_attack3b_dead_fastpath_passes_everything_known_gap` |
| 7 | Stream injection (work launched onto a stream the timing event pairs never wait for) | WP20 stream sentinel: after all windows close, one call per timed group re-timed two ways (event pair on the timing stream vs full-device drain under a wall clock); a drained reading dwarfing the events rides the WP9 fail path (`fresh_recheck mode=stream` -> REVERT + suspicious, zero new driver judgment); threshold wide enough that honest launch/sync overhead never trips; env failure degrades to `unknown`, never convicts; a kernel chaining its side stream back (`wait_stream`) reads the same both ways and passes | CAUGHT (asserted, unit-level with real sleeps; a CPU box has no CUDA stream to inject into) | `tests/redteam/test_redteam_stream.py` (8 tests, incl. `test_attack4_side_stream_work_makes_the_drain_dwarf_the_events`, `test_honest_kernel_overhead_never_trips`, `test_env_failure_degrades_to_unknown_never_convicts`, `test_driver_reverts_and_marks_suspicious_on_mode_stream`) |
| 8 | Thread injection / JIT fork around the timed region | proc census (pre-import / post-first-call / post-windows), accounting-only by design — the anomaly rides the eval JSON, never the verdict | READING ONLY — recorded posture (pre-existing) | `tests/redteam/test_redteam_timer_proc.py::test_attack5_a_thread_started_for_the_timed_region_shows`, `::test_proc_census_is_accounting_only_in_the_loop` |
| 9 | Timer monkey-patch (rebind the eval's window routine or `torch.cuda.Event` in-process) | timer identity sentinel: clock/window identities captured before the kernel imports, re-resolved after all windows close; any moved id reads `mismatch` and rides `fresh_recheck mode=timer`; an unreadable sentinel is `unknown`, never a conviction | CAUGHT (asserted) | `tests/redteam/test_redteam_timer_proc.py::test_attack6_rebinding_the_evals_clock_is_caught`, `::test_attack6_rebinding_torch_cuda_event_is_caught`, `::test_timer_mismatch_rides_fresh_recheck_mode_timer` |
| 10 | Eval-file editing (a looser tolerance, a correctness block that always passes, a bench whose timer skips the kernel) | harness guard (`[loop] harness_guard`, default `revert`): a delivered commit whose diff touches `eval/`, `gpu_entry.sh`, `ref_bin/`, `.gitattributes` is REVERTed before any job is bought, paths named in the record; every pushed job tree carries the card's tracked harness bytes via the job overlay | CAUGHT (asserted) | `tests/test_harness_guard.py::test_a_delivery_that_edits_the_harness_is_reverted_without_a_job`, `::test_the_job_tree_carries_the_cards_harness_whatever_the_lineage_holds` (matrix cites the defense tests; no duplicate built, per the brief's no-duplicate-build rule) |

Memo for the paper: rows 1-3, 5, 7, 9, 10 are defenses with passing catch
assertions; rows 4, 6 are open rows recorded honestly (the matrix's value is
that the boundary between caught and not-caught is asserted, not narrated);
row 8 is a deliberate accounting-only reading.

## 3. Task 2 (gate/judge accounting) — audit-confirmed as already implemented

Zero code change in this package. The brief's two fields, with evidence:

| Field | Where | Evidence |
|---|---|---|
| `gate_degraded` (strict card's `unavailable` rows beyond the ratio -> a record-side flag, zero verdict weight) | bench emits it on the eval JSON | `eval/benchkit.py:380` (`gate_degraded()`, ratio default 0.5, `PK_GATE_DEGRADED_RATIO` override at `:134-136`), emission at `:1916-1920` |
| | driver recomputes and carries it | `loop/driver.py:1229-1233` (`_precision_degraded`), record construction `:3965` and `:4065`, config `_wp20_apply_cfg` `:1330-1341` (wired `:18603`) |
| | record field | `inner/trajectory.py:153-155` |
| `corr_margin_ratio` (the worst check's error/tolerance on a REVERT row) | parsed once | `loop/parse.py:479` (`corr_margin_ratio()`), carried `:595` |
| | record field | `inner/trajectory.py:151`, record construction `loop/driver.py:3964` |
| Adjacent WP20 accounting already present (not in 105's shrunken scope, listed for completeness) | judge edge-band recheck (`:3931`), `fresh_recheck_form`, `board_span_unstable`, milestone tag-wipe WARN | asserted in `tests/redteam/test_redteam_accounting.py` |

Consumption tests (pre-existing, passing): `tests/redteam/test_redteam_accounting.py`
— gate_degraded live through the bench (`test_gate_degraded_marks_an_all_unavailable_block`,
`test_gate_degraded_ratio_env_overrides`), corr_margin_ratio
(`test_corr_margin_ratio_parsed_from_the_checks`, `test_revert_row_carries_the_judge_edge_it_sat_on`),
and the zero-verdict control (`test_zero_verdict_control_all_rows_quiet_on_a_baseline_eval`).

## 4. Task 3 (asset boundary) — new regression test

New file `tests/test_redteam_asset_boundary.py` (runs in the default suite):

1. **Marker scan**: the fixtures' technique markers (cheat kernel symbols, the
   parked artifact filenames) appear in NO tracked file outside `tests/` and
   the `CHANGES-*.md` records — the scan covers `src/` (incl.
   `card_template/`), `meta/`, `skills/`, `scripts/`, `kernels/`, `tasks/`,
   `hw_kb/`, `global_config/`, and `docs/` (the known-gap page is kept at
   defense-level wording so it stays inside the boundary).
2. **No skip / no xfail**: no `pytest.mark.skip` / `pytest.mark.xfail` /
   inline `pytest.skip` in `tests/redteam/` (the module-level
   `importorskip("torch")` is an environment guard, not a case skip).
3. **Collection floor**: the default suite collects the matrix at its WP20R
   delivery size (40) or larger — shrinking the matrix must be a reviewed
   decision.

Grep evidence (technique markers outside the sanctioned homes):

```
$ grep -rln "CHEAT_KERNEL\|FALLBACK_WRONG_FASTPATH\|FALLBACK_DEAD_FASTPATH\|.side_buffer.pt\|.ptr_buffer.pt\|cheater's bet" \
    --include='*.py' --include='*.md' --include='*.toml' --include='*.json' . | grep -v '^./tests/'
  (empty)
$ grep -rln ... .                       # unfiltered: the only hits are
  ./tests/test_fresh_recheck.py         # the WP9 defense's own cheat scaffold (pre-existing; inside tests/)
  ./tests/redteam/test_redteam_cache.py
  ./tests/redteam/test_redteam_fallback.py
  ./tests/redteam/test_redteam_recon.py
```

## 5. Delta (files) and regression

New/changed by WP20R (nothing else):

| File | Change |
|---|---|
| `tests/redteam/test_redteam_cache.py` | +attack 1c fixture (shape-keyed cache), `Path` import |
| `tests/redteam/test_redteam_recon.py` | NEW — attack 2b (recon replay), 3 assertions of the documented open-row posture |
| `tests/test_redteam_asset_boundary.py` | NEW — task 3 boundary regression (3 tests) |
| `docs/wp20r-known-gaps.md` | NEW — G1 record + minimal proposal for adjudication |
| `CHANGES-WP20R.md` | NEW — this file |

Regression (the project venv's Python, CPU only, no GPU used):

- baseline `origin/main` (31b0712): **2594 passed** (333.20 s)
- branch tip: **2601 passed** (2594 + 7 new: 4 red-team fixtures, 3 boundary
  tests), 0 failed, 0 skipped.

## 6. Decisions requested (chief engineer)

1. **G1 disposition** (`docs/wp20r-known-gaps.md`): approve the fresh-recheck
   latency probe (or the smaller latency-delta fingerprint) as a separate,
   adjudicated package — or close it as accepted residual risk for the paper's
   measurement-integrity section (the open row is already asserted, so the
   paper can state it verbatim).
2. The pre-existing open rows (dead fast path, census accounting-only) remain
   as recorded; no action requested.
