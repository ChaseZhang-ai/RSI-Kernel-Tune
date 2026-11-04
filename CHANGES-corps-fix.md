# CHANGES-corps-fix — the audit-corps fix-d batch (monitor / resume / trajectory)

Branch `feat/corps-fix-d`, cut from origin/main `8977b63`. Nine B-class bugs
from the R2 audit (handoff/修复bug.md, triage T2/T3/T4), fixed one per commit,
each reproduced first via the ported regression pins (red on this base), then
turned green.

## 1. The bugs and the fixes

| Bug | Verdict | Commit | What was wrong | What changed |
|---|---|---|---|---|
| BUG-SEED-14 (P1) | fixed | `83aecda` | `monitor.classify` had no consumer for run.py's per-profile `hook.processes` count; a failed sitecustomize injection read as info "the bench does not import the kernel as a module" (the wrong cause) and info does not raise severity, so a passing card read `status: "ok"` while the NaN / input-shift checks had never executed | classify adds its own evidence: `processes == 0` in any profile that ran yields one warn (profile `"-"`) naming the profiles and the injection chain; the misleading info only fires when the hook itself was alive in the base run; the card's own verdict still dominates (a failed base stays alarm) |
| BUG-SEED-15 (P1) | fixed | `32a3df5` | run.py printed the final MARKER and the result JSON as two prints; a kill between them left MARKER over no payload, `parse_log` returned None and the per-profile rebuild (written exactly for cut-short jobs) was unreachable — every earlier profile's alarm-level finding was lost | both halves: run.py writes marker + JSON in one `write()` + flush; `parse_log` falls through to the per-profile rebuild whenever `_json_after` comes back empty (rebuilt partial keeps `complete: False` and the "cut short after N profile(s)" warn; a complete final JSON still wins) |
| BUG-SEED-16 (P1) | fixed | `33d19f8` | `compute_reopen` only pulled companions named by the candidates themselves (`d.companions & reopen`): with A declaring ⊗ B and B silent, reopening B kept A — the pair split across the reopen, the kept half's evidence describing the old tree; `dependency_warnings` (moved to milestone.py) checked only ⇒/⇐ echoes, so a one-sided ⊗ passed silently | the fixpoint also admits a decision NAMED by any reopened member (symmetric closure, chains close from any entry point; ⇒/⇐ untouched); `dependency_warnings` flags a one-sided ⊗ with the same message shape as the ⇒ echo; the contradictory Decision/ReopenPlan docstrings aligned (companions widen the reopen closure symmetrically; the `compute_resume` prefix cut still ignores them) |
| BUG-SEED-41 (P2) | fixed | `b2c045d` | `pk_monitor_hook._dump` wrapped the whole write path in `except Exception: pass` — an unwritable PK_MON_OUT lost every summary with zero trace, bypassing the module's own `_err` convention (which depends on `_dump` succeeding) | a failed dump is recorded via `_err`, said on stderr once (module-level flag), and the payload gets a last-resort write to the temp dir (`pk-mon-dump-fallback-<pid>.json`) so the data is not simply gone |
| BUG-SEED-42 (P2) | fixed | `6980a65` | the hook counts each call's index `i` from zero per process but call records carried no pid, and `_calls` deduplicated on `(fn, i)` — merged multi-process records (DataLoader workers / mp) let the processes overwrite each other | the hook writes `pid` into every call record; `_calls` keys on `(pid, fn, i)` (pre-pid records share the None key and read exactly as before); cross-run joins use a pid-insensitive view (`_base_calls` / `_call_key`, `_in_sig` keyed `(fn, i)`) — two profiles are different processes by construction, so comparisons match calls by what they were, not who held them |
| BUG-SEED-43 (P2) | fixed | `ce142b6` | `load_records` fed every non-empty line of status.jsonl to a bare `json.loads`; append_record is a plain append, so one torn tail line made the whole trajectory unreadable and every resume impossible | a torn FINAL line is dropped with a stderr warning ("torn"); a broken MIDDLE line still raises (a hole mid-history is corruption; silently resuming over it would build the prefix-reuse plan on a trajectory that never existed); the file is read utf-8 with `errors="replace"` so a cut multibyte character cannot raise before the parser sees it |
| BUG-SEED-25 (P2) | fixed | `f888397` | `_DIFF_SIG` matched only decimal fractions, so scientific-notation diffs lost their exponent: `1.5e-05` and `1.5e-07` shared the signature "1.5" and three genuinely different failing attempts fired `constant_output_flag` (driver.py production consumer) | the numeric branch admits an optional exponent (and bare integers like `max_diff=0`); inf/nan unchanged; bit-identical repeats still flag |
| BUG-SEED-26 (P2) | fixed | `0eb84f6` | `validate_milestone` located REQUIRED_SECTIONS markers with `text.find`: "### 4. M2 pipeline" contains the substring "## 4.", so a milestone missing the real "## 4. Known dead ends" passed on the strength of a numbered subheading (the same trap `_section_body` already fixed for the decisions cut) | the presence check uses the same line-anchored `_heading_at`; presence and order are judged on real heading lines |
| BUG-SEED-28 (P2) | fixed | `2709bd6` | `parse_correctness` read any "N/M FAILED" marker as a fail (including "0/5 FAILED" — zero failures); `_first_json_with_key` hand-counted braces without knowing JSON strings, so a `}` inside a string value broke the depth and quietly returned None for the whole output | the count decides (`passed = fails == 0`); the scan goes through `JSONDecoder.raw_decode` (the idiom run.py already uses); `parse_perf`'s empty result carries a `note` naming which diagnosis it was (marker present but unparseable vs no marker) |

Two further commits carry the ported regression suite:

- `65dc63b` — `tests/test_monitor_resume_fixes.py`: the archived pre-rename fix
  batch's 20 pins, rebranded to this package. 15 were red on `8977b63`
  (reproduction evidence for the table above), 5 were green and pin the
  behavior the fixes must not disturb (complete-JSON precedence, live-hook ok,
  old records without a `processes` verdict, middle-line raise, pid-ignorant
  joins).
- `06b2c84` — the ported file's docstring reworded to stay outside the WP25
  branding gate's forbidden-pattern surface (it named the archive paths
  verbatim).

One commit carries the digest telemetry half of the ported pins:

- `988aa34` — `build_digest` stamps each curve point with its row's
  `clock_drift` flag and adds the best row's `sm_clock_mhz` to the achieved
  block (omitted when no gpu_state). Report-only; the archived batch surfaced
  both, this tree carried only the `clock` header. Report-only telemetry, no
  gate reads it.
- `bcd7674` — retires the `IterationRecord.clock_drift` line in
  `tests/dataflow_audit_baseline.json`: the curve stamping is the field's
  first read point, and the dataflow audit's own rule ("registered entries are
  stale — the consumer has landed, delete the line") enforces the deletion;
  `test_annotations_are_current_and_substantiated` failed on it until the line
  went. `gate_spread_pct` / `ref_host_enqueue_ms` / `variant_timings` stay
  registered (still readerless).

## 2. Deliberate decisions

- **SEED-43 middle-line semantics**: the corps fix-d brief's prose ("中间行失败
  仍告警丢弃该行不炸全程") and the archived pin
  (`test_load_records_still_raises_on_a_broken_middle_line`, endorsed by
  triage-t5: "顺带钉住不因上条修复放开中段") disagree. The pin wins: a middle
  line failing to parse still raises. A silently skipped middle row would make
  resume decisions on a trajectory with a hole; tolerance is exactly the torn
  tail.
- **test_dependency_warnings_flag_one_sided_downstream_claims extended**: the
  existing test asserted the exact warning list, and the `_ARROW_MS` fixture
  genuinely carries a one-sided ⊗ D0.3 — which is now (correctly) warned. The
  ⇒ assertion is kept verbatim as a membership check; a dedicated one-sided ⊗
  pin was added.
- **Digest clock telemetry** is not one of the nine ledger bugs, but the
  ported pin (`test_digest_curve_carries_clock_drift_and_best_sm_clock`)
  requires it; implemented as report-only additions, no gate semantic touched.

## 3. Territory and freeze compliance

- Touched files: `monitor/__init__.py`, `monitor/runtime/run.py`,
  `monitor/runtime/hook/pk_monitor_hook.py`, `resume/invalidation.py`,
  `milestone.py`, `inner/trajectory.py`, `inner/results.py`, plus tests.
- No forbidden file touched (loop/driver.py, loop/parse.py, sitecustomize.py,
  loop/cli.py, launch.py, loop/agent_policy.py, taskcard.py, baseline_cli.py,
  timing_audit.py, scripts/ all byte-identical to `8977b63`).
- Frozen surfaces untouched: `timing.py measure_paired`, precision constants,
  meta/ discipline text, gate semantics, LICENSE/NOTICE.

## 4. Regression

Full suite (all of `tests/`, venv python 3.14.4, this tree's `src/` on
PYTHONPATH): **2715 passed, 0 failed** (614.86 s). The pre-batch base collected
2692; +20 ported pins, +2 new milestone pins = 2715 exactly.
