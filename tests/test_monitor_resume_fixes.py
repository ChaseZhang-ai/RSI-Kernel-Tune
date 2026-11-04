"""Regression tests for the monitor / resume fix batch (audit corps fix-d).

Ported from the audit archive's untracked-tests copy of
test_monitor_resume_fixes.py (the pre-rename fix batch), rebranded to this
package: module path peregrine_kernel, hook file pk_monitor_hook.

Covers: classify flagging a monitor whose hook never ran (BUG-SEED-14),
parse_log rebuilding a partial result when the final JSON was killed mid-dump
(BUG-SEED-15), one-sided ⊗ companion declarations reopening the pair from
either side (BUG-SEED-16), the hook recording its own dump failures
(BUG-SEED-41), per-process calls not overwriting each other (BUG-SEED-42),
load_records tolerating a torn final line only (BUG-SEED-43), max_diff
signatures keeping their exponent (BUG-SEED-25), the digest carrying the clock
telemetry on the curve and the best row, and the inner results parser
(BUG-SEED-28).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from peregrine_kernel import monitor
from peregrine_kernel.inner.results import parse_correctness, parse_perf
from peregrine_kernel.inner.trajectory import (
    IterationRecord,
    _DIFF_SIG,
    append_record,
    build_digest,
    constant_output_flag,
    load_records,
)
from peregrine_kernel.resume import Decision, ReopenPlan, compute_reopen


# --- fixtures ---------------------------------------------------------------------
def _call(fn="call", i=0, *, nan=0, abs_sum=10.0, pid=None):
    return {"fn": fn, "i": i, "pid": pid, "inputs_nonfinite": 0, "in_sums": [1.0],
            "outputs": [{"nan": nan, "posinf": 0, "neginf": 0, "abs_sum": abs_sum,
                         "sq_sum": abs_sum / 2}]}


def _p_rec(profile, card_pass=True, *, processes=1, calls=(), **kw):
    """A per-profile record the way runtime/run.py's _merge_hooks writes it."""
    return {"profile": profile, "card_pass": card_pass, "rc": 0, "seconds": 1.0,
            "cases": [],
            "hook": {"processes": processes, "calls": list(calls), "perturbed": 0,
                     "reseeded": 0, "errors": []}, **kw}


def _ir(i, *, verdict="KEEP", error="", **kw):
    kw.setdefault("bwd_median_ms", None if error else 2.0)
    return IterationRecord(iter=i, decision_id="D", change="c",
                           correctness=None if verdict == "NEUTRAL" else not error,
                           verdict=verdict, error=error, **kw)


# --- BUG-SEED-14: the hook never ran -> not ok ------------------------------------
def test_classify_zero_hook_processes_is_a_warn_with_the_real_cause():
    res = {"complete": True, "profiles": [_p_rec("base", processes=0),
                                          _p_rec("seed1", processes=0)]}
    v = monitor.classify(res)
    assert v["status"] == "warn"                       # was "ok" before the fix
    dead = [f for f in v["findings"] if "monitor hook never ran" in f["what"]]
    assert dead and dead[0]["profile"] == "-" and "sitecustomize" in dead[0]["what"]
    # the wrong attribution that used to stand in for it is gone
    assert not any("does not import the kernel as a module" in f["what"]
                   for f in v["findings"])


def test_classify_zero_hook_processes_does_not_mask_a_real_alarm():
    res = {"complete": True, "profiles": [_p_rec("base", card_pass=False, processes=0)]}
    v = monitor.classify(res)
    assert v["status"] == "alarm"                       # the card's verdict still counts
    assert any("monitor hook never ran" in f["what"] for f in v["findings"])


def test_classify_with_a_live_hook_stays_ok_and_old_records_unaffected():
    res = {"complete": True, "profiles": [_p_rec("base", processes=1, calls=[_call()])]}
    v = monitor.classify(res)
    assert v["status"] == "ok"
    assert not any("monitor hook never ran" in f["what"] for f in v["findings"])
    # records written before run.py counted ``processes`` carry no verdict on the
    # hook: no finding either way
    old = {"complete": True, "profiles": [
        {"profile": "base", "card_pass": True, "rc": 0, "seconds": 1.0, "cases": [],
         "hook": {"calls": [], "perturbed": 0, "reseeded": 0}}]}
    v = monitor.classify(old)
    assert v["status"] == "ok"
    assert not any("monitor hook never ran" in f["what"] for f in v["findings"])


# --- BUG-SEED-15: a killed final JSON still yields the finished profiles ----------
def test_parse_log_rebuilds_partial_when_final_json_is_truncated():
    blocks = [_p_rec("base", calls=[_call(pid=7)]),
              _p_rec("repeat", card_pass=False, calls=[_call(i=0, pid=8)])]
    text = "".join(f"{monitor.PROFILE_MARKER}\n{json.dumps(b)}\nnoise\n" for b in blocks)
    text += monitor.MARKER + "\n" + '{"profiles":[{"profile":"base"'   # killed mid-dump
    res = monitor.parse_log(text)
    assert res is not None and res["complete"] is False
    assert [r["profile"] for r in res["profiles"]] == ["base", "repeat"]
    v = monitor.classify(res)
    assert v["status"] == "alarm"          # the verdict flip survives the kill
    assert any("cut short after 2" in f["what"] for f in v["findings"])


def test_parse_log_partial_when_the_marker_stands_alone():
    block = _p_rec("base")
    text = f"{monitor.PROFILE_MARKER}\n{json.dumps(block)}\n{monitor.MARKER}\n"
    res = monitor.parse_log(text)
    assert res is not None and res["complete"] is False and len(res["profiles"]) == 1


def test_parse_log_a_complete_final_json_still_wins():
    final = {"profiles": [_p_rec("base")], "skipped": [], "complete": True}
    text = (f"{monitor.PROFILE_MARKER}\n{json.dumps(_p_rec('base'))}\n"
            f"{monitor.MARKER}\n{json.dumps(final)}")
    res = monitor.parse_log(text)
    assert res == final and res["complete"] is True


# --- BUG-SEED-16: a one-sided ⊗ declaration keeps the pair together ---------------
def test_reopen_pulls_a_one_sided_companion_both_ways():
    traj = [Decision("D1"), Decision("D2", depends_on={"D1"}, companions={"D1"})]
    assert compute_reopen(traj, {"D2"}) == ReopenPlan(reopen=["D1", "D2"], kept=[])
    assert compute_reopen(traj, {"D1"}) == ReopenPlan(reopen=["D1", "D2"], kept=[])


def test_reopen_follows_a_companion_chain_to_the_fixpoint():
    traj = [Decision("A", companions={"B"}), Decision("B", companions={"C"}),
            Decision("C")]
    for changed in ("A", "B", "C"):
        assert compute_reopen(traj, {changed}).reopen == ["A", "B", "C"]


# --- BUG-SEED-25: max_diff signatures keep their exponent --------------------------
def test_diff_signature_distinguishes_exponent_and_bare_forms():
    assert _DIFF_SIG.findall("dq: max_diff=1.5e-05 tol=1e-3") == ["1.5e-05"]
    assert _DIFF_SIG.findall("dq: max_diff=1.5e-07 tol=1e-3") == ["1.5e-07"]
    assert _DIFF_SIG.findall("max_diff=4.28e+00") == ["4.28e+00"]
    assert _DIFF_SIG.findall("max_diff=0 max_diff=0.0 max_diff=inf max_diff=nan") == \
        ["0", "0.0", "inf", "nan"]
    assert _DIFF_SIG.findall("max_diff=7.560892") == ["7.560892"]


def test_tiny_moving_diffs_are_not_a_constant_output_streak():
    # 1.5e-05 vs 1.5e-07 used to both read as "1.5": three genuinely different
    # readings fired the constant-output flag
    recs = [_ir(1, verdict="REVERT", error="correctness FAIL: dq max_diff=1.5e-05"),
            _ir(2, verdict="REVERT", error="correctness FAIL: dq max_diff=1.5e-07"),
            _ir(3, verdict="REVERT", error="correctness FAIL: dq max_diff=1.5e-05")]
    assert constant_output_flag(recs) is None
    same = [_ir(i, verdict="REVERT", error="correctness FAIL: dq max_diff=1.5e-05")
            for i in (1, 2, 3)]
    flag = constant_output_flag(same)
    assert flag is not None and flag["iters"] == [1, 2, 3]


# --- BUG-SEED-41: the hook records its own dump failures ---------------------------
def _load_hook_module():
    import importlib.util
    path = Path(monitor.RUNTIME_DIR) / "hook" / "pk_monitor_hook.py"
    spec = importlib.util.spec_from_file_location("pk_monitor_hook_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dump_failure_is_recorded_and_reported_once(tmp_path, capsys):
    hook = _load_hook_module()
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")          # a file where the dump directory should be
    hook._state.clear()
    hook._state.update({"out": str(blocker), "calls": [{"fn": "call"}]})
    hook._dump()
    hook._dump()
    assert not list(tmp_path.glob("hook-*.json"))       # nothing landed, as before
    errs = hook._state["errors"]                        # but it is recorded now
    assert len(errs) == 2 and errs[0].startswith("dump:")
    assert capsys.readouterr().err.count("pk_monitor_hook") == 1   # said once


def test_dump_failure_names_the_directory_on_stderr(tmp_path, capsys):
    hook = _load_hook_module()
    hook._state.clear()
    hook._state.update({"out": str(tmp_path / "gone"), "calls": []})
    # make makedirs fail: put a plain file in the parent at the target name
    (tmp_path / "gone").write_text("x")
    hook._dump()
    assert "cannot write" in capsys.readouterr().err


# --- BUG-SEED-42: merged per-process calls do not overwrite each other ------------
def test_calls_from_two_processes_both_reach_the_findings():
    base = _p_rec("base", processes=2, calls=[_call(i=0, pid=100, nan=2),
                                              _call(i=0, pid=101, nan=3)])
    v = monitor.classify({"complete": True, "profiles": [base]})
    warns = [f for f in v["findings"] if f["severity"] == "warn" and "NaN" in f["what"]]
    assert len(warns) == 2   # one per process; the old (fn, i) key kept only the last


def test_cross_run_comparison_joins_calls_ignoring_pids():
    def rec(profile, pid, abs_sum, card_pass=True):
        return _p_rec(profile, card_pass, processes=1,
                      calls=[_call(pid=pid, abs_sum=abs_sum)])
    res = {"complete": True,
           "profiles": [rec("base", 100, 10.0), rec("repeat", 200, 10.1)]}
    v = monitor.classify(res)
    assert v["status"] == "warn" and "race" in v["findings"][0]["what"]
    # unchanged inputs across two different pids still read as "not applicable"
    res = {"complete": True,
           "profiles": [rec("base", 100, 10.0), rec("amp", 200, 10.0)]}
    v = monitor.classify(res)
    assert v["not_applicable"] == ["amp"]


# --- BUG-SEED-43: load_records tolerates a torn final line only -------------------
def test_load_records_skips_a_torn_final_line(tmp_path, capsys):
    p = tmp_path / "status.jsonl"
    append_record(p, _ir(0))
    with open(p, "a") as fh:
        fh.write('{"iter": 1, "decision_id": "D2.1", "chan')   # killed mid-append
    recs = load_records(p)
    assert [r.iter for r in recs] == [0]
    assert "torn" in capsys.readouterr().err


def test_load_records_still_raises_on_a_broken_middle_line(tmp_path):
    p = tmp_path / "status.jsonl"
    append_record(p, _ir(0))
    with open(p, "a") as fh:
        fh.write("not json at all\n")
        fh.flush()
    append_record(p, _ir(1))
    with pytest.raises(ValueError):
        load_records(p)


# --- the digest carries the clock telemetry on the curve and the best row ---------
def test_digest_curve_carries_clock_drift_and_best_sm_clock():
    recs = [_ir(0, bwd_median_ms=3.0, device_median_ms=2.9),
            _ir(1, bwd_median_ms=2.5, device_median_ms=2.4, clock_drift=True,
                gpu_state={"sm_clock_mhz": 1590.0, "temperature_c": 61.0}),
            _ir(2, verdict="REVERT", error="dq max_diff=0.9")]
    d = build_digest(recs, {"value": 2.0})
    assert [c["clock_drift"] for c in d["curve"]] == [False, True, False]
    assert d["achieved"]["iter"] == 1
    assert d["achieved"]["sm_clock_mhz"] == 1590.0
    # a best with no gpu_state simply omits the field
    d2 = build_digest([_ir(0, bwd_median_ms=3.0)], {"value": 2.0}, gate_window="e2e")
    assert "sm_clock_mhz" not in d2["achieved"]
    assert d2["curve"][0]["clock_drift"] is False


# --- BUG-SEED-28: the inner results parser ----------------------------------------
def test_parse_correctness_zero_failed_summary_is_not_a_fail():
    r = parse_correctness("  PASS  a\n  PASS  b\n0/2 FAILED\n")
    assert r["passed"] is True and r["fails"] == 0
    # a real N/M FAILED still fails, whatever the per-case lines say
    r = parse_correctness("  PASS  a\n1/2 FAILED\n")
    assert r["passed"] is False and r["fails"] == 1


def test_parse_perf_reports_why_it_found_nothing():
    r = parse_perf("=== JSON ===\n{'trunc")
    assert r["passed"] is None and "note" in r
    assert parse_perf("no json here")["note"]


def test_json_scan_survives_braces_inside_strings():
    text = 'noise {"performance": {"bwd_median_ms": 1.0, "label": "a}b{c"}} {"x": 1}'
    assert parse_perf(text)["bwd_median_ms"] == 1.0
