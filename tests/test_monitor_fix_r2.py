"""Regression tests for the audit corps' monitor fix batch (r2, monitor cluster).

Covers: classify flagging a monitor whose hook was injected into no process
(BUG-SEED-14), parse_log rebuilding a partial result when the final JSON dump was
killed mid-line (BUG-SEED-15), and the hook recording its own dump failures
(BUG-SEED-41).
"""

from __future__ import annotations

import json
from pathlib import Path

from peregrine_kernel import monitor


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


# --- BUG-SEED-41: the hook records its own dump failures --------------------------
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
    # Adapted at the merge: the r2 variant recorded each failure in a state
    # "errors" list; the merged hook (the r3 line's, ruled OURS) records the
    # failure as a one-shot flag and says so once on stderr -- a summary that is
    # going nowhere can never show a recorded list, so the flag is the record.
    assert hook._state["dump_failed"] is True
    assert capsys.readouterr().err.count("pk-monitor") == 1   # said once, not twice


# --- BUG-SEED-42: per-process calls do not overwrite each other -------------------
def test_calls_from_two_processes_both_reach_the_findings():
    # two processes both counted their first fwd call from 0: the NaN one is recorded
    # FIRST, the clean one LAST — the old (fn, i) key let the clean record overwrite
    # the NaN evidence
    base = _p_rec("base", processes=2,
                  calls=[_call(fn="fwd", i=0, pid=100, nan=3),
                         _call(fn="fwd", i=0, pid=101, nan=0)])
    v = monitor.classify({"complete": True, "profiles": [base]})
    assert v["status"] == "warn" and "NaN" in v["findings"][0]["what"]
    assert "3 NaN" in v["findings"][0]["what"]    # the poisoned record, not the clean one
    base_calls = monitor._calls(base)
    assert len(base_calls) == 2                   # both processes' records survive


def test_cross_profile_call_join_ignores_pids():
    # the pid only separates records inside one profile: base and its repeat ran in
    # different processes, and their comparison must still happen
    def rec(profile, pid, abs_sum):
        return _p_rec(profile, processes=1, calls=[_call(pid=pid, abs_sum=abs_sum)])
    res = {"complete": True, "profiles": [rec("base", 100, 10.0), rec("repeat", 200, 10.1)]}
    v = monitor.classify(res)
    assert v["status"] == "warn" and "race" in v["findings"][0]["what"]
    # unchanged inputs across two different pids still read as "not applicable"
    res = {"complete": True, "profiles": [rec("base", 100, 10.0), rec("amp", 200, 10.0)]}
    v = monitor.classify(res)
    assert v["not_applicable"] == ["amp"]
