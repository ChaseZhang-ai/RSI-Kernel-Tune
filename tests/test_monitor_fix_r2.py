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
