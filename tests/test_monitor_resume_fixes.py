"""Regression tests ported from the ForgeKernel fix wave (monitor / resume batch).

Only the mechanisms that exist in this lineage are pinned here; each names the fix
it carries:

* parse_log: a complete final JSON still wins over the per-profile blocks (the
  killed-final-JSON partial rebuild is NOT in this lineage -- parse_log returns
  None when the marker's own JSON is truncated, see the BUG-PORT ledger);
* the cross-run comparison joins the per-(fn, i) calls of two profiles ignoring
  pids, so equal inputs with diverging checksums read as a race and unchanged
  inputs read as not-applicable;
* the inner results scanner survives braces inside JSON string values.

Dropped from the source file, with reasons (upstream fix absent or superseded
here -- see the BUG-PORT / port ledger):

* classify's dead-hook warn ("the monitor hook never ran", per-process call
  keys): classify never reads the ``processes`` count runtime/run.py writes, and
  _calls keys by (fn, i) only;
* parse_log rebuilding a partial result when the final JSON was killed mid-dump:
  parse_log returns None in that case (docstring promises, code does not);
* one-sided companion declarations reopening the pair from either side:
  compute_reopen follows companions only from the reopened decision's own set;
* _DIFF_SIG keeping exponents / constant_output_flag on tiny moving diffs: the
  signature regex here has no exponent alternative;
* the hook recording its own dump failures: _dump swallows every exception;
* load_records tolerating a torn final line only: any bad line raises;
* the digest carrying WP1 clock telemetry in curve/achieved: superseded by the
  WP15 top-level ``clock`` header (the telemetry itself is carried, elsewhere);
* parse_correctness reading "0/2 FAILED" as a fail / parse_perf's ``note``.
"""

from __future__ import annotations

import json

from peregrine_kernel import monitor
from peregrine_kernel.inner.results import parse_perf


def _p_rec(profile, card_pass=True, *, calls=()):
    """A per-profile record the way runtime/run.py writes it."""
    return {"profile": profile, "card_pass": card_pass, "rc": 0, "seconds": 1.0,
            "cases": [],
            "hook": {"calls": list(calls), "perturbed": 0, "reseeded": 0,
                     "errors": []}}


def _call(i=0, *, abs_sum=10.0):
    return {"fn": "call", "i": i, "inputs_nonfinite": 0, "in_sums": [1.0],
            "outputs": [{"nan": 0, "posinf": 0, "neginf": 0, "abs_sum": abs_sum,
                         "sq_sum": abs_sum / 2}]}


# --- a complete final JSON still wins over the profile blocks ---------------------
def test_parse_log_a_complete_final_json_still_wins():
    final = {"profiles": [_p_rec("base")], "skipped": [], "complete": True}
    text = (f"{monitor.PROFILE_MARKER}\n{json.dumps(_p_rec('base'))}\n"
            f"{monitor.MARKER}\n{json.dumps(final)}")
    res = monitor.parse_log(text)
    assert res == final and res["complete"] is True


# --- the cross-run comparison joins calls ignoring pids ---------------------------
def test_cross_run_comparison_joins_calls_ignoring_pids():
    def rec(profile, pid, abs_sum, card_pass=True):
        block = _p_rec(profile, card_pass, calls=[_call(abs_sum=abs_sum)])
        block["hook"]["calls"][0]["pid"] = pid
        return block

    res = {"complete": True,
           "profiles": [rec("base", 100, 10.0), rec("repeat", 200, 10.1)]}
    v = monitor.classify(res)
    assert v["status"] == "warn" and "race" in v["findings"][0]["what"]
    # unchanged inputs across two different pids still read as "not applicable"
    res = {"complete": True,
           "profiles": [rec("base", 100, 10.0), rec("amp", 200, 10.0)]}
    v = monitor.classify(res)
    assert v["not_applicable"] == ["amp"]


# --- the inner results scanner ----------------------------------------------------
def test_json_scan_survives_braces_inside_strings():
    text = 'noise {"performance": {"bwd_median_ms": 1.0, "label": "a}b{c"}} {"x": 1}'
    assert parse_perf(text)["bwd_median_ms"] == 1.0
