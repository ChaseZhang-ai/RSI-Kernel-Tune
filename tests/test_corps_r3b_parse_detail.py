"""BUG-CH-E-03 regression: parse_eval survives non-numeric ``error`` values in
correctness check rows.

An own-bench card (a hand-written eval/bench.py is a supported shape) writes a
human-readable verdict string into a check row's ``error`` when it judges FAIL.
``_correctness_fail_detail`` sorted those rows with a bare ``float(c["error"])``
that only special-cased ``None`` — so one string row raised ValueError straight
through parse_eval (five no-network call sites, no try), into the run's safety
net, ending a hours-long run as ``error``. The sister consumer of the same field
(``corr_margin_ratio``) has always guarded with
``isinstance(e, (int, float)) and not isinstance(e, bool)``; this pins the same
guard on the ranking key while the row still displays verbatim.
"""

from rsi_kernel_tune.loop.parse import (
    _correctness_error,
    _correctness_fail_detail,
    parse_eval,
)


def _log(corr: dict) -> str:
    import json
    return "=== JSON ===\n" + json.dumps(
        {"correctness": corr, "performance": None})


def test_string_error_rows_do_not_crash_parse_eval():
    """The reported crash: an own-bench FAIL whose checks rows carry readable
    verdict strings instead of numbers."""
    log = _log({"pass": False, "checks": [
        {"workload": "s1", "group": "q", "output": "o",
         "error": "got 3, expected 1 (way over tol)", "tolerance": 0.1},
        {"workload": "s1", "group": "q", "output": "p",
         "error": "shape mismatch", "tolerance": None},
    ]})
    ev = parse_eval(log)
    assert ev.correctness is False
    # the rows still name their locus and verdict, verbatim
    assert "got 3, expected 1 (way over tol)" in ev.error
    assert "shape mismatch" in ev.error


def test_bench_verdict_with_string_rows_keeps_its_runners_up():
    """Even the common path crashed: a bench that writes its own top-level error
    verdict still calls _correctness_fail_detail for the runners-up."""
    corr = {"pass": False, "error": "q/o judged FAIL by bench",
            "checks": [
                {"group": "q", "output": "o", "error": "judged FAIL", "tolerance": 0.1},
                {"group": "q", "output": "p", "error": 2.5, "tolerance": 0.5},
            ]}
    ev = parse_eval(_log(corr))
    assert ev.correctness is False
    assert "q/o judged FAIL by bench" in ev.error
    assert "worst checks" in ev.error


def test_numeric_rows_still_rank_above_string_rows():
    """Numeric errors keep their ranking; unrankable rows sort last-worst
    (float("inf")) instead of exploding, matching corr_margin_ratio's guard."""
    corr = {"pass": False, "checks": [
        {"group": "a", "output": "x", "error": 0.2, "tolerance": 0.1},
        {"group": "b", "output": "y", "error": "hand verdict", "tolerance": None},
    ]}
    detail = _correctness_fail_detail(corr)
    assert detail.index("b/y") < detail.index("a/x")   # inf ranks first (worst)
    assert "0.2" in detail


def test_bool_error_value_is_not_ranked_as_a_number():
    """isinstance(True, int) is True in Python — the guard must exclude bools
    exactly as corr_margin_ratio does."""
    assert _correctness_error({"pass": False, "checks": [
        {"group": "g", "output": "o", "error": True, "tolerance": 1}]}) != ""


def test_none_error_rows_keep_ranking_as_worst():
    """None (no numeric reading) ranked first (float("inf") under reverse=True)
    before the fix and must keep doing so."""
    detail = _correctness_fail_detail({"pass": False, "checks": [
        {"group": "a", "output": "x", "error": None, "tolerance": 1},
        {"group": "b", "output": "y", "error": 5.0, "tolerance": 1},
    ]})
    assert detail.index("a/x") < detail.index("b/y")
