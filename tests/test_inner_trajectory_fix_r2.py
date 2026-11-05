"""Regression tests for the audit corps' trajectory fix batch (r2, inner cluster).

Covers: load_records tolerating a torn final line only (BUG-SEED-43) and
max_diff signatures keeping their exponent (BUG-SEED-25).
"""

from __future__ import annotations

import pytest

from rsi_kernel_tune.inner.trajectory import (
    IterationRecord,
    _DIFF_SIG,
    append_record,
    constant_output_flag,
    load_records,
)


def _ir(i, *, verdict="KEEP", error="", **kw):
    kw.setdefault("bwd_median_ms", None if error else 2.0)
    return IterationRecord(iter=i, decision_id="D", change="c",
                           correctness=None if verdict == "NEUTRAL" else not error,
                           verdict=verdict, error=error, **kw)


# --- BUG-SEED-43: a torn final line does not poison the trajectory ----------------
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


# --- BUG-SEED-25: max_diff signatures keep their exponent -------------------------
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
