"""Regression tests for the audit corps' trajectory fix batch (r2, inner cluster).

Covers: load_records tolerating a torn final line only (BUG-SEED-43).
"""

from __future__ import annotations

import pytest

from peregrine_kernel.inner.trajectory import (
    IterationRecord,
    append_record,
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
