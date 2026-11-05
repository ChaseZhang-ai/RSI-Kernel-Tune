"""Regression tests for the audit-corps fix wave 7 (the parse/inner/trajectory
surface): non-finite eval readings must not hijack the plateau minimum, scalar
eval readings must be real numbers, scientific-notation correctness diffs must
keep their signatures, the correctness summary must trust its own failure count,
the perf JSON scan must respect string literals, and a torn trajectory line must
not sink the run's whole history."""
import json

from rsi_kernel_tune.inner.results import (_first_json_with_key,
                                            parse_correctness)
from rsi_kernel_tune.inner.trajectory import (IterationRecord, _DIFF_SIG,
                                               constant_output_flag)
from rsi_kernel_tune.loop.parse import (Exhausted, ProbeRequest, Record,
                                         _float_list, detect_plateau, parse_eval)


def _eval_text(**perf):
    body = {"correctness": {"pass": True}, "performance": perf}
    return "=== JSON ===\n" + json.dumps(body)


def _rec(it, ms, verdict="KEEP"):
    # dual-window record, same reading in both (the plateau math is window-agnostic)
    return IterationRecord(iter=it, decision_id="D", change="c", correctness=True,
                           verdict=verdict, bwd_median_ms=ms, device_median_ms=ms)


# --- BUG-SEED-22: NaN readings reached detect_plateau and hijacked its minimum ---

def test_float_list_drops_non_finite_entries():
    # same filter readings_spread_pct applies: a NaN reading is a broken
    # measurement, not a small one
    assert _float_list([1.0, float("nan"), 3.0, float("inf"), True, "2.0"]) == [1.0, 3.0]
    assert _float_list("not a list") == []


def test_parse_eval_drops_non_finite_scalar_readings():
    ev = parse_eval(_eval_text(device_median_ms=float("nan"), bwd_median_ms=2.5,
                               ref_device_ms=float("nan"), vs_ref=float("nan"),
                               host_enqueue_ms=float("inf")))
    assert ev.device_median_ms is None
    assert ev.bwd_median_ms == 2.5
    assert ev.ref_device_ms is None
    assert ev.vs_ref is None
    assert ev.host_enqueue_ms is None


def test_parse_eval_drops_non_finite_workload_readings():
    ev = parse_eval(_eval_text(bwd_median_ms=2.5, workloads=[
        {"name": "w0", "device_median_ms": float("nan"),
         "device_readings_ms": [1.0, float("nan"), 3.0], "vs_ref_device": 1.0},
        {"name": "w1", "device_median_ms": 2.0,
         "device_readings_ms": [2.0, float("inf")], "vs_ref_device": float("nan")},
    ]))
    w0, w1 = ev.workloads
    assert "device_median_ms" not in w0
    assert w0["device_readings_ms"] == [1.0, 3.0]
    assert w0["vs_ref_device"] == 1.0
    assert w1["device_median_ms"] == 2.0
    assert w1["device_readings_ms"] == [2.0]
    assert "vs_ref_device" not in w1


def test_a_nan_reading_cannot_hijack_the_plateau_minimum():
    # six-row shape: no new best inside the window -> plateau True
    rows = [_rec(1, 1.00), _rec(2, 1.02), _rec(3, 1.03), _rec(4, 1.04), _rec(5, 1.05)]
    assert detect_plateau(rows, window=3) is True
    # the driver builds its records from parse_eval's output, so the NaN eval must
    # arrive as "no reading": the row drops out of the evaluated set instead of
    # sitting first in the min() and turning the plateau verdict permanently False
    ev = parse_eval(_eval_text(device_median_ms=float("nan"), bwd_median_ms=2.5))
    assert ev.device_median_ms is None
    nan_row = _rec(0, ev.device_median_ms)
    assert detect_plateau([nan_row] + rows, window=3) is True


# --- BUG-T4C-01: scalar eval readings swallowed bools/strings the list fields ---

def test_scalar_eval_readings_reject_bool_and_string():
    # the list fields have always demanded a real number; the scalars took
    # anything float() swallowed, so a malformed bench's `true` became a
    # 1.0 ms gate reading
    ev = parse_eval(_eval_text(device_median_ms=True, bwd_median_ms="2.5",
                               ref_device_ms="0.9", vs_ref=False,
                               mfu="42", host_bound_ratio=True))
    assert ev.device_median_ms is None
    assert ev.bwd_median_ms is None
    assert ev.ref_device_ms is None
    assert ev.vs_ref is None
    assert ev.mfu is None
    assert ev.host_bound_ratio is None


def test_scalar_eval_readings_still_accept_real_numbers():
    ev = parse_eval(_eval_text(device_median_ms=2, bwd_median_ms=2.5,
                               ref_device_ms=3, vs_ref=1.2, mfu=36))
    assert ev.device_median_ms == 2.0
    assert ev.bwd_median_ms == 2.5
    assert ev.ref_device_ms == 3.0
    assert ev.vs_ref == 1.2
    assert ev.mfu == 36.0


def test_scalar_eval_readings_reject_garbage_without_raising():
    # a non-numeric string used to raise ValueError out of parse_eval; now the
    # field simply reads as absent
    ev = parse_eval(_eval_text(device_median_ms="fast", bwd_median_ms=[1.0]))
    assert ev.device_median_ms is None
    assert ev.bwd_median_ms is None


# --- BUG-SEED-25: the diff signature truncated scientific notation to the ------
# mantissa, so distinct small-diff FAILs read as one bit-identical signature and
# falsely fired the constant-output diagnosis.

def _fail_rec(it, err):
    return IterationRecord(iter=it, decision_id=f"d{it}", change="c",
                           correctness=False, verdict="REVERT", error=err)


def test_diff_signature_keeps_the_exponent():
    assert _DIFF_SIG.findall("max_diff=1.5e-05") == ["1.5e-05"]
    assert _DIFF_SIG.findall("max_diff=1.5e-07") == ["1.5e-07"]
    assert _DIFF_SIG.findall("max_diff=1.5E+02 exceeds 1e-2") == ["1.5E+02"]
    # plain decimals, inf and nan signatures unchanged
    assert _DIFF_SIG.findall("max_diff=4.28") == ["4.28"]
    assert _DIFF_SIG.findall("max_diff=inf") == ["inf"]
    assert _DIFF_SIG.findall("max_diff=nan") == ["nan"]


def test_distinct_scientific_diffs_do_not_fire_constant_output():
    # three distinct FAILs whose mantissas agree: a real numeric bug that moves
    # (1.5e-05 -> 1.5e-07 -> 1.5e-09) must not read as "the output never changes"
    recs = [_fail_rec(1, "FAIL: max_diff=1.5e-05 exceeds 1e-2"),
            _fail_rec(2, "FAIL: max_diff=1.5e-07 exceeds 1e-2"),
            _fail_rec(3, "FAIL: max_diff=1.5e-09 exceeds 1e-2")]
    assert constant_output_flag(recs) is None


def test_identical_scientific_diffs_still_fire_constant_output():
    recs = [_fail_rec(1, "FAIL: max_diff=1.5e-05 exceeds 1e-2"),
            _fail_rec(2, "FAIL: max_diff=1.5e-05 exceeds 1e-2"),
            _fail_rec(3, "FAIL: max_diff=1.5e-05 exceeds 1e-2")]
    flag = constant_output_flag(recs)
    assert flag is not None and flag["kind"] == "constant_output"
    assert flag["signature"] == ["1.5e-05"]


# --- BUG-SEED-28: "0/M FAILED" contradicted its own counter (passed=False with --
# fails=0), and the hand-rolled brace count returned None for any JSON whose
# string values contained a closing brace.

def test_zero_failed_summary_trusts_its_count():
    r = parse_correctness("  PASS  case_a\n0/3 FAILED")
    assert r["passed"] is True
    assert r["fails"] == 0
    assert [c["name"] for c in r["cases"]] == ["case_a"]


def test_failed_summary_with_failures_still_fails():
    r = parse_correctness("  PASS  case_a\n  FAIL  case_b\n1/2 FAILED")
    assert r["passed"] is False
    assert r["fails"] == 1


def test_json_scan_respects_string_literals():
    # a literal '}' inside a string value used to end the brace count early and
    # return None for otherwise clean input
    text = '{"note": "a literal } brace", "performance": {"bwd_median_ms": 2.5}}'
    obj = _first_json_with_key(text, "performance")
    assert obj is not None and obj["performance"] == {"bwd_median_ms": 2.5}
    assert _first_json_with_key('{"performance": {"esc \\" quote": 1}}',
                                "performance") == {"performance": {'esc " quote': 1}}


def test_json_scan_still_finds_the_plain_block():
    assert _first_json_with_key('{"performance": {"x": 1}}',
                                "performance") == {"performance": {"x": 1}}
    assert _first_json_with_key("no json at all", "performance") is None
    assert _first_json_with_key('{"other": {"a": 1}}', "performance") is None


# --- BUG-SEED-43: one torn line (a run killed mid-append) made load_records -----
# raise JSONDecodeError, so the whole trajectory — and every resume of it — was
# unreadable. A torn FINAL line (the only corruption an append can produce) is now
# dropped with a stderr note; a broken line anywhere else still raises, since
# silently truncating the middle would mis-read the run's history.

def _keep_rec(it):
    return IterationRecord(iter=it, decision_id=f"d{it}", change=f"c{it}",
                           correctness=True, verdict="KEEP")


def test_load_records_tolerates_a_torn_final_line(tmp_path, capsys):
    from rsi_kernel_tune.inner.trajectory import append_record, load_records
    p = tmp_path / "status.jsonl"
    append_record(p, _keep_rec(1))
    with open(p, "a") as fh:
        fh.write('{"iter": 2, "decision_id": "d2", "change"')
    recs = load_records(p)
    assert [r.iter for r in recs] == [1]          # the history stays readable
    err = capsys.readouterr().err
    assert "torn" in err and "status.jsonl" in err


def test_load_records_still_raises_on_a_broken_middle_line(tmp_path):
    # Adapted at the merge: the lenient any-line drop (the r3 line's variant) was
    # deduped in favor of the narrow contract -- the middle of an append-only file
    # cannot be torn, so a broken line there is real corruption and must be loud.
    import pytest

    from rsi_kernel_tune.inner.trajectory import append_record, load_records
    p = tmp_path / "status.jsonl"
    append_record(p, _keep_rec(1))
    with open(p, "a") as fh:
        fh.write("not json at all\n")
    append_record(p, _keep_rec(2))
    with pytest.raises(ValueError):
        load_records(p)


def test_load_records_clean_file_warns_nothing(tmp_path, capsys):
    from rsi_kernel_tune.inner.trajectory import append_record, load_records
    p = tmp_path / "status.jsonl"
    append_record(p, _keep_rec(1))
    append_record(p, _keep_rec(2))
    assert [r.iter for r in load_records(p)] == [1, 2]
    assert capsys.readouterr().err == ""


# --- BUG-SEED-50: an unanchored scan classified any turn containing the marker --
# text anywhere in it, so a turn that merely echoed "PROBE_REQUEST: ..." burned a
# GPU probe and an echoed "EXHAUSTED: ..." stopped the run.

def _import_parse_student_turn():
    from rsi_kernel_tune.loop.parse import parse_student_turn
    return parse_student_turn


def test_an_echoed_probe_marker_does_not_burn_a_probe():
    parse = _import_parse_student_turn()
    text = ('I will not use PROBE_REQUEST: notes/x.py now that the probe budget '
            'is spent; the record stands.')
    try:
        turn = parse(text)
        assert not isinstance(turn, ProbeRequest)
    except ValueError:
        pass    # no record block either: unparseable is fine, a probe is not


def test_an_echoed_exhausted_marker_does_not_stop_the_run():
    parse = _import_parse_student_turn()
    text = ('The protocol reminder says to end with EXHAUSTED: no decisions left, '
            'but I have one more lever to try.\ndecision_id: D9\nchange: unroll')
    turn = parse(text)
    assert not isinstance(turn, Exhausted)


def test_an_echoed_marker_yields_the_record_block():
    parse = _import_parse_student_turn()
    text = ('As the prompt says, a probe is PROBE_REQUEST: notes/whatever.py — '
            'meanwhile:\ndecision_id: D7\nchange: vectorize\nfiles_touched: kernel.py')
    turn = parse(text)
    assert isinstance(turn, Record) and turn.decision_id == "D7"


def test_own_line_markers_still_parse():
    parse = _import_parse_student_turn()
    assert parse("prose first\nPROBE_REQUEST: notes/x.py").path == "notes/x.py"
    assert parse("  EXHAUSTED: nothing left to try\n").reason == "nothing left to try"
    assert parse("EXHAUSTED: done").reason == "done"
