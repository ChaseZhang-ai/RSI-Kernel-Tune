"""forgekernel.benchx.api: the interface every evaluation of a candidate goes through.

The real Bench is task benchx's; what is pinned here is what everything else codes against:
the data classes, the protocol, ``FakeBench``, the cold-start budget and the tag strings.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

import forgekernel.benchx as benchx
from forgekernel.benchx import api
from forgekernel.benchx.api import (COLD_START_BUDGET_S, CONF_DEV_ONLY, CONF_FP4_BOUND,
                                    CONF_SFU_BOUND, CONF_SM_COUNT_SENSITIVE, MODE_DEV,
                                    MODE_TARGET, MODES, Bench, BenchCall, BenchResult,
                                    FakeBench, WorkloadReading, cold_start_over_budget)


def _reading(name="w0", t_ms=1.0, **kw) -> WorkloadReading:
    return WorkloadReading(name=name, t_ms=t_ms, **kw)


def _result(correct=True, s=0.5, workloads=None, **kw) -> BenchResult:
    return BenchResult(correct=correct, workloads=workloads if workloads is not None
                       else [_reading(s=s)], mean_s=s, **kw)


# --- the interface says what it is ----------------------------------------------------------


def test_the_module_tells_everyone_not_to_write_a_second_evaluator():
    doc = api.__doc__
    assert "Do not write a second evaluator: wait for benchx" in doc
    assert "card's own clock" in doc and "card's own judge" in doc


def test_the_package_exports_the_interface():
    assert len(benchx.__all__) == len(set(benchx.__all__))
    for name in benchx.__all__:
        assert getattr(benchx, name) is getattr(api, name), name


def test_the_tag_strings_other_tasks_match_on_are_named_once():
    assert (MODE_DEV, MODE_TARGET, MODES) == ("dev", "target", ("dev", "target"))
    assert CONF_DEV_ONLY == "dev-only"
    assert CONF_FP4_BOUND == "fp4-bound"
    assert CONF_SFU_BOUND == "sfu-bound"
    assert CONF_SM_COUNT_SENSITIVE == "sm-count-sensitive"


def test_the_fields_are_the_contract_the_other_tasks_code_against():
    """Names and order: the other tasks build these positionally and read them by name."""
    assert [f.name for f in dataclasses.fields(BenchResult)] == [
        "correct", "workloads", "mean_s", "fast1", "integrity", "job", "elapsed_s", "error",
        "cold_start_s", "confidence"]
    assert [f.name for f in dataclasses.fields(WorkloadReading)] == [
        "name", "axes", "t_ms", "t_b", "t_sol", "s", "correct", "error"]


# --- readings -------------------------------------------------------------------------------


def test_a_reading_is_empty_until_the_card_measures_it():
    w = WorkloadReading("decode_b64")
    assert (w.name, w.axes, w.t_ms, w.t_b, w.t_sol, w.s, w.correct, w.error) == \
        ("decode_b64", {}, None, None, None, None, None, "")
    assert WorkloadReading("w", {"batch": 64}, 0.5, 1.0, 0.25, 0.5, True).axes == {"batch": 64}


def test_a_result_carries_its_readings_and_what_else_is_known():
    r = _result(workloads=[_reading("a", 1.0), _reading("b", 2.0)], fast1=1,
                integrity={"cold_start": "over-budget"}, job="1789171541", elapsed_s=12.5,
                cold_start_s=101.0, confidence=["dev-only"])
    assert r.by_name() == {"a": r.workloads[0], "b": r.workloads[1]}
    assert (r.correct, r.mean_s, r.error) == (True, 0.5, "")
    plain = BenchResult(correct=False, workloads=[], mean_s=None)
    assert (plain.fast1, plain.integrity, plain.job, plain.elapsed_s, plain.error) == \
        (0, {}, "", 0.0, "")
    assert (plain.cold_start_s, plain.confidence) == (None, [])


def test_two_results_do_not_share_a_list_they_both_defaulted():
    a, b = BenchResult(True, [], None), WorkloadReading("w")
    a.confidence.append("dev-only")
    assert BenchResult(True, [], None).confidence == []
    b.axes["x"] = 1
    assert WorkloadReading("w").axes == {}


# --- the cold-start budget -----------------------------------------------------------------


def test_the_cold_start_budget_is_100_seconds_and_only_a_measured_reading_can_be_past_it():
    assert COLD_START_BUDGET_S == 100.0
    assert not cold_start_over_budget(None)      # not measured is unknown, not "fast"
    assert not cold_start_over_budget(0.0) and not cold_start_over_budget(99.9)
    assert not cold_start_over_budget(COLD_START_BUDGET_S)
    assert cold_start_over_budget(COLD_START_BUDGET_S + 0.001)


# --- the protocol --------------------------------------------------------------------------


def test_a_bench_is_whatever_answers_bench():
    assert isinstance(FakeBench(), Bench)

    class Inline:                                # a caller's own stand-in, not derived from anything
        def bench(self, card_dir, candidate, *, workloads=None, mode="dev"):
            return _result()

    assert isinstance(Inline(), Bench)
    assert not isinstance(object(), Bench)
    assert not isinstance(_result(), Bench)


# --- the fake ------------------------------------------------------------------------------


def test_the_fake_hands_back_one_result_and_records_every_call(tmp_path):
    answer, card = _result(s=0.7), tmp_path / "card"
    fake = FakeBench(answer)
    candidates = [tmp_path / "a.py", {"language": "cuda_cpp", "sources": {}}, "b.py"]
    assert all(fake.bench(card, c) is answer for c in candidates)
    assert fake.calls == [BenchCall(card, c, None, "dev") for c in candidates]


def test_the_fake_plays_a_sequence_and_repeats_its_last_answer():
    first, last = _result(s=0.1), _result(s=0.9)
    fake = FakeBench([first, last])
    assert [fake.bench("c", "k") for _ in range(4)] == [first, last, last, last]
    assert len(fake.calls) == 4


def test_the_fake_answers_from_the_call_it_was_given():
    fast, slow = _result(s=0.9), _result(correct=False, s=None, error="correctness FAIL")
    fake = FakeBench(lambda call: fast if str(call.candidate).endswith("fast.py") else slow)
    assert fake.bench(Path("/cards/x"), "/k/fast.py", workloads=[0, 3], mode="target") is fast
    assert fake.bench("/cards/x", "/k/slow.py", workloads="0,3,5-9") is slow
    first, second = fake.calls
    assert (first.card_dir, first.candidate, first.workloads, first.mode) == \
        (Path("/cards/x"), "/k/fast.py", [0, 3], "target")
    assert (second.workloads, second.mode) == ("0,3,5-9", "dev")


def test_the_fake_without_a_script_passes_and_says_nothing_was_read():
    fake = FakeBench()
    r = fake.bench("c", "k")
    assert (r.correct, r.workloads, r.mean_s, r.cold_start_s) == (True, [], None, None)
    assert fake.calls == [BenchCall(Path("c"), "k", None, "dev")]


def test_the_fake_refuses_a_mode_it_does_not_know_and_records_nothing_for_it():
    fake = FakeBench(_result())
    with pytest.raises(ValueError, match="mode must be one of"):
        fake.bench("c", "k", mode="Dev")
    assert fake.calls == []


def test_a_recorded_call_cannot_be_edited_after_the_fact():
    fake = FakeBench()
    fake.bench("c", "k")
    with pytest.raises(dataclasses.FrozenInstanceError):
        fake.calls[0].mode = "target"
