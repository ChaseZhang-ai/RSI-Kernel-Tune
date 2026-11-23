"""The bench reader: fake GPU job logs in the card's own format -> BenchResult.

The logs are built the way eval/leaderboard.py writes them (a human preamble, then
``=== JSON ===`` and the eval JSON: correctness + performance + the
``leaderboard.sol_board`` block), so the reader is tested against the card's format, not
against a format of our own.
"""

import json

import pytest

from forgekernel.benchx import reader
from forgekernel.benchx.api import BenchResult, FakeBench, WorkloadReading
from forgekernel.leaderboard import sol_score


# -- fixtures: the card's own log format ------------------------------------------------------


def problem(workloads=(("wl0", {"m": 128}),), dps=True) -> dict:
    group = [{"index": i, "name": name, "workload": {"uuid": f"u{i}", "axes": axes},
              "sol_board": {"kernel_id": "k", "workload_index": i, "t_b_ms": 1.0,
                            "t_sol_ms": 0.2, "source": "test"}}
             for i, (name, axes) in enumerate(workloads)]
    p = {
        "schema": 1, "suite": "sol-execbench",
        "definition": {"name": "demo_op", "inputs": {}, "outputs": {},
                       "reference": "def run(x):\n    return x"},
        "workload": group[0]["workload"], "dps": dps, "kernel_file": "kernel.py",
        "entry": "run", "sol_board": group[0]["sol_board"],
    }
    if len(group) > 1:
        p["group"] = group
    return p


def correctness(passed=True, rows=None, error=None) -> dict:
    out = {"pass": passed, "status": "PASSED" if passed else "FAILED",
           "max_abs": 0.0, "max_rel": 0.0, "matched_ratio": 1.0, "checks": [],
           "n_workloads": len(rows or []), "max_diff": 0.0}
    if rows:
        out["workloads"] = rows
    if error:
        out["status"] = "FAILED"
        out["error"] = error
    return out


def perf_single(t_ms=0.5, t_b=1.0, t_sol=0.2, s=None, basis="cupti_kernel_span",
                clock="locked sm=1500 mem=3996") -> dict:
    return {
        "bwd_median_ms": round(t_ms * 1.2, 4), "min_ms": round(t_ms * 0.95, 4),
        "device_median_ms": t_ms, "device_window_basis": basis,
        "host_enqueue_ms": 0.01, "host_bound_ratio": 0.02,
        "in_job_ref": "not timed (problem.json time_reference)",
        "leaderboard": {
            "suite": "sol-execbench", "definition": "demo_op", "workload": "u0",
            "clock": clock, "protocol": {"aggregate": "mean"},
            "kernel_per_set_ms": [t_ms],
            "sol_board": {"t_b_ms": t_b, "t_sol_ms": t_sol,
                          "sol_score": sol_score(t_ms, t_b, t_sol) if s is None else s,
                          "vs_t_b": round(t_b / t_ms, 4), "kernel_id": "k",
                          "workload_index": 0},
        },
    }


def perf_group(rows) -> dict:
    blocks = []
    for name, t_ms, t_b, t_sol in rows:
        blocks.append({
            "name": name, "device_median_ms": t_ms, "bwd_median_ms": round(t_ms * 1.2, 4),
            "mins_ms": None, "device_window_basis": "cupti_kernel_span",
            "leaderboard": {"sol_board": {
                "t_b_ms": t_b, "t_sol_ms": t_sol,
                "sol_score": round(sol_score(t_ms, t_b, t_sol), 4)}}})
    scores = {b["name"]: b["leaderboard"]["sol_board"]["sol_score"] for b in blocks}
    return {
        "device_median_ms": 1.0, "bwd_median_ms": 1.1, "n_workloads": len(blocks),
        "aggregate": "geomean", "device_window_basis": "cupti_kernel_span",
        "workloads": blocks,
        "leaderboard": {
            "suite": "sol-execbench", "definition": "demo_op", "workload": "u0",
            "clock": "locked sm=1500 mem=3996", "protocol": {"aggregate": "geomean"},
            "sol_board": {"sol_score_mean": round(sum(scores.values()) / len(scores), 4),
                          "sol_score": scores, "vs_t_b_geomean": 1.2,
                          "kernel_id": "k", "workload_index": [0, 1, 2]},
        },
    }


def log(corr, perf, *, tail="") -> str:
    body = {"correctness": corr, "performance": perf}
    return ("=== gpu_entry.sh (demo_op, sol-execbench) ===\n"
            "  task:      all\n  clocks:    locked sm=1500 mem=3996\n"
            "[leaderboard] correctness: PASSED\n\n"
            "=== JSON ===\n" + json.dumps(body, indent=2) + "\n" + tail)


EMPTY_LOG = """\
=== gpu_entry.sh (demo_op, sol-execbench) ===
  task:      all
[leaderboard] scoped kernel: COMPILE_ERROR -- boom
"""

REFUSED_LOG = """\
=== gpu_entry.sh (demo_op, sol-execbench) ===
  task:      all
  clocks:    lock not held (asked sm=1500 mem=3996, read 1155,3996)

[gpu_entry.sh] REFUSED: this card times under the board's clock lock, and it did not hold here (lock not held (asked sm=1500 mem=3996, read 1155,3996)). LB_REQUIRE_CLOCK_LOCK=0 times anyway (the readings are then not comparable to the board's).
"""

PROBE = ("=== gpu_entry.sh (demo_op, sol-execbench) ===\n  task:      cmd\n"
         "[benchx] COLD_START_S {cold:.3f} (card demo_op, kernel kernel.py)\n")


# -- one job ----------------------------------------------------------------------------------


def test_single_workload_reading_is_the_cards():
    p = problem()
    result = reader.build_result([log(correctness(), perf_single())], problem=p, jobs=["7"],
                                 cold_log=PROBE.format(cold=42.5))
    assert result.correct is True
    assert result.error == ""
    assert result.job == "7"
    assert result.cold_start_s == 42.5
    assert result.integrity["cold_start"] == "ok"
    assert result.confidence == ["dev-only"]
    assert result.fast1 == 1 and result.fast1_ratio() == 1.0
    (w,) = result.workloads
    assert w.name == "u0"                     # a one-workload card is named by its uuid
    assert w.axes == {"m": 128}
    assert w.t_ms == 0.5
    assert w.t_b == 1.0 and w.t_sol == 0.2
    assert w.s == pytest.approx(sol_score(0.5, 1.0, 0.2))
    assert w.correct is True
    assert result.mean_s == w.s               # the card's score, not re-derived
    assert result.integrity["clock"] == "locked sm=1500 mem=3996"
    assert result.integrity["reading_basis"] == "cupti_kernel_span"


def test_unlocked_reading_carries_the_clock_note():
    p = problem()
    result = reader.build_result(
        [log(correctness(), perf_single(clock="unlocked"))], problem=p)
    assert result.correct
    assert "clock lock" in result.integrity["clock_note"]


def test_group_card_reading_is_the_cards_mean():
    p = problem(workloads=(("wl0", {"m": 128}), ("wl1", {"m": 256}), ("wl2", {"m": 512})))
    rows = [("wl0", 0.5, 1.0, 0.2), ("wl1", 1.2, 2.0, 0.4), ("wl2", 3.0, 2.0, 0.4)]
    corr = correctness(rows=[{"name": n, "status": "PASSED"} for n, *_ in rows])
    result = reader.build_result([log(corr, perf_group(rows))], problem=p, jobs=["9"])
    assert result.correct
    assert [w.name for w in result.workloads] == ["wl0", "wl1", "wl2"]
    assert [w.t_ms for w in result.workloads] == [0.5, 1.2, 3.0]
    assert result.fast1 == 2                  # wl0 and wl1 read faster than their T_b
    card_scores = [round(sol_score(0.5, 1.0, 0.2), 4), round(sol_score(1.2, 2.0, 0.4), 4),
                   round(sol_score(3.0, 2.0, 0.4), 4)]
    # The card's own mean: over the scores it rounded and wrote, not over re-derived ones.
    assert result.mean_s == pytest.approx(round(sum(card_scores) / 3, 4))
    for w, s in zip(result.workloads, card_scores):
        assert w.s == pytest.approx(s)             # the card's rounded per-workload score


def test_correctness_failure_keeps_the_cards_words_and_no_readings():
    p = problem(workloads=(("wl0", {"m": 128}), ("wl1", {"m": 256})))
    corr = correctness(False, rows=[{"name": "wl0", "status": "PASSED"},
                                    {"name": "wl1", "status": "FAILED"}],
                       error="workload wl1 {\"m\": 256}: max|y - judge| over tolerance")
    result = reader.build_result([log(corr, None)], problem=p, jobs=["11"])
    assert result.correct is False
    assert result.error == corr["error"]      # verbatim, the card's own sentence
    assert result.mean_s is None and result.fast1 == 0
    assert result.fast1_ratio() is None
    assert result.timed() == []
    assert result.integrity["failure_kind"] == "incorrect"
    wl0, wl1 = result.workloads
    assert wl0.correct is True and wl0.t_ms is None
    assert wl1.correct is False and wl1.error == corr["error"]


def test_a_job_that_died_before_a_verdict_is_not_a_half_reading():
    p = problem()
    result = reader.build_result([EMPTY_LOG], problem=p, jobs=["12"])
    assert result.correct is False
    assert "no eval JSON" in result.error
    assert result.timed() == [] and result.mean_s is None
    assert result.integrity["failure_kind"] == "no_verdict"


def test_a_clock_lock_refusal_is_named_as_the_machine_fault_it_is():
    p = problem()
    result = reader.build_result([REFUSED_LOG], problem=p, jobs=["13"])
    assert result.correct is False
    assert "clock lock" in result.error and "did not hold" in result.error
    assert result.integrity.get("failure_kind") != "no_verdict"


# -- cold start -------------------------------------------------------------------------------


def test_cold_start_is_unmeasured_not_zero():
    p = problem()
    result = reader.build_result([log(correctness(), perf_single())], problem=p,
                                 cold_log="=== gpu_entry.sh ===\n  task: cmd\n",
                                 cold_error="")
    assert result.cold_start_s is None
    assert result.integrity["cold_start"] == "unmeasured"
    assert "COLD_START_S" in result.integrity["cold_start_note"]


def test_cold_start_probe_that_never_ran_says_so():
    result = reader.build_result([log(correctness(), perf_single())], problem=problem(),
                                 cold_log=None,
                                 cold_error="ClockLockRefusedError: refused")
    assert result.cold_start_s is None
    assert result.integrity["cold_start"] == "unmeasured"
    assert "ClockLockRefusedError" in result.integrity["cold_start_note"]


def test_cold_start_over_budget_is_flagged():
    from forgekernel.benchx.api import COLD_START_BUDGET_S

    result = reader.build_result([log(correctness(), perf_single())], problem=problem(),
                                 cold_log=PROBE.format(cold=COLD_START_BUDGET_S + 0.5))
    assert result.cold_start_s == pytest.approx(COLD_START_BUDGET_S + 0.5)
    assert result.integrity["cold_start"] == "over-budget"
    assert "budget" in result.integrity["cold_start_note"]


def test_parse_cold_start_reads_the_first_line_and_survives_junk():
    assert reader.parse_cold_start("[benchx] COLD_START_S 3.25 (card demo_op)") == 3.25
    assert reader.parse_cold_start("[benchx] COLD_START_S 3.25\n"
                                   "[benchx] COLD_START_S 9.0") == 3.25
    assert reader.parse_cold_start("[benchx] COLD_START_S nope") is None
    assert reader.parse_cold_start("") is None
    assert reader.parse_cold_start(None) is None


def test_cold_start_state_is_explicit_about_all_three_cases():
    from forgekernel.benchx.api import COLD_START_BUDGET_S

    assert reader.cold_start_state(12.0, probe_ran=True) == {"cold_start": "ok"}
    assert reader.cold_start_state(COLD_START_BUDGET_S + 1, probe_ran=True)["cold_start"] \
        == "over-budget"
    missing = reader.cold_start_state(None, probe_ran=False, error="")
    assert missing["cold_start"] == "unmeasured"
    assert "did not run" in missing["cold_start_note"]
    assert "no [benchx] COLD_START_S line" in \
        reader.cold_start_state(None, probe_ran=True)["cold_start_note"]


# -- repeats ----------------------------------------------------------------------------------


def test_repeat_takes_the_median_and_records_the_spread():
    p = problem()
    logs = [log(correctness(), perf_single(t_ms=t)) for t in (1.0, 1.2, 1.4)]
    result = reader.build_result(logs, problem=p, jobs=["21", "22", "23"],
                                 cold_log=PROBE.format(cold=10.0))
    assert result.correct
    assert result.job == "21,22,23"
    (w,) = result.workloads
    assert w.t_ms == 1.2                     # the median, not a mean
    assert w.s == pytest.approx(sol_score(1.2, 1.0, 0.2))   # board formula, median input
    rep = result.integrity["repeat"]
    assert rep["n"] == 3
    assert rep["workloads"]["u0"]["min_ms"] == 1.0
    assert rep["workloads"]["u0"]["max_ms"] == 1.4
    assert rep["workloads"]["u0"]["spread_pct"] == pytest.approx(33.333, abs=1e-2)
    assert result.cold_start_s == 10.0       # the first probe's reading, not re-probed


def test_repeat_is_conservative_about_correctness():
    p = problem()
    good = log(correctness(), perf_single())
    bad = log(correctness(False, error="max_diff=9.9 tol=1e-2"), None)
    result = reader.build_result([good, bad], problem=p, jobs=["31", "32"])
    assert result.correct is False
    assert "max_diff=9.9" in result.error


def test_repeat_noted_even_with_one_job():
    result = reader.build_result([log(correctness(), perf_single())], problem=problem())
    assert result.integrity["repeat"] == {"n": 1}


# -- the value types ---------------------------------------------------------------------------


def test_bench_result_round_trips_through_json(tmp_path):
    p = problem()
    result = reader.build_result([log(correctness(), perf_single())], problem=p,
                                 jobs=["1"], cold_log=PROBE.format(cold=7.5),
                                 workload_subset=["wl0"])
    path = result.save(tmp_path / "r.json")
    back = BenchResult.load(path)
    assert back.to_dict() == result.to_dict()
    assert back.workloads == [WorkloadReading.from_dict(w.to_dict())
                              for w in result.workloads]
    assert back.integrity["workload_subset"] == {"requested": ["wl0"],
                                                 "card_workloads": 1}


def test_fake_bench_returns_a_copy_each_call():
    bench = FakeBench(t_ms={"wl0": 0.5}, s={"wl0": 0.7})
    a = bench.bench("card", "cand.py", workloads="all", mode="dev")
    b = bench.bench("card", "cand.py")
    assert a.mean_s == 0.7 and a.workloads[0].t_ms == 0.5
    a.workloads[0].t_ms = 99.0
    assert b.workloads[0].t_ms == 0.5        # the preset is not shared state
    assert bench.calls[0] == {"card_dir": bench.calls[0]["card_dir"],
                              "candidate": "cand.py", "workloads": "all",
                              "mode": "dev"}
