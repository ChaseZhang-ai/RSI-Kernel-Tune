"""compare / best-of: pure arithmetic over bench results, hand-checkable cases."""

import json

import pytest

from rsi_kernel_tune.benchx import compare as compare_mod
from rsi_kernel_tune.benchx.api import BenchResult, WorkloadReading


def result(readings, *, mean_s=None, correct=True, error="") -> BenchResult:
    """``readings``: [(name, t_ms, t_b, t_sol, s)] or [(name, t_ms, correct)]."""
    out = []
    for row in readings:
        if len(row) == 3:
            name, t_ms, ok = row
            out.append(WorkloadReading(name=name, t_ms=t_ms, correct=ok))
        else:
            name, t_ms, t_b, t_sol, s = row
            out.append(WorkloadReading(name=name, t_ms=t_ms, t_b=t_b, t_sol=t_sol,
                                       s=s, correct=True))
    fast1 = sum(1 for w in out if w.t_ms is not None and w.t_b is not None
                and w.t_ms < w.t_b)
    return BenchResult(correct=correct, workloads=out, mean_s=mean_s, fast1=fast1,
                       error=error)


def test_compare_is_a_per_workload_table_with_deltas_and_a_summary():
    a = result([("wl0", 1.0, 2.0, 0.5, 0.6667), ("wl1", 2.0, 4.0, 1.0, 0.6667)],
               mean_s=0.6667)
    b = result([("wl0", 1.2, 2.0, 0.5, 0.5333), ("wl1", 1.5, 4.0, 1.0, 0.8333)],
               mean_s=0.6833)
    table = compare_mod.compare({"a.json": a, "b.json": b})
    assert [r["workload"] for r in table["workloads"]] == ["wl0", "wl1"]
    row0, row1 = table["workloads"]
    assert row0["fastest"] == "a.json" and row0["best_t_ms"] == 1.0
    assert row1["fastest"] == "b.json"
    by_label = {e["candidate"]: e for e in row0["entries"]}
    assert by_label["a.json"]["delta_pct"] == 0.0
    assert by_label["b.json"]["delta_pct"] == pytest.approx(20.0)
    assert table["summary"]["a.json"] == {"correct": True, "mean_s": 0.6667,
                                          "fast1": 2, "fast1_ratio": 1.0,
                                          "geomean_ms": pytest.approx(
                                              (1.0 * 2.0) ** 0.5),
                                          "n_timed": 2}
    assert table["summary"]["b.json"]["fast1_ratio"] == 1.0


def test_compare_says_which_workloads_a_candidate_lacks():
    a = result([("wl0", 1.0, 2.0, 0.5, 0.6667)])
    b = result([("wl1", 1.5, 4.0, 1.0, 0.8333)])
    table = compare_mod.compare({"a.json": a, "b.json": b})
    missing = {r["workload"]: r for r in table["workloads"]}
    assert missing["wl0"]["entries"][1]["t_ms"] is None
    assert missing["wl0"]["entries"][1]["delta_pct"] is None
    assert table["summary"]["b.json"]["n_timed"] == 1


def test_best_of_routes_each_workload_to_its_fastest_correct_candidate():
    a = result([("wl0", 1.0, 2.0, 0.5, 0.6667), ("wl1", 2.0, 4.0, 1.0, 0.6667)],
               mean_s=0.6667)
    b = result([("wl0", 1.2, 2.0, 0.5, 0.5333), ("wl1", 1.5, 4.0, 1.0, 0.8333)],
               mean_s=0.6833)
    picked = compare_mod.best_of({"a.json": a, "b.json": b})
    assert [(r["workload"], r["winner"]) for r in picked["routes"]] == [
        ("wl0", "a.json"), ("wl1", "b.json")]
    assert picked["routes"][0]["t_ms"] == 1.0
    assert picked["routes"][1]["s"] == 0.8333
    assert picked["fallback"] == "b.json"          # the highest mean S
    assert picked["fallback_mean_s"] == 0.6833


def test_best_of_gives_a_tie_to_the_earlier_file():
    a = result([("wl0", 1.0, 2.0, 0.5, 0.6)], mean_s=0.60)     # simpler, first
    b = result([("wl0", 1.005, 2.0, 0.5, 0.6)], mean_s=0.60)   # 0.5% faster: a tie
    picked = compare_mod.best_of({"a.json": a, "b.json": b})
    (route,) = picked["routes"]
    assert route["winner"] == "a.json"
    assert route["within_noise"] == ["a.json", "b.json"]

    c = result([("wl0", 0.97, 2.0, 0.5, 0.6)], mean_s=0.60)    # 3% faster: not a tie
    picked = compare_mod.best_of({"a.json": a, "c.json": c})
    assert picked["routes"][0]["winner"] == "c.json"
    assert picked["routes"][0]["within_noise"] == ["c.json"]


def test_best_of_never_routes_to_an_incorrect_candidate():
    a = result([("wl0", 1.0, 2.0, 0.5, 0.6667)], mean_s=0.6667)
    bad = result([("wl0", 0.1, False)], mean_s=None, correct=False, error="boom")
    picked = compare_mod.best_of({"bad.json": bad, "a.json": a})
    assert picked["routes"][0]["winner"] == "a.json"
    assert picked["fallback"] == "a.json"


def test_a_workload_no_candidate_got_right_has_no_winner():
    bad = result([("wl0", None, False)], correct=False, error="boom")
    picked = compare_mod.best_of({"bad.json": bad})
    (route,) = picked["routes"]
    assert route["winner"] is None and route["t_ms"] is None
    assert picked["fallback"] is None and picked["fallback_mean_s"] is None


def test_geometry_is_a_geometric_mean_of_the_positive_readings():
    assert compare_mod.geometry([1.0, 4.0]) == pytest.approx(2.0)
    assert compare_mod.geometry([0.0025, 0.01]) == pytest.approx(0.005)
    assert compare_mod.geometry([]) is None


# -- the CLI ----------------------------------------------------------------------------------


def two_result_files(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    result([("wl0", 1.0, 2.0, 0.5, 0.6667), ("wl1", 2.0, 4.0, 1.0, 0.6667)],
           mean_s=0.6667).save(a)
    result([("wl0", 1.2, 2.0, 0.5, 0.5333), ("wl1", 1.5, 4.0, 1.0, 0.8333)],
           mean_s=0.6833).save(b)
    return a, b


def test_the_compare_command_prints_and_writes_the_table(tmp_path, capsys):
    from rsi_kernel_tune.ext import bench as bench_cmd

    a, b = two_result_files(tmp_path)
    out = tmp_path / "table.json"
    assert bench_cmd.main(["compare", str(a), str(b), "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "wl0" in printed and "a.json" in printed
    table = json.loads(out.read_text())
    assert table["workloads"][0]["fastest"] == "a.json"


def test_the_best_of_command_writes_the_routing_table(tmp_path, capsys):
    from rsi_kernel_tune.ext import bench as bench_cmd

    a, b = two_result_files(tmp_path)
    out = tmp_path / "routes.json"
    assert bench_cmd.main(["best-of", str(a), str(b), "--json", str(out)]) == 0
    picked = json.loads(out.read_text())
    assert [r["workload"] for r in picked["routes"]] == ["wl0", "wl1"]
    assert picked["fallback"] == "b.json"
    assert "fallback" in capsys.readouterr().out


def test_compare_needs_two_readable_results(tmp_path, capsys):
    from rsi_kernel_tune.ext import bench as bench_cmd

    a, _ = two_result_files(tmp_path)
    assert bench_cmd.main(["compare", str(a)]) == 2
    assert "at least two" in capsys.readouterr().err
    assert bench_cmd.main(["compare", str(a), str(tmp_path / "nope.json")]) == 2
    assert "not a bench result file" in capsys.readouterr().err
    junk = tmp_path / "junk.json"
    junk.write_text("{\"hello\": 1}")
    assert bench_cmd.main(["compare", str(a), str(junk)]) == 2
    assert "does not read as a BenchResult" in capsys.readouterr().err
