"""`shapes` matrix and routing: the readings table and who wins each workload.

Everything here is pure: readings are handed in as bench results, so the three cases the
task file calls out (a candidate incorrect on a workload, a tie inside the noise floor, the
eager reference being fastest) are exact numbers, not GPU noise.
"""

from __future__ import annotations

import json

import pytest

from rsi_kernel_tune.benchx.api import BenchResult, WorkloadReading
from rsi_kernel_tune.shapes import cluster as C
from rsi_kernel_tune.shapes import evalx, matrix as M


def _workloads() -> list[C.Workload]:
    return [C.Workload(index=i, name=f"wl{i}", axes={"M": m}, t_b=tb, t_sol=ts, elements=m)
            for i, (m, tb, ts) in enumerate([(8, 0.032, 0.004), (12, 0.048, 0.006),
                                             (1024, 4.8, 0.6), (16384, 72.0, 9.0)])]


def _result(times: dict, *, correct: dict | None = None, overall: bool = True,
            mean_s: float | None = None) -> BenchResult:
    correct = correct or {}
    rows = []
    for name, t in times.items():
        ok = correct.get(name, t is not None)
        rows.append(WorkloadReading(name=name, t_ms=t if ok else None, correct=ok,
                                    t_b=None, t_sol=None,
                                    s=(1 / (1 + t)) if (ok and t is not None) else None,
                                    error="" if ok else "the card said no"))
    return BenchResult(correct=overall, workloads=rows, mean_s=mean_s, fast1=0,
                       cold_start_s=2.0, confidence=["dev-only"])


def _cand(label: str, kind: str, source: str = "def run(*args):\n    return None\n"):
    return evalx.Candidate(label=label, kind=kind, source=source, path=None)


def _matrix(rows: list[tuple[evalx.Candidate, BenchResult]]) -> M.Matrix:
    return M.from_readings(rows, workloads=_workloads(), card="card",
                           card_sha="deadbeef", now="2026-01-01T00:00:00+0000")


def _eager(times: dict, **kw) -> tuple[evalx.Candidate, BenchResult]:
    return (_cand("eager", evalx.KIND_EAGER), _result(times, **kw))


def test_matrix_cells_and_absent_workloads():
    rows = [(_cand("c0", evalx.KIND_CLUSTER), _result({"wl0": 1.0, "wl1": 2.0})),
            _eager({n: 3.0 for n in ("wl0", "wl1", "wl2", "wl3")})]
    mx = _matrix(rows)
    assert mx.labels() == ["c0", "eager"]
    c0 = mx.row("c0")
    assert c0.cell("wl0").t_ms == 1.0 and c0.cell("wl0").correct is True
    assert c0.cell("wl2").correct is False          # absent == not correct, not a zero
    assert "does not mention" in c0.cell("wl2").error
    assert mx.row("eager").cell("wl3").t_ms == 3.0
    doc = mx.as_json()
    assert doc["workloads"][0]["axes"] == {"M": 8}
    assert doc["rows"][0]["cells"]["wl0"]["t_ms"] == 1.0
    # the json round-trips (matrix.json is a cache as well as a report)
    again = M.Matrix.load(mx.save(tmp_json(mx)))
    assert again.labels() == mx.labels()
    assert again.row("c0").cell("wl1").t_ms == 2.0


def tmp_json(mx: M.Matrix):
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp()) / "matrix.json"
    return mx.save(tmp)


def test_route_picks_the_fastest_correct_candidate():
    rows = [(_cand("c0", evalx.KIND_CLUSTER), _result({"wl0": 1.0, "wl1": 1.0,
                                                       "wl2": 1.0, "wl3": 1.0})),
            _eager({n: 2.0 for n in ("wl0", "wl1", "wl2", "wl3")})]
    routes = M.route(_matrix(rows), eager="eager")
    assert [r.winner for r in routes.routes] == ["c0"] * 4
    assert routes.fallback == "eager"
    assert routes.guarantee["checked"] is True
    assert routes.guarantee["violations"] == []
    assert all(r.vs_eager == pytest.approx(0.5) for r in routes.routes)


def test_incorrect_cells_are_excluded():
    rows = [(_cand("c0", evalx.KIND_CLUSTER),
             _result({"wl0": 1.0, "wl1": 1.0, "wl2": 0.5, "wl3": 0.5},
                     correct={"wl2": False, "wl3": False})),
            _eager({"wl0": 2.0, "wl1": 2.0, "wl2": 2.0, "wl3": 2.0})]
    routes = M.route(_matrix(rows), eager="eager")
    winners = [r.winner for r in routes.routes]
    # wl2/wl3: c0's reading is a wrong answer, so it never competes -- the reference runs
    assert winners == ["c0", "c0", "eager", "eager"]
    assert "no correct candidate" in routes.routes[2].reason


def test_a_tie_inside_the_noise_floor_goes_to_the_simpler_candidate():
    # 1.0 vs 1.01 is a 1% difference: inside the 1.5% floor, so the *cluster specialist*
    # wins even though it is listed second and reads 0.01 ms slower.
    rows = [(_cand("extra", evalx.KIND_EXTRA), _result({n: 1.0 for n in
                                                       ("wl0", "wl1", "wl2", "wl3")})),
            (_cand("c0", evalx.KIND_CLUSTER), _result({n: 1.01 for n in
                                                      ("wl0", "wl1", "wl2", "wl3")})),
            _eager({n: 4.0 for n in ("wl0", "wl1", "wl2", "wl3")})]
    routes = M.route(_matrix(rows), eager="eager")
    assert [r.winner for r in routes.routes] == ["c0"] * 4
    assert routes.routes[0].within_noise == ["c0", "extra"]
    # outside the floor the faster candidate wins whatever its kind
    faster_extra = M.route(_matrix([(_cand("extra", evalx.KIND_EXTRA),
                                     _result({n: 1.0 for n in
                                              ("wl0", "wl1", "wl2", "wl3")})),
                                    (_cand("c0", evalx.KIND_CLUSTER),
                                     _result({n: 1.2 for n in
                                              ("wl0", "wl1", "wl2", "wl3")})),
                                    _eager({n: 4.0 for n in
                                            ("wl0", "wl1", "wl2", "wl3")})]),
                            eager="eager")
    assert [r.winner for r in faster_extra.routes] == ["extra"] * 4


def test_the_eager_reference_is_the_floor_unless_allow_slower():
    # c0 is correct everywhere but slower than eager on wl2 and wl3
    rows = [(_cand("c0", evalx.KIND_CLUSTER),
             _result({"wl0": 1.0, "wl1": 1.0, "wl2": 3.0, "wl3": 3.0})),
            _eager({"wl0": 2.0, "wl1": 2.0, "wl2": 2.0, "wl3": 2.0})]
    routes = M.route(_matrix(rows), eager="eager")
    assert [r.winner for r in routes.routes] == ["c0", "c0", "eager", "eager"]
    assert routes.routes[2].slower_than_reference is False
    assert "+50.0% slower than the eager reference" in routes.routes[2].reason
    for route in routes.routes:
        assert route.t_ms <= route.eager_t_ms * 1.015         # the guarantee holds

    allowed = M.route(_matrix(rows), eager="eager", allow_slower=True)
    assert [r.winner for r in allowed.routes] == ["c0"] * 4
    assert allowed.routes[2].slower_than_reference is True
    assert allowed.guarantee["violations"] == ["wl2", "wl3"]
    assert allowed.guarantee["allow_slower"] is True


def test_a_workload_with_no_correct_candidate_routes_to_the_fallback():
    rows = [(_cand("c0", evalx.KIND_CLUSTER),
             _result({"wl0": 1.0, "wl1": 1.0, "wl2": 1.0, "wl3": 1.0},
                     correct={n: False for n in ("wl0", "wl1", "wl2", "wl3")})),
            (_cand("overall", evalx.KIND_OVERALL),
             _result({"wl0": 5.0, "wl1": 5.0, "wl2": 5.0, "wl3": 5.0})),
            _eager({n: 9.0 for n in ("wl0", "wl1", "wl2", "wl3")})]
    routes = M.route(_matrix(rows), eager="eager", fallback="overall")
    assert [r.winner for r in routes.routes] == ["overall"] * 4
    assert routes.fallback == "overall" and routes.fallback_kind == "overall"
    # the fallback itself is incorrect here: the reference takes over
    rows_bad = [(_cand("c0", evalx.KIND_CLUSTER),
                 _result({"wl0": 1.0, "wl1": 1.0, "wl2": 1.0, "wl3": 1.0},
                         correct={n: False for n in ("wl0", "wl1", "wl2", "wl3")})),
                (_cand("overall", evalx.KIND_OVERALL),
                 _result({"wl0": 5.0}, correct={"wl0": False, "wl1": False, "wl2": False,
                                                "wl3": False})),
                _eager({n: 9.0 for n in ("wl0", "wl1", "wl2", "wl3")})]
    taken = M.route(_matrix(rows_bad), eager="eager", fallback="overall")
    assert [r.winner for r in taken.routes] == ["eager"] * 4
    assert "routed to the eager reference" in taken.routes[0].reason


def test_mean_s_scores_the_routes_and_the_reference():
    rows = [(_cand("c0", evalx.KIND_CLUSTER),
             _result({"wl0": 0.5, "wl1": 0.5, "wl2": 0.5, "wl3": 0.5})),
            _eager({n: 1.0 for n in ("wl0", "wl1", "wl2", "wl3")})]
    routes = M.route(_matrix(rows), eager="eager")
    assert routes.mean_s == pytest.approx(1 / 1.5)         # 0.5 ms readings
    assert routes.mean_s_eager == pytest.approx(0.5)       # 1.0 ms readings
    assert routes.mean_s > routes.mean_s_eager
    doc = routes.as_json()
    assert doc["mean_s"] == pytest.approx(1 / 1.5)
    assert doc["routes"][0]["winner"] == "c0"
    assert doc["routes"][0]["s"] == pytest.approx(1 / 1.5)


def test_without_an_eager_reading_the_guarantee_is_unchecked_not_claimed():
    rows = [(_cand("c0", evalx.KIND_CLUSTER), _result({n: 1.0 for n in
                                                       ("wl0", "wl1", "wl2", "wl3")})),
            _eager({n: 1.0 for n in ("wl0", "wl1", "wl2", "wl3")},
                   overall=False, correct={n: False for n in
                                           ("wl0", "wl1", "wl2", "wl3")})]
    routes = M.route(_matrix(rows), eager="eager")
    assert routes.guarantee["checked"] is False
    assert any("not correct on every workload" in n for n in routes.notes)
    assert [r.winner for r in routes.routes] == ["c0"] * 4


def test_reading_cache_is_keyed_by_the_candidate_and_the_card(tmp_path):
    path = tmp_path / "matrix.json"
    cache = evalx.ReadingCache(path, card_sha="card-a")
    cand = evalx.Candidate(label="c0", kind=evalx.KIND_CLUSTER,
                           source="def run(*args):\n    pass\n")
    cache.put(cand, _result({"wl0": 1.0}))
    cache.save()
    again = evalx.ReadingCache.load(path, card_sha="card-a")
    assert again.get(cand)["workloads"][0]["t_ms"] == 1.0
    other = evalx.ReadingCache.load(path, card_sha="card-b")
    assert other.get(cand) is None                  # a different card: nothing comparable
    changed = evalx.Candidate(label="c0", kind=evalx.KIND_CLUSTER,
                              source="def run(*args):\n    return 1\n")
    assert again.get(changed) is None                # a changed source re-benches


def test_bench_all_benches_every_candidate_on_the_whole_card_and_caches(tmp_path):
    from rsi_kernel_tune.benchx.api import FakeBench

    cands = [evalx.Candidate(label="c0", kind=evalx.KIND_CLUSTER,
                             source="def run(*args):\n    pass\n",
                             path=tmp_path / "c0.py"),
             evalx.Candidate(label="eager", kind=evalx.KIND_EAGER,
                             source="def run(*args):\n    return None\n",
                             path=tmp_path / "eager.py")]
    for cand in cands:
        cand.path.write_text(cand.source, encoding="utf-8")
    bench = FakeBench(t_ms={"wl0": 1.0, "wl1": 2.0})
    cache = evalx.ReadingCache(tmp_path / "matrix.json", card_sha="c")
    rows = evalx.bench_all(cands, tmp_path, bench, cache=cache)
    assert [c.label for c, _ in rows] == ["c0", "eager"]
    assert len(bench.calls) == 2 and all(call.workloads is None for call in bench.calls)
    assert (tmp_path / "matrix.json").is_file()
    # a second pass reuses both readings: no new bench call
    rows2 = evalx.bench_all(cands, tmp_path, bench, cache=evalx.ReadingCache.load(
        tmp_path / "matrix.json", card_sha="c"))
    assert len(bench.calls) == 2 and len(rows2) == 2
    # a raising bench is one failed row, not a crashed matrix
    def boom(call):
        raise RuntimeError("the job died")
    rows3 = evalx.bench_all(cands[:1], tmp_path, FakeBench(script=boom),
                            cache=None)
    assert rows3[0][1]["correct"] is False
    assert "the job died" in rows3[0][1]["error"]


def test_format_helpers_show_the_numbers():
    rows = [(_cand("c0", evalx.KIND_CLUSTER), _result({"wl0": 1.0, "wl1": 2.0})),
            _eager({n: 3.0 for n in ("wl0", "wl1", "wl2", "wl3")})]
    mx = _matrix(rows)
    text = M.format_matrix(mx)
    assert "c0" in text and "1.0000" in text and "! = the card judged" in text
    routes = M.route(mx, eager="eager")
    shown = M.format_routes(routes)
    assert "wl0" in shown and "mean S" in shown and "guarantee checked" in shown
    assert json.dumps(routes.as_json())[:1] == "{"
