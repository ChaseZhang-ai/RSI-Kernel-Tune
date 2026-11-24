"""forgekernel.opkb.query: which records a problem reads -- same family, self rules, order."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import forgekernel.ext.lessons as lessons_cli
from forgekernel.opkb import query, store

REPO = Path(__file__).resolve().parents[1]


def _rec(rid: str, *, family: str = "rmsnorm", name: str = "069_rms_norm",
         definition: str | None = None, value: float | None = 0.5,
         created: str = "2026-11-24T10:00:00+0800", metric: str = "sol") -> dict:
    outcome: dict = {"window": "device", "stop": "target"}
    if value is not None:
        outcome["metric_kind"] = metric
        outcome["value"] = value
    problem = {"name": name, "suite": "sol-execbench"}
    if definition is not None:
        problem["definition"] = definition
    return {"schema": 1, "id": rid, "created": created, "family": family,
            "problem": problem, "reference": {"kind": "eager"}, "outcome": outcome,
            "source": {"run": "run"}}


def _task_dir(tmp_path: Path, name: str = "sol-069_rms_norm", *, definition: str | None = "069_rms_norm",
              suite: str = "sol-execbench") -> Path:
    task = tmp_path / "tasks" / name
    (task / "leaderboard" / "card" / "eval" / "leaderboard").mkdir(parents=True)
    (task / "task.toml").write_text(
        '[task]\nlanguage = "triton"\narch = "sm100"\n\n[leaderboard]\n'
        f'suite = "{suite}"\ncard = "leaderboard/card"\n', encoding="utf-8")
    problem = {"schema": 1, "suite": suite, "definition": {"name": definition},
               "workload": {"axes": {"B": 4096}}}
    (task / "leaderboard" / "card" / "eval" / "leaderboard" / "problem.json").write_text(
        json.dumps(problem), encoding="utf-8")
    return task


# --- resolving the problem a query is about -----------------------------------------------------


def test_a_task_directory_gives_the_task_name_the_definition_and_the_suite(tmp_path):
    target = query.resolve_target(str(_task_dir(tmp_path)))
    assert target.name == "sol-069_rms_norm"
    assert target.definition == "069_rms_norm"
    assert target.suite == "sol-execbench"
    assert target.family == "rmsnorm"
    assert set(target.keys) == {"sol-069_rms_norm", "069_rms_norm"}


def test_a_task_without_a_card_resolves_by_its_directory_name(tmp_path):
    task = tmp_path / "tasks" / "my_rms_norm"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('[task]\nlanguage = "cuda"\n', encoding="utf-8")
    target = query.resolve_target(str(task))
    assert (target.name, target.definition, target.suite) == ("my_rms_norm", None, None)
    assert target.family == "rmsnorm"


def test_a_corrupt_task_toml_still_resolves_by_name(tmp_path):
    task = tmp_path / "tasks" / "069_rms_norm"
    task.mkdir(parents=True)
    (task / "task.toml").write_text("this is not toml [", encoding="utf-8")
    assert query.resolve_target(str(task)).family == "rmsnorm"


def test_a_bare_problem_name_resolves_to_its_family():
    target = query.resolve_target("069_rms_norm")
    assert target.family == "rmsnorm" and target.keys == ("069_rms_norm",)


@pytest.mark.parametrize("spec", ["", "/no/such/task/dir"])
def test_what_is_not_a_problem_is_refused(spec):
    with pytest.raises(query.QueryError):
        query.resolve_target(spec)


def test_an_existing_directory_that_is_not_a_task_is_refused(tmp_path):
    with pytest.raises(query.QueryError, match="neither a task directory"):
        query.resolve_target(str(tmp_path))


def test_a_run_directory_resolves_through_its_run_json():
    fixture = Path(__file__).parent / "fixtures" / "opkb_run"
    target = query.resolve_target(str(fixture))
    assert (target.name, target.definition, target.suite) == \
        ("069_rms_norm", "069_rms_norm", "sol-execbench")
    assert target.family == "rmsnorm"


def test_a_run_directory_without_a_task_name_is_not_a_problem(tmp_path):
    (tmp_path / "run.json").write_text("{}", encoding="utf-8")
    with pytest.raises(query.QueryError, match="neither a task directory"):
        query.resolve_target(str(tmp_path))


# --- selection -----------------------------------------------------------------------------------


def test_only_the_same_family_is_returned():
    rows = [_rec("a", family="rmsnorm"), _rec("b", family="gemm"),
            _rec("c", family="norm")]
    got = query.select(rows, query.resolve_target("069_rms_norm"))
    assert [r["id"] for r in got] == ["a"]


def test_no_same_family_records_is_an_empty_list_not_an_error():
    rows = [_rec("a", family="gemm")]
    assert query.select(rows, query.resolve_target("069_rms_norm")) == []


def test_the_problems_own_records_are_included_by_default_and_excludable():
    rows = [_rec("own", name="sol-069_rms_norm", definition="069_rms_norm"),
            _rec("sibling", name="084_rms_norm", definition="084_rms_norm")]
    target = query.resolve_target("069_rms_norm")
    assert [r["id"] for r in query.select(rows, target)] == ["own", "sibling"]
    assert target.same_problem(rows[0]) and not target.same_problem(rows[1])
    assert [r["id"] for r in query.select(rows, target, exclude_self=True)] == ["sibling"]


def test_a_record_matches_either_identity_key(tmp_path):
    # the target comes from a task directory: both its task name and its definition match
    target = query.resolve_target(str(_task_dir(tmp_path)))
    by_name = _rec("by-name", name="sol-069_rms_norm")
    by_definition = _rec("by-definition", name="069_rms_norm")
    assert target.same_problem(by_name) and target.same_problem(by_definition)
    assert not target.same_problem(_rec("other", name="084_rms_norm"))


def test_order_is_value_descending_then_created_descending_then_id():
    rows = [_rec("a", value=0.9, created="2026-11-01T00:00:00+0800"),
            _rec("b", value=0.9, created="2026-11-02T00:00:00+0800"),
            _rec("c", value=None, created="2026-11-03T00:00:00+0800"),
            _rec("d", value=0.5, created="2026-11-04T00:00:00+0800"),
            _rec("e", value=0.9, created="2026-11-02T00:00:00+0800")]
    got = [r["id"] for r in query.select(rows, query.resolve_target("069_rms_norm"))]
    assert got == ["b", "e", "a", "d", "c"]


def test_order_drops_a_record_with_no_outcome_block_to_the_end():
    rows = [_rec("with", value=0.1),
            {"schema": 1, "id": "without", "created": "2026-12-01T00:00:00+0800",
             "family": "rmsnorm", "problem": {"name": "084_rms_norm"},
             "reference": {"kind": "none"}, "outcome": {"stop": "budget"},
             "source": {"run": "r"}}]
    assert [r["id"] for r in query.select(rows, query.resolve_target("069_rms_norm"))] == \
        ["with", "without"]


def test_k_limits_the_result_and_zero_returns_nothing():
    rows = [_rec(str(i), value=0.1 * i) for i in range(1, 8)]
    assert len(query.select(rows, query.resolve_target("069_rms_norm"), k=3)) == 3
    assert query.select(rows, query.resolve_target("069_rms_norm"), k=0) == []


def test_describe_names_the_family_problem_metric_and_stop():
    line = query.describe(_rec("x", value=0.71))
    assert line.startswith("x  [rmsnorm]  069_rms_norm")
    assert "sol=0.71" in line and "stop=target" in line


# --- the CLI --------------------------------------------------------------------------------------


def _root_with(tmp_path, *rows) -> Path:
    root = tmp_path / "op_kb"
    for row in rows:
        store.write_record(root, row)
    return root


def test_the_query_command_lists_and_json_prints_full_records(tmp_path, capsys):
    root = _root_with(tmp_path, _rec("a", value=0.9), _rec("b", value=0.5))
    assert lessons_cli.main(["query", "069_rms_norm", "--root", str(root)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in out] == ["a", "b"]
    assert lessons_cli.main(["query", "069_rms_norm", "--root", str(root), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [r["id"] for r in data] == ["a", "b"] and data[0]["schema"] == 1


def test_the_query_command_says_so_when_the_family_has_no_records(tmp_path, capsys):
    root = _root_with(tmp_path, _rec("a", family="gemm"))
    assert lessons_cli.main(["query", "069_rms_norm", "--root", str(root)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert 'no records of the "rmsnorm" family' in captured.err + captured.out


def test_the_query_command_accepts_a_run_directory(tmp_path, capsys):
    from forgekernel.opkb import record

    fixture = Path(__file__).parent / "fixtures" / "opkb_run"
    root = tmp_path / "op_kb"
    store.write_record(root, record.extract_run(fixture).record)
    assert lessons_cli.main(["query", str(fixture), "--root", str(root)]) == 0
    assert "069_rms_norm" in capsys.readouterr().out


def test_the_query_command_rejects_a_non_problem(tmp_path, capsys):
    assert lessons_cli.main(["query", str(tmp_path)]) == 2
    assert "neither a task directory" in capsys.readouterr().err
