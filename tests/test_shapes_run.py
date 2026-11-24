"""`shapes plan` and `shapes run`: the cluster tasks, their argv, and the resumable run.

Nothing here starts a real GPU job: every `forge` child command is a fake that records its
argv and leaves the files the next step would find. The plan tests are pure (a card in,
a plan document out); the run tests cover the normal path, a failing step, resuming, and
``--dry-run``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import tomllib

from forgekernel.shapes import ShapesError
from forgekernel.shapes import cluster as C
from forgekernel.shapes import plan as P
from forgekernel.shapes import run as R
from forgekernel.ext import shapes as cli

FIXTURE = Path(__file__).parent / "fixtures" / "shapes_card"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _task(tmp_path: Path, *, name: str = "fixture_shapes_gemm", rows=None) -> Path:
    task = tmp_path / "tasks" / name
    shutil.copytree(FIXTURE, task, dirs_exist_ok=True)
    if rows is not None:
        path = task / "eval/leaderboard/problem.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["group"] = doc["group"][:rows]
        doc["workload"] = doc["group"][0]["workload"]
        path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return task


# --- plan ---------------------------------------------------------------------------------

def test_plan_partitions_the_card_and_names_the_tasks(tmp_path):
    task = _task(tmp_path)
    plan = P.build_plan(task)
    doc = plan.as_json()
    assert doc["op"] == "fixture_shapes_gemm"
    assert [c["name"] for c in doc["clusters"]] == [f"fixture_shapes_gemm-c{i}"
                                                    for i in range(4)]
    seen: list[int] = []
    for cluster in doc["clusters"]:
        assert _NAME_RE.match(cluster["name"])
        assert cluster["name"] != doc["op"]            # never the original task's name
        assert cluster["target_ms"] and cluster["target_ms"] > 0
        argv = cluster["forge_init_argv"]
        assert argv[-2] == "--workload"
        picked = [int(x) for x in argv[-1].split(",")]
        assert picked == cluster["workloads"]
        seen += picked
    assert sorted(seen) == list(range(8)) and len(seen) == len(set(seen))
    assert doc["params"] == {"max_clusters": 4, "split_ratio": 2.0, "min_size": 2}
    assert str(task) in doc["task"]
    assert (Path(doc["card"]) / "eval/leaderboard/problem.json").is_file()
    assert doc["sol_goal"] == 0.90


def test_plan_derives_the_init_arguments_from_the_card(tmp_path):
    doc = P.build_plan(_task(tmp_path)).as_json()
    base = doc["init_argv_base"]
    assert base[:2] == ["--leaderboard", "sol-execbench"]
    assert "--dataset" in base and "tests/fixtures/shapes_dataset" in base
    assert "--definition" in base and "L1/fixture_shapes_gemm" in base
    assert "--no-sol-board" in base
    assert any("derived from the card's provenance" in w for w in doc["warnings"])
    assert any("--no-sol-board" in w for w in doc["warnings"])


def test_init_args_override_and_replace_the_workload_selector(tmp_path):
    task = _task(tmp_path)
    doc = P.build_plan(
        task, init_args="--leaderboard sol-execbench --dataset /data --definition L1/x "
                        "--sol-board board.json --workload 0-7 --init-kernel seed.py"
    ).as_json()
    assert doc["init_argv_base"] == ["--leaderboard", "sol-execbench", "--dataset", "/data",
                                     "--definition", "L1/x", "--sol-board", "board.json",
                                     "--init-kernel", "seed.py"]
    for cluster in doc["clusters"]:
        assert cluster["forge_init_argv"].count("--workload") == 1
        assert cluster["forge_init_argv"][-1] == ",".join(str(i)
                                                          for i in cluster["workloads"])
    # a leading op name is tolerated, not passed on as an argument
    doc2 = P.build_plan(task, init_args="tasks/fixture_shapes_gemm --leaderboard "
                                        "sol-execbench --dataset /d --definition L1/x")
    assert doc2.as_json()["init_argv_base"][0] == "--leaderboard"


def test_plan_is_deterministic(tmp_path):
    task = _task(tmp_path)
    one = json.dumps(P.build_plan(task, now="2026-01-01T00:00:00+0000").as_json(),
                     sort_keys=True)
    two = json.dumps(P.build_plan(task, now="2026-01-01T00:00:00+0000").as_json(),
                     sort_keys=True)
    assert one == two


def test_a_one_workload_card_says_there_is_nothing_to_cluster(tmp_path):
    with pytest.raises(P.NothingToCluster, match="nothing to cluster"):
        P.build_plan(_task(tmp_path, rows=1))


def test_a_directory_that_is_not_a_board_card_is_a_usage_error(tmp_path):
    plain = tmp_path / "tasks" / "plain"
    plain.mkdir(parents=True)
    (plain / "task.toml").write_text("[task]\nlanguage = 'cutedsl'\narch = 'sm100'\n",
                                     encoding="utf-8")
    with pytest.raises(ShapesError, match="not a leaderboard task"):
        P.build_plan(plain)
    with pytest.raises(ShapesError, match="no such task"):
        P.build_plan(tmp_path / "missing")


def test_plan_json_round_trip_and_schema(tmp_path):
    task = _task(tmp_path)
    plan = P.build_plan(task)
    path = P.write_plan_doc(plan.as_json(), Path(plan.as_json()["product_dir"]))
    assert P.load_plan(path)["clusters"] == plan.as_json()["clusters"]
    assert P.load_plan(task)["op"] == "fixture_shapes_gemm"      # a task dir resolves too
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        P.load_plan(bad)
    with pytest.raises(FileNotFoundError):
        P.load_plan(tmp_path / "none" / "plan.json")


def test_plan_for_uses_the_saved_plan_else_builds_one(tmp_path):
    task = _task(tmp_path)
    doc, used = P.plan_for(task)
    assert used is None and doc["op"] == "fixture_shapes_gemm"
    written = P.write_plan_doc(doc, Path(doc["product_dir"]))
    doc2, used2 = P.plan_for(task)
    assert used2 == written and doc2["created"] == doc["created"]


def test_strip_workload_flags():
    assert P.strip_workload_flags(["--a", "--workload", "1,2", "--b"]) == ["--a", "--b"]
    assert P.strip_workload_flags(["--workload=3", "--check-workload", "4"]) == []
    assert P.strip_workload_flags(["--workload"]) == []


def test_ensure_excluded_writes_git_info_exclude_only(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    result = P.ensure_excluded(repo, ["tasks/a-c0", "tasks/a/shapes"])
    assert result["error"] == ""
    assert result["added"] == ["/tasks/a-c0/", "/tasks/a/shapes/"]
    text = Path(result["path"]).read_text(encoding="utf-8")
    assert "/tasks/a-c0/" in text
    assert not (repo / ".gitignore").exists()
    again = P.ensure_excluded(repo, ["tasks/a-c0"])
    assert again["added"] == []                      # idempotent
    assert Path(result["path"]).read_text(encoding="utf-8") == text
    missing = P.ensure_excluded(tmp_path / "not-a-repo", ["tasks/a"])
    assert missing["error"] and missing["added"] == []


def test_excluded_rels_are_repository_relative(tmp_path):
    doc = {"task": str(tmp_path / "tasks" / "op"),
           "product_dir": str(tmp_path / "tasks" / "op" / "shapes"),
           "clusters": [{"name": "op-c0"}, {"name": "op-c1"}]}
    rels = P.excluded_rels(doc)
    assert "tasks/op-c0" in rels and "tasks/op-c1" in rels
    assert not any(r.startswith("/") or r.startswith("tmp") for r in rels)


def test_the_plan_cli_writes_the_plan_and_prints_it(tmp_path, monkeypatch, capsys):
    task = _task(tmp_path)
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(P, "ensure_excluded",
                        lambda root, rels: calls.append((str(root), rels))
                        or {"path": "", "added": [], "error": ""})
    assert cli.main(["plan", str(task)]) == 0
    out = capsys.readouterr().out
    assert "fixture_shapes_gemm-c0" in out and "forge init" in out
    assert (task / "shapes" / "plan.json").is_file()
    assert calls and "tasks/fixture_shapes_gemm-c0" in calls[0][1]
    # --dry-run writes nothing at all
    shutil.rmtree(task / "shapes")
    assert cli.main(["plan", str(task), "--dry-run"]) == 0
    assert not (task / "shapes").exists()
    assert "dry run" in capsys.readouterr().out


def test_the_plan_cli_exits_zero_for_one_workload_and_two_for_a_bad_path(tmp_path, capsys):
    assert cli.main(["plan", str(_task(tmp_path, rows=1))]) == 0
    assert "nothing to cluster" in capsys.readouterr().out
    assert cli.main(["plan", str(tmp_path / "nope")]) == 2
    assert "no such task" in capsys.readouterr().err
    assert cli.main(["plan", str(_task(tmp_path, name="op2")), "--split-ratio", "1"]) == 2


# --- run ----------------------------------------------------------------------------------

class FakeForge:
    """The `forge` CLI as a fake: what a real call would create, plus the argv record."""

    def __init__(self, root: Path, *, fail: tuple[str, ...] = (), solution: bool = True):
        self.root = root
        self.calls: list[list[str]] = []
        self.fail = set(fail)
        self.solution = solution
        self.count = 0

    def __call__(self, argv, *, cwd=None, env=None):
        self.calls.append(list(argv))
        command = argv[3]
        if command in self.fail:
            return R.Proc(rc=1, output=f"{command} failed on purpose\n")
        if command == "init":
            name = argv[4]
            task = self.root / "tasks" / name
            task.mkdir(parents=True, exist_ok=True)
            (task / "task.toml").write_text(
                f"# {name}\n\n[task]\nlanguage = \"cutedsl\"\narch = \"sm100\"\n",
                encoding="utf-8")
            return R.Proc(rc=0, output=f"wrote {task}\n")
        if command == "new":
            name = argv[4]
            self.count += 1
            run = self.root / ".artifacts" / f"{name}_260101-00000{self.count}-abcd"
            run.mkdir(parents=True, exist_ok=True)
            (run / "run.json").write_text(json.dumps({"task": name}), encoding="utf-8")
            return R.Proc(rc=0, output=f"run directory {run.relative_to(self.root)}\n")
        if command == "run":
            return R.Proc(rc=0, output="the loop finished\n")
        if command == "export":
            out = Path(argv[argv.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            if self.solution:
                out.write_text(json.dumps({
                    "name": "x", "definition": "fixture_shapes_gemm", "author": "t",
                    "spec": {"languages": ["pytorch"], "target_hardware": ["B200"],
                             "entry_point": "kernel.py::run"},
                    "sources": [{"path": "kernel.py",
                                 "content": "def run(*args):\n    return None\n"}]}),
                    encoding="utf-8")
            return R.Proc(rc=0, output=f"wrote {out}\n")
        return R.Proc(rc=2, output=f"unknown command {command}\n")


def _run(tmp_path: Path, **kw):
    task = _task(tmp_path)
    doc = P.build_plan(task).as_json()
    spawn = FakeForge(tmp_path, **{k: v for k, v in kw.items()
                                   if k in ("fail", "solution")})
    rc, runs = R.run_plan(doc, root=tmp_path, spawn=spawn,
                          log=kw.pop("log", None),
                          max_hours=kw.get("max_hours"), cluster_hours=kw.get("cluster_hours"),
                          dry_run=kw.get("dry_run", False),
                          clock=kw.get("clock", __import__("time").monotonic))
    return task, doc, spawn, rc, runs


def test_run_drives_every_cluster_and_records_it(tmp_path):
    task, doc, spawn, rc, runs = _run(tmp_path)
    assert rc == R.EXIT_OK
    names = [c["name"] for c in doc["clusters"]]
    assert all(runs.state(n).status == R.STATUS_SOLVED for n in names)
    assert len(spawn.calls) == 4 * len(names)          # init, new, run, export per cluster
    for call in spawn.calls:
        assert call[:3] == [__import__("sys").executable, "-m", "forgekernel"]
    commands = [call[3] for call in spawn.calls]
    assert commands.count("init") == len(names)
    assert commands.count("export") == len(names)
    ledger = json.loads((task / "shapes" / "runs.json").read_text(encoding="utf-8"))
    assert set(ledger["clusters"]) == set(names)
    first = ledger["clusters"][names[0]]
    assert first["status"] == "solved"
    assert first["steps"] == {"init": 0, "new": 0, "run": 0, "export": 0}
    assert Path(first["run"]).is_dir() and Path(first["solution"]).is_file()
    # the plan's args are the ones handed to `forge init`
    init = next(call for call in spawn.calls if call[3] == "init")
    assert init[4] == names[0]
    assert init[-2] == "--workload" and init[-1] == ",".join(
        str(i) for i in doc["clusters"][0]["workloads"])


def test_run_writes_the_cluster_target_and_budget(tmp_path):
    task, doc, _spawn, rc, runs = _run(tmp_path, cluster_hours=1.5)
    assert rc == R.EXIT_OK
    name = doc["clusters"][0]["name"]
    text = (task.parent.parent / "tasks" / name / "task.toml").read_text(encoding="utf-8")
    cfg = tomllib.loads(text)
    assert cfg["target"]["ms"] == pytest.approx(doc["clusters"][0]["target_ms"], rel=1e-6)
    assert cfg["budget"]["hours"] == 1.5
    assert runs.state(name).target_ms == pytest.approx(doc["clusters"][0]["target_ms"])


def test_a_failing_step_marks_the_cluster_failed_and_the_run_resumable(tmp_path):
    task, doc, spawn, rc, runs = _run(tmp_path, fail=("run",))
    assert rc == R.EXIT_UNFINISHED
    name = doc["clusters"][0]["name"]
    state = runs.state(name)
    assert state.status == R.STATUS_FAILED
    assert "run exited 1" in state.error
    assert "failed on purpose" in state.error
    ledger = json.loads((task / "shapes" / "runs.json").read_text(encoding="utf-8"))
    assert ledger["clusters"][name]["status"] == "failed"
    # a failing export is the same kind of failure
    _task2 = tmp_path / "second"
    _task2.mkdir()
    task2, doc2, spawn2, rc2, runs2 = _run(_task2, solution=False)
    assert rc2 == R.EXIT_UNFINISHED
    assert "export exited 0" in runs2.state(doc2["clusters"][0]["name"]).error


def test_run_resumes_and_skips_what_is_done(tmp_path):
    task, doc, spawn, rc, runs = _run(tmp_path)
    names = [c["name"] for c in doc["clusters"]]
    first = runs.state(names[0])
    # pretend the second cluster never started and the first is done
    ledger_path = task / "shapes" / "runs.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["clusters"][names[0]]["status"] = "solved"
    ledger["clusters"][names[1]]["status"] = "planned"
    ledger["clusters"][names[1]]["run"] = ""
    ledger["clusters"][names[1]]["solution"] = ""
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    spawn.calls.clear()
    rc2, runs2 = R.run_plan(doc, root=tmp_path, spawn=spawn)
    assert rc2 == R.EXIT_OK
    touched = {call[4].split("/")[-1] for call in spawn.calls if call[3] in ("init", "new")}
    assert names[0] not in touched                     # solved: not rebuilt
    assert names[1] in touched                         # planned: built now
    assert runs2.state(names[0]).solution == first.solution


def test_run_resumes_an_interrupted_cluster_from_its_run_directory(tmp_path):
    task, doc, _spawn, rc, runs = _run(tmp_path, dry_run=True)     # nothing created
    name = doc["clusters"][0]["name"]
    run_dir = tmp_path / ".artifacts" / f"{name}_broken"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"task": name}), encoding="utf-8")
    (tmp_path / "tasks" / name).mkdir(parents=True, exist_ok=True)
    (tmp_path / "tasks" / name / "task.toml").write_text("[task]\n", encoding="utf-8")
    product = Path(doc["product_dir"])
    product.mkdir(parents=True, exist_ok=True)
    ledger = {"schema": 1, "plan": "", "sol_goal": 0.9,
              "clusters": {name: {"status": "running", "run": str(run_dir), "task":
                                  str(tmp_path / "tasks" / name)}}}
    (product / "runs.json").write_text(json.dumps(ledger), encoding="utf-8")
    spawn = FakeForge(tmp_path)
    rc, runs2 = R.run_plan(doc, root=tmp_path, spawn=spawn)
    assert rc == R.EXIT_OK
    first = spawn.calls[0]
    assert first[3] == "run" and first[4] == str(run_dir)   # no init, no new
    assert runs2.state(name).status == R.STATUS_SOLVED


def test_dry_run_runs_nothing_and_writes_nothing(tmp_path):
    task, doc, spawn, rc, runs = _run(tmp_path, dry_run=True)
    assert rc == R.EXIT_OK
    assert spawn.calls == []
    assert not (task / "shapes" / "runs.json").exists()
    assert all(s.status == R.STATUS_PLANNED for s in runs.clusters.values())


def test_the_dry_run_printout_names_every_command(tmp_path, capsys):
    task = _task(tmp_path)
    doc = P.build_plan(task).as_json()
    printed: list[str] = []
    rc, _runs = R.run_plan(doc, root=tmp_path, spawn=FakeForge(tmp_path), dry_run=True,
                           log=printed.append)
    assert rc == R.EXIT_OK
    text = "\n".join(printed)
    for word in ("forgekernel init", "forgekernel new", "forgekernel run",
                 "forgekernel export", "[target] ms", "--workload"):
        assert word in text, word


def test_max_hours_stops_before_the_next_cluster(tmp_path):
    task = _task(tmp_path)
    doc = P.build_plan(task).as_json()
    now = {"t": 0.0}

    def clock() -> float:
        now["t"] += 0.6                     # two checks run the budget out
        return now["t"]

    spawn = FakeForge(tmp_path)
    rc, runs = R.run_plan(doc, root=tmp_path, spawn=spawn, max_hours=1.0, clock=clock)
    assert rc == R.EXIT_UNFINISHED
    statuses = [runs.state(c["name"]).status for c in doc["clusters"]]
    assert statuses[0] == R.STATUS_SOLVED and "planned" in statuses[1:]
    notes = " ".join(n for s in runs.clusters.values() for n in s.notes)
    assert "--max-hours" in notes


def test_run_dir_parsing(tmp_path):
    run = tmp_path / ".artifacts" / "op-c0_260101-000000-abcd"
    run.mkdir(parents=True)
    (run / "run.json").write_text("{}", encoding="utf-8")
    text = "some banner\nrun directory .artifacts/op-c0_260101-000000-abcd\n"
    assert R.run_dir_from(text, root=tmp_path, task_name="op-c0") == run
    # an absolute announcement works too
    text = f"run directory {run}\n"
    assert R.run_dir_from(text, root=tmp_path, task_name="op-c0") == run
    assert R.run_dir_from("nothing here", root=tmp_path, task_name="op-c0") == run
    assert R.run_dir_from("nothing", root=tmp_path, task_name="op-c9") is None


def test_toml_editing_replaces_or_appends(tmp_path):
    text = "[task]\nlanguage = \"cutedsl\"\n\n[target]\nratio = 1.2\ntiming = \"device\"\n"
    new = R.set_toml_keys(text, "target", {"ms": 0.5}, drop=("ratio",))
    cfg = tomllib.loads(new)
    assert cfg["target"]["ms"] == 0.5 and "ratio" not in cfg["target"]
    assert cfg["target"]["timing"] == "device"            # the rest of the table survives
    appended = R.set_toml_keys(text.replace('\n[target]\nratio = 1.2\ntiming = "device"\n',
                                            "\n"), "budget", {"hours": 2})
    assert tomllib.loads(appended)["budget"]["hours"] == 2
    assert R.set_toml_keys(text, "target", {}) == text
    # a comment inside the table stays, and the new key is added beside it
    commented = "[target]\n# the gate\ntiming = \"device\"\n"
    merged = R.set_toml_keys(commented, "target", {"ms": 1.5})
    assert "# the gate" in merged and tomllib.loads(merged)["target"]["ms"] == 1.5


def test_apply_target_leaves_a_missing_task_toml_alone(tmp_path):
    notes = R.apply_target(tmp_path / "nothing", target_ms=1.0, hours=2.0)
    assert notes and "missing" in notes[0]


def test_cluster_targets_match_the_plan(tmp_path):
    doc = P.build_plan(_task(tmp_path)).as_json()
    targets = R.cluster_targets(doc, 0.90)
    assert {c["name"]: round(c["target_ms"], 12) for c in doc["clusters"]} == \
           {k: round(v, 12) for k, v in targets.items()}
    higher = R.cluster_targets(doc, 0.95)
    assert all(higher[k] < targets[k] for k in targets)


def test_the_run_cli_prints_the_ledger(tmp_path, capsys):
    task = _task(tmp_path)
    doc = P.build_plan(task).as_json()
    plan_path = P.write_plan_doc(doc, Path(doc["product_dir"]))
    assert cli.main(["run", str(plan_path), "--root", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "forgekernel init" in out and "cluster" in out
    assert cli.main(["run", str(tmp_path / "nope.json")]) == 2
