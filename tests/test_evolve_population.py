"""The population state machine: run, resume, concurrency, migration, archive, metric.

Everything GPU- and agent-shaped is faked at the seams the population itself uses
(``new_fn`` / ``run_fn`` / ``metric_fn`` / ``best_fn``), so these tests are pure
filesystem + scripted readings -- no ``forge run``, no model, no GPU.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from forgekernel.evolve import archive as archive_mod
from forgekernel.evolve import lineage as lineage_mod
from forgekernel.evolve import metric as metric_mod
from forgekernel.evolve import population as population_mod
from forgekernel.evolve.metric import Metric
from forgekernel.evolve.population import Config, Population, initial_state
from forgekernel.task import load_task

KERNEL = "import torch\n\ndef run(x, w):\n    return x * w\n"


# -- fixtures ----------------------------------------------------------------------------


def _ordinary(root: Path, name="rmsnorm") -> Path:
    d = root / "tasks" / name
    d.mkdir(parents=True)
    (d / "task.toml").write_text('[task]\nlanguage = "cutedsl"\narch = "sm100"\n'
                                 '[reference]\nrun = "reference/run.py:run"\n'
                                 'source = "reference"\n')
    (d / "description.md").write_text("y = x * w\n")
    (d / "shapes.json").write_text('[{"name": "s", "axes": {"m": 4}}]')
    (d / "reference").mkdir()
    (d / "reference" / "run.py").write_text(KERNEL)
    return d


def _board(root: Path, name="brmsnorm") -> Path:
    d = root / "tasks" / name
    card = d / "leaderboard" / "card" / "eval" / "leaderboard"
    card.mkdir(parents=True)
    (card / "problem.json").write_text(json.dumps({
        "schema": 1, "suite": "sol-execbench",
        "definition": {"name": "rmsnorm"}, "workload": {"uuid": "w0", "axes": {"m": 4}},
        "kernel_file": "kernel.py", "entry": "run",
        "reference_impl": {"kind": "definition"},
        "source": {"dataset_root": str(root), "problem": "L1/rmsnorm"},
        "sol_board": {"kernel_id": 7, "workload_index": 0, "t_b_ms": 1.0,
                      "t_sol_ms": 0.5, "clock_mhz": {}}}))
    (d / "task.toml").write_text('[task]\nlanguage = "cutedsl"\narch = "sm100"\n'
                                 '[leaderboard]\nsuite = "sol-execbench"\n'
                                 'card = "leaderboard/card"\n')
    return d


def _key(run_dir: Path) -> tuple[int, int]:
    task_name = Path(run_dir).name.rsplit("__", 1)[0]
    if ".e" in task_name:
        _, rest = task_name.split(".e", 1)
        i, gen = rest.split(".g", 1)
        return int(i), int(gen)
    if "-g" in task_name:
        return 0, int(task_name.rsplit("-g", 1)[1])
    return 0, 1


class Harness:
    def __init__(self, pop, script, calls):
        self.pop = pop
        self.script = script
        self.calls = calls

    def run(self) -> int:
        return self.pop.run()

    def state(self) -> dict:
        return json.loads((self.pop.root / population_mod.STATE_FILE).read_text())


def build(tmp_path, monkeypatch, script, *, lineages=1, langs=None, generations=None,
          parallel=1, seeds=None, migration_from=2, refuge=1, elite=None, max_fails=2,
          base="rmsnorm", board=False, best_kernel=KERNEL, versions=0,
          metric_raises=None, extra_run=None):
    repo = tmp_path / "repo"
    base_dir = _board(repo) if board else _ordinary(repo, base)
    task = load_task(base_dir)
    root = tmp_path / "pop"
    config = Config(problem=lineage_mod.base_name(task.name), base_task=str(base_dir),
                    lineages=lineages, generations=generations, parallel=parallel,
                    langs=langs or [task.language], migration_from=migration_from,
                    refuge=refuge, elite=elite, max_fails=max_fails)
    state = initial_state(root, config, task, seeds, log=lambda *_: None)
    calls = {"new": [], "run": [], "inflight": 0, "max_inflight": 0}
    lock = threading.Lock()
    counter = {"n": 0}

    def fake_new(task_dir: Path):
        calls["new"].append(Path(task_dir))
        counter["n"] += 1
        run_dir = tmp_path / "runs" / f"{Path(task_dir).name}__{counter['n']}"
        run_dir.mkdir(parents=True)
        return run_dir

    def fake_run(run_dir: Path, env: dict) -> int:
        with lock:
            calls["inflight"] += 1
            calls["max_inflight"] = max(calls["max_inflight"], calls["inflight"])
        calls["run"].append(Path(run_dir))
        time.sleep(0.02)
        if extra_run:
            extra_run(run_dir)
        with lock:
            calls["inflight"] -= 1
        return 0

    def fake_best(run, task):
        i, gen = _key(run.dir)
        entry = script.get((i, gen))
        if entry is None:
            return None
        if isinstance(entry, dict):
            return entry
        return {"sha": f"{i:02d}{gen:02d}" + "a" * 36, "window": "device",
                "median_ms": entry[1], "speedup": entry[0],
                "ratios": [], "kernel": best_kernel}

    def fake_metric(run, task, *, best=None, s_floor=0.005):
        i, gen = _key(run.dir)
        if metric_raises and (i, gen) in metric_raises:
            raise KeyboardInterrupt()
        entry = script.get((i, gen))
        if entry is None:
            return None
        if isinstance(entry, dict):     # sol entry
            return Metric(kind="sol", value=entry["value"], floor=s_floor,
                          ms=entry.get("ms"), source="test",
                          confidence=("dev-only",))
        return Metric(kind="speedup", value=entry[0], floor=0.015, ms=entry[1],
                      source="test")

    def fake_versions(run, task):
        if not versions:
            return []
        return [{"sha": f"h{k}" + "0" * 38, "ms": 1.0 - k / 100,
                 "t_ms": 1.0 - k / 100, "verdict": "KEEP",
                 "source": f"git:h{k}:kernel.py", "kernel": f"# v{k}\n",
                 "kernel_rel": "kernel.py"} for k in range(versions)]

    monkeypatch.setattr(lineage_mod, "add_exclude_pattern", lambda *a, **k: None)
    monkeypatch.setattr(population_mod, "open_run", lambda path: SimpleNamespace(
        dir=Path(path), operators_dir=Path(path) / "operators", ws=Path(path) / "ws",
        task_name=Path(path).name))
    monkeypatch.setattr(archive_mod, "collect_versions", fake_versions)
    pop = Population(root, state, config, base_task=task, repo=repo,
                     new_fn=fake_new, run_fn=fake_run, metric_fn=fake_metric,
                     best_fn=fake_best, log=lambda *_: None)
    pop.init_kernel_supported = False
    return Harness(pop, script, calls)


# -- one lineage: the old chain ----------------------------------------------------------


def test_one_lineage_walks_the_old_chain(tmp_path, monkeypatch):
    h = build(tmp_path, monkeypatch,
              {(0, 1): (1.30, 1.00), (0, 2): (1.01, 1.00)}, lineages=1)
    assert h.run() == 0
    assert [p.name for p in h.calls["new"]] == ["rmsnorm", "rmsnorm-g2"]
    g2 = tmp_path / "repo" / "tasks" / "rmsnorm-g2"
    assert (g2 / "reference" / "kernel.py").read_text() == KERNEL
    assert load_task(g2).mode == "source"
    state = h.state()
    assert state["next_generation"] == 3
    assert state["stopped"]["reason"] == "all_retired"
    assert state["generations"][1]["decisions"]["0"]["action"] == "retire"


def test_one_lineage_without_a_best_exits_1(tmp_path, monkeypatch):
    h = build(tmp_path, monkeypatch, {(0, 1): None}, lineages=1)
    assert h.run() == 1
    assert h.state()["stopped"]["reason"] == "all_retired"


def test_the_generation_cap_stops(tmp_path, monkeypatch):
    h = build(tmp_path, monkeypatch,
              {(0, 1): (1.30, 1.0), (0, 2): (1.30, 1.0), (0, 3): (1.30, 1.0)},
              lineages=1, generations=2)
    assert h.run() == 0
    assert [p.name for p in h.calls["new"]] == ["rmsnorm", "rmsnorm-g2"]
    assert h.state()["stopped"]["reason"] == "generations"


def test_there_is_no_generation_cap_by_default(tmp_path, monkeypatch):
    # the same scripted chain tests/test_self_evolve.py runs: six gains over the floor,
    # then one inside it
    script = {(0, g): (1.2, 1.0) for g in range(1, 7)}
    script[(0, 7)] = (1.005, 1.0)
    h = build(tmp_path, monkeypatch, script, lineages=1)
    assert h.run() == 0
    assert [p.name for p in h.calls["new"]] == ["rmsnorm"] + \
        [f"rmsnorm-g{k}" for k in range(2, 8)]


def test_stalled_population_stops_instead_of_looping(tmp_path, monkeypatch):
    h = build(tmp_path, monkeypatch,
              {(0, 1): (1.0, 1.0), (1, 1): (1.0, 1.0)}, lineages=2, refuge=0)
    assert h.run() == 0
    assert h.state()["stopped"]["reason"] in ("all_retired", "stalled")
    assert len(h.calls["new"]) == 2


# -- concurrency -------------------------------------------------------------------------


def test_parallel_caps_the_inflight_runs(tmp_path, monkeypatch):
    script = {(i, 1): (1.30, 1.0) for i in range(3)}
    h = build(tmp_path, monkeypatch, script, lineages=3, parallel=2, generations=1,
              refuge=0)
    assert h.run() == 0
    assert len(h.calls["run"]) == 3
    assert h.calls["max_inflight"] == 2
    assert h.calls["max_inflight"] <= 2


# -- resume ------------------------------------------------------------------------------


def test_resume_after_an_interruption_does_not_rerun_finished_generations(tmp_path,
                                                                         monkeypatch):
    script = {(0, 1): (1.30, 1.0), (0, 2): (1.30, 1.0),
              (1, 1): (1.30, 1.0), (1, 2): (1.30, 1.0)}
    h = build(tmp_path, monkeypatch, script, lineages=2, generations=2, parallel=1,
              metric_raises={(0, 2)})
    assert h.run() == 130
    state = h.state()
    assert state["stopped"]["reason"] == "interrupted"
    assert state["generations"][0]["state"] == "done"
    assert state["generations"][1]["state"] == "in_progress"
    new_before = len(h.calls["new"])
    gen2_run0 = state["generations"][1]["lineages"][0]["run"]
    gen2_run1 = state["generations"][1]["lineages"][1]["run"]

    # a second process continues the same population
    monkeypatch.setattr(population_mod, "open_run", lambda path: SimpleNamespace(
        dir=Path(path), operators_dir=Path(path) / "operators", ws=Path(path) / "ws",
        task_name=Path(path).name))
    h2 = Population.load(h.pop.root, base_task=h.pop.base_task, repo=h.pop.repo,
                         new_fn=h.pop.new_fn, run_fn=h.pop.run_fn,
                         metric_fn=lambda run, task, *, best=None, s_floor=0.005:
                         Metric(kind="speedup", value=1.30, floor=0.015, ms=1.0),
                         best_fn=h.pop.best_fn, log=lambda *_: None)
    h2.init_kernel_supported = False
    assert h2.run() == 0
    state2 = json.loads((h.pop.root / population_mod.STATE_FILE).read_text())
    assert state2["stopped"]["reason"] == "generations"
    assert len(h.calls["new"]) == new_before          # generation 1 was not run again
    # generation 2 kept the run directories it had already recorded
    assert state2["generations"][1]["lineages"][0]["run"] == gen2_run0
    assert state2["generations"][1]["lineages"][1]["run"] == gen2_run1
    events = [json.loads(x)["event"]
              for x in (h.pop.root / population_mod.LEDGER_FILE).read_text().splitlines()]
    assert events.count("generation_started") == 2    # gen 1 and gen 2 (not re-logged)
    assert "population_resumed" in events


# -- migration ---------------------------------------------------------------------------


def test_migration_hands_the_elite_to_a_stalled_laggard(tmp_path, monkeypatch):
    script = {
        (0, 1): (1.02, 1.30), (0, 2): (1.00, 1.30),
        (1, 1): (1.30, 1.00), (1, 2): (1.01, 0.99), (1, 3): (1.00, 0.98),
        (2, 1): (1.20, 1.10), (2, 2): (1.02, 1.08), (2, 3): (1.00, 1.08),
        (0, 3): (1.15, 0.85),
    }
    h = build(tmp_path, monkeypatch, script, lineages=3,
              langs=["cutedsl", "cutedsl", "cuda"], generations=3, parallel=1,
              migration_from=2, elite=1, refuge=1)
    assert h.run() == 0
    state = h.state()
    g2 = {int(k): v for k, v in state["generations"][1]["decisions"].items()}
    assert g2[0]["action"] == "migrate" and g2[0]["donor"] == 1
    assert g2[2]["action"] == "self"                    # the refuge (cuda) stays
    dest = tmp_path / "repo" / "tasks" / "rmsnorm.e0.g3"
    donor = h.pop.root / "variants" / "l1-g2" / "kernel.py"
    assert dest.is_dir()
    assert (dest / "reference" / "kernel.py").read_text() == donor.read_text()
    events = [json.loads(x) for x in
              (h.pop.root / population_mod.LEDGER_FILE).read_text().splitlines()]
    assert any(e["event"] == "migration" and e["donor"] == 1 for e in events)
    assert (h.pop.root / "variants" / "l0-g3" / "kernel.py").is_file()
    text = (h.pop.root / "archive.md").read_text()
    assert "| lineage | gen |" in text
    assert "### l1-g2" in text and "# Variant: l1-g2" in text
    assert "待榜单确认" not in text                    # ordinary task: no board marker


def test_refuge_retires_when_it_stalls(tmp_path, monkeypatch):
    script = {(0, 1): (1.30, 1.0), (1, 1): (1.20, 1.1),
              (0, 2): (1.30, 1.0), (1, 2): (1.00, 1.1)}
    h = build(tmp_path, monkeypatch, script, lineages=2, langs=["cutedsl", "cuda"],
              generations=2, migration_from=2, elite=1, refuge=1)
    h.run()
    state = h.state()
    d = state["generations"][1]["decisions"]
    assert d["1"]["action"] == "retire"
    assert state["lineages"][1]["retired"] is True
    assert not (tmp_path / "repo" / "tasks" / "rmsnorm.e1.g3").exists()


def test_from_lib_seeds_start_the_lineages(tmp_path, monkeypatch):
    seed = tmp_path / "seed_kernel.py"
    seed.write_text("# seed\n" + KERNEL)
    seeds = [{"id": "hyra-051", "path": str(seed), "language": "cutedsl",
              "license": "Apache-2.0"}]
    h = build(tmp_path, monkeypatch, {(0, 1): (1.30, 1.0), (1, 1): (1.20, 1.1)},
              lineages=2, seeds=seeds, generations=1)
    assert h.run() == 0
    l0 = tmp_path / "repo" / "tasks" / "rmsnorm.e0.g1"
    l1 = tmp_path / "repo" / "tasks" / "rmsnorm.e1.g1"
    assert (l0 / "reference" / "kernel.py").read_text() == seed.read_text()
    assert (l1 / "reference" / "run.py").read_text() == KERNEL
    assert h.state()["lineages"][0]["seed"]["id"] == "hyra-051"


def test_parallel_default_is_two():
    assert Config(problem="x", base_task="y").parallel == 2


# -- archive -----------------------------------------------------------------------------


def test_archive_lists_every_passing_version(tmp_path, monkeypatch):
    h = build(tmp_path, monkeypatch, {(0, 1): (1.30, 1.0)}, lineages=1,
              generations=1, versions=2)
    assert h.run() == 0
    vdir = h.pop.root / "variants" / "l0-g1"
    assert len(list((vdir / "history").iterdir())) == 2
    versions = json.loads((vdir / "versions.json").read_text())
    assert [v["verdict"] for v in versions] == ["KEEP", "KEEP"]
    text = (h.pop.root / "archive.md").read_text()
    assert "## Correctness-passing versions" in text
    assert (vdir / "parent.txt").read_text().strip() == "init"
    result = json.loads((vdir / "result.json").read_text())
    assert result["reference"] == "base" and result["metric"]["kind"] == "speedup"


# -- metric readings ---------------------------------------------------------------------


def _run_with_eval(tmp_path, *, sol_block, task_name="brmsnorm"):
    run = tmp_path / "run"
    op = run / "operators" / task_name
    (op / "jobs" / "42").mkdir(parents=True)
    (op / "best_confirmation.json").write_text(json.dumps({
        "sha": "a" * 40, "window": "device", "median_ms": 0.8, "median_ratio": 1.1,
        "jobs": [{"job": "42", "ms": 0.8, "ratio": 1.1}]}))
    (op / "jobs" / "42" / "log.txt").write_text(
        "noise\n=== JSON ===\n" + json.dumps({
            "correctness": {"pass": True},
            "performance": {"device_median_ms": 0.8,
                            "workloads": [{"name": "w0", "device_median_ms": 0.8}],
                            "leaderboard": {"suite": "sol-execbench",
                                            "sol_board": sol_block}}}) + "\n")
    return SimpleNamespace(dir=run, operators_dir=run / "operators", ws=run / "ws",
                           task_name=task_name)


def _best(speedup=None):
    return {"sha": "a" * 40, "median_ms": 0.8, "speedup": speedup, "ratios": [],
            "kernel": KERNEL}


def test_metric_reads_the_cards_sol_block(tmp_path, monkeypatch):
    task = load_task(_board(tmp_path))
    run = _run_with_eval(tmp_path, sol_block={
        "sol_score": 0.75, "t_b_ms": 1.0, "t_sol_ms": 0.5, "kernel_id": 7,
        "workload_index": 0})
    monkeypatch.setattr(metric_mod.launch, "best_of", lambda run, task: _best(1.1))
    m = metric_mod.metric_of(run, task)
    assert m.kind == "sol" and m.value == pytest.approx(0.75)
    assert m.ms == pytest.approx(0.8)
    assert m.workloads[0]["name"] == "w0"
    assert "dev-only" in m.confidence and m.board_pending


def test_metric_reads_a_group_sol_block(tmp_path, monkeypatch):
    task = load_task(_board(tmp_path))
    run = _run_with_eval(tmp_path, sol_block={
        "sol_score_mean": 0.6, "sol_score": {"w0": 0.5, "w1": 0.7},
        "kernel_id": 7, "workload_index": [0, 1]})
    monkeypatch.setattr(metric_mod.launch, "best_of", lambda run, task: _best(None))
    m = metric_mod.metric_of(run, task)
    assert m.kind == "sol" and m.value == pytest.approx(0.6)
    assert {w["s"] for w in m.workloads} == {0.5, 0.7}


def test_metric_falls_back_to_speedup_without_a_sol_block(tmp_path, monkeypatch):
    task = load_task(_board(tmp_path))
    run = _run_with_eval(tmp_path, sol_block=None)
    monkeypatch.setattr(metric_mod.launch, "best_of", lambda run, task: _best(1.25))
    m = metric_mod.metric_of(run, task)
    assert m.kind == "speedup" and m.value == pytest.approx(1.25)
    assert m.floor == pytest.approx(0.005)      # board tasks use --s-floor


def test_metric_uses_the_campaign_hook_when_installed(tmp_path, monkeypatch):
    import sys
    task = load_task(_board(tmp_path))
    run = _run_with_eval(tmp_path, sol_block=None)
    mod = type(sys)("forgekernel.campaign.solscore")
    mod.score_kernel = lambda obj: 0.42
    monkeypatch.setitem(sys.modules, "forgekernel.campaign.solscore", mod)
    monkeypatch.setattr(metric_mod.launch, "best_of", lambda run, task: _best(1.1))
    m = metric_mod.metric_of(run, task)
    assert (m.kind, m.value) == ("sol", pytest.approx(0.42))
    assert "solscore" in m.source


def test_metric_uses_benchx_when_installed(tmp_path, monkeypatch):
    import sys
    task = load_task(_board(tmp_path))
    run = _run_with_eval(tmp_path, sol_block=None)
    reader = type(sys)("forgekernel.benchx.reader")
    reader.read_run = lambda *a, **k: SimpleNamespace(
        mean_s=0.9, confidence=["dev-only", "fp4-bound"],
        workloads=[{"name": "w0", "t_ms": 0.7}])
    monkeypatch.setitem(sys.modules, "forgekernel.benchx", type(sys)("forgekernel.benchx"))
    monkeypatch.setitem(sys.modules, "forgekernel.benchx.reader", reader)
    monkeypatch.setattr(metric_mod.launch, "best_of", lambda run, task: _best(1.1))
    m = metric_mod.metric_of(run, task)
    assert (m.kind, m.value) == ("sol", pytest.approx(0.9))
    assert "fp4-bound" in m.confidence


def test_ordinary_metric_uses_the_confirmation_spread(tmp_path, monkeypatch):
    task = load_task(_ordinary(tmp_path))
    run = _run_with_eval(tmp_path, sol_block=None, task_name="rmsnorm")
    monkeypatch.setattr(metric_mod.launch, "best_of", lambda run, task: {
        "sha": "a" * 40, "median_ms": 0.8, "speedup": 1.03,
        "ratios": [1.01, 1.03, 1.0504], "kernel": KERNEL})
    m = metric_mod.metric_of(run, task)
    assert m.kind == "speedup"
    assert m.floor == pytest.approx(0.04)      # the spread of the same-job ratios


# -- the command -------------------------------------------------------------------------


def test_cli_dry_run_prints_and_writes_nothing(tmp_path, monkeypatch, capsys):
    from forgekernel.ext import evolve as evolve_cli
    repo = tmp_path / "repo"
    base = _ordinary(repo)
    monkeypatch.setattr(evolve_cli.launch, "repo_root", lambda: repo)
    rc = evolve_cli.main([str(base), "--dry-run", "--lineages", "2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry-run" in out
    assert "rmsnorm.e0.g1" in out and "rmsnorm.e1.g1" in out
    assert not (repo / ".artifacts").exists()
    assert not (repo / "tasks" / "rmsnorm.e0.g1").exists()


def test_cli_rejects_an_unknown_language(tmp_path, monkeypatch, capsys):
    from forgekernel.ext import evolve as evolve_cli
    repo = tmp_path / "repo"
    base = _ordinary(repo)
    monkeypatch.setattr(evolve_cli.launch, "repo_root", lambda: repo)
    rc = evolve_cli.main([str(base), "--langs", "tilelang", "--dry-run"])
    assert rc == 2
    assert "tilelang" in capsys.readouterr().err


def test_cli_status_and_show_read_a_population(tmp_path, monkeypatch, capsys):
    from forgekernel.ext import evolve as evolve_cli
    h = build(tmp_path, monkeypatch, {(0, 1): (1.30, 1.0)}, lineages=1,
              generations=1)
    assert h.run() == 0
    rc = evolve_cli.main(["status", str(h.pop.root)])
    out = capsys.readouterr().out
    assert rc == 0 and "generation 1" in out and "rmsnorm" in out
    rc = evolve_cli.main(["show", str(h.pop.root)])
    out = capsys.readouterr().out
    assert rc == 0 and "# Evolve archive" in out and "l0-g1" in out


def test_cli_refuses_a_missing_population(capsys):
    from forgekernel.ext import evolve as evolve_cli
    assert evolve_cli.main(["status", "/nonexistent/pop"]) == 2
    assert "does not exist" in capsys.readouterr().err
