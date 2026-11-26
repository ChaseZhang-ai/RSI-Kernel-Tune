"""Lineage task directories: ordinary source references and board re-inits.

The board path is tested through the argv handed to ``forge init`` (monkeypatched)
and the sealed/exports it asks for -- no dataset, no GPU, no network.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rsi_kernel_tune.evolve import lineage
from rsi_kernel_tune.evolve.lineage import LineageError, Reference, Request, make_task
from rsi_kernel_tune.task import LANGUAGES, load_task

KERNEL = "import torch\n\ndef run(x, w):\n    return x * w\n"
DONOR = "import torch\n\ndef run(x, w):\n    return (x * w).contiguous()\n"


# -- fixtures ----------------------------------------------------------------------------


def _ordinary(root: Path, name="rmsnorm", language="cutedsl") -> Path:
    d = root / "tasks" / name
    d.mkdir(parents=True)
    (d / "task.toml").write_text(f'[task]\nlanguage = "{language}"\narch = "sm100"\n'
                                 '[reference]\nrun = "reference/run.py:run"\n'
                                 'source = "reference"\n'
                                 '[budget]\nhours = 2.5\n')
    (d / "description.md").write_text("y = x * w\n")
    (d / "shapes.json").write_text('[{"name": "s", "axes": {"m": 4}}]')
    (d / "reference").mkdir()
    (d / "reference" / "run.py").write_text(KERNEL)
    return d


def _board(root: Path, *, group=False, name="brmsnorm") -> Path:
    d = root / "tasks" / name
    card = d / "leaderboard" / "card"
    (card / "eval" / "leaderboard").mkdir(parents=True)
    definition = {"name": "rmsnorm"}
    workload = {"uuid": "w0", "axes": {"m": 4}}
    problem = {
        "schema": 1, "suite": "sol-execbench",
        "definition": definition, "workload": workload,
        "dps": True, "kernel_file": "kernel.py", "entry": "run",
        "reference_impl": {"kind": "definition"},
        "source": {"dataset": "SOL-ExecBench", "dataset_root": str(root),
                   "problem": "L1/rmsnorm"},
        "sol_board": {"kernel_id": 7, "workload_index": 0,
                      "t_b_ms": 1.0, "t_sol_ms": 0.5, "clock_mhz": {}},
    }
    if group:
        problem["group"] = [
            {"index": 0, "name": "wl0", "workload": {"uuid": "w0", "axes": {"m": 4}},
             "sol_board": {"kernel_id": 7, "workload_index": 0, "t_b_ms": 1.0,
                           "t_sol_ms": 0.5, "clock_mhz": {}}},
            {"index": 1, "name": "wl1", "workload": {"uuid": "w1", "axes": {"m": 8}},
             "sol_board": {"kernel_id": 7, "workload_index": 1, "t_b_ms": 2.0,
                           "t_sol_ms": 1.0, "clock_mhz": {}}},
        ]
        problem["workload"] = problem["group"][0]["workload"]
        problem.pop("sol_board", None)
    (card / "eval" / "leaderboard" / "problem.json").write_text(json.dumps(problem))
    d.mkdir(parents=True, exist_ok=True)
    (d / "task.toml").write_text('[task]\nlanguage = "cutedsl"\narch = "sm100"\n'
                                 '[leaderboard]\nsuite = "sol-execbench"\n'
                                 'card = "leaderboard/card"\n')
    return d


def _req(base: Path, dest: Path, **kw) -> Request:
    return Request(base=load_task(base), dest=dest, language=kw.pop("language", "cutedsl"),
                   generation=kw.pop("generation", 2), lineages=kw.pop("lineages", 3),
                   step=kw.pop("step", 1.05), add_exclude=kw.pop("add_exclude", False),
                   workdir=kw.pop("workdir", dest.parent / "solutions"), **kw)


@pytest.fixture(autouse=True)
def _no_exclude():
    """Requests built here set ``add_exclude=False`` (see ``_req``), so no test touches
    the real repository's .git/info/exclude except the one that points the module at a
    throwaway repository."""
    yield


# -- ordinary tasks ----------------------------------------------------------------------


def test_ordinary_handoff_writes_a_source_reference(tmp_path):
    base = _ordinary(tmp_path)
    donor = tmp_path / "donor.py"
    donor.write_text(DONOR)
    dest = tmp_path / "tasks" / "rmsnorm.e0.g2"
    rec = make_task(_req(base, dest, reference=Reference(
        kind="self", kernel=donor, language="cutedsl", variant="l0-g1")))
    assert rec["reference"]["kind"] == "self"
    assert (dest / "reference" / "kernel.py").read_text() == DONOR
    assert (dest / "description.md").read_text() == "y = x * w\n"
    assert (dest / "shapes.json").is_file()
    task = load_task(dest)
    assert task.mode == "source"
    assert task.reference.run_fn == "run"
    assert task.target.ratio == pytest.approx(1.05)
    assert task.budget.hours == 2.5
    assert (dest / "prior" / "donor.py").read_text() == DONOR


def test_ordinary_first_generation_copies_the_base_reference(tmp_path):
    base = _ordinary(tmp_path)
    dest = tmp_path / "tasks" / "rmsnorm.e0.g1"
    rec = make_task(_req(base, dest, generation=1, reference=None))
    assert rec["reference"]["kind"] == "base"
    assert (dest / "reference" / "run.py").read_text() == KERNEL
    task = load_task(dest)
    assert task.mode == "source"
    assert task.target.ratio == pytest.approx(1.05)


def test_one_lineage_generation_one_keeps_the_base_target(tmp_path):
    base = _ordinary(tmp_path)
    dest = tmp_path / "tasks" / "rmsnorm"
    # the generation-1 directory of a one-lineage population is the base task itself;
    # next_generation() in scripts/self_evolve.py leaves the target alone there
    req = _req(base, dest, generation=1, lineages=1, reference=None)
    assert not req.verifies_target
    base2 = _ordinary(tmp_path / "other")
    dest2 = tmp_path / "other" / "tasks" / "rmsnorm-g2"
    task = load_task(make_task(_req(base2, dest2, generation=2, lineages=1,
                                    reference=None))["task"])
    assert task.target.ratio == pytest.approx(1.05)     # later generations: --step


def test_ordinary_cuda_donor_is_refused(tmp_path):
    base = _ordinary(tmp_path)
    donor = tmp_path / "donor.cu"
    donor.write_text("// kernel\n")
    with pytest.raises(LineageError, match=r"\.cu"):
        make_task(_req(base, tmp_path / "tasks" / "rmsnorm-g2", reference=Reference(
            kind="self", kernel=donor, language="cuda", variant="l0-g1")))


def test_unknown_language_is_refused_with_the_accepted_list(tmp_path):
    base = _ordinary(tmp_path)
    with pytest.raises(LineageError) as ei:
        make_task(_req(base, tmp_path / "tasks" / "rmsnorm-g2", language="tilelang"))
    assert list(LANGUAGES)[0] in str(ei.value)
    assert "tilelang" in str(ei.value)


def test_effort_and_language_are_patched(tmp_path):
    base = _ordinary(tmp_path)
    dest = tmp_path / "tasks" / "rmsnorm-g2"
    make_task(_req(base, dest, language="cutedsl", effort="max"))
    task = load_task(dest)
    assert task.student.effort == "max"
    assert task.language == "cutedsl"


def test_copy_task_repatches_a_retry(tmp_path):
    base = _ordinary(tmp_path)
    prev = tmp_path / "tasks" / "rmsnorm-g2"
    make_task(_req(base, prev, reference=None))
    dest = tmp_path / "tasks" / "rmsnorm-g3"
    rec = lineage.copy_task(_req(base, dest, generation=3), prev)
    assert rec["reference"]["kind"] == "retry"
    task = load_task(dest)
    assert task.target.ratio == pytest.approx(1.05)
    assert task.language == "cutedsl"


# -- board tasks -------------------------------------------------------------------------


def _capture_init(monkeypatch, base: Path, calls: list):
    root = base.parent.parent          # <root>/tasks/<name> -> <root>

    def fake_init(argv):
        calls.append(list(argv))
        dest = root / "tasks" / argv[0]
        shutil.copytree(base, dest)
        if "--kernel-file" in argv:
            kf = argv[argv.index("--kernel-file") + 1]
            p = dest / "leaderboard" / "card" / "eval" / "leaderboard" / "problem.json"
            problem = json.loads(p.read_text())
            problem["kernel_file"] = kf
            p.write_text(json.dumps(problem))
        return 0
    monkeypatch.setattr(lineage.launch, "cmd_init", fake_init)


def _capture_export(monkeypatch, calls: list):
    def fake_export(argv):
        calls.append(list(argv))
        out = Path(argv[argv.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
        return 0
    monkeypatch.setattr(lineage.launch, "cmd_export", fake_export)


def test_board_reinit_uses_the_sealed_reference(tmp_path, monkeypatch):
    base = _board(tmp_path)
    init_calls, export_calls = [], []
    _capture_init(monkeypatch, base, init_calls)
    _capture_export(monkeypatch, export_calls)
    donor = tmp_path / "donor.py"
    donor.write_text(DONOR)
    dest = tmp_path / "tasks" / "brmsnorm.e0.g2"
    rec = make_task(_req(base, dest, blackbox=True, generation=2, reference=Reference(
        kind="self", kernel=donor, run=tmp_path / "run", language="cutedsl",
        variant="l0-g1")))
    argv = init_calls[0]
    assert argv[0] == dest.name
    assert argv[argv.index("--leaderboard") + 1] == "sol-execbench"
    assert argv[argv.index("--definition") + 1] == "L1/rmsnorm"
    assert argv[argv.index("--workload") + 1] == "w0"
    assert argv[argv.index("--dataset") + 1] == str(tmp_path)
    assert "--sealed-ref" in argv
    assert "--no-sol-board" in argv          # a one-workload card cannot rebuild the record
    assert export_calls and export_calls[0][0] == str(tmp_path / "run")
    assert rec["reference"]["kind"] == "self"
    task = load_task(dest)
    assert task.leaderboard is not None
    assert task.target.ratio == pytest.approx(1.05)


def test_board_readable_handoff_prefers_init_kernel(tmp_path, monkeypatch):
    base = _board(tmp_path)
    init_calls, export_calls = [], []
    _capture_init(monkeypatch, base, init_calls)
    _capture_export(monkeypatch, export_calls)
    donor = tmp_path / "donor.py"
    donor.write_text(DONOR)
    dest = tmp_path / "tasks" / "brmsnorm.e0.g2"
    make_task(_req(base, dest, init_kernel_supported=True, reference=Reference(
        kind="self", kernel=donor, run=tmp_path / "run", language="cutedsl")))
    argv = init_calls[0]
    assert argv[argv.index("--init-kernel") + 1] == str(donor)
    assert "--sealed-ref" not in argv
    assert not export_calls
    assert (dest / "prior" / "donor.py").read_text() == DONOR


def test_board_falls_back_to_sealed_when_warmstart_is_absent(tmp_path, monkeypatch):
    base = _board(tmp_path)
    init_calls, export_calls = [], []
    _capture_init(monkeypatch, base, init_calls)
    _capture_export(monkeypatch, export_calls)
    donor = tmp_path / "donor.py"
    donor.write_text(DONOR)
    dest = tmp_path / "tasks" / "brmsnorm.e0.g2"
    make_task(_req(base, dest, init_kernel_supported=False, reference=Reference(
        kind="foreign", kernel=donor, run=tmp_path / "run", language="cutedsl",
        variant="l1-g1")))
    assert "--sealed-ref" in init_calls[0]
    assert export_calls
    # the honest record of what happened: blackbox handoff, not a readable one
    result = json.loads((dest / "leaderboard" / "card" / "eval" / "leaderboard" /
                         "problem.json").read_text())
    assert result["suite"] == "sol-execbench"


def test_board_record_is_rebuilt_when_the_card_covers_every_workload(tmp_path, monkeypatch):
    base = _board(tmp_path, group=True)
    init_calls = []
    _capture_init(monkeypatch, base, init_calls)
    _capture_export(monkeypatch, [])
    donor = tmp_path / "donor.py"
    donor.write_text(DONOR)
    dest = tmp_path / "tasks" / "brmsnorm.e0.g2"
    make_task(_req(base, dest, generation=2, blackbox=True, reference=Reference(
        kind="self", kernel=donor, run=tmp_path / "run", language="cutedsl")))
    argv = init_calls[0]
    assert argv[argv.index("--workload") + 1] == "0,1"
    assert "--sol-board" in argv
    assert "--no-sol-board" not in argv
    rec = json.loads(Path(argv[argv.index("--sol-board") + 1]).read_text())
    assert rec["name"] == "rmsnorm"
    assert [w["baseline_latency_ms"] for w in rec["workloads"]] == [1.0, 2.0]
    assert [w["sol_ms"] for w in rec["workloads"]] == [0.5, 1.0]


def test_board_generation_one_without_a_donor_just_copies(tmp_path, monkeypatch):
    base = _board(tmp_path)
    dest = tmp_path / "tasks" / "brmsnorm.e0.g1"

    def boom(*a, **k):
        raise AssertionError("no forge init for a plain same-problem copy")
    monkeypatch.setattr(lineage.launch, "cmd_init", boom)
    make_task(_req(base, dest, generation=1, reference=None))
    task = load_task(dest)
    assert task.leaderboard is not None
    assert task.target.ratio == pytest.approx(1.05)


def test_board_record_helper_rejects_a_subrange_card():
    problem = {
        "definition": {"name": "x"},
        "group": [{"index": 3, "workload": {"axes": {}},
                   "sol_board": {"t_b_ms": 1.0, "t_sol_ms": 0.5}}],
    }
    assert lineage.board_record_from_card(problem) is None


def test_board_seed_without_a_readable_handoff_is_dropped(tmp_path, monkeypatch):
    base = _board(tmp_path)
    init_calls = []
    _capture_init(monkeypatch, base, init_calls)
    seed = tmp_path / "seed.py"
    seed.write_text(KERNEL)
    dest = tmp_path / "tasks" / "brmsnorm.e0.g1"
    rec = make_task(_req(base, dest, generation=1, init_kernel_supported=False,
                         reference=Reference(kind="seed", kernel=seed,
                                             language="cutedsl", seed_id="lib-1")))
    assert "--init-kernel" not in init_calls[0]
    assert "--sealed-ref" not in init_calls[0]
    assert rec["reference"]["kind"] == "base"
    assert "seed dropped" in rec["handoff"]
    assert load_task(dest).leaderboard is not None


def test_board_language_change_reinits_with_kernel_cu(tmp_path, monkeypatch):
    base = _board(tmp_path)              # the card is a Python kernel card
    init_calls = []
    _capture_init(monkeypatch, base, init_calls)
    dest = tmp_path / "tasks" / "brmsnorm.e0.g1"
    make_task(_req(base, dest, language="cuda", generation=1, reference=None))
    argv = init_calls[0]
    assert argv[argv.index("--kernel-file") + 1] == "kernel.cu"
    task = load_task(dest)
    assert task.language == "cuda"


def test_board_language_and_card_file_must_agree(tmp_path, monkeypatch):
    base = _board(tmp_path)
    _capture_init(monkeypatch, base, [])

    def lying_init(argv):
        dest = tmp_path / "tasks" / argv[0]
        shutil.copytree(base, dest)      # card still says kernel.py
        return 0
    monkeypatch.setattr(lineage.launch, "cmd_init", lying_init)
    with pytest.raises(LineageError, match="kernel_file"):
        make_task(_req(base, tmp_path / "tasks" / "brmsnorm.e0.g1", language="cuda",
                       generation=1, reference=None))


# -- .git/info/exclude -------------------------------------------------------------------


def test_exclude_pattern_goes_to_the_local_exclude(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "tasks").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.setattr(lineage.launch, "repo_root", lambda: repo)
    dest = repo / "tasks" / "rmsnorm-g2"
    dest.mkdir()
    lineage.add_exclude_pattern(repo / "tasks", dest)
    exclude = repo / ".git" / "info" / "exclude"
    assert "/tasks/rmsnorm-g2/" in exclude.read_text()
    lineage.add_exclude_pattern(repo / "tasks", dest)     # idempotent
    assert exclude.read_text().count("/tasks/rmsnorm-g2/") == 1


# -- reference-library seeds (soft dependency) -------------------------------------------


def _fake_klib(tmp_path, monkeypatch, entries):
    mod = type(sys)("rsi_kernel_tune.klib")
    mod.find = lambda problem, k: entries[:k]
    monkeypatch.setitem(sys.modules, "rsi_kernel_tune.klib", mod)
    # `import rsi_kernel_tune.klib as klib` reaches the parent package's attribute when one is
    # already set; patch it too, so the fake is what a test sees whatever ran before it.
    import rsi_kernel_tune
    monkeypatch.setattr(rsi_kernel_tune, "klib", mod, raising=False)


def test_find_seeds_filters_licenses_and_missing_files(tmp_path, monkeypatch):
    good = tmp_path / "good.py"
    good.write_text(KERNEL)
    _fake_klib(tmp_path, monkeypatch, [
        {"id": "a", "path": str(good), "language": "triton", "license": "Apache-2.0"},
        {"id": "b", "path": str(good), "language": "triton", "license": "unknown"},
        {"id": "c", "path": str(tmp_path / "gone.py"), "language": "triton",
         "license": "MIT"},
        {"id": "d", "path": str(good), "language": "triton", "license": "MIT"},
    ])
    seeds = lineage.find_seeds("rmsnorm", 4, log=lambda *_: None)
    assert [s["id"] for s in seeds] == ["a", "d"]
    assert seeds[0]["path"] == str(good)


def test_find_seeds_without_the_library_is_a_clean_fallback(monkeypatch):
    # None in sys.modules is the standard stand-in for a package that is not installed
    # (deleting the module is not enough once rsi_kernel_tune.klib exists on disk: base's stub).
    monkeypatch.setitem(sys.modules, "rsi_kernel_tune.klib", None)
    logs = []
    assert lineage.find_seeds("rmsnorm", 2, log=logs.append) == []
    assert logs and "not installed" in logs[0]
