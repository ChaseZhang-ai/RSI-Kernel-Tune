"""scripts/self_evolve.py: each generation's best kernel becomes the next one's reference."""

from __future__ import annotations

import importlib.util
import math
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from peregrine_kernel.task import dump_task_toml, load_task, read_task_toml

from test_task import _kb

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("self_evolve", REPO / "scripts" / "self_evolve.py")
se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(se)

KERNEL = "import torch\n\ndef run(x, w):\n    return x * w\n"


def _task(root: Path, name="rmsnorm") -> Path:
    d = root / "tasks" / name
    d.mkdir(parents=True)
    (d / "task.toml").write_text('[task]\nlanguage = "cutedsl"\narch = "sm100"\n'
                                 '[agent.teacher]\nmodel = "claude-opus-5-5"\neffort = "high"\n'
                                 '[budget]\nhours = 2.5\n')
    (d / "description.md").write_text("y = x * w\n")
    (d / "shapes.json").write_text('[{"name": "s", "axes": {"m": 4}}]')
    return d


@pytest.fixture
def chain(tmp_path, monkeypatch):
    """Generations stood in for: each 'run' reports the next scripted speedup."""
    kb = _kb(tmp_path)
    monkeypatch.setattr(se, "load_task", lambda d: load_task(d, kb_root=kb))
    monkeypatch.setattr(se.launch, "repo_root", lambda: tmp_path)
    ran: list[Path] = []
    gains: list = []
    monkeypatch.setattr(se.launch, "cmd_new", lambda argv: ran.append(Path(argv[0])) or 0)
    monkeypatch.setattr(se.launch, "cmd_run", lambda argv: 0)
    monkeypatch.setattr(se, "latest_run", lambda name: SimpleNamespace(
        dir=tmp_path / "run", operators_dir=tmp_path / "run" / "operators",
        ws=tmp_path / "run" / "ws"))
    # the carry ledger lookup stays hermetic: no repo hw_kb in a chain test
    monkeypatch.setattr(se, "REPO_ROOT", tmp_path)

    def best_of(run, task):
        g = gains[len(ran) - 1]
        if g is None:
            return None
        speedup, ratios = g if isinstance(g, tuple) else (g, [])
        return {"sha": "abc123def456", "speedup": speedup, "ratios": ratios, "kernel": KERNEL}
    monkeypatch.setattr(se.launch, "best_of", best_of)
    orig = se.next_generation
    monkeypatch.setattr(se, "next_generation",
                        lambda task, best, dest, step, **kw: orig(
                            task, best, dest, step, **kw) and dest)
    return tmp_path, ran, gains, kb


def test_dump_task_toml_round_trips(tmp_path):
    d = _task(tmp_path)
    cfg = read_task_toml(d)
    assert tomllib.loads(dump_task_toml(cfg)) == cfg


def test_a_gain_writes_the_next_generation_and_runs_it(chain):
    root, ran, gains, kb = chain
    gains += [1.30, 1.01]
    first = _task(root)
    assert se.main([str(first), "--max-gen", "4"]) == 0
    g2 = root / "tasks" / "rmsnorm-g2"
    assert ran == [first.resolve(), g2]
    assert (g2 / "reference" / "kernel.py").read_text() == KERNEL
    t2 = load_task(g2, kb_root=kb)
    assert t2.mode == "source" and t2.target.ratio == 1.05
    assert t2.reference.run_fn == "run" and t2.budget.hours == 2.5
    assert t2.teacher.effort == "high"
    assert not (root / "tasks" / "rmsnorm-g3").exists()


def test_the_chain_stops_at_max_gen(chain):
    root, ran, gains, _ = chain
    gains += [1.5, 1.5, 1.5]
    assert se.main([str(_task(root)), "--max-gen", "2"]) == 0
    assert [p.name for p in ran] == ["rmsnorm", "rmsnorm-g2"]


def test_there_is_no_generation_cap_by_default(chain):
    root, ran, gains, _ = chain
    gains += [1.2] * 6 + [1.005]
    assert se.main([str(_task(root))]) == 0
    assert [p.name for p in ran] == ["rmsnorm"] + [f"rmsnorm-g{k}" for k in range(2, 8)]


def test_the_chain_stops_at_the_noise_floor(chain):
    root, ran, gains, _ = chain
    # 1.03x clears the 1.5% minimum, but the generation's own same-job ratios spread 4%
    gains += [(1.03, [1.01, 1.03, 1.0504])]
    assert se.main([str(_task(root))]) == 0
    assert len(ran) == 1
    assert not (root / "tasks" / "rmsnorm-g2").exists()


def test_noise_floor():
    assert se.noise_floor({"ratios": []}) == se.NOISE_FLOOR_MIN
    assert se.noise_floor({"ratios": [1.10, 1.101]}) == se.NOISE_FLOOR_MIN
    assert se.noise_floor({"ratios": [1.0, 1.05, 1.02]}) == pytest.approx(0.05)


def test_noise_floor_with_a_zero_ratio_stops_the_chain_instead_of_crashing():
    # BUG-SEED-27: a zero ratio is nonsense data (a timed kernel never takes 0 ms);
    # the old (max - min) / min raised ZeroDivisionError on it. The floor is now
    # infinite — meaningless data must stop the chain, never wave it through.
    assert se.noise_floor({"ratios": [1.10, 0.0]}) == math.inf
    assert se.noise_floor({"ratios": [0.0, 1.10]}) == math.inf


def test_a_generation_without_a_best_ends_the_chain(chain):
    root, ran, gains, _ = chain
    gains += [None]
    assert se.main([str(_task(root))]) == 1
    assert len(ran) == 1


def test_an_existing_next_generation_is_not_overwritten(chain, capsys):
    # BUG-SEED-27: a collision with an existing generation directory is now a
    # readable error on stderr and exit 2. The old assertion pinned the bare
    # FileExistsError traceback itself — rewritten for the new semantics.
    root, ran, gains, _ = chain
    gains += [1.3]
    first = _task(root)
    (root / "tasks" / "rmsnorm-g2").mkdir()
    assert se.main([str(first)]) == 2
    assert not (root / "tasks" / "rmsnorm-g2" / "task.toml").exists()
    assert "exists" in capsys.readouterr().err


def test_a_chain_resumes_from_a_mid_generation_directory(chain):
    # BUG-SEED-27: starting the chain from tasks/rmsnorm-g2 used to build the
    # first child as rmsnorm-g2 — the directory the run itself lives in — and die
    # on FileExistsError. The numbering continues from where you start: the child
    # of -g2 is -g3.
    root, ran, gains, _ = chain
    gains += [1.3, 1.01]
    start = _task(root, name="rmsnorm-g2")
    assert se.main([str(start)]) == 0
    g3 = root / "tasks" / "rmsnorm-g3"
    assert ran == [start.resolve(), g3]
    assert (g3 / "reference" / "kernel.py").read_text() == KERNEL
    assert not (root / "tasks" / "rmsnorm-g4").exists()
