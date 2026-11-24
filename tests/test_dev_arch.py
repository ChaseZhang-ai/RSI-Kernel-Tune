"""dev_arch: the card compiles and runs for the development GPU, the export for the board.

Before `dev_arch` existed, every card named one architecture (the board's) in three places:
the CuTe DSL arch, the CUDA gencode and the recon tree's reference build. On a development
machine of another compute capability that card cannot build anything. These tests pin both
sides: a card with `dev_arch` compiles for it, and a card without one is what it always was.
"""

from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from forgekernel import leaderboard as lb
from forgekernel import taskcard
from forgekernel.localenv import load_machine
from forgekernel.task import TaskError, load_task
from forgekernel.targethw import arch_of_cc, cc_of_arch, cuda_arch, device_arch, gencode
from forgekernel.targethw import target_sms

from test_launch import _run_with_check, repo  # noqa: F401  (repo is the fixture)

REPO = Path(__file__).resolve().parents[1]
BASE_TOML = '[task]\nlanguage = "cutedsl"\narch = "sm100"\n'
DEV_TOML = '[task]\nlanguage = "cutedsl"\narch = "sm100"\ndev_arch = "sm103"\n'
REF_TOML = DEV_TOML + '[reference]\nrun = "reference/run.py:run"\nsource = "reference"\n'
PLAIN_REF_TOML = BASE_TOML + '[reference]\nrun = "reference/run.py:run"\nsource = "reference"\n'


# -- the helpers ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, cc", [
    ("sm103", (10, 3)), ("sm_103a", (10, 3)), ("103", (10, 3)), ("sm100", (10, 0)),
    ("sm90", (9, 0)), ("sm_86", (8, 6)),
])
def test_arch_spellings_map_to_a_compute_capability(text, cc):
    assert cc_of_arch(text) == cc


def test_arch_spellings_are_normalised():
    assert arch_of_cc(10, 3) == "sm103" and arch_of_cc(8, 6) == "sm86"
    assert cuda_arch("sm103") == "sm_103a" and cuda_arch("sm100") == "sm_100a"
    assert cuda_arch("sm86") == "sm_86"          # no accelerated variant
    assert gencode("sm103") == "-gencode=arch=compute_103a,code=sm_103a"
    assert gencode("sm100") == "-gencode=arch=compute_100a,code=sm_100a"


def test_a_non_architecture_is_refused():
    for bad in ("", "sm", "turing", "sm100x", "gpu"):
        with pytest.raises(Exception):
            cc_of_arch(bad)


def test_fk_target_sms_is_an_optional_override():
    assert target_sms({"FK_TARGET_SMS": "148"}) == 148
    assert target_sms({}) is None
    assert target_sms({"FK_TARGET_SMS": "nope"}) is None
    assert target_sms({"FK_TARGET_SMS": "-3"}) is None


def test_device_arch_without_a_device_is_none(monkeypatch):
    import forgekernel.targethw as th

    assert device_arch() is None or isinstance(device_arch(), str)
    class Boom:
        def __getattr__(self, name):
            raise RuntimeError("no driver")
    monkeypatch.setitem(__import__("sys").modules, "torch", Boom())
    assert th.device_arch() is None


# -- task.toml ---------------------------------------------------------------------------

def test_dev_arch_is_read_and_needs_no_knowledge_base(tmp_path):
    d = tmp_path / "tasks" / "op"
    (d / "reference").mkdir(parents=True)
    (d / "task.toml").write_text(REF_TOML)
    (d / "description.md").write_text("y = x\n")
    (d / "shapes.json").write_text(json.dumps([{"name": "s", "axes": {"m": 4}}]))
    (d / "reference" / "run.py").write_text("def run(x):\n    return x\n")
    task = load_task(d)
    assert task.arch == "sm100" and task.dev_arch == "sm103"
    assert task.compiled_arch == "sm103"
    assert not (REPO / "hw_kb" / "cutedsl" / "sm103").is_dir()      # no KB, still valid
    assert any("sm103" in note for note in task.notes)


def test_without_dev_arch_the_compiled_arch_is_the_graded_one():
    task = load_task(_task_dir(PLAIN_REF_TOML, name="op"))
    assert task.dev_arch is None and task.compiled_arch == task.arch == "sm100"


@pytest.mark.parametrize("value", ["103", "", "turing"])
def test_a_bad_dev_arch_is_refused(tmp_path, value):
    d = _task_dir(BASE_TOML.replace('arch = "sm100"',
                                    f'arch = "sm100"\ndev_arch = "{value}"'))
    with pytest.raises(TaskError) as err:
        load_task(d)
    assert any("dev_arch" in p for p in err.value.problems)


@pytest.mark.parametrize("value", ["sm103", "sm_103a"])
def test_both_arch_spellings_are_accepted_and_normalised(tmp_path, value):
    d = _task_dir(BASE_TOML.replace('arch = "sm100"',
                                    f'arch = "sm100"\ndev_arch = "{value}"'))
    task = load_task(d)
    assert task.compiled_arch == "sm_103a" or taskcard._cute_arch(task.compiled_arch) == "sm_103a"


def test_a_task_may_not_be_named_after_an_extension_command(tmp_path):
    d = _task_dir(PLAIN_REF_TOML, name="target")
    with pytest.raises(TaskError) as err:
        load_task(d)
    assert any("extension command" in p for p in err.value.problems)


def _task_dir(toml: str, name: str = "op") -> Path:
    import tempfile

    base = Path(tempfile.mkdtemp(prefix="fk_devarch_"))
    d = base / "tasks" / name
    (d / "reference").mkdir(parents=True)
    (d / "task.toml").write_text(toml)
    (d / "description.md").write_text("y = x\n")
    (d / "shapes.json").write_text(json.dumps([{"name": "s", "axes": {"m": 4}}]))
    (d / "reference" / "run.py").write_text("def run(x):\n    return x\n")
    return d


# -- the card ------------------------------------------------------------------------------

def test_a_card_with_dev_arch_compiles_for_the_development_gpu(repo):
    run, task, check = _run_with_check(repo, toml=REF_TOML)
    op_dir = taskcard.write_card(run, task, check, load_machine(repo, environ={}))
    entry = (op_dir / "gpu_entry.sh").read_text()
    assert 'CUTE_DSL_ARCH:-sm_103a' in entry
    assert 'FK_CUDA_ARCH:-sm_103a' in entry
    submitted = tomllib.loads((op_dir / "submit.toml").read_text())
    assert submitted["env"]["CUTE_DSL_ARCH"] == "sm_103a"
    assert submitted["env"]["FK_CUDA_ARCH"] == "sm_103a"
    # the graded architecture still picks the knowledge base
    taskcard.write_loop_toml(run, task, op_dir, deny=[])
    loop_toml = tomllib.loads((op_dir / "loop.toml").read_text())
    assert loop_toml["loop"]["kb_arch"] == "sm100"


def test_a_card_without_dev_arch_is_what_it_always_was(repo):
    run, task, check = _run_with_check(repo, toml=PLAIN_REF_TOML)
    op_dir = taskcard.write_card(run, task, check, load_machine(repo, environ={}))
    entry = (op_dir / "gpu_entry.sh").read_text()
    assert 'CUTE_DSL_ARCH:-sm_100a' in entry
    assert "FK_CUDA_ARCH" not in entry              # the byte-level promise of "no dev_arch"
    submitted = tomllib.loads((op_dir / "submit.toml").read_text())
    assert submitted["env"]["CUTE_DSL_ARCH"] == "sm_100a"
    assert "FK_CUDA_ARCH" not in submitted["env"]


def test_a_leaderboard_card_takes_the_dev_arch_too(tmp_path):
    problem = SimpleNamespace(name="p", suite=lb.SUITE_SOL)
    board = lb.render_gpu_entry(problem)
    dev = lb.render_gpu_entry(problem, dev_arch="sm103")
    assert 'CUTE_DSL_ARCH:-sm_100a' in board and "FK_CUDA_ARCH" not in board
    assert 'CUTE_DSL_ARCH:-sm_103a' in dev and 'FK_CUDA_ARCH:-sm_103a' in dev
    assert board.replace('"${CUTE_DSL_ARCH:-sm_100a}"', "") != dev


def test_the_card_template_builds_cuda_for_the_machine_it_runs_on(monkeypatch):
    card = _card_module()
    src = "int x;\n"
    monkeypatch.delenv("FK_CUDA_ARCH", raising=False)
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    assert card.cuda_kernel_flags(src)[-1] == card.CUDA_GENCODE == lb.CUDA_GENCODE
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_103a")
    assert card.cuda_kernel_flags(src)[-1] == "-gencode=arch=compute_103a,code=sm_103a"
    monkeypatch.setenv("FK_CUDA_ARCH", "sm_100a")
    assert card.cuda_kernel_flags(src)[-1] == "-gencode=arch=compute_100a,code=sm_100a"
    # a source's own arch flag wins, as the board's rules say
    assert card.cuda_kernel_flags("// fk-cuda-cflags: -arch=sm_90a\n")[-1] == "-arch=sm_90a"


def test_the_main_and_card_flag_builders_agree(monkeypatch):
    card = _card_module()
    src = "// fk-cuda-cflags: -Xptxas=-v\nint x;\n"
    for env in ({}, {"CUTE_DSL_ARCH": "sm_103a"}, {"FK_CUDA_ARCH": "sm_103a"}):
        for key in ("FK_CUDA_ARCH", "CUTE_DSL_ARCH"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        assert lb.cuda_kernel_flags(src) == card.cuda_kernel_flags(src)
    assert lb.CUDA_GENCODE == card.CUDA_GENCODE
    assert lb.CUDA_DEFAULT_CFLAGS == card.CUDA_DEFAULT_CFLAGS


def test_the_export_is_always_the_boards_architecture(monkeypatch):
    """A development machine's FK_CUDA_ARCH must not leak into the board's solution JSON."""
    monkeypatch.setenv("FK_CUDA_ARCH", "sm_103a")
    src = "int x;\n"
    assert lb.cuda_kernel_flags(src)[-1].endswith("sm_103a")          # the card's build
    assert lb.cuda_kernel_flags(src, gencode=lb.CUDA_GENCODE)[-1] == lb.CUDA_GENCODE


def test_the_recon_tree_builds_for_the_machine_it_runs_on(monkeypatch, tmp_path):
    """The binref tool is a card file (it reads its own binref.json when imported)."""
    binref = _binref_module(tmp_path)
    monkeypatch.delenv("FK_CUDA_ARCH", raising=False)
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    assert binref.device_gencode() == binref.BOARD_GENCODE
    assert binref.device_gencode({"FK_CUDA_ARCH": "sm_103a"}) \
        == "-gencode=arch=compute_103a,code=sm_103a"
    assert binref.device_gencode({"CUTE_DSL_ARCH": "sm_90a"}) \
        == "-gencode=arch=compute_90a,code=sm_90a"
    assert binref.device_gencode({"FK_CUDA_ARCH": "nonsense"}) == binref.BOARD_GENCODE


def _binref_module(tmp_path: Path):
    import shutil

    work = tmp_path / "binref_load"
    work.mkdir()
    shutil.copy2(REPO / "src" / "forgekernel" / "binref_tool.py", work / "binref.py")
    (work / "binref.json").write_text(json.dumps({"module": "ref", "sources": []}))
    spec = importlib.util.spec_from_file_location("_test_binref_tool", work / "binref.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _card_module():
    path = REPO / "src" / "forgekernel" / "card_template" / "eval" / "leaderboard.py"
    spec = importlib.util.spec_from_file_location("_test_card_lb_devarch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
