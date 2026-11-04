"""What the README points a new user at: the shipped tasks and the top-level CLI."""

from __future__ import annotations

from pathlib import Path

from peregrine_kernel import __main__ as forge_main
from peregrine_kernel import __version__
from peregrine_kernel.task import load_task

ROOT = Path(__file__).resolve().parents[1]


def test_the_example_task_loads():
    task = load_task(ROOT / "tasks" / "rmsnorm")
    assert task.mode == "source" and task.language == "cutedsl"
    assert [s.name for s in task.shapes] == ["prefill", "decode"]
    assert (ROOT / "tasks" / "rmsnorm" / "reference" / "run.py").is_file()
    assert (task.budget.hours, task.budget.max_meta) == (2.0, 1)


def test_the_task_template_is_a_valid_task_but_for_its_name():
    task = load_task(ROOT / "tasks" / "_template", name="template")
    assert task.mode == "generate" and task.target.ratio == 1.2


def test_forge_alone_prints_its_usage_and_version(capsys):
    assert forge_main.main([]) == 0
    out = capsys.readouterr().out
    assert "peregrine new tasks/<op>" in out and "peregrine run .artifacts/<op>_<id>" in out
    assert forge_main.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"peregrine_kernel {__version__}"
