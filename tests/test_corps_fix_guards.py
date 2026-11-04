"""Regression tests for the audit-corps fix wave 3 (the guard bypass surface and the
kernel loading path): tree-root shadow modules, the correctness monitor's hook
precedence on PYTHONPATH, the multi-name and dynamic import scan, the C++ torch API
spellings, and the kernel directory's position on sys.path."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_harness_guard import _backend, _card, _commit, _git, _tree  # noqa: E402

from peregrine_kernel.loop import cli, harness_guard  # noqa: E402


# --- BUG-SEED-07: tree-root shadow modules are harness --------------------------------

@pytest.mark.parametrize("path", ["sitecustomize.py", "usercustomize.py", "json.py",
                                  "os.py", "time.py", "subprocess.py", "random.py",
                                  "torch.py", "numpy.py", "triton.py"])
def test_root_shadow_modules_are_harness(path):
    """A root file named like a module Python or the judge imports is imported INSTEAD
    of that module by every process that reaches the tree root first on sys.path."""
    assert harness_guard.is_harness_path(path, "kernel.py") is True


@pytest.mark.parametrize("path", ["utils.py", "helpers.py", "evaluation.py", "my_ops.py",
                                  "kernel.py", "kern/json.py", "json.py.bak",
                                  "notes/json.py", "window.toml", "README.md"])
def test_the_students_own_root_files_are_not_harness(path):
    """The kernel may import root siblings by name: only the names Python itself or the
    judge imports are harness, and only at the tree root."""
    assert harness_guard.is_harness_path(path, "kernel.py") is False


def test_the_kernel_file_is_never_a_shadow_even_when_its_name_is():
    assert not harness_guard.is_harness_path("json.py", "json.py")


def test_tree_harness_files_lists_root_shadow_modules(tmp_path):
    card, ws = _card(tmp_path)
    (ws / "sitecustomize.py").write_text("STUDENT\n")
    (ws / "json.py").write_text("STUDENT\n")
    head = _commit(ws, "shadow")
    assert harness_guard.tree_harness_files(ws, head, "kernel.py") is not None
    tree = harness_guard.tree_harness_files(ws, head, "kernel.py")
    assert "sitecustomize.py" in tree and "json.py" in tree
    assert "kernel.py" not in tree                     # the kernel file is not harness


def test_a_delivery_adding_root_shadows_is_caught_and_dropped_from_jobs(tmp_path):
    """The delivery guard and the job overlay must see the shadow modules the same way
    they see an edited eval/ -- and leave the student's own root files alone."""
    card, ws = _card(tmp_path)
    be = _backend(ws, card)
    base = be.head()
    (ws / "sitecustomize.py").write_text("import os; os.environ['STUDENT'] = '1'\n")
    (ws / "json.py").write_text("STUDENT SHADOW\n")
    (ws / "utils.py").write_text("# the student's own root sibling\n")
    (ws / "kernel.py").write_text("def run(): return 1\n")
    head = _commit(ws, "shadow")
    assert be.harness_changes(base, head) == ["json.py", "sitecustomize.py"]
    job = be._with_job_env(head)
    tree = _tree(ws, job)
    assert "json.py" not in tree and "sitecustomize.py" not in tree
    assert "kernel.py" in tree and "utils.py" in tree
    assert _git(ws, "rev-parse", "HEAD") == head        # the lineage itself is untouched


def test_untracked_root_shadows_show_as_working_tree_drift(tmp_path):
    card, ws = _card(tmp_path)
    be = _backend(ws, card)
    anchor = be.head()
    (ws / "sitecustomize.py").write_text("STUDENT\n")
    (ws / "json.py").write_text("STUDENT\n")
    assert be.harness_drift(anchor) == ["json.py", "sitecustomize.py"]


def test_preflight_refuses_a_fresh_start_over_root_shadows(tmp_path, capsys):
    card, ws = _card(tmp_path)
    (ws / "sitecustomize.py").write_text("STUDENT\n")
    (ws / "json.py").write_text("STUDENT\n")
    _commit(ws, "shadow")
    be = _backend(ws)
    rc = cli._harness_guard_preflight(card, ws, "kernel.py", be, resuming=False,
                                      dry_run=False)
    assert rc == cli.EXIT_HARNESS_DRIFT
    assert "-sitecustomize.py" in capsys.readouterr().err


def test_preflight_accepts_a_workspace_with_harmless_root_files(tmp_path, capsys):
    card, ws = _card(tmp_path)
    (ws / "utils.py").write_text("# the student's own root sibling\n")
    _commit(ws, "sibling")
    be = _backend(ws)
    assert cli._harness_guard_preflight(card, ws, "kernel.py", be, resuming=False,
                                        dry_run=False) == 0
    assert "identical to the workspace" in capsys.readouterr().out
