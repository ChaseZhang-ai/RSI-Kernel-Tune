"""Regression tests for the audit-corps R2 guard + benchkit fix cluster.

One section per ledger bug:

* BUG-SEED-07: a tree-root python module a harness process executes without the tree
  naming it (sitecustomize.py, a stdlib shadow like json.py) is harness -- the job
  overlay drops it, the delivery diff names it, the hygiene drift sees it
  (loop/harness_guard.py).
* BUG-SEED-08: the goal-ref library scan reads every name of an import statement and
  the string literals of importlib.import_module / __import__ (loop/goal_ref.py).
* BUG-C3-01: the torch C++ call forms (torch:: / at:: / c10::) are library findings
  on .cu / .cpp sources (loop/goal_ref.py).
* BUG-SEED-09: load_entry APPENDS the kernel's directory to sys.path, so the stdlib
  and the already-loaded reference are not shadowed (card_template/eval/benchkit.py).
* BUG-C5-02: a Python kernel that dies at import is a structured IMPORT_ERROR verdict,
  not a bare crash of the bench process (card_template/eval/benchkit.py).
* BUG-C5-03: an empty input_groups workload is an INPUT_ERROR, never a fake PASS, and
  the NCU window refuses it instead of dividing by zero (card_template/eval/benchkit.py).

CPU only where a GPU would be needed; the harness tests shell out to git.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "src" / "peregrine_kernel" / "card_template" / "eval"


def _load_eval(name: str):
    """The card-template eval module under its own name (benchkit.py is copied into
    cards and loaded by path there too, so tests do the same)."""
    spec = importlib.util.spec_from_file_location(f"pk_guard_fixes_{name}",
                                                  EVAL_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from peregrine_kernel.loop import cli, harness_guard  # noqa: E402
from peregrine_kernel.loop.goal_ref import library_findings  # noqa: E402

lb = _load_eval("benchkit")


# --- BUG-SEED-07: a tree-root module the harness imports without naming it is harness --

def test_a_root_import_hijack_is_harness_and_the_kernel_is_not():
    # the demonstrated attacks: the interpreter's startup hook, a stdlib shadow
    assert harness_guard.is_harness_path("sitecustomize.py", "kernel.py")
    assert harness_guard.is_harness_path("usercustomize.py", "kernel.py")
    assert harness_guard.is_harness_path("./sitecustomize.py", "kernel.py")
    assert harness_guard.is_harness_path("json.py", "kernel.py")
    assert harness_guard.is_harness_path("csv.py", "kernel.py")
    assert harness_guard.is_harness_path("os.py", "kernel.py")
    # the kernel file and the old rules stay
    assert not harness_guard.is_harness_path("kernel.py", "kernel.py")
    # the default kernel_file="" keeps the existing convention: the kernel file itself
    # is not harness (nothing imports it implicitly), but a root stdlib shadow is
    assert not harness_guard.is_harness_path("kernel.py")
    assert harness_guard.is_harness_path("time.py")
    assert not harness_guard.is_harness_path("eval/kernel.py", "eval/kernel.py")
    # deeper paths are the student's, whatever the name
    assert not harness_guard.is_harness_path("deep/dir/module.py", "kernel.py")
    assert not harness_guard.is_harness_path("notes/json.py", "kernel.py")
    # a root module nothing imports implicitly (a helper the tree names itself) is not
    assert not harness_guard.is_harness_path("newmod.py", "kernel.py")
    assert not harness_guard.is_harness_path("gpu_entry.sh.bak", "kernel.py")
    assert not harness_guard.is_harness_path("window.toml", "kernel.py")
    # the harness prefixes still win, at depth too
    assert harness_guard.is_harness_path("eval/sitecustomize.py", "kernel.py")


def test_git_pathspecs_carry_the_root_module_glob():
    # git ls-files / ls-tree / diff (the listings, the overlay and the delivery and
    # hygiene checks in cli.py) take the wildcard; is_harness_path narrows the depth
    # and the name afterwards.
    assert harness_guard.ROOT_PY_PATHSPEC == "*.py"
    assert "*.py" in harness_guard.HARNESS_PATHSPECS
    assert tuple(p.rstrip("/") for p in harness_guard.HARNESS_PREFIXES) \
        == harness_guard.HARNESS_PATHSPECS[:-1]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, msg: str) -> str:
    _git(repo, "add", "-A")
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-m", msg], check=True)
    return _git(repo, "rev-parse", "HEAD")


def test_tree_listings_carry_root_hijack_modules(tmp_path):
    """tree_harness_files feeds the overlay drop (cli.py _overlay_wanted) and the
    hygiene drift: a root sitecustomize.py / json.py in a job tree must be IN it."""
    tree = tmp_path / "tree"
    (tree / "eval").mkdir(parents=True)
    (tree / "kernel.py").write_text("def run(): pass\n")
    (tree / "eval/bench.py").write_text("print('judge')\n")
    (tree / "sitecustomize.py").write_text("import os\nos.environ['HACK'] = '1'\n")
    (tree / "json.py").write_text("loads = lambda *a: {'pass': True}\n")
    (tree / "deep").mkdir()
    (tree / "deep/lib.py").write_text("x = 1\n")          # not root: not harness
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    head = _commit(tree, "seed")
    want = {"eval/bench.py", "sitecustomize.py", "json.py"}
    assert set(harness_guard.tree_harness_files(tree, head, "kernel.py")) == want
    assert set(harness_guard.harness_only(["sitecustomize.py", "json.py", "newmod.py",
                                           "deep/lib.py"], "kernel.py")) == want - {"eval/bench.py"}
