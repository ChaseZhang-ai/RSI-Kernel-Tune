"""``forge init --init-kernel``: the card's kernel file starts from a real implementation.

Uses the tiny SOL dataset fixture from test_leaderboard (no torch, no GPU, no network: the
board's T_b / T_SOL come from a saved record). What is pinned: the bytes land in the card
unchanged, a copy lands in the card's prior/ with its provenance, prior/ is part of the
initial commit (so it is never the student's change), and a file that cannot be the card's
kernel is refused before anything is written."""

from __future__ import annotations

import json
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from rsi_kernel_tune import launch, taskcard
from rsi_kernel_tune.scaffold import main as scaffold_main
from rsi_kernel_tune.task import load_task
from rsi_kernel_tune.warmstart import seed

from test_leaderboard import base, sol_root  # noqa: F401 -- fixtures

# The definition in the sol_root fixture: hidden_states, weight -> output (DPS: 3 params).
KERNEL = ("import torch\n\n\ndef run(hidden_states, weight, output):\n"
          "    output.copy_(hidden_states)   \n")

BOARD = {"id": 77, "name": "001_rms", "tags": ["L1"],
         "workloads": [{"axes": {"batch_size": 4}, "baseline_latency_ms": 0.03,
                        "sol_ms": 0.01},
                       {"axes": {"batch_size": 9}, "baseline_latency_ms": 0.05,
                        "sol_ms": 0.02}]}


@pytest.fixture
def board_file(tmp_path: Path) -> Path:
    p = tmp_path / "board.json"
    p.write_text(json.dumps(BOARD))
    return p


def _init(tmp_path, base, sol_root, board_file, *extra, name="warmop"):
    """``forge init --leaderboard`` with --init-kernel; returns (repo, card dir, kernel)."""
    kernel = tmp_path / "start_kernel.py"
    kernel.write_text(KERNEL)
    rc = scaffold_main([name, "--leaderboard", "sol-execbench", "--dataset", str(sol_root),
                        "--definition", "L1/001_rms", "--workload", "0",
                        "--sol-board", str(board_file), "--base-dir", str(tmp_path / "repo"),
                        "--init-kernel", str(kernel), *extra])
    assert rc == 0
    repo = tmp_path / "repo"
    return repo, repo / "operators" / name, kernel


def test_the_card_kernel_is_the_init_kernel_byte_for_byte(tmp_path, base, sol_root, board_file):
    _, card, kernel = _init(tmp_path, base, sol_root, board_file)
    assert (card / "kernel.py").read_bytes() == kernel.read_bytes()
    problem = json.loads((card / taskcard.PROBLEM).read_text())
    assert problem["kernel_file"] == "kernel.py"


def test_a_copy_lands_in_prior_with_its_provenance(tmp_path, base, sol_root, board_file):
    _, card, kernel = _init(tmp_path, base, sol_root, board_file)
    copy = card / "prior" / kernel.name
    assert copy.read_bytes() == kernel.read_bytes()
    records = seed.read_sources(card)
    assert len(records) == 1
    rec = records[0]
    assert rec["role"] == "init-kernel" and rec["origin"] == str(kernel)
    assert rec["files"][0]["sha256"] == seed.sha256(copy)
    assert rec["license"] == "unknown"           # no LICENSE next to tmp_path/kernel.py


def test_prior_is_part_of_the_initial_commit_not_a_student_change(tmp_path, base, sol_root,
                                                                  board_file):
    _, card, kernel = _init(tmp_path, base, sol_root, board_file)
    tracked = subprocess.run(["git", "-C", str(card), "ls-files"], capture_output=True,
                             text=True, check=True).stdout.split()
    assert f"prior/{kernel.name}" in tracked and "prior/SOURCES.json" in tracked
    assert "kernel.py" in tracked
    # the initial commit holds it, and the tree is clean: nothing the student did
    shown = subprocess.run(["git", "-C", str(card), "show", "HEAD:kernel.py"],
                           capture_output=True, check=True).stdout
    assert shown == kernel.read_bytes()
    status = subprocess.run(["git", "-C", str(card), "status", "--porcelain"],
                            capture_output=True, text=True, check=True).stdout
    assert status.strip() == ""


def test_a_prior_directory_is_copied_with_its_licence(tmp_path, base, sol_root, board_file):
    src = tmp_path / "refimpl"
    src.mkdir()
    (src / "kernel.py").write_text("import torch\n\ndef run(a):\n    return a\n")
    (src / "LICENSE").write_text("MIT License\n\nCopyright (c) 2026 X\n\n"
                                 "Permission is hereby granted, free of charge, to any "
                                 "person obtaining a copy\n")
    _, card, _ = _init(tmp_path, base, sol_root, board_file, "--prior", str(src))
    assert (card / "prior" / "refimpl" / "kernel.py").read_text().startswith("import torch")
    records = {r["path"]: r for r in seed.read_sources(card)}
    assert records["refimpl"]["license"] == "MIT" and records["refimpl"]["kind"] == "tree"


def test_the_language_is_inferred_and_an_unsupported_import_is_refused(tmp_path, base, sol_root,
                                                                      board_file):
    triton = tmp_path / "triton_kernel.py"
    triton.write_text("import triton\nimport triton.language as tl\n\n"
                      "def run(*args):\n    pass\n")
    rc = scaffold_main(["op", "--leaderboard", "sol-execbench", "--dataset", str(sol_root),
                        "--definition", "L1/001_rms", "--workload", "0",
                        "--sol-board", str(board_file), "--base-dir", str(tmp_path / "repo2"),
                        "--init-kernel", str(triton)])
    assert rc == 0                       # a Triton init kernel is a task kernel now (task triton)
    card = tmp_path / "repo2" / "operators" / "op"
    assert (card / "kernel.py").read_bytes() == triton.read_bytes()
    assert (card / "prior" / "triton_kernel.py").is_file()

    cupy = tmp_path / "cupy_kernel.py"
    cupy.write_text("import cupy\n\ndef run(*args):\n    pass\n")
    with pytest.raises(SystemExit, match="cupy"):
        scaffold_main(["op", "--leaderboard", "sol-execbench", "--dataset", str(sol_root),
                       "--definition", "L1/001_rms", "--workload", "0",
                       "--base-dir", str(tmp_path / "repo3"), "--init-kernel", str(cupy)])


def test_an_arity_mismatch_is_refused_with_the_expected_call(tmp_path, base, sol_root,
                                                             board_file):
    kernel = tmp_path / "short.py"
    kernel.write_text("def run(hidden_states, weight):\n    pass\n")
    with pytest.raises(SystemExit) as ei:
        scaffold_main(["op", "--leaderboard", "sol-execbench", "--dataset", str(sol_root),
                       "--definition", "L1/001_rms", "--workload", "0",
                       "--base-dir", str(tmp_path / "repo4"), "--init-kernel", str(kernel)])
    assert "run takes 2 positional parameter(s)" in str(ei.value)
    assert not (tmp_path / "repo4" / "operators" / "op").exists()


def test_a_cuda_init_kernel_becomes_kernel_cu(tmp_path, base, sol_root, board_file):
    cu = tmp_path / "start.cu"
    cu.write_text('#include <torch/extension.h>\n\nvoid run(torch::Tensor hidden_states,'
                  ' torch::Tensor weight, torch::Tensor output) {}\n\n'
                  'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }\n')
    rc = scaffold_main(["cuop", "--leaderboard", "sol-execbench", "--dataset", str(sol_root),
                        "--definition", "L1/001_rms", "--workload", "0",
                        "--sol-board", str(board_file), "--base-dir", str(tmp_path / "repo5"),
                        "--init-kernel", str(cu)])
    assert rc == 0
    card = tmp_path / "repo5" / "operators" / "cuop"
    assert (card / "kernel.cu").read_bytes() == cu.read_bytes()
    problem = json.loads((card / taskcard.PROBLEM).read_text())
    assert problem["kernel_file"] == "kernel.cu"


def test_kernel_file_must_agree_with_the_init_kernel(tmp_path, base, sol_root, board_file):
    kernel = tmp_path / "start.py"
    kernel.write_text(KERNEL)
    with pytest.raises(SystemExit, match="must be the card's kernel.py"):
        scaffold_main(["op", "--leaderboard", "sol-execbench", "--dataset", str(sol_root),
                       "--definition", "L1/001_rms", "--workload", "0",
                       "--base-dir", str(tmp_path / "repo6"), "--init-kernel", str(kernel),
                       "--kernel-file", "kernel.cu"])


def test_a_missing_prior_is_refused(tmp_path, base, sol_root):
    with pytest.raises(SystemExit, match="no such file or directory"):
        scaffold_main(["op", "--prior", str(tmp_path / "nope"),
                       "--base-dir", str(tmp_path / "repo7"), "--no-git"])


def test_a_plain_operator_gets_the_same_kernel_and_prior(tmp_path):
    kernel = tmp_path / "start.py"
    kernel.write_text("import torch\n\ndef run(x):\n    return x\n")
    rc = scaffold_main(["plainop", "--base-dir", str(tmp_path / "repo8"), "--no-git",
                        "--init-kernel", str(kernel), "--kernel-file", "plainop.py"])
    assert rc == 0
    op = tmp_path / "repo8" / "operators" / "plainop"
    assert (op / "plainop.py").read_bytes() == kernel.read_bytes()
    assert (op / "prior" / "start.py").read_bytes() == kernel.read_bytes()


def test_priors_alone_keep_the_generated_stub(tmp_path, base, sol_root, board_file):
    """--prior without --init-kernel: the reference library rides along, the stub stays."""
    src = tmp_path / "lib"
    src.mkdir()
    (src / "alt.py").write_text("import torch\n\ndef run(a):\n    return a\n")
    rc = scaffold_main(["op", "--leaderboard", "sol-execbench", "--dataset", str(sol_root),
                        "--definition", "L1/001_rms", "--workload", "0",
                        "--sol-board", str(board_file), "--base-dir", str(tmp_path / "repo9"),
                        "--prior", str(src)])
    assert rc == 0
    card = tmp_path / "repo9" / "operators" / "op"
    assert "NotImplementedError" in (card / "kernel.py").read_text()
    assert (card / "prior" / "lib" / "alt.py").read_text().startswith("import torch")


def test_the_loop_toml_carries_the_warmstart_context(tmp_path, base, sol_root, board_file,
                                                     monkeypatch):
    repo, card, _ = _init(tmp_path, base, sol_root, board_file)
    task_dir = repo / "tasks" / "warmop"
    (task_dir / "leaderboard").mkdir(parents=True)
    shutil.copytree(card, task_dir / "leaderboard" / "card", dirs_exist_ok=True)
    (task_dir / "task.toml").write_text(
        '[task]\nlanguage = "cutedsl"\narch = "sm100"\n\n'
        '[leaderboard]\nsuite = "sol-execbench"\ncard = "leaderboard/card"\n')
    (task_dir / "prior.md").write_text("# manifest\n")
    task = load_task(task_dir)
    # `local_knowledge_dirs()` reads `_reference/` beside the checkout, which a fresh clone
    # does not have: pin both branches instead of depending on this machine's tree.
    knowledge = tmp_path / "reference" / "KernelWiki"
    knowledge.mkdir(parents=True)

    def loop_cfg():
        return tomllib.loads(taskcard.write_board_loop_toml(
            None, task, task_dir / "leaderboard" / "card", deny=[],
            ncu=False).read_text())["loop"]

    monkeypatch.setattr(taskcard, "local_knowledge_dirs", lambda: [knowledge])
    cfg = loop_cfg()
    assert cfg["student_online"] is True
    assert cfg["prior_dir"] == str(task_dir / "leaderboard" / "card" / "prior")
    assert cfg["prior_manifest"] == str(task_dir / "prior.md")
    assert cfg["local_knowledge_dirs"] == [str(knowledge)]
    monkeypatch.setattr(taskcard, "local_knowledge_dirs", lambda: [])
    assert "local_knowledge_dirs" not in loop_cfg()      # no tree: the key is simply absent
    lines = taskcard.context_lines(card, task)
    assert "student_online = true" in lines
    assert "[online]" in launch.describe(task, _Machine())


def test_forge_init_leaderboard_carries_the_warmstart_into_the_task(tmp_path, base, sol_root,
                                                                    board_file, monkeypatch):
    """The whole `forge init --leaderboard --init-kernel` path (launch.new_board_task): the
    task it writes names the inferred language, and its frozen card carries the kernel."""
    repo = tmp_path / "tasksrepo"
    repo.mkdir()
    monkeypatch.setattr(launch, "repo_root", lambda: repo)
    kernel = tmp_path / "start_kernel.py"
    kernel.write_text(KERNEL)
    rc = launch.new_board_task(["warmop", "--leaderboard", "sol-execbench",
                                "--dataset", str(sol_root), "--definition", "L1/001_rms",
                                "--workload", "0", "--sol-board", str(board_file),
                                "--init-kernel", str(kernel)])
    assert rc == 0
    task_dir = repo / "tasks" / "warmop"
    cfg = tomllib.loads((task_dir / "task.toml").read_text())
    assert cfg["task"]["language"] == "cutedsl"
    card = task_dir / "leaderboard" / "card"
    assert (card / "kernel.py").read_bytes() == kernel.read_bytes()
    assert (card / "prior" / kernel.name).is_file()
    loaded = load_task(task_dir)
    assert loaded.language == "cutedsl" and loaded.student.online is True


class _Machine:
    def describe(self) -> str:
        return "a fake GPU"
