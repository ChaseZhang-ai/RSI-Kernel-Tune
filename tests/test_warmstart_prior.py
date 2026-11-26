"""``forge prior build``: the run-context manifest (rsi_kernel_tune.warmstart.build_prior).

The manifest is the only agent-facing document this task adds, and every line of it is
derived from files that already exist: the card's problem.json, the prior/ record the card
carries, the reference-library index and the local knowledge trees. These tests pin that
derivation -- including the licence rule (only Apache-2.0 / MIT / BSD rows are usable) and
that the file reaches a prompt through ``{PRIOR_PATH}``."""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path

import pytest

from rsi_kernel_tune.ext import prior as prior_cli
from rsi_kernel_tune.loop.driver import render_prompt
from rsi_kernel_tune.scaffold import main as scaffold_main
from rsi_kernel_tune.warmstart import seed

from test_leaderboard import base, sol_root  # noqa: F401 -- fixtures

# The manifest builder by module (the package re-exports its build_prior function too).
bp = importlib.import_module("rsi_kernel_tune.warmstart.build_prior")

MIT = ("MIT License\n\nCopyright (c) 2026 A\n\nPermission is hereby granted, free of "
       "charge, to any person obtaining a copy of this software\n")
APACHE = ("Apache License\nVersion 2.0, January 2004\nhttp://www.apache.org/licenses/\n")
GPL = "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n"

BOARD = {"id": 77, "name": "001_rms", "tags": ["L1"],
         "workloads": [{"axes": {"batch_size": 4}, "baseline_latency_ms": 0.03,
                        "sol_ms": 0.01},
                       {"axes": {"batch_size": 9}, "baseline_latency_ms": 0.05,
                        "sol_ms": 0.02}]}

KERNEL = "import torch\n\n\ndef run(hidden_states, weight, output):\n    output.copy_(hidden_states)\n"


def _licensed(tmp_path: Path, name: str, text: str, source: str) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "kernel.py").write_text(source)
    (d / "LICENSE").write_text(text)
    return d


@pytest.fixture
def warm(tmp_path: Path, base, sol_root):
    """A card made by ``forge init --init-kernel --prior`` plus a task directory naming it:
    an Apache-2.0 starting kernel, an MIT implementation, a GPL one."""
    start = _licensed(tmp_path, "start", APACHE,
                      "import torch\n\n\ndef run(hidden_states, weight, output):\n"
                      "    output.copy_(hidden_states)\n")
    mit = _licensed(tmp_path, "mit_impl", MIT,
                    "import torch\n\ndef run(hidden_states, weight, output):\n"
                    "    output.copy_(hidden_states * 2)\n")
    gpl = _licensed(tmp_path, "gpl_impl", GPL,
                    "import torch\n\ndef run(hidden_states, weight, output):\n"
                    "    output.copy_(hidden_states)\n")
    board = tmp_path / "board.json"
    board.write_text(json.dumps(BOARD))
    repo = tmp_path / "repo"
    rc = scaffold_main(["warmop", "--leaderboard", "sol-execbench",
                        "--dataset", str(sol_root), "--definition", "L1/001_rms",
                        "--workload", "0", "--sol-board", str(board),
                        "--base-dir", str(repo),
                        "--init-kernel", str(start / "kernel.py"),
                        "--prior", str(mit / "kernel.py"), "--prior", str(gpl / "kernel.py")])
    assert rc == 0
    task_dir = repo / "tasks" / "warmop"
    task_dir.mkdir(parents=True)
    shutil.copytree(repo / "operators" / "warmop", task_dir / "leaderboard" / "card")
    (task_dir / "task.toml").write_text(
        '[task]\nlanguage = "cutedsl"\narch = "sm100"\n\n'
        '[leaderboard]\nsuite = "sol-execbench"\ncard = "leaderboard/card"\n')
    return task_dir, {"start": start, "mit": mit, "gpl": gpl}


def test_the_manifest_states_the_rules_and_the_scored_workloads(warm):
    task_dir, _ = warm
    path = bp.build_prior(task_dir)
    assert path == task_dir / "prior.md"
    text = path.read_text()
    assert "Existing implementations and rules: 001_rms" in text
    assert "destination-passing style" in text and "one `submission.py`" in text
    assert "monkey-patching a timing function" in text
    assert "hidden_states" in text and "bfloat16" in text       # the definition's table
    # T_b / T_SOL and the tolerance come from the card's workload row
    assert "0.03" in text and "0.01" in text
    assert "max_atol=1e-05" in text and "max_rtol=0.05" in text


def test_only_licences_that_permit_incorporation_are_listed_as_usable(warm):
    task_dir, _ = warm
    by_license = {r["license"]: r["path"] for r in seed.read_sources(task_dir / "leaderboard" / "card")}
    assert set(by_license) == {"Apache-2.0", "MIT", "GPL-3.0"}
    text = bp.build_prior(task_dir).read_text()
    usable, _, not_usable = text.partition("Present but NOT reusable")
    assert f"prior/{by_license['Apache-2.0']}" in usable      # the starting kernel
    assert f"prior/{by_license['MIT']}" in usable
    assert f"prior/{by_license['GPL-3.0']}" not in usable
    assert "Start from the highest-scoring entry whose licence permits reuse" in usable


def test_the_excluded_list_names_everything_that_may_not_be_copied(warm):
    task_dir, _ = warm
    by_license = {r["license"]: r["path"] for r in seed.read_sources(task_dir / "leaderboard" / "card")}
    excluded = bp.build_prior(task_dir).read_text().split("Present but NOT reusable", 1)[1]
    assert f"`prior/{by_license['GPL-3.0']}` — license: GPL-3.0" in excluded
    assert "the starting point" in excluded or "the starting point" in \
        bp.build_prior(task_dir).read_text()


def test_the_manifest_is_reproducible(warm):
    task_dir, _ = warm
    first = bp.build_prior(task_dir).read_bytes()
    second = bp.build_prior(task_dir).read_bytes()
    assert first == second


def test_without_an_index_the_klib_section_says_so(warm, tmp_path):
    task_dir, _ = warm
    text = bp.build_prior(task_dir, klib=True,
                          index=tmp_path / "missing.json").read_text()
    assert "## Reference library (klib)" in text
    assert "not on this machine" in text


def test_with_an_index_the_matching_entry_and_its_score_are_listed(warm, tmp_path):
    task_dir, dirs = warm
    by_license = {r["license"]: r["path"] for r in seed.read_sources(task_dir / "leaderboard" / "card")}
    # A library laid out as the seedlib schema has it: <root>/index.json +
    # <root>/kernels/<id>/kernel.py, the entry naming that file relative to the root and
    # carrying the schema's fields (the index is validated when that task has merged).
    library = tmp_path / "library"
    (library / "kernels" / "kw-1").mkdir(parents=True)
    shutil.copy2(dirs["mit"] / "kernel.py", library / "kernels" / "kw-1" / "kernel.py")
    index = library / "index.json"
    index.write_text(json.dumps({"version": 1, "entries": [{
        "id": "kw-1", "problem_keys": ["001_rms", "other"], "family": "other",
        "path": "kernels/kw-1/kernel.py", "language": "pytorch", "entry": "run",
        "source": {"repo": "KernelWiki", "commit": "abc1234", "path": "artifacts/1/kernel.py"},
        "license": "MIT", "reported": {"score": 0.8, "board": "sol-execbench"},
        "verified_local": {"t_ms": 0.02}}]}))
    text = bp.build_prior(task_dir, klib=True, index=index).read_text()
    assert "kw-1" in text and "KernelWiki" in text and "0.8" in text
    # the prior row whose content is that entry carries its score
    rows = [line for line in text.splitlines() if line.startswith("| `prior/")]
    assert any("board:" in row and by_license["MIT"] in row for row in rows)
    # and the locally measured time anchors the target: 0.02 ms / [target] ratio
    assert "0.02 / 1.2" in text and "0.0166667" in text


def test_the_manifest_reaches_a_prompt_through_prior_path(warm, tmp_path):
    task_dir, _ = warm
    path = bp.build_prior(task_dir)
    env = bp.env_for(path)
    assert env == {"FK_PRIOR_PATH": str(path)}
    assert bp.env_line(path) == f"FK_PRIOR_PATH={path}"
    template = tmp_path / "student.md"
    template.write_text("Existing implementations: `{PRIOR_PATH}`\n")
    rendered = render_prompt(template, {"{PRIOR_PATH}": env["FK_PRIOR_PATH"]})
    assert str(path) in rendered and "{PRIOR_PATH}" not in rendered


def test_the_cli_writes_the_manifest_and_prints_the_environment_line(warm, capsys):
    task_dir, _ = warm
    out = task_dir / "custom.md"
    assert prior_cli.main(["build", str(task_dir), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert out.is_file() and f"FK_PRIOR_PATH={out}" in printed
    with pytest.raises(SystemExit) as ei:
        prior_cli.main(["build", "--help"])
    assert ei.value.code == 0


def test_the_cli_reports_a_bad_task(tmp_path, capsys):
    assert prior_cli.main(["build", str(tmp_path / "nope")]) == 2
    assert "[forge prior]" in capsys.readouterr().err


def test_a_missing_local_checkout_is_not_an_error(warm, monkeypatch, tmp_path):
    task_dir, _ = warm
    monkeypatch.setattr(bp, "reference_root", lambda: tmp_path / "no_reference")
    text = bp.build_prior(task_dir).read_text()
    assert "## Local knowledge you may read" in text and "Not present on this machine" in text


def test_the_local_knowledge_section_names_the_query_tools(warm, tmp_path, monkeypatch):
    task_dir, _ = warm
    root = tmp_path / "reference"
    (root / "KernelWiki").mkdir(parents=True)
    (root / "ncu-report-skill").mkdir()
    monkeypatch.setattr(bp, "reference_root", lambda: root)
    text = bp.build_prior(task_dir).read_text()
    assert "scripts/query.py --tag tcgen05 --compact" in text
    assert "scripts/get_page.py" in text and "ncu-report-skill" in text
