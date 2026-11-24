"""tests for the export gates (`forge export`) and the eval JSON's integrity block read-back.

The export path is the last place a check can stand between the harness and the board: a
block-level finding refuses the write, the cold-start gate refuses a candidate the board
would time out, and the override records itself in the exported description.  The
parse_eval tests pin that the new block is carried and changes nothing.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from forgekernel import launch                    # noqa: E402
from forgekernel import integrity as icheck       # noqa: E402
from forgekernel.loop.parse import parse_eval     # noqa: E402

CLEAN = "import torch\n\ndef run(x, out):\n    out.copy_(x)\n"
BLOCKED = ("import os\nos.environ['TF32'] = '0'\n\n"
           "def run(x, out):\n    out.copy_(x)\n")


def _fake_run(tmp_path: Path) -> types.SimpleNamespace:
    d = tmp_path / "run"
    (d / "operators" / "op").mkdir(parents=True)
    return types.SimpleNamespace(dir=d, operators_dir=d / "operators", task_dir=tmp_path,
                                 task_name="op")


def _fake_task(mode: str):
    return types.SimpleNamespace(mode=mode, name="op", leaderboard=None)


@pytest.fixture
def export_env(tmp_path, monkeypatch):
    """``cmd_export`` with a run, a task and a confirmed best kernel, all faked."""
    run = _fake_run(tmp_path)
    kernel = {"sha": "a" * 40, "window": "device", "median_ms": 0.01, "speedup": 1.2,
              "ratios": [], "kernel": CLEAN}

    monkeypatch.setattr(launch, "open_run", lambda path: run)
    monkeypatch.setattr(launch, "load_task", lambda task_dir, name=None: _fake_task("source"))
    monkeypatch.setattr(launch, "best_of", lambda r, t: kernel)
    return types.SimpleNamespace(run=run, kernel=kernel, tmp=tmp_path)


# --- the static review gate -------------------------------------------------------------

def test_a_clean_export_writes_the_kernel_and_nothing_else(export_env, capsys) -> None:
    assert launch.cmd_export([str(export_env.run.dir)]) == 0
    captured = capsys.readouterr()
    assert captured.out == CLEAN                      # stdout is the exported bytes, only
    assert "integrity review: 0 block" in captured.err


def test_a_block_level_finding_refuses_the_export(export_env, capsys) -> None:
    export_env.kernel["kernel"] = BLOCKED
    assert launch.cmd_export([str(export_env.run.dir)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "export refused" in captured.err and "block-level integrity" in captured.err
    assert "env-assign" in captured.err


def test_the_override_flag_exports_and_says_so(export_env, capsys) -> None:
    export_env.kernel["kernel"] = BLOCKED
    assert launch.cmd_export([str(export_env.run.dir),
                              "--allow-integrity-findings"]) == 0
    captured = capsys.readouterr()
    assert captured.out == BLOCKED
    assert "overridden" in captured.err


def test_a_board_export_records_the_override_in_the_description(export_env, monkeypatch,
                                                                tmp_path, capsys) -> None:
    export_env.kernel["kernel"] = BLOCKED
    monkeypatch.setattr(launch, "load_task",
                        lambda task_dir, name=None: _fake_task(launch.MODE_LEADERBOARD))
    card = export_env.run.operators_dir / "op"
    (card / "eval" / "leaderboard").mkdir(parents=True)
    (card / "eval" / "leaderboard" / "problem.json").write_text(json.dumps(
        {"kernel_file": "kernel.py"}))
    monkeypatch.setattr("forgekernel.leaderboard.export_solution",
                        lambda op_dir, **kw: {"name": kw["name"], "description": "d",
                                              "sources": [{"content": BLOCKED}]})
    assert launch.cmd_export([str(export_env.run.dir),
                              "--allow-integrity-findings"]) == 0
    sol = json.loads(capsys.readouterr().out)
    assert sol["description"].endswith("integrity: 1 block-level finding(s) overridden")
    # without the flag the same card is refused before any JSON is produced
    assert launch.cmd_export([str(export_env.run.dir)]) == 2
    assert capsys.readouterr().out == ""


def test_a_clean_board_export_gets_no_integrity_line(export_env, monkeypatch, capsys) -> None:
    monkeypatch.setattr(launch, "load_task",
                        lambda task_dir, name=None: _fake_task(launch.MODE_LEADERBOARD))
    card = export_env.run.operators_dir / "op"
    (card / "eval" / "leaderboard").mkdir(parents=True)
    (card / "eval" / "leaderboard" / "problem.json").write_text(json.dumps(
        {"kernel_file": "kernel.py"}))
    monkeypatch.setattr("forgekernel.leaderboard.export_solution",
                        lambda op_dir, **kw: {"name": kw["name"], "description": "d",
                                              "sources": [{"content": CLEAN}]})
    assert launch.cmd_export([str(export_env.run.dir)]) == 0
    assert "integrity:" not in json.loads(capsys.readouterr().out)["description"]


def test_the_target_rules_are_a_soft_dependency_of_the_export() -> None:
    """targethw's own block rules reach the export through export_check."""
    verdict = icheck.export_check(
        "import torch\nFLAGS = ['-gencode', 'arch=compute_103a,code=sm_103a']\n"
        "def run(x, out):\n    out.copy_(x)\n", path=False)
    assert verdict["ok"] is False
    rules = {f["rule"] for f in verdict["findings"]}
    assert {"arch-literal", "b300-only", "gencode-dev-only"} <= rules, rules
    assert all(f["severity"] == "block" for f in verdict["findings"])


def test_the_licence_gate_reaches_the_export_check() -> None:
    verdict = icheck.export_check(CLEAN, path=False,
                                  origins=[{"source": "a/k.py", "license": "GPL-3.0"}])
    assert verdict["ok"] is False and verdict["licence"]["problems"]
    clean = icheck.export_check(
        "# Copyright (c) 2024 X\n" + CLEAN, path=False,
        origins=[{"source": "a/k.py", "license": "MIT", "notice": "Copyright (c) 2024 X"}])
    assert clean["ok"] is True


# --- the cold-start gate ----------------------------------------------------------------

def test_the_cold_start_gate_reads_a_bench_result(tmp_path) -> None:
    result = tmp_path / "bench.json"
    result.write_text(json.dumps({"correct": True, "cold_start_s": 12.5}))
    gate = icheck.cold_start_gate(result)
    assert gate["state"] == "ok" and gate["cold_start_s"] == 12.5
    result.write_text(json.dumps({"correct": True, "cold_start_s": 150.0}))
    gate = icheck.cold_start_gate(result)
    assert gate["state"] == "over-budget" and "timed out at 0" in gate["why"]
    result.write_text(json.dumps({"correct": True, "cold_start_s": None}))
    assert icheck.cold_start_gate(result)["state"] == "unmeasured"
    assert icheck.cold_start_gate(tmp_path / "absent.json")["state"] == "unmeasured"


def test_the_cold_start_budget_is_benchx_s() -> None:
    from forgekernel.benchx.api import COLD_START_BUDGET_S
    gate = icheck.cold_start_gate()
    assert gate["budget_s"] == COLD_START_BUDGET_S == 100.0


def test_an_over_budget_cold_start_refuses_the_export(export_env, capsys) -> None:
    bench = export_env.run.dir / "bench.json"
    bench.write_text(json.dumps({"correct": True, "cold_start_s": 140.0}))
    assert launch.cmd_export([str(export_env.run.dir)]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "cold start" in captured.err
    # the escape hatch a campaign has too
    assert launch.cmd_export([str(export_env.run.dir),
                              "--allow-slow-cold-start"]) == 0
    assert capsys.readouterr().out == CLEAN


def test_an_unmeasured_cold_start_is_said_not_assumed(export_env, capsys) -> None:
    assert launch.cmd_export([str(export_env.run.dir)]) == 0
    assert "cold start: not measured" in capsys.readouterr().err


def test_a_named_bench_result_is_read_too(export_env, tmp_path, capsys) -> None:
    other = tmp_path / "elsewhere.json"
    other.write_text(json.dumps({"cold_start_s": 200.0}))
    assert launch.cmd_export([str(export_env.run.dir), "--bench-result", str(other)]) == 2
    assert "cold start" in capsys.readouterr().err


# --- parse_eval carries the block, and the readings do not move ------------------------

def _eval_json(integrity: dict | None) -> str:
    obj = {"correctness": {"pass": True, "status": "PASSED"},
           "performance": {"bwd_median_ms": 0.03, "device_median_ms": 0.01,
                           "host_enqueue_ms": 0.004, "device_window_basis": "cupti_kernel_span",
                           "leaderboard": {"sol_board": {"sol_score": 0.8}},
                           "n_workloads": 1,
                           "workloads": [{"name": "wl0", "device_median_ms": 0.01}]}}
    if integrity is not None:
        obj["integrity"] = integrity
    return "log line\n=== JSON ===\n" + json.dumps(obj) + "\n"


def test_parse_eval_reads_the_integrity_block_verbatim() -> None:
    block = {"fresh_input_recheck": {"status": "fail", "workloads": ["wl0"],
                                     "reason": "matched 0.9"},
             "global_state": {"status": "clean", "diff": []},
             "reference_stability": {"status": "pass", "max_rel_err": 0.0},
             "timer_consistency": {"status": "ok", "events_ms": 1, "span_ms": 0.9,
                                   "wall_ms": 1.1},
             "implausible_speedup": False}
    ev = parse_eval(_eval_json(block))
    assert ev.integrity == block
    assert ev.correctness is True                    # the block never folds into the verdict
    assert ev.device_median_ms == 0.01


def test_parse_eval_without_the_block_has_none_and_the_same_readings() -> None:
    with_block = parse_eval(_eval_json({"fresh_input_recheck": {"status": "pass"}}))
    without = parse_eval(_eval_json(None))
    assert without.integrity is None and with_block.integrity is not None
    for field in ("correctness", "bwd_median_ms", "device_median_ms", "host_enqueue_ms",
                  "device_window_basis", "n_workloads", "workloads", "error"):
        assert getattr(with_block, field) == getattr(without, field), field


def test_a_non_object_integrity_value_is_ignored() -> None:
    text = _eval_json(None).replace('"performance"', '"integrity": ["nope"], "performance"')
    assert parse_eval(text).integrity is None
