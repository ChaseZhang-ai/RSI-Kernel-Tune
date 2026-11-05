"""BUG-CH-E-02 regression: the launch admission sentinel is written only after
every synchronous refusal gate, immediately before the loop starts.

The sentinel used to be written the moment the fresh-start contamination gate
passed — but a dozen refusal gates still lay ahead (goal generation, baselines,
target derivation, target workloads, harness drift, strict precision, KB, job
env). Any of them refusing left the sentinel permanently on disk, so the
operator's unfixed retry evaluated as a RESUME: the three fresh-start-only gates
(EXIT_TARGET_WORKLOADS / EXIT_HARNESS_DRIFT / EXIT_PRECISION_UNREPORTED)
degraded from refusal to warning and the run carried the unfixed config
silently. Now a refused launch leaves no sentinel: the retry is judged as the
fresh start it still is (refuse -> refuse, never refuse -> warn).

The admission itself is preserved — it moves to just before run_two_layer, so a
crash once the loop is running still restarts into a resume.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from peregrine_kernel.loop import cli

SENTINEL = cli._LAUNCH_SENTINEL_NAME


def _root(tmp_path: Path):
    """A minimal repo root: operators/<op>/ plus an openable hw_kb tree."""
    root = tmp_path / "repo"
    op_dir = root / "operators" / "demo"
    op_dir.mkdir(parents=True)
    kb = root / "hw_kb" / "cutedsl" / "sm90"
    kb.mkdir(parents=True)
    (kb / "index.md").write_text("# kb\n", encoding="utf-8")
    return root, op_dir


def _workspace(tmp_path: Path, *, pushable: bool) -> Path:
    ws = tmp_path / "ws"
    subprocess.run(["git", "init", "-q", str(ws)], check=True)
    if pushable:
        subprocess.run(["git", "-C", str(ws), "remote", "add", "origin",
                        str(tmp_path / "origin.git")], check=True)
    return ws


def _argv(tmp_path: Path, ws: Path, oracle: Path) -> list[str]:
    return ["demo", "--workspace", str(ws), "--oracle", str(oracle),
            "--target-ms", "1.0", "--agent-cmd", "true",
            "--allow-200k-window", "--allow-single-workload", "--hours", "0.1"]


def _sentinel_path(op_dir: Path) -> Path:
    return op_dir / "student" / SENTINEL


def _records(op_dir: Path) -> Path:
    return op_dir / "student" / "status.jsonl"


def test_a_refused_launch_leaves_no_sentinel(tmp_path, monkeypatch, capsys):
    """Strict precision contract + a bench that cannot report the block refuses
    a fresh start (exit 20) AFTER where the sentinel used to be written — and
    must not leave one behind."""
    root, op_dir = _root(tmp_path)
    ws = _workspace(tmp_path, pushable=False)
    monkeypatch.setattr(cli, "_repo_root", lambda: root)
    rc = cli.main(_argv(tmp_path, ws, tmp_path / "oracle.py"))
    assert rc == cli.EXIT_PRECISION_UNREPORTED
    assert not _sentinel_path(op_dir).exists()


def test_a_refused_retry_is_refused_again_not_resumed(tmp_path, monkeypatch, capsys):
    """The reported downgrade, end to end: launch #1 refuses (fresh), launch #2
    with nothing fixed must refuse AGAIN — under the old placement the sentinel
    from launch #1 turned launch #2 into a resume and the same gate only
    warned."""
    root, op_dir = _root(tmp_path)
    ws = _workspace(tmp_path, pushable=False)
    monkeypatch.setattr(cli, "_repo_root", lambda: root)
    argv = _argv(tmp_path, ws, tmp_path / "oracle.py")
    assert cli.main(argv) == cli.EXIT_PRECISION_UNREPORTED
    assert cli.main(argv) == cli.EXIT_PRECISION_UNREPORTED
    err = capsys.readouterr().err
    assert err.count("refusing a fresh start under precision_contract") == 2
    assert not _sentinel_path(op_dir).exists()


def test_a_repaired_retry_starts_fresh_and_then_gets_admitted(tmp_path, monkeypatch):
    """Fix the card (precision helper + recorder call) and the same launch gets
    all the way to run_two_layer — where the sentinel must already be on disk,
    so a crash from that point on still resumes."""
    root, op_dir = _root(tmp_path)
    ws = _workspace(tmp_path, pushable=True)
    (op_dir / "eval").mkdir()
    (op_dir / "eval" / "precision.py").write_text("class PrecisionRecorder:\n    pass\n",
                                                  encoding="utf-8")
    (op_dir / "eval" / "bench.py").write_text(
        "import precision\nprecision.PrecisionRecorder()\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_repo_root", lambda: root)
    monkeypatch.setattr(cli, "job_env_preflight", lambda ctx, policy: None)

    seen = {}

    def fake_run_two_layer(ctx, max_meta):
        seen["admitted"] = cli._launch_admitted(op_dir, _records(op_dir))
        raise RuntimeError("stop-the-test-here")

    monkeypatch.setattr(cli, "run_two_layer", fake_run_two_layer)
    with pytest.raises(RuntimeError, match="stop-the-test-here"):
        cli.main(_argv(tmp_path, ws, tmp_path / "oracle.py"))
    assert seen["admitted"] is True
    assert _sentinel_path(op_dir).exists()
