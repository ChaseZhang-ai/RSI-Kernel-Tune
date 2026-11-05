"""Regression pins for the BUG-CORPS-C-09 and BUG-CORPS-C-10 fixes (audit wave R3C).

C-09: the goal-ref spool sits inside the agent's workspace; the side of it the harness
writes (responses/, logs/, context.json) must be in the session's deny_write, so the
agent cannot pre-write a response that makes ``pending()`` skip serving its own
evaluation, or forge the verdict text its eval command prints. The harness itself
(outside the agent's sandbox) keeps writing the answers.

C-10: the session's eval/exec/check command runs ``python -m`` from the agent's
working directory; without interpreter isolation a ``rsi_kernel_tune`` package
planted in that directory shadows this module and forges the CLI the verdicts ride on.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from rsi_kernel_tune.loop import goal_ref as gr


def _card(root: Path, op: str = "demo") -> Path:
    """A minimal git-inited card (ensure_ws clones it for the session workspace)."""
    op_dir = root / "operators" / op
    op_dir.mkdir(parents=True)
    (op_dir / "kernel.py").write_text("def run(x, out):\n    raise NotImplementedError\n",
                                      encoding="utf-8")
    gr._git(op_dir, "init", "-q", "-b", "main")
    gr._git(op_dir, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    gr._git(op_dir, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m",
            "card")
    return op_dir


# -- BUG-CORPS-C-10 ------------------------------------------------------------------------

def test_the_session_cli_command_isolates_the_interpreter():
    cmd = gr.command("eval", "/some/ws")
    assert " -P " in cmd and " -s " in cmd and "-m rsi_kernel_tune.loop.goal_ref" in cmd
    assert "PYTHONPATH=" in cmd          # this checkout, not the working directory


def test_a_shadow_package_in_the_working_directory_cannot_hijack_the_cli(tmp_path):
    """End to end: cwd is the agent-writable workspace, and a planted rsi_kernel_tune
    package would answer first without -P (cwd precedes PYTHONPATH for ``python -m``).
    With it, the real module answers."""
    ws = tmp_path / "ws"
    shadow = ws / "rsi_kernel_tune" / "loop"
    shadow.mkdir(parents=True)
    (ws / "rsi_kernel_tune" / "__init__.py").write_text("", encoding="utf-8")
    (shadow / "__init__.py").write_text("", encoding="utf-8")
    (shadow / "goal_ref.py").write_text(
        "import sys\n"
        "sys.stdout.write('SHADOW-CLI-ANSWER\\n')\n"
        "sys.exit(0)\n", encoding="utf-8")
    r = subprocess.run(gr.command("status", str(tmp_path / "goal_ref")), shell=True,
                       cwd=str(ws), capture_output=True, text=True, timeout=120)
    assert "SHADOW-CLI-ANSWER" not in r.stdout + r.stderr
    assert "speed bar" in r.stdout       # the real module's status output


# -- BUG-CORPS-C-09 ------------------------------------------------------------------------

def test_the_harness_answers_while_the_session_cannot_write_the_answer(tmp_path):
    """pending() skips a request whose response file exists — a response the agent
    could write itself would silently stand in for a real evaluation. The responses
    side is deny_write now; the harness (outside the sandbox) still writes it."""
    op_dir = _card(tmp_path)
    paths = gr.Paths.of(op_dir)
    ws = gr.ensure_ws(paths, op_dir)
    (ws / "kernel.py").write_text("x", encoding="utf-8")
    # generate() writes this before the policy installs; this test skips the session
    (paths.spool / "context.json").write_text(json.dumps({"kernel_file": "kernel.py"}),
                                              encoding="utf-8")
    ro = gr.session_read_only_paths(op_dir)
    for p in (paths.spool / "responses", paths.spool / "logs",
              paths.spool / "context.json"):
        assert p in ro and p.exists()    # denied only once the harness created them
    assert paths.spool / "requests" not in ro   # the eval command creates requests

    rid, _req = gr._new_request(ws, "eval", "kernel.py", {})

    class _Backend:
        last_job_id = ""

        def submit_gpu(self, branch, mode):     # never reached: the deadline passed
            raise AssertionError("no GPU job for an expired deadline")

    svc = gr.Service(op="demo", op_dir=op_dir, cfg={}, paths=paths, tree=ws,
                     kernel_file="kernel.py", bar={"kind": "none", "text": "-"},
                     deadline=time.time() - 1.0, backend=_Backend(), log=lambda s: None)
    assert svc.serve_pending() == 1
    answer = (paths.spool / "responses" / f"{rid}.txt").read_text(encoding="utf-8")
    assert gr.DEADLINE_LINE in answer    # the harness, not the agent, wrote the verdict
