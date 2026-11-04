"""Loop CLI + agent-policy fixes, audit corps R2 (BUG-SEED-10/17/18/19/21/48, BUG-C3-02).

One section per fix, in ledger order:

* BUG-SEED-10 — an interrupted agent turn ends the whole process group
  (``start_new_session=True`` + ``_end_process_group``), so a model process under
  the shell wrapper cannot outlive the loop;
* BUG-SEED-18 — the clock-lock budget is a TOTAL of ``clock_lock_retries`` submits;
  the docstrings say so (doc-only fix);
* BUG-SEED-19 — an overlay build failure degrades the job but leaves a structured
  trace, and never reads as "the tree's copies match the card";
* BUG-SEED-21 — a poll timeout counts toward giving an in-flight job up, like a
  log that never arrives;
* BUG-SEED-48 — ``_retry_call`` announces each retry and does not sleep after the
  last failure;
* BUG-C3-02 — the ref predicates count only action keys (ref_cmd / ref_mode /
  submit_op), never the label keys, so a label-only declaration is not mistaken
  for a declared reference;
* BUG-SEED-17 — agent-policy command findings are deduped on (source, text), so a
  second (teacher) command is checked instead of being dropped wholesale.

The BUG-SEED-10/18/19/21/48 cases are ported from the R1 archive
(audit-r2-2026-11-05/forgekernel-untracked-tests/test_cli_submit_fixes.py), adapted
to this tree's names and semantics.
"""

from __future__ import annotations

import json
import subprocess
import time

import pytest

from peregrine_kernel.loop import cli


# -- BUG-SEED-10: an interrupted agent turn ends the whole process group ----------------------


def _pgrep_alive(pattern: str) -> bool:
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return bool(out.stdout.strip())


def _wait_gone(pattern: str, deadline_s: float = 15.0) -> bool:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if not _pgrep_alive(pattern):
            return True
        time.sleep(0.2)
    return not _pgrep_alive(pattern)


def test_end_process_group_kills_the_grandchild_not_just_the_wrapper():
    """The helper the interrupted-turn path must use reaches the whole group: a
    background grandchild under the shell wrapper dies with it (proc.kill() would
    orphan the sleep)."""
    proc = subprocess.Popen(["bash", "-c", "sleep 3001 & wait # pk-group-probe"],
                            start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not _pgrep_alive("sleep 3001"):
            time.sleep(0.05)
        assert _pgrep_alive("sleep 3001"), "the grandchild never started"
    finally:
        cli._end_process_group(proc, grace_s=2.0)
    assert proc.wait(timeout=5) is not None
    assert _wait_gone("sleep 3001")


def test_an_interrupted_agent_turn_ends_the_whole_session(monkeypatch, tmp_path):
    """_agent_turn used to kill only the shell wrapper on Ctrl-C, orphaning the model
    process under it. Now the Popen runs in its own session and the exception path
    ends that whole group: nothing outlives the loop.

    Ported from the R1 archive (test_an_interrupted_agent_turn_ends_the_whole_session);
    ``forgekernel`` -> ``peregrine_kernel``."""
    marker = "pk-agent-turn-probe-2917"
    ended: list = []
    real_end = cli._end_process_group

    def recording_end(proc, *a, **k):
        ended.append(proc.pid)
        return real_end(proc, *a, **k)

    monkeypatch.setattr(cli, "_end_process_group", recording_end)
    real_popen = cli.subprocess.Popen
    popen_kw: dict = {}

    def recording_popen(*a, **kw):
        popen_kw.update(kw)
        return real_popen(*a, **kw)

    monkeypatch.setattr(cli.subprocess, "Popen", recording_popen)

    be = cli.RealBackend("op", tmp_path, None, "n", "e", agent_heartbeat_s=0.001)
    real_print = print

    def second_print_is_interrupt(*a, **k):
        real_print(*a, **k)
        raise KeyboardInterrupt      # the watchdog's signal, arriving on this thread

    monkeypatch.setattr(cli, "print", second_print_is_interrupt, raising=False)
    # The turn leaves a GRANDCHILD running (sleep in the background, bash waiting):
    # the old proc.kill() reached only the wrapper (here the wrapper IS the process,
    # via exec) and the grandchild outlived the loop. pgrep names the grandchild, so
    # this assertion fails against the old kill.
    cmd = f"exec bash -c 'sleep 300042 & wait # {marker}'"
    with pytest.raises(KeyboardInterrupt):
        be._agent_turn(cmd, "prompt", "student")

    assert popen_kw.get("start_new_session") is True
    assert ended                       # the group was ended by name, not proc.kill()
    assert _wait_gone("sleep 300042")  # the sleep under the shell died with the group
