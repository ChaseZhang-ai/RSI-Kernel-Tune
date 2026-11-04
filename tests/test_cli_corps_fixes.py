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


# -- BUG-SEED-18: the clock-lock budget is a TOTAL of clock_lock_retries submits --------------


class _Proc:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_clock_lock_budget_is_a_total_of_clock_lock_retries_submits(monkeypatch, capsys):
    """submit_gpu's clock-lock budget is a TOTAL of ``clock_lock_retries`` submits —
    the first submit plus ``clock_lock_retries - 1`` resubmits — the way the loop
    actually counts; the lost-runner budget next door counts the other way
    (``lost_retries`` extra submits AFTER the first). The progress line names the
    refusal against that total.

    Ported from the R1 archive
    (test_clock_lock_retry_counts_total_submits_and_says_so), adapted: this tree's
    fix is doc-only (the counting was already the total), so the test pins the
    implemented counting and phrasing rather than the archive's reworded line."""
    from peregrine_kernel.submit.jobs import CLOCK_LOCK_REFUSED_MARKER

    refused = f"{CLOCK_LOCK_REFUSED_MARKER} lock not held\n"
    be = cli.RealBackend("op", None, None, "n", "e")
    be.clock_lock_retry_sleep_s = 0
    seen: list = []

    def once(branch, mode):
        seen.append(mode)
        be.last_job_id = str(len(seen))
        return refused

    monkeypatch.setattr(be, "_submit_gpu_once", once)
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    with pytest.raises(cli.ClockLockRefusedError):
        be.submit_gpu("b", "all")
    assert len(seen) == be.clock_lock_retries == 3   # 3 submits in total, not 3 resubmits
    out = capsys.readouterr().out
    assert "resubmitting (1/3)" in out and "resubmitting (2/3)" in out


# -- BUG-SEED-19: an overlay build failure leaves a structured trace --------------------------


def test_an_overlay_build_failure_is_recorded_as_a_structured_trace(monkeypatch, tmp_path,
                                                                    capsys):
    """A card whose overlay cannot be built (a git hiccup, an unreadable card file)
    degrades the job to the tree's own copies — fail-open, on purpose. But the failure
    used to leave one log line and then fall into the dedup'd announce branch, which
    could read "the tree's copies match the card — no overlay": a lie that hides the
    one job that did NOT run the card's judge. The failure now carries a WARNING line
    and a structured record beside the in-flight file in the workspace's git dir.

    Ported from the R1 archive
    (test_an_overlay_build_failure_is_recorded_as_a_structured_trace), adapted per
    the R2 ledger: the failure is injected through harness_guard.blob_sha (OSError)."""
    from peregrine_kernel.loop import harness_guard

    ws = tmp_path / "ws"
    ws.mkdir()
    subprocess.run(["git", "-C", str(ws), "init", "-q"], check=True, capture_output=True)
    be = cli.RealBackend("op", ws, None, "n", "e")
    monkeypatch.setattr(be, "_overlay_wanted",
                        lambda ref: ({"gpu_entry.sh": b"#!/bin/bash\n"}, []))

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(harness_guard, "blob_sha", boom)
    sha = be._with_job_env("abc1234")
    assert sha == "abc1234"                               # fail-open: the job still runs
    out = capsys.readouterr().out
    assert "WARNING" in out and "the tree's own copies" in out
    assert "the tree's copies match the card" not in out  # no misleading success line
    lines = (ws / ".git" / cli.RealBackend.overlay_fallback_file).read_text().splitlines()
    rec = json.loads(lines[-1])
    assert rec["ref"] == "abc1234" and "disk full" in rec["error"]
    assert "tree's own copies" in rec["note"]


def test_an_unreadable_card_file_never_reads_as_no_overlay_needed(tmp_path, capsys):
    """The same fail-open one level down: _overlay_wanted that cannot read the card's
    file degraded silently, and an empty overlay then announced "the tree's copies
    match the card". A degraded run must say WARNING and leave the record."""
    ws = tmp_path / "ws"
    (ws / ".git").mkdir(parents=True)
    be = cli.RealBackend("op", ws, None, "n", "e")
    be.job_env_files = {"gpu_entry.sh": tmp_path / "card" / "missing.sh"}  # unreadable
    sha = be._with_job_env("abc1234")
    assert sha == "abc1234"                               # still fail-open
    out = capsys.readouterr().out
    assert "WARNING" in out and "cannot read the card's" in out
    assert "the tree's copies match the card" not in out
    rec = json.loads((ws / ".git" / cli.RealBackend.overlay_fallback_file)
                     .read_text().splitlines()[-1])
    assert rec["ref"] == "abc1234" and "missing.sh" in rec["error"]


# -- BUG-SEED-21: a poll timeout counts toward giving an in-flight job up ---------------------


def _poll_timeout_backend(monkeypatch, tmp_path):
    from types import SimpleNamespace

    ws = tmp_path / "ws"
    (ws / ".git").mkdir(parents=True)
    n = {"submits": 0}

    def fake_forge(op, verb, *a, **k):
        if verb == "submit":
            n["submits"] += 1
            return SimpleNamespace(returncode=0, stderr="",
                                   stdout=f"[submit] job={654999 + n['submits']}\n")
        if verb == "status":
            return SimpleNamespace(returncode=0, stderr="",
                                   stdout='{"id": 9, "status": "Running"}')
        return SimpleNamespace(returncode=0, stderr="", stdout="log body")

    monkeypatch.setattr(cli, "_forge", fake_forge)
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    be = cli.RealBackend("demo", ws, None, "n", "e", poll_s=0.0, poll_timeout_s=0.0)
    monkeypatch.setattr(cli.RealBackend, "_branch_sha", lambda self, b: "abc")
    return be, ws / ".git" / cli.RealBackend.inflight_file, n


def test_a_poll_timeout_counts_toward_giving_the_job_up(monkeypatch, tmp_path):
    """poll_until_terminal's BackendError used to fly straight out of _submit_gpu_once,
    so a recorded job whose poll never settled was re-adopted after every restart,
    forever — the record neither counted the miss nor was ever cleared. The timeout now
    counts its misses in the record like an unreadable log does; at the threshold the
    record is cleared and the next submit is fresh.

    Ported from the R1 archive (test_a_poll_timeout_counts_toward_giving_the_job_up);
    fk-inflight.json -> RealBackend.inflight_file."""
    be, rec_path, n = _poll_timeout_backend(monkeypatch, tmp_path)
    for expected_misses in (1, 2):
        with pytest.raises(cli.BackendError, match="job 655000"):
            be.submit_gpu("br", "all")
        rec = json.loads(rec_path.read_text())
        assert rec["log_failures"] == expected_misses     # the record survives, still adoptable
    with pytest.raises(cli.BackendError, match="job 655000"):
        be.submit_gpu("br", "all")
    assert not rec_path.exists()                          # third miss: the job is given up
    with pytest.raises(cli.BackendError, match="job 655001"):
        be.submit_gpu("br", "all")
    assert n["submits"] == 2                              # only the 1st and this submit ran
