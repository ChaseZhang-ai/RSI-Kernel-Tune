"""Regression tests for the audit-corps fix wave 4: the process-life-cycle, ssh job
channel, and cli-miscellaneous surface.

Each test names the BUG it pins: SEED-10 (an interrupted agent turn must take its whole
process group down), SEED-18 (the clock-lock retry budget's docs match its attempts
count), SEED-48 (a failed retry budget does not sleep after its last attempt, and a
call that ran to its own timeout is not retried), SEED-21 (a poll that never reaches a
terminal state counts toward the in-flight give-up), SEED-19 (a harness overlay build
failure refuses the reading instead of silently re-judging with the lineage's own
copies), C3-02 (reference label keys do not read as a declared reference), SEED-20 (the
remote job tree carries the op tree's identity, not just its basename), SEED-11 (the
ssh job deadline starts when the job begins, not while it queues for the host's GPU
lock), SEED-12 (an ssh drop after the job began is Lost and ends the remote session),
and T2C-02 (the ssh extract step is bounded and rides the transport channel).
"""
import hashlib
import json as _json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from peregrine_kernel.loop import cli  # noqa: E402
from peregrine_kernel.loop.cli import RunnerLostError  # noqa: E402
from peregrine_kernel.loop.driver import BackendError  # noqa: E402
from peregrine_kernel.submit import direct, jobs  # noqa: E402


# --- BUG-SEED-10: an interrupted agent turn ends its whole process group ------------


def test_an_interrupted_agent_turn_ends_its_whole_process_group(tmp_path, monkeypatch):
    """The turn is a shell=True Popen; proc.kill() killed only the shell, and the agent
    (the leg script's grandchild) survived as an orphan spending API usage. The turn
    must run in a session of its own and its interrupt path must end that GROUP."""
    real_popen = subprocess.Popen
    holder = {}

    def popen(cmd, **kw):
        proc = real_popen(cmd, **kw)
        holder["proc"] = proc
        return proc

    monkeypatch.setattr(cli.subprocess, "Popen", popen)

    class _NeverPump:
        """Stands in for the pump thread: the turn stays in its heartbeat loop until
        the interrupt lands."""

        def start(self):
            pass

        def is_alive(self):
            return True

        def join(self, timeout=None):
            if timeout is None:
                raise KeyboardInterrupt      # the interrupt lands on the caller's thread
            return None

    monkeypatch.setattr(cli.threading, "Thread", lambda *a, **k: _NeverPump())
    be = cli.RealBackend("op", tmp_path, None, "t", "t@t")
    be.agent_heartbeat_s = 0        # join(timeout=None): the interrupt lands on beat 1
    # The command's leg: shell -> bash -> sleep. The sleep is the "agent" grandchild
    # a bare proc.kill() leaves behind.
    with pytest.raises(KeyboardInterrupt):
        be._agent_turn("bash -c 'sleep 47.3 & wait'", "prompt", "student")
    assert holder["proc"].poll() is not None
    left = subprocess.run(["pgrep", "-f", r"sleep 47\.3"], capture_output=True,
                          text=True).stdout
    assert left.strip() == "", "the agent grandchild outlived the interrupted turn"


def test_agent_turn_start_new_session_is_passed(tmp_path, monkeypatch):
    """The group terminator assumes Popen(start_new_session=True) built the group: the
    turn's Popen must actually pass it."""
    seen = {}
    real_popen = subprocess.Popen

    def popen(cmd, **kw):
        seen.update(kw)
        return real_popen("true", **kw)

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(cli.threading, "Thread", lambda *a, **k:
                        type("_P", (), {"start": lambda self: None,
                                        "is_alive": lambda self: False,
                                        "join": lambda self, timeout=None: None})())
    be = cli.RealBackend("op", tmp_path, None, "t", "t@t")
    be._agent_turn("true", "prompt", "student")
    assert seen.get("start_new_session") is True


# --- BUG-SEED-18: the retry budgets say what they spend ------------------------------

def _lost_backend(monkeypatch, tmp_path, *, lost_retries):
    """A backend whose every job comes back Lost (no runner, no verdict)."""
    ws = tmp_path / "ws"
    (ws / ".git").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)

    def fake_forge(op, verb, *a, **k):
        calls.append(verb)
        if verb == "submit":
            return type("P", (), {"returncode": 0,
                                  "stdout": f"[submit] job={len(calls)}\n"})()
        if verb == "status":
            return type("P", (), {"returncode": 0,
                                  "stdout": '{"id": 1, "status": "Lost"}\n'})()
        return type("P", (), {"returncode": 0, "stdout": ""})()

    monkeypatch.setattr(cli, "_forge", fake_forge)
    be = cli.RealBackend("demo", ws, None, "n", "e", poll_s=0.0,
                         lost_retries=lost_retries)
    return be, calls


def test_the_lost_budget_spends_its_extra_submits_then_stops(monkeypatch, tmp_path, capsys):
    """lost_retries reads "extra submits": the same value as clock_lock_retries buys
    one MORE attempt on this path (lost_retries + 1 total), and the run log never
    prints a count past the budget (the old "(4/3)" overflow)."""
    be, calls = _lost_backend(monkeypatch, tmp_path, lost_retries=1)
    with pytest.raises(RunnerLostError):
        be.submit_gpu("br", "all")
    assert calls.count("submit") == 2          # initial + 1 extra submit
    out = capsys.readouterr().out
    assert "extra submit 1/1" in out
    assert "/1)" not in out.replace("extra submit 1/1", ""), \
        "a lost count past the budget was printed"


def test_the_clock_lock_budget_prints_its_attempt_count(monkeypatch, tmp_path, capsys):
    """The resubmit line names the ATTEMPT about to be spent (an initial submit plus
    clock_lock_retries - 1 resubmits), not a resubmit number that over-reads it."""
    ws = tmp_path / "ws"
    ws.mkdir()
    logs = iter([jobs.CLOCK_LOCK_REFUSED_MARKER + " lock not held\n"] * 5)
    seen = []

    def once(branch, mode):
        seen.append(mode)
        return next(logs)

    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    be = cli.RealBackend("op", ws, None, "t", "t@t")
    be.clock_lock_retry_sleep_s = 0
    monkeypatch.setattr(be, "_submit_gpu_once", once)
    with pytest.raises(cli.ClockLockRefusedError):
        be.submit_gpu("b", "all")
    assert len(seen) == be.clock_lock_retries  # attempts semantics, as ever
    out = capsys.readouterr().out
    assert "attempt 2 of 3" in out and "attempt 3 of 3" in out
    assert "attempt 4" not in out


# --- BUG-SEED-48: the retry budget does not sleep past its last attempt --------------

def _sleeps_recorder(monkeypatch):
    sleeps = []
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)
    return sleeps


def test_a_failing_retry_budget_never_sleeps_after_its_last_attempt(monkeypatch):
    """Three failing attempts with a 10 s base sleep cost two backoffs, not three: the
    old unconditional tail sleep added the last multiplier after the final failure."""
    sleeps = _sleeps_recorder(monkeypatch)
    be = cli.RealBackend("op", Path("/ws"), None, "n", "e", git_retries=3,
                         git_retry_sleep=10.0)
    proc = type("P", (), {"returncode": 1, "stderr": "boom"})()
    with pytest.raises(BackendError):
        be._retry_call(lambda: proc, what="submit all")
    assert sleeps == [10.0, 20.0]


def test_a_call_that_ran_to_its_own_timeout_is_not_retried(monkeypatch):
    """A hung submit comes back as one 600 s timeout, not three of them plus backoff
    (~31 minutes before this could say anything)."""
    import subprocess as sp

    sleeps = _sleeps_recorder(monkeypatch)
    be = cli.RealBackend("op", Path("/ws"), None, "n", "e", git_retries=3,
                         git_retry_sleep=10.0)
    calls = []

    def fn():
        calls.append(1)
        raise sp.TimeoutExpired("cmd", 600)

    with pytest.raises(BackendError, match="timed out"):
        be._retry_call(fn, what="submit all")
    assert calls == [1] and sleeps == []


# --- BUG-SEED-21: a poll that never reaches a terminal state counts toward give-up ---

def test_a_poll_that_never_reaches_terminal_counts_toward_giving_up(monkeypatch, tmp_path):
    """A poll timeout used to raise before the give-up counting: the record sat there
    and every restart re-adopted the same job, each paying another full poll timeout.
    The misses now count, and the record is let go once retrying is clearly futile."""
    ws = tmp_path / "ws"
    (ws / ".git").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)

    def fake_forge(op, verb, *a, **k):
        calls.append(verb)
        if verb == "submit":
            return type("P", (), {"returncode": 0, "stdout": "[submit] job=7\n"})()
        if verb == "status":
            return type("P", (), {"returncode": 0,
                                  "stdout": '{"id": 7, "status": "Running"}\n'})()
        return type("P", (), {"returncode": 0, "stdout": ""})()

    monkeypatch.setattr(cli, "_forge", fake_forge)
    be = cli.RealBackend("demo", ws, None, "n", "e", poll_s=0.0, poll_timeout_s=0.5)
    monkeypatch.setattr(cli.RealBackend, "_branch_sha", lambda self, b: "abc")
    rec_path = ws / ".git" / "pk-inflight.json"
    rec_path.write_text(_json.dumps({"job": "7", "branch": "br", "mode": "all",
                                     "sha": "abc"}))
    for attempt in range(1, cli._INFLIGHT_LOG_ATTEMPTS + 1):
        with pytest.raises(BackendError, match="terminal state"):
            be.submit_gpu("br", "all")
        if attempt < cli._INFLIGHT_LOG_ATTEMPTS:
            assert _json.loads(rec_path.read_text())["log_failures"] == attempt
    assert not rec_path.exists(), "a job that never reports is adopted forever"
    # the record is gone: the next submit is a real resubmission, not another adoption
    with pytest.raises(BackendError, match="terminal state"):
        be.submit_gpu("br", "all")
    assert calls.count("submit") == 1


# --- BUG-SEED-19: a failed harness overlay build is never silently fail-open ---------

from test_harness_guard import _backend, _card, _commit, _git, _tree  # noqa: E402
from test_loop_driver import EVAL, FakeBackend, NCU, REC, _ctx  # noqa: E402

from peregrine_kernel.loop.driver import run_inner_iteration  # noqa: E402
from peregrine_kernel.inner.trajectory import load_records  # noqa: E402


def _overlay_backend(ws, card):
    be = _backend(ws, card)
    base = be.head()
    return be, base


def test_an_unreadable_card_file_marks_the_overlay_failed(tmp_path):
    card, ws = _card(tmp_path)
    be, base = _overlay_backend(ws, card)
    (card / "eval" / "bench.py").chmod(0o000)
    try:
        be._with_job_env(base)
        assert "cannot read the card's" in be.last_overlay_failure
    finally:
        (card / "eval" / "bench.py").chmod(0o644)
    be._with_job_env(base)
    assert be.last_overlay_failure == ""        # the next push reports only itself


def test_a_card_that_stopped_being_a_repository_marks_the_overlay_failed(tmp_path):
    card, ws = _card(tmp_path)
    be, base = _overlay_backend(ws, card)
    shutil.rmtree(card / ".git")
    be._with_job_env(base)
    assert "no longer a git repository" in be.last_overlay_failure


def test_a_card_that_tracks_no_harness_is_config_not_failure(tmp_path):
    """The pinned legitimate case: a card with no tracked harness overlays nothing and
    records no failure (the preflight warned at launch; there is no judge to overlay)."""
    card, ws = _card(tmp_path)
    be, base = _overlay_backend(ws, card)
    _git(card, "rm", "-q", "-r", "--cached", "gpu_entry.sh", "eval", "ref_bin")
    _git(card, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "untrack")
    sha = be._with_job_env(base)
    assert be.last_overlay_failure == ""
    assert {"gpu_entry.sh", "eval/bench.py"} <= _tree(ws, sha)


def test_a_failing_overlay_commit_marks_the_overlay_failed(tmp_path):
    card, ws = _card(tmp_path)
    (ws / "eval" / "bench.py").write_text("student judge\n")
    head = _commit(ws, "student")
    be = _backend(ws, card)
    be._commit_overlay = lambda ref, changes: (_ for _ in ()).throw(
        BackendError("git commit-tree failed (test)"))
    sha = be._with_job_env(head)
    assert sha == head                          # the raw tree rides, as before
    assert "could not overlay the card's files" in be.last_overlay_failure


def test_the_gate_refuses_a_reading_from_an_overlaid_failed_job(tmp_path):
    """An eval whose push could not overlay the card's harness ran the lineage's own
    judge: the row is a REVERT that names the overlay failure, never a KEEP."""
    class OverlayFailedBackend(FakeBackend):
        def push_branch(self, branch):
            self.calls.append(("push", branch))
            self.last_overlay_failure = "the card vanished mid-run (test)"

    be = OverlayFailedBackend([REC("D1.1", 0)], [EVAL(2.4), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.ncu_on_revert = False
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert rec.verdict == "REVERT" and rec.correctness is False
    assert "harness overlay failed" in rec.error and "not judgeable" in rec.error
    rows = load_records(ctx.status_path)
    assert rows[-1].verdict == "REVERT" and "harness overlay failed" in rows[-1].error


def test_the_gate_still_keeps_a_healthy_overlay_reading(tmp_path):
    """No failure flag, no refusal: the normal path is untouched."""
    be = FakeBackend([REC("D1.1", 0)], [EVAL(2.4), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.ncu_on_revert = False
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert rec.verdict == "KEEP" and rec.correctness is True


# --- BUG-C3-02: reference label keys are not a declared reference ---------------------

def test_label_only_reference_declarations_declare_nothing():
    """ref_name / ref_shape / ref_hardware / ref_ncu_cmd only annotate a reference.
    A [loop] that names one without giving anything to measure declares no reference:
    neither the rides-the-job judgment nor the window preflight may read it as one."""
    cfg = {"loop": {"ref_name": "cuDNN", "ref_shape": "B=1,H=128",
                    "ref_hardware": "H100", "ref_ncu_cmd": "ncu --target-processes all"}}
    spec = cli._resolve_ref_spec(cfg)
    assert spec, "the labels still resolve (display uses them)"
    assert cli._ref_rides_job(spec) is False
    assert cli._arm_has_reference(cfg) is False


def test_measurable_reference_declarations_still_declare():
    cfg_cmd = {"loop": {"ref_cmd": "python3 eval/measure_ref.py",
                        "ref_name": "cuDNN"}}
    assert cli._ref_rides_job(cli._resolve_ref_spec(cfg_cmd)) is True
    assert cli._arm_has_reference(cfg_cmd) is True
    cfg_mode = {"loop": {"ref_mode": "faref", "ref_submit_op": "_recon",
                         "ref_name": "FA"}}
    assert cli._ref_rides_job(cli._resolve_ref_spec(cfg_mode)) is False
    assert cli._arm_has_reference(cfg_mode) is True


def test_label_only_run_preflights_as_na(tmp_path):
    """A fresh target-ms-only run whose [loop] only labels a reference preflights as
    "n/a" — it used to be refused with EXIT_WINDOW_UNDECLARED."""
    from peregrine_kernel.loop.cli import _window_preflight
    (tmp_path / "window.toml").write_text("")       # absent contract, no reference
    state, contract, problems = _window_preflight(
        tmp_path, {"loop": {"ref_name": "cuDNN", "ref_shape": "B=1"}}, resuming=False)
    assert state == "n/a" and problems == []


# --- BUG-SEED-20: the remote job tree carries the op tree's identity ------------------

def _clone_op_tree(tmp_path, name="repo"):
    """Two checkouts of one operator tree, in same-named directories on one host."""
    trees = []
    for i in (0, 1):
        d = tmp_path / f"host{i}" / name / "operators" / "op"
        d.mkdir(parents=True)
        (d / "gpu_entry.sh").write_text("#!/bin/bash\n")
        (d / "submit.toml").write_text('[repo]\ndelivery = "ssh"\nbranch = "main"\n'
                                       '[exec]\nhost = "box"\nworkdir = "/data/fk"\n')
        trees.append(d)
    return trees


def test_same_named_clones_get_different_remote_trees(tmp_path):
    """Two clones of one repository share the local basename, so the old run/op tree
    name collided and one job's extract rm -rf'd the tree another clone's job ran."""
    (a, b), cfgs = _clone_op_tree(tmp_path), []
    for d in (a, b):
        cfg = direct.SubmitConfig(branch="main", op_dir=str(d), exec_host="box",
                                  exec_workdir="/data/fk")
        cfgs.append(cfg)
        tree = direct.remote_tree(cfg)
        assert tree.startswith("/data/fk/repo/op-")
        assert tree == direct.remote_tree(cfg)      # deterministic: resubmits find it
    assert direct.remote_tree(cfgs[0]) != direct.remote_tree(cfgs[1])


def test_the_run_segment_still_carries_the_run_id(tmp_path):
    """In run mode the snapshot directory (task + time + token) stays the run segment;
    only the op segment gains the identity digest."""
    d = tmp_path / "runs" / "demo_260105-101112-abcd" / "operators" / "op"
    d.mkdir(parents=True)
    cfg = direct.SubmitConfig(branch="main", op_dir=str(d), exec_host="box",
                              exec_workdir="/data/fk")
    tree = direct.remote_tree(cfg)
    assert Path(tree).parts[3] == "demo_260105-101112-abcd"
    assert tree.endswith(f"op-{hashlib.sha256(str(d.resolve()).encode()).hexdigest()[:8]}")


# --- BUG-SEED-11: the ssh job deadline starts when the job begins ---------------------

from test_direct_delivery import _ssh_cfg, _ssh_run  # noqa: E402
import fcntl as fcntl_mod  # noqa: E402


class _QueuedSsh:
    """The exec step's session queues for the host GPU lock: no begin marker until
    ``queue_polls`` waits have passed, then the job runs and finishes."""

    def __init__(self, queue_polls):
        self.queue_polls = queue_polls
        self.polls = 0
        self.waits = []          # every wait() timeout argument, in order
        self.began = False

    def run(self, argv, input=None, capture_output=False, timeout=None):
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    def popen(self, argv, stdout=None, stderr=None, text=None, stdin=None, **kw):
        fake = self

        class _P:
            pid = 987654321          # never a real pid: killpg must be a no-op here

            @staticmethod
            def kill():
                pass

            @staticmethod
            def wait(timeout=None):
                fake.waits.append(timeout)
                if fake.polls < fake.queue_polls:
                    fake.polls += 1
                    raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout)
                if not fake.began:
                    fake.began = True
                    stdout.write(f"{jobs.BEGIN_MARKER}\n")
                    stdout.flush()
                    raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout)
                return 0

        return _P()


def test_the_ssh_deadline_starts_when_the_job_begins_not_while_it_queues(tmp_path,
                                                                         monkeypatch):
    """The host GPU lock's queue used to sit inside the job's deadline: a queued job
    was killed and charged Failed for time it never ran. The session prints the begin
    marker only once it holds the lock, and the deadline starts there."""
    monkeypatch.setattr(direct, "RUN_HEADROOM_S", 0)
    cfg = _ssh_cfg(tmp_path)                     # timeout 60 -> deadline 60 s
    fake = _QueuedSsh(queue_polls=3)
    status, log = _ssh_run(cfg, fake)
    assert status == jobs.STATUS_SUCCEEDED
    assert fake.waits[:4] == [1.0, 1.0, 1.0, 1.0], "the queue ran on the job's clock"
    assert fake.waits[4] == pytest.approx(60.0), "the deadline did not start whole at begin"


def test_the_session_prints_the_begin_marker_only_under_the_lock(tmp_path):
    """The session script is the ssh job's clock gate, so its quoting is load-bearing:
    with the host lock held by someone else, nothing may print the marker; once the
    lock is ours, the marker comes and the entry runs under it."""
    lock = tmp_path / "gpu.lock"
    lock.write_text("")
    log = tmp_path / "log.txt"
    with open(lock, "w") as fh, open(log, "w") as lf:
        fcntl_mod.flock(fh, fcntl_mod.LOCK_EX)
        proc = subprocess.Popen(
            ["sh", "-c", direct._SESSION_SCRIPT, str(lock), "bash", "-c", "echo job ran"],
            cwd=tmp_path, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(1.0)
        assert jobs.BEGIN_MARKER not in log.read_text(), \
            "the marker showed while the session was still queued"
        fcntl_mod.flock(fh, fcntl_mod.LOCK_UN)
    assert proc.wait(timeout=30) == 0
    text = log.read_text()
    assert jobs.BEGIN_MARKER in text and "job ran" in text
    assert text.index(jobs.BEGIN_MARKER) < text.index("job ran")
