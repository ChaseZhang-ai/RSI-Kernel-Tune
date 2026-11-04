"""Regression tests for the audit-corps fix wave 5 (the launch/monitor/resume/agent-policy
lifecycle surface): the run verdict never reads DONE out of the stop word, the
correctness monitor reports a hook that never injected and recovers a cut-short log,
the reopen plan keeps companion pairs together, the agent policy sees every agent
command, the monitor hook leaves a trace when it cannot write, and the policy hook
fails closed on an unreadable event."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_launch import _confirmed, _loop_ends, _start, _task, _Teacher  # noqa: E402

from peregrine_kernel import launch, monitor  # noqa: E402
from peregrine_kernel.runs import list_runs  # noqa: E402


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A scratch place for runs and locks; the harness itself stays the real repository."""
    monkeypatch.setenv("PEREGRINE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("PEREGRINE_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("PEREGRINE_OPERATORS_DIR", raising=False)
    monkeypatch.setenv("PK_DELIVERY", "local")
    monkeypatch.setenv("PK_GPU", "0")
    monkeypatch.setattr(launch, "probe_ncu", lambda task, op_dir: True)
    return tmp_path


# --- BUG-SEED-13: a correctness alarm (or a missing confirmation) on a target stop ----
# must end the run stopped, not done ---------------------------------------------------


def _drive_to_the_end(repo, monkeypatch, *, stop, confirm):
    """`peregrine new` + `peregrine run` with the loop stood in: it ends with ``stop``
    and ``confirm`` as the final confirmation exactly as given (no stand-in defaults)."""
    from peregrine_kernel.loop import cli as loop_cli
    task_dir = _task(repo)
    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", _Teacher().method)
    monkeypatch.setattr(launch, "calibrate", lambda task, op_dir: {
        "match": True, "tolerance": {"y": 2e-6}, "problems": []})
    monkeypatch.setattr(loop_cli, "main", _loop_ends(stop=stop, confirm=confirm))
    return _start(task_dir), list_runs()[0]


def test_a_target_stop_with_a_correctness_alarm_is_not_done(repo, monkeypatch):
    """The loop stopped at the target, but the confirmed kernel's correctness monitor
    raised an alarm: `peregrine run` exits 2 and the run is stopped — DONE requires a
    confirmation with no alarm, whatever the stop word says."""
    rc, run = _drive_to_the_end(repo, monkeypatch, stop="target",
                                confirm=_confirmed(8.0, monitor="alarm"))
    assert rc == 2
    rec = run.record()
    assert rec["status"] == "stopped" and rec["stop"] == "target"
    assert rec["target_met"] is False


def test_a_target_stop_on_a_confirmed_best_is_done(repo, monkeypatch):
    """The mirror side of the contract: a target stop whose final confirmation meets
    the target with no alarm still ends done (the stop word alone decides nothing,
    either way)."""
    rc, run = _drive_to_the_end(repo, monkeypatch, stop="target", confirm=_confirmed(8.0))
    assert rc == 0
    rec = run.record()
    assert rec["status"] == "done" and rec["stop"] == "target" and rec["target_met"] is True


def test_a_target_stop_without_any_confirmation_is_not_done(repo, monkeypatch):
    """A target stop with no best_confirmation.json at all: `met` is false, so the run
    ends stopped instead of reading DONE out of the stop word."""
    from peregrine_kernel.loop import cli as loop_cli

    def main(argv):
        from peregrine_kernel.loop.cli import SUMMARY_FILE
        d = Path(os.environ["PEREGRINE_OPERATORS_DIR"]) / argv[0]
        (d / SUMMARY_FILE).write_text(json.dumps({"stop": "target"}))
        return 0

    task_dir = _task(repo)
    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", _Teacher().method)
    monkeypatch.setattr(launch, "calibrate", lambda task, op_dir: {
        "match": True, "tolerance": {"y": 2e-6}, "problems": []})
    monkeypatch.setattr(loop_cli, "main", main)
    assert _start(task_dir) == 2
    rec = list_runs()[0].record()
    assert rec["status"] == "stopped" and rec["stop"] == "target" and rec["target_met"] is False


# --- BUG-SEED-14: a hook that never injected is reported, not misattributed -----------


def test_classify_reports_a_hook_that_never_injected():
    """Zero hook-<pid>.json files means sitecustomize never ran: the input reshaping and
    the output checks did not execute, so the verdict is a warn naming the injection —
    not an ok that explains the silence as a bench that never imports the kernel."""
    result = {"complete": True, "profiles": [
        {"profile": "base", "card_pass": True, "hook": {"processes": 0, "calls": []}},
        {"profile": "seed1", "card_pass": True, "hook": {"processes": 0, "calls": []}},
    ]}
    v = monitor.classify(result)
    assert v["status"] == "warn"
    assert v["ran"] == [] and v["not_applicable"] == []
    whats = [f["what"] for f in v["findings"]]
    assert sum("did not inject" in w for w in whats) == 2
    assert not any("does not import the kernel" in w for w in whats)
    assert not any("did not change" in w for w in whats)


def test_classify_reports_one_profile_the_hook_never_injected():
    """A profile whose bench ran but whose hook did not inject is dropped from the
    counted profiles and named, while the injected base stays counted."""
    result = {"complete": True, "profiles": [
        {"profile": "base", "card_pass": True,
         "hook": {"processes": 1, "calls": [{"fn": "kernel", "i": 0, "outputs": []}]}},
        {"profile": "seed1", "card_pass": True, "hook": {"processes": 0, "calls": []}},
    ]}
    v = monitor.classify(result)
    assert v["status"] == "warn" and v["ran"] == ["base"]
    assert any(f["profile"] == "seed1" and f["severity"] == "warn"
               and "did not inject" in f["what"] for f in v["findings"])


def test_classify_keeps_the_old_reading_for_records_without_a_process_count():
    """Records from before the runner counted its hook files carry no ``processes``:
    a call-less base stays the ordinary info finding, not an injection failure."""
    result = {"complete": True, "profiles": [
        {"profile": "base", "card_pass": True, "hook": {"calls": []}},
    ]}
    v = monitor.classify(result)
    assert v["status"] == "ok" and v["ran"] == ["base"]
    assert any("no kernel call was observed" in f["what"] for f in v["findings"])


# --- BUG-SEED-15: a final marker without its JSON line falls back to the rebuild ------


def test_parse_log_rebuilds_when_the_final_marker_has_no_json():
    """The runner prints the final marker and its JSON on two lines: a job killed
    between them leaves the marker without a result line. The per-profile blocks
    still on the log are recovered instead of the whole parse reading as None."""
    log = (f"{monitor.PROFILE_MARKER}\n"
           + json.dumps({"profile": "base", "card_pass": True,
                         "hook": {"processes": 1, "calls": []}}) + "\n"
           + f"{monitor.PROFILE_MARKER}\n"
           + json.dumps({"profile": "seed1", "card_pass": False,
                         "hook": {"processes": 1, "calls": [], "perturbed": 3}}) + "\n"
           + f"{monitor.MARKER}\n")          # the final JSON line never landed
    res = monitor.parse_log(log)
    assert res is not None and res["complete"] is False
    assert [r["profile"] for r in res["profiles"]] == ["base", "seed1"]
    # and the recovered findings survive classification: the seed1 FAIL is not lost
    v = monitor.classify(res)
    assert v["status"] == "alarm"
    assert any(f["profile"] == "seed1" and f["severity"] == "alarm"
               for f in v["findings"])


def test_parse_log_with_a_complete_final_result_is_unchanged():
    """A log whose final JSON line landed parses as before: the final result wins,
    complete, with the skipped list the rebuild cannot know."""
    prof = json.dumps({"profile": "base", "card_pass": True, "hook": {"calls": []}})
    final = {"version": 1, "profiles": [{"profile": "base", "card_pass": True,
                                         "hook": {"calls": []}}],
             "skipped": [{"profile": "seed1", "why": "budget"}], "complete": True}
    log = (f"{monitor.PROFILE_MARKER}\n{prof}\n"
           f"{monitor.MARKER}\n{json.dumps(final)}\n")
    res = monitor.parse_log(log)
    assert res == final and res["complete"] is True


def test_parse_log_without_any_final_marker_still_rebuilds():
    """The pre-existing cut-short shape (no final marker at all) keeps rebuilding."""
    log = (f"{monitor.PROFILE_MARKER}\n"
           + json.dumps({"profile": "base", "card_pass": True, "hook": {"calls": []}}) + "\n")
    res = monitor.parse_log(log)
    assert res is not None and res["complete"] is False
    assert [r["profile"] for r in res["profiles"]] == ["base"]


# --- BUG-SEED-16: companion pairs are not split across a reopen -----------------------


def test_a_one_sided_companion_declaration_reopens_both_sides():
    """D2 declares ⊗ D3 and D3 declares nothing back: reopening D2 pulls D3 in too —
    companions land in one commit, so half a pair kept would validate a tree the
    other half rewrites."""
    from peregrine_kernel.resume import Decision, compute_reopen
    traj = [Decision("D1"), Decision("D2", companions={"D3"}), Decision("D3")]
    plan = compute_reopen(traj, {"D2"})
    assert plan.reopen == ["D2", "D3"] and plan.kept == ["D1"]


def test_a_one_sided_companion_declaration_reopens_the_declarer_too():
    """The direction that always worked stays working: the companion (D3) declaring ⊗
    D2 is pulled in when D2 is reopened — and now so is anything D2 named."""
    from peregrine_kernel.resume import Decision, compute_reopen
    traj = [Decision("D1"), Decision("D2"), Decision("D3", companions={"D2"})]
    plan = compute_reopen(traj, {"D2"})
    assert plan.reopen == ["D2", "D3"] and plan.kept == ["D1"]


def test_a_two_sided_companion_declaration_behaves_as_before():
    """Both sides declaring ⊗ each other: the symmetric closure changes nothing."""
    from peregrine_kernel.resume import Decision, compute_reopen
    traj = [Decision("D1"), Decision("D2", companions={"D3"}),
            Decision("D3", companions={"D2"})]
    plan = compute_reopen(traj, {"D2"})
    assert plan.reopen == ["D2", "D3"] and plan.kept == ["D1"]
    plan = compute_reopen(traj, {"D3"})
    assert plan.reopen == ["D2", "D3"] and plan.kept == ["D1"]


def test_companion_closure_does_not_reach_past_the_changed_set():
    """A decision that declares no companion and is not changed stays kept even when a
    companion pair elsewhere reopens: the closure is the pair, not the trajectory."""
    from peregrine_kernel.resume import Decision, compute_reopen
    traj = [Decision("D1"), Decision("D2", companions={"D3"}), Decision("D3"),
            Decision("D4")]
    plan = compute_reopen(traj, {"D2"})
    assert plan.reopen == ["D2", "D3"] and plan.kept == ["D1", "D4"]


def _companion_milestone(d03_depends: str = "- Depends on: ⇐ D0.1") -> str:
    """A three-decision milestone whose D0.2 declares ⊗ D0.3; ``d03_depends`` decides
    whether D0.3 declares the companion edge back."""
    return f"""# op milestone

## 0. Problem statement and target
x

## 1. Problem contract (Math Spec)
x

## 2. Decision DAG + phase order
x

## 3. Decision details
### D0.1 root
- Action: a
- Purpose: b
- Trigger condition: c
- Cost: d
- Depends on: ⇒ D0.2, D0.3

### D0.2 producer
- Action: a
- Purpose: b
- Trigger condition: c
- Cost: d
- Depends on: ⇐ D0.1 | ⊗ D0.3

### D0.3 consumer
- Action: a
- Purpose: b
- Trigger condition: c
- Cost: d
{d03_depends}

M0 Gate:
- Mechanism-verification line: tensor pipe active % ≥60%

## 4. Known dead ends
x

## 5. Global correctness / performance gate
x

## 6. Rules
x

## Worked Example
x
"""


def test_dependency_warnings_flag_a_one_sided_companion_declaration():
    """⊗ declared on one side only is warned, in the style of the ⇒ echo check: the
    reopen plan can only keep the pair together when both sides declare it."""
    from peregrine_kernel.milestone import dependency_warnings
    w = dependency_warnings(_companion_milestone())
    assert w == ["§3 D0.2 declares ⊗ D0.3 but D0.3 does not list ⊗ D0.2"
                 " — companions land in one commit, and the reopen plan can only"
                 " keep the pair together when both sides declare it"]


def test_dependency_warnings_stay_silent_on_a_two_sided_companion_declaration():
    from peregrine_kernel.milestone import dependency_warnings
    assert dependency_warnings(
        _companion_milestone("- Depends on: ⇐ D0.1 | ⊗ D0.2")) == []


# --- BUG-SEED-17: every agent command is seen by the policy gate -----------------------


def test_command_findings_see_the_teacher_command_too():
    """All agent commands used to share the "agent command" source, so the dedup key
    collided and the second command's findings were dropped whole: the teacher leg —
    always the second — could carry --settings past the gate unseen."""
    from peregrine_kernel.loop import agent_policy as ap
    student = 'claude -p "optimize the kernel"'
    teacher = 'claude --settings /tmp/evil.json -p "recon the reference"'
    assert ap.command_findings([student, teacher], {}) == [
        "agent command: --settings outranks the project settings the "
        "agent policy is delivered in, and could turn its sandbox or hooks off"]
    findings = ap.command_findings(["claude -p --bare", teacher], {})
    assert len(findings) == 2
    assert "--bare" in findings[0] and "--settings" in findings[1]


def test_command_findings_report_an_identical_command_once():
    from peregrine_kernel.loop import agent_policy as ap
    assert len(ap.command_findings(["claude -p --bare", "claude -p --bare"], {})) == 1


def test_command_findings_still_read_a_shared_script_once(tmp_path):
    """A script file named by two commands is scanned once: its (path, text) pair is
    what repeats, and the dedup keeps working on it."""
    from peregrine_kernel.loop import agent_policy as ap
    script = tmp_path / "agent.sh"
    script.write_text("claude -p --bare\n")
    findings = ap.command_findings([f"bash {script}", f"bash {script}"], {})
    assert len(findings) == 1
    assert findings[0].startswith(str(script)) and "--bare" in findings[0]
