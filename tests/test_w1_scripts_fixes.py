"""W1 scripts+docs line: regression tests for the audit-corps fixes.

One test per finding that needs one; the findings live in the audit corps' W1 list
(blackbox-guard cases stay in test_blackbox_guard.py, self_evolve's in
test_self_evolve.py). Everything here is stdlib-only in-process: no tmux, no GPU, no
network.
"""
from __future__ import annotations

import importlib.util
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- W1-LIF-04: a refusal hands the reply back on stdout --------------------------------

def test_a_refusal_hands_the_reply_back_on_stdout(tmp_path, monkeypatch, capsys):
    """The docstring contract: exit 75/76/77 all come with the reply on stdout. The
    refusal branch used to write only stderr, so the caller that classifies the exit
    code got no turn output at all."""
    mod = _load("claude_goal_w1", "scripts/claude-goal.py")
    cfg = tmp_path / "cfg"
    (cfg / "projects" / "p").mkdir(parents=True)
    (cfg / "projects" / "p" / "sidw1.jsonl").write_text(
        json.dumps({"type": "system", "subtype": "model_refusal_fallback",
                    "apiRefusalCategory": "safety", "originalModel": "orig",
                    "fallbackModel": "fb"}) + "\n", encoding="utf-8")

    leg = mod.GoalLeg("claude-test-1", "test-model", [], "do the thing", env={})
    leg.cfg, leg.sid = cfg, "sidw1"
    leg.turn_dir = tmp_path
    leg.reply = tmp_path / "reply.md"
    leg.reply.write_text("the turn's reply text", encoding="utf-8")
    leg.spec = {"condition": "c", "check": "true"}
    monkeypatch.setattr(leg, "setup_turn", lambda: None)
    monkeypatch.setattr(leg, "launch", lambda: None)
    monkeypatch.setattr(leg, "watch", lambda tr: ("refusal", '{"category": "safety"}'))

    rc = leg.run()
    out = capsys.readouterr()
    assert rc == 77, (rc, out.out, out.err)
    assert "the turn's reply text" in out.out          # the contract: reply on stdout
    assert "SAFEGUARD REFUSAL" in out.err              # the explanation on stderr
    assert json.loads((tmp_path / "result.json").read_text().splitlines()[-1])[
        "outcome"] == "refusal"


# --- W1-LIF-05: a signal-killed child exits 128+n, not a wrapped negative ---------------

def test_timeout_run_maps_a_signal_killed_child_to_128_plus_n():
    """proc.wait() returns a negative status for a child killed by a signal;
    sys.exit(-11) lands as 245 and reads as an ordinary program exit. 128+n (the
    convention the signal-forwarding branch already speaks) says what actually
    happened."""
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "timeout-run.py"), "30",
         sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 128 + signal.SIGKILL, (r.returncode, r.stdout, r.stderr)
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "timeout-run.py"), "30",
         sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 128 + signal.SIGTERM, (r.returncode, r.stdout, r.stderr)


def test_timeout_run_still_returns_124_on_expiry_and_zero_on_clean_exit():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "timeout-run.py"), "1",
         sys.executable, "-c", "import time; time.sleep(30)"],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 124, (r.returncode, r.stdout, r.stderr)
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "timeout-run.py"), "30",
         sys.executable, "-c", "pass"],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
# --- W1-LIF-06: a recipe flag with no value is a FAIL verdict, not a traceback ----------

@pytest.mark.parametrize("script,flag,verdict", [
    ("skills/timeline/pm_timeline_run.py", "--interval", "PMTL-FAIL reason=bad-args"),
    ("skills/stall/stall_price_run.py", "--pm-interval", "STALL-FAIL reason=bad-args"),
])
def test_a_trailing_recipe_flag_without_a_value_is_a_fail_verdict(script, flag, verdict):
    """Both scripts state "the verdict travels in the line, not the exit code"; a
    `--interval` / `--pm-interval` given last used to die on a bare IndexError
    traceback instead."""
    r = subprocess.run([sys.executable, str(REPO / script), flag],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert verdict in r.stdout
    assert "Traceback" not in r.stderr
    assert "IndexError" not in r.stderr


# --- W1-LIF-07 / SEED-27: scripts/self_evolve.py chain robustness -----------------------

from test_task import _kb  # noqa: E402  (the shared hermetic KB helper)
from peregrine_kernel.task import load_task  # noqa: E402

SE = _load("self_evolve_w1", "scripts/self_evolve.py")
KERNEL = "import torch\n\ndef run(x, w):\n    return x * w\n"


def _gen_task(root: Path, name: str = "rmsnorm") -> Path:
    d = root / "tasks" / name
    d.mkdir(parents=True)
    (d / "task.toml").write_text('[task]\nlanguage = "cutedsl"\narch = "sm100"\n'
                                 '[agent.teacher]\nmodel = "claude-opus-5-5"\neffort = "high"\n'
                                 '[budget]\nhours = 2.5\n')
    (d / "description.md").write_text("y = x * w\n")
    (d / "shapes.json").write_text('[{"name": "s", "axes": {"m": 4}}]')
    return d


def _evolve(tmp_path, monkeypatch):
    """A two-command stand-in harness: `new`/`run` return the configured exit codes,
    the best-kernel read reports the configured gains. A factory, not a fixture: a
    test may stand up several independent chains."""
    kb = _kb(tmp_path)
    monkeypatch.setattr(SE, "load_task", lambda d: load_task(d, kb_root=kb))
    monkeypatch.setattr(SE.launch, "repo_root", lambda: tmp_path)
    # new_seq/run_seq hand each generation's exit code in order; an empty sequence
    # means every call succeeds (0)
    state = {"new_seq": [], "run_seq": [], "ran_new": [], "ran_run": [], "gains": []}

    def _cmd_new(argv):
        state["ran_new"].append(Path(argv[0]))
        return state["new_seq"].pop(0) if state["new_seq"] else 0

    def _cmd_run(argv):
        state["ran_run"].append(Path(argv[0]))
        return state["run_seq"].pop(0) if state["run_seq"] else 0

    monkeypatch.setattr(SE.launch, "cmd_new", _cmd_new)
    monkeypatch.setattr(SE.launch, "cmd_run", _cmd_run)
    monkeypatch.setattr(SE, "latest_run", lambda name: SimpleNamespace(
        dir=tmp_path / "run", operators_dir=tmp_path / "run" / "operators",
        ws=tmp_path / "run" / "ws"))
    monkeypatch.setattr(SE, "REPO_ROOT", tmp_path)

    def best_of(run, task):
        g = state["gains"][len(state["ran_new"]) - 1]
        if g is None:
            return None
        speedup, ratios = g if isinstance(g, tuple) else (g, [])
        return {"sha": "abc123def456", "speedup": speedup, "ratios": ratios,
                "kernel": KERNEL}
    monkeypatch.setattr(SE.launch, "best_of", best_of)
    orig = SE.next_generation
    monkeypatch.setattr(SE, "next_generation",
                        lambda task, best, dest, step, **kw: orig(
                            task, best, dest, step, **kw) and dest)
    return tmp_path, state


def test_a_harness_failure_is_not_a_natural_chain_stop(tmp_path, monkeypatch, capsys):
    """W1-LIF-07: `peregrine new`/`peregrine run` failing used to exit 0 at gen>1 —
    byte-for-byte the natural chain break. An infrastructure failure gets its own
    reason line and exit 3."""
    root, state = _evolve(tmp_path, monkeypatch)
    first = _gen_task(root)
    state["gains"] += [1.30]              # gen 1 clears the floor, writes gen 2
    state["new_seq"] += [0, 2]            # gen 1's new succeeds, gen 2's fails
    assert SE.main([str(first), "--max-gen", "4"]) == 3
    assert "not a chain stop" in capsys.readouterr().out
    assert len(state["ran_new"]) == 2     # gen 1 new + the failed gen 2 new

    root2, state2 = _evolve(tmp_path, monkeypatch)
    first2 = _gen_task(root2, name="rms2")   # a fresh chain beside the first
    state2["gains"] += [1.30, 1.30]       # gen 2 runs, its `peregrine run` fails
    state2["run_seq"] += [0, 5]
    rc = SE.main([str(first2), "--max-gen", "4"])
    assert rc == 3, rc
    assert len(state2["ran_run"]) == 2    # gen 1 run + the failed gen 2 run


def test_a_natural_chain_break_still_exits_zero(tmp_path, monkeypatch):
    """The other half of W1-LIF-07: a generation that completes with no confirmed best
    (or a gain inside the floor) is a RESULT — it keeps exiting 0."""
    root, state = _evolve(tmp_path, monkeypatch)
    state["gains"] += [1.30, None]        # gen 2 completes without a confirmed best
    assert SE.main([str(_gen_task(root)), "--max-gen", "4"]) == 0
