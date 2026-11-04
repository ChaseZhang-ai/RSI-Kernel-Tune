"""W1 scripts+docs line: regression tests for the audit-corps fixes.

One test per finding that needs one; the findings live in the audit corps' W1 list
(blackbox-guard cases stay in test_blackbox_guard.py, self_evolve's in
test_self_evolve.py). Everything here is stdlib-only in-process: no tmux, no GPU, no
network.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

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
