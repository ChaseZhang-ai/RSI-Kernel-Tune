"""BUG-SEED-45 — the agent policy hook let an unparseable event through.

main() failed closed on an unreadable policy file (exit 2, a BLOCKED line on
stderr — a guard that cannot read its orders must not allow everything), but an
event that failed to parse as JSON hit a bare ``return 0``: the same guard went
fail-open on its other input, silently allowing every tool call of a turn whose
event stream broke. An unparseable event now fails closed the same way the
unreadable policy does. Runs the hook as a real subprocess, the way Claude Code
drives it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / "scripts" / "agent_policy_hook.py"

POLICY = {"role": "teacher", "oracle_mode": "blackbox", "offline": False,
          "deny_urls": ["github.com/Org/RefLib"]}


def _hook(policy_path: Path, stdin: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(HOOK), str(policy_path)], input=stdin,
                          text=True, capture_output=True)


def test_an_unparseable_event_fails_closed(tmp_path):
    policy = tmp_path / "p.json"
    policy.write_text(json.dumps(POLICY))
    bad = _hook(policy, '{"tool_name": "Bash", "tool_input": ')   # truncated event
    assert bad.returncode == 2
    assert "BLOCKED" in bad.stderr


def test_a_well_formed_event_is_untouched(tmp_path):
    policy = tmp_path / "p.json"
    policy.write_text(json.dumps(POLICY))
    ok = _hook(policy, json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls"}}))
    assert ok.returncode == 0 and ok.stderr == ""
    blocked = _hook(policy, json.dumps({"tool_name": "WebFetch", "tool_input": {
        "url": "https://github.com/org/RefLib/blob/main/k.cuh"}}))
    assert blocked.returncode == 2 and "reference" in blocked.stderr


def test_empty_stdin_is_still_the_historical_allow(tmp_path):
    # an empty event stream parses as {} and stays a pass (no JSON error to close on)
    policy = tmp_path / "p.json"
    policy.write_text(json.dumps(POLICY))
    assert _hook(policy, "").returncode == 0
