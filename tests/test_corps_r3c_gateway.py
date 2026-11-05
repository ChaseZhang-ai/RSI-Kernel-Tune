"""Regression pins for the BUG-CORPS-C-06 / C-07 / C-12 fixes (audit wave R3C).

C-07: the script reader follows .py and extensionless entries, reads the scripts a
wrapper names in turn, and sees ``command cd`` / a python ``os.chdir``; the launcher
scripts this repo actually ships must stay finding-free.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from peregrine_kernel.loop import agent_policy as ap
from peregrine_kernel.loop import agentctx as ac

GW = "https://gw.example/llm"


def _env(tmp_path: Path, **extra) -> dict:
    return {"HOME": str(tmp_path / "home"), **extra}


# -- BUG-CORPS-C-07 ------------------------------------------------------------------------

def test_the_reader_follows_a_chain_of_wrapper_scripts(tmp_path):
    """agent-claude.sh execs claude-goal.sh execs claude-goal.py: the second hop is
    read too, and its --setting-sources/--bare shapes are findings even though the
    top-level command never spells them."""
    hop2 = tmp_path / "second-hop.py"
    hop2.write_text("subprocess.run(['claude', '-p', '--bare'])\n", encoding="utf-8")
    hop1 = tmp_path / "first-hop.sh"
    hop1.write_text(f'exec python3 "{hop2}" "$@"\n', encoding="utf-8")
    findings = ap.command_findings([f"bash {hop1} student"], {})
    assert any("second-hop.py" in f and "--bare" in f for f in findings)


def test_a_python_wrappers_os_chdir_is_a_finding(tmp_path):
    wrapper = tmp_path / "goalwrap.py"
    wrapper.write_text("import os\nos.chdir('/tmp/elsewhere')\n"
                       "subprocess.run(['claude', '-p'])\n", encoding="utf-8")
    findings = ap.command_findings([f"{sys.executable} {wrapper}"], {})
    assert any("os.chdir" in f for f in findings)


def test_a_string_literal_cd_in_a_python_wrapper_is_not_its_own_behavior(tmp_path):
    """scripts/claude-goal.py writes a bash launcher whose text contains
    ``cd <dir>``: that is the generated script's behavior, not the python file's, and
    must not refuse every /goal session."""
    wrapper = tmp_path / "claude-goal-like.py"
    wrapper.write_text('launcher.write_text("cd /srv/agent-ws || exit 97\\n")\n'
                       'subprocess.run(["claude", "-p"])\n', encoding="utf-8")
    assert ap.command_findings([f"{sys.executable} {wrapper}"], {}) == []


def test_command_cd_and_a_nested_scripts_cd_are_findings(tmp_path):
    nested = tmp_path / "inner.sh"
    nested.write_text("command cd /srv/real && claude -p\n", encoding="utf-8")
    outer = tmp_path / "outer.sh"
    outer.write_text(f'bash {nested}\n', encoding="utf-8")
    assert ap.command_findings(["command cd /x && claude -p"], {})
    findings = ap.command_findings([f"bash {outer}"], {})
    assert any("inner.sh" in f for f in findings)


def test_extensionless_and_py_entries_are_read(tmp_path):
    bare = tmp_path / "run-agent"
    bare.write_text("cd /srv/elsewhere && claude -p\n", encoding="utf-8")
    assert ap.command_findings(["./run-agent"], {}, cwd=str(tmp_path))
    assert ap.command_findings([f"bash {bare}"], {})


def test_a_binary_entry_is_skipped_not_scanned(tmp_path):
    """The installed claude executable is a compiled binary: pattern hits inside it
    would refuse every launch, so a NUL byte in the head ends the read."""
    binary = tmp_path / "claude-like"
    binary.write_bytes(b"BUN\x00" + b"\0\0\0cd /x\n" + b"payload")
    assert ap.command_findings([f"{binary} -p"], {}) == []


def test_the_repo_s_own_launcher_scripts_stay_finding_free():
    """scripts/agent-claude.sh -> claude-goal.sh -> claude-goal.py is the production
    chain; the deeper read must not start refusing it."""
    scripts = Path(ap.__file__).resolve().parents[3] / "scripts"
    agent_sh = scripts / "agent-claude.sh"
    if not agent_sh.is_file():        # a source tree without scripts/ (packaged)
        pytest.skip("no scripts/ beside this checkout")
    findings = ap.command_findings([f"bash {agent_sh} claude-opus-5[1m]"], {},
                                   cwd=str(scripts))
    assert findings == []
