"""Regression pins for the BUG-CORPS-C-06 / C-07 / C-12 fixes (audit wave R3C).

C-06: a ``--model $VAR`` whose variable is assigned somewhere readable must resolve to
legs even when the variable is not one of the conventional flag names, and a variable
assigned nowhere at all must raise rather than silently produce no leg (a gate with
``legs=[]`` is blind).

C-07: the script reader follows .py and extensionless entries, reads the scripts a
wrapper names in turn, and sees ``command cd`` / a python ``os.chdir``; the launcher
scripts this repo actually ships must stay finding-free.

C-12: an ``unset ANTHROPIC_BASE_URL`` governs from its statement on and stops at a
re-declaration of the variable (whose value is the gateway) — not at the file level,
which made a mixed wrapper's gateway legs invisible.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rsi_kernel_tune.loop import agent_policy as ap
from rsi_kernel_tune.loop import agentctx as ac

GW = "https://gw.example/llm"


def _env(tmp_path: Path, **extra) -> dict:
    return {"HOME": str(tmp_path / "home"), **extra}


# -- BUG-CORPS-C-06 ------------------------------------------------------------------------

def test_a_model_variable_outside_the_flag_whitelist_still_yields_legs():
    """A wrapper that holds the model in any assigned variable — not only
    MODEL_FLAG / STUDENT_FLAG / TEACHER_FLAG — must produce legs. The old whitelist
    resolved nothing and the gate ran with legs=[] (the audited shape)."""
    facts = ac.inspect_agent_cmd(
        "ROLE_MODEL=claude-opus-5\nclaude -p --model $ROLE_MODEL", env={})
    assert [leg.model for leg in facts.legs] == ["claude-opus-5"]
    assert facts.legs[0].variable == "ROLE_MODEL"


def test_a_bare_id_behind_a_gateway_via_an_unlisted_variable_is_a_finding(tmp_path):
    """The end-to-end failure the silent skip caused: a bare claude id riding a
    non-whitelisted variable was no finding at all."""
    facts = ac.inspect_agent_cmd(
        f"GATEWAY_MODEL=claude-opus-5\nANTHROPIC_BASE_URL={GW}\n"
        f"claude -p --model $GATEWAY_MODEL", env=_env(tmp_path))
    findings = ac.window_findings(facts)
    assert len(findings) == 1 and "claude-opus-5" in findings[0]


def test_a_model_variable_assigned_nowhere_raises():
    """No assignment anywhere and not in the environment: the leg's model is
    unknowable, so the reader refuses instead of skipping it (fail-closed)."""
    with pytest.raises(ValueError, match=r"\$UNSEEN_VAR"):
        ac.inspect_agent_cmd("claude -p --model $UNSEEN_VAR", env={})


def test_a_model_variable_only_in_the_environment_is_a_leg(tmp_path):
    facts = ac.inspect_agent_cmd("claude -p --model $EXPORTED_MODEL",
                                 env=_env(tmp_path, EXPORTED_MODEL="claude-opus-5"))
    assert [leg.model for leg in facts.legs] == ["claude-opus-5"]


def test_a_composite_model_token_resolves_its_variable(tmp_path):
    """``$FLAG[1m]`` used to lose the name to a crude strip and evaporated the same
    way; the reference reader must resolve the variable inside the token."""
    facts = ac.inspect_agent_cmd(
        "STUDENT_FLAG=claude-opus-5\nclaude -p --model $STUDENT_FLAG[1m]", env={})
    assert [leg.model for leg in facts.legs] == ["claude-opus-5[1m]"]


def test_a_variable_filled_from_the_wrappers_arguments_yields_no_leg_but_no_raise():
    """``MODEL=${1:?model}`` (scripts/agent-claude.sh's shape) is resolved from the
    invocation's arguments at runtime: the name is known, the value is not readable
    statically — no leg, and no refusal of the standard production wrapper."""
    facts = ac.inspect_agent_cmd('MODEL=${1:?model, e.g. claude-opus-5-5}\n'
                                 'exec claude -p --model "$MODEL"', env={})
    assert facts.legs == []


# -- BUG-CORPS-C-12 ------------------------------------------------------------------------

def test_unset_then_redeclared_gateway_is_visible(tmp_path):
    """A source-level ``unset`` followed by a re-declaration: the leg after the
    re-declaration runs on the (declared) gateway and is a finding for a bare id. The
    whole-source flag used to call every leg first-party and hid it."""
    facts = ac.inspect_agent_cmd(
        "unset ANTHROPIC_BASE_URL\n"
        "claude -p --model claude-opus-5\n"
        f"export ANTHROPIC_BASE_URL={GW}\n"
        "claude -p --model claude-sonnet-5", env=_env(tmp_path))
    legs = {leg.model: leg for leg in facts.legs}
    assert legs["claude-opus-5"].first_party and not legs["claude-opus-5"].on_gateway
    assert legs["claude-sonnet-5"].on_gateway and not legs["claude-sonnet-5"].first_party
    findings = ac.window_findings(facts)
    assert [f for f in findings if "claude-sonnet-5" in f]


def test_an_unset_inside_a_branch_does_not_outlive_esac(tmp_path):
    """The reader cannot know which case branch ran, so an unset inside one is not
    trusted past ``esac``: the later leg falls back to the environment (gateway there
    -> finding), instead of inheriting a first-party state it may not have."""
    script = tmp_path / "mixed.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'case "$ROLE" in\n'
        "  teacher) unset ANTHROPIC_BASE_URL ;;\n"
        "esac\n"
        "claude -p --model claude-opus-5\n", encoding="utf-8")
    facts = ac.inspect_agent_cmd(f"bash {script} student",
                                 env=_env(tmp_path, ANTHROPIC_BASE_URL=GW))
    assert facts.legs and not facts.legs[0].first_party
    assert len(ac.window_findings(facts)) == 1


def test_a_plain_source_unset_still_makes_the_later_leg_first_party(tmp_path):
    """The good shape is unchanged: unset at the top, every claude call after it runs
    the official endpoint."""
    facts = ac.inspect_agent_cmd(
        "unset ANTHROPIC_BASE_URL\nclaude -p --model claude-opus-5",
        env=_env(tmp_path, ANTHROPIC_BASE_URL=GW))
    assert ac.window_findings(facts) == []


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
