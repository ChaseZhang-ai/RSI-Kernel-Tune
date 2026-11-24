"""The agent backend registry, the two adapters, and what a task may name.

What these tests pin down:

- discovery: both backends are found without importing them, an unknown name lists the
  ones there are, and the interface module is not a backend;
- the Claude command is what it always was, plus ``FK_AGENT_ROLE`` in front (the loop
  gained a role for the usage ledger and the refusal check, not a new invocation);
- the Codex launcher carries every flag the CLI needs, and codex never serves the student,
  a sealed reference, or a /goal session;
- provider.grant picks the endpoint variable from the *wire*, so a table that speaks the
  wrong protocol is refused before a run exists.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from forgekernel import agentbackends, agents, provider
from forgekernel import task as task_mod
from forgekernel.task import AgentSpec, TaskError, load_task
from test_task import MINIMAL, _kb, _task

ROOT = Path(__file__).resolve().parents[1]
CLAUDE_SCRIPT = str(ROOT / "scripts" / "agent-claude.sh")
CODEX_SCRIPT = str(ROOT / "scripts" / "agent-codex.sh")


def _spec(backend: str = "claude", *, model: str = "claude-opus-5-5", effort: str | None = "high",
          provider_name: str = "native", timeout_s: int = 7200) -> AgentSpec:
    return AgentSpec(backend=backend, model=model, effort=effort, provider=provider_name,
                     timeout_s=timeout_s)


def _gateways(tmp_path, text: str, monkeypatch) -> None:
    gc = tmp_path / "gc"
    gc.mkdir(exist_ok=True)
    (gc / "backends.toml").write_text(text)
    monkeypatch.setattr(provider, "global_config_dir", lambda: gc)


# -- discovery -------------------------------------------------------------------------


def test_the_registry_finds_both_backends_without_importing_them():
    assert agentbackends.names() == ("claude", "codex")
    assert set(agentbackends.all_backends()) == {"claude", "codex"}
    assert agentbackends.load("claude").name == "claude"
    assert agentbackends.load("codex").name == "codex"


def test_an_unknown_backend_lists_the_ones_there_are():
    with pytest.raises(agentbackends.BackendError, match=r"claude, codex"):
        agentbackends.load("nope")
    with pytest.raises(agentbackends.BackendError, match=r"registered: claude, codex"):
        agentbackends.get("nope")
    assert agentbackends.get("nope", agentbackends.load("claude")).name == "claude"


def test_the_interface_module_is_not_a_backend():
    assert "base" not in agentbackends.names()
    with pytest.raises(agentbackends.BackendError, match="no agent backend 'base'"):
        agentbackends.load("base")


def test_a_module_whose_backend_is_named_otherwise_is_refused(monkeypatch):
    module = types.ModuleType("fake_backend_mod")
    module.BACKEND = agentbackends.load("claude")
    monkeypatch.setitem(sys.modules, "fake_backend_mod", module)
    monkeypatch.setattr(agentbackends, "modules", lambda: {"other": "fake_backend_mod"})
    with pytest.raises(agentbackends.BackendError, match="BACKEND.name is 'claude'"):
        agentbackends.load("other")


def test_a_module_without_a_backend_is_refused(monkeypatch):
    module = types.ModuleType("empty_backend_mod")
    monkeypatch.setitem(sys.modules, "empty_backend_mod", module)
    monkeypatch.setattr(agentbackends, "modules", lambda: {"empty": "empty_backend_mod"})
    with pytest.raises(agentbackends.BackendError, match="does not define BACKEND"):
        agentbackends.load("empty")


def test_task_reads_its_backends_from_the_registry():
    assert task_mod.BACKENDS == agentbackends.names()
    assert "claude" in task_mod.BACKENDS and "codex" in task_mod.BACKENDS


# -- what each adapter declares -------------------------------------------------------


def test_the_claude_backend_serves_both_roles_through_the_hook_policy():
    backend = agentbackends.load("claude")
    assert backend.roles == ("teacher", "student")
    assert backend.wire == "anthropic" and backend.policy_support == "hooks"
    assert backend.script == "scripts/agent-claude.sh"
    assert backend.script_path() == Path(CLAUDE_SCRIPT)


def test_the_codex_backend_serves_only_the_teacher_and_has_no_policy_hooks():
    backend = agentbackends.load("codex")
    assert backend.roles == ("teacher",)
    assert backend.wire == "openai" and backend.policy_support == "none"
    assert backend.script == "scripts/agent-codex.sh"
    assert backend.script_path() == Path(CODEX_SCRIPT)


# -- the commands ---------------------------------------------------------------------


def test_the_claude_command_is_what_it_always_was_plus_the_role():
    cmd = agents.agent_command(_spec(), "teacher")
    assert cmd == (f"env FK_AGENT_ROLE=teacher bash {CLAUDE_SCRIPT} "
                   f"claude-opus-5-5 high 7200")
    # no effort: the (quoted, empty) argument keeps the script's contract positional
    assert agents.agent_command(_spec(effort=None), "student").endswith(
        f"bash {CLAUDE_SCRIPT} claude-opus-5-5 '' 7200")


def test_the_codex_command_is_the_same_script_contract():
    cmd = agents.agent_command(_spec("codex", model="gpt-5.5-codex"), "teacher")
    assert cmd == (f"env FK_AGENT_ROLE=teacher bash {CODEX_SCRIPT} gpt-5.5-codex high 7200")


def test_a_codex_student_is_refused_by_agent_command():
    with pytest.raises(ValueError, match=r"codex backend cannot serve the student role"):
        agents.agent_command(_spec("codex", model="gpt-5.5-codex"), "student")


def test_a_codex_goal_session_is_refused_with_a_reason():
    with pytest.raises(ValueError, match=r"cannot serve a /goal session"):
        agents.agent_command(_spec("codex", model="gpt-5.5-codex"), "teacher",
                             env={"FK_AGENT_DRIVER": "goal"})
    # the same refusal from the launcher, for a caller that builds the command itself
    r = subprocess.run(["bash", CODEX_SCRIPT, "m", "high", "60"], input="p",
                       capture_output=True, text=True,
                       env={**os.environ, "FK_AGENT_DRIVER": "goal", "FK_AGENT_PY": sys.executable})
    assert r.returncode == 2 and "goal" in r.stderr


def test_the_caller_may_ask_for_another_permission_mode(tmp_path):
    fake = tmp_path / "claude"
    fake.write_text("#!/usr/bin/env bash\ncat >/dev/null\necho \"P $*\"\n")
    fake.chmod(0o755)
    base = {**os.environ, "FK_CLAUDE_BIN": str(fake), "FK_AGENT_PY": sys.executable}
    r = subprocess.run(["bash", str(ROOT / "scripts" / "agent-claude.sh"), "m"],
                       input="p", capture_output=True, text=True,
                       env={**base, "FK_AGENT_PERMISSION_MODE": "plan"})
    assert r.returncode == 0 and r.stdout.strip() == "P -p --model m --permission-mode plan"
    r = subprocess.run(["bash", str(ROOT / "scripts" / "agent-claude.sh"), "m"],
                       input="p", capture_output=True, text=True,
                       env={k: v for k, v in base.items() if k != "FK_AGENT_PERMISSION_MODE"})
    assert r.stdout.strip() == "P -p --model m --dangerously-skip-permissions"


def test_the_codex_launcher_carries_every_documented_flag():
    r = subprocess.run(["bash", CODEX_SCRIPT, "gpt-5.5-codex", "high", "600"], input="",
                       capture_output=True, text=True,
                       env={**os.environ, "FK_AGENT_ECHO": "1", "FK_AGENT_PY": sys.executable})
    assert r.returncode == 0, r.stderr
    argv = r.stdout.splitlines()
    assert argv[1].endswith("timeout-run.py") and argv[2] == "600"
    for flag in ("exec", "--json", "--color", "never",
                 "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check"):
        assert flag in argv, flag
    assert argv[argv.index("-c") + 1] == 'model_reasoning_effort="high"'
    assert argv[argv.index("-m") + 1] == "gpt-5.5-codex"
    assert argv[-1] == "-"          # the prompt comes on stdin


def test_the_codex_launcher_leaves_the_effort_out_when_it_is_empty():
    r = subprocess.run(["bash", CODEX_SCRIPT, "m", "", "60"], input="", capture_output=True,
                       text=True, env={**os.environ, "FK_AGENT_ECHO": "1",
                                       "FK_AGENT_PY": sys.executable})
    assert r.returncode == 0 and "-c" not in r.stdout.splitlines()


def test_the_codex_launcher_says_so_when_codex_is_missing():
    r = subprocess.run(["bash", CODEX_SCRIPT, "m", "high", "60"], input="", capture_output=True,
                       text=True, env={**os.environ, "FK_CODEX_BIN": "/nonexistent/codex",
                                       "FK_AGENT_PY": sys.executable})
    assert r.returncode == 127 and "not installed" in r.stderr


# -- provider: the wire decides -------------------------------------------------------


def test_the_wire_decides_which_endpoint_variable_a_gateway_sets(tmp_path, monkeypatch):
    _gateways(tmp_path, '[oa]\nbase_url = "https://gw.example/openai"\nwire = "openai"\n'
                        'key_env = "OA_KEY"\n', monkeypatch)
    grant = provider.grant("codex", "oa", env={"OA_KEY": "k"})
    assert grant == {"OPENAI_BASE_URL": "https://gw.example/openai",
                     "FK_PROVIDER_KEY_ENV": "OA_KEY"}


def test_a_provider_of_the_wrong_wire_is_refused(tmp_path, monkeypatch):
    _gateways(tmp_path, '[an]\nbase_url = "https://gw.example/anthropic"\n'
                        'wire = "anthropic"\nkey_env = "AN_KEY"\n', monkeypatch)
    with pytest.raises(provider.ProviderError,
                       match=r"speaks anthropic, the codex CLI speaks openai"):
        provider.grant("codex", "an", env={"AN_KEY": "k"})


def test_an_openai_table_is_not_offered_to_claude(tmp_path, monkeypatch):
    _gateways(tmp_path, '[oa]\nbase_url = "https://gw.example/openai"\nwire = "openai"\n'
                        'key_env = "OA_KEY"\n', monkeypatch)
    with pytest.raises(provider.ProviderError,
                       match=r"speaks openai, the claude CLI speaks anthropic"):
        provider.grant("claude", "oa", env={"OA_KEY": "k"})


def test_an_unregistered_backend_has_no_provider_channel_and_lists_the_names(tmp_path,
                                                                           monkeypatch):
    _gateways(tmp_path, '[an]\nbase_url = "https://x"\nwire = "anthropic"\nkey_env = "K"\n',
              monkeypatch)
    with pytest.raises(provider.ProviderError, match=r"registered backends: claude, codex"):
        provider.grant("nope", "an", env={"K": "k"})
    assert provider.grant("nope", "native", env={}) == {}


def test_the_key_never_appears_on_a_codex_command_line(tmp_path, monkeypatch):
    _gateways(tmp_path, '[oa]\nbase_url = "https://gw.example/openai"\nwire = "openai"\n'
                        'key_env = "OA_KEY"\n', monkeypatch)
    cmd = agents.agent_command(_spec("codex", model="gpt-5.5-codex", provider_name="oa"),
                               "teacher", env={"OA_KEY": "topsecret"})
    assert "topsecret" not in cmd and "OPENAI_BASE_URL=https://gw.example/openai" in cmd


# -- what a task may name -------------------------------------------------------------


def _codex(role: str) -> str:
    return f'[agent.{role}]\nbackend = "codex"\nmodel = "gpt-5.5-codex"\n'


def test_codex_as_the_student_is_a_task_problem(tmp_path):
    with pytest.raises(TaskError, match=r"cannot serve the student role \(it serves: \['teacher'\]\)"):
        load_task(_task(tmp_path, MINIMAL + _codex("student")), kb_root=_kb(tmp_path))


def test_codex_as_the_teacher_on_a_source_task_is_fine(tmp_path):
    toml = MINIMAL + ('[reference]\nrun = "reference/run.py:run"\nsource = "reference"\n'
                      + _codex("teacher"))
    t = load_task(_task(tmp_path, toml, files={"reference/run.py": "def run(x, w):\n    return x\n"}),
                  kb_root=_kb(tmp_path))
    assert t.mode == task_mod.MODE_SOURCE and t.teacher.backend == "codex"


def test_codex_on_a_blackbox_task_is_refused(tmp_path):
    toml = MINIMAL + '[reference]\nrun = "reference/run.py:run"\n' + _codex("teacher")
    with pytest.raises(TaskError, match="sealed reference"):
        load_task(_task(tmp_path, toml, files={"reference/run.py": "def run(x):\n    return x\n"}),
                  kb_root=_kb(tmp_path))


def test_codex_on_a_leaderboard_task_with_a_sealed_reference_is_refused(tmp_path):
    toml = (MINIMAL + '[leaderboard]\nsuite = "sol-execbench"\ncard = "leaderboard/card"\n'
            'ref = "leaderboard/ref"\n' + _codex("teacher"))
    task = _task(tmp_path, toml, shapes=False, description=None,
                 files={"leaderboard/card/eval/leaderboard/problem.json":
                        '{"workload": {"uuid": "w", "axes": {"m": 4}}}',
                        "leaderboard/ref/milestone.md": "sealed\n"})
    with pytest.raises(TaskError, match="sealed reference"):
        load_task(task, kb_root=_kb(tmp_path))


def test_codex_on_a_leaderboard_task_without_a_reference_tree_is_fine(tmp_path):
    toml = (MINIMAL + '[leaderboard]\nsuite = "sol-execbench"\ncard = "leaderboard/card"\n'
            + _codex("teacher"))
    task = _task(tmp_path, toml, shapes=False, description=None,
                 files={"leaderboard/card/eval/leaderboard/problem.json":
                        '{"workload": {"uuid": "w", "axes": {"m": 4}}}'})
    t = load_task(task, kb_root=_kb(tmp_path))
    assert t.mode == task_mod.MODE_LEADERBOARD and t.teacher.backend == "codex"


def test_claude_may_serve_every_task(tmp_path):
    toml = MINIMAL + '[reference]\nrun = "reference/run.py:run"\n'
    t = load_task(_task(tmp_path, toml, files={"reference/run.py": "def run(x):\n    return x\n"}),
                  kb_root=_kb(tmp_path))
    assert t.mode == task_mod.MODE_BLACKBOX
    assert (t.teacher.backend, t.student.backend) == ("claude", "claude")


def test_a_generate_task_may_name_codex_but_not_a_goal_session(tmp_path):
    # no reference at all: load is fine (the reference is written by the teacher), and the
    # /goal session that would have written it is refused later, with a reason.
    t = load_task(_task(tmp_path, MINIMAL + _codex("teacher")), kb_root=_kb(tmp_path))
    assert t.mode == task_mod.MODE_GENERATE and t.teacher.backend == "codex"
    with pytest.raises(ValueError, match="goal"):
        agents.agent_command(t.teacher, "teacher", env={"FK_AGENT_DRIVER": "goal"})
