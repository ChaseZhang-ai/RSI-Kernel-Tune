"""The Codex backend (teacher only): ``codex exec`` under ``scripts/agent-codex.sh``.

Why only the teacher
--------------------
Codex has no equivalent of the Claude Code hooks the agent policy is enforced with, so
``policy_support = "none"``: an offline student, no ssh, no self-submitted jobs and no
reads of a sealed reference would all be prompt-level requests rather than refusals.
``rsi_kernel_tune.task._read_agent`` therefore refuses ``backend = "codex"`` for the student
role and for any task whose reference is sealed (blackbox mode, or a leaderboard task
whose ``[leaderboard] ref`` recon tree exists).

``FK_AGENT_DRIVER=goal`` (a /goal session, used when a task has no reference) is a Claude
Code feature (``scripts/claude-goal.sh``); this backend refuses it instead of silently
running something else.

Not exercised on this machine
-----------------------------
Codex is not installed here, so the argv below follows Atrex's ``codex.py`` (Apache-2.0,
see NOTICE) and the Codex CLI's documented ``exec`` mode; ``docs/agent-backends.md`` says
what was and was not confirmed, and ``scripts/agent-codex.sh`` parses the JSON event
stream tolerantly because its shape may change between versions.
"""

from __future__ import annotations

import os

from .base import BackendError, ScriptBackend


class CodexBackend(ScriptBackend):
    name = "codex"
    wire = "openai"
    script = "scripts/agent-codex.sh"
    roles = ("teacher",)
    policy_support = "none"

    def goal_problem(self, env: dict | None = None) -> str | None:
        env = os.environ if env is None else env
        if env.get("FK_AGENT_DRIVER") != "goal":
            return None
        return ("the codex backend cannot serve a /goal session (FK_AGENT_DRIVER=goal): "
                "the session is a Claude Code feature (scripts/claude-goal.sh, Claude "
                "Code's own goal judging). A task without a reference needs backend = "
                "\"claude\", or a reference written another way.")

    def build_command(self, spec, role: str, env: dict | None = None) -> list[str]:
        why = self.goal_problem(env)
        if why is not None:
            raise BackendError(why)
        return super().build_command(spec, role, env)


BACKEND = CodexBackend()
