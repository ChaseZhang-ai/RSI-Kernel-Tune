"""The Claude Code backend: ``claude -p`` under ``scripts/agent-claude.sh``.

Serves both roles. It is the only backend that can serve the student: the agent policy
(offline turns, no ssh, no self-submitted jobs, no reads of a sealed reference) is
enforced through Claude Code's own ``.claude/settings.json`` and the PreToolUse hook in
``scripts/agent_policy_hook.py`` -- a denied read is refused, not merely discouraged.

A turn with ``FK_AGENT_DRIVER=goal`` (a task without a reference) is an interactive
``claude`` driven by ``/goal``; ``agent-claude.sh`` hands it to ``scripts/claude-goal.sh``.
"""

from __future__ import annotations

from .base import ScriptBackend


class ClaudeBackend(ScriptBackend):
    name = "claude"
    wire = "anthropic"
    script = "scripts/agent-claude.sh"
    roles = ("teacher", "student")
    policy_support = "hooks"


BACKEND = ClaudeBackend()
