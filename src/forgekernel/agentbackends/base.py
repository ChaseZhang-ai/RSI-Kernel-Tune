"""One agent CLI the harness can drive: the interface every backend implements.

A task names its backend (``[agent] backend = "<name>"``); ``agentbackends/__init__.py``
finds the module named after it (``src/forgekernel/agentbackends/<name>.py``), and that
module exports one instance ``BACKEND``. ``forgekernel.agents.agent_command`` assembles
what the loop runs for a role turn; ``forgekernel.provider`` asks the same adapter what
points its CLI at another endpoint.

What a backend may and may not do is **declared, not inferred**:

- ``roles``: the roles of the two-agent loop it can serve. The student is the tighter
  role -- the agent policy's offline turns, no ssh, no self-submission and no reads of a
  sealed reference are enforced through Claude Code's own settings and hooks -- so a
  backend with ``policy_support == "none"`` serves only the teacher, and only on a task
  without a sealed reference (``forgekernel.task._read_agent`` refuses the rest).
- ``policy_support``: ``"hooks"`` when ``scripts/agent_policy_hook.py`` can be wired into
  the CLI's settings so a denied read is *refused*; ``"none"`` when nothing but the prompt
  stands behind the policy.
- ``wire``: the protocol its provider endpoint speaks (``"anthropic"`` / ``"openai"``);
  ``provider.grant`` only hands it a table of that wire.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:       # task.py imports this package, so never import it back at runtime
    from forgekernel.task import AgentSpec

ROLES = ("teacher", "student")
POLICY_SUPPORT = ("hooks", "none")


class BackendError(ValueError):
    """A backend that is not registered, or that cannot do what was asked of it."""


class AgentBackend(ABC):
    """One agent CLI: a module-level singleton in its own module (``BACKEND``)."""

    name: ClassVar[str]                 # "claude" / "codex": the module's name, too
    wire: ClassVar[str]                 # the provider protocol: "anthropic" / "openai"
    script: ClassVar[str]               # the launcher, repo-relative
    roles: ClassVar[tuple[str, ...]]    # the roles it may serve
    policy_support: ClassVar[str]       # "hooks" / "none" (see the module docstring)

    def script_path(self) -> Path:
        from forgekernel.paths import repo_root
        return repo_root() / self.script

    @abstractmethod
    def build_command(self, spec: "AgentSpec", role: str,
                      env: dict | None = None) -> list[str]:
        """The argv of one ``role`` turn -- **without** the prompt, which the loop writes
        to the child's stdin and which the CLI answers on stdout."""

    def grant(self, provider: str, env: dict | None = None, *,
              root: Path | None = None) -> dict[str, str]:
        """The environment assignments that point this CLI at provider table
        ``provider`` (``native`` declares none); ProviderError when it does not fit or
        its key is not set."""
        from forgekernel.provider import grant_for_wire
        return grant_for_wire(self.wire, self.name, provider, env, root=root)

    def goal_problem(self, env: dict | None = None) -> str | None:
        """Why a /goal session cannot be served by this backend; None when it can."""
        return None


class ScriptBackend(AgentBackend):
    """A backend whose turn is one of the repository's launcher scripts.

    Both scripts share one contract (``scripts/agent-claude.sh``'s)::

        bash <script> <model> [effort] [timeout_s]      prompt on stdin, reply on stdout

    so a backend that only changes *which* CLI the script drives, and how its output is
    parsed there, is nothing but the class variables above.
    """

    def build_command(self, spec: "AgentSpec", role: str,
                      env: dict | None = None) -> list[str]:
        return ["bash", str(self.script_path()), spec.model, spec.effort or "",
                str(spec.timeout_s)]
