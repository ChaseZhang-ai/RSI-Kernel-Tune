"""Agent commands from a task's ``[agent]`` (merged per role: rsi_kernel_tune.task).

The harness builds the command line itself (a script under ``scripts/``, named by the
backend's adapter in ``rsi_kernel_tune.agentbackends``): the task names the backend, the
provider, the model, the effort and a turn's timeout, and the user never writes a CLI
invocation. One script serves both a ``claude -p`` turn and, when the caller asks for one
(FK_AGENT_DRIVER=goal), a /goal session.

Every command carries ``FK_AGENT_ROLE=<role>`` in front of it: the launcher scripts write
their usage record (and their refusal check) per role, and the loop's ``run_agent`` does
not otherwise tell them which role the turn is for.
"""

from __future__ import annotations

import shlex

from rsi_kernel_tune.agentbackends import ROLES, BackendError, load as load_backend

# The Claude Code launcher, as it always was (a caller that only wants "the script" should
# ask the backend: ``agentbackends.load("claude").script``).
AGENT_SCRIPT = "scripts/agent-claude.sh"

ROLE_ENV = "FK_AGENT_ROLE"


def agent_command(spec, role: str, env: dict | None = None) -> str:
    """The shell command the loop runs for one ``role`` turn (prompt on stdin). A provider
    other than ``native`` rides as environment assignments in front of it: the endpoint and
    the name of the variable that holds the key, never the key."""
    try:
        backend = load_backend(spec.backend)
    except BackendError as e:
        raise ValueError(f"backend {spec.backend!r} is not supported: {e}") from None
    if role not in ROLES:
        raise ValueError(f"unknown agent role {role!r} (roles: {list(ROLES)})")
    if role not in backend.roles:
        raise ValueError(f"the {backend.name} backend cannot serve the {role} role "
                         f"(it serves: {list(backend.roles)})")
    grant = backend.grant(spec.provider, env)
    argv = backend.build_command(spec, role, env)
    prefix = ["env", f"{ROLE_ENV}={role}", *(f"{k}={v}" for k, v in grant.items())]
    return " ".join(shlex.quote(a) for a in [*prefix, *argv])
