"""Run context files: what a run knows about the GPU it is graded on and about its operator.

A prompt in ``meta/`` is written once and says nothing about any operator (invariant L1). What
differs from run to run -- the GPU the result is graded on, how far the reference is from what
the hardware allows, what was measured before on other operators of the same class, the kernels
that already exist for this one -- is data, and it reaches an agent as FILES the prompt names:

    name        environment variable     what the file holds
    target      FK_TARGET_PROFILE_PATH    the GPU the result is graded on: its facts, and the
                                          rules a submission must satisfy (the machine the kernel
                                          is developed on may differ: the file says how)
    roofline    FK_ROOFLINE_PATH          the GPU's measured peaks, and per workload the time the
                                          hardware allows and how far the reference and the current
                                          best are from it
    lessons     FK_LESSONS_PATH           what was measured before on other operators of the
                                          same class (approaches, numbers, traps)
    prior       FK_PRIOR_PATH             kernels already written for THIS operator or its family
                                          (paths, sources, licenses, measured scores) + its rules

The driver binds each *path* into the prompts as ``{TARGET_PROFILE_PATH}`` and so on
(loop/driver.py ``_bindings``, through ``binding``); the teacher's and the student's prompts end
with the section that says what to do with them. A variable that is not set binds ``(none)``, and
the prompts tell the agent to proceed exactly as without the file -- so a run that sets none of
them behaves as before.

A prompt builder on a machine where a path would not resolve can inline a file's *text*
instead: ``render(text)`` replaces every placeholder with ``read``'s text, and ``bindings()``
gives ``{placeholder: text}`` for all four. The two views are the same files; whichever one a
caller uses, nothing is invented when a file is unset -- ``(none)`` for a path, the plain note
from ``read`` for a text.

Who measures writes: targethw the target profile, roofline the roofline, opkb the lessons,
warmstart and seedlib the prior implementations. What each file holds is theirs to define. Two
rules are not: the content is measured or mechanically extracted (never a narrative an LLM wrote
about what might work), and it is run-time input -- data a prompt reads, never advice that
outranks the prompt.

::

    from rsi_kernel_tune import runcontext
    runcontext.path_for("target")                       # Path | None, read from the environment
    runcontext.env_for({"target": p, "prior": q})       # the variables that make them the context
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

# name -> (environment variable, prompt placeholder)
CONTEXTS: dict[str, tuple[str, str]] = {
    "target": ("FK_TARGET_PROFILE_PATH", "TARGET_PROFILE_PATH"),
    "roofline": ("FK_ROOFLINE_PATH", "ROOFLINE_PATH"),
    "lessons": ("FK_LESSONS_PATH", "LESSONS_PATH"),
    "prior": ("FK_PRIOR_PATH", "PRIOR_PATH"),
}

# name -> environment variable (the same four, in the order an agent meets them)
CONTEXT_FILES = {name: variable for name, (variable, _) in CONTEXTS.items()}

# What a binding holds when the run sets no file (the prompts name this value and say to go on).
NONE_BINDING = "(none)"

# A context file is a page or two. A truncated one is still useful (its head is the summary);
# an unbounded one would crowd out the task itself.
MAX_CHARS = 64_000

_EMPTY = {
    "target": "(no target profile set: FK_TARGET_PROFILE_PATH is unset)",
    "roofline": "(no measured roofline set: FK_ROOFLINE_PATH is unset)",
    "lessons": "(no operator-family lessons set: FK_LESSONS_PATH is unset)",
    "prior": "(no ready-made implementations listed: FK_PRIOR_PATH is unset)",
}


class ContextError(ValueError):
    """A context path was named but cannot be read."""


def _absolute(value) -> str | None:
    """``value`` as an absolute path with ``~`` expanded; None when it says nothing. Absolute,
    because the agents run with the student's workspace as their working directory: a relative
    path would be read from there, not from where the run was started."""
    text = str(value).strip() if value is not None else ""
    return os.path.abspath(os.path.expanduser(text)) if text else None


def path_for(name: str, env: Mapping | None = None) -> Path | None:
    """The file ``name`` (one of ``CONTEXT_FILES``) names in ``env`` -- the process environment
    by default: absolute, ``~`` expanded; ``None`` when it is not set or blank. The file need
    not exist: whether it does is the agent's to see, not a reason to refuse a run."""
    value = (os.environ if env is None else env).get(CONTEXT_FILES[name])
    absolute = _absolute(value)
    return Path(absolute) if absolute is not None else None


def paths(env: Mapping | None = None) -> dict[str, Path | None]:
    """Every run context file by name (``None`` where the run sets none)."""
    return {name: path_for(name, env) for name in CONTEXT_FILES}


def binding(name: str, env: Mapping | None = None) -> str:
    """What the prompts' placeholder for ``name`` is bound to: the absolute path, or
    ``NONE_BINDING`` when the run sets no file."""
    path = path_for(name, env)
    return NONE_BINDING if path is None else str(path)


def env_for(context: Mapping) -> dict[str, str]:
    """The environment variables that make ``context`` (run context name -> file) what a run
    sees -- for a subprocess started with them, or a test. A name whose file is ``None`` or
    blank sets nothing, so the run proceeds without it. A name that is not a run context file
    is a mistake and raises ``KeyError``."""
    out: dict[str, str] = {}
    for name, value in context.items():
        variable = CONTEXT_FILES[name]       # a name that is not a run context file is a mistake
        absolute = _absolute(value)
        if absolute is not None:
            out[variable] = absolute
    return out


def env_name(name: str) -> str:
    """The environment variable of context ``name``."""
    try:
        return CONTEXTS[name][0]
    except KeyError:
        raise KeyError(f"unknown run context {name!r}; known: {sorted(CONTEXTS)}") from None


def placeholder(name: str) -> str:
    """The prompt placeholder of context ``name`` (without braces)."""
    try:
        return CONTEXTS[name][1]
    except KeyError:
        raise KeyError(f"unknown run context {name!r}; known: {sorted(CONTEXTS)}") from None


def read(name: str, environ: Mapping | None = None) -> str:
    """The text of context ``name``; the empty-context note when unset; ``ContextError`` when
    the variable is set but the file cannot be read (a silently missing target profile would
    let a run proceed on assumptions the operator never chose)."""
    environ = os.environ if environ is None else environ
    var = env_name(name)
    value = str(environ.get(var) or "").strip()
    if not value:
        return _EMPTY[name]
    p = Path(value).expanduser()
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise ContextError(f"{var}={value}: {e.strerror or e}") from None
    if len(text) > MAX_CHARS:
        text = (text[:MAX_CHARS].rstrip()
                + f"\n\n[truncated: {p.name} is {len(text)} characters, the first "
                  f"{MAX_CHARS} are shown]")
    return text


def bindings(environ: Mapping | None = None) -> dict[str, str]:
    """``{placeholder: text}`` for every context (unset ones get their note)."""
    return {ph: read(name, environ) for name, (_, ph) in CONTEXTS.items()}


def render(text: str, environ: Mapping | None = None) -> str:
    """``text`` with every context placeholder replaced by the file's text."""
    for ph, value in bindings(environ).items():
        text = text.replace("{" + ph + "}", value)
    return text
