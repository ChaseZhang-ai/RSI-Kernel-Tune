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

The driver binds each path into the prompts as ``{TARGET_PROFILE_PATH}`` and so on
(loop/driver.py ``_bindings``); the teacher's and the student's prompts end with the section that
says what to do with them. A variable that is not set binds ``(none)``, and the prompts tell the
agent to proceed exactly as without the file -- so a run that sets none of them behaves as before.

Who measures writes: targethw the target profile, roofline the roofline, opkb the lessons,
warmstart and seedlib the prior implementations. What each file holds is theirs to define. Two
rules are not: the content is measured or mechanically extracted (never a narrative an LLM wrote
about what might work), and it is run-time input -- data a prompt reads, never advice that
outranks the prompt.

::

    from forgekernel import runcontext
    runcontext.path_for("target")                       # Path | None, read from the environment
    runcontext.env_for({"target": p, "prior": q})       # the variables that make them the context
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

CONTEXT_FILES = {
    "target":    "FK_TARGET_PROFILE_PATH",   # target device profile: facts + rules for THIS project
    "roofline":  "FK_ROOFLINE_PATH",         # measured peaks + per-workload distance to SOL
    "lessons":   "FK_LESSONS_PATH",          # what was measured before on other operators of the class
    "prior":     "FK_PRIOR_PATH",            # existing implementations for THIS operator (paths, sources, scores) + its rules
}

# What a binding holds when the run sets no file (the prompts name this value and say to go on).
NONE_BINDING = "(none)"


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
