"""Runtime context files: the facts a run is given, injected through the environment.

A run is configured by files that are *data*, not by the prompts: the target profile, the
measured roofline, the operator-family lessons and the list of ready-made implementations
for this problem. Each is named by an environment variable and reaches the prompts as a
placeholder, so the prompt templates themselves stay operator-agnostic (invariant L1) and
the same template serves every task:

    name      environment variable        prompt placeholder
    target    FK_TARGET_PROFILE_PATH      {TARGET_PROFILE_PATH}
    roofline  FK_ROOFLINE_PATH            {ROOFLINE_PATH}
    lessons   FK_LESSONS_PATH             {LESSONS_PATH}
    prior     FK_PRIOR_PATH               {PRIOR_PATH}

Two ways to use it:

* a launcher (campaign, evolve) renders the file and passes the environment on:
  ``env_for({"target": target_profile.materialize(repo_root)})``;
* a prompt builder turns the environment into the placeholder values:
  ``bindings()`` -> ``{"TARGET_PROFILE_PATH": "<the file's text>", ...}``, then
  ``render(text)``. An unset context renders as a short plain statement, never as a raw
  ``{PLACEHOLDER}`` the model would read as an instruction to fill in.

The placeholder holds the file's *text* (not its path): the agents run on another machine
(ssh delivery) and on a snapshot of the tree, so a path would not resolve there.
"""

from __future__ import annotations

import os
from pathlib import Path

# name -> (environment variable, placeholder)
CONTEXTS: dict[str, tuple[str, str]] = {
    "target": ("FK_TARGET_PROFILE_PATH", "TARGET_PROFILE_PATH"),
    "roofline": ("FK_ROOFLINE_PATH", "ROOFLINE_PATH"),
    "lessons": ("FK_LESSONS_PATH", "LESSONS_PATH"),
    "prior": ("FK_PRIOR_PATH", "PRIOR_PATH"),
}

# A context file is a page or two. A truncated one is still useful (its head is the
# summary); an unbounded one would crowd out the task itself.
MAX_CHARS = 64_000

_EMPTY = {
    "target": "(no target profile set: FK_TARGET_PROFILE_PATH is unset)",
    "roofline": "(no measured roofline set: FK_ROOFLINE_PATH is unset)",
    "lessons": "(no operator-family lessons set: FK_LESSONS_PATH is unset)",
    "prior": "(no ready-made implementations listed: FK_PRIOR_PATH is unset)",
}


class ContextError(ValueError):
    """A context path was named but cannot be read."""


def env_name(name: str) -> str:
    try:
        return CONTEXTS[name][0]
    except KeyError:
        raise KeyError(f"unknown run context {name!r}; known: {sorted(CONTEXTS)}") from None


def placeholder(name: str) -> str:
    try:
        return CONTEXTS[name][1]
    except KeyError:
        raise KeyError(f"unknown run context {name!r}; known: {sorted(CONTEXTS)}") from None


def env_for(paths: dict[str, str | Path], base: dict | None = None) -> dict[str, str]:
    """Environment with one variable per context in ``paths`` (merged over ``base``).

    Only names in ``CONTEXTS`` are accepted: a typo silently ignored would leave the
    agents without the facts they were meant to get.
    """
    unknown = sorted(set(paths) - set(CONTEXTS))
    if unknown:
        raise KeyError(f"unknown run context(s) {unknown}; known: {sorted(CONTEXTS)}")
    env = dict(os.environ if base is None else base)
    for name, path in paths.items():
        env[env_name(name)] = str(Path(path).expanduser().resolve())
    return env


def paths(environ: dict | None = None) -> dict[str, Path]:
    """The context files the environment names (only those set and non-empty)."""
    environ = os.environ if environ is None else environ
    out: dict[str, Path] = {}
    for name, (var, _) in CONTEXTS.items():
        value = str(environ.get(var) or "").strip()
        if value:
            out[name] = Path(value).expanduser()
    return out


def read(name: str, environ: dict | None = None) -> str:
    """The text of context ``name``; the empty-context note when unset; ContextError when
    the variable is set but the file cannot be read (a silently missing target profile
    would let a run proceed on assumptions the operator never chose)."""
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


def bindings(environ: dict | None = None) -> dict[str, str]:
    """``{placeholder: text}`` for every context (unset ones get their note)."""
    return {ph: read(name, environ) for name, (_, ph) in CONTEXTS.items()}


def render(text: str, environ: dict | None = None) -> str:
    """``text`` with every context placeholder replaced."""
    for ph, value in bindings(environ).items():
        text = text.replace("{" + ph + "}", value)
    return text
