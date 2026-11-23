"""Command extensions: one module per command, discovered at import time.

A module in this package that defines ``COMMAND`` (the name ``forge`` dispatches on) and
``main(argv) -> int`` becomes a command:

  ``forge <COMMAND> [args...]``  ->  ``main([args...])``

Nothing else is needed: no registration table, no import list. ``forge --help`` lists the
commands (``help_lines()``), and ``forge help <COMMAND>`` runs the module's own ``--help``.

The built-in commands (init / new / run / status / harvest / export, and the per-op
subcommands) win over an extension of the same name: an extension may not shadow them,
and a task may not be named after an extension (``forge target`` must stay the command).
A module whose import fails is skipped (and reported by ``problems()``) rather than
breaking every other command.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

COMMAND_ATTR = "COMMAND"
MAIN_ATTR = "main"

_cache: dict[str, ModuleType] | None = None
_cache_problems: list[str] = []


def _iter_modules():
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        yield info.name


def _discover() -> dict[str, ModuleType]:
    global _cache, _cache_problems
    if _cache is not None:
        return _cache
    found: dict[str, ModuleType] = {}
    problems: list[str] = []
    for name in _iter_modules():
        modname = f"{__name__}.{name}"
        try:
            mod = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE001 -- one broken extension must not hide the rest
            problems.append(f"ext/{name}.py does not import: {type(e).__name__}: {e}")
            continue
        command = getattr(mod, COMMAND_ATTR, None)
        main = getattr(mod, MAIN_ATTR, None)
        if not isinstance(command, str) or not command:
            problems.append(f"ext/{name}.py defines no COMMAND")
            continue
        if not callable(main):
            problems.append(f"ext/{name}.py defines no main(argv)")
            continue
        if command in found:
            problems.append(f"ext/{name}.py repeats command {command!r} "
                            f"({found[command].__name__} already has it)")
            continue
        found[command] = mod
    _cache, _cache_problems = found, problems
    return found


def reload() -> dict[str, ModuleType]:
    """Drop the cache and discover again (a test that installs a module)."""
    global _cache
    _cache = None
    return _discover()


def commands() -> dict[str, ModuleType]:
    """``{command: module}``; extensions that fail to import are omitted."""
    return _discover()


def problems() -> list[str]:
    """Every reason an extension was skipped (for a diagnostic command)."""
    _discover()
    return list(_cache_problems)


def is_command(name: str) -> bool:
    return name in _discover()


def dispatch(argv: list[str]) -> int | None:
    """Run the extension command ``argv[0]``; None when no extension has that name."""
    if not argv:
        return None
    mod = _discover().get(argv[0])
    if mod is None:
        return None
    rc = mod.main(list(argv[1:]))
    return 0 if rc is None else int(rc)


def help_lines() -> list[str]:
    """One ``forge <name>  <first docstring line>`` per extension, sorted by name."""
    lines = []
    for command, mod in sorted(commands().items()):
        doc = (mod.__doc__ or "").strip().splitlines()
        summary = doc[0].strip() if doc else ""
        lines.append(f"  forge {command}" + (f"{' ' * max(1, 18 - len(command))}{summary}"
                                             if summary else ""))
    return lines
