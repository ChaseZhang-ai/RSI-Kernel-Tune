"""forge's command extensions.

A module in this package answers to one command name and lives apart from the loop's own
CLI (``forgekernel.launch`` / ``forgekernel.submit``)::

    COMMAND = "bench"                 # the name `forge <COMMAND> ...` answers to
    def main(argv: list[str]) -> int: # the command itself, exit code included

``forgekernel.ext.command_main`` discovers them: ``forge <name> ...`` reaches the
extension that declares ``COMMAND = "name"``, and returns ``None`` when no extension
claims the name -- so the built-in commands are never shadowed and an unknown name still
falls through to the submit CLI's error.

Discovery reads a module's declared ``COMMAND`` from its source, not by importing it: an
unrelated command (``forge <op> status``) never pays an extension's imports, and a broken
extension is only reported when its own command is asked for.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from pathlib import Path

_PACKAGE = __name__
_COMMAND_RE = re.compile(r"^COMMAND\s*=\s*[\"']([A-Za-z0-9_.-]+)[\"']", re.MULTILINE)


def _extensions() -> dict[str, Path]:
    """{module name: its source file} for every extension module."""
    out = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        path = Path(__file__).resolve().parent / f"{info.name}.py"
        out[info.name] = path
    return out


def _declared(name: str, path: Path) -> str:
    """The ``COMMAND`` a module's source declares ("" when it declares none)."""
    try:
        m = _COMMAND_RE.search(path.read_text(encoding="utf-8"))
    except OSError:
        return ""
    return m.group(1) if m else ""


def command_for(name: str) -> str | None:
    """The extension module name answering to ``name``, or None."""
    exts = _extensions()
    if name in exts:
        declared = _declared(name, exts[name])
        if declared == name:
            return name
        # A file whose COMMAND differs; a scan below may still claim the name.
    for mod_name, path in exts.items():
        if mod_name == name:
            continue
        if _declared(mod_name, path) == name:
            return mod_name
    return None


def command_main(argv: list[str]) -> int | None:
    """Run ``argv[0]``'s extension; None when no extension claims that name."""
    if not argv:
        return None
    mod_name = command_for(argv[0])
    if mod_name is None:
        return None
    mod = importlib.import_module(f"{_PACKAGE}.{mod_name}")
    return int(mod.main(argv[1:]))
