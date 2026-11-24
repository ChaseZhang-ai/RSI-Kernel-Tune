"""Command extensions: each module in this package is one ``forge`` command.

A new tool goes here and is ``forge <name> ...`` with nothing else registered:

    # src/forgekernel/ext/hello.py
    COMMAND = "hello"                      # the command: this module's own name
    def main(argv: list[str]) -> int:      # what ``forge hello ...`` runs; its exit status
        ...

How it works:

- **Found, never imported, until its command is called.** Discovery only lists this directory
  (``pkgutil.iter_modules``), so a broken extension cannot break ``forge --help`` or any built-in
  command. Called, it says what is wrong with it --
  ``[forge] extension command '<name>' failed to load: ...`` -- and answers 2. What a loaded
  extension's ``main`` raises is its own, as for a built-in command.
- **The command name is the module's name.** ``COMMAND`` must say the same: a name that could
  only be read by running the module could route nothing when the module is the thing that is
  broken. A module whose name starts with ``_`` is no command (a helper shared by extensions).
- **Built-in commands come first.** ``__main__`` looks here only after every built-in, so an
  extension named like one is never reached (and is left out of the usage text). The second
  word matters too: ``loop``, ``baseline``, ``window``, ``leaderboard``, ``monitor``,
  ``ncu-trace`` and ``ncu-report`` after any first word are built-in operator commands, so an
  extension must not have a subcommand of one of those names.
- **A task may not take an extension's name** (``task.load_task``): ``forge <name>`` would
  never reach the task's operator commands.

This package adds no command of its own. The first wave of them (bench, lib, target, prior,
campaign, evolve, board) is the tasks' to write.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Collection
from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True)
class Extension:
    """One extension command: ``name`` is both the command and the module's name here."""
    name: str
    module: str            # dotted, under this package

    def load(self) -> ModuleType:
        """Import the module. Raises whatever is wrong with it."""
        return importlib.import_module(self.module)


def commands() -> dict[str, Extension]:
    """Every extension command by name, in name order: the modules (and packages) under this
    one whose name does not start with ``_``. Nothing is imported."""
    import pkgutil

    found = {}
    for info in pkgutil.iter_modules(list(__path__)):
        if not info.name.startswith("_"):
            found[info.name] = Extension(name=info.name, module=f"{__name__}.{info.name}")
    return dict(sorted(found.items()))


def command_names() -> frozenset[str]:
    """The names a task may not take. Never raises: a task is read whatever is the matter with
    this directory, and an extension that cannot be listed cannot be reached either."""
    try:
        return frozenset(commands())
    except Exception:  # noqa: BLE001 -- listing is best effort; see run() for the real error
        return frozenset()


def run(name: str, argv: list[str]) -> int:
    """``forge <name> <argv...>``: import the extension now and call its ``main`` with what
    follows the command. A module that is not an extension (it does not import, its ``COMMAND``
    is not its name, it has no callable ``main``) is reported and answers 2."""
    extension = commands()[name]
    try:
        module = extension.load()
        main = module.main
        if module.COMMAND != name:
            raise TypeError(f"COMMAND is {module.COMMAND!r} in {extension.module}: a command is "
                            f"named after its module")
        if not callable(main):
            raise TypeError("main is not callable")
    except Exception as e:  # noqa: BLE001 -- whatever is wrong with it is the extension's to fix
        sys.stderr.write(f"[forge] extension command '{name}' failed to load: "
                         f"{type(e).__name__}: {e}\n")
        return 2
    return main(argv) or 0


def usage(builtins: Collection[str] = ()) -> str:
    """The part of ``forge --help`` that names the extension commands found -- by name only,
    since nothing of theirs is imported to write it. Those named like a built-in (``builtins``)
    are left out: they can never be reached."""
    header = "Extension commands (src/forgekernel/ext/, one module each):"
    try:
        names = [n for n in commands() if n not in builtins]
    except Exception as e:  # noqa: BLE001 -- the usage must print whatever is the matter here
        return f"{header} none could be listed ({type(e).__name__}: {e})"
    if not names:
        return f"{header} none"
    return header + "\n" + "".join(f"  forge {n} ...                (`forge {n} --help`)\n"
                                   for n in names)
