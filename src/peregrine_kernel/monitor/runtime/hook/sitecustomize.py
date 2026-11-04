"""Correctness monitor loader. Python imports ``sitecustomize`` at start-up from the
first directory on sys.path that has one; the monitor's runner puts this directory
first on PYTHONPATH for the card's bench. It installs the hook (a no-op unless
PK_MON_PROFILE is set) and then hands over to whatever ``sitecustomize`` the image
itself ships, so the job environment is otherwise unchanged."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

try:
    sys.path.insert(0, _HERE)
    import pk_monitor_hook

    pk_monitor_hook.install()
except Exception as _e:  # noqa: BLE001 — the monitor never costs the bench
    sys.stderr.write(f"[pk-monitor] hook not installed: {_e}\n")
finally:
    try:
        sys.path.remove(_HERE)
    except ValueError:
        pass


def _student_roots() -> list:
    """Directory roots the student controls, derived from the deployed layout
    (``<workspace>/.pk_monitor/runtime/hook``): the workspace itself, everything under
    it included -- its ``eval/``, a tree-root ``sitecustomize.py`` most of all. Empty
    when this copy does not sit in the deployed layout."""
    parts = _HERE.split(os.sep)
    if parts[-3:-1] != [".pk_monitor", "runtime"]:
        return []
    return [os.path.normpath(os.path.join(_HERE, "..", "..", ".."))]


def _chain() -> None:
    import importlib

    saved = list(sys.path)
    ours = sys.modules.pop("sitecustomize", None)
    try:
        roots = _student_roots()
        # drop this directory (already imported) and every student-owned entry: the
        # next ``sitecustomize`` must be one the image ships, never one the student's
        # tree root carries -- chaining it would hand the bench over before any
        # harness code has run
        sys.path[:] = [p for p in sys.path if not _owned(p, roots)]
        try:
            importlib.import_module("sitecustomize")
        except ImportError:
            if ours is not None:
                sys.modules["sitecustomize"] = ours
    finally:
        sys.path[:] = saved


def _owned(entry: str, roots: list) -> bool:
    directory = os.path.abspath(entry or os.curdir)
    if directory == _HERE:
        return True
    return any(directory == r or directory.startswith(r + os.sep) for r in roots)


_chain()
