"""Correctness monitor loader. Python imports ``sitecustomize`` at start-up from the
first directory on sys.path that has one; the monitor's runner puts this directory
first on PYTHONPATH for the card's bench. It installs the hook (a no-op unless
RKT_MON_PROFILE is set) and then hands over to whatever ``sitecustomize`` the image
itself ships, so the job environment is otherwise unchanged."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

try:
    sys.path.insert(0, _HERE)
    import rkt_monitor_hook

    rkt_monitor_hook.install()
except Exception as _e:  # noqa: BLE001 — the monitor never costs the bench
    sys.stderr.write(f"[rkt-monitor] hook not installed: {_e}\n")
finally:
    try:
        sys.path.remove(_HERE)
    except ValueError:
        pass


def _chain_target() -> str | None:
    """The next ``sitecustomize.py`` along sys.path (ours removed) — skipping any
    that lives inside the judged tree (``RKT_MON_ROOT``) or the current directory:
    a student workspace can plant one at its root, and the old plain
    ``importlib.import_module("sitecustomize")`` happily executed it. The image's
    own copy (the harness's, a site-packages one) sits outside those roots and
    still chains. Decided by file path alone — no name lists."""
    roots = []
    for r in (os.environ.get("RKT_MON_ROOT"), os.getcwd()):
        if not r:
            continue
        try:
            roots.append(os.path.realpath(r))
        except OSError:
            continue
    for p in sys.path:
        cand = os.path.join(os.path.abspath(p or os.curdir), "sitecustomize.py")
        if not os.path.isfile(cand):
            continue
        try:
            real = os.path.realpath(cand)
        except OSError:
            continue
        if any(real == r or real.startswith(r + os.sep) for r in roots):
            continue
        return cand
    return None


def _chain() -> None:
    import importlib.util

    saved = list(sys.path)
    ours = sys.modules.pop("sitecustomize", None)
    try:
        sys.path[:] = [p for p in sys.path
                       if os.path.abspath(p or os.curdir) != _HERE]
        path = _chain_target()
        if path is None:
            raise ImportError("no other sitecustomize on sys.path")
        spec = importlib.util.spec_from_file_location("sitecustomize", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["sitecustomize"] = mod
        spec.loader.exec_module(mod)
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
