"""Agent backends: each module here is one agent CLI the harness can drive.

Discovery mirrors ``forgekernel.ext``: this directory is *listed*, never imported, until a
backend is asked for, so a broken backend cannot break ``forge --help``, task loading or
another backend.

A backend is a module whose name is the backend name and which exports one instance
``BACKEND`` of ``forgekernel.agentbackends.base.AgentBackend`` (its ``.name`` must equal
the module name). Adding one is: write the module, add the launcher script it names, and
document it in ``docs/agent-backends.md`` -- nothing else registers it (``task.BACKENDS``
and ``provider.grant`` both read this registry).

    # src/forgekernel/agentbackends/mine.py
    from .base import ScriptBackend
    class MineBackend(ScriptBackend):
        name = "mine"; wire = "openai"; script = "scripts/agent-mine.sh"
        roles = ("teacher",); policy_support = "none"
    BACKEND = MineBackend()
"""

from __future__ import annotations

import importlib

from .base import POLICY_SUPPORT, ROLES, AgentBackend, BackendError, ScriptBackend

__all__ = ["AgentBackend", "BackendError", "ScriptBackend", "ROLES", "POLICY_SUPPORT",
           "all_backends", "get", "names"]

# The modules of this package that are not backends: the interface itself.
_INTERFACE_MODULES = frozenset({"base"})


def modules() -> dict[str, str]:
    """Every backend module here by name, in name order. Nothing is imported.

    ``base`` is the interface module (no ``BACKEND`` of its own), so it is the one name
    that is listed here but is not a backend."""
    import pkgutil

    found = {}
    for info in pkgutil.iter_modules(list(__path__)):
        if not info.name.startswith("_") and info.name not in _INTERFACE_MODULES:
            found[info.name] = f"{__name__}.{info.name}"
    return dict(sorted(found.items()))


def names() -> tuple[str, ...]:
    """The registered backend names. Never raises: a name that cannot be listed cannot be
    used either."""
    try:
        return tuple(modules())
    except Exception:  # noqa: BLE001 -- listing is best effort, as in forgekernel.ext
        return ()


def load(name: str) -> AgentBackend:
    """The backend ``name``, imported now. BackendError when it is not registered or its
    module does not carry a matching ``BACKEND``."""
    module_name = modules().get(name)
    if module_name is None:
        available = ", ".join(names()) or "none"
        raise BackendError(f"no agent backend {name!r} (registered: {available}); a backend "
                           f"is a module in src/forgekernel/agentbackends/")
    module = importlib.import_module(module_name)
    backend = getattr(module, "BACKEND", None)
    if not isinstance(backend, AgentBackend):
        raise BackendError(f"{module_name} does not define BACKEND as an AgentBackend")
    if backend.name != name:
        raise BackendError(f"{module_name}: BACKEND.name is {backend.name!r}, but the module "
                           f"is {name!r}")
    return backend


def get(name: str, default: AgentBackend | None = None) -> AgentBackend | None:
    try:
        return load(name)
    except BackendError:
        if default is not None:
            return default
        raise


def all_backends() -> dict[str, AgentBackend]:
    """Every registered backend, loaded, by name."""
    return {n: load(n) for n in names()}
