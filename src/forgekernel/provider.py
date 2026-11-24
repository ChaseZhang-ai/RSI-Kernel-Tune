"""Bind one agent turn to a provider: the only reader of global_config/backends.toml.

``native`` inherits the environment untouched: the agent CLI's own login, or the credentials
local.env puts in the environment. Any other name is a table of backends.toml that points
the CLI at another endpoint, with the key read from the variable the table names.

Which protocol a backend's CLI speaks, and which variable carries its endpoint, comes from
the backend's adapter (``forgekernel.agentbackends``): ``wire`` is checked against the
table, and the environment handed back names the key variable rather than the key.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from forgekernel.agentbackends import BackendError, load as load_backend
from forgekernel.paths import global_config_dir

BACKENDS_FILE = "backends.toml"
NATIVE = "native"
# The endpoint variable of each wire protocol: what the turn's launcher reads
# (scripts/agent-claude.sh, scripts/agent-codex.sh). The name of the variable holding the
# key rides separately, and the key itself never appears on a command line.
BASE_URL_ENV = "ANTHROPIC_BASE_URL"      # the anthropic one, as it always was
BASE_URL_ENV_BY_WIRE = {"anthropic": BASE_URL_ENV, "openai": "OPENAI_BASE_URL"}
KEY_ENV_ENV = "FK_PROVIDER_KEY_ENV"


class ProviderError(ValueError):
    pass


def _table(root: Path | None = None) -> dict:
    p = (root or global_config_dir()) / BACKENDS_FILE
    if not p.is_file():
        return {}
    return tomllib.loads(p.read_text(encoding="utf-8"))


def names(root: Path | None = None) -> tuple[str, ...]:
    return (NATIVE, *_table(root))


def grant_for_wire(wire: str | None, backend: str, name: str, env: dict | None = None,
                   root: Path | None = None) -> dict[str, str]:
    """``grant`` for a backend whose CLI speaks ``wire`` (None: no provider channel)."""
    if name == NATIVE:
        return {}
    table = _table(root)
    if name not in table:
        raise ProviderError(f"provider {name!r} is not in global_config/{BACKENDS_FILE}")
    p = table[name]
    if wire is None:
        raise ProviderError(f"the {backend} CLI has no provider channel; use provider = "
                            f"{NATIVE!r}")
    if p.get("wire") != wire:
        raise ProviderError(f"provider {name!r} speaks {p.get('wire')}, the {backend} CLI "
                            f"speaks {wire}")
    for key in ("base_url", "key_env"):
        if not isinstance(p.get(key), str) or not p[key].strip():
            raise ProviderError(f"provider {name!r} has no {key}")
    env = os.environ if env is None else env
    if not env.get(p["key_env"]):
        raise ProviderError(f"provider {name!r} needs {p['key_env']}; set it in local.env or "
                            f"the environment")
    base_url_env = BASE_URL_ENV_BY_WIRE.get(wire)
    if base_url_env is None:
        raise ProviderError(f"the {backend} CLI speaks {wire}, which has no endpoint "
                            f"variable; use provider = {NATIVE!r}")
    return {base_url_env: p["base_url"], KEY_ENV_ENV: p["key_env"]}


def grant(backend: str, name: str, env: dict | None = None,
          root: Path | None = None) -> dict[str, str]:
    """The environment additions that point ``backend`` at provider ``name``; ProviderError
    when the provider does not fit the backend or its key is not set."""
    if name == NATIVE:
        return {}
    try:
        adapter = load_backend(backend)
    except BackendError:
        raise ProviderError(
            f"the {backend} CLI has no provider channel (registered backends: "
            f"{', '.join(_registered()) or 'none'}); use provider = {NATIVE!r}") from None
    return adapter.grant(name, env, root=root)


def _registered() -> tuple[str, ...]:
    from forgekernel.agentbackends import names as backend_names
    return backend_names()
