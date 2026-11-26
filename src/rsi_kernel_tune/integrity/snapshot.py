"""The evaluation's own state, as the card reads it -- one implementation, two callers.

Check B asks whether the candidate changed the process it was judged in: a
``torch.backends`` flag (TF32, cuDNN autotune), the float32 matmul precision, an
environment variable the GPU stack reads, or the number of live threads.  The *runtime*
implementation is in the frozen card (``card_template/eval/leaderboard.py``): the card is
copied into a task at ``forge init`` and runs in a job image where this package does not
exist, so the code that actually measures has to live there, not here.

This module is the host-side door to that same code, so a test, a report or a host tool
can read the state and diff two snapshots without a GPU and without a second copy of the
rules drifting from the first:

    from rsi_kernel_tune.integrity.snapshot import snapshot, diff
    before = snapshot()
    ...
    settings, threads = diff(before, snapshot())

``snapshot()`` is ``{"flags": {...}, "env": {...}, "threads": n}``; ``diff`` returns the
lines that changed, split into (settings the candidate must not change, thread-count
changes, which are a warning).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

CARD = Path(__file__).resolve().parents[1] / "card_template" / "eval" / "leaderboard.py"

_card_cache = None


def card_module():
    """The card's own module, loaded for its integrity helpers (torch required)."""
    global _card_cache
    if _card_cache is not None:
        return _card_cache
    spec = importlib.util.spec_from_file_location("rsi_kernel_tune_card_integrity", CARD)
    if spec is None or spec.loader is None:      # pragma: no cover -- a broken install
        raise ImportError(f"cannot load the card template at {CARD}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _card_cache = module
    return module


def flags() -> dict[str, str]:
    """The global flags, read the card's way (a missing one reads ``<unavailable>``)."""
    return dict(card_module()._global_state()["flags"])


def snapshot() -> dict:
    """The evaluation's state: flags, GPU-related environment variables, thread count."""
    return card_module()._global_state()


def diff(before: dict | None, after: dict | None) -> tuple[list[str], list[str]]:
    """(settings changed, thread counts changed) between two snapshots."""
    return card_module()._global_state_diff(before, after)


def changed(before: dict | None, after: dict | None) -> list[str]:
    """Every line that changed, settings and threads together."""
    settings, threads = diff(before, after)
    return list(settings) + list(threads)
