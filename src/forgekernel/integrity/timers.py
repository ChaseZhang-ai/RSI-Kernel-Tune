"""Check C's rule, for readers outside the job: the three clocks of one call.

The card times one call three ways after the timed window -- CUDA events around it, a
CUPTI span of the work it launched, and the wall clock -- and reports whether their
relationship is the one a per-call measurement must have::

    wall >= events >= span

The card is the implementation (it is what runs in the job); this module is the tested
host-side door to the same rule, so the thresholds and their reasoning live in one place
(``docs/integrity.md`` explains each bound):

* ``span < 0.5 x events``  -- work escaped the window the events bracketed
* ``events < 0.5 x span``  -- the events did not see the work CUPTI did
* ``wall < 0.5 x span``    -- impossible for one call
* ``wall > 20 x max(...)`` and over 1 ms -- the reading is host-bound, not device time

``verdict()`` returns ``("ok" | "suspicious" | "unavailable", why)``.
"""

from __future__ import annotations

from .snapshot import card_module

SPAN_SHARE = 0.5
EVENTS_SHARE = 0.5
WALL_SHARE = 0.5
WALL_MAX = 20.0
WALL_FLOOR_MS = 1.0
ITERS = 5


def verdict(events_ms: float | None, span_ms: float | None,
            wall_ms: float | None) -> tuple[str, str]:
    """The card's timer-consistency rule on three readings of one call."""
    return card_module()._timer_verdict(events_ms, span_ms, wall_ms)


def suspicious(events_ms: float | None, span_ms: float | None,
               wall_ms: float | None) -> bool:
    return verdict(events_ms, span_ms, wall_ms)[0] == "suspicious"
