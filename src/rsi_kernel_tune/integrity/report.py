"""The eval JSON's ``integrity`` block: its shape, its validation, and its summary line.

The card writes the block (``card_template/eval/leaderboard.py``) and ``loop/parse.py``
reads it into ``EvalResult.integrity``; this module is the schema both sides are held to,
so a field rename cannot pass silently.  It changes nothing about the reading: the block
is carried beside ``correctness`` / ``performance``, never inside them.

::

    "integrity": {
      "fresh_input_recheck": {"status": "pass | fail | skipped", "workloads": [...],
                              "reason": "...", "max_rel_err": 0.0},
      "global_state":        {"status": "clean | changed", "diff": ["..."],
                              "threads": ["..."]},          # threads: warning only
      "reference_stability": {"status": "pass | fail | skipped", "max_rel_err": 0.0,
                              "reason": "..."},
      "timer_consistency":   {"status": "ok | suspicious | unavailable",
                              "events_ms": 0, "span_ms": 0, "wall_ms": 0, "reason": "..."},
      "implausible_speedup": false                 # or true, or null (no board row)
    }

``status`` values are fixed; anything else is a card bug and ``validate`` says so.  The
block is advisory except where the card turns it into a correctness failure
(``INTEGRITY: ...`` in ``correctness.error``) -- the loop gates on correctness, and the
block explains why.
"""

from __future__ import annotations

FRESH = "fresh_input_recheck"
GLOBAL = "global_state"
REFERENCE = "reference_stability"
TIMERS = "timer_consistency"
IMPLAUSIBLE = "implausible_speedup"

STATUSES = {
    FRESH: ("pass", "fail", "skipped"),
    GLOBAL: ("clean", "changed"),
    REFERENCE: ("pass", "fail", "skipped"),
    TIMERS: ("ok", "suspicious", "unavailable"),
}
BLOCK_KEYS = (FRESH, GLOBAL, REFERENCE, TIMERS, IMPLAUSIBLE)
FAIL_TEXTS = {
    FRESH: "fresh-input recheck mismatch",
    GLOBAL: "global state changed after the candidate ran",
    REFERENCE: "reference output changed after the candidate ran",
}


def block(*, fresh: dict | None = None, global_state: dict | None = None,
          reference: dict | None = None, timers: dict | None = None,
          implausible: bool | None = None) -> dict:
    """A block with every key present, defaults filled in."""
    return {
        FRESH: dict(fresh or {"status": "skipped", "workloads": [], "reason": "not run"}),
        GLOBAL: dict(global_state or {"status": "clean", "diff": []}),
        REFERENCE: dict(reference or {"status": "skipped", "max_rel_err": 0.0,
                                      "reason": "not run"}),
        TIMERS: dict(timers or {"status": "unavailable", "events_ms": None, "span_ms": None,
                                "wall_ms": None, "reason": "not run"}),
        IMPLAUSIBLE: implausible,
    }


def validate(value) -> list[str]:
    """Every way ``value`` is not the documented block (empty = it is)."""
    problems: list[str] = []
    if not isinstance(value, dict):
        return [f"the integrity block is {type(value).__name__}, not an object"]
    for key in BLOCK_KEYS:
        if key not in value:
            problems.append(f"missing key {key!r}")
    for key, allowed in STATUSES.items():
        entry = value.get(key)
        if entry is None:
            continue
        if not isinstance(entry, dict):
            problems.append(f"{key}: {type(entry).__name__}, not an object")
            continue
        status = entry.get("status")
        if status not in allowed:
            problems.append(f"{key}.status={status!r}, one of {allowed} expected")
    if GLOBAL in value and isinstance(value[GLOBAL], dict):
        diff = value[GLOBAL].get("diff")
        if diff is not None and not isinstance(diff, list):
            problems.append("global_state.diff is not a list")
    if IMPLAUSIBLE in value and not isinstance(value[IMPLAUSIBLE], (bool, type(None))):
        problems.append("implausible_speedup is not a bool or null")
    return problems


def failures(value) -> list[str]:
    """The ``INTEGRITY: ...`` lines the block implies (the card writes the same ones)."""
    out: list[str] = []
    if not isinstance(value, dict):
        return out
    fresh = value.get(FRESH) or {}
    if isinstance(fresh, dict) and fresh.get("status") == "fail":
        workloads = [str(w) for w in (fresh.get("workloads") or []) if w]
        out.append("INTEGRITY: " + FAIL_TEXTS[FRESH]
                   + (f" on {', '.join(workloads)}" if workloads else "")
                   + f": {fresh.get('reason') or ''}")
    glob = value.get(GLOBAL) or {}
    if isinstance(glob, dict) and glob.get("status") == "changed":
        out.append("INTEGRITY: " + FAIL_TEXTS[GLOBAL] + ": "
                   + "; ".join(str(d) for d in (glob.get("diff") or [])))
    ref = value.get(REFERENCE) or {}
    if isinstance(ref, dict) and ref.get("status") == "fail":
        out.append("INTEGRITY: " + FAIL_TEXTS[REFERENCE] + f": {ref.get('reason') or ''}")
    return out


def warnings(value) -> list[str]:
    """What the block flags without failing the run."""
    out: list[str] = []
    if not isinstance(value, dict):
        return out
    glob = value.get(GLOBAL) or {}
    if isinstance(glob, dict):
        for line in glob.get("threads") or ():
            out.append(f"thread count changed: {line}")
    timers = value.get(TIMERS) or {}
    if isinstance(timers, dict) and timers.get("status") == "suspicious":
        out.append("timer consistency: " + str(timers.get("reason") or "suspicious"))
    if value.get(IMPLAUSIBLE):
        out.append("implausible speedup: a reading below 0.6 x T_SOL")
    return out


def describe(value) -> str:
    """One line, for a log or a report."""
    if not isinstance(value, dict):
        return "no integrity block"
    parts = []
    for key in (FRESH, GLOBAL, REFERENCE, TIMERS):
        entry = value.get(key) or {}
        if isinstance(entry, dict):
            parts.append(f"{key.replace('_', '-')} {entry.get('status', '?')}")
    if value.get(IMPLAUSIBLE):
        parts.append("implausible-speedup true")
    return "; ".join(parts)
