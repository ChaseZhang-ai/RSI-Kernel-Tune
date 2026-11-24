"""Evaluation integrity: the checks that keep a reading honest, and the export gates.

Two places need this, and they are the two ways a number leaves the harness:

**the card** (``card_template/eval/leaderboard.py``, copied into every task at
``forge init``) runs the *runtime* checks after it has timed a candidate -- a fresh-input
recheck, a global-state snapshot, a reference-stability recheck, a three-clock consistency
test, an implausibility bound.  They live in the card, not here, because the card runs in a
job image where this package does not exist; the block they write goes into the eval JSON
and ``loop/parse.py`` carries it through as ``EvalResult.integrity``.

**the export** (``forge export``, campaign's ``compose.export``) runs the *static* checks on
the file that is about to be uploaded:

* :func:`lint_source` -- the board's prohibited behaviours, in source form (see
  ``forgekernel.integrity.lint``; every rule has a positive and a negative test);
* ``forgekernel.targethw.rules.check_submission`` -- the target rules (no architecture
  literal, no hard-coded SM count), when that task is installed (a soft dependency, the
  same way campaign resolves its slots);
* :func:`license_check` -- a derived file's licences and its notice block
  (``forgekernel.integrity.license`` / ``.notices``);
* :func:`cold_start_gate` -- a candidate whose first call takes longer than the board's
  compile budget is a zero however fast its steady state is (``forgekernel.benchx.api``
  owns the number).

``export_check(path)`` is the one call campaign's ``tools.resolve`` finds (its slot probes
``export_check`` / ``submission_check`` / ``check_submission``): a dict with ``ok`` and the
findings.  ``forge export`` refuses on ``ok == False`` unless
``--allow-integrity-findings`` is given.

Nothing here judges a reading: the card's gate is the board's protocol, and these checks
only add failure categories (``INTEGRITY: ...``) or warnings beside it.
"""

from __future__ import annotations

from pathlib import Path

from . import license as license_mod
from . import lint as lint_mod
from . import multiseed as multiseed_mod
from . import notices as notices_mod
from . import report as report_mod
from . import snapshot as snapshot_mod
from . import timers as timers_mod
from .fallback import leading_comment_block, wrap_with_fallback
from .lint import (BLOCK, INFO, SEVERITIES, WARN, Finding, blocks, lint_file, lint_paths,
                   lint_source, lint_texts, summarize, warn_or_worse)
from .license import ALLOWED_LICENSES, Origin, allowed_license, license_check, normalize_license
from .notices import banner

__all__ = [
    "ALLOWED_LICENSES", "BLOCK", "Finding", "INFO", "Origin", "SEVERITIES", "WARN",
    "allowed_license", "banner", "blocks", "cold_start_gate", "export_check",
    "leading_comment_block", "license", "license_check", "lint", "lint_file", "lint_paths",
    "lint_source", "lint_texts", "multiseed", "normalize_license", "notices", "report",
    "snapshot", "summarize", "timers", "wrap_with_fallback", "warn_or_worse",
]

license = license_mod
lint = lint_mod
notices = notices_mod
multiseed = multiseed_mod
report = report_mod
snapshot = snapshot_mod
timers = timers_mod


def _targethw_findings(source: str, filename: str, language: str | None = None) -> list[Finding]:
    """The target rules (targethw, a soft dependency) as this package's findings."""
    try:
        from forgekernel.targethw import rules            # noqa: PLC0415 -- optional
    except ImportError:
        return []
    try:
        found = rules.check_submission(source, language)
    except Exception as exc:                              # noqa: BLE001
        return [Finding(rule="targethw-unavailable", severity=INFO, file=filename, line=0,
                        snippet=type(exc).__name__, detail=str(exc)[:200])]
    out = []
    for f in found:
        out.append(Finding(rule=getattr(f, "rule", "targethw"),
                           severity=getattr(f, "severity", INFO), file=filename,
                           line=int(getattr(f, "line", 0) or 0),
                           snippet=str(getattr(f, "snippet", "") or ""),
                           detail=str(getattr(f, "detail", "") or "")))
    return out


def export_check(candidate, *, origins=None, targethw: bool = True,
                 language: str | None = None, path: bool = True) -> dict:
    """Every static check on the file about to be exported.

    ``candidate`` is a path (``path=True``, the default) or the source text.
    ``origins`` is the licence list campaign carries (see ``license.license_check``).
    Returns ``{"ok", "blocks", "warnings", "infos", "findings", "path", "bytes", ...}``;
    ``ok`` is False when a ``block``-level finding exists.
    """
    if path and isinstance(candidate, (str, Path)) and Path(candidate).exists():
        p = Path(candidate)
        try:
            source = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"ok": False, "blocks": 1, "warnings": 0, "infos": 0, "path": str(p),
                    "findings": [Finding(rule="unreadable", severity=BLOCK, file=str(p),
                                         line=0, snippet=type(exc).__name__,
                                         detail=str(exc)[:200]).to_dict()],
                    "error": f"cannot read {p}: {exc}"}
        filename = str(p)
    else:
        source, filename = str(candidate), "<submission>"
    findings = lint_source(source, filename, submission=True)
    if targethw:
        findings += _targethw_findings(source, filename, language)
    licence = {}
    if origins:
        licence_findings = license_check(source, list(origins))
        findings += licence_findings
        licence = {"origins": [Origin.of(o).__dict__ for o in origins],
                   "problems": [f.detail or f.rule for f in blocks(licence_findings)]}
    seen: set[tuple[str, int]] = set()
    unique: list[Finding] = []
    for f in findings:
        key = (f.rule, f.line)
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)
    unique.sort(key=lambda f: (f.line, f.rule))
    blocked = blocks(unique)
    return {
        "ok": not blocked,
        "blocks": len(blocked),
        "warnings": len(warn_or_worse(unique)) - len(blocked),
        "infos": len(unique) - len(warn_or_worse(unique)),
        "findings": [f.to_dict() for f in unique],
        "summary": summarize(unique),
        "path": filename,
        "bytes": len(source.encode("utf-8")),
        **({"licence": licence} if origins else {}),
    }


def cold_start_gate(*paths, limit: float | None = None) -> dict:
    """The export's cold-start gate, read from a benchx result when one is readable.

    ``state`` is ``"ok"`` / ``"over-budget"`` / ``"unmeasured"``; ``unmeasured`` is what a
    machine with no bench result says, and the caller prints it as "not measured" rather
    than treating it as fast.  The budget is ``forgekernel.benchx.api.COLD_START_BUDGET_S``
    -- one number, in one place.
    """
    from forgekernel.benchx.api import COLD_START_BUDGET_S, cold_start_over_budget
    budget = COLD_START_BUDGET_S if limit is None else float(limit)
    reading, why = None, ""
    for given in paths:
        if given is None:
            continue
        try:
            data = Path(given)
        except TypeError:
            continue
        if not data.is_file():
            continue
        try:
            result = _read_bench_result(data)
        except Exception as exc:                          # noqa: BLE001
            why = f"{data}: {type(exc).__name__}: {exc}"
            continue
        reading = result
        break
    if reading is None:
        return {"state": "unmeasured", "cold_start_s": None, "budget_s": budget,
                "why": why or "the bench wrote no result for this candidate (not measured)"}
    cold = reading.get("cold_start_s")
    if cold is None:
        return {"state": "unmeasured", "cold_start_s": None, "budget_s": budget,
                "why": "the bench result carries no cold_start_s (not measured)"}
    over = (cold_start_over_budget(float(cold)) if limit is None
            else float(cold) > budget)
    if over:
        return {"state": "over-budget", "cold_start_s": float(cold), "budget_s": budget,
                "why": f"cold start {float(cold):.1f}s > {budget:.0f}s -- a board job that "
                       f"compiles this long is timed out at 0"}
    return {"state": "ok", "cold_start_s": float(cold), "budget_s": budget, "why": ""}


def _read_bench_result(path: Path) -> dict:
    """A benchx result file as a dict (its own format, or a wrapper with ``result``)."""
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("result"), dict):
        data = data["result"]
    if not isinstance(data, dict):
        raise ValueError("not a BenchResult object")
    return data
