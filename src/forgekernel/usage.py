"""The agent usage ledger: one JSONL line per agent turn, and its summaries.

Who writes it: ``scripts/agent_trailer.py``, which every launcher script calls after a
structured turn (``claude -p --output-format json`` / ``codex exec --json``). The loop sets
``FK_AGENT_USAGE_FILE=<run dir>/agent_usage.jsonl`` for the whole run (launch._entered), so
teacher, student and judge turns all land in one file.

What a record holds (missing numbers are ``null``, never 0 -- "the CLI did not say" is not
"no tokens")::

    ts, role, backend, model_requested, model_used, session_id, rc,
    input_tokens, output_tokens, cache_read_input_tokens,
    cache_creation_input_tokens, total_tokens,
    cost_usd, duration_s, refusal_check, warning?

**Display only.** There is no budget here, no cap and no blocking: token spend is not a
constraint, and a run's tokens are reported, per role, in ``forge status`` and in the run's
report. What this file is *read* for is the other tasks that display it (the throughput
task's report) and the audit trail of a run.
"""

from __future__ import annotations

import json
from pathlib import Path

LEDGER_NAME = "agent_usage.jsonl"
TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens",
              "cache_creation_input_tokens", "total_tokens")
WARNING_KINDS = ("refusal_fallback", "model_mismatch")


def ledger_path(run_dir: str | Path) -> Path:
    """The ledger of one run."""
    return Path(run_dir) / LEDGER_NAME


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def read_ledger(path: str | Path) -> list[dict]:
    """Every readable record of the ledger, in file order. A missing file is no records;
    a line that is not a JSON object is skipped (a half-written last line of a killed run
    is the normal case) -- the ledger is a report, and one bad line must not hide the
    rest."""
    p = Path(path)
    if not p.is_file():
        return []
    out: list[dict] = []
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def _bucket() -> dict:
    return {"turns": 0, **{k: 0 for k in TOKEN_KEYS}, "cost_usd": 0.0,
            "duration_s": 0.0, "unknown_usage": 0}


def _add(bucket: dict, record: dict) -> None:
    bucket["turns"] += 1
    total = _number(record.get("total_tokens"))
    if total is None:
        bucket["unknown_usage"] += 1
    for key in TOKEN_KEYS:
        value = _number(record.get(key))
        if value is not None:
            bucket[key] += value
    for key in ("cost_usd", "duration_s"):
        value = _number(record.get(key))
        if value is not None:
            bucket[key] += value


def summarize(records: list[dict]) -> dict:
    """Totals, and the same totals per role and per model.

    ``unknown_usage`` is the number of turns whose ``total_tokens`` was missing, so a
    reader can tell a sum that is complete from one that is only a floor."""
    totals = _bucket()
    by_role: dict[str, dict] = {}
    by_model: dict[str, dict] = {}
    warnings = {kind: 0 for kind in WARNING_KINDS}
    for record in records:
        _add(totals, record)
        role = record.get("role") if isinstance(record.get("role"), str) else "unknown"
        _add(by_role.setdefault(role or "unknown", _bucket()), record)
        model = record.get("model_used") or record.get("model_requested") or "unknown"
        _add(by_model.setdefault(model, _bucket()), record)
        warning = record.get("warning")
        if warning in warnings:
            warnings[warning] += 1
    return {"turns": totals["turns"], "totals": totals, "by_role": by_role,
            "by_model": by_model, "warnings": warnings}


def _tokens(n: float) -> str:
    return f"{int(n):,}"


def _money(x: float) -> str:
    return f"{x:,.4f}" if x and x < 1 else f"{x:,.2f}"


def _duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _row(name: str, bucket: dict) -> str:
    unknown = f" ({bucket['unknown_usage']} without usage)" if bucket["unknown_usage"] else ""
    return (f"| {name}{unknown} | {bucket['turns']} | {_tokens(bucket['input_tokens'])} "
            f"| {_tokens(bucket['output_tokens'])} "
            f"| {_tokens(bucket['cache_read_input_tokens'])} "
            f"| {_tokens(bucket['cache_creation_input_tokens'])} "
            f"| {_tokens(bucket['total_tokens'])} | {_money(bucket['cost_usd'])} "
            f"| {_duration(bucket['duration_s'])} |")


_HEADER = ("| | turns | input | output | cache read | cache write | total | cost (USD) "
           "| agent time |\n|---|---|---|---|---|---|---|---|---|")


def render_markdown(summary: dict) -> str:
    """The ledger as a report section body: totals, then per role and per model."""
    if not summary.get("turns"):
        return "No agent turns were recorded."
    lines = ["Token usage per agent turn, as the agent CLIs reported it (display only: no "
             "budget, no cap).", "",
             _HEADER, _row("**all turns**", summary["totals"]), ""]
    lines += ["### By role", "", _HEADER]
    order = [r for r in ("teacher", "student", "judge") if r in summary["by_role"]]
    order += sorted(r for r in summary["by_role"] if r not in order)
    lines += [_row(role, summary["by_role"][role]) for role in order]
    lines += ["", "### By model", "", _HEADER]
    lines += [_row(model, summary["by_model"][model])
              for model in sorted(summary["by_model"])]
    flagged = {k: v for k, v in summary["warnings"].items() if v}
    if flagged:
        lines += ["", "Downgraded turns: " + ", ".join(
            f"{kind} x{count}" for kind, count in sorted(flagged.items()))
            + " (see docs/safeguard-refusals.md; the turn did not run on the model the "
              "task asked for)."]
    return "\n".join(lines)


def summary_line(summary: dict) -> str:
    """One line for ``forge status``: turns, tokens, cost, agent time."""
    totals = summary.get("totals") or _bucket()
    if not summary.get("turns"):
        return ""
    text = (f"{summary['turns']} turn(s), {_tokens(totals['total_tokens'])} tokens, "
            f"${_money(totals['cost_usd'])}, {_duration(totals['duration_s'])} agent time")
    if totals["unknown_usage"]:
        text += f" ({totals['unknown_usage']} turn(s) without usage)"
    return text
