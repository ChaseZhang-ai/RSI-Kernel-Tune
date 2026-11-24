"""Render selected records into the file a run reads as ``{LESSONS_PATH}``.

The file is written for a teacher agent (English, one block per record) and is assembled
**only from record fields**: every number in it is one of the record's own values, printed in
its own JSON spelling, and no sentence judges them. That is the same discipline
``docs/design/milestone-distillation.md`` §4.1bis asks for: mechanical rearrangement of
structured fields, no new claims.

Three explanation lines come first, then one block per record:

    # Operator-family lessons (mechanically extracted facts)

    ...three lines: what this is, where it comes from, how far the numbers carry...
    ## <problem>  (<family>)
    same problem                     <- only when the record is this problem's own earlier run
    - problem: ...
    - outcome: ...
    - kept / failed / meta / certified_kb / recon / usage / source

``same problem`` is the marker the user asked for: a later run of the *same* problem is
allowed to see what was measured on it before (the "distill from zero" requirement was
dropped), and the marker is what tells the teacher "this problem's own history" apart from
"another operator of this family". ``exclude_self=True`` removes those blocks entirely.

Same input, same bytes: record order is total (``query.order``), every field is read in a
fixed sequence, and nothing time-dependent is written.

::

    from forgekernel.opkb import render
    text = render.build(store.load_records(root), target, k=5)
"""

from __future__ import annotations

import json
from pathlib import Path

from forgekernel.opkb import query

TITLE = "# Operator-family lessons (mechanically extracted facts)"
INTRO = (
    "This file is data extracted mechanically from the finished runs listed below; no agent "
    "wrote any sentence of it.",
    "The records belong to other operators of this family; a block marked \"same problem\" is "
    "an earlier run of this problem itself.",
    "Numbers carry over only between close shapes and dtypes, and nothing here is a decision "
    "for this problem -- it is a prior to check against your own measurements.",
)
SAME_PROBLEM_MARK = "same problem"


def empty_lessons(target: query.Target) -> str:
    """The minimal file for a family with no records: says so, and is not an error.

    A teacher that reads it learns one true thing -- there is no measured prior for this
    class of kernel yet -- and the prompt's own rule ("proceed exactly as without the file")
    stays the operative instruction."""
    lines = [
        TITLE,
        "",
        f'No records of the "{target.family}" operator family are in the lesson base yet: no '
        f'measured prior exists for this class of kernel.',
        "This is not an error -- proceed as if this file were not set.",
        "",
    ]
    return "\n".join(lines)


def _num(value) -> str:
    """A number in its JSON spelling, so a rendered digit is always a substring of the record."""
    return json.dumps(value, ensure_ascii=False)


def _one_line(text) -> str:
    """Display text on one line (a stored newline cannot break the block layout)."""
    return str(text).replace("\r", " ").replace("\n", " ")


def _problem_line(problem: dict) -> str:
    parts = []
    for key in ("name", "definition", "suite", "track", "arch", "language", "workloads"):
        if key not in problem:
            continue
        value = problem[key]
        parts.append(f"{key}={_num(value) if isinstance(value, (int, float)) else value}")
    return "- problem: " + "; ".join(parts)


def _outcome_line(outcome: dict) -> str:
    parts = []
    metric, value = outcome.get("metric_kind"), outcome.get("value")
    if metric is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
        window = f" ({outcome['window']} window)" if outcome.get("window") else ""
        parts.append(f"{metric}={_num(value)}{window}")
    elif outcome.get("window"):
        parts.append(f"window={outcome['window']}")
    if "stop" in outcome:
        parts.append(f"stop={outcome['stop']}")
    if "student_iterations" in outcome:
        parts.append(f"student_iterations={_num(outcome['student_iterations'])}")
    if "hours" in outcome:
        parts.append(f"hours={_num(outcome['hours'])}")
    return "- outcome: " + "; ".join(parts) if parts else ""


def _kept_lines(kept) -> list[str]:
    lines = ["- kept:"]
    for item in kept:
        if not isinstance(item, dict):
            continue
        bits = [f"iter {_num(item['iter'])}" if "iter" in item else "iter ?"]
        if "change" in item:
            bits.append(_one_line(item["change"]))
        readings = []
        if "before_ms" in item:
            readings.append(f"before {_num(item['before_ms'])} ms")
        if "after_ms" in item:
            readings.append(f"after {_num(item['after_ms'])} ms")
        if readings:
            bits.append("(" + " -> ".join(readings) + ")")
        lines.append("  - " + ": ".join(bits[:2]) + (("  " + bits[2]) if len(bits) > 2 else ""))
    return lines


def _failed_lines(failed) -> list[str]:
    lines = ["- failed:"]
    for item in failed:
        if not isinstance(item, dict):
            continue
        bits = [f"iter {_num(item['iter'])}" if "iter" in item else "iter ?"]
        if "kind" in item:
            bits[0] += f" ({item['kind']})"
        if "change" in item:
            bits.append(_one_line(item["change"]))
        if "error" in item:
            bits.append("error: " + _one_line(item["error"]))
        lines.append("  - " + "  ".join(bits))
    return lines


def _meta_lines(meta) -> list[str]:
    lines = ["- meta:"]
    for item in meta:
        if not isinstance(item, dict):
            continue
        lines.append("  - round {}: {}".format(_num(item["round"]) if "round" in item else "?",
                                               item.get("outcome", "?")))
    return lines


def _recon_lines(recon: dict) -> list[str]:
    lines = ["- recon (verbatim entries from the reconnaissance deliverables):"]
    for item in recon.get("verdicts") or ():
        if not isinstance(item, dict) or "line" not in item:
            continue
        lines.append("  - " + _one_line(item["line"]))
    for item in recon.get("candidates") or ():
        if not isinstance(item, dict) or "decision" not in item:
            continue
        text = f"  - {_one_line(item['decision'])} = {_one_line(item.get('value', ''))}"
        if item.get("verify"):
            text += f"  [verify: {item['verify']}]"
        lines.append(text)
    return lines


def block(record: dict, target: query.Target) -> list[str]:
    """One record's block (marker line included when it is this problem's own run)."""
    problem = record.get("problem") if isinstance(record.get("problem"), dict) else {}
    outcome = record.get("outcome") if isinstance(record.get("outcome"), dict) else {}
    label = str(problem.get("definition") or problem.get("name") or record.get("id") or "?")
    lines = ["", f"## {label}  ({record.get('family') or '?'})"]
    if target.same_problem(record):
        lines.append(SAME_PROBLEM_MARK)
    lines.append(_problem_line(problem))
    reference = record.get("reference") if isinstance(record.get("reference"), dict) else {}
    if "kind" in reference:
        lines.append(f"- reference: kind={reference['kind']}")
    line = _outcome_line(outcome)
    if line:
        lines.append(line)
    if record.get("created"):
        lines.append(f"- created: {record['created']}")
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    bits = []
    if "run" in source:
        bits.append(f"run={source['run']}")
    if "best_sha" in source:
        bits.append(f"best_sha={source['best_sha']}")
    if bits:
        lines.append("- source: " + "; ".join(bits))
    for key, fn in (("kept", _kept_lines), ("failed", _failed_lines), ("meta", _meta_lines)):
        rows = record.get(key)
        if isinstance(rows, list) and rows:
            lines.extend(fn(rows))
    kb = record.get("certified_kb")
    if isinstance(kb, list) and kb:
        lines.append("- certified_kb:")
        lines.extend(f"  - {entry}" for entry in kb)
    recon = record.get("recon")
    if isinstance(recon, dict) and (recon.get("verdicts") or recon.get("candidates")):
        lines.extend(_recon_lines(recon))
    usage = record.get("usage")
    if isinstance(usage, dict) and usage:
        lines.append("- usage: " + "; ".join(f"{k}={_num(v)}" for k, v in usage.items()))
    return lines


def build(records: list[dict], target: query.Target, *, k: int = 5,
          exclude_self: bool = False) -> str:
    """The lesson file's text for ``target`` from ``records`` (bytes are input-determined)."""
    chosen = query.select(records, target, k=k, exclude_self=exclude_self)
    if not chosen:
        return empty_lessons(target)
    lines = [TITLE, "", *INTRO]
    for record in chosen:
        lines.extend(block(record, target))
    return "\n".join(lines) + "\n"


def write(path: str | Path, text: str) -> Path:
    """Write ``text`` to ``path`` (parent directories created)."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p
