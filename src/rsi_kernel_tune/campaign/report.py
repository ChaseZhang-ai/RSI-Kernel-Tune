"""The campaign's own paperwork: the gap list, the instrument panel, the ratio ledger.

Three outputs, all mechanical, all reading the same two sources (the board snapshot and
the campaign's ledger):

* **`report.md`** -- the four-track instrument panel, the biggest gaps, per-problem state,
  and the local/board ratio with the corrected estimates it implies;
* **`report.csv`** -- one line per problem: gap, leader, best known, ours, start, state,
  export, local score, board score, ratio, corrected estimate;
* **`report.json`** -- everything above, for another tool to read.

The instrument panel is the "how far from first" answer: per track, our mean score
(missing problems count 0), the best-known mean, and the incumbent's mean -- and the same
three weighted into one overall number. `targets.instrument` computes it, so these
numbers and `campaign targets` can never disagree.

A local estimate is always shown next to its correction: a dev score from the B300
multiplied by the family's board/dev ratio, marked with the basis. With no board reading
for that family the estimate stays `None` and the report says "uncorrected" rather than
pretending the correction is 1.0.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from rsi_kernel_tune.campaign import solscore, sync
from rsi_kernel_tune.campaign import targets as targets_mod

REPORT_MD = "report.md"
REPORT_CSV = "report.csv"
REPORT_JSON = "report.json"


def _fmt(v: float | None, width: int = 6, digits: int = 4) -> str:
    return "" if v is None else f"{v:.{digits}f}".rjust(width)


def state_of(work_dir: Path, op: str | None) -> dict:
    if not op:
        return {"op": "-", "planned": False, "status": "-", "export": None,
                "export_kind": "", "candidates": 0, "runs": [], "error": ""}
    p = Path(work_dir) / "state" / f"{op}.json"
    state = {}
    if p.is_file():
        try:
            state = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
    return {"op": op, "planned": True, "status": state.get("status", "planned"),
            "export": (state.get("export") or {}).get("path"),
            "export_kind": (state.get("export") or {}).get("kind", ""),
            "candidates": len(state.get("candidates") or []),
            "runs": state.get("runs") or [], "error": state.get("error", "")}


def build(rows: list[targets_mod.TargetRow], snap: targets_mod.Snapshot, incumbent: str,
          why: dict, *, work_dir: str | Path, plan: dict | None = None,
          entries: list[dict] | None = None) -> dict:
    """The whole report as a dict; the writers below read only this."""
    work = Path(work_dir)
    entries = entries if entries is not None else sync.Ledger(work / sync.LEDGER_FILE).entries()
    ratio_doc = sync.ratios(entries)
    planned = {e["name"]: e.get("op") for e in (plan or {}).get("problems") or []}
    dev = sync.best_by_source(entries, sync.DEV)
    board = sync.best_by_source(entries, sync.BOARD)

    problems = []
    for r in rows:
        family = targets_mod.family_of(r.name)
        dev_s = _score(dev.get(r.name))
        board_s = _score(board.get(r.name))
        ratio = (ratio_doc["problems"].get(r.name) or {}).get("ratio")
        corrected = sync.corrected_estimate(dev_s, family, ratio_doc)
        basis = "dev x family ratio"
        if board_s is not None:
            corrected, basis = board_s, "board reading"
        elif corrected is None:
            basis = "uncorrected (no board/dev ratio for this family)"
        st = state_of(work, planned.get(r.name))
        problems.append({
            "kernel_id": r.kernel_id, "name": r.name, "track": r.track,
            "subset": solscore.TRACK_SUBSET[r.track], "family": family,
            "leader": r.leader, "best_known": r.best_known, "best_known_by": r.best_known_by,
            "gap": r.gap, "ours": r.ours, "starter": r.starter,
            "starter_detail": r.starter_detail, "public_s": r.public_s,
            "priority": r.priority, "priority_why": r.priority_why,
            "dev_s": dev_s, "board_s": board_s, "ratio": ratio,
            "estimated_s": corrected, "estimate_basis": basis,
            **{k: st[k] for k in ("op", "planned", "status", "export", "export_kind",
                                  "candidates", "runs", "error")}})
    return {"schema": 1, "snapshot": str(snap.dir), "incumbent": incumbent,
            "incumbent_why": why, "instrument": targets_mod.instrument(rows, snap, incumbent),
            "ratios": ratio_doc, "warnings": snap.warnings, "problems": problems,
            "ledger_entries": len(entries)}


def _score(entry: dict | None) -> float | None:
    if not entry or entry.get("mean_s") is None:
        return None
    try:
        return float(entry["mean_s"])
    except (TypeError, ValueError):
        return None


def instrument_markdown(inst: dict) -> list[str]:
    lines = ["| track | ours | best known | leader (incumbent) | done | problems |",
             "|---|---|---|---|---|---|"]
    for t in solscore.TRACKS:
        d = inst["tracks"][t]
        lines.append(f"| {t} | {d['ours']:.4f} | {d['best_known']:.4f} | "
                     f"{d['leader']:.4f} | {d['done']} | {d['problems']} |")
    o = inst["overall"]
    lines.append(f"| **overall** | {o['ours']:.4f} | {o['best_known']:.4f} | "
                 f"{o['leader']:.4f} | | 235 |")
    return lines


def gap_markdown(problems: list[dict], limit: int = 15) -> list[str]:
    lines = ["| gap | id | track | problem | leader | best known | holder | ours | start | state |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(problems, key=lambda x: -x["gap"])[:limit]:
        best = "" if p["best_known"] is None else f"{p['best_known']:.4f}"
        ours = _fmt(p["ours"])
        lines.append(f"| {p['gap']:+.4f} | {p['kernel_id']} | {p['track']} | {p['name']} | "
                     f"{p['leader']:.4f} | {best} | {p['best_known_by']} | {ours} | "
                     f"{p['starter']} | {p['status']} |")
    return lines


def ratio_markdown(doc: dict, problems: list[dict]) -> list[str]:
    lines = ["", "## local vs board", ""]
    if not doc.get("n_pairs"):
        lines += ["No problem has both a local reading and a board reading yet, so no",
                  "correction exists -- the estimates above are raw dev-machine numbers.",
                  ""]
        return lines
    lines += [f"{doc['n_pairs']} problem(s) with both readings; overall ratio "
              f"{doc['overall']['mean_ratio']:.4f} (board S / dev S).", "",
              "| family | n | mean ratio |", "|---|---|---|"]
    for fam, d in doc["by_family"].items():
        lines.append(f"| {fam} | {d['n']} | {d['mean_ratio']:.4f} |")
    lines += ["", "| reading tag | n | mean ratio |", "|---|---|---|"]
    for tag, d in doc["by_confidence"].items():
        lines.append(f"| {tag} | {d['n']} | {d['mean_ratio']:.4f} |")
    measured = [p for p in problems if p["dev_s"] is not None or p["board_s"] is not None]
    if measured:
        lines += ["", "| problem | dev S | board S | ratio | estimate | basis |",
                  "|---|---|---|---|---|---|"]
        for p in measured[:40]:
            lines.append(f"| {p['name']} | {_fmt(p['dev_s'])} | {_fmt(p['board_s'])} | "
                         f"{_fmt(p['ratio'], 6, 3)} | {_fmt(p['estimated_s'])} | "
                         f"{p['estimate_basis']} |")
    return lines


def markdown(doc: dict, *, gap_limit: int = 15) -> str:
    problems = doc["problems"]
    why = doc["incumbent_why"]
    lines = ["# campaign report", "",
             f"- snapshot: `{doc['snapshot']}`",
             f"- incumbent: **{doc['incumbent']}** ({why.get('by')}, overall "
             f"{why.get('overall', 0):.6f})",
             f"- ledger: {doc['ledger_entries']} entries", "",
             "## instrument panel (how far from first)", ""]
    lines += instrument_markdown(doc["instrument"])
    lines += ["", f"## biggest gaps (top {gap_limit})", ""]
    lines += gap_markdown(problems, limit=gap_limit)
    lines += ratio_markdown(doc["ratios"], problems)
    planned = [p for p in problems if p["planned"]]
    if planned:
        lines += ["", "## planned problems", "",
                  "| problem | track | status | candidates | export | estimate |",
                  "|---|---|---|---|---|---|"]
        for p in planned:
            name = Path(p["export"]).name if p["export"] else "-"
            lines.append(f"| {p['name']} | {p['track']} | {p['status']} | "
                         f"{p['candidates']} | {name} | {_fmt(p['estimated_s'])} |")
            if p["error"]:
                lines.append(f"| ^ | | | | | error: {p['error'][:120]} |")
    if doc["warnings"]:
        lines += ["", "## warnings", ""] + [f"- {w}" for w in doc["warnings"]]
    lines.append("")
    return "\n".join(lines)


def csv_text(doc: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["kernel_id", "name", "track", "family", "leader", "best_known",
                "best_known_by", "gap", "ours", "starter", "status", "export", "dev_s",
                "board_s", "ratio", "estimated_s", "estimate_basis"])
    for p in doc["problems"]:
        w.writerow([p["kernel_id"], p["name"], p["track"], p["family"],
                    f"{p['leader']:.6f}",
                    "" if p["best_known"] is None else f"{p['best_known']:.6f}",
                    p["best_known_by"], f"{p['gap']:.6f}",
                    "" if p["ours"] is None else f"{p['ours']:.6f}", p["starter"],
                    p["status"], p["export"] or "", _num(p["dev_s"]), _num(p["board_s"]),
                    _num(p["ratio"]), _num(p["estimated_s"]), p["estimate_basis"]])
    return buf.getvalue()


def _num(v: float | None) -> str:
    return "" if v is None else f"{v:.6f}"


def write(doc: dict, out_dir: str | Path, *, gap_limit: int = 15) -> dict:
    """Write report.md / report.csv / report.json under ``out_dir``."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    md, cs, js = d / REPORT_MD, d / REPORT_CSV, d / REPORT_JSON
    md.write_text(markdown(doc, gap_limit=gap_limit), encoding="utf-8")
    cs.write_text(csv_text(doc), encoding="utf-8")
    js.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"markdown": md, "csv": cs, "json": js}
