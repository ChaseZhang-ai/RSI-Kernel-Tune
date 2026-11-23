"""The ledger and the local-vs-board ratio: what we measured, and what it was worth.

Every benchmark, every board reading and every refusal goes to one append-only
``ledger.jsonl`` -- one JSON object per line, no rewriting, no aggregation baked in. It
is the only record that survives a campaign, so both directions of the traffic the board
task and this one exchange go through it:

* **out** -- `record_results(rows, ledger)` is the entry `forge board import <campaign>`
  calls with its decoded submission results (background §8's soft dependency);
* **in** -- `campaign sync --from-saved <dir>` reads JSON the user saved from the board's
  web pages (a per-kernel leaderboard page, or a submission result) and appends the same
  kind of rows. No network, no token: the pages are the source either way.

``ratios.json`` is the derived view: for every problem with both a local reading and a
board reading, ``ratio = board S / dev S``; aggregated by operator family and by the
reading's confidence tags. It exists because the development machine is a B300 and the
board times a B200 (background §4): FP4-dense kernels read faster here than they score
there, SFU-heavy ones read much faster, and a persistent kernel tuned for a different SM
count can go either way. Until a problem has its own pair, the family's mean ratio is the
best available correction -- and it is always shown *as* a correction, never replacing
the measurement it came from.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from forgekernel.campaign import solscore, targets

LEDGER_FILE = "ledger.jsonl"
RATIOS_FILE = "ratios.json"
DEV = "dev"
BOARD = "board"
EVENTS = ("bench", "starter", "board", "compose", "pack", "submit", "error", "note")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Ledger:
    """Append-only JSONL. ``append`` is the only writer; readers take a snapshot."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, entry: dict) -> dict:
        row = {"ts": _now(), **entry}
        if row.get("event") not in EVENTS:
            raise ValueError(f"unknown ledger event {row.get('event')!r}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def entries(self) -> list[dict]:
        if not self.path.is_file():
            return []
        out = []
        for n, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"event": "error", "note": f"line {n} does not parse"})
        return out


def record_reading(ledger: Ledger, *, problem: str, track: str, kernel_id: int | None,
                   reading, source: str = DEV, candidate: str = "", run: str = "",
                   sha: str = "", event: str = "bench", note: str = "") -> dict:
    """One measured reading (campaign.tools.BenchReading) as a ledger line."""
    if source not in (DEV, BOARD):
        raise ValueError(f"source must be {DEV!r} or {BOARD!r}")
    return ledger.append({
        "event": event, "problem": problem, "track": track, "kernel_id": kernel_id,
        "source": source, "candidate": candidate or reading.candidate, "run": run, "sha": sha,
        "mean_s": reading.mean_s, "fast1": reading.fast1,
        "geomean_ms": reading.geomean_ms, "correct": bool(reading.correct),
        "cold_start_s": reading.cold_start_s, "job": reading.job,
        "confidence": list(reading.confidence or []), "note": note,
        "workloads": {w.name: {"t_ms": w.t_ms, "t_b": w.t_b, "t_sol": w.t_sol, "s": w.s}
                      for w in reading.workloads}})


def record_results(rows: list[dict], ledger_path: str | Path) -> int:
    """Append decoded board results (what `forge board import` reads) to the ledger.

    Each row is one submission result; only ``problem`` (or ``kernel_name``) is required.
    Recognised keys: ``mean_s`` / ``sol_score`` (the problem's score), ``source``
    ("board" by default), ``candidate``, ``sha``, ``note``, ``track``, ``kernel_id``,
    ``url``. Returns how many lines were written."""
    led = Ledger(ledger_path)
    n = 0
    for row in rows:
        name = row.get("problem") or row.get("kernel_name") or row.get("name")
        if not name:
            continue
        score = row.get("mean_s", row.get("sol_score", row.get("score")))
        led.append({"event": "board", "problem": str(name), "track": row.get("track") or "",
                    "kernel_id": row.get("kernel_id"),
                    "source": BOARD, "candidate": row.get("candidate") or "",
                    "sha": row.get("sha") or "", "mean_s": _f(score),
                    "fast1": row.get("fast1"), "correct": row.get("correct", True),
                    "confidence": list(row.get("confidence") or ["board"]),
                    "note": row.get("note") or "", "url": row.get("url") or ""})
        n += 1
    return n


def _f(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


# --- reading saved pages -------------------------------------------------------------------

def rows_from_saved(path: str | Path, *, username: str | None = None) -> list[dict]:
    """Rows out of JSON the user saved from the board.

    Two shapes are recognised, because both appear on the site:

    * a **per-kernel leaderboard page** (``{"data": {"kernel_name": ..., "rankings": [...]}}``)
      -- every non-reference entry is used, or only ``username``'s when that is given;
    * a **submission result** (one flat object with a score and a problem name) -- used
      as-is.
    """
    p = Path(path).expanduser()
    files = sorted(p.rglob("*.json")) if p.is_dir() else [p]
    out: list[dict] = []
    for f in files:
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        obj = obj.get("data", obj) if isinstance(obj, dict) else obj
        if isinstance(obj, dict) and obj.get("rankings"):
            name = obj.get("kernel_name") or obj.get("name")
            for r in obj["rankings"]:
                if not isinstance(r, dict) or r.get("is_reference"):
                    continue
                if username and r.get("username") != username:
                    continue
                out.append({"problem": name, "kernel_id": obj.get("kernel_id"),
                            "mean_s": r.get("sol_score"), "candidate": r.get("username"),
                            "source": BOARD, "fast1": r.get("fast_1_count"),
                            "url": str(f), "note": "saved per-kernel page"
                            + ("" if username else f" (user {r.get('username')})")})
        elif isinstance(obj, dict) and (obj.get("problem") or obj.get("kernel_name")
                                        or obj.get("submission_id")):
            out.append({"problem": obj.get("problem") or obj.get("kernel_name"),
                        "kernel_id": obj.get("kernel_id"),
                        "mean_s": obj.get("mean_s", obj.get("sol_score", obj.get("score"))),
                        "track": obj.get("track"), "source": BOARD,
                        "confidence": obj.get("confidence") or ["board"],
                        "url": str(f), "note": obj.get("note") or "saved result page"})
    return [r for r in out if r.get("problem")]


def import_saved(path: str | Path, ledger_path: str | Path, *,
                 username: str | None = None) -> tuple[int, list[str]]:
    """``--from-saved``: append every recognised row; ``(rows written, warnings)``."""
    rows = rows_from_saved(path, username=username)
    warns = []
    if not rows:
        warns.append(f"{path}: no per-kernel page and no result page was recognised")
        if username is None:
            warns.append("a per-kernel page lists every team -- pass --username to take "
                         "only yours")
    return record_results(rows, ledger_path), warns


# --- the ratio ------------------------------------------------------------------------------

def best_by_source(entries: list[dict], source: str) -> dict[str, dict]:
    """problem -> the highest-scoring correct entry of that source (a board reading wins
    over a local one only when it exists; they are kept apart here)."""
    out: dict[str, dict] = {}
    for e in entries:
        if e.get("source") != source or e.get("mean_s") is None or e.get("correct") is False:
            continue
        name = e.get("problem")
        if not name:
            continue
        cur = out.get(name)
        if cur is None or float(e["mean_s"]) > float(cur["mean_s"]):
            out[name] = e
    return out


def ratios(entries: list[dict]) -> dict:
    """``{problems, by_family, by_confidence, overall}`` from the ledger's own readings."""
    dev = best_by_source(entries, DEV)
    board = best_by_source(entries, BOARD)
    problems = {}
    for name in sorted(set(dev) & set(board)):
        d, b = float(dev[name]["mean_s"]), float(board[name]["mean_s"])
        if d <= 0:
            continue
        problems[name] = {"dev_s": d, "board_s": b, "ratio": b / d,
                          "track": board[name].get("track") or dev[name].get("track") or "",
                          "family": targets.family_of(name),
                          "confidence": sorted(set(dev[name].get("confidence") or [])),
                          "candidate": dev[name].get("candidate") or ""}
    by_family: dict[str, list[float]] = {}
    by_conf: dict[str, list[float]] = {}
    for name, p in problems.items():
        by_family.setdefault(p["family"], []).append(p["ratio"])
        for tag in (p["confidence"] or ["untagged"]):
            by_conf.setdefault(tag, []).append(p["ratio"])
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None          # noqa: E731
    return {"schema": 1, "computed": _now(), "n_pairs": len(problems),
            "problems": problems,
            "by_family": {k: {"n": len(v), "mean_ratio": mean(v)} for k, v in sorted(by_family.items())},
            "by_confidence": {k: {"n": len(v), "mean_ratio": mean(v)}
                              for k, v in sorted(by_conf.items())},
            "overall": {"n": len(problems), "mean_ratio": mean([p["ratio"] for p in problems.values()])},
            "dev_s_not_on_board": sorted(set(dev) - set(board)),
            "board_s_without_dev": sorted(set(board) - set(dev))}


def write_ratios(work_dir: str | Path, entries: list[dict]) -> Path:
    p = Path(work_dir) / RATIOS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ratios(entries), indent=1, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    return p


def correction(family: str, ratio_doc: dict) -> float | None:
    """The factor a local reading of ``family`` should be multiplied by to estimate its
    board score: the family's mean ratio when it has one, else the overall mean, else
    None (no correction available -- and None is what a caller must show, not 1.0)."""
    fam = (ratio_doc.get("by_family") or {}).get(family)
    if fam and fam.get("mean_ratio"):
        return float(fam["mean_ratio"])
    overall = (ratio_doc.get("overall") or {}).get("mean_ratio")
    return float(overall) if overall else None


def corrected_estimate(dev_s: float | None, family: str, ratio_doc: dict) -> float | None:
    """``dev_s x correction`` -- the local estimate as the board would score it, or None
    when either half is unknown."""
    f = correction(family, ratio_doc)
    return None if (dev_s is None or f is None) else float(dev_s) * f


def summary(entries: list[dict]) -> str:
    doc = ratios(entries)
    lines = [f"ledger: {len(entries)} entries, {doc['n_pairs']} problem(s) with both a dev "
             f"and a board reading"]
    if doc["overall"]["mean_ratio"]:
        lines.append(f"  board/dev overall {doc['overall']['mean_ratio']:.4f}")
    for fam, d in doc["by_family"].items():
        lines.append(f"  family {fam:<32} n={d['n']:<3} ratio {d['mean_ratio']:.4f}")
    for tag, d in doc["by_confidence"].items():
        lines.append(f"  tag {tag:<35} n={d['n']:<3} ratio {d['mean_ratio']:.4f}")
    if doc["dev_s_not_on_board"]:
        lines.append(f"  measured here, never submitted: {len(doc['dev_s_not_on_board'])}")
    return "\n".join(lines)


def ours_from_ledger(entries: list[dict]) -> dict[str, float]:
    """problem -> our best score, board reading preferred (it is the one that counts)."""
    out = dict((k, float(v["mean_s"])) for k, v in best_by_source(entries, DEV).items())
    out.update((k, float(v["mean_s"])) for k, v in best_by_source(entries, BOARD).items())
    return out


def track_of_problem(snap, name: str) -> str | None:
    for kid in snap.kernel_ids:
        if snap.name_of(kid) == name:
            return solscore.track_of(kid)
    return None
