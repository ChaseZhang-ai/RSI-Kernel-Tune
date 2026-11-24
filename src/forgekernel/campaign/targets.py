"""What to work on: the board snapshot as a per-problem gap list.

The board publishes three things per problem: the *collection* rankings (one score per
team per track), the *per-kernel* rankings (one score per team per problem), and the
`kernels/<id>` payload with the timing baseline. This module reads the saved snapshot
(`reports/board-snapshot-<date>/`) and answers the only question a campaign needs at
the start: **for each problem, how far is the team we must beat from the best score
anyone has reached, and is a starting point at hand?**

Definitions, pinned to what the data check out as:

* ``leader`` -- the score of *the incumbent*: the team with the highest weighted
  overall score (sum over the four tracks of the team's per-problem scores divided by
  the 235 problems, i.e. an unscored problem counts 0). ``--incumbent`` overrides it.
  This is the number the background's table quotes (Amir M. Mir | SF Tensor:
  0.818852 / 0.820129 / 0.905855 / 0.805878, overall 0.83008) and reproducing it to
  the last printed digit is `tests/test_campaign_targets.py`'s first assertion.
* ``best_known`` -- the highest score any non-reference entry has on the per-kernel
  leaderboard, and who holds it. Averaged per track it is the "ceiling" row
  (0.841240 / 0.857025 / 0.936968 / 0.819011 -- also asserted).
* ``gap = best_known - (ours or leader)`` -- what is on the table for us.
* Rows sort by gap descending; a tie goes to the problem with an exact starting point
  (a reference library kernel for this very problem) over one with only a family
  neighbour.

The feasibility prior (background §3.5, task §6) is a second, *prior-only* signal: the
public Cursor-235 solution's per-problem score (v1.0) from
``cursor_problem_level_metrics.csv``. It never changes what "the gap" is; it only
reorders `--pilot auto`, which multiplies the gap by a factor (>= 0.7: x1.2, < 0.4:
x0.7, in between or unknown: x1.0) and prints the factor's reason next to the pick.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

from forgekernel.campaign import solscore

SNAPSHOT_GLOB = "board-snapshot-*"
# Why a problem scores what it scores, in one word, for the report's tables.
STARTER_EXACT = "exact"
STARTER_FAMILY = "family"
STARTER_BASELINE = "baseline"
STARTER_UNKNOWN = "unknown"


def default_snapshot(repo: Path) -> Path | None:
    """The newest ``reports/board-snapshot-*`` beside the repository, or beside its
    parent -- the background's layout puts `_reference/`, `reports/` and `prompts/` next
    to the base repository, and a git worktree lives one level deeper
    (``<root>/worktrees/<task>``), so a few ancestors are tried before giving up. None
    when none exists -- the caller then has to pass one."""
    start = Path(repo).resolve()
    for base in [start, *list(start.parents)[:4]]:
        reports = base / "reports"
        if not reports.is_dir():
            continue
        hits = sorted(p for p in reports.glob(SNAPSHOT_GLOB) if p.is_dir())
        if hits:
            return hits[-1]
    return None


@dataclass
class Snapshot:
    """The saved board pages. ``records`` is keyed by kernel id (1..235)."""
    dir: Path
    records: dict[int, dict] = field(default_factory=dict)
    collections: dict[str, dict] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)
    cursor: dict[tuple[str, str], float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def kernel_ids(self) -> list[int]:
        return sorted(self.records)

    def track_of(self, kernel_id: int) -> str | None:
        return solscore.track_of(kernel_id)

    def name_of(self, kernel_id: int) -> str:
        rec = self.records.get(kernel_id) or {}
        return self.names.get(kernel_id) or str(rec.get("kernel_name") or rec.get("kernel_title") or kernel_id)


def load_snapshot(path: str | Path) -> Snapshot:
    """Read a snapshot directory: `perkernel/<id>.json`, `collection_<n>.json`,
    `leaderboards.json`, and `cursor_problem_level_metrics.csv` when present."""
    d = Path(path).expanduser().resolve()
    if not d.is_dir():
        raise FileNotFoundError(f"no board snapshot at {d}")
    snap = Snapshot(dir=d)

    for p in sorted((d / "perkernel").glob("*.json")) if (d / "perkernel").is_dir() else []:
        obj = json.loads(p.read_text(encoding="utf-8"))
        obj = obj.get("data", obj)
        try:
            kid = int(obj.get("kernel_id", p.stem))
        except (TypeError, ValueError):
            snap.warnings.append(f"{p.name}: no kernel_id")
            continue
        snap.records[kid] = obj

    lb = d / "leaderboards.json"
    if lb.is_file():
        data = json.loads(lb.read_text(encoding="utf-8")).get("data", {})
        for k in data.get("kernel_leaderboards") or []:
            try:
                snap.names[int(k["kernel_id"])] = str(k.get("name") or k.get("display_name"))
            except (KeyError, TypeError, ValueError):
                continue
    for p in sorted(d.glob("collection_*.json")):
        data = json.loads(p.read_text(encoding="utf-8")).get("data", {})
        key = str(data.get("collection_name") or p.stem)
        snap.collections[key] = data

    cur = d / "cursor_problem_level_metrics.csv"
    if cur.is_file():
        for row in csv.DictReader(cur.open(encoding="utf-8")):
            try:
                snap.cursor[(row["artifact_id"], row["subset"])] = float(row["selected_SOL"])
            except (KeyError, TypeError, ValueError):
                continue
    else:
        snap.warnings.append(f"no {cur.name}: the feasibility prior is off (gap ordering only)")

    missing = [t for t in solscore.TRACKS if t not in snap.collections]
    if missing:
        snap.warnings.append(f"no collection page for {', '.join(missing)}")
    if not snap.records:
        raise ValueError(f"{d}: no per-kernel pages under perkernel/")
    return snap


# --- the incumbent and the two per-problem numbers ----------------------------------------

def _rankings(record: dict) -> list[dict]:
    return [r for r in (record.get("rankings") or [])
            if isinstance(r, dict) and not r.get("is_reference") and r.get("sol_score") is not None]


def problem_scores(snap: Snapshot, username: str) -> dict[int, float]:
    """One team's score on every problem it appears on (a missing problem is absent)."""
    out = {}
    for kid, rec in snap.records.items():
        for r in _rankings(rec):
            if r.get("username") == username:
                out[kid] = float(r["sol_score"])
                break
    return out


def overall_scores(snap: Snapshot) -> dict[str, float]:
    """Every team's weighted overall score (unscored problems count 0)."""
    per: dict[str, dict[int, float]] = {}
    for kid, rec in snap.records.items():
        for r in _rankings(rec):
            u = str(r.get("username"))
            per.setdefault(u, {})[kid] = float(r["sol_score"])
    total = sum(solscore.track_size(t) for t in solscore.TRACKS)
    return {u: sum(scores.values()) / total for u, scores in per.items()}


def pick_incumbent(snap: Snapshot, username: str | None = None) -> tuple[str, dict]:
    """``(username, why)`` -- the team whose per-problem scores are the bar to beat."""
    over = overall_scores(snap)
    if not over:
        raise ValueError(f"{snap.dir}: no non-reference ranking anywhere")
    if username:
        if username not in over:
            cands = sorted(over, key=lambda u: -over[u])[:5]
            raise ValueError(f"incumbent {username!r} is not on this board; the top five "
                             f"are: {', '.join(cands)}")
        return username, {"by": "explicit", "overall": over[username]}
    best = max(over.items(), key=lambda kv: kv[1])
    return best[0], {"by": "highest weighted overall score", "overall": best[1],
                     "runners_up": sorted(((u, s) for u, s in over.items() if u != best[0]),
                                          key=lambda kv: -kv[1])[:3]}


def best_known(snap: Snapshot, kernel_id: int) -> tuple[float | None, str]:
    rs = _rankings(snap.records.get(kernel_id) or {})
    if not rs:
        return None, ""
    top = max(rs, key=lambda r: float(r["sol_score"]))
    return float(top["sol_score"]), str(top.get("username") or "")


def track_means(snap: Snapshot, scores: dict[int, float] | None) -> dict[str, float]:
    return {t: solscore.track_mean(scores or {}, t) for t in solscore.TRACKS}


def best_known_means(snap: Snapshot) -> dict[str, float]:
    per = {kid: (best_known(snap, kid)[0] or 0.0) for kid in snap.kernel_ids}
    return {t: solscore.track_mean(per, t) for t in solscore.TRACKS}


# --- rows ----------------------------------------------------------------------------------

@dataclass
class TargetRow:
    kernel_id: int
    name: str
    track: str
    leader: float
    best_known: float | None
    best_known_by: str
    ours: float | None = None
    starter: str = STARTER_UNKNOWN
    starter_detail: str = ""
    public_s: float | None = None      # the Cursor-235 prior (v1.0), data only
    priority: float | None = None      # gap x feasibility factor (--pilot auto)
    priority_why: str = ""

    @property
    def gap(self) -> float:
        base = self.ours if self.ours is not None else self.leader
        return (self.best_known or 0.0) - base


def _feasibility(public: float | None) -> tuple[float, str]:
    if public is None:
        return 1.0, "public solution unknown -> neutral"
    if public >= 0.7:
        return 1.2, f"public solution {public:.3f} >= 0.7 -> x1.2"
    if public < 0.4:
        return 0.7, f"public solution {public:.3f} < 0.4 -> x0.7"
    return 1.0, f"public solution {public:.3f} -> neutral"


def rows(snap: Snapshot, *, incumbent: str | None = None, ours: dict[str, float] | None = None,
         starters: dict[str, tuple[str, str]] | None = None,
         with_prior: bool = True) -> tuple[list[TargetRow], str, dict]:
    """Every problem of the board as a TargetRow, with the incumbent's name resolved.

    ``ours``: problem name -> best score we hold (from the ledger). ``starters``:
    problem name -> (situation, detail) from `starter_situation`."""
    who, why = pick_incumbent(snap, incumbent)
    mine = problem_scores(snap, who)
    rows_out = []
    for kid in snap.kernel_ids:
        track = snap.track_of(kid)
        if track is None:
            continue
        name = snap.name_of(kid)
        bk, by = best_known(snap, kid)
        public = snap.cursor.get((name, solscore.TRACK_SUBSET[track])) if with_prior else None
        row = TargetRow(kernel_id=kid, name=name, track=track,
                        leader=float(mine.get(kid, 0.0)), best_known=bk, best_known_by=by,
                        ours=(ours or {}).get(name), public_s=public)
        if starters and name in starters:
            row.starter, row.starter_detail = starters[name]
        factor, why_prior = _feasibility(public)
        row.priority = row.gap * factor
        row.priority_why = why_prior
        rows_out.append(row)
    return rows_out, who, why


def ranked(rows_in: list[TargetRow], *, by: str = "gap") -> list[TargetRow]:
    """Gap descending (ties: an exact starting point first, then kernel id); ``by="priority"``
    applies the feasibility factor (what `--pilot auto` reads)."""
    exact = {STARTER_EXACT: 0, STARTER_FAMILY: 1, STARTER_BASELINE: 2, STARTER_UNKNOWN: 3}
    if by == "priority":
        return sorted(rows_in, key=lambda r: (-(r.priority or 0.0), exact.get(r.starter, 9),
                                              r.kernel_id))
    return sorted(rows_in, key=lambda r: (-r.gap, exact.get(r.starter, 9), r.kernel_id))


def instrument(rows_in: list[TargetRow], snap: Snapshot, incumbent: str) -> dict:
    """The "how far from first" dashboard: per track, our mean / best-known mean / the
    incumbent's mean, and the weighted overall of each."""
    ours_by_track = {t: solscore.track_mean({r.kernel_id: (r.ours or 0.0) for r in rows_in}, t)
                     for t in solscore.TRACKS}
    best_by_track = {t: solscore.track_mean({r.kernel_id: (r.best_known or 0.0)
                                             for r in rows_in}, t) for t in solscore.TRACKS}
    lead_by_track = {t: solscore.track_mean({r.kernel_id: r.leader for r in rows_in}, t)
                     for t in solscore.TRACKS}
    return {"incumbent": incumbent, "tracks": {
        t: {"ours": ours_by_track[t], "best_known": best_by_track[t],
            "leader": lead_by_track[t],
            "problems": solscore.track_size(t),
            "done": sum(1 for r in rows_in if r.ours is not None and r.track == t)}
        for t in solscore.TRACKS},
        "overall": {"ours": solscore.weighted_overall(ours_by_track),
                    "best_known": solscore.weighted_overall(best_by_track),
                    "leader": solscore.weighted_overall(lead_by_track)}}


# --- starting points (soft: the reference library index) ----------------------------------

def family_of(name: str) -> str:
    """The operator family a problem name belongs to. `forgekernel.family.family_key`
    when that module exists (the opkb/targethw task), else the name without its
    `NNN_` prefix -- mechanical either way, never guessed."""
    try:
        from forgekernel import family as family_mod  # type: ignore
    except ImportError:
        family_mod = None
    fn = getattr(family_mod, "family_key", None) if family_mod is not None else None
    if fn is not None:
        try:
            return str(fn(name))
        except Exception:                                       # noqa: BLE001
            pass
    parts = name.split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else name


def load_klib_index(root: str | Path) -> dict:
    """`kernel_library/index.json` (seedlib) as ``{id: entry}``; {} when absent.

    The shipped index is ``forgekernel.klib.schema``'s shape (``{"version", "entries"}``),
    so it is read through that loader when it is installed; a hand-written fixture (a list,
    or ``{"kernels": [...]}``) is read tolerantly either way."""
    path = Path(root) / "kernel_library" / "index.json"
    if not path.is_file():
        return {}
    entries = None
    try:
        from forgekernel.klib import schema as klib_schema  # type: ignore
    except ImportError:
        klib_schema = None
    if klib_schema is not None:
        try:
            loaded = klib_schema.load_index(path)
        except Exception:      # noqa: BLE001 -- a fixture, or another klib version's shape
            loaded = None
        if loaded is not None:
            entries = [e.to_dict() for e in loaded]
    if entries is not None:
        return {str(e.get("id")): e for e in entries if e.get("id")}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        entries = obj.get("entries", obj.get("kernels", obj))
    else:
        entries = obj
    out = {}
    if isinstance(entries, list):
        for e in entries:
            if isinstance(e, dict) and e.get("id"):
                out[str(e["id"])] = e
    elif isinstance(entries, dict):
        out = {str(k): v for k, v in entries.items() if isinstance(v, dict)}
    return out


def starter_situation(snap: Snapshot, root: str | Path) -> dict[str, tuple[str, str]]:
    """Per problem: (exact | family | baseline | unknown, a one-line detail).

    "exact" needs a reference-library entry whose ``problem_keys`` names this very
    problem; "family" only a same-family entry. With no reference library every
    problem is "baseline" -- the torch reference implementation and its generated
    baselines are always available to `forge lib baselines` -- and the detail says the
    library was not built."""
    index = load_klib_index(root)
    by_problem: dict[str, list[dict]] = {}
    by_family: dict[str, list[dict]] = {}
    for e in index.values():
        if e.get("usable_in_submission") is False:      # in the library, not a start
            continue
        for key in (e.get("problem_keys") or []):
            by_problem.setdefault(str(key), []).append(e)
        if e.get("family"):
            by_family.setdefault(str(e["family"]), []).append(e)
    out = {}
    for kid in snap.kernel_ids:
        name = snap.name_of(kid)
        fam = family_of(name)
        exact = by_problem.get(name) or []
        family = [e for e in by_family.get(fam, []) if e not in exact]
        if exact:
            out[name] = (STARTER_EXACT, f"{len(exact)} reference kernel(s) name this problem")
        elif family:
            out[name] = (STARTER_FAMILY, f"{len(family)} same-family reference(s), none exact")
        elif index:
            out[name] = (STARTER_BASELINE, f"no reference for family {fam!r}; torch baselines only")
        else:
            out[name] = (STARTER_UNKNOWN, "no reference library yet; torch baselines only")
    return out


# --- text / files -------------------------------------------------------------------------

def format_table(rows_in: list[TargetRow], *, limit: int | None = None,
                 with_prior: bool = True) -> str:
    lines = [f"{'gap':>7} {'id':>4} {'track':<16} {'problem':<52} {'leader':>6} {'best':>6} "
             f"{'ours':>6} {'start':<8} best-by"]
    for r in (rows_in if limit is None else rows_in[:limit]):
        best = f"{r.best_known:6.4f}" if r.best_known is not None else f"{'-':>6}"
        ours = f"{r.ours:6.4f}" if r.ours is not None else f"{'-':>6}"
        lines.append(f"{r.gap:+7.4f} {r.kernel_id:4d} {r.track:<16} {r.name[:52]:<52} "
                     f"{r.leader:6.4f} {best} {ours} {r.starter:<8} {r.best_known_by}")
    return "\n".join(lines)


def format_instrument(inst: dict) -> str:
    lines = [f"incumbent: {inst['incumbent']}",
             f"{'track':<16} {'ours':>8} {'best':>8} {'leader':>8} {'done':>5} {'n':>4}"]
    for t in solscore.TRACKS:
        d = inst["tracks"][t]
        lines.append(f"{t:<16} {d['ours']:8.4f} {d['best_known']:8.4f} {d['leader']:8.4f} "
                     f"{d['done']:5d} {d['problems']:4d}")
    o = inst["overall"]
    lines.append(f"{'OVERALL':<16} {o['ours']:8.4f} {o['best_known']:8.4f} {o['leader']:8.4f}")
    return "\n".join(lines)


def to_json(rows_in: list[TargetRow], snap: Snapshot, incumbent: str, why: dict,
            inst: dict) -> dict:
    return {"schema": 1, "snapshot": str(snap.dir), "incumbent": incumbent,
            "incumbent_why": why, "instrument": inst,
            "warnings": snap.warnings,
            "problems": [{"kernel_id": r.kernel_id, "name": r.name, "track": r.track,
                          "subset": solscore.TRACK_SUBSET[r.track],
                          "leader": r.leader, "best_known": r.best_known,
                          "best_known_by": r.best_known_by, "gap": r.gap, "ours": r.ours,
                          "starter": r.starter, "starter_detail": r.starter_detail,
                          "public_s": r.public_s, "priority": r.priority,
                          "priority_why": r.priority_why} for r in rows_in]}


def to_csv(rows_in: list[TargetRow]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["kernel_id", "name", "track", "leader", "best_known", "best_known_by",
                "gap", "ours", "starter", "public_s", "priority"])
    for r in rows_in:
        w.writerow([r.kernel_id, r.name, r.track, f"{r.leader:.6f}",
                    "" if r.best_known is None else f"{r.best_known:.6f}", r.best_known_by,
                    f"{r.gap:.6f}", "" if r.ours is None else f"{r.ours:.6f}", r.starter,
                    "" if r.public_s is None else f"{r.public_s:.6f}", f"{r.priority:.6f}"])
    return buf.getvalue()
