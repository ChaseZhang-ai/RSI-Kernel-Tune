"""Pick the records a problem's teacher should read: same family, newest and best first.

A problem is identified by its name (the board's problem name) or by its task directory
(``tasks/<op>``); the second also yields the board definition name and the suite from the
frozen card, so a record matches when **either** identity agrees with it. The record's
identity keys are ``problem.name`` (the task/operator name) and ``problem.definition`` (the
board's definition name) -- a run of ``tasks/sol-069_rms_norm`` and a run of
``tasks/069_rms_norm`` are the same problem when both resolve to definition
``069_rms_norm``.

Selection: records of the **same operator family** only (``rsi_kernel_tune.family`` -- a
different family's numbers describe a different kernel), then the problem's own earlier runs
are included by default (the user dropped "distill from zero"; the render marks them
``same problem``) and excluded with ``exclude_self=True``. Order: ``outcome.value``
descending, then ``created`` descending, then id (so a tie is deterministic).

::

    from rsi_kernel_tune.opkb import query
    target = query.resolve_target("tasks/sol-069_rms_norm")
    rows = query.select(store.load_records(root), target, k=5)
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from rsi_kernel_tune.family import family_key
from rsi_kernel_tune.opkb.record import parse_time

DEFINITION_FILE = "eval/leaderboard/problem.json"


class QueryError(ValueError):
    """The argument does not name a problem (not a task directory, or empty)."""


@dataclass(frozen=True)
class Target:
    """The problem a query is about, and the identity keys a record is matched against."""
    spec: str
    family: str
    keys: tuple[str, ...] = ()     # normalized identity strings: task name and/or definition
    name: str | None = None        # the task/operator name when known
    definition: str | None = None  # the board's definition name when known
    suite: str | None = None

    def same_problem(self, record: dict) -> bool:
        """Whether ``record`` is an earlier run of THIS problem (not merely its family)."""
        problem = record.get("problem") if isinstance(record.get("problem"), dict) else {}
        for key in ("name", "definition"):
            value = problem.get(key)
            if isinstance(value, str) and _norm(value) in self.keys:
                return True
        return False


def _norm(text) -> str:
    return str(text or "").strip().lower().rstrip("/").rsplit("/", 1)[-1]


def _card_identity(problem_file: Path) -> tuple[str | None, str | None]:
    """``(definition name, suite)`` from a card's ``problem.json``, or ``(None, None)``."""
    try:
        problem = json.loads(problem_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    if not isinstance(problem, dict):
        return None, None
    definition = suite = None
    defn = problem.get("definition")
    if isinstance(defn, dict) and isinstance(defn.get("name"), str):
        definition = defn["name"]
    elif isinstance(defn, str) and defn:
        definition = defn
    if isinstance(problem.get("suite"), str):
        suite = problem["suite"]
    return definition, suite


def _task_identity(task_dir: Path) -> tuple[str, str | None, str | None]:
    """``(task name, definition name, suite)`` of a task directory."""
    name = task_dir.name
    try:
        cfg = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        cfg = {}
    board = cfg.get("leaderboard") if isinstance(cfg.get("leaderboard"), dict) else {}
    card = board.get("card")
    definition = suite = None
    if isinstance(card, str) and card:
        problem_file = Path(card) / DEFINITION_FILE
        if not problem_file.is_absolute():
            problem_file = task_dir / problem_file
        definition, suite = _card_identity(problem_file)
    return name, definition, suite


def _run_identity(run_dir: Path) -> tuple[str, str | None, str | None] | None:
    """``(task name, definition name, suite)`` of a run directory, or None when it is not one."""
    try:
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    task = str(run.get("task") or "").strip() if isinstance(run, dict) else ""
    if not task:
        return None
    definition, suite = _card_identity(run_dir / "operators" / task / DEFINITION_FILE)
    return task, definition, suite


def resolve_target(spec: str) -> Target:
    """The problem ``spec`` names: a task directory (``tasks/<op>``), a run directory
    (``.artifacts/<op>_<id>``, read through its ``run.json``), or a problem name."""
    text = str(spec or "").strip()
    if not text:
        raise QueryError("give a problem name, a task directory (tasks/<op>) or a run "
                         "directory (.artifacts/<op>_<id>)")
    candidate = Path(text).expanduser()
    looks_like_path = "/" in text or "\\" in text or candidate.exists()
    identity = None
    if candidate.is_dir():
        if (candidate / "task.toml").is_file():
            identity = _task_identity(candidate)
        elif (candidate / "run.json").is_file():
            identity = _run_identity(candidate)
    if identity is not None:
        name, definition, suite = identity
        label = definition or name
        keys = {_norm(name)}
        if definition:
            keys.add(_norm(definition))
        return Target(spec=text, family=family_key(label), keys=tuple(sorted(keys)),
                      name=name, definition=definition, suite=suite)
    if looks_like_path:
        raise QueryError(f"{text}: neither a task directory (no task.toml) nor a run "
                         f"directory (no run.json)")
    keys = {_norm(text)}
    return Target(spec=text, family=family_key(text), keys=tuple(sorted(keys)), name=text)


def _value(record: dict) -> float | None:
    outcome = record.get("outcome") if isinstance(record.get("outcome"), dict) else {}
    v = outcome.get("value")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _created_stamp(record: dict) -> float:
    dt = parse_time(record.get("created"))
    if dt is None:
        return 0.0
    try:
        return dt.timestamp()
    except (OverflowError, OSError, ValueError):
        return 0.0


def order(records: list[dict]) -> list[dict]:
    """Best first, then newest, then id -- a total order, so rendering is byte-stable."""
    rows = sorted(records, key=lambda r: str(r.get("id") or ""))
    rows.sort(key=lambda r: (1 if _value(r) is not None else 0, _value(r) or 0.0,
                             _created_stamp(r)), reverse=True)
    return rows


def select(records: list[dict], target: Target, *, k: int = 5,
           exclude_self: bool = False) -> list[dict]:
    """Up to ``k`` records of ``target``'s family, ordered (see ``order``).

    The family filter is exact: a record of another family is never returned, so a problem
    whose family has no records yet honestly gets an empty list. ``exclude_self`` drops the
    target problem's own earlier runs."""
    if k <= 0:
        return []
    rows = [r for r in records
            if isinstance(r, dict) and r.get("family") == target.family]
    if exclude_self:
        rows = [r for r in rows if not target.same_problem(r)]
    return order(rows)[:k]


def describe(record: dict) -> str:
    """One line: id, family, problem, metric, stop."""
    problem = record.get("problem") if isinstance(record.get("problem"), dict) else {}
    outcome = record.get("outcome") if isinstance(record.get("outcome"), dict) else {}
    bits = [str(record.get("id") or "?"), f"[{record.get('family') or '?'}]",
            str(problem.get("definition") or problem.get("name") or "?")]
    metric = outcome.get("metric_kind")
    value = outcome.get("value")
    if metric and value is not None:
        bits.append(f"{metric}={value:g}")
    if outcome.get("stop"):
        bits.append(f"stop={outcome['stop']}")
    bits.append(str(record.get("created") or "?"))
    return "  ".join(bits)
