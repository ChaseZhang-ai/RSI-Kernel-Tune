"""One lesson record: the objective facts a finished run left on disk.

A record is built from a run directory and nothing else. Every value is read out of a file
the run itself wrote -- there is no inference, no summary, no field an agent filled in:

    <run>/run.json                              task, id, created/updated, stop
    <run>/task/task.toml                        language, arch, [reference], [leaderboard]
    <run>/operators/<op>/eval/leaderboard/problem.json   suite, definition, workloads, board
    <run>/operators/<op>/loop.toml              [loop] gate_window (which timing window gates)
    <run>/operators/<op>/student/status.jsonl   the iteration trajectory (inner/trajectory.py)
    <run>/operators/<op>/meta_status.jsonl      the meta-step ledger (meta_trajectory.py)
    <run>/operators/<op>/loop_summary.json      stop reason, the KB recipes the harvest recorded
    <run>/operators/<op>/best_confirmation.json the confirmed best sha and its readings
    <run>/operators/*/report.md, candidates.md  the reconnaissance deliverables (structured
                                                entries only: verdict lines and table rows)
    <run>/agent_usage.jsonl                     the usage ledger (backends; optional)

A field whose source is absent is **omitted**, never defaulted: "no reading" and "a reading
of zero" are different facts, and a later run's teacher must not be handed the second when
the first is true. Free text (``change``, ``error``, and the reconnaissance values) is copied
verbatim, truncated at a fixed length, and marked ``verbatim: true``.

What is deliberately NOT collected: the milestone (its body describes a lesson plan somebody
wrote), the recon report's *prose*, probe notes, SASS dumps and the other narrative
artifacts. Only the reconnaissance's fixed-format entries come across -- the candidate
architecture table's rows and the tools' ``*-OK`` verdict lines -- because those are
mechanical readings of the reference (the user allowed the reference's measured structure
into the lesson base; a narrative paragraph is not a reading).

::

    from forgekernel.opkb import record
    ext = record.extract_run(".artifacts/069_rms_norm_251124-101500-ab12")
    ext.record["family"], len(ext.record["kept"]), ext.notes
"""

from __future__ import annotations

import json
import re
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from forgekernel.family import family_key
from forgekernel.inner.trajectory import GATE_WINDOWS, gate_ms, load_records
from forgekernel.meta_trajectory import load_meta_records

try:  # the SOL arithmetic is campaign's one implementation; a checkout without it degrades
    from forgekernel.campaign import solscore as _solscore
except Exception:  # noqa: BLE001 -- a soft dependency: the SOL metric is then simply absent
    _solscore = None

SCHEMA = 1
SCHEMA_KEY = "schema"

PROBLEM_FILE = "eval/leaderboard/problem.json"
RUN_FILE = "run.json"
SNAPSHOT = "task"
TASK_FILE = "task.toml"
STATUS_FILE = "student/status.jsonl"
META_STATUS_FILE = "meta_status.jsonl"
SUMMARY_FILE = "loop_summary.json"
CONFIRMATION_FILE = "best_confirmation.json"
LOOP_FILE = "loop.toml"
RECON_REPORT = "report.md"
RECON_CANDIDATES = "candidates.md"
USAGE_FILE = "agent_usage.jsonl"

# The metric a record's outcome.value is: the board's SOL score when the card carries the
# board's T_b / T_SOL, else the same-job speedup over the reference.
METRIC_SOL = "sol"
METRIC_SPEEDUP = "speedup"
METRIC_KINDS = (METRIC_SOL, METRIC_SPEEDUP)

# What the reference was to this run. "eager" = the task's own readable reference (the
# definition's PyTorch run, or a source-mode run.py); "sealed" = a reference the run could
# not read (blackbox, a board solution, a sealed binary); "generated" = a task that shipped
# no reference and had one written (task.MODE_GENERATE); "none" = no reference at all.
REFERENCE_KINDS = ("eager", "sealed", "generated", "none")

# Length caps. A cap is part of the schema: lint enforces the same numbers, so a record
# written by harvest always passes lint.
MAX_CHANGE = 1000
MAX_ERROR = 1500
MAX_ENTRIES = 200            # kept / failed / meta rows
MAX_KB_ENTRIES = 100
MAX_RECON_VERDICTS = 24
MAX_RECON_CANDIDATES = 200
MAX_RECON_LINE = 400
MAX_RECON_CELL = 600
MAX_ID = 120

TRUNCATION_MARK = " [truncated]"


class RecordError(ValueError):
    """The directory is not a run whose record can be extracted; the message says why."""


@dataclass
class Extraction:
    """A record plus what could not be read (one note per missing/ignored source)."""
    record: dict
    notes: list[str] = field(default_factory=list)


# -- small mechanical helpers ------------------------------------------------------------------


def truncate(text, limit: int) -> str:
    """``text`` as a string, cut to ``limit`` characters with an explicit marker.

    The marker is the only thing added: what is kept is the source's own characters, which is
    what ``verbatim: true`` promises. ``None`` becomes ``""``."""
    s = "" if text is None else str(text)
    if len(s) <= limit:
        return s
    return s[: max(0, limit - len(TRUNCATION_MARK))] + TRUNCATION_MARK


def _as_float(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def parse_time(text) -> datetime | None:
    """An ISO 8601 timestamp as the run writes it (``2026-11-24T16:42:01+0800``)."""
    s = str(text or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def hours_between(start, end) -> float | None:
    """Hours from ``start`` to ``end`` (both ISO text), or None when either is unreadable."""
    a, b = parse_time(start), parse_time(end)
    if a is None or b is None:
        return None
    try:
        return round((b - a).total_seconds() / 3600.0, 4)
    except (OverflowError, TypeError):
        return None


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_toml(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _read_jsonl(path: Path) -> list[dict]:
    """Every JSON object of a jsonl file; unreadable lines are skipped (a ledger is not SSOT)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _slug(text) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text or "").strip()).strip("-.")


def record_id(label: str, run_id: str) -> str:
    """The record's file name stem: ``<problem>-<run id>`` (deterministic per run)."""
    base = _slug(label) or "run"
    suffix = _slug(run_id)
    rid = f"{base}-{suffix}" if suffix and suffix != base else base
    return truncate(rid, MAX_ID)


def _mtime_iso(path: Path) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(path.stat().st_mtime))
    except OSError:
        return ""


# -- run artifacts ----------------------------------------------------------------------------


def _gate_window(op_dir: Path, task: dict) -> str:
    """The timing window the run gated on: the card's ``[loop] gate_window`` (which taskcard
    writes from ``[target] timing``), else the task's own ``[target] timing``, else the loop's
    default (``device``)."""
    cfg = _read_toml(op_dir / LOOP_FILE)
    chosen = (cfg.get("loop") or {}).get("gate_window")
    if chosen not in GATE_WINDOWS:
        chosen = (task.get("target") or {}).get("timing")
    return chosen if chosen in GATE_WINDOWS else "device"


def _task_snapshot(run_dir: Path, notes: list[str]) -> dict:
    task = _read_toml(run_dir / SNAPSHOT / TASK_FILE)
    if not task:
        notes.append(f"no readable {SNAPSHOT}/{TASK_FILE}: language / arch / reference kind "
                     f"fall back to the card")
    return task


def _card(op_dir: Path, notes: list[str]) -> dict | None:
    card = _read_json(op_dir / PROBLEM_FILE)
    if not isinstance(card, dict):
        notes.append(f"no readable {PROBLEM_FILE}: suite / definition / workload count and the "
                     f"SOL board are taken from what else the run wrote")
        return None
    return card


def _definition(card: dict | None) -> str | None:
    if not card:
        return None
    defn = card.get("definition")
    if isinstance(defn, dict):
        name = defn.get("name")
        return str(name) if isinstance(name, str) and name else None
    if isinstance(defn, str) and defn:
        return defn
    return None


def _workload_count(card: dict | None) -> int | None:
    if not card:
        return None
    group = card.get("group")
    if isinstance(group, list) and group:
        return len(group)
    if isinstance(card.get("workload"), dict):
        return 1
    return None


def _track(card: dict | None) -> str | None:
    """The board track of the card's problem: from the board's kernel id, mechanically."""
    if not card or _solscore is None:
        return None
    blocks = [card.get("sol_board")]
    for member in card.get("group") or ():
        if isinstance(member, dict):
            blocks.append(member.get("sol_board"))
    for block in blocks:
        if isinstance(block, dict):
            kid = _as_int(block.get("kernel_id"))
            if kid is not None:
                try:
                    return _solscore.track_of(kid)
                except (TypeError, ValueError):
                    return None
    return None


def _reference_kind(task: dict, card: dict | None) -> str:
    """Which kind of reference the run was judged against (see ``REFERENCE_KINDS``)."""
    ref = task.get("reference")
    if isinstance(ref, dict):
        return "sealed" if ref.get("blackbox") else "eager"
    impl = (card or {}).get("reference_impl")
    kind = impl.get("kind") if isinstance(impl, dict) else None
    if kind == "definition":
        return "eager"
    if kind in ("binary", "sealed", "baseline_solution"):
        return "sealed"
    if isinstance(task.get("leaderboard"), dict):
        return "none"
    return "generated" if task else "none"


def _outcome_metric(op_dir: Path, card: dict | None, confirm: dict | None,
                    notes: list[str]) -> tuple[str | None, float | None]:
    """``(metric_kind, value)`` of the confirmed best: the board's SOL mean when the card
    carries the board's T_b / T_SOL and the confirmation has per-workload latencies on the
    device window, else the confirmed same-job speedup; ``(None, None)`` without either."""
    if not isinstance(confirm, dict):
        return None, None
    if card and _solscore is not None:
        try:
            board = _solscore.card_boards(op_dir)
        except (OSError, ValueError, KeyError, TypeError) as e:
            board = None
            notes.append(f"the card's SOL board is unreadable ({e})")
        if board:
            reading = None
            workloads = confirm.get("workloads")
            if isinstance(workloads, list) and workloads:
                reading = {"workloads": workloads}
            else:
                ms = _as_float(confirm.get("median_ms"))
                if ms is not None:
                    reading = {"wl0": ms}
            if reading is not None:
                try:
                    score = _solscore.score_kernel(reading, board, mean=True)
                except (TypeError, ValueError, KeyError):
                    score = None
                if score is not None:
                    return METRIC_SOL, float(score)
    ratio = _as_float(confirm.get("median_ratio"))
    if ratio is not None:
        return METRIC_SPEEDUP, ratio
    return None, None


def _kept(records, gate_window: str) -> list[dict]:
    out: list[dict] = []
    for r in records:
        if r.verdict != "KEEP":
            continue
        item: dict = {"iter": int(r.iter)}
        if r.change:
            item["change"] = truncate(r.change, MAX_CHANGE)
            item["verbatim"] = True
        before = _as_float(r.metric_before)
        after = _as_float(gate_ms(r, gate_window))
        if before is not None:
            item["before_ms"] = before
        if after is not None:
            item["after_ms"] = after
        out.append(item)
        if len(out) >= MAX_ENTRIES:
            break
    return out


def _failed(records) -> list[dict]:
    out: list[dict] = []
    for r in records:
        if r.verdict != "REVERT":
            continue
        item: dict = {"iter": int(r.iter)}
        if r.change:
            item["change"] = truncate(r.change, MAX_CHANGE)
        if r.error:
            item["error"] = truncate(r.error, MAX_ERROR)
        if "change" in item or "error" in item:
            item["verbatim"] = True
        item["kind"] = "correctness" if r.correctness is False else "revert"
        out.append(item)
        if len(out) >= MAX_ENTRIES:
            break
    return out


def _meta(meta_records) -> list[dict]:
    out: list[dict] = []
    for m in meta_records:
        out.append({"round": int(m.meta_step), "outcome": str(m.outcome)})
        if len(out) >= MAX_ENTRIES:
            break
    return out


def _certified_kb(summary: dict) -> list[str]:
    raw = summary.get("kb_harvest") if isinstance(summary, dict) else None
    if not isinstance(raw, list):
        return []
    return [str(p) for p in raw if isinstance(p, str) and p.strip()][:MAX_KB_ENTRIES]


# -- reconnaissance: structured entries only --------------------------------------------------

_VERDICT_LINE = re.compile(r"^([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*-OK)\b(.*)$")
_FIELD = re.compile(r"([A-Za-z_][A-Za-z0-9_.-]*)=(\S+)")
_TABLE_ROW = re.compile(r"^\|(.+)\|\s*$")
_VERIFY_TAG = re.compile(r"\[VERIFY:\s*([A-Za-z-]+)\s*\]", re.IGNORECASE)


def verdict_lines(text: str) -> list[dict]:
    """The reconnaissance tools' ``*-OK`` verdict lines, as ``{kind, line, fields}``.

    Every one of them is printed only after the tool's own self-checks pass (``SASS-OK``
    after the kernel match / instruction count / tensor family checks, ``SCHED-OK`` after the
    event transcription, and so on), so the line is a machine reading of the reference, not a
    claim. ``fields`` is the ``key=value`` pairs of the line; unparsed remainders stay in
    ``line`` verbatim."""
    out: list[dict] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        m = _VERDICT_LINE.match(line)
        if m is None:
            continue
        kind = m.group(1)
        fields = {k: truncate(v, 120) for k, v in _FIELD.findall(m.group(2))}
        item = {"kind": kind, "line": truncate(line, MAX_RECON_LINE), "fields": fields,
                "verbatim": True}
        key = item["line"]
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= MAX_RECON_VERDICTS:
            break
    return out


def candidate_rows(text: str) -> list[dict]:
    """The rows of the reconnaissance's candidate architecture table(s).

    The table is the recon's fixed-format deliverable: one row per structural decision with
    its value and its ``[VERIFY: ...]`` channel (``meta/recon_prompt.md`` §Deliverables 2).
    Only rows of a markdown table are read -- the report's prose is not; a row with no
    decision cell is skipped."""
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    header: list[str] | None = None
    for raw in text.splitlines():
        m = _TABLE_ROW.match(raw.strip())
        if m is None:
            header = None
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if header is None:
            header = cells
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue                                  # the |---|---| rule
        decision = cells[0].strip("*").strip() if cells else ""
        if not decision:
            continue
        value = cells[1] if len(cells) > 1 else ""
        verify = ""
        tag = _VERIFY_TAG.search(" ".join(cells))
        if tag is not None:
            verify = tag.group(1).lower()
        else:
            col = next((i for i, h in enumerate(header) if h.upper().startswith("VERIFY")), None)
            if col is not None and col < len(cells):
                bare = cells[col].strip("`").strip()
                verify = re.split(r"[\s（(`]", bare, maxsplit=1)[0].lower() if bare else ""
        key = (decision, value)
        if key in seen:
            continue
        seen.add(key)
        item: dict = {"decision": truncate(decision, MAX_RECON_CELL),
                      "value": truncate(value, MAX_RECON_CELL)}
        if verify:
            item["verify"] = truncate(verify, 40)
        item["verbatim"] = True
        out.append(item)
        if len(out) >= MAX_RECON_CANDIDATES:
            break
    return out


def _recon_block(run_dir: Path, task_name: str) -> dict | None:
    """The reconnaissance's structured entries, or None when the run wrote no recon artifacts.

    The recon tree is teacher-only and lives beside the operator card (``operators/_recon`` by
    default, ``operators/_<task>ref`` when the reference is sealed). Only ``report.md`` /
    ``candidates.md`` are opened -- the notes/ scripts are not."""
    ops = run_dir / "operators"
    if not ops.is_dir():
        return None
    trees = [d for d in sorted(ops.iterdir())
             if d.is_dir() and (d / RECON_REPORT).is_file()]
    if not trees:
        return None
    chosen = None
    for preferred in ("_recon", f"_{task_name}ref"):
        chosen = next((d for d in trees if d.name == preferred), None)
        if chosen is not None:
            break
    chosen = chosen or trees[0]
    report = chosen / RECON_REPORT
    try:
        report_text = report.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        candidates_text = (chosen / RECON_CANDIDATES).read_text(encoding="utf-8",
                                                                errors="replace")
    except OSError:
        candidates_text = ""
    verdicts = verdict_lines(report_text)
    candidates = candidate_rows(candidates_text)
    if not verdicts and not candidates:
        return None
    return {"verbatim": True, "verdicts": verdicts, "candidates": candidates}


# -- usage ledger (soft dependency on the backends task) ---------------------------------------

_USAGE_TOTAL_KEYS = ("tokens", "total_tokens", "token_count")
_USAGE_IN_KEYS = ("input_tokens", "prompt_tokens", "tokens_in")
_USAGE_OUT_KEYS = ("output_tokens", "completion_tokens", "tokens_out")
_USAGE_TURN_KEYS = ("turns", "turn", "calls", "requests")
_ROLES = ("teacher", "student")


def _first_number(obj: dict, keys) -> float | None:
    for key in keys:
        value = _as_float(obj.get(key))
        if value is not None:
            return value
    return None


def usage_totals(path: Path) -> dict | None:
    """Token / turn totals per role from the usage ledger, or None when nothing is recognized.

    The ledger is the backends task's ``<run>/agent_usage.jsonl``; its exact field names are
    that task's to define and this reader must not assume them. It accepts the shapes a
    per-turn ledger plausibly has (a ``role`` + ``tokens`` row; ``input_tokens`` /
    ``output_tokens``; flat ``teacher_tokens`` columns) and ignores every row it does not
    recognize. A ledger in an unrecognized format therefore leaves the record without a
    ``usage`` block -- never with invented numbers."""
    rows = _read_jsonl(path)
    if not rows:
        return None
    totals = {f"{role}_tokens": 0 for role in _ROLES}
    totals.update({f"{role}_turns": 0 for role in _ROLES})
    seen = False
    for obj in rows:
        role = str(obj.get("role") or obj.get("agent") or obj.get("actor") or "").strip().lower()
        if role in _ROLES:
            tokens = _first_number(obj, _USAGE_TOTAL_KEYS)
            if tokens is None:
                parts = [_first_number(obj, _USAGE_IN_KEYS), _first_number(obj, _USAGE_OUT_KEYS)]
                present = [p for p in parts if p is not None]
                tokens = sum(present) if present else None
            turns = _first_number(obj, _USAGE_TURN_KEYS)
            if tokens is not None:
                totals[f"{role}_tokens"] += int(round(tokens))
                seen = True
            if turns is not None:
                totals[f"{role}_turns"] += int(round(turns))
                seen = True
        else:
            for r in _ROLES:
                tokens = _first_number(obj, (f"{r}_tokens",))
                turns = _first_number(obj, (f"{r}_turns",))
                if tokens is not None:
                    totals[f"{r}_tokens"] += int(round(tokens))
                    seen = True
                if turns is not None:
                    totals[f"{r}_turns"] += int(round(turns))
                    seen = True
    return totals if seen else None


# -- the extraction ---------------------------------------------------------------------------


def _problem_block(task_name: str, card: dict | None, task: dict) -> dict:
    problem: dict = {}
    if card and isinstance(card.get("suite"), str):
        problem["suite"] = card["suite"]
    problem["name"] = task_name
    definition = _definition(card)
    if definition:
        problem["definition"] = definition
    track = _track(card)
    if track:
        problem["track"] = track
    section = task.get("task") if isinstance(task.get("task"), dict) else {}
    arch = section.get("arch")
    if isinstance(arch, str) and arch:
        problem["arch"] = arch
    language = section.get("language")
    if isinstance(language, str) and language:
        problem["language"] = language
    workloads = _workload_count(card)
    if workloads is not None:
        problem["workloads"] = workloads
    return problem


def extract_run(run_dir: str | Path) -> Extraction:
    """Read one run directory into a record (see the module docstring for every source).

    Raises ``RecordError`` when the directory is not a run at all (no ``run.json``, no task
    name, no operator card directory) or when a trajectory file it must read is unreadable.
    A *missing* source is not an error: the field is omitted and a note says which file was
    absent."""
    run_dir = Path(run_dir).expanduser()
    run = _read_json(run_dir / RUN_FILE)
    if not isinstance(run, dict):
        raise RecordError(f"{run_dir} is not a run directory (no readable {RUN_FILE})")
    task_name = str(run.get("task") or "").strip()
    if not task_name:
        raise RecordError(f"{run_dir / RUN_FILE} names no task")
    op_dir = run_dir / "operators" / task_name
    if not op_dir.is_dir():
        raise RecordError(f"{op_dir} does not exist — the run has no operator card")

    notes: list[str] = []
    task = _task_snapshot(run_dir, notes)
    card = _card(op_dir, notes)
    gate_window = _gate_window(op_dir, task)

    status = op_dir / STATUS_FILE
    records = []
    if status.is_file():
        try:
            records = load_records(status)
        except (OSError, ValueError, KeyError) as e:
            raise RecordError(f"{status}: {e}") from None
    else:
        notes.append(f"no {op_dir.name}/{STATUS_FILE}: no iteration rows")

    meta_records = []
    meta_path = op_dir / META_STATUS_FILE
    if meta_path.is_file():
        try:
            meta_records = load_meta_records(meta_path)
        except (OSError, ValueError, KeyError) as e:
            raise RecordError(f"{meta_path}: {e}") from None
    else:
        notes.append(f"no {op_dir.name}/{META_STATUS_FILE}: no meta-step rows")

    summary = _read_json(op_dir / SUMMARY_FILE)
    summary = summary if isinstance(summary, dict) else {}
    if not summary:
        notes.append(f"no readable {op_dir.name}/{SUMMARY_FILE}: stop reason / certified KB "
                     f"entries come from run.json only")
    confirm = _read_json(op_dir / CONFIRMATION_FILE)
    if not isinstance(confirm, dict):
        confirm = None
        notes.append(f"no readable {op_dir.name}/{CONFIRMATION_FILE}: no outcome metric and no "
                     f"best sha")

    definition = _definition(card)
    label = definition or task_name
    created = str(run.get("updated") or run.get("created") or "") or _mtime_iso(
        run_dir / RUN_FILE)

    outcome: dict = {"window": gate_window}
    metric_kind, metric_value = _outcome_metric(op_dir, card, confirm, notes)
    if metric_kind is not None:
        outcome["metric_kind"] = metric_kind
        outcome["value"] = metric_value
    stop = summary.get("stop") or run.get("stop")
    if isinstance(stop, str) and stop:
        outcome["stop"] = stop
    hours = hours_between(run.get("created"), run.get("updated"))
    if hours is not None:
        outcome["hours"] = hours
    if status.is_file():
        outcome["student_iterations"] = len(records)

    record: dict = {
        SCHEMA_KEY: SCHEMA,
        "id": record_id(label, str(run.get("id") or run_dir.name)),
        "created": created,
        "family": family_key(label),
        "problem": _problem_block(task_name, card, task),
        "reference": {"kind": _reference_kind(task, card)},
        "outcome": outcome,
    }
    if status.is_file():
        record["kept"] = _kept(records, gate_window)
        record["failed"] = _failed(records)
    if meta_path.is_file():
        record["meta"] = _meta(meta_records)
    kb = _certified_kb(summary)
    if kb:
        record["certified_kb"] = kb
    usage = usage_totals(run_dir / USAGE_FILE)
    if usage is not None:
        record["usage"] = usage
    recon = _recon_block(run_dir, task_name)
    if recon is not None:
        record["recon"] = recon
    source: dict = {"run": run_dir.name}
    if isinstance(confirm, dict) and isinstance(confirm.get("sha"), str) and confirm["sha"]:
        source["best_sha"] = confirm["sha"]
    record["source"] = source
    return Extraction(record=record, notes=notes)


def serialize(record: dict) -> str:
    """The record's on-disk form: one JSON object, one space of indent, trailing newline."""
    return json.dumps(record, indent=1, ensure_ascii=False) + "\n"
