"""The throughput governor: a ``DriverBackend`` wrapper that protects student iterations.

Motivation (issue #8, reproduced in the task file): a 10.6-hour run spent 92.5 % of its
wall clock in teacher turns and the student landed only 7 effective iterations; three meta
steps burned on "resource-budget addendum / performance attribution / measurement
verification", and the run finished at 0.547x (decode 0.289x). Tokens were not the
problem — iterations were. So this module does two things, and only these two:

1. **Iteration throughput.** While the student has not yet completed
   ``min_student_iters`` iterations, the teacher's cumulative wall clock may not exceed
   ``hours * teacher_share`` without a warning; past 1.25x that share, the teacher turn is
   refused. Independently, a phase may not run more than
   ``max_teacher_turns_per_phase`` consecutive teacher turns (no student turn in between)
   without a warning; past +4 more, the teacher turn is refused. Student turns are never
   braked or refused.
2. **Architecture escape.** When the best reading is still more than 2x the target after
   the M0 gate and the last two iterations bought no improvement beyond the noise floor,
   the next ``diagnose`` / ``patch`` turn is prefixed with a fixed text asking the teacher
   to prefer a structural patch of the earliest milestone (R1) over more tuning. Once per
   meta round; advice only.

There are **no token caps** (the user explicitly said tokens are not the constraint).

Soft brake = a fixed block prepended to that turn's prompt. Hard stop =
``GovernorStop`` (an ``AgentError`` subclass). How a hard stop lands: ``loop/cli.main``
catches ``AgentError`` from the meta layer, logs "stopping cleanly", keeps the run's best
(confirmation still runs), writes the summary, and ``launch.drive`` returns 2 — the run
ends gracefully. The summary's ``stop`` reads ``agent_failed`` (the cli's existing label;
changing it would mean editing the cli's meta-layer catch, which this task does not own).
See docs/throughput.md.

The governor is wired at the two ``RealBackend`` construction sites in ``loop/cli.py``;
``FK_GOVERNOR=off`` disables it (the inner backend is returned unwrapped). The policy
travels from ``launch.run_loop`` as JSON in ``FK_BUDGET_POLICY``.

Ledger: ``<run operator tree>/student/governor_ledger.jsonl``, one line per turn plus
event lines (``policy``, ``soft_brake``, ``hard_stop``, ``escape_fired``,
``over_share``). ``forge status`` and ``report.md`` read it.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from forgekernel.loop.driver import AGENT_ROLE_STUDENT, AGENT_ROLE_TEACHER, AgentError

POLICY_ENV = "FK_BUDGET_POLICY"
DISABLE_ENV = "FK_GOVERNOR"
LEDGER_NAME = "governor_ledger.jsonl"
USAGE_NAME = "agent_usage.jsonl"

GATE_WINDOWS = ("device", "e2e")

# The soft brake, verbatim (task 14 section 4.4). Prepended, never replaces the prompt.
SOFT_BRAKE_TEXT = (
    "[THROUGHPUT] This phase has used most of the time the run can spend on it without the kernel\n"
    "getting any further. Finish now: write the files / verdict this step requires, list what remains\n"
    "unresolved in one short block, and send no further requests. An unresolved question is an\n"
    "acceptable outcome; an unbounded investigation is not.\n"
)

# The escape rule (task 14 section 4.5), with N and X filled in.
ESCAPE_TEXT_TEMPLATE = (
    "[ESCAPE RULE] After {n} student iterations the best reading is still {x} times the target, and "
    "the last\n"
    "two iterations did not improve it beyond the noise floor. A gap this large is rarely closed by\n"
    "tuning inside the current structure. Treat this as R1 (a foundation ceiling) unless the measured\n"
    "evidence in front of you shows the current structure is within a few percent of its own ceiling:\n"
    "prefer a structural patch of the earliest milestone over more parameter tuning.\n"
)

PHASES = ("judge", "recon", "distill", "m0_audit", "diagnose", "patch", "kb", "student",
          "other")

# First-line title -> phase. Every prompt family has an entry here (tests check each
# meta/*.md and the language subdirectories against it). The judge prompt has no meta
# file; it is the opening line of forgekernel.judge.judge_prompt.
_PHASE_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("judge", re.compile(r"^You are writing the correctness judge\b")),
    ("recon", re.compile(r"^# Recon\b")),
    ("distill", re.compile(r"^# Meta-Agent — Initial distillation authoring step\b")),
    ("m0_audit", re.compile(r"^# Meta-Agent — Mandatory M0 finish audit\b")),
    ("diagnose", re.compile(r"^# Meta-Agent — Diagnosis step\b")),
    ("patch", re.compile(r"^# Meta-Agent — Patch step\b")),
    ("kb", re.compile(r"^# Meta-Agent \(teacher\) — "
                      r"(?:Finish-time KB harvest|Write one certified)\b")),
    ("student", re.compile(r"^# Student — One iteration\b")),
)

# The [budget] keys this module owns. No token key exists, deliberately.
POLICY_KEYS = ("hours", "teacher_share", "min_student_iters", "max_teacher_turns_per_phase")

# The escape rule's fixed shape (task 14 section 4.5). The thresholds are not config:
# they are the rule as specified, and the only knob is whether the run has a target.
ESCAPE_MIN_ITERS = 3
ESCAPE_TARGET_FACTOR = 2.0
ESCAPE_NOISE_FLOOR = 0.015
ESCAPE_RECENT_ITERS = 2

M0_GATE_ID = "M0-gate"


class GovernorStop(AgentError):
    """A teacher turn was refused by the throughput governor.

    ``AgentError`` on purpose: the loop CLI's meta-layer catch (``loop/cli.py``) already
    stops such a run cleanly — the confirmed best is kept, the report is written and
    ``forge run`` exits 2. Nothing in driver.py is changed to make this land.
    """

    def __init__(self, message: str, *, rules: Iterable[str] = (), detail: Mapping | None = None):
        super().__init__(message)
        self.rules = tuple(rules)
        self.detail = dict(detail or {})


# -- phases ---------------------------------------------------------------------------------

def first_line(prompt: str) -> str:
    """The first non-empty line of a rendered prompt (its title)."""
    for line in prompt.splitlines():
        if line.strip():
            return line.strip()
    return ""


def detect_phase(prompt: str) -> str:
    """The phase a prompt belongs to, from its first line; ``other`` when unknown."""
    title = first_line(prompt)
    if not title:
        return "other"
    for phase, pattern in _PHASE_PATTERNS:
        if pattern.match(title):
            return phase
    return "other"


def resolve_role(role: str | None, phase: str) -> str:
    """The turn's role: explicit when the driver says so (``_TeacherView`` passes
    ``role="teacher"``), otherwise inferred from the phase.

    An unrecognised title (``other``) resolves to **student**: the hard guarantee is
    that a student turn is never stopped, so a turn this module cannot identify is left
    alone rather than refused. Every teacher turn the driver issues carries the explicit
    role, so this costs the rules nothing.
    """
    if role in (AGENT_ROLE_STUDENT, AGENT_ROLE_TEACHER):
        return role
    return AGENT_ROLE_TEACHER if phase not in ("student", "other") else AGENT_ROLE_STUDENT


# -- the policy -----------------------------------------------------------------------------

def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass(frozen=True)
class GovernorPolicy:
    """The [budget] knobs plus the run facts the escape rule needs."""

    hours: float
    teacher_share: float = 0.5
    min_student_iters: int = 5
    max_teacher_turns_per_phase: int = 12
    # Run facts, filled by loop/cli.py (never from the environment).
    target_ms: float | None = None
    gate_window: str = "device"
    status_path: Path | None = None

    @property
    def share_s(self) -> float:
        """Wall clock (seconds) the teacher may spend before the soft brake."""
        return float(self.hours) * 3600.0 * float(self.teacher_share)

    @property
    def hard_share_s(self) -> float:
        """The share's 1.25x: past this a teacher turn is refused."""
        return self.share_s * 1.25

    @property
    def hard_turns(self) -> int:
        """max_teacher_turns_per_phase + 4: past this a teacher turn is refused."""
        return self.max_teacher_turns_per_phase + 4

    def as_dict(self) -> dict:
        return {"hours": self.hours,
                "teacher_share": self.teacher_share,
                "min_student_iters": self.min_student_iters,
                "max_teacher_turns_per_phase": self.max_teacher_turns_per_phase}


def validate_policy_values(data: Mapping[str, Any]) -> dict:
    """Check the policy numbers; returns the coerced values or raises ``ValueError``."""
    out: dict = {}
    hours = data.get("hours")
    if not _is_number(hours) or float(hours) <= 0:
        raise ValueError(f"hours must be a positive number, got {hours!r}")
    out["hours"] = float(hours)
    share = data.get("teacher_share", 0.5)
    if not _is_number(share) or not (0.0 < float(share) <= 1.0):
        raise ValueError(f"teacher_share must be a number in (0, 1], got {share!r}")
    out["teacher_share"] = float(share)
    min_iters = data.get("min_student_iters", 5)
    if not _is_int(min_iters) or min_iters < 0:
        raise ValueError(f"min_student_iters must be a non-negative integer, "
                         f"got {min_iters!r}")
    out["min_student_iters"] = int(min_iters)
    max_turns = data.get("max_teacher_turns_per_phase", 12)
    if not _is_int(max_turns) or max_turns < 1:
        raise ValueError(f"max_teacher_turns_per_phase must be a positive integer, "
                         f"got {max_turns!r}")
    out["max_teacher_turns_per_phase"] = int(max_turns)
    return out


def governor_disabled(environ: Mapping[str, str] | None = None) -> bool:
    """``FK_GOVERNOR=off`` (also 0/false/no/disabled, case-insensitively) turns it off."""
    env = os.environ if environ is None else environ
    return str(env.get(DISABLE_ENV, "")).strip().lower() in ("off", "0", "false", "no",
                                                              "disabled")


def policy_from_env(*, hours: float | None = None, target_ms: float | None = None,
                    gate_window: str = "device", status_path: str | Path | None = None,
                    environ: Mapping[str, str] | None = None) -> GovernorPolicy | None:
    """The run's policy: ``FK_BUDGET_POLICY`` (JSON) over the caller's defaults.

    Returns ``None`` when ``FK_GOVERNOR`` disables the governor — the caller then leaves
    the backend unwrapped. A malformed payload, an unknown key (including any token
    key) or an out-of-range value raises ``ValueError``: a run must not start under a
    policy nobody can read.
    """
    env = os.environ if environ is None else environ
    if governor_disabled(env):
        return None
    base: dict[str, Any] = {}
    if hours is not None:
        base["hours"] = hours
    raw = str(env.get(POLICY_ENV, "") or "").strip()
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"{POLICY_ENV} is not valid JSON: {e}") from None
        if not isinstance(payload, dict):
            raise ValueError(f"{POLICY_ENV} must be a JSON object, got "
                             f"{type(payload).__name__}")
        unknown = sorted(set(payload) - set(POLICY_KEYS))
        if unknown:
            raise ValueError(f"{POLICY_ENV}: unknown key(s) {', '.join(unknown)} "
                             f"(known: {', '.join(POLICY_KEYS)})")
        base.update(payload)
    if "hours" not in base:
        raise ValueError(f"the throughput policy has no hours (give hours= or set "
                         f"{POLICY_ENV})")
    values = validate_policy_values(base)
    if gate_window not in GATE_WINDOWS:
        raise ValueError(f"gate_window must be one of {GATE_WINDOWS}, got {gate_window!r}")
    if target_ms is not None and (not _is_number(target_ms) or float(target_ms) <= 0):
        raise ValueError(f"target_ms must be a positive number or None, got {target_ms!r}")
    return GovernorPolicy(**values,
                          target_ms=None if target_ms is None else float(target_ms),
                          gate_window=gate_window,
                          status_path=Path(status_path) if status_path is not None else None)


# -- the ledger -----------------------------------------------------------------------------

def ledger_path(status_path: str | Path) -> Path:
    """``governor_ledger.jsonl`` next to the student's ``status.jsonl``."""
    return Path(status_path).parent / LEDGER_NAME


def _read_rows(path: str | Path | None) -> list[dict]:
    """Every JSON object in a jsonl file; unreadable files and bad lines are skipped."""
    if path is None:
        return []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def student_iterations(status_path: str | Path | None) -> int:
    """Completed student iterations = the rows on ``student/status.jsonl``.

    That is the record the rest of the harness calls the iteration count (``forge
    status`` prints the same number); rows the driver archives without a turn (a gate
    the delivery closed) are iterations too, and count.
    """
    return len(_read_rows(status_path))


def _reading(row: Mapping, gate_window: str) -> float | None:
    key = "device_median_ms" if gate_window == "device" else "bwd_median_ms"
    v = row.get(key)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
        return float(v)
    return None


def _is_keep(row: Mapping) -> bool:
    return row.get("verdict") == "KEEP" and row.get("correctness") is not False


@dataclass(frozen=True)
class EscapeState:
    """Whether the architecture-escape rule fires, and with which numbers."""

    fire: bool
    n: int = 0
    x: float = 0.0
    best_ms: float | None = None
    target_ms: float | None = None
    reason: str = ""


def escape_state(rows: list[dict], *, target_ms: float | None, gate_window: str = "device",
                 min_iters: int = ESCAPE_MIN_ITERS, factor: float = ESCAPE_TARGET_FACTOR,
                 noise_floor: float = ESCAPE_NOISE_FLOOR,
                 recent: int = ESCAPE_RECENT_ITERS) -> EscapeState:
    """The three conditions of the escape rule, read off ``status.jsonl`` rows.

    Fires only when all three hold:
    1. at least ``min_iters`` student iterations since the last M0-gate KEEP;
    2. the best reading in the gate window is still more than ``factor`` x the target;
    3. neither of the last ``recent`` evaluated iterations was a KEEP that improved the
       running best by at least ``noise_floor`` (1.5 %).
    """
    if target_ms is None or not _is_number(target_ms) or float(target_ms) <= 0:
        return EscapeState(False, reason="no target")
    target = float(target_ms)
    gate = -1
    for i, row in enumerate(rows):
        if row.get("decision_id") == M0_GATE_ID and _is_keep(row):
            gate = i
    if gate < 0:
        return EscapeState(False, reason="no M0-gate KEEP")
    after = rows[gate + 1:]
    n = len(after)
    if n < min_iters:
        return EscapeState(False, n=n, reason=f"{n} iteration(s) since M0 (< {min_iters})")
    readings = [v for v in (_reading(r, gate_window) for r in rows) if v is not None]
    if not readings:
        return EscapeState(False, n=n, reason="no reading in the gate window")
    best = min(readings)
    if best <= factor * target:
        return EscapeState(False, n=n, best_ms=best, target_ms=target,
                           reason=f"best {best:g} ms is within {factor:g}x the target")
    evaluated = [r for r in after if _reading(r, gate_window) is not None]
    if len(evaluated) <= recent:
        return EscapeState(False, n=n, best_ms=best, target_ms=target,
                           reason=f"only {len(evaluated)} evaluated iteration(s) since M0")
    before = evaluated[:-recent]
    prior_keeps = [_reading(r, gate_window) for r in before if _is_keep(r)
                   and _reading(r, gate_window) is not None]
    prior_best = min(prior_keeps) if prior_keeps else min(
        _reading(r, gate_window) for r in before)
    for row in evaluated[-recent:]:
        v = _reading(row, gate_window)
        if (_is_keep(row) and v is not None
                and v < prior_best * (1.0 - noise_floor)):
            return EscapeState(False, n=n, best_ms=best, target_ms=target,
                               reason="a recent KEEP improved beyond the noise floor")
    return EscapeState(True, n=n, x=best / target, best_ms=best, target_ms=target)


def _append_line(path: Path, row: dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return True
    except OSError:
        return False


def _takes_role(backend) -> bool:
    try:
        return "role" in inspect.signature(backend.run_agent).parameters
    except (TypeError, ValueError):
        return False


class GovernedBackend:
    """A ``DriverBackend`` that applies the throughput policy around every agent turn.

    Everything except ``run_agent`` is delegated to the wrapped backend (attributes are
    forwarded both on read and on write, so ``backend.kernel_file = ...`` after
    construction still lands on the inner object).
    """

    def __init__(self, inner, policy: GovernorPolicy, ledger: str | Path, *,
                 clock: Callable[[], float] = time.monotonic):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_policy", policy)
        object.__setattr__(self, "_ledger", Path(ledger))
        object.__setattr__(self, "_clock", clock)
        object.__setattr__(self, "_streak_phase", "")
        object.__setattr__(self, "_streak", 0)
        object.__setattr__(self, "_escape_fired", False)
        object.__setattr__(self, "_over_share_recorded", False)
        object.__setattr__(self, "_warned", False)
        object.__setattr__(self, "_spent_s", 0.0)
        object.__setattr__(self, "_accepts_role", _takes_role(inner))
        self._append({"event": "policy", "ts": round(self._clock(), 3),
                      **policy.as_dict(),
                      "target_ms": policy.target_ms,
                      "gate_window": policy.gate_window})

    # -- transparency ---------------------------------------------------------------------

    def __getattr__(self, name):
        try:
            inner = object.__getattribute__(self, "_inner")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(inner, name)

    def __setattr__(self, name, value):
        # This task's internals all start with "_"; public attributes belong to the
        # backend (job_env_files, kernel_file, ...) and must reach the real object.
        if name.startswith("_") or "_inner" not in self.__dict__:
            object.__setattr__(self, name, value)
        else:
            setattr(self.__dict__["_inner"], name, value)

    # -- the governed turn ----------------------------------------------------------------

    def run_agent(self, prompt: str, *, role: str | None = None) -> str:
        phase = detect_phase(prompt)
        role = resolve_role(role, phase)
        iters = student_iterations(self._policy.status_path)
        if role == AGENT_ROLE_STUDENT:
            # A student turn is the boundary between meta rounds: the consecutive-teacher
            # counter resets, and the escape rule may fire again in the next meta round.
            object.__setattr__(self, "_streak_phase", "")
            object.__setattr__(self, "_streak", 0)
            object.__setattr__(self, "_escape_fired", False)
            return self._run_turn(prompt, role, phase, iters, brakes=(), escape=None,
                                  count=0)

        count = self._streak + 1 if phase == self._streak_phase else 1
        brakes, stops, used_s = self._precheck(phase, iters, count)
        if stops:
            detail = "; ".join(self._stop_text(rule, kind, used, limit)
                               for rule, kind, used, limit in stops)
            message = (f"throughput governor: {detail} — refusing a teacher turn "
                       f"(phase {phase!r}, student iterations {iters}); the run stops "
                       f"cleanly on its confirmed best")
            if (any(s[0] == "teacher_share" for s in stops)
                    and not self._over_share_recorded):
                object.__setattr__(self, "_over_share_recorded", True)
                self._append({"event": "over_share", "ts": round(self._clock(), 3),
                              "teacher_s": round(used_s, 3),
                              "share_s": round(self._policy.share_s, 3),
                              "student_iters": iters})
            self._append({"event": "hard_stop", "ts": round(self._clock(), 3),
                          "phase": phase, "role": role,
                          "student_iters": iters,
                          "rules": [s[0] for s in stops],
                          "detail": detail})
            raise GovernorStop(message, rules=[s[0] for s in stops],
                               detail={"phase": phase, "student_iters": iters})
        escape = self._escape_for(phase)
        for rule in brakes:
            self._append({"event": "soft_brake", "ts": round(self._clock(), 3),
                          "rule": rule, "phase": phase, "role": role,
                          "student_iters": iters,
                          "teacher_s": round(used_s, 3),
                          "share_s": round(self._policy.share_s, 3),
                          "turns_in_phase": count,
                          "limit": self._policy.max_teacher_turns_per_phase})
        object.__setattr__(self, "_streak_phase", phase)
        object.__setattr__(self, "_streak", count)
        return self._run_turn(prompt, role, phase, iters, brakes=brakes, escape=escape,
                              count=count, teacher_before_s=used_s)

    # -- prechecks ------------------------------------------------------------------------

    def _teacher_seconds(self) -> float:
        """Completed teacher wall clock on this run's ledger (both backends, resumes).

        Falls back to this instance's in-memory total when the ledger cannot be read,
        so an unwritable ledger degrades the accounting but never the brakes.
        """
        total = 0.0
        try:
            text = self._ledger.read_text(encoding="utf-8")
        except OSError:
            return self._spent_s
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if (isinstance(row, dict) and row.get("event") == "turn"
                    and row.get("role") == AGENT_ROLE_TEACHER):
                v = row.get("elapsed_s")
                if _is_number(v):
                    total += float(v)
        return total

    def _precheck(self, phase: str, iters: int, count: int):
        """(brakes, stops, teacher_seconds) for the turn about to run."""
        p = self._policy
        brakes: list[str] = []
        stops: list[tuple[str, str, float, float]] = []
        used_s = self._teacher_seconds()
        if iters < p.min_student_iters:
            if used_s >= p.hard_share_s:
                stops.append(("teacher_share", "share", used_s, p.hard_share_s))
            elif used_s >= p.share_s:
                brakes.append("teacher_share")
        if count >= p.hard_turns:
            stops.append(("consecutive_phase_turns", "turns", float(count),
                          float(p.hard_turns)))
        elif count >= p.max_teacher_turns_per_phase:
            brakes.append("consecutive_phase_turns")
        return brakes, stops, used_s

    def _stop_text(self, rule: str, kind: str, used: float, limit: float) -> str:
        p = self._policy
        if kind == "share":
            return (f"teacher wall clock {used:.0f} s >= 1.25 x share {limit:.0f} s "
                    f"({p.hours:g} h x {p.teacher_share:g})")
        return (f"{int(used)} teacher turn(s) in one phase >= {int(limit)} "
                f"(max_teacher_turns_per_phase {p.max_teacher_turns_per_phase} + 4)")

    def _escape_for(self, phase: str) -> EscapeState | None:
        """The escape state for this teacher turn, or None when it cannot apply."""
        if phase not in ("diagnose", "patch") or self._escape_fired:
            return None
        if self._policy.target_ms is None or self._policy.status_path is None:
            return None
        state = escape_state(_read_rows(self._policy.status_path),
                             target_ms=self._policy.target_ms,
                             gate_window=self._policy.gate_window)
        if not state.fire:
            return None
        object.__setattr__(self, "_escape_fired", True)
        self._append({"event": "escape_fired", "ts": round(self._clock(), 3),
                      "phase": phase, "iters": state.n, "best_ms": state.best_ms,
                      "target_ms": state.target_ms, "x": round(state.x, 4)})
        return state

    # -- running a turn -------------------------------------------------------------------

    def _run_turn(self, prompt: str, role: str, phase: str, iters: int, *,
                  brakes: Iterable[str], escape: EscapeState | None, count: int,
                  teacher_before_s: float | None = None) -> str:
        text = prompt
        if brakes:
            text = SOFT_BRAKE_TEXT + "\n" + text
        if escape is not None:
            # the structural advice reads first, then "finish now"
            text = ESCAPE_TEXT_TEMPLATE.format(n=escape.n, x=f"{escape.x:.2f}".rstrip("0")
                                               .rstrip(".")) + "\n" + text
        started = self._clock()
        error = ""
        try:
            return self._call(text, role)
        except BaseException as e:  # noqa: BLE001 — recorded, then re-raised untouched
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            elapsed = max(0.0, float(self._clock() - started))
            self._append({"event": "turn", "ts": round(self._clock(), 3),
                          "role": role, "phase": phase,
                          "elapsed_s": round(elapsed, 3),
                          "student_iters": iters,
                          "brakes": list(brakes),
                          "escape": escape is not None,
                          "turns_in_phase": count,
                          "error": error})
            if role == AGENT_ROLE_TEACHER:
                object.__setattr__(self, "_spent_s", self._spent_s + elapsed)
                used_after = (teacher_before_s if teacher_before_s is not None else 0.0) \
                    + elapsed
                if (iters < self._policy.min_student_iters
                        and used_after >= self._policy.share_s
                        and not self._over_share_recorded):
                    object.__setattr__(self, "_over_share_recorded", True)
                    self._append({"event": "over_share", "ts": round(self._clock(), 3),
                                  "teacher_s": round(used_after, 3),
                                  "share_s": round(self._policy.share_s, 3),
                                  "student_iters": iters})

    def _call(self, prompt: str, role: str) -> str:
        if self._accepts_role:
            return self._inner.run_agent(prompt, role=role)
        return self._inner.run_agent(prompt)

    def _append(self, row: dict) -> None:
        if _append_line(self._ledger, row):
            return
        if not self._warned:
            object.__setattr__(self, "_warned", True)
            print(f"[governor] warning: cannot write {self._ledger} — throughput "
                  f"bookkeeping is in memory only", file=sys.stderr, flush=True)


def govern(backend, policy: GovernorPolicy | None, ledger: str | Path, *,
           clock: Callable[[], float] = time.monotonic) -> Any:
    """``backend`` wrapped in the governor, unless the policy is None (FK_GOVERNOR=off)."""
    if policy is None:
        return backend
    if isinstance(backend, GovernedBackend):
        return backend
    return GovernedBackend(backend, policy, ledger, clock=clock)


# -- reading the ledger (status / report) ---------------------------------------------------

@dataclass
class LedgerSummary:
    """What a run's governor ledger says, tolerant of any partial or old file."""

    policy: dict = field(default_factory=dict)
    turns: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    by_role: dict = field(default_factory=dict)    # role -> {"turns", "seconds"}
    by_phase: dict = field(default_factory=dict)   # phase -> {"turns", "seconds"}


def _bump(table: dict, key: str, seconds: float) -> None:
    cell = table.setdefault(key, {"turns": 0, "seconds": 0.0})
    cell["turns"] += 1
    cell["seconds"] += seconds


def read_ledger(path: str | Path | None) -> LedgerSummary:
    """Parse the governor ledger; never raises (a missing file is an empty summary)."""
    summary = LedgerSummary()
    for row in _read_rows(path):
        event = row.get("event")
        if event == "policy":
            summary.policy = {k: row.get(k) for k in POLICY_KEYS if k in row}
            summary.policy["target_ms"] = row.get("target_ms")
            summary.policy["gate_window"] = row.get("gate_window")
        elif event == "turn":
            summary.turns.append(row)
            seconds = float(row["elapsed_s"]) if _is_number(row.get("elapsed_s")) else 0.0
            _bump(summary.by_role, str(row.get("role") or "?"), seconds)
            _bump(summary.by_phase, str(row.get("phase") or "other"), seconds)
        else:
            summary.events.append(row)
    return summary


def throughput_flags(summary: LedgerSummary, *, student_iters: int) -> dict[str, bool]:
    """The three report flags. ``budget_starved`` needs the run's final iteration count."""
    over = any(e.get("event") == "over_share" for e in summary.events)
    escape = any(e.get("event") == "escape_fired" for e in summary.events)
    min_iters = summary.policy.get("min_student_iters", 5)
    try:
        min_iters = int(min_iters)
    except (TypeError, ValueError):
        min_iters = 5
    return {"teacher_over_share": over,
            "budget_starved": over and student_iters < min_iters,
            "escape_fired": escape}


def _turns_text(table: dict) -> str:
    """Turns and seconds per key, in the order the ledger first saw them."""
    if not table:
        return "none"
    return "; ".join(f"{k} {v['turns']} / {v['seconds']:.1f} s" for k, v in table.items())


def status_lines(ledger: str | Path | None, status_path: str | Path | None) -> list[str]:
    """The two ``forge status`` lines: turns by role, and the rules currently triggered."""
    summary = read_ledger(ledger)
    if not summary.turns and not summary.events:
        return []
    iters = student_iterations(status_path)
    flags = throughput_flags(summary, student_iters=iters)
    active = [name for name, on in flags.items() if on]
    soft = sum(1 for e in summary.events if e.get("event") == "soft_brake")
    hard = sum(1 for e in summary.events if e.get("event") == "hard_stop")
    return [f"  throughput: {_turns_text(summary.by_role)}",
            f"  throughput rules: {', '.join(active) if active else 'none'} "
            f"(soft brakes {soft}, hard stops {hard})"]


def _usage_seconds(row: Mapping) -> float:
    for key in ("duration_s", "seconds", "elapsed_s"):
        if _is_number(row.get(key)):
            return float(row[key])
    return 0.0


def _usage_tokens(row: Mapping) -> float:
    if _is_number(row.get("total_tokens")):
        return float(row["total_tokens"])
    parts = [row.get("prompt_tokens"), row.get("completion_tokens"),
             row.get("input_tokens"), row.get("output_tokens")]
    known = [float(v) for v in parts if _is_number(v)]
    if known:
        return sum(known)
    if _is_number(row.get("tokens")):
        return float(row["tokens"])
    return 0.0


def read_usage(path: str | Path | None) -> dict:
    """Aggregate ``agent_usage.jsonl`` (written by the backends task, when present).

    Tolerant to the field names that ledger may use; rows without a role are grouped
    under ``?``. Returns ``{}`` when the file does not exist or holds nothing usable.
    """
    totals: dict[tuple[str, str], dict] = {}
    for row in _read_rows(path):
        role = str(row.get("role") or "?")
        phase = str(row.get("phase") or "?")
        cell = totals.setdefault((role, phase), {"tokens": 0.0, "seconds": 0.0, "rows": 0})
        cell["tokens"] += _usage_tokens(row)
        cell["seconds"] += _usage_seconds(row)
        cell["rows"] += 1
    return totals


def report_lines(ledger: str | Path | None, status_path: str | Path | None, *,
                 usage_path: str | Path | None = None) -> list[str]:
    """The ``## Throughput`` section of ``report.md``."""
    summary = read_ledger(ledger)
    iters = student_iterations(status_path)
    lines = ["## Throughput", ""]
    if not summary.turns and not summary.events:
        lines.append("- governor: no ledger (FK_GOVERNOR=off, or the loop did not run)")
        return lines
    p = summary.policy
    policy_bits = [f"{k}={p[k]}" for k in POLICY_KEYS if p.get(k) is not None]
    if policy_bits:
        lines.append("- policy: " + ", ".join(policy_bits))
    lines.append(f"- student iterations: {iters} (student/status.jsonl)")
    flags = throughput_flags(summary, student_iters=iters)
    lines.append("- flags: " + ", ".join(
        f"{name}={'yes' if on else 'no'}" for name, on in flags.items()))
    lines.append(f"- turns by role: {_turns_text(summary.by_role)}")
    lines.append(f"- turns by phase: {_turns_text(summary.by_phase)}")
    soft = sum(1 for e in summary.events if e.get("event") == "soft_brake")
    hard = sum(1 for e in summary.events if e.get("event") == "hard_stop")
    if soft or hard:
        lines.append(f"- brakes: {soft} soft, {hard} hard")
    usage = read_usage(usage_path)
    if usage:
        cells = []
        for (role, phase), cell in sorted(usage.items()):
            cells.append(f"{role}/{phase} {int(round(cell['tokens']))} tokens / "
                         f"{cell['seconds']:.1f} s")
        lines.append("- usage (agent_usage.jsonl): " + "; ".join(cells))
    return lines
