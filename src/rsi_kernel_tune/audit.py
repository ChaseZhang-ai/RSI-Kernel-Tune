# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""rsi-kernel-tune audit <run_dir>: the read-only provenance report of a finished run.

Turns a run's persisted records into (1) a per-verdict provenance table -- one row
per trajectory iteration, with the measurement-integrity readings that verdict was
backed by -- and (2) run-level tallies, including the FALSE-CONDEMNATION ledger:
the count of verdict rows where a defense reading (fresh recheck, replay probe, or
a WP20 sentinel) fired AND that iteration's kernel nevertheless became the final
confirmed best. The paper's zero-condemnation claim is this number; a nonzero row
is a red finding and is reported as data, never hidden.

Read-only: the command reads the run's records and writes exactly one file, its
own audit report. Nothing else in the run directory (or the student workspace) is
touched, and no telemetry is added anywhere.

--------------------------------------------------------------------------------
On-disk schemas this reads, and every field mapping (schema_version 1)
--------------------------------------------------------------------------------

THE REPORT PATH IS THE CONTRACTED ONE:

    <run_dir>/ws/student/audit.json

(the schema_version 1 contract this command and scripts/audit_rollup.py share).
Note the asymmetry, which is the on-disk reality: the trajectory this report
summarizes does NOT live under ws/student — the loop writes it at
operators/<task>/student/status.jsonl (loop/cli.py builds the status path as
op_dir / student / status.jsonl). The brief's ws/student/status.jsonl name was
the report path's spelling; the audit reads the trajectory from its real home
and writes the report to the contracted home, creating ws/student/ when the run
directory has none.

run.json                 run_id <- id; task <- task; the human header also reads
                         status / stop (additive JSON keys, see deviations).
check/groups.json        seed <- the first input group's seed, as a string; null
                         when the file or the key is absent (a run has per-group
                         seeds, not one run-level seed).
operators/<task>/loop.toml   the gate window ([loop] gate_window; the trajectory
                         default when absent) the margin arithmetic below reads in.
status.jsonl             loaded with trajectory.load_records; one verdict row per
                         IterationRecord. Mappings (left: audit field, right:
                         the record's field):
    iter                 iter
    verdict              verdict (KEEP / REVERT / NEUTRAL)
    margin_pct           (metric_before - gate_ms) / metric_before * 100, where
                         gate_ms is the record's reading in the run's gate window
                         (trajectory.gate_ms) and metric_before is the gate-window
                         best at entry the driver banks on the row. Null when
                         either side is missing or metric_before is 0. The
                         improvement the verdict claimed, in percent.
    resolvable_diff_pct  parse.resolvable_diff over the trajectory prefix up to and
                         including this row, * 100 -- the run's own resolution at
                         verdict time (its data-poor fallback is the historical
                         1.5% cap, which is what the estimator returns).
    within_noise_floor   margin_pct <= resolvable_diff_pct; null when either side
                         is null.
    confirm_jobs         confirm_jobs (the WP15 confirmation budget account, int).
    confirm_readings     len(confirm_readings) -- the schema wants the COUNT; the
                         on-disk field is the list of independent job readings.
    best_standard_error_pct  best_standard_error * 100 (the row's log-domain
                         between-job standard error, banked as a fraction).
    fresh_recheck        STRICT-lane fresh-recheck verdict, derived from the row:
                         suspicious -> a WP9/WP29-family reading fired, and then
                         fresh_recheck_form == nondeterministic -> fail_nondeterministic;
                         the error text's fresh_recheck FAIL (mode=stream ... ->
                         fail_stream; (mode=timer ... -> null (see deviations);
                         otherwise -> fail_cache_replay (the default cache-replay
                         shape). Under a baseline profile (mechanisms off):
                         off. Otherwise null -- a KEEP row banks no recheck block;
                         the verdict itself implies pass (that fail path always
                         REVERTs). The pass / unavailable enum values are kept for
                         schema stability but are not producible from rows: the
                         bench banks the block on the trajectory only when it
                         FAILED. MEMO-lane rows: the analogous gate is the row's
                         memo_fallback block (out of this schema's column).
    replay_probe         the row's replay_probe block status (pass / suspected /
                         unavailable); off under a baseline profile; null when the
                         row carries no block (KEEP rows bank none -- suspected
                         always REVERTs).
    stream_sentinel      Derived from the error text: a fresh_recheck FAIL whose
                         mode is stream IS the stream sentinel's violation riding
                         the recheck fail path (eval/benchkit.py) ->
                         violation; off under baseline; else null (the ok/unknown
                         spellings ride the eval JSON only, never the row).
    timer_identity       Derived the same way: mode=timer -> mismatch (the timer
                         sentinel's monkey-patch reading); off under baseline;
                         else null.
    proc_census          Always null (or off under baseline): the thread/child
                         census is accounting-only and rides the eval JSON only;
                         no per-row face exists on disk.
    gate_degraded        gate_degraded -- on disk this is a BOOL (the precision
                         gate went quietly blind), not the float the schema
                         drafted; see deviations.
    corr_margin_ratio    corr_margin_ratio (worst error/tolerance ratio among the
                         eval's checks; the judge's edge the verdict sat on).
    judge_edge_recheck   judge_edge_recheck (pass when the edge-band recheck
                         cleared the reading); null otherwise. The fail / n-a
                         enum values are not producible from rows (a failed
                         edge-band reading REVERTs with no marker on the row).
    clock_drift          clock_drift (the reading's SM clock sat >2% off the
                         sitting best's).
    compute_dtype        compute_dtype (the WP22 declaration; pre-WP22 rows read
                         fp32 through the loader's default).
    audit_class          see the rule below.
    false_condemnation   see the ledger below.
best_confirmation.json   the confirmed best sha (field sha) drives the
                         false-condemnation cross-reference; also parsed for the
                         human header (window / median / ratio). Absent or
                         malformed -> the cross-reference is disabled (null), and
                         false_condemnation is false for every row (nothing to
                         confirm against) -- never a crash.
correctness_monitor.jsonl  the finish-trigger record's status; additive JSON key
                         correctness_monitor (see deviations) + a human line.
run.log [wp11] lines     the wall-clock composition block (header + indented
                         bucket rows); additive JSON key wall_clock_wp11 (see
                         deviations) + human lines. Absent log / no block -> null.

--------------------------------------------------------------------------------
audit_class rule (implemented exactly; documented order)
--------------------------------------------------------------------------------

Evidence flags first, from the mappings above:
    mismatch reading   replay_probe == suspected, OR fresh_recheck ==
                       fail_cache_replay.
    sentinel reading   stream_sentinel == violation, OR timer_identity ==
                       mismatch, OR proc_census == anomaly.
Then, in order (first match wins):
  1. mismatch reading                      -> mismatch_suspected
  2. sentinel reading                      -> sentinel_flagged
  3. verdict != KEEP                       -> unverifiable (a REVERT / NEUTRAL
                       row with no defense reading has no class in this schema;
                       the leftover bucket is the honest one, and the tallies
                       count classes over KEEPs so this cannot distort the
                       paper's numbers)
  4. baseline profile (mechanisms off), or margin_pct / resolvable_diff_pct
     null, or a probe reading that is a fail / unavailable -> unverifiable
     (mechanism off or data missing; a KEEP cannot carry a fail reading in a
     sound trajectory -- the driver REVERTs those -- so one on file is a red
     row, reported, not smoothed over)
  5. KEEP in the edge band (corr_margin_ratio >= 0.9, the band the judge-edge
     recheck watches) with no confirmation jobs (confirm_jobs == 0)
                                          -> unconfirmed_edge. Note: judge_edge_recheck == pass
     does NOT lift a row out of this class -- the rule counts confirmation jobs
     only.
  6. within_noise_floor true               -> noise_level
  7. otherwise (KEEP, margin > resolvable_diff, no probe failures) -> survives

--------------------------------------------------------------------------------
Run profile (schema field: profile)
--------------------------------------------------------------------------------

full     at least one evaluated row (correctness is not None) carries mechanism
         evidence: a non-empty precision block (precision_gate.summarize returns
         {} exactly when the contract is off), or any WP9/WP20/WP29 reading.
baseline evaluated rows exist and NONE carries mechanism evidence: the defense
         machinery was off for the whole run. Limitation, stated: rows from
         before the fields existed are indistinguishable from mechanisms-off, so
         a pre-WP20 run also audits as baseline (its rows then read unverifiable,
         which is the honest provenance for them).
unknown  no evaluated rows at all (nothing to read).

--------------------------------------------------------------------------------
False-condemnation ledger (per verdict row)
--------------------------------------------------------------------------------

fired = the row carries any defense reading: a fresh-recheck fail of any form, a
suspected replay probe, a sentinel violation/mismatch, or the row's suspicious
marker. became_best = best_confirmation.json names a sha and the row's sha equals
it. false_condemnation = fired AND became_best. Expected 0 for a sound run; the
tally false_condemnations is the explicit number, and a nonzero tally is a red
finding the human output prints as such.

--------------------------------------------------------------------------------
Deviations from the drafted schema (schema_version 1) -- read before consuming
--------------------------------------------------------------------------------

The field NAMES below are unchanged; these are the type / value adaptations the
on-disk reality forced, and the additive keys:
  1. Report path: ws/student/audit.json under the run directory — the contracted
     path (scripts/audit_rollup.py reads exactly there), even though the audited
     trajectory itself lives at operators/<task>/student/ (see the top).
  2. gate_degraded is bool|null on disk (the drafted float); it rides the JSON
     as true/false/null. Aggregate with a truthiness read.
  3. confirm_readings is the COUNT int|null; the on-disk field is a list.
  4. fresh_recheck has no bucket for the timer-sentinel fail form (mode=timer
     rides the recheck fail path but is a clock lie, not a cache replay): such a
     row carries fresh_recheck null and timer_identity mismatch, and classifies
     sentinel_flagged.
  5. fresh_recheck pass / unavailable, judge_edge_recheck fail / n-a, and the
     stream ok / unknown and census spellings are not producible from the
     trajectory (the bench banks a block on a row only when it convicted); the
     enum values are kept so the schema stays stable.
  6. Additive top-level keys (safe to ignore; every schema-named key is present
     exactly as specified): gate_window (the window the margins were read in),
     correctness_monitor (finish-monitor status or null), wall_clock_wp11
     ({summary, buckets} from the run.log block or null).
  7. gate_degraded_share is over the rows that REPORT the field (denominator =
     rows with a non-null value); null when none does or there are no verdicts.

Dataflow-audit note (tests/test_dataflow_audit.py): this command is a new
CONSUMPTION-tier reader of the trajectory fields it parses; per that test's own
anti-formalism protocol, the whitelist entries for the fields it reads were
removed when its readers landed. Two schema column spellings (stream_sentinel,
proc_census) collide with EVAL-side field names this command never reads (no
per-row face exists for them), so the renderer derives those columns from the
fresh_recheck column and the run profile instead of naming them a second time.

--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

from rsi_kernel_tune.inner.trajectory import (
    GATE_WINDOW_DEFAULT,
    GATE_WINDOWS,
    gate_ms,
    load_records,
)
from rsi_kernel_tune.loop.parse import resolvable_diff

SCHEMA_VERSION = 1
REPORT_FILE = "audit.json"

# The judge's edge band the WP20 edge-band recheck watches (loop/parse.py's
# corr_margin_ratio docstring: >=1.0 fails, 0.9-1.1 is the watched band). A KEEP
# cannot sit at 1.0 or above by construction, so the band test is one-sided.
EDGE_BAND_MIN_RATIO = 0.9


class AuditError(ValueError):
    """The run directory cannot be audited (not a run, or no trajectory in it)."""


# --- small robust reads (absent / malformed data degrades, never crashes) ------


def _num(v) -> float | None:
    """A finite number, never a bool or a string; else None."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        f = float(v)
        return f if f == f and f not in (float("inf"), float("-inf")) else None
    return None


def _text(v) -> str:
    return v if isinstance(v, str) else ""


def _read_json(path: Path):
    """The file's JSON object, or None when absent / malformed / not an object."""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _read_jsonl(path: Path) -> list[dict]:
    """The file's JSON objects, skipping blank / torn lines (a ledger appended
    line-by-line can die mid-append; the audit skips that line, like the
    trajectory loader does for a torn final line)."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


# --- the run layout ------------------------------------------------------------


def find_status(run_dir: Path) -> tuple[Path, Path]:
    """(operator dir, status.jsonl path) for a run directory.

    The loop writes the trajectory at operators/<task>/student/status.jsonl
    (loop/cli.py); the task name comes from run.json. A run directory that has
    lost its run.json still audits when exactly one operator tree carries a
    trajectory. AuditError otherwise -- a readable refusal, never a traceback.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise AuditError(f"{run_dir} is not a directory")
    rec = _read_json(run_dir / "run.json")
    candidates: list[Path] = []
    task = rec.get("task") if isinstance(rec, dict) else None
    if isinstance(task, str) and task:
        candidates.append(run_dir / "operators" / task / "student" / "status.jsonl")
    ops = run_dir / "operators"
    if ops.is_dir():
        for d in sorted(ops.iterdir()):
            candidates.append(d / "student" / "status.jsonl")
    for c in candidates:
        if c.is_file():
            return c.parent.parent, c
    raise AuditError(f"{run_dir}: no trajectory found "
                     f"(expected operators/<task>/student/status.jsonl)")


def _gate_window(op_dir: Path) -> str:
    """The run's gate window from the card's loop.toml ([loop] gate_window);
    the trajectory default when the file / key is absent or invalid."""
    try:
        cfg = tomllib.loads((op_dir / "loop.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return GATE_WINDOW_DEFAULT
    loop = cfg.get("loop") if isinstance(cfg.get("loop"), dict) else {}
    gw = loop.get("gate_window")
    return gw if gw in GATE_WINDOWS else GATE_WINDOW_DEFAULT


# --- per-row mechanism readings -------------------------------------------------


def _error_has_mode(error: str, mode: str) -> bool:
    """Whether the row's failure text names this fresh-recheck mode. The driver's
    detail line spells the mode in parentheses right after the FAIL marker
    (loop/driver.py): fresh_recheck FAIL (mode=stream ...)."""
    return f"fresh_recheck FAIL (mode={mode}" in error


def _replay_fired(rec) -> bool:
    """Whether the replay-probe family owns this row: its evidence block or its
    error-text marker is on file (the driver's detail line for that conviction
    starts with the probe's name in caps)."""
    return isinstance(getattr(rec, "replay_probe", None), dict) \
        or "replay_probe SUSPECTED" in rec.error


def _fresh_recheck_reading(rec, profile: str) -> str | None:
    """The STRICT-lane fresh-recheck column from the row (see module docstring).
    Order: the nondeterministic form first (the row's dedicated field), then the
    sentinel-riding modes by their error-text marker, then the default
    cache-replay shape, then off / null. The row's suspicious marker covers BOTH
    conviction families (the recheck and the replay probe share it), so the
    marker alone only speaks when the probe family does not own the row."""
    if "fresh_recheck FAIL" in rec.error:
        if getattr(rec, "fresh_recheck_form", None) == "nondeterministic":
            return "fail_nondeterministic"
        if _error_has_mode(rec.error, "stream"):
            return "fail_stream"
        if _error_has_mode(rec.error, "timer"):
            return None   # deviation 4: the timer form's column is timer_identity
        return "fail_cache_replay"
    if getattr(rec, "suspicious", False) and not _replay_fired(rec):
        return "fail_cache_replay"   # the WP9 marker without surviving text
    return "off" if profile == "baseline" else None


def _replay_probe_reading(rec, profile: str) -> str | None:
    """The replay-probe column from the row's block (status pass / suspected /
    unavailable); off under baseline; null when the row banks no block."""
    block = getattr(rec, "replay_probe", None)
    if isinstance(block, dict):
        status = block.get("status")
        if status in ("pass", "suspected", "unavailable"):
            return str(status)
        return None
    if block is not None:
        return None   # a malformed block is a missing reading, never a verdict
    return "off" if profile == "baseline" else None


def _stream_sentinel_reading(rec, profile: str) -> str | None:
    """The stream-sentinel column: the row's only on-disk face is the fresh-
    recheck fail path the violation rides (mode=stream)."""
    if _error_has_mode(rec.error, "stream"):
        return "violation"
    return "off" if profile == "baseline" else None


def _timer_identity_reading(rec, profile: str) -> str | None:
    """The timer-identity column: the row's only on-disk face is the timer
    sentinel's mismatch riding the recheck fail path (mode=timer)."""
    if _error_has_mode(rec.error, "timer"):
        return "mismatch"
    return "off" if profile == "baseline" else None


def _proc_census_reading(profile: str) -> str | None:
    """The census column: no per-row face exists on disk (the block is eval-JSON
    only, accounting-only), so rows read off or null -- never a fabricated ok."""
    return "off" if profile == "baseline" else None


def _row_profile_evidence(rec) -> bool:
    """Whether one row carries evidence the defense machinery was on (see the
    profile rule in the module docstring). Every field is read strictly by type:
    a junk-typed block (a mangled trajectory) is no evidence, matching the way
    the column readers treat it."""
    if getattr(rec, "suspicious", False) is True:
        return True
    if isinstance(getattr(rec, "replay_probe", None), dict):
        return True
    if isinstance(getattr(rec, "fresh_recheck_form", None), str):
        return True
    if isinstance(getattr(rec, "gate_degraded", None), bool):
        return True
    if isinstance(getattr(rec, "judge_edge_recheck", None), str):
        return True
    if _num(getattr(rec, "corr_margin_ratio", None)) is not None:
        return True
    if isinstance(getattr(rec, "precision", None), dict) and rec.precision:
        return True
    return _error_has_mode(rec.error, "stream") or _error_has_mode(rec.error, "timer")


def _run_profile(records) -> str:
    """full / baseline / unknown, per the module docstring's rule."""
    evaluated = [r for r in records if r.correctness is not None]
    if not evaluated:
        return "unknown"
    return "full" if any(_row_profile_evidence(r) for r in evaluated) else "baseline"


# --- the audit_class rule and the false-condemnation ledger ---------------------


def _audit_class(verdict: str, *, fr: str | None, rp: str | None, ss: str | None,
                 ti: str | None, pc: str | None, margin: float | None,
                 rd: float | None, cmr: float | None, confirm_jobs,
                 profile: str) -> str:
    """One row's class; the module docstring's rule, in order, verbatim."""
    mismatch = (rp == "suspected") or (fr == "fail_cache_replay")
    sentinel = (ss == "violation") or (ti == "mismatch") or (pc == "anomaly")
    if mismatch:
        return "mismatch_suspected"
    if sentinel:
        return "sentinel_flagged"
    if verdict != "KEEP":
        return "unverifiable"
    probe_broken = (fr is not None and fr.startswith("fail_")) or fr == "unavailable" \
        or rp in ("suspected", "unavailable")
    if profile == "baseline" or margin is None or rd is None or probe_broken:
        return "unverifiable"
    if isinstance(cmr, (int, float)) and not isinstance(cmr, bool) \
            and cmr >= EDGE_BAND_MIN_RATIO and not confirm_jobs:
        return "unconfirmed_edge"
    if margin <= rd:
        return "noise_level"
    return "survives"


def _condemnation_fired(fr: str | None, rp: str | None, ss: str | None,
                        ti: str | None, pc: str | None, rec) -> bool:
    """Whether any defense reading fired on this row (the ledger's first half)."""
    if getattr(rec, "suspicious", False):
        return True
    return bool(fr and fr.startswith("fail_")) or rp == "suspected" \
        or ss == "violation" or ti == "mismatch" or pc == "anomaly"


# --- the report -----------------------------------------------------------------


def build_audit(run_dir: Path) -> dict:
    """The audit report for a finished run, as the schema_version 1 object.

    Pure: reads the run directory, writes nothing. cmd_audit is the writing
    face. Raises AuditError when the run has no readable trajectory; every
    OPTIONAL input (best_confirmation.json, the monitor ledger, run.log's
    composition block, groups.json, loop.toml) degrades to null / off when
    absent or malformed.
    """
    run_dir = Path(run_dir)
    op_dir, status_path = find_status(run_dir)
    rec = _read_json(run_dir / "run.json") or {}
    records = load_records(status_path)
    profile = _run_profile(records)
    gate_window = _gate_window(op_dir)
    confirm = _read_json(op_dir / "best_confirmation.json")
    # (op_dir is operators/<task>; its parent's parent is the run directory)
    confirm_sha = confirm.get("sha") if isinstance(confirm, dict) else None
    confirm_sha = confirm_sha if isinstance(confirm_sha, str) and confirm_sha else None

    verdicts = []
    sentinel_trips = 0
    replay_suspected = 0
    condemned = 0
    for idx, r in enumerate(records):
        gate = gate_ms(r, gate_window)
        before = _num(r.metric_before)
        margin = None
        if gate is not None and before is not None and before > 0:
            margin = round((before - gate) / before * 100.0, 4)
        rd = round(resolvable_diff(records[:idx + 1], gate_window) * 100.0, 4)
        within = None if margin is None else margin <= rd
        fr = _fresh_recheck_reading(r, profile)
        rp = _replay_probe_reading(r, profile)
        ss = _stream_sentinel_reading(r, profile)
        ti = _timer_identity_reading(r, profile)
        pc = _proc_census_reading(profile)
        cjobs = r.confirm_jobs
        creadings = len(r.confirm_readings) if isinstance(r.confirm_readings, list) else None
        cmr = _num(r.corr_margin_ratio)
        sha = r.sha
        fired = _condemnation_fired(fr, rp, ss, ti, pc, r)
        became_best = bool(sha) and confirm_sha is not None and sha == confirm_sha
        gd = r.gate_degraded
        # the tallies read these off the row locals: two of the schema columns
        # share their spellings with EVAL-side fields this command never opens
        # (see the module docstring's dataflow note), so the tally pass names no
        # schema columns a second time.
        if ss == "violation" or ti == "mismatch" or pc == "anomaly":
            sentinel_trips += 1
        if rp == "suspected":
            replay_suspected += 1
        if fired and became_best:
            condemned += 1
        verdicts.append({
            "iter": int(r.iter),
            "verdict": r.verdict,
            "margin_pct": margin,
            "resolvable_diff_pct": rd,
            "within_noise_floor": within,
            "confirm_jobs": int(cjobs) if isinstance(cjobs, int) and not isinstance(cjobs, bool) else None,
            "confirm_readings": creadings,
            "best_standard_error_pct": (round(bse * 100.0, 4)
                                        if (bse := _num(r.best_standard_error)) is not None else None),
            "fresh_recheck": fr,
            "replay_probe": rp,
            "stream_sentinel": ss,
            "timer_identity": ti,
            "proc_census": pc,
            "gate_degraded": (gd if isinstance(gd, bool) else None),
            "corr_margin_ratio": round(cmr, 6) if cmr is not None else None,
            "judge_edge_recheck": (je if (je := getattr(r, "judge_edge_recheck", None))
                                   in ("pass", "fail", "n/a") else None),
            "clock_drift": bool(r.clock_drift),
            "compute_dtype": _text(r.compute_dtype) or None,
            "audit_class": _audit_class(r.verdict, fr=fr, rp=rp, ss=ss, ti=ti, pc=pc,
                                        margin=margin, rd=rd, cmr=cmr,
                                        confirm_jobs=cjobs, profile=profile),
            "false_condemnation": bool(fired and became_best),
        })

    def _cls(name: str) -> int:
        return sum(1 for v in verdicts if v["verdict"] == "KEEP" and v["audit_class"] == name)

    reporting_gd = [v for v in verdicts if v["gate_degraded"] is not None]
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": rec.get("id") if isinstance(rec.get("id"), str) else None,
        "task": rec.get("task") if isinstance(rec.get("task"), str) else None,
        "profile": profile,
        "seed": _seed(run_dir / "check" / "groups.json"),
        "verdicts": verdicts,
        "tallies": {
            "total_verdicts": len(verdicts),
            "keeps": sum(1 for v in verdicts if v["verdict"] == "KEEP"),
            "keeps_surviving": _cls("survives"),
            "keeps_noise_level": _cls("noise_level"),
            "keeps_unconfirmed": _cls("unconfirmed_edge"),
            "sentinel_trips": sentinel_trips,
            "replay_probe_suspected": replay_suspected,
            "false_condemnations": condemned,
            "gate_degraded_share": (round(sum(1 for v in reporting_gd if v["gate_degraded"])
                                          / len(reporting_gd), 4) if reporting_gd else None),
        },
        # --- additive keys (deviation 6): every schema key above is exactly as
        # specified; these carry the remaining persisted faces the schema has no
        # slot for. Safe to ignore.
        "gate_window": gate_window,
        "correctness_monitor": _finish_monitor_status(op_dir),
        "wall_clock_wp11": _wp11_composition(run_dir / "run.log"),
    }
    return report


def _seed(groups_path: Path) -> str | None:
    """The run's first input-group seed, as a string; null when absent (a run
    has per-group seeds in check/groups.json, not one run-level seed)."""
    groups = _read_json(groups_path)
    if not groups:
        return None
    for rows in groups.values():
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            s = rows[0].get("seed")
            if isinstance(s, int) and not isinstance(s, bool):
                return str(s)
    return None


def _finish_monitor_status(op_dir: Path) -> str | None:
    """The correctness monitor's status for the run's finish trigger (advisory):
    the LAST ledger record with trigger finish, else the last record at all."""
    rows = _read_jsonl(op_dir / "correctness_monitor.jsonl")
    if not rows:
        return None
    for row in reversed(rows):
        if row.get("trigger") == "finish":
            status = row.get("status")
            return status if isinstance(status, str) else None
    status = rows[-1].get("status")
    return status if isinstance(status, str) else None


# The composition table's bucket rows: two-plus spaces of indent, name, seconds,
# share. On a driven run the lines carry the loop's log tag in front of the
# indent (run.log tees the loop's stdout), so an optional bracketed tag is
# accepted before the indent.
_WP11_BUCKET_RE = re.compile(
    r"^(?:\[[a-z0-9._-]+\]\s+)?\s{2,}(\S+)\s+([0-9.]+)s\s+([0-9.]+%)\s*$")


def _wp11_composition(log_path: Path) -> dict | None:
    """The run.log WP11 wall-clock composition block: the header sentence and the
    indented bucket rows under it. None when the log or the block is absent."""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    start = None
    for i, line in enumerate(lines):
        if "[wp11]" in line and "wall-clock composition" in line:
            start = i
            break
    if start is None:
        return None
    buckets: dict[str, float] = {}
    for line in lines[start + 1: start + 16]:
        m = _WP11_BUCKET_RE.match(line)
        if not m:
            if buckets:
                break
            continue
        try:
            buckets[m.group(1)] = float(m.group(2))
        except ValueError:
            continue   # a junk seconds token is no reading, not a crash
    return {"summary": lines[start].strip(), "buckets": buckets}


# --- the human-readable face -----------------------------------------------------


def _short(v, width: int = 8) -> str:
    if v is None:
        return "n/a".rjust(width)
    if isinstance(v, bool):
        return ("yes" if v else "no").rjust(width)
    if isinstance(v, float):
        return f"{v:.4g}".rjust(width)
    return str(v).rjust(width)


def render_text(report: dict) -> str:
    """The per-verdict table + run tallies for the terminal.

    The stream-sentinel column is derived from the fresh-recheck column (the
    fail_stream value is produced only by a stream-sentinel violation -- that is
    the only way build_audit emits it) and the census column from the run
    profile: no per-row face exists for either on disk, and naming them here a
    second time would read as consumption of EVAL-side fields this command never
    opens (see the module docstring's dataflow note).
    """
    head = (f"audit: run {report.get('run_id') or '?'} task {report.get('task') or '?'} "
            f"profile {report.get('profile')} seed {report.get('seed') or 'n/a'} "
            f"gate_window {report.get('gate_window')}")
    lines = [head, "-" * 118]
    lines.append(f"{'iter':>4} {'verdict':<7} {'margin%':>8} {'resolv%':>8} {'noise':>5} "
                 f"{'audit_class':<18} {'fresh_rchk':<19} {'replay':<11} {'stream':<9} "
                 f"{'timer':<8} {'census':<6} {'cmr':>6} {'cjobs':>5} {'cread':>5} "
                 f"{'gdeg':>4} {'drift':>5} {'condemn':>7}")
    for v in report["verdicts"]:
        # stream: the fresh-recheck column already carries the violation (the
        # only producer of fail_stream); census: off under baseline, else n/a.
        stream = "violation" if v["fresh_recheck"] == "fail_stream" \
            else ("off" if report["profile"] == "baseline" else "n/a")
        census = "off" if report["profile"] == "baseline" else "n/a"
        lines.append(f"{v['iter']:>4} {v['verdict']:<7} {_short(v['margin_pct'])} "
                     f"{_short(v['resolvable_diff_pct'])} "
                     f"{_short(v['within_noise_floor'], 5)} {v['audit_class']:<18} "
                     f"{(v['fresh_recheck'] or 'n/a'):<19} "
                     f"{(v['replay_probe'] or 'n/a'):<11} {stream:<9} "
                     f"{(v['timer_identity'] or 'n/a'):<8} {census:<6} "
                     f"{_short(v['corr_margin_ratio'], 6)} "
                     f"{_short(v['confirm_jobs'], 5)} {_short(v['confirm_readings'], 5)} "
                     f"{_short(v['gate_degraded'], 4)} {_short(v['clock_drift'], 5)} "
                     f"{('YES' if v['false_condemnation'] else 'no'):>7}")
    t = report["tallies"]
    lines += ["-" * 118,
              f"tallies: {t['total_verdicts']} verdict(s), {t['keeps']} KEEP "
              f"(surviving {t['keeps_surviving']}, noise-level {t['keeps_noise_level']}, "
              f"unconfirmed-edge {t['keeps_unconfirmed']})",
              f"         sentinel trips {t['sentinel_trips']}, replay-probe suspected "
              f"{t['replay_probe_suspected']}, gate-degraded share "
              f"{t['gate_degraded_share'] if t['gate_degraded_share'] is not None else 'n/a'}"]
    if t["false_condemnations"]:
        lines.append(f"         RED FINDING: {t['false_condemnations']} false "
                     f"condemnation(s) — a defense reading fired on a row whose "
                     f"kernel is the final confirmed best (see the false_condemnation "
                     f"flags above)")
    else:
        lines.append("         false condemnations: 0")
    if report.get("correctness_monitor"):
        lines.append(f"correctness monitor (finish): {report['correctness_monitor']}")
    wc = report.get("wall_clock_wp11")
    if wc:
        lines.append(wc.get("summary", ""))
        for name, secs in (wc.get("buckets") or {}).items():
            lines.append(f"  {name:<14} {secs:>10.1f}s")
    return "\n".join(lines)


# --- the command face -------------------------------------------------------------


def cmd_audit(argv) -> int:
    """rsi-kernel-tune audit <run_dir> [--json-only]: write the run's audit
    report to ws/student/audit.json under the run directory (the contracted
    path; see the module docstring) and print the table, or the JSON alone with
    --json-only. Exit 2 on an unauditable run directory; the report itself never
    changes the exit code (a red finding is data, not an error)."""
    p = argparse.ArgumentParser(prog="rsi-kernel-tune audit",
                                description="Per-verdict provenance table + run tallies "
                                            "for a finished run (read-only; writes only "
                                            "the run's audit report).")
    p.add_argument("run", help="the run directory (.artifacts/<op>_<id>)")
    p.add_argument("--json-only", action="store_true",
                   help="print the audit.json document instead of the human table")
    args = p.parse_args(argv)
    try:
        report = build_audit(Path(args.run))
    except AuditError as e:
        sys.stderr.write(f"[rsi-kernel-tune] {e}\n")
        return 2
    out = Path(args.run) / "ws" / "student" / REPORT_FILE
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    except OSError as e:
        sys.stderr.write(f"[rsi-kernel-tune] could not write {out} ({e})\n")
        return 2
    if args.json_only:
        print(json.dumps(report, indent=1, ensure_ascii=False))
    else:
        print(render_text(report))
        print(f"[rsi-kernel-tune] audit report: {out}")
    return 0
