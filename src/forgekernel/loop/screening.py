"""WP17: the screening primitive — contract C9's single implementation.

Four channels screen candidate designs against the sitting base before anything
takes the standard full eval gate: the student's verdict PROBE rounds, the
teacher's SWEEP grids, the MC-DELIVERY prescreen, and the M0 ARCH_PROBE. Each
was hand-rolled its own copy of the same five pieces; this module is the one
place they now live (the channels' verdict semantics are untouched — the
migration is subtraction, not invention; every helper here renders the bytes its
channel always rendered, and later channels — WP16's pattern search and compound
bundles, WP15's two-level ladder — consume these helpers instead of growing a
fifth copy):

  * frozen spec   — the driver owns the spec (teacher-authored recipe, driver
                    canonical-JSON hash); a submitted file that does not
                    hash-match is refused without a job.
  * paired timing — one GPU job times base and every variant interleaved
                    (``paired_order``), so every reading shares the node draw.
  * node dirty    — a pass whose GPU was left held (the NODE-DIRTY marker the
                    tools print after an orphan sweep) stops and is re-bought
                    once; a reading whose own gpu_state telemetry sits outside
                    the measurement band (``reading_dirty``) is marked and
                    produces no verdict — an honest hole over a dirty verdict.
  * budget clamp  — a bracket never buys jobs it does not have
                    (``clamp_bracket``); the job-side pass clock clamps once,
                    in lever_pricing_run (the tools' shared kernel).
  * the verdict   — a settled job's log carries a machine-readable closing
                    verdict (SWEEP-OK / MC-OK / ...); ``log_settled`` recovers a
                    first read that stopped short of it, ``settled_job`` applies
                    the re-buy-once policy, ``pair_outcome`` is the ARCH_PROBE
                    stamp's above/within-floor decision core.

Discipline this module must never grow (WP17 red lines): a screening reading is
never a gate; the analytic floor ranks candidates (``floor_rank``) and never
prices them; a node-dirty reading is marked, never judged.
"""

from __future__ import annotations

import hashlib
import json
import re

# --- frozen spec ----------------------------------------------------------------
# The driver owns a screening spec (the teacher's recipe on the milestone line, or
# the driver-generated prescreen spec); a submitted file is judged by content
# identity, never by trust. Key order and formatting do not matter; any added,
# dropped or changed variant does.

SWEEP_CRITERION = "min same-job ratio above floor"   # == sweep_run.CRITERION
SWEEP_MAX_VARIANTS = 8


def canonical_spec_hash(obj) -> str:
    """sha256 over the canonical JSON form (sorted keys, no whitespace): the
    content identity of a screening spec (the sweep recipe's canonical-JSON
    hash, one place for every channel that freezes one)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def sweep_spec_refused(recipe: dict) -> str | None:
    """Why the driver would not run ``recipe`` as written, or None: at most
    SWEEP_MAX_VARIANTS variants, one-token unique names (never `base`), every
    variant carrying at least one edit pair, and the criterion — when spelled —
    the one mechanical rule the sweep implements."""
    variants = recipe.get("variants")
    if not isinstance(variants, list) or not variants:
        return "recipe names no variants"
    if len(variants) > SWEEP_MAX_VARIANTS:
        return f"{len(variants)} variants (max {SWEEP_MAX_VARIANTS})"
    seen = set()
    for var in variants:
        if not isinstance(var, dict):
            return "a variant must be an object"
        name = re.sub(r"\s+", "_", str(var.get("name", "")).strip())
        if not name or name == "base" or name in seen:
            return f"variant name {name!r} is empty, reserved or duplicated"
        seen.add(name)
        edits = var.get("edits")
        if not isinstance(edits, list) or not edits:
            return f"variant {name!r} has no edits"
    if (crit := recipe.get("criterion")) and str(crit).strip() != SWEEP_CRITERION:
        return f"unknown criterion {crit!r}"
    return None


# --- paired timing -----------------------------------------------------------------
# The interleaved pass order lives ONCE, in lever_pricing_run.paired_order (the
# tools' shared timing kernel — the sweep, the prescreen and the pricing pass
# all call it); the driver-side face of the discipline is this comment and the
# settled-job policy below: one GPU job, base first, variants as listed, every
# reading sharing the job's one node draw.


# --- node dirty --------------------------------------------------------------------
# A timed-out command is killed as a PROCESS GROUP; the GPU is swept for compute
# processes still holding it, and whatever survives is printed as NODE-DIRTY: the
# pass stops (its later readings would be the orphan's share) and the driver
# re-buys the same job once on a fresh node. lever_pricing_run owns the sweep
# itself; this module owns the driver side: recognizing the marker and the
# once-only re-buy policy.

NODE_DIRTY_MARKER = "NODE-DIRTY"
TOOL_TRACEBACK = "Traceback (most recent call last)"
ABLATE_FAIL_RE = re.compile(r"^ABLATE-FAIL reason=(.*)$", re.M)

# WP1: two readings whose SM clocks sit further apart than this are not one
# measurement condition (a >2% clock move alone moves most kernels more than the
# resolution any quiet run can resolve). The gpu_state telemetry (timing.py) is
# what makes the check possible; a reading without it is simply never flagged.
CLOCK_DRIFT_BAND = 0.02
# temperature: two windows more than this many degrees apart are not one thermal
# condition either (a hot node reads systematically slow); both blocks must
# carry a temperature for the check to fire.
TEMP_DRIFT_BAND_C = 5.0


def node_dirty_reason(fail: str | None, log: str) -> str | None:
    """Why a screening job's node must be re-bought, or None: the caller's fail
    line, a sibling ABLATE-FAIL line, or the NODE-DIRTY marker — the pricing,
    sweep, prescreen and timeline checkers' one rule."""
    if fail and fail.startswith("node-dirty"):
        return f"PRICE-FAIL {fail}"
    if (m := ABLATE_FAIL_RE.search(log or "")) and m.group(1).strip().startswith("node-dirty"):
        return f"ABLATE-FAIL {m.group(1).strip()}"
    if NODE_DIRTY_MARKER in (log or ""):
        return "the job reported NODE-DIRTY"
    return None


def clock_drift(a, b) -> float | None:
    """Relative SM-clock distance of two gpu_state blocks, or None when either
    lacks a usable clock — an unmeasured condition, never a drift claim."""
    va, vb = (a or {}).get("sm_clock_mhz"), (b or {}).get("sm_clock_mhz")
    for v in (va, vb):
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
            return None
    return abs(va - vb) / min(va, vb)


def reading_dirty(gpu_state, ref_state=None, *, band: float = CLOCK_DRIFT_BAND,
                  temp_band_c: float = TEMP_DRIFT_BAND_C) -> str | None:
    """Why ``gpu_state`` (one reading's gpu_state telemetry block) sits outside
    the measurement condition, or None. ``ref_state`` is the block to read
    against (the pass's first reading, or the sitting best's); without one, only
    the reading's own impossible values flag. None — even when the block is
    missing or empty — means unmeasured, never dirty: a reading without
    telemetry is never flagged (the WP1 rule), and a flagged reading produces no
    verdict (the WP17 rule: honest marking over a dirty verdict)."""
    if not isinstance(gpu_state, dict) or not gpu_state:
        return None
    drift = clock_drift(gpu_state, ref_state)
    if drift is not None and drift > band:
        return f"sm clock {drift:.1%} off the reference reading (band {band:.0%})"
    ta, tb = gpu_state.get("temperature_c"), (ref_state or {}).get("temperature_c")
    if (ref_state and isinstance(ta, (int, float)) and not isinstance(ta, bool)
            and isinstance(tb, (int, float)) and not isinstance(tb, bool)
            and abs(ta - tb) > temp_band_c):
        return (f"temperature {ta:.0f}°C vs {tb:.0f}°C reference "
                f"(band ±{temp_band_c:.0f}°C)")
    return None


# --- budget clamp -------------------------------------------------------------------
def clamp_bracket(want: int, listed: int, budget: int) -> int:
    """How many candidates a screening bracket may probe: the requested width,
    clamped to what is listed and to the jobs the budget actually has (the
    ARCH_PROBE rule — it never buys jobs it does not have; a wider bracket
    degrades with the cut recorded)."""
    return min(max(0, int(want)), max(0, int(listed)), max(0, int(budget)))


# --- machine-readable verdicts -------------------------------------------------------
def log_settled(log: str, verdict_of, be, what: str, *, word: str, log_fn) -> str:
    """``log`` once it carries its channel's closing verdict: as read, or the
    job's archive read again (a first read that stopped short of the verdict may
    be complete on a second read — the free recovery). ``word`` is the channel's
    verdict word (SWEEP / MC / ...) exactly as the two hand-rolled checkers
    spelled it; ``log_fn`` is the caller's run-log sink."""
    from forgekernel.loop.driver import BackendError  # lazy: driver imports this module
    if verdict_of(log) is not None:
        return log
    job = str(getattr(be, "last_job_id", "") or "")
    reread = getattr(be, "reread_log", None)
    if not job or not callable(reread):
        return log
    log_fn(f"{what}: job {job} log carries no {word} verdict — re-reading the archive")
    try:
        again = reread(job)
    except BackendError as e:
        log_fn(f"{what}: re-read of job {job} failed ({e})")
        return log
    return again if len(again or "") > len(log or "") else log


def settled_job(be, branch: str, cmd: str, what: str, verdict_of, *, word: str,
                noun: str, rerun: bool = True, log_fn) -> "tuple[str, str]":
    """One screening job, settled: (log, job id). Re-bought once (``rerun``)
    when its GPU was left dirty (``node_dirty_reason``) or its log still carries
    no closing verdict and no tool traceback (a crashed tool is not re-bought).
    BackendError propagates. ``word`` / ``noun`` reproduce each channel's log
    wording byte for byte (SWEEP/recipe, MC/prescreen)."""
    attempts = 2 if rerun else 1
    log = ""
    for attempt in range(attempts):
        log = log_settled(be.submit_gpu(branch, cmd), verdict_of, be, what,
                          word=word, log_fn=log_fn)
        last = attempt == attempts - 1
        if not last and (dirty := node_dirty_reason(None, log)):
            log_fn(f"{what}: {dirty} — re-running the same {noun} once")
            continue
        if not last and verdict_of(log) is None and TOOL_TRACEBACK not in log:
            log_fn(f"{what}: job {getattr(be, 'last_job_id', '') or '?'} log still carries no "
                   f"{word} verdict — resubmitting the same {noun} once")
            continue
        break
    return log, str(getattr(be, "last_job_id", "") or "")


def pair_outcome(t0, t1, floor: float | None) -> "tuple[float | None, bool]":
    """The ARCH_PROBE stamp's decision core for a pair of time readings
    ``(delta, decisive)``: ``delta`` is the gap relative to the winner (the same
    ratio currency every other driver judgment reads), ``decisive`` True when it
    clears the resolution floor — the freeze follows the winner; anything within
    the floor freezes the teacher's preferred candidate. A missing reading is
    not decisive and carries no delta."""
    if t0 is None or t1 is None:
        return None, False
    win_t, lose_t = min(t0, t1), max(t0, t1)
    delta = (lose_t - win_t) / win_t
    return delta, delta > (floor if floor is not None else 0.015)


# --- the analytic floor as an ADVISORY rank (WP17 task 2; red line ①) ---------------
# The floor ranks candidates for a budget-limited probe; it never prices one.
# analytic_floor.py stays the single floor implementation — this module only
# merges a recipe's declared problem view over the task's own and reads the
# result.
def recipe_floor(recipe: dict, base_problem: dict | None = None) -> dict | None:
    """``analytic_floor.floor_for`` of one ARCH_PROBE candidate recipe, or None.

    The recipe may declare a ``"problem"`` view (the same schema floor_for
    reads: axes / inputs / outputs; arch and family default from the task's own
    problem so a tile-shape override is two keys, not a second card). No
    declared view and no task problem — None: nothing is ranked, the file order
    stands. A floor here is a SORT KEY, never a price (red line ①)."""
    view = recipe.get("problem") if isinstance(recipe, dict) else None
    if not isinstance(view, dict) and not isinstance(base_problem, dict):
        return None
    from forgekernel.loop.analytic_floor import floor_for  # the single implementation
    problem = dict(base_problem) if isinstance(base_problem, dict) else {}
    if isinstance(view, dict):
        for key in ("arch", "family", "op_type", "name", "inputs", "outputs"):
            if isinstance(view.get(key), (dict, list, str)):
                problem[key] = view[key]
        # axes merge per-key: a tile-shape override names the axes it changes,
        # not the whole problem again
        axes = dict(problem.get("axes") or {})
        if isinstance(view.get("axes"), dict):
            axes.update(view["axes"])
        if axes:
            problem["axes"] = axes
    try:
        out = floor_for(problem)
    except Exception:  # noqa: BLE001 — a malformed declared view ranks nothing
        return None
    return out if out and out.get("floor_ms") is not None else None


def floor_rank(requests: list, floor_of) -> list:
    """``requests`` ordered by their analytic floor, fastest floor first — the
    advisory pre-sort for a budget-limited probe (red line ①: a ranking, never
    a price, never a verdict). Stable: candidates without a floor keep their
    file order behind those with one, and an empty floor set returns the input
    order unchanged (the default-parameters behavior, byte for byte)."""
    keyed = [(i, r, floor_of(r)) for i, r in enumerate(requests)]
    with_floor = [(f["floor_ms"], i, r) for i, r, f in keyed if f]
    if not with_floor:
        return list(requests)
    without = [(i, r) for i, r, f in keyed if not f]
    ranked = sorted(with_floor, key=lambda t: (t[0], t[1]))
    out = [r for _, _, r in ranked] + [r for _, r in without]
    return out


def floor_note(ranked: list, floor_of) -> str:
    """The stamp's advisory note for a floor-ranked bracket, or "" when nothing
    was ranked. Says the ranking for what it is — advisory, never a verdict —
    and names each ranked candidate's floor."""
    rows = [(r, floor_of(r)) for r in ranked]
    rows = [(r, f) for r, f in rows if f]
    if not rows:
        return ""
    body = "; ".join(f"{r} floor={f['floor_ms']:.4g} ms" for r, f in rows)
    return (f"analytic-floor pre-rank (advisory only — a sort suggestion, never a "
            f"price or a verdict): {body}")


# --- the (family, arch) readings ledger (WP17 task 2) --------------------------------
LEDGER_SCHEMA = ("ts", "op_family", "arch", "candidate", "name", "tier",
                 "floor_ms", "bytes_min", "flops", "ms", "verdict", "job")


def ledger_row(*, ts: str, op_family: str, arch: str, candidate: str, name: str = "",
               tier: str = "mock", floor_ms=None, bytes_min=None, flops=None,
               ms=None, verdict: str = "measured", job: str = "") -> dict:
    """One ``evidence/arch_probe_ledger.jsonl`` row — the (family, arch) machine
    reading cache WP19's price cache and WP16's graveyard consume (the schema is
    the three-package seam; the integration step owns its consumers). Advisory
    bookkeeping only: nothing reads it back into a gate."""
    row = {"ts": ts, "op_family": op_family, "arch": arch, "candidate": candidate,
           "name": name, "tier": tier, "floor_ms": floor_ms, "bytes_min": bytes_min,
           "flops": flops, "ms": ms, "verdict": verdict, "job": job}
    return {k: row[k] for k in LEDGER_SCHEMA}


def append_ledger(path, row: dict) -> bool:
    """Append one ledger row as JSON lines. False (silently, with the caller
    logging) when the row cannot be written — a ledger is bookkeeping, its
    failure must never touch the probe outcome."""
    try:
        path = _path_of(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
        return True
    except (OSError, TypeError, ValueError):
        return False


def _path_of(path):
    from pathlib import Path
    return path if isinstance(path, Path) else Path(path)


# --- ARCH_PROBE fidelity tiers (WP17 task 3) ------------------------------------------
FIDELITY_TIERS = ("mock", "semi-real")
_FIDELITY_RE = re.compile(r"\s*fidelity\s*[:：]\s*([A-Za-z-]+)\s*", re.I)


def recipe_fidelity(recipe: dict) -> str:
    """The recipe's declared fidelity tier: ``"semi-real"`` when it says so, else
    ``"mock"`` (the historical default — every recipe that predates the field is
    a mock, byte for byte). Anything else spelled falls back to mock: the tier
    annotates the stamp, it never gates."""
    raw = recipe.get("fidelity") if isinstance(recipe, dict) else None
    tier = str(raw or "").strip().lower()
    return tier if tier in FIDELITY_TIERS else "mock"


def near_diff_reprobe(readings: dict, probed: list, floor: float | None,
                      spare_jobs: int) -> list:
    """The candidates a within-floor pair must re-measure once (WP17 task 3: a
    near-diff winner is a noise draw until a second draw says otherwise), or []
    when nothing is due: the decisive pair's gap must sit within the resolution
    floor, and the budget must still have the one job (``spare_jobs`` — the
    re-probe never buys jobs it does not have; the frozen runner takes both
    recipes in that one job, so one re-probe costs one round). The pair's own
    order (winner first) is the re-probe order."""
    decisive = sorted(((r, readings[r]) for r in probed
                       if readings.get(r) is not None), key=lambda rt: rt[1])
    if len(decisive) < 2 or spare_jobs < 1:
        return []
    (win, _), (lose, _) = decisive[0], decisive[1]
    delta, is_decisive = pair_outcome(decisive[0][1], decisive[1][1], floor)
    if is_decisive or delta is None:
        return []
    return [win, lose]
