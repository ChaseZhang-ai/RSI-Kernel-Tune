"""Structured inner-loop trajectory: status.jsonl (machine SSOT) + Layer-1 digest.

Each iteration appends one IterationRecord. The record holds OBJECTIVE facts only
— the optimization technique applied and the measured outcome (correctness, perf,
NCU) — for BOTH kept and reverted iterations. The inner agent's SUBJECTIVE analysis
(why it expected a gain, its bottleneck theory) is deliberately NOT recorded, so
the meta agent diagnoses from measured hardware truth rather than the student's
prose. build_digest() assembles the operator-agnostic Layer-1 digest (design §4.1bis).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class IterationRecord:
    iter: int
    decision_id: str
    change: str            # objective: the optimization technique / code change applied
    correctness: bool | None  # None = not evaluated (a scaffold-phase NEUTRAL row)
    verdict: str           # KEEP | REVERT | NEUTRAL (NEUTRAL = committed, not evaluated)
    bwd_median_ms: float | None = None
    mfu: float | None = None
    # In-job reading of THE declared reference — the single-ref contract: whatever
    # library rides the eval job to be timed back-to-back with the student must BE
    # this run's declared ref, never a stand-in (the retired "instrument"/ruler
    # concept: a second library timed only to calibrate node speed. It made every
    # vs_* number ambiguous — a fwd kernel's 1.0418x read as "beat FA" when the
    # denominator was cuDNN). Runs whose ref cannot ride the job (blackbox: the ref
    # package is readable source) time only the student, and these stay None —
    # ratios for such runs are median-vs-median across jobs, never same-job.
    # When present, the same-job ratio is what cross-round comparisons are read
    # from: absolute latency for one unchanged commit moves ~6% between jobs
    # (node bimodality), wider than most single-round gains.
    ref_median_ms: float | None = None
    vs_ref: float | None = None
    # Second timing window, when the harness measures one (see EvalResult in
    # loop/parse.py for the full account). bwd_median_ms is a per-call window and
    # includes host work; device_median_ms is a pipelined batch window ≈ device time.
    # The two sides of a ratio can pay different host taxes, so vs_ref (e2e) is a
    # call-latency claim while vs_ref_device is the kernel-quality claim. None when
    # the harness only times one window.
    device_median_ms: float | None = None
    ref_device_ms: float | None = None
    vs_ref_device: float | None = None
    # Precondition of the device window (see EvalResult in loop/parse.py): how long
    # the host alone takes to enqueue one call, and that over the device reading.
    # Recorded per iteration so the criterion lives in the trajectory the meta agent
    # and the digest read, not only in a job log nobody re-opens.
    host_enqueue_ms: float | None = None
    host_bound_ratio: float | None = None
    ref_host_bound_ratio: float | None = None
    # What one call of the in-job reference costs to enqueue (WP2 parsed it into
    # EvalResult; WP9 records it) — the ref-side host tax's evolution over the run,
    # a thick python wrapper showing up as drift the device window never explains.
    # Kept out of the digest (the teacher reads it off the row like
    # ref_host_bound_ratio).
    ref_host_enqueue_ms: float | None = None
    # The basis device_median_ms was read on (plain_batch / cuda_graph /
    # cupti_kernel_span). host_bound_ratio is None BY DESIGN under a host-free
    # basis, and a reader of the trajectory needs this to tell "unmeasured" from
    # "cannot apply"; None = recorded before the field existed, or a bench that
    # did not declare one.
    device_window_basis: str | None = None
    metric_before: float | None = None
    error: str = ""        # objective failure signature (compile error / correctness diff) for REVERTs
    profiler: dict = field(default_factory=dict)  # NCU key metrics — recorded for ALL iters incl. REVERT
    sha: str = ""          # commit sha of the landed iteration (for prefix-reuse resume); "" if none
    # Which GPU machine the eval job ran on (backend.last_cluster at eval time). One
    # kernel's absolute latency moves more between machines than most
    # single-round gains, so every absolute-ms comparison the driver makes —
    # best, plateau, patch judgment, target — reads same-machine rows only.
    # "" = recorded before the field existed, or a backend that cannot say;
    # the driver takes such rows to be from the run's own machine.
    cluster: str = ""
    # HEAD the student built this iteration on. Lineage, not order: after a
    # reset / meta reopen the next iteration's base is an earlier tree, and the
    # axis harvest pairs a KEEP with its ancestor's census, not its neighbour's.
    base_sha: str = ""
    # The tree's SASS contract axes (orch_tools summary: sync topology, in-flight
    # depth, supply order, cache-policy bits) off the same NCU job, when the
    # per-iteration census ran. Kept OUT of ``profiler`` so the Layer-1 digest
    # stays flat; the harvest reads it to bill a KEEP that moved an axis.
    orch_summary: dict = field(default_factory=dict)
    # Every kernel the student's profiled window launched (inner.ncu.kernel_table:
    # name / launches / time_ns / pct), off the same NCU job as ``profiler``. Kept
    # OUT of ``profiler`` (flat digest); read by the timed-window contract
    # (forgekernel.window_contract) to price the launches that exist only to
    # rebuild a reference-side input, and shown to the teacher next to the
    # reference's table. [] = recorded before the field existed / no NCU.
    kernels: list = field(default_factory=list)
    # The bench's window-outside prep step (eval JSON ``performance.prep``): what
    # it produced (name / shape / dtype, already held to window.toml by the
    # driver) and what it cost. Banked so the outside step is a reading on the
    # teacher's ledger, never a place work can be moved into unseen. {} = no
    # prep step / recorded before the field existed.
    prep: dict = field(default_factory=dict)
    # Multi-workload cards: the readings above are the geometric mean over
    # ``n_workloads`` workloads of one fixed shape; ``workloads`` keeps the
    # per-workload readings (see EvalResult in loop/parse.py). 1 / [] for a
    # single-input bench and for rows recorded before the field existed.
    n_workloads: int = 1
    workloads: list = field(default_factory=list)
    # A row whose tree is the tree of an earlier evaluated row (nothing changed
    # outside notes/, or a gate the delivery itself closed) carries THAT row's
    # reading instead of buying a fresh eval: ``reading_of`` is the iter of the
    # row whose job produced every measured field above. None = this row's own
    # job. A re-measure of an unchanged tree is one more noise draw — kept as a
    # reading it lets the best be set by luck (see loop/driver.py
    # _unchanged_tree_reading).
    reading_of: int | None = None
    # The precision gate's reading of this row's eval (loop/precision_gate.summarize):
    # contract, status (pass / fail / unavailable / unreported), the worst
    # e_cand / e_ref and the stress profiles' statuses. {} = contract off, no eval, or
    # recorded before the field existed.
    precision: dict = field(default_factory=dict)
    # Run-level spread of the raw readings behind the gate currency (WP1): the
    # mirrored device readings' (max - min) / min of this row's own job — one number
    # for "how loud was this reading", so a best set on a quiet draw is
    # distinguishable from one set on a noisy one. None = no raw readings reported
    # (plain measure_windows bench, or a row from before the field existed). The
    # per-workload spreads live in the ``workloads`` blocks.
    gate_spread_pct: float | None = None
    # The job's GPU state around the timed windows (eval JSON top-level
    # ``gpu_state``: sm/mem clock, temperature, power — timing.py's telemetry,
    # per-field means). {} = the bench reported none, or a row from before the
    # field existed. The evidence a cross-job comparison reads before it treats
    # two numbers as one measurement condition.
    gpu_state: dict = field(default_factory=dict)
    # Set by the driver when this row's SM clock sat >2% off the sitting best's at
    # reading time (loop/driver.py _CLOCK_DRIFT_BAND): the two rows are not one
    # measurement condition, and this row must not take the watermark on its own —
    # it goes through the suspicious-watermark confirmation first.
    clock_drift: bool = False
    # WP4: a KEEP slower than the best, +x%
    regression_vs_best_pct: float | None = None
    # WP9: the row's fresh_recheck FAILED — the timed calls did not compute what
    # correctness measured on the same instance (a state/pointer-keyed cache).
    # Set by the driver on the REVERT it records for that reading; the objective
    # flag travels the trajectory so a later resume cannot mistake the row for an
    # ordinary regression. Digest stays row-free (the teacher reads it on the row).
    suspicious: bool = False
    # --- WP11 (task 4): wall-clock composition telemetry -------------------------
    #
    # Where an iteration's wall clock actually went: the student's agent turns, the
    # eval / NCU GPU jobs (submit -> terminal), the teacher's turns that led into
    # this iteration, and the time the job sat waiting for the GPU. Measurement
    # only — nothing reads these fields for a verdict, a budget or a stop; the
    # run-end composition table is their only consumer. All None by default (a row
    # recorded before the field existed, or a channel that cannot say — e.g.
    # queue wait is only observable on the local delivery, and a scaffold row
    # buys no job). Coexists with the WP1/WP2/WP4 field additions by the same
    # append-only rule: load_records drops unknown keys of older files.
    #
    # Merged-job口径 (WP11 rework note): on the default merged path eval_job_s
    # carries the WHOLE merged job's wall (eval segment + profile segment — the
    # two share one unpack/build, so a per-segment split of it would be
    # fiction), and ncu_job_s carries the profile segment's own wall as the
    # entry script echoes it at the segment boundary. The buckets therefore do
    # NOT sum to the merged job (eval_job_s already includes the profile
    # segment); the composition table reads the profile share from ncu_job_s
    # and everything else from eval_job_s. On the two-job path each field is
    # its own job's wall, as always.
    agent_turn_s: float | None = None
    eval_job_s: float | None = None
    ncu_job_s: float | None = None
    teacher_s: float | None = None
    queue_wait_s: float | None = None
    # WP10 MC-DELIVERY: the objective per-variant table of a multi-candidate
    # delivery (the driver's prescreen: which student-written variant passed the
    # correctness smoke, its latency readings, which one became the delivery and
    # took the full gate). The driver generates it; like the WP1 fields above it
    # is an objective measurement record of the student's own work, not a
    # teacher-curated message. None = no multi-candidate delivery this row (or a
    # row from before the field existed).
    variant_timings: list | None = None
    # --- WP14 (shape-regime decision channel) ------------------------------------
    # The per-shape ledger (contract C8: THIS is the schema's single definition —
    # every consumer reads these fields, only trajectory.py defines them). A group
    # card's gate reading is a geomean over workloads of one kernel, so "小 shape
    # 结构 A、大 shape 结构 B" is a combination the aggregate can never name: the
    # ledger gives each workload its own measured history.
    # per_shape_ratios: this row's per-workload vs-reference ratios keyed by the
    # workload's name ({workload: ratio}), the same-job currency the pricing table
    # reads (vs_ref_device, falling back to vs_ref). {} for a single-input bench
    # (the top-level fields already are that one shape's reading) and for rows
    # recorded before the field existed.
    # per_shape_best: the running historical best per workload as of this row —
    # {workload: {"ms": gate-window best ms, "iter": the row that set it}} —
    # inherited across rows and moved only by a KEEP whose workload reading beats
    # the incumbent (a REVERT row inherits unchanged). Snapshot arithmetic lives in
    # per_shape_ledger() below; the read-side recompute is per_shape_best_readings().
    per_shape_ratios: dict = field(default_factory=dict)
    per_shape_best: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.verdict not in ("KEEP", "REVERT", "NEUTRAL"):
            raise ValueError(f"invalid verdict: {self.verdict!r}")


_FIELDS = {f.name for f in fields(IterationRecord)}

# Pre-rename key names (the in-job reference was recorded under cublas_* whatever
# library it actually was). Mapped on load so resuming an older run keeps its
# ratio history; written files use the new names only.
_LEGACY_KEYS = {"cublas_median_ms": "ref_median_ms", "vs_cublas": "vs_ref",
                "cublas_device_ms": "ref_device_ms", "vs_cublas_device": "vs_ref_device"}

# The two timing windows a bench can measure (see EvalResult in loop/parse.py for
# the full account). "e2e" is the per-call window — host work lands inside it —
# and has been the gate since the series began; "device" is the pipelined batch
# window ≈ device time. Which one drives the loop's verdicts (best, target stop,
# plateau, rollback anchor) is the operator's gate_window, chosen once per run:
# gating a training-shaped kernel on host-inclusive latency optimizes python
# plumbing the deployment would never pay, and gating an inference-shaped one on
# device time hides a host tax the caller pays on every call. Neither window is
# right in general, so the choice is config, not a constant — but the DEFAULT is
# device: the benchmark question is kernel quality, the
# claim every reference comparison and cross-run ranking is read from. The e2e
# window remains available per run (--gate-window / loop.toml) for kernels whose
# deployment pays the host tax on every call. The default is deliberately loud on
# under-equipped runs: a baseline archive or bench that never measured the device
# window refuses to start / stops as gate_unmeasured rather than silently gating
# on the other window.
GATE_WINDOW_E2E = "e2e"
GATE_WINDOW_DEVICE = "device"
GATE_WINDOWS = (GATE_WINDOW_E2E, GATE_WINDOW_DEVICE)
GATE_WINDOW_DEFAULT = GATE_WINDOW_DEVICE


def gate_ms(rec, gate_window: str = GATE_WINDOW_DEFAULT) -> float | None:
    """The record's reading in the gate window; None when that window was not measured.

    Works on anything carrying ``bwd_median_ms``/``device_median_ms`` (IterationRecord,
    EvalResult). None under gate_window="device" is a bench-contract violation for an
    evaluated KEEP — the driver refuses it loudly rather than letting every verdict
    silently degrade to "no reading" (a device-gated loop where the bench only times
    e2e would never hit target and never plateau).
    """
    if gate_window not in GATE_WINDOWS:
        raise ValueError(f"gate_window must be one of {GATE_WINDOWS}, got {gate_window!r}")
    if gate_window == GATE_WINDOW_DEVICE:
        return rec.device_median_ms
    return rec.bwd_median_ms


def append_record(path: str | Path, rec: IterationRecord) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


# --- WP14: the per-shape ledger's arithmetic (single definition, contract C8) ----
#
# A group card's gate currency is the geomean over its workloads, and until now the
# per-workload readings rode each row's ``workloads`` blocks with no account that
# said which row was ever the best FOR EACH SHAPE. per_shape_ledger is the one
# writer-side definition of that account; per_shape_best_readings is the one
# reader-side definition (a recompute from the KEEP rows' own workload blocks, so
# the digest never trusts a snapshot it can recompute).


def _wl_gate_ms(w: dict, gate_window: str) -> float | None:
    """One workload block's reading in the gate window (None when unmeasured)."""
    if gate_window == GATE_WINDOW_DEVICE:
        v = w.get("device_median_ms")
    elif gate_window == GATE_WINDOW_E2E:
        v = w.get("bwd_median_ms")
    else:
        raise ValueError(f"gate_window must be one of {GATE_WINDOWS}, got {gate_window!r}")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _wl_ratio(w: dict) -> float | None:
    """One workload block's same-job vs-reference ratio (the pricing table's
    currency: vs_ref_device, falling back to vs_ref)."""
    v = w.get("vs_ref_device", w.get("vs_ref"))
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def per_shape_ledger(records: list, workloads: list, iter_no: int, *, improve: bool,
                     gate_window: str = GATE_WINDOW_DEFAULT) -> "tuple[dict, dict]":
    """(per_shape_ratios, per_shape_best) for the row about to be appended.

    ``workloads`` is the eval's per-workload blocks (IterationRecord.workloads
    schema). The ratios are this row's own objective readings; the best snapshot
    inherits the newest row that carries one, and — only when ``improve`` (a KEEP)
    — moves a workload's entry whose gate-window reading beats the incumbent.
    ([] workloads -> ({}, {}) : the single-input shape is the top-level fields.)
    """
    if not workloads:
        return {}, {}
    ratios = {}
    for w in workloads:
        name, r = str(w.get("name", "?")), _wl_ratio(w)
        if r is not None:
            ratios[name] = r
    prev: dict = {}
    for r in reversed(records):
        if r.per_shape_best:
            prev = dict(r.per_shape_best)
            break
    best = prev
    if improve:
        best = {k: dict(v) for k, v in prev.items()}
        for w in workloads:
            ms = _wl_gate_ms(w, gate_window)
            if ms is None:
                continue
            name = str(w.get("name", "?"))
            cur = best.get(name)
            if cur is None or not isinstance(cur.get("ms"), (int, float)) or ms < cur["ms"]:
                best[name] = {"ms": ms, "iter": int(iter_no)}
    return ratios, best


def per_shape_best_readings(records: list,
                            gate_window: str = GATE_WINDOW_DEFAULT) -> dict:
    """{workload: {"ms", "iter"}} — each workload's historical best, recomputed
    from the KEEP rows' own workload blocks (the read-side definition; a run of
    pre-WP14 rows is accounted too, from the same blocks the digest already read).
    Workloads whose readings are missing from every KEEP row are absent."""
    out: dict = {}
    for r in records:
        if r.verdict != "KEEP":
            continue
        for w in r.workloads or ():
            ms = _wl_gate_ms(w, gate_window)
            if ms is None:
                continue
            name = str(w.get("name", "?"))
            cur = out.get(name)
            if cur is None or ms < cur["ms"]:
                out[name] = {"ms": ms, "iter": int(r.iter)}
    return out


def load_records(path: str | Path) -> list[IterationRecord]:
    """Load records, dropping any unknown keys (tolerates pre-refactor status files)."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            d = json.loads(line)
            d = {_LEGACY_KEYS.get(k, k): v for k, v in d.items()}
            out.append(IterationRecord(**{k: v for k, v in d.items() if k in _FIELDS}))
    return out


# A correctness FAIL whose per-tensor max_diff repeats BIT-IDENTICALLY across
# distinct failing attempts means the scored outputs never changed — the edits
# are not reaching the computation that produces them (outputs still hold their
# init constants: a broken inter-kernel handoff, a writeback that never lands, a
# stale compile cache). Real numeric bugs move when the code moves; four attempts
# in a row landing on 4.504025/7.560892/7.823140 to the last digit is not a
# numeric bug, and no one — student, meta, or log reader — was told (observed on
# an MHA bwd run: four such iterations were only diagnosed by hand afterwards).
_DIFF_SIG = re.compile(r"max_diff=([0-9]+\.[0-9]+|inf|nan)")
CONSTANT_OUTPUT_MIN_REPEATS = 3


def constant_output_flag(records: list[IterationRecord]) -> dict | None:
    """The trailing streak of REVERTs whose diff signatures are bit-identical.

    Only correctness-FAIL records that report max_diff values participate; rows
    without a diff (probe-blocked, build errors, missing eval) neither extend nor
    break the streak — the observed pattern interleaved both. Returns None below
    CONSTANT_OUTPUT_MIN_REPEATS identical signatures.
    """
    streak: list[int] = []
    sig: tuple[str, ...] | None = None
    for r in reversed(records):
        if r.verdict == "KEEP":
            break  # a KEEP means outputs demonstrably respond to the code
        if r.verdict != "REVERT" or not r.error:
            continue
        found = tuple(_DIFF_SIG.findall(r.error))
        if not found:
            continue
        if sig is None:
            sig = found
        if found != sig:
            break
        streak.append(r.iter)
    if sig is None or len(streak) < CONSTANT_OUTPUT_MIN_REPEATS:
        return None
    return {
        "kind": "constant_output",
        "iters": sorted(streak),
        "signature": list(sig),
        "hint": ("{} consecutive correctness FAILs have a bit-identical max_diff — the scored "
                 "output does not respond to these changes at all, so this is most likely not a "
                 "numerical bug: the output is still the initialization constant (a broken "
                 "hand-off of intermediates between kernels / a write-back that never lands / a "
                 "compile cache pointing at old code). First verify that \"the end-to-end output "
                 "is not constant and changes with the computation\", then fix the numerics."
                 .format(len(streak))),
    }


# The SOL header's fields (WP2 telemetry): the roofline readings the diagnosis
# reads FIRST, whichever the newest iteration that measured them carries. Only
# present fields are emitted — records profiled by a card without the expanded
# metric set (or an arch the query filter dropped names on) contribute nothing,
# and the header is {} rather than a row of nulls.
_SOL_HEADER_KEYS = ("compute_sol_pct", "memory_sol_pct", "l2_hit_rate_pct",
                    "waves_per_sm", "tensor_pct", "dram_pct")


def sol_header(records: list[IterationRecord]) -> dict:
    """The newest iteration's SOL readings as the digest's top line ({} when none)."""
    for r in reversed(records):
        prof = r.profiler or {}
        head = {k: prof[k] for k in _SOL_HEADER_KEYS if prof.get(k) is not None}
        if head:
            return {"iter": r.iter, **head}
    return {}


def build_digest(records: list[IterationRecord], target: dict,
                 gate_window: str = GATE_WINDOW_DEFAULT) -> dict:
    """Assemble the Layer-1 digest (target/achieved/curve/decisions/regressions/profiler).

    Objective-only: decisions carry the technique + measured metrics; regressions
    carry the objective failure signature; profiler carries NCU for every iteration
    that has it (KEEP and REVERT), tagged with the verdict.

    ``gate_window`` picks which timing window "best" and the target gap are read
    from — the digest must agree with the driver's verdicts, or the meta agent
    diagnoses a different run than the one that happened. Both windows' readings
    stay in the rows regardless; the window only decides which one ranks.
    """
    kept = [r for r in records
            if r.verdict == "KEEP" and gate_ms(r, gate_window) is not None]
    best = min(kept, key=lambda r: gate_ms(r, gate_window)) if kept else None
    achieved = None
    if best is not None:
        tv = target.get("value")
        best_ms = gate_ms(best, gate_window)
        achieved = {
            "bwd_median_ms": best.bwd_median_ms,
            "device_median_ms": best.device_median_ms,
            "gate_window": gate_window,
            "mfu": best.mfu,
            "iter": best.iter,
            "gap_pct": (best_ms - tv) / tv * 100 if tv else None,
        }
        if best.workloads:
            # the gate reading is a geomean: show which workload carries or drags it.
            # Median + spread only — the raw per-rep arrays stay in the trajectory
            # (the digest is the agents' context, and an array per workload per
            # iteration buys no decision they can make).
            achieved["aggregate"] = f"geomean over {best.n_workloads} workloads"
            achieved["workloads"] = [{k: v for k, v in w.items()
                                      if "readings" not in k} for w in best.workloads]
    out = {
        # TOP of the digest: the SOL roofline header (WP2 telemetry) — the diagnosis's
        # first reading. Which side of the machine saturates first (compute vs memory
        # SOL), is the grid tail-bound (waves < 1), does a working set that should sit
        # in L2 keep missing (l2_hit_rate_pct). From the NEWEST iteration that measured
        # them (KEEP and REVERT alike); empty until a card/NCU reports the new metrics
        # — records from before the telemetry expansion leave it {}, not missing.
        "sol": sol_header(records),
        "target": target,
        "achieved": achieved,
        # WP14 per-shape ledger: each workload's historical best (gate window), the
        # account a geomean cannot show. {} until some KEEP row carried per-workload
        # readings (a single-input bench stays {} — the top-level fields are that
        # one shape's history).
        "per_shape_best": per_shape_best_readings(records, gate_window),
        "curve": [{"iter": r.iter, "bwd_median_ms": r.bwd_median_ms,
                   **({"device_median_ms": r.device_median_ms}
                      if r.device_median_ms is not None else {}),
                   **({"cluster": r.cluster} if r.cluster else {}),
                   "verdict": r.verdict}
                  for r in records],
        # metric_before is the gate-window best at entry (the driver writes it that
        # way), so metric_after must read the same window or the delta is fiction.
        "decisions": [{"iter": r.iter, "id": r.decision_id, "change": r.change,
                       "correctness": r.correctness, "metric_before": r.metric_before,
                       "metric_after": gate_ms(r, gate_window), "verdict": r.verdict}
                      for r in records],
        "regressions": [{"id": r.decision_id, "error": r.error or "regression", "iter": r.iter}
                        for r in records if r.verdict == "REVERT"],
        "profiler": [{"iter": r.iter, "verdict": r.verdict, **r.profiler}
                     for r in records if r.profiler],
        # red flags derived from the trajectory shape itself; empty list when clean
        "flags": [f for f in (constant_output_flag(records),) if f],
    }
    return out
