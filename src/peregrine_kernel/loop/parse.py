"""Deterministic parsers for the two-layer driver.

The driver shells out to an agent for each student/meta turn and to the GPU
harness for each eval/probe. These pure functions turn that free-form text into
typed decisions the orchestration core can branch on — so the loop logic is
unit-testable without an agent or a GPU.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from typing import NamedTuple
from dataclasses import dataclass, field

from peregrine_kernel.inner.trajectory import GATE_WINDOW_DEFAULT, GATE_WINDOW_DEVICE, gate_ms


@dataclass
class Record:
    """A student's completed decision — OBJECTIVE fields only (the technique + file).

    The student's subjective analysis (hypothesis, bottleneck theory) is not
    parsed / not recorded; the meta agent reasons from measured NCU truth instead.
    """
    decision_id: str
    change: str
    files_touched: str = ""


@dataclass
class ProbeRequest:
    """The student wants a GPU layout-probe run before committing the decision.

    ``manifest=True``: ``path`` names a PROBE-MANIFEST file (WP19 task 2) —
    one ≤4-line list of self-contained probe scripts the driver runs in a
    single job, each with its own verdict. The student-side submit surface
    never grows this shape: only the meta-step / KB-certification loops read
    it."""
    path: str
    manifest: bool = False


@dataclass
class ReconRequest:
    """The teacher wants a reconnaissance script run against the reference.

    Distinct from ProbeRequest because it runs in a different tree and answers a
    different question. A probe validates the student's own code; a recon script
    measures the *reference* — profiling it, disassembling it, or computing against
    hardware limits — and it runs in the teacher-only submission tree so that neither
    the script nor what it learns can reach a student job.
    """
    path: str


@dataclass
class Exhausted:
    """No further phase-order decision is available in the milestone."""
    reason: str


StudentTurn = Record | ProbeRequest | ReconRequest | Exhausted

_FIELD = re.compile(r"^\s*([a-z_]+)\s*:\s*(.*\S)\s*$")


_PATH_LWRAP = "`'\"*([<"
_PATH_RWRAP = "`'\"*)]>,;:."


def _clean_path(tok: str) -> str:
    """A marker's path with the markdown an agent wraps it in stripped: the
    ``\\S+`` capture keeps a closing backtick / quote / paren, and that one
    character reaches the GPU machine's ``bash -c`` as an unmatched quote — the job
    dies before its begin marker, which the ssh runner reads as a transport
    failure, for a path that never existed."""
    return tok.lstrip(_PATH_LWRAP).rstrip(_PATH_RWRAP)


def parse_student_turn(text: str, *, allow_recon: bool = False) -> StudentTurn:
    """Classify a student agent's final message into the three terminal shapes.

    PROBE_REQUEST / EXHAUSTED take precedence (they are explicit single-line
    markers — matched only at the start of a line, so a marker the turn merely
    echoes mid-sentence — quoting the protocol, narrating it, or quoting a
    reminder — does not classify the turn; an unanchored scan sent an echoed
    marker to the probe branch and burned a GPU job on it, and an echoed
    EXHAUSTED ended the run). Otherwise the record block fields are extracted
    from anywhere in the text (agents often emit prose before the block).

    RECON_REQUEST is only a valid shape for the reconnaissance caller
    (``allow_recon=True``). Recognizing it everywhere let a student turn that
    merely echoed the marker — likely in a blackbox run, where the vocabulary
    saturates the prompts — reach the probe branch and burn a GPU job on a
    path that exists only in the teacher's tree.
    """
    probe = re.search(r"^\s*PROBE_REQUEST:\s*(\S+)", text, re.MULTILINE)
    if probe:
        return ProbeRequest(path=_clean_path(probe.group(1)))
    # WP19 (task 2): a PROBE-MANIFEST names a ≤4-line list of self-contained
    # probe scripts for one batched job. Parsed AFTER PROBE_REQUEST (an
    # explicit single-script request always wins) and before the record
    # fields — the manifest is a meta-side vocabulary; a student turn is not
    # expected to carry it, and a stray echo falls through to the ordinary
    # shapes exactly like a stray RECON_REQUEST does.
    manifest = re.search(r"PROBE-MANIFEST:\s*(\S+)", text)
    if manifest:
        return ProbeRequest(path=_clean_path(manifest.group(1)), manifest=True)
    if allow_recon:
        recon = re.search(r"RECON_REQUEST:\s*(\S+)", text)
        if recon:
            return ReconRequest(path=_clean_path(recon.group(1)))
    # Line-anchored like PROBE_REQUEST above: the protocol puts the marker on its
    # own line ("End your turn with EXACTLY one of: ... 'EXHAUSTED: <reason>'"),
    # and prose that merely mentions the phrase must not stop the run.
    exhausted = re.search(r"^\s*EXHAUSTED:\s*(.+)", text, re.MULTILINE)
    if exhausted:
        return Exhausted(reason=exhausted.group(1).strip())

    fields: dict[str, str] = {}
    for line in text.splitlines():
        m = _FIELD.match(line)
        if m and m.group(1) in ("decision_id", "change", "files_touched"):
            fields.setdefault(m.group(1), m.group(2))
    if "decision_id" not in fields:
        raise ValueError("student turn is neither a record, a probe request, nor exhausted")
    return Record(
        decision_id=fields["decision_id"],
        change=fields.get("change", ""),
        files_touched=fields.get("files_touched", ""),
    )


@dataclass
class EvalResult:
    correctness: bool
    bwd_median_ms: float | None = None
    mfu: float | None = None
    # In-job reading of THE declared reference, when the run's ref can ride the eval
    # job (single-ref contract — see IterationRecord in inner/trajectory.py). Absolute
    # latency for an unchanged commit moves ~6% between jobs — wider than most
    # single-round gains — so the absolute is partly a reading of which node the round
    # landed on; the same-job ratio holds across that swing. None when the ref cannot
    # ride the job (blackbox runs time only the student — no stand-in library).
    ref_median_ms: float | None = None
    vs_ref: float | None = None
    # Second timing window, when the harness measures one. bwd_median_ms is a per-call
    # window (record/call/record/sync each rep): the stream starts idle, so host work —
    # launch latency, python wrappers — lands inside it. device_median_ms is a pipelined
    # batch window (one record pair around N back-to-back calls, divided by N): the
    # stream never drains, host cost is hidden, and what remains is ≈ device time.
    # Observed live: the same cuDNN kernel reads 0.28 ms in the first window and 0.25 ms
    # in the second, and the two sides of a ratio can pay *different* host taxes (a
    # python ctx manager on the library side vs a thin pointer path on the student's) —
    # so an e2e ratio is not a kernel-quality ratio. The gate stays on the e2e window;
    # this pair exists so kernel-vs-kernel claims can be read from matching windows.
    device_median_ms: float | None = None
    ref_device_ms: float | None = None
    vs_ref_device: float | None = None
    # The device window's PRECONDITION: it equals device time only while the host can
    # enqueue faster than the device drains. host_enqueue_ms times the host alone (N
    # calls, no synchronize in between); host_bound_ratio is that over the device
    # reading. Near 1 means the pipeline never filled and the "device" number is
    # python dispatch cost. Parsed and recorded rather than left in the job log —
    # a criterion that never reaches the trajectory cannot gate anything and cannot
    # be fed back to the student, which is the same as not having written it.
    host_enqueue_ms: float | None = None
    host_bound_ratio: float | None = None
    # Same check for the in-job reference, when the bench times one. A host-bound
    # REFERENCE is a measurement defect rather than a kernel fact, and it invalidates
    # vs_ref_device in the reference's favour-losing direction.
    ref_host_bound_ratio: float | None = None
    # The basis the device reading was taken on (timing.py's device_window_basis:
    # plain_batch / cuda_graph / cupti_kernel_span). Decides whether the enqueue
    # clock describes the reading at all — under a host-free basis it does not, and
    # the host-bound audit must not divide the two (on dsa_sparse: three rounds of
    # "your reading is inflated" against a CUPTI span that cannot contain host time).
    device_window_basis: str | None = None
    # What one call of the in-job reference costs to enqueue, when the bench timed
    # it: the comparison a thick student wrapper is reported against (as a
    # deployment cost, never as a gate) under a host-free basis.
    ref_host_enqueue_ms: float | None = None
    # The bench's own words for a correctness fail (forbidden-op verdict, bad-shape
    # detail, import traceback). A bare "correctness FAIL" sent four agents in a row
    # chasing phantom numeric bugs while the real verdict — a scanner match on the
    # entry symbol — sat unread in the job log; the student and the meta diagnoser
    # can only be as right as the failure text they are shown.
    error: str = ""
    # The bench's window-outside prep step, when the timed call receives
    # symmetrized inputs (timing.measure_windows(fn, prep=...)): ``outputs`` =
    # {name: {shape, dtype}} of what it produced, plus its own median_ms /
    # device_median_ms. The driver holds ``outputs`` to window.toml
    # (peregrine_kernel.window_contract.check_prep_report) and banks the block on the
    # iteration record. None when the bench ran no prep step.
    prep: dict | None = None
    # Multi-workload cards (timing.measure_windows_workloads): the top-level
    # readings above are the geometric mean over ``n_workloads`` workloads of one
    # fixed shape, and ``workloads`` keeps the per-workload readings compact —
    # [{name, device_median_ms, bwd_median_ms, ref_device_ms?, vs_ref_device?,
    # vs_ref?}] — for the log line, the teacher's digest and the confirmation
    # record. Empty and 1 for a single-input bench.
    n_workloads: int = 1
    workloads: list = field(default_factory=list)
    # The eval JSON's top-level ``precision`` block (src/peregrine_kernel/card_template/eval/precision.py:
    # kernel vs reference error against a float64 truth). The bench never folds it into
    # ``correctness``; the driver applies the card's [loop] precision_contract
    # (loop/precision_gate.py). None = the bench reported none.
    precision: dict | None = None
    # The raw device readings behind device_median_ms — the mirrored pair of per-window
    # medians a paired bench reports (timing.py ``device_readings_ms``), and the
    # reference's own pair. The median alone threw the spread away: a 0.8% "win" whose
    # two raw readings sit 2% apart is not the same fact as one whose pair agrees to
    # 0.1%, and nothing downstream could tell. [] = a bench that reports no raw
    # readings (plain measure_windows, or a record from before they rode the JSON).
    device_readings_ms: list = field(default_factory=list)
    ref_device_readings_ms: list = field(default_factory=list)
    # The eval JSON's top-level ``gpu_state`` block (timing.py's clock / thermal /
    # power telemetry around the windows). None = the bench reported none.
    gpu_state: dict | None = None
    # Set by the driver (never the bench): the reading's SM clock moved >2% from the
    # sitting best's, so the two are not one measurement condition — see
    # loop/driver.py _CLOCK_DRIFT_BAND.
    clock_drift: bool = False
    # WP9: the timing-fresh-input recheck (eval/benchkit.py, run after ALL
    # timing windows close): {"mode": "process"|"regen", "pass": bool|None,
    # "layers": [...]}. pass=False — the timed calls did not compute what
    # correctness measured on the same instance — is a reward-hack reading the
    # driver turns into a suspicious REVERT; None layers are 'no reading'.
    fresh_recheck: dict | None = None
    # WP15 (N-B14, contract C14): the job's SPRT early-stop decision, when the
    # bench armed the sequential controller and it DECIDED: {"decided": <windows>,
    # "direction": "loss"|"win", "windows": <the sides' full budget>}. A
    # loss-direction stop is the WP11 rule (the reading cannot be a win);
    # a WIN-direction stop is the tail task — its reading is one truncated draw
    # that beat the margin, so the driver must run it through the standing
    # watermark-confirmation protocol before it can take the watermark, enter
    # the plateau window or price a lever. None = no decision (full sampling,
    # or SPRT not armed).
    sprt: dict | None = None
    # --- WP20: measure-side defense accounting (readings, never verdicts) -----
    # The worst correctness error/tolerance ratio among the eval's checks: how
    # close to the judge's edge the worst output sat (>=1.0 fails). The edge-band
    # judge recheck and the corr_margin_ratio bookkeeping read this.
    corr_margin_ratio: float | None = None
    # The precision gate went quietly blind: unavailable rows beyond the card's
    # gate_degraded_ratio (the bench-side reading; the driver recomputes with the
    # loop.toml ratio when the precision block is present).
    gate_degraded: bool | None = None
    # The precision contract the card ran under, as the bench reported it
    # (`strict (default)` / `off (problem)` / `off (env: PK_PRECISION=0)`).
    precision_audit: str | None = None
    # The stream sentinel's post-window span audit ({"status": "ok"|"violation"|
    # "unknown", ...}): a timed call that drained far more than its event window
    # measured. A violation rides the fresh_recheck fail path (mode=stream).
    stream_sentinel: dict | None = None
    # `ok` | `mismatch` | `unknown`: the clock / window identities re-resolved
    # against the pre-import capture. A mismatch rides fresh_recheck (mode=timer).
    timer_integrity: str | None = None
    # The thread / child census ({"status": "ok"|"anomaly", ...}): accounting
    # only — threads or children appearing after the timed region, beyond the
    # first-call baseline.
    proc_census: dict | None = None
    # WP29: the replay probe's block ({"status": "pass"|"suspected"|
    # "unavailable", "ratio_threshold", "shapes": [...]}): after all windows
    # closed the bench measured one honest-compute latency probe per timed
    # group on FRESH input draws (a third seed) in a child process. status ==
    # "suspected" — the probe median dwarfs the judged median, i.e. the timed
    # calls did not recompute their inputs (the content-keyed replay whose
    # miss path recomputes honestly) — is a reward-hack reading the driver
    # turns into a suspicious REVERT; unavailable never convicts.
    replay_probe: dict | None = None
    # M3, the two-lane contract (docs/industrial-lane.md §6.4): the eval JSON's
    # lane stamp, MEMO cards only — {"memo": true}. None (absent) = STRICT; the
    # STRICT eval JSON stays byte-identical to pre-M3. Every MEMO reading
    # inherits the lane from this stamp; the lanes' numbers are never compared.
    lane: dict | None = None
    # M3: the MEMO lane's unseen-input hard gate (the WP9 fresh child, verdict
    # re-purposed — {"mode": "regen"|"process", "pass": bool|None, ...}): on a
    # fresh draw the timed stream never showed, the cache's miss path must still
    # answer correctly. pass=False is a hard correctness REVERT (never marked
    # suspicious — replaying timed content is legal in this lane); pass=None is
    # a no-reading (the WP9 posture: env failure never convicts). None on
    # STRICT rows (the fresh_recheck key carries the STRICT verdict instead).
    memo_fallback: dict | None = None
    # Set by the driver (never the bench): the edge-band judge recheck cleared —
    # `pass` on the fresh draw that replaced a failed edge-band reading. A field,
    # not an attribute, so a dataclasses.replace (the watermark confirmation's
    # median) carries it through.
    judge_edge_recheck: str | None = None


_WORKLOAD_KEYS = ("device_median_ms", "bwd_median_ms", "ref_device_ms", "vs_ref_device", "vs_ref")
# Raw per-rep readings and the host enqueue clock ride the workload blocks too (WP1):
# the medians were kept and the spread dropped, which made every workload's noise
# figure unrecoverable from the trajectory.
_WORKLOAD_LIST_KEYS = ("device_readings_ms", "ref_device_readings_ms")


def readings_spread_pct(readings) -> float | None:
    """``(max - min) / min`` over finite positive readings, as a fraction; None when
    fewer than two usable readings — an absent verdict, never a passing one."""
    vals = [float(v) for v in (readings or ())
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and float(v) > 0.0 and float(v) == float(v)]
    if len(vals) < 2:
        return None
    return (max(vals) - min(vals)) / min(vals)


def _finite_float(v) -> float | None:
    """A scalar reading, read the way the list fields read theirs: a real number,
    never a bool or a string, and finite. The list fields already demanded
    ``isinstance(v, (int, float)) and not isinstance(v, bool)``; the scalars used
    to take anything ``float()`` swallowed — a bench JSON that wrote
    ``"device_median_ms": true`` parsed as a 1.0 ms gate reading, and ``"2.5"``
    as 2.5, straight into best / plateau / watermark arithmetic."""
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
        return float(v)
    return None


def _workloads(perf: dict) -> list:
    """Compact per-workload readings from a multi-workload performance block."""
    out = []
    for w in perf.get("workloads") or ():
        if not isinstance(w, dict):
            continue
        item = {"name": str(w.get("name", len(out)))}
        # A NaN/inf reading is a malformed measurement, not a small one: like
        # readings_spread_pct below, drop it here — a NaN that reaches the
        # trajectory takes over every min() it lands in (a NaN best hijacks the
        # plateau comparison, so the meta loop can never fire).
        for key in _WORKLOAD_KEYS:
            v = w.get(key)
            if (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(float(v))):
                item[key] = float(v)
        for key in _WORKLOAD_LIST_KEYS:
            v = w.get(key)
            if isinstance(v, (list, tuple)):
                item[key] = [float(x) for x in v
                             if isinstance(x, (int, float)) and not isinstance(x, bool)
                             and math.isfinite(float(x))]
        host = w.get("host_enqueue_ms")
        if (isinstance(host, (int, float)) and not isinstance(host, bool)
                and math.isfinite(float(host))):
            item["host_enqueue_ms"] = float(host)
        # WP15: the workload's SPRT early-stop decision (timing.py's controller),
        # so the driver's card rule (a loss-stopped workload cannot acquit the
        # card) and the digest can read it off the row.
        if isinstance(w.get("sprt"), dict):
            item["sprt"] = dict(w["sprt"])
        spread = readings_spread_pct(item.get("device_readings_ms"))
        if spread is not None:
            item["device_spread_pct"] = spread
        out.append(item)
    return out


def _float_list(raw) -> list:
    """The finite numeric entries of ``raw`` as floats; [] for anything else (a bench
    that reports no raw readings, or a malformed one). NaN/inf are dropped like
    readings_spread_pct drops them — a NaN reading is a broken measurement, and one
    let into a readings list poisons the spreads and medians computed from it."""
    if not isinstance(raw, (list, tuple)):
        return []
    return [float(v) for v in raw
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v))]


def describe_workload_readings(workloads: list) -> str:
    """``name device-ms (xratio)`` per workload, for one log / digest line."""
    parts = []
    for w in workloads or ():
        ms = w.get("device_median_ms", w.get("bwd_median_ms"))
        text = f"{w.get('name')} {ms:.4f} ms" if isinstance(ms, (int, float)) else f"{w.get('name')} n/a"
        ratio = w.get("vs_ref_device", w.get("vs_ref"))
        if isinstance(ratio, (int, float)):
            text += f" (x{ratio:.4f})"
        parts.append(text)
    return "; ".join(parts)


# --- WP11 (task 5): the eval failure's error bandwidth ------------------------
#
# A failing eval used to reach the record as one vague sentence ("no eval JSON
# block in the job log (job died before the verdict?)") while the actual evidence
# — the nvcc/ptxas error, the import traceback, the killed-job marker — sat in the
# job log the parser was already holding. The verdict-probe channel has shipped a
# bounded log-tail protocol since long ago; this is the same protocol copied here.
# The constants mirror loop/driver.py's _VERDICT_PROBE_TAIL_LINES /
# _VERDICT_PROBE_TAIL_CHARS (the probe keeps its own copy first; these were raised
# to shared status by copying, per the probe's own file-boundary rule — this file
# ships to job images only through the driver, so the two must be kept in sync by
# this comment until a shared home is worth the churn).
EVAL_TAIL_LINES = 150
EVAL_TAIL_CHARS = 8000

_EVAL_TAIL_MARK = "--- job log tail (last {n} lines, oldest first) ---"


def job_log_tail(text: str, *, lines: int = EVAL_TAIL_LINES,
                 chars: int = EVAL_TAIL_CHARS) -> str:
    """The bounded tail of a job log, per the verdict-probe protocol: the last
    ``lines`` lines, then clipped to the last ``chars`` characters."""
    if not text:
        return ""
    tail = "\n".join(text.strip().splitlines()[-lines:])
    return tail[-chars:]


def _with_log_tail(error: str, text: str) -> str:
    """``error`` with the job log's bounded tail attached (WP11 task 5).

    Only failure paths call this. The tail is the student's own tree's job output
    — objective fact about its own job, no reference surface. A marker line keeps
    the tail tellable from the parser's own words, and an empty log attaches
    nothing (the error stays exactly what it was)."""
    tail = job_log_tail(text)
    if not tail:
        return error
    return f"{error}\n{_EVAL_TAIL_MARK.format(n=EVAL_TAIL_LINES)}\n{tail}"


def _last_json_object(text: str) -> dict | None:
    """The last JSON object carrying a 'correctness' or 'performance' key."""
    dec = json.JSONDecoder()
    found = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "{":
            try:
                obj, j = dec.raw_decode(text, i)
            except json.JSONDecodeError:
                i += 1
                continue
            if isinstance(obj, dict) and ("correctness" in obj or "performance" in obj):
                found = obj
            i = j
        else:
            i += 1
    return found


CORRECTNESS_FAIL_TOP_K = 3
"""How many of a numeric FAIL's worst check rows ride the feedback (WP15, A-G6:
the aggregate max_abs alone sent agents re-rounding the whole kernel when one
output on one input group carried the failure; the worst-k rows name the
locus)."""

_WORST_ELEMENTS_KEY = "worst_elements"
"""Reserved seam (WP15): when the bench reports per-element worst offenders
(``[{output, group, index, value, expected, abs_diff}, ...]`` — an emission
only the eval template can compute; outside this package's fence), the rows
ride the feedback verbatim ahead of the per-check summary. Absent today, so
the feedback ranks the per-(group, output) check rows the JSON already
carries."""


def _correctness_fail_detail(corr_obj: dict) -> str:
    """Top-k worst rows of a numeric FAIL (WP15 A-G6): element-level rows when
    the bench emitted them (the reserved seam above), else the per-(group,
    output) check rows ranked by normalized error with the tolerance each
    needed — the failing locus and the gap, not one aggregate number."""
    rows = corr_obj.get(_WORST_ELEMENTS_KEY)
    if isinstance(rows, list) and rows:
        parts = []
        for r in rows[:CORRECTNESS_FAIL_TOP_K]:
            if not isinstance(r, dict):
                continue
            where = f"{r.get('group', '?')}/{r.get('output', '?')}"
            idx = f"[{r.get('index')}]" if r.get("index") is not None else ""
            parts.append(f"{where}{idx}: got {r.get('value')} expected {r.get('expected')} "
                         f"(abs_diff {r.get('abs_diff')})")
        if parts:
            return "worst elements — " + "; ".join(parts)
    checks = [c for c in corr_obj.get("checks") or () if isinstance(c, dict)]
    if not checks:
        return ""
    ranked = sorted(checks, key=lambda c: (float("inf") if c.get("error") is None
                                           else float(c["error"])), reverse=True)
    parts = []
    for c in ranked[:CORRECTNESS_FAIL_TOP_K]:
        where = "/".join(str(c.get(k)) for k in ("workload", "group") if c.get(k))
        loc = f"{where}/{c.get('output', '?')}" if where else str(c.get("output", "?"))
        err = c.get("error")
        tol = c.get("tolerance")
        why = f" ({c['why']})" if c.get("why") else ""
        parts.append(f"{loc}: error {err} vs tol {tol}{why}"
                     + (" — worst" if len(parts) == 0 else ""))
    return "; ".join(parts)


def _correctness_error(corr_obj: dict) -> str:
    """The bench's failure verdict, verbatim when it wrote one.

    Harness-level rejections (forbidden-op scan, import failure, bad return shape)
    arrive as a single ``error`` string; numeric fails arrive as per-tensor detail
    dicts instead, so summarise those into one line — and since WP15 the summary
    carries the top-k worst rows (``_correctness_fail_detail``), not just the
    aggregate: a FAIL whose feedback names the locus and the gap ends the
    diagnosis in one round instead of three.
    """
    if err := str(corr_obj.get("error") or "").strip():
        detail = _correctness_fail_detail(corr_obj)
        # the bench's own verdict already names the first failing (group, output);
        # the ranked rows add the RUNNERS-UP the first-fail stop never reached.
        return f"{err}; worst checks: {detail}" if detail and detail not in err else err
    parts = []
    for name in ("dq", "dk", "dv"):
        det = corr_obj.get(name)
        if not isinstance(det, dict):
            continue
        if det_err := det.get("error"):
            parts.append(f"{name}: {det_err}")
        elif "max_diff" in det:
            parts.append(f"{name}: max_diff={det['max_diff']} tol={det.get('tolerance')}")
    if parts:
        return "; ".join(parts)
    detail = _correctness_fail_detail(corr_obj)
    return detail or "no failure detail in the eval JSON"


def corr_margin_ratio(corr_obj: dict) -> float | None:
    """The worst normalized-error / tolerance ratio among the correctness checks
    (WP20 task 3): how close to the judge's edge the worst output sat — >=1.0
    fails, the 0.9-1.1 band is the edge the judge recheck watches. The ratio was
    buried verbatim in the error text (max_diff / tolerance strings); a REVERT
    row that carries it can be aggregated afterwards. None when no check carries
    both numbers (a harness-level fail, or a legacy verdict) — an absent
    reading, never a passing one."""
    checks = corr_obj.get("checks")
    if not isinstance(checks, list):
        return None
    worst = None
    for c in checks:
        if not isinstance(c, dict):
            continue
        e, t = c.get("error"), c.get("tolerance")
        if isinstance(e, (int, float)) and not isinstance(e, bool) \
                and isinstance(t, (int, float)) and not isinstance(t, bool) and t > 0:
            r = float(e) / float(t)
            worst = r if worst is None else max(worst, r)
    return worst


def parse_eval(text: str) -> EvalResult:
    """Parse the gpu_entry ``=== JSON ===`` eval block (correctness + perf)."""
    obj = _last_json_object(text)
    if not obj:
        from peregrine_kernel.submit.jobs import clock_lock_refusal
        if (why := clock_lock_refusal(text)) is not None:
            # RealBackend.submit_gpu never returns such a log; another submitter
            # might — say what happened instead of "the job died"
            return EvalResult(correctness=False,
                              error=f"the job timed nothing: the card's declared clock lock "
                                    f"did not hold on its node (a node fault, not the "
                                    f"kernel): {why[:300]}")
        # WP11 (task 5): the tail rides every failure error — the job died before
        # its verdict, and its log is the only evidence of why. Same sentence as
        # always, the evidence now attached.
        return EvalResult(correctness=False,
                          error=_with_log_tail(
                              "no eval JSON block in the job log (job died before the verdict?)",
                              text))
    if scope := obj.get("workload_scope"):
        # a group card timed on part of its group (PK_LB_WORKLOAD_SCOPE, the seeded
        # meta tools' setting) — a pricing reading, never a gate: the gate is the
        # geomean over every workload of the group
        return EvalResult(correctness=False,
                          error=f"eval JSON carries workload_scope={json.dumps(scope)}: a group card "
                                f"timed on part of its group is a meta reading, not a gate")
    corr_obj = obj.get("correctness", {}) or {}
    correct = bool(corr_obj.get("pass", False))
    perf = obj.get("performance") or {}
    ms = perf.get("bwd_median_ms")
    mfu = perf.get("bwd_mfu_pct", perf.get("mfu"))

    def _opt(key: str, legacy: str | None = None) -> float | None:
        # legacy: pre-rename bench key (cublas_* — the in-job reference was recorded
        # under that name whatever library it was). Read so older cards' bench.py
        # keeps parsing; new benches emit the ref_* names only. The same guard the
        # list fields apply: a real finite number or nothing (a NaN gate reading
        # takes over every min() it joins and silently disables the plateau
        # trigger; a bool/string "reading" is a malformed bench, not a small one).
        return _finite_float(perf.get(key, perf.get(legacy) if legacy else None))

    def _opt_any(*keys: str) -> float | None:
        for key in keys:
            v = _opt(key)
            if v is not None:
                return v
        return None

    # The prep block rides inside performance (timing.py's contract); a bench
    # that placed it at the top level is read too.
    prep = perf.get("prep") if isinstance(perf.get("prep"), dict) else obj.get("prep")
    if not isinstance(prep, dict):
        prep = None

    workloads = _workloads(perf)
    n_wl = perf.get("n_workloads")
    n_wl = int(n_wl) if isinstance(n_wl, int) and not isinstance(n_wl, bool) and n_wl > 0 \
        else max(1, len(workloads))

    return EvalResult(
        prep=prep,
        precision=obj.get("precision") if isinstance(obj.get("precision"), dict) else None,
        # WP1 telemetry: the bench may report gpu_state at the eval JSON's top
        # level or inside the performance block (timing.py emits it there);
        # top level wins when both exist.
        gpu_state=(obj.get("gpu_state") if isinstance(obj.get("gpu_state"), dict)
                   else (perf.get("gpu_state")
                         if isinstance(perf.get("gpu_state"), dict) else None)),
        device_readings_ms=_float_list(perf.get("device_readings_ms")),
        ref_device_readings_ms=_float_list(perf.get("ref_device_readings_ms")),
        n_workloads=n_wl,
        workloads=workloads,
        correctness=correct,
        # WP11 (task 5): a failing eval carries its job log's bounded tail — the
        # COMPILE_ERROR verdict's nvcc/ptxas error and a numeric FAIL's surrounding
        # output are in the log the parser holds, so the record now holds them too.
        error="" if correct else _with_log_tail(_correctness_error(corr_obj), text),
        bwd_median_ms=_finite_float(ms),
        mfu=_finite_float(mfu),
        ref_median_ms=_opt("ref_median_ms", "cublas_median_ms"),
        vs_ref=_opt("vs_ref", "vs_cublas"),
        device_median_ms=_opt("device_median_ms"),
        ref_device_ms=_opt("ref_device_ms", "cublas_device_ms"),
        vs_ref_device=_opt("vs_ref_device", "vs_cublas_device"),
        host_enqueue_ms=_opt_any("host_enqueue_ms", "enqueue_ms"),
        host_bound_ratio=_opt("host_bound_ratio"),
        ref_host_bound_ratio=_opt_any("ref_host_bound_ratio", "cublas_host_bound_ratio"),
        device_window_basis=(str(perf["device_window_basis"])
                             if perf.get("device_window_basis") else None),
        ref_host_enqueue_ms=_opt_any("ref_host_enqueue_ms", "cublas_host_enqueue_ms"),
        fresh_recheck=(obj.get("fresh_recheck")
                       if isinstance(obj.get("fresh_recheck"), dict) else None),
        sprt=(perf.get("sprt") if isinstance(perf.get("sprt"), dict)
              else (obj.get("sprt") if isinstance(obj.get("sprt"), dict) else None)),
        # --- WP20: the eval JSON's accounting rows (readings, never verdicts) ---
        corr_margin_ratio=corr_margin_ratio(corr_obj),
        gate_degraded=(obj["gate_degraded"]
                       if isinstance(obj.get("gate_degraded"), bool) else None),
        precision_audit=(str(obj["precision_audit"])
                         if isinstance(obj.get("precision_audit"), str) else None),
        stream_sentinel=(obj["stream_sentinel"]
                         if isinstance(obj.get("stream_sentinel"), dict) else None),
        timer_integrity=(str(obj["timer_integrity"])
                         if isinstance(obj.get("timer_integrity"), str) else None),
        proc_census=(obj["proc_census"]
                     if isinstance(obj.get("proc_census"), dict) else None),
        replay_probe=(obj["replay_probe"]
                      if isinstance(obj.get("replay_probe"), dict) else None),
        # --- M3: the lane stamp and the memo_fallback gate (MEMO cards only;
        # absent on a STRICT eval JSON -> None, the fresh_recheck pattern) ---
        lane=(obj["lane"] if isinstance(obj.get("lane"), dict) else None),
        memo_fallback=(obj["memo_fallback"]
                       if isinstance(obj.get("memo_fallback"), dict) else None),
    )


PROBE_MAX_DIFF_CEILING = 1e3
"""Largest ``max_diff`` the driver will accept behind a PASS.

Not a correctness tolerance — the driver cannot know a probe's tensor scale, so
the real bound stays where it can be computed (``eval/bench.py``, relative to
``|ref32|_max``). This is a plausibility floor under that: for fp16 gradients an
error norm above 1e3 means the kernel is catastrophically wrong, so a PASS above
it is a FAIL either way. What it actually catches is the number not being an
error norm at all — probes have reported values like 8e10 and 8.1e15, smuggling
sub-check bitmasks through the one field the driver echoes back.
"""

_PROBE_VERDICT_RE = re.compile(r"PROBE\s+(PASS|FAIL)(?:\s+max_diff=(\S+))?([^\n]*)")
_PROBE_NON_NUMERIC = {"n/a", "na", "none", "-"}
_PROBE_NOTE_LEAD = " \t#-/:;|"
_PROBE_READING_MAX = 96

# The operator entry script's closing line (gpu_entry.sh prints it after the
# task, whatever the task was). Both spellings of the tag are in the field.
ENTRY_DONE_RE = re.compile(r"^\[gpu_entry(?:\.sh)?\] task=(?P<task>\S+) finished", re.M)

# Any line the entry script itself printed: proof the log is a job log (as opposed
# to bare verdict text), so its closing line can be demanded.
_ENTRY_LINE_RE = re.compile(r"^\[gpu_entry(?:\.sh)?\] ", re.M)

# The entry script's own bookends, walked as a stack: `timeout=` opens a run,
# `task=... finished` closes it cleanly, `KILLED:` closes it dead. A cmd job may
# nest entry runs (orch_census streams `bash gpu_entry.sh ncu`; a KB/teacher probe
# shells out the same way, and a ladder rung may run an inner entry under its own
# cap), so a kill line mid-log can be an inner rung the probe handled, and a
# `finished` line mid-log can be an inner run while the outer one is still open.
_ENTRY_EVENT_RE = re.compile(
    r"^\[gpu_entry(?:\.sh)?\] (?:(?P<open>timeout=)|(?P<done>task=\S+ finished)|(?P<killed>KILLED:))",
    re.M)

# Kill evidence written by the layers around the entry script, anchored to the
# spelling those layers use (a probe quoting a child's wrapper line on the FACT
# channel is not the wrapper): bash's report of the SIGKILL from
# `timeout --signal=KILL` (carries no code), the runner's closing line (written
# only for a non-zero rc) and its run-timeout line (peregrine_kernel.submit.direct).
_JOB_ABEND_RES = (
    re.compile(r"gpu_entry\.sh: line \d+:\s+\d+ Killed\b"),
    re.compile(r"^\[direct\] gpu_entry .+? exited rc=(?P<rc>[1-9]\d*)", re.M),  # mode may carry spaces
    re.compile(r"^\[direct\] gpu_entry .+? exceeded \S+ -- killed", re.M),
)

JOB_END_WHOLE, JOB_END_KILLED, JOB_END_CUT = "", "killed", "cut"


class JobEnd(NamedTuple):
    """How a job log ends: ``kind`` is ``JOB_END_WHOLE`` (``""``: the job ran to
    completion, or the text is not a job log at all), ``JOB_END_KILLED`` (the log
    carries kill evidence) or ``JOB_END_CUT`` (the outer entry run never closed
    and nothing says why: the job died silently, or the archive is cut).
    ``reason`` is the sentence for a log line / feedback; ``""`` when whole."""
    kind: str
    reason: str = ""


def job_abend(log: str) -> JobEnd:
    """Whether ``log`` says the job ran to completion, and if not, why.

    A PROBE line is a claim about the rung that printed it. The rungs after it
    (a launch that hangs, a relaunch, a teardown) print nothing when the job is
    killed under them, and a log that ends in a kill is not a log of a job that
    ran to completion — whatever it printed before. Bare verdict text with no
    entry-script line at all (tests, hand-run probes) is left as read.

    Entry runs are walked as a stack (see ``_ENTRY_EVENT_RE``): the job ran to
    completion when every run that opened has closed and the last one closed
    with its ``finished`` line — an inner run killed under its own cap and
    handled by the probe does not veto an outer run that finished, and an inner
    ``finished`` line does not stand in for an outer run that is still open. Only
    an open outer run consults the wrapper layers' kill lines.
    """
    if not _ENTRY_LINE_RE.search(log):
        return JobEnd(JOB_END_WHOLE)
    depth, last = 0, None
    for m in _ENTRY_EVENT_RE.finditer(log):
        if m["open"]:
            depth += 1
        else:
            depth = max(depth - 1, 0)
            last = "done" if m["done"] else "killed"
    if depth == 0 and last == "done":
        return JobEnd(JOB_END_WHOLE)
    hits = [m for m in (rx.search(log) for rx in _JOB_ABEND_RES) if m]
    if last == "killed" or hits:
        # bash's "Killed" line carries no code; the wrapper lines after it do
        rc = next((m.group("rc") for m in hits if m.groupdict().get("rc")), None)
        return JobEnd(JOB_END_KILLED,
                      f"the job was killed (exit {rc})" if rc else "the job was killed")
    return JobEnd(JOB_END_CUT, (
        "the entry script never printed its closing line and the log shows no kill — "
        "the archive is cut (the log archiver lags the job) or the job died silently"))


def _as_finite_float(raw: str | None) -> float | None:
    """The token as an error norm / reading, or None when it is not a finite number."""
    if raw is None:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def _clip(text: str, limit: int = _PROBE_READING_MAX) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass(frozen=True)
class ProbeContract:
    """One ``PROBE PASS/FAIL`` line of a probe job log, as the probe printed it.

    A probe is allowed (and told) to print one line per contract it checks. Those
    lines used to be collapsed into a single bit plus the first float-parseable
    ``max_diff`` — so a five-contract experiment whose decisive contract PASSed
    with a reading table and whose control contract FAILed on a tooling hiccup
    surfaced as ``PROBE FAIL (max_diff=unreported)``. Each line is now kept.
    """
    index: int             # 1-based, log order
    verdict: str           # "PASS" | "FAIL"
    raw: str | None        # the max_diff token exactly as printed; None when absent
    note: str = ""         # trailing text on the line (contract name / reason)

    @property
    def diff(self) -> float | None:
        return _as_finite_float(self.raw)

    @property
    def reading(self) -> str:
        """What the line reported, for logs: a number, ``n/a``, a reading string,
        or ``unreported`` when the line carried no max_diff at all."""
        return "unreported" if self.raw is None else _clip(self.raw)

    def describe(self) -> str:
        head = f"#{self.index} {self.verdict} max_diff={self.reading}"
        return f"{head} — {_clip(self.note, 120)}" if self.note else head


def _contracts(text: str) -> tuple[ProbeContract, ...]:
    out = []
    for i, m in enumerate(_PROBE_VERDICT_RE.finditer(text), start=1):
        note = m.group(3).strip().lstrip(_PROBE_NOTE_LEAD).strip()
        out.append(ProbeContract(i, m.group(1), m.group(2) or None, note))
    return tuple(out)


@dataclass(frozen=True)
class ProbeVerdict:
    """A probe's GPU verdict, after the driver has audited the probe's own claim.

    ``passed`` is the driver's judgement, not the probe's: a self-reported PASS is
    downgraded when the accompanying ``max_diff`` cannot be an error norm. ``found``
    separates "the probe ran and said FAIL" from "no verdict line at all" — those
    reached the student as the same message before, so a probe that never built
    was fed back as a numerical failure.

    ``contracts`` keeps every verdict line the probe printed; ``reading`` keeps the
    representative ``max_diff`` token when it was not a number (``n/a``, or a
    ``k=v,...`` reading string a verdict experiment packed into the field), so a
    line that reported something is never logged as ``unreported``.
    """
    passed: bool
    max_diff: float | None = None
    numeric: bool = True
    found: bool = True
    reason: str = ""
    contracts: tuple[ProbeContract, ...] = ()
    reading: str | None = None

    @property
    def diff_text(self) -> str:
        """How the diff should read in a log line / in feedback to the agent."""
        if self.max_diff is not None:
            return f"{self.max_diff}"
        if self.reading:
            return _clip(self.reading)
        return "n/a (non-numeric probe)" if not self.numeric else "unreported"

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.contracts if c.verdict == "PASS")

    @property
    def summary(self) -> str:
        """Per-contract tally for multi-line probes: ``2/4 PASS; FAIL #1 max_diff=n/a
        — <note>, #3 ...``. Empty for a single-line probe (the diff_text says it all)."""
        if len(self.contracts) < 2:
            return ""
        fails = [c for c in self.contracts if c.verdict == "FAIL"]
        text = f"{self.n_pass}/{len(self.contracts)} PASS"
        if fails:
            text += "; FAIL " + ", ".join(c.describe() for c in fails)
        return text

    def contract_table(self) -> str:
        """One line per contract, for feedback blocks."""
        return "\n".join(c.describe() for c in self.contracts)


def _audit_pass(raw: str | None) -> ProbeVerdict:
    """Decide whether one self-reported PASS survives the driver's audit."""
    if raw is None:
        return ProbeVerdict(False, reason=(
            "reported PASS with no max_diff=... — a PASS must declare what it "
            "verified: a numeric diff, or max_diff=n/a for a non-numeric contract"))
    if raw.strip().lower() in _PROBE_NON_NUMERIC:
        return ProbeVerdict(True, max_diff=None, numeric=False)
    try:
        diff = float(raw)
    except ValueError:
        return ProbeVerdict(False, reason=(
            f"reported PASS with an unparseable max_diff={raw!r} — report a number, "
            f"or max_diff=n/a for a non-numeric contract"))
    if diff != diff or diff in (float("inf"), float("-inf")):  # NaN / inf
        return ProbeVerdict(False, reason=(
            f"reported PASS with max_diff={raw} — a non-finite diff is a failure"))
    if abs(diff) > PROBE_MAX_DIFF_CEILING:
        return ProbeVerdict(False, reason=(
            f"reported PASS with max_diff={diff}, above the {PROBE_MAX_DIFF_CEILING:g} "
            f"ceiling — that is not an error norm. Do not encode sub-check codes in "
            f"max_diff; report the real diff, or max_diff=n/a plus one PROBE line per "
            f"contract"))
    return ProbeVerdict(True, max_diff=diff)


def _representative(contracts: tuple[ProbeContract, ...]) -> tuple[float | None, str | None]:
    """(first finite numeric diff, first non-numeric token) among the given lines."""
    diff = next((c.diff for c in contracts if c.diff is not None), None)
    reading = next((c.raw for c in contracts if c.raw is not None and c.diff is None), None)
    return diff, reading


def parse_probe_verdict(text: str, *, audit: bool = True) -> ProbeVerdict:
    """Audit the ``PROBE PASS/FAIL max_diff=<x>`` verdict(s) in a probe job's log.

    The verdict is written by the probe the student authored, so it is a claim, not
    evidence. Three ways that claim used to be taken at face value, all of which
    let a probe certify a kernel the eval then rejected:

    * only the *first* verdict was read, so a probe printing PASS for check 1 and
      FAIL for check 2 certified as PASS. Every verdict now has to agree — any FAIL
      loses.
    * a log with no verdict at all (build error, killed job) was indistinguishable
      from a numerical FAIL. It is now ``found=False``.
    * ``max_diff`` was decorative. It is now audited (see ``_audit_pass``).

    ``audit=True`` is the certification reading (student probes, KB recipe probes):
    a PASS is a claim that a kernel is numerically right, so its ``max_diff`` must
    be an error norm. ``audit=False`` is the verdict-experiment reading (diagnose /
    m0-audit): the lines are answers to pre-registered hypotheses and ``max_diff``
    carries the reading — a number, ``n/a``, or a ``k=v,...`` string — and is
    recorded as printed. A PASS is never downgraded for the shape of its reading
    there, because a downgraded PASS was reaching the teacher as "FAIL", i.e. as
    the hypothesis being falsified, which is the opposite of what the experiment
    said (e.g. ``PROBE PASS max_diff=ref_exposed=106.3,...`` on grouped-gemm).
    """
    contracts = _contracts(text)
    if not contracts:
        return ProbeVerdict(False, found=False, reason=(
            "no 'PROBE PASS/FAIL' line in the job log — the probe did not build, "
            "or the job died before reaching the verdict"))

    fails = tuple(c for c in contracts if c.verdict == "FAIL")
    if fails:
        diff, reading = _representative(fails)
        reason = ""
        if len(fails) < len(contracts):
            reason = (f"{len(fails)} of {len(contracts)} PROBE lines reported FAIL — "
                      f"a probe passes only when every check passes")
        return ProbeVerdict(False, max_diff=diff, reason=reason,
                            contracts=contracts, reading=reading)

    if not audit:
        diff, reading = _representative(contracts)
        diffs = [c.diff for c in contracts if c.diff is not None]
        if reading is not None and reading.strip().lower() in _PROBE_NON_NUMERIC:
            reading = None  # `n/a` on every line: a non-numeric experiment, as before
        return ProbeVerdict(True, max_diff=max(diffs) if diffs else None,
                            numeric=bool(diffs) or reading is not None,
                            contracts=contracts, reading=reading)

    # all PASS: every one must survive the audit, and the worst diff is the one
    # that represents the probe.
    audited = [_audit_pass(c.raw) for c in contracts]
    rejected = next((v for v in audited if not v.passed), None)
    if rejected is not None:
        return ProbeVerdict(False, reason=rejected.reason, contracts=contracts)
    diffs = [v.max_diff for v in audited if v.max_diff is not None]
    # Every line said PASS — but a PASS certifies only a job that ran to completion.
    # A multi-rung probe that printed PASS for its first rung and then hung in the
    # next was scored PASS on that one line (on attn-bwd: `T-trace` PASS, `S-smoke`
    # hung 1800 s, rc=137, and the hanging tree was delivered as certified).
    end = job_abend(text)
    if end.kind == JOB_END_KILLED:
        return ProbeVerdict(False, max_diff=max(diffs) if diffs else None,
                            numeric=bool(diffs), contracts=contracts, reason=(
                                f"every PROBE line said PASS but {end.reason} — the rungs "
                                f"after the last PROBE line (a launch that hung, a "
                                f"relaunch, a teardown) have no verdict. Bound every "
                                f"rung so a hang costs its own cap, not the job, and "
                                f"print one PROBE line per rung"))
    if end.kind == JOB_END_CUT:
        # Not a rejection of the probe: nothing in the log blames it. NO VERDICT
        # sends the student back for a run that ends, not hunting for a defect.
        return ProbeVerdict(False, found=False, max_diff=max(diffs) if diffs else None,
                            numeric=bool(diffs), contracts=contracts, reason=(
                                f"every PROBE line said PASS but {end.reason}. If the "
                                f"probe printed every line it should have, re-emit the "
                                f"same PROBE_REQUEST unchanged"))
    return ProbeVerdict(True, max_diff=max(diffs) if diffs else None,
                        numeric=bool(diffs), contracts=contracts)


# --- the resolvable difference (WP1; cross-package contract C1; WP15 v2) ------
#
# The 1.5% noise floor was a constant standing in for a measurement nobody kept:
# the spread of the run's OWN same-job paired ratios. Those ratios ride every
# evaluated record (vs_ref_device / vs_ref — same job, so node speed cancels),
# and a quiet run resolves far finer than 1.5%. WP4 / WP6 consume this function
# in parallel — the name, module and positional signature are a contract.
#
# WP15 (v2, contract C8): the estimator is LAYERED. The v1 shape had two known
# biases (42-report §2): the residual baseline was the rolling MAX (one lucky or
# genuinely-faster draw pinned every residual positive and the band wide — the
# run's own progress read as noise), and the job-internal window readings the
# records already carry (device_readings_ms / ref_device_readings_ms — up to a
# dozen pure window-noise samples per few records) were never consulted. v2:
#
#   sigma_within   per-window paired ratios (ref_i / kernel_i) of the recent
#                  records, deviated around each record's OWN robust center —
#                  kernel quality cancels within a record, what pools is pure
#                  window noise (the _SprtController._candidate_pairs family);
#   sigma_between  the record-level ratios' residuals against the robust center
#                  of the OTHERS (leave-one-out — no rolling max to pin them),
#                  plus the confirmation jobs' ratios the driver injects
#                  (``raw={"ratios": [...]}`` — same-sha re-measurements are the
#                  only between-job samples that carry no kernel difference);
#   band           clamp(3 * max(sigma_within, sigma_between), floor, cap).
#
# The name, positional signature, clamp bounds and the data-poor fallback are
# untouched (C8): fewer than RESOLVABLE_MIN_SAMPLES ratio samples returns cap —
# the historical constant, so the old behavior is the fallback, never a mode.
# _resolvable_diff_v1 keeps the v1 arithmetic verbatim for the consistency
# matrix (tests + CHANGES) and for any caller that must reproduce old numbers.

RESOLVABLE_SAMPLES = 6
"""How many recent evaluated records the resolution estimate reads."""

RESOLVABLE_MIN_SAMPLES = 4
"""Fewer ratio samples than this cannot speak: the estimate returns the cap
(the historical 1.5% behavior)."""

RESOLVABLE_DIFF_FLOOR = 0.005
"""Never claim to resolve better than this: a floor on the estimated margin
(3 sigma of a very quiet run clamps here, ~0.5%)."""

RESOLVABLE_DIFF_CAP = 0.015
"""Never claim to resolve worse than the historical noise floor: the cap the
estimate clamps to, and the fallback when the data is insufficient."""

RESOLVABLE_MAX_WINDOW_PAIRS = 24
"""Cap on the pooled within-job window pairs (a noise estimate, not a census)."""


def _gate_ratio(rec, gate_window: str) -> float | None:
    """The record's in-job paired ratio in the gate window, or None. vs_ref_device
    is a same-job reading (ref and student timed back-to-back), so node speed
    cancels and what remains is measurement spread — the quantity the resolution
    estimate needs."""
    if gate_window == GATE_WINDOW_DEVICE:
        v = getattr(rec, "vs_ref_device", None)
    else:
        v = getattr(rec, "vs_ref", None)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _mad_sigma(values) -> float | None:
    """1.4826 x MAD(values) — the robust sigma of one family; None when fewer
    than two usable values (a spread no single sample can speak about)."""
    vals = [float(v) for v in (values or ())
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(vals) < 2:
        return None
    med = statistics.median(vals)
    return 1.4826 * statistics.median(abs(v - med) for v in vals)


def _window_pairs(rec, gate_window: str) -> list[float]:
    """One record's per-window paired ratios (ref_i / kernel_i, index-aligned raw
    device readings), deviated around the record's OWN robust center — what
    remains is that record's pure window noise, its kernel level cancelled.
    [] when the record carries no raw readings (a plain measure_windows bench, a
    pre-WP1 row, or the e2e gate window — the raw arrays are device-window)."""
    if gate_window != GATE_WINDOW_DEVICE:
        return []
    groups = []
    kr = getattr(rec, "device_readings_ms", None)
    rr = getattr(rec, "ref_device_readings_ms", None)
    if kr and rr:
        groups.append((rr, kr))
    for w in getattr(rec, "workloads", None) or ():
        if isinstance(w, dict) and w.get("device_readings_ms") and w.get("ref_device_readings_ms"):
            groups.append((w["ref_device_readings_ms"], w["device_readings_ms"]))
    out = []
    for refs, kers in groups:
        n = min(len(refs), len(kers))
        ratios = [float(r) / float(k) for r, k in zip(refs[:n], kers[:n])
                  if float(k) > 0.0 and float(r) > 0.0]
        if len(ratios) < 2:
            continue
        center = statistics.median(ratios)
        out.extend(r - center for r in ratios)
    return out


def _drift_clean(samples: list, min_keep: int) -> list:
    """Drop rows whose SM clock sat >2% off the best's (they are not one
    measurement condition and inflate every noise estimate), but only while the
    remainder can still speak (>= ``min_keep``) — two-level fallback: an
    all-drift history measures with what it has, exactly as before WP15."""
    clean = [s for s in samples if not getattr(s, "clock_drift", False)]
    return clean if len(clean) >= min_keep else samples


def _resolvable_diff_v1(records, gate_window: str, *, floor: float, cap: float) -> float:
    """The v1 estimator, verbatim (rolling-max residuals): kept for the v1<->v2
    consistency matrix and as documentation of the fallback arithmetic. Not
    consumed by the loop — resolvable_diff owns the production path."""
    evaluated = [r for r in records if gate_ms(r, gate_window) is not None]
    ratios = [v for r in evaluated[-RESOLVABLE_SAMPLES:]
              if (v := _gate_ratio(r, gate_window)) is not None]
    if len(ratios) < RESOLVABLE_MIN_SAMPLES:
        return cap
    best = max(ratios)
    residuals = [best - v for v in ratios]
    med = statistics.median(residuals)
    mad = statistics.median(abs(v - med) for v in residuals)
    sigma = 1.4826 * mad
    return min(cap, max(floor, 3.0 * sigma))


def resolvable_diff(records, gate_window: str = GATE_WINDOW_DEFAULT, *,
                    floor: float = RESOLVABLE_DIFF_FLOOR,
                    cap: float = RESOLVABLE_DIFF_CAP,
                    raw: dict | None = None) -> float:
    """The relative difference this run's own recent measurements can resolve.

    Data (WP15 v2, layered — see the block comment above):
    * **within-job**: per-window paired ratios of the recent records, deviated
      around each record's own center (pure window noise, kernel cancelled);
    * **between-job**: the record-level ratios' leave-one-out residuals (the v1
      rolling-max baseline — which let one draw pin every residual and widen the
      band exactly where the run was making progress — is gone), plus any
      confirmation-job ratios injected via ``raw``.

    Returns ``clamp(3 x max(sigma_within, sigma_between), floor, cap)``. Fewer
    than ``RESOLVABLE_MIN_SAMPLES`` between-job ratio samples (a young run, or a
    bench without an in-job reference) returns ``cap`` — the historical constant,
    so the old behavior is the data-poor fallback, never a separate mode. A
    record with clock_drift=True is dropped from the sample while enough clean
    rows remain (drifted readings inflate sigma; the fallback keeps a history
    that is ALL drift measurable, as before).

    ``raw`` (contract C8's injection point): {"pairs": [...], "ratios": [...]} —
    within-job window deviations and between-job job-level ratios the CALLER
    holds outside the trajectory (the confirmation jobs' same-sha re-measurements).
    None (the default) reads everything off the records; an injected dict is
    POOLED with the records' own samples, never a replacement.

    ``floor`` / ``cap`` are loop.toml [loop] resolvable_floor / resolvable_cap
    (defaults here); detect_plateau forwards its caller's.
    """
    evaluated = [r for r in records if gate_ms(r, gate_window) is not None]
    recent = _drift_clean(evaluated[-RESOLVABLE_SAMPLES:], RESOLVABLE_MIN_SAMPLES)
    ratios = [v for r in recent if (v := _gate_ratio(r, gate_window)) is not None]
    extra = [float(v) for v in ((raw or {}).get("ratios") or ())
             if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0.0]
    if len(ratios) + len(extra) < RESOLVABLE_MIN_SAMPLES:
        return cap
    # within-job: window deviations off the records' raw readings (+ injected pairs)
    pairs = []
    for r in recent:
        pairs.extend(_window_pairs(r, gate_window))
    pairs.extend(float(v) for v in ((raw or {}).get("pairs") or ())
                 if isinstance(v, (int, float)) and not isinstance(v, bool))
    pairs = pairs[:RESOLVABLE_MAX_WINDOW_PAIRS]
    sigma_within = _mad_sigma(pairs) or 0.0
    # between-job: leave-one-out residuals of the pooled job-level ratios. The
    # center for each datum is the median of the OTHERS — no datum can pin its
    # own baseline (the v1 bias), and a robust center survives one outlier.
    datums = ratios + extra
    residuals = [statistics.median(datums[:i] + datums[i + 1:]) - v
                 for i, v in enumerate(datums) if len(datums) > 1]
    sigma_between = _mad_sigma(residuals) or 0.0
    sigma = max(sigma_within, sigma_between)
    return min(cap, max(floor, 3.0 * sigma))


def combined_workload_test(records, gate_window: str = GATE_WINDOW_DEFAULT, *,
                           floor: float = RESOLVABLE_DIFF_FLOOR,
                           cap: float = RESOLVABLE_DIFF_CAP,
                           z_trigger: float = 2.0) -> dict | None:
    """Per-workload Stouffer combined test over the recent records (WP15; NEW
    function name per contract C8 — the verdict main path never calls this).

    Why: the gate ratio is the geomean over the card's workloads, and the
    confirmation band rides the same pooled currency — a gain that is small but
    CONSISTENT across every workload (each +0.7%, the geomean +0.7%) hides under
    a band fitted to the noisiest workload, while a gain carried by one workload
    and a loss on another reads the same geomean. Stouffer's method combines each
    workload's own z (its recent median ratio minus parity, over its own robust
    sigma — the resolvable_diff MAD family) into one z with weights √n each;
    heterogeneous per-workload noise stops diluting the signal the geomean
    averages away.

    Judgment is NOT here: the return value is advisory. The driver uses
    ``significant`` only as one more CONFIRMATION-BAND trigger (a suspicious
    reading gets re-measured — never a verdict by itself) and renders the row as
    a teacher advisory in the plateau/audit block's existing slot.

    Returns None when the records cannot speak (no per-workload ratios, or fewer
    than RESOLVABLE_MIN_SAMPLES records with readings — same data floor as
    resolvable_diff)."""
    evaluated = _drift_clean([r for r in records if gate_ms(r, gate_window) is not None],
                             RESOLVABLE_MIN_SAMPLES)
    recent = evaluated[-RESOLVABLE_SAMPLES:]
    per: dict[str, list[float]] = {}
    for r in recent:
        for w in getattr(r, "workloads", None) or ():
            if not isinstance(w, dict):
                continue
            key = str(w.get("name"))
            v = w.get("vs_ref_device", w.get("vs_ref")) if gate_window == GATE_WINDOW_DEVICE \
                else w.get("vs_ref")
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0.0:
                per.setdefault(key, []).append(float(v))
    if sum(len(v) for v in per.values()) < RESOLVABLE_MIN_SAMPLES or len(per) < 2:
        return None
    rows, z_sum, w_sum = [], 0.0, 0.0
    for name, vals in per.items():
        med = statistics.median(vals)
        sigma = _mad_sigma(vals)
        if not sigma:
            continue
        z = (med - 1.0) / sigma
        weight = len(vals) ** 0.5
        z_sum += weight * z
        w_sum += weight
        rows.append({"name": name, "n": len(vals), "median_ratio": round(med, 4),
                     "sigma_pct": round(sigma * 100, 2), "z": round(z, 2)})
    if not w_sum:
        return None
    z = z_sum / w_sum
    # one-sided p of the GAIN direction (z > 0): normal CDF via Abramowitz &
    # Stegun 26.2.17 — report-only, so the rational approximation is the honest
    # precision. A loss-direction z reads p=1: this test exists to surface small
    # consistent WINS, never to call a loss.
    t = 1.0 / (1.0 + 0.2316419 * z) if z >= 0 else 0.0
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
             + t * (-1.821255978 + t * 1.330274429))))
    phi = 1.0 - 0.3989422804014327 * pow(2.718281828459045, -0.5 * z * z) * poly \
        if z >= 0 else 0.0
    p = max(0.0, min(1.0, 1.0 - phi))
    return {"z": round(z, 3), "p_one_sided": round(p, 5),
            "significant": z >= z_trigger, "z_trigger": z_trigger,
            "workloads": rows}


def best_standard_error(records, gate_window: str = GATE_WINDOW_DEFAULT,
                        *, min_within: int = 2) -> dict | None:
    """The running best's log-domain standard error and CI95 (WP15; NEW function
    name per contract C8 — an objective statistic row, never a judgment).

    The best is min-of-draws: statistically it sits ABOVE the true best-branch
    mean by an extreme-value offset, and every later gain is priced against that
    bar. The row cannot remove the offset (only repetition can), but it can SAY
    how wide the luck is: se of the best estimate in log domain, from

    * the best row's CONFIRMATION readings when it carries >= 2 (the between-job
      se of its median — the honest one), else
    * the best row's own raw device readings (the within-job se of a median,
      1.2533 x MAD(log)/sqrt(n) for the median-of-n estimator), else None.

    ci95 = best_ms * exp(+/- 1.96 * se_log). The best row is picked exactly as
    the driver's _best_ms does (min gate reading over rows with one, drifted
    rows dropped while enough clean rows remain), so the digest line and the
    watermark bar always describe the same number."""
    evaluated = _drift_clean([r for r in records if gate_ms(r, gate_window) is not None],
                             1)
    if not evaluated:
        return None
    best = min(evaluated, key=lambda r: gate_ms(r, gate_window))
    best_ms = gate_ms(best, gate_window)
    readings = [float(v) for v in (getattr(best, "confirm_readings", None) or ())
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0.0]
    if len(readings) >= 2:
        logs = [math.log(v) for v in readings]
        se = (statistics.stdev(logs) / (len(logs) ** 0.5)) if len(logs) > 1 else None
        basis, n = "confirm_jobs", len(readings)
    else:
        raw = [float(v) for v in (getattr(best, "device_readings_ms", None) or ())
               if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0.0]
        if len(raw) < min_within:
            # single-input rows keep no raw arrays in the trajectory (the WP1
            # digest-budget decision); the per-workload blocks do — read those
            for w in getattr(best, "workloads", None) or ():
                if isinstance(w, dict):
                    raw.extend(float(v) for v in (w.get("device_readings_ms") or ())
                               if isinstance(v, (int, float)) and not isinstance(v, bool)
                               and v > 0.0)
        if len(raw) < min_within:
            return {"best_ms": best_ms, "se_log": None, "n": 1, "basis": None,
                    "ci95": None, "iter": best.iter,
                    "best_low_confidence": bool(getattr(best, "best_low_confidence", False))}
        logs = [math.log(v) for v in raw]
        mad = statistics.median(abs(x - statistics.median(logs)) for x in logs)
        se = 1.2533 * mad / (len(logs) ** 0.5)
        basis, n = "within_job", len(raw)
    if se is None:
        return {"best_ms": best_ms, "se_log": None, "n": n, "basis": basis,
                "ci95": None, "iter": best.iter,
                "best_low_confidence": bool(getattr(best, "best_low_confidence", False))}
    half = 1.96 * se
    return {"best_ms": best_ms, "se_log": round(se, 5), "n": n, "basis": basis,
            "ci95": [round(best_ms * pow(2.718281828459045, -half), 4),
                     round(best_ms * pow(2.718281828459045, half), 4)],
            "iter": best.iter,
            "best_low_confidence": bool(getattr(best, "best_low_confidence", False))}


def detect_plateau(records, window: int = 3, noise_floor: float = 0.015,
                   gate_window: str = GATE_WINDOW_DEFAULT, *,
                   floor: float | None = None,
                   cap: float | None = None,
                   raw: dict | None = None) -> bool:
    """True when the last ``window`` evaluated iters set no new best beyond noise.

    Evaluated = has a reading in the gate window (REVERTs with None are skipped).
    The plateau is the meta-loop trigger: phase-internal levers have stopped paying —
    so it must watch the same window the loop gates on, or host-side progress on a
    device-gated run (and vice versa) reads as a plateau and hands the run to meta.

    "New best" must clear the measurement noise, not an absolute epsilon. The
    running best is the luckiest draw so far (max-of-noise ratchet), and the
    same kernel re-measured spans ~1.37% on this harness (4-sample span)
    — so the margin is relative: what the run's own paired ratios can resolve
    (``resolvable_diff``, clamped to [resolvable_floor, noise_floor]), falling
    back to the historical ``noise_floor`` (default 1.5%) while the ratio
    history is too thin to speak. The old absolute eps (0.005 ms ≈ 0.2% of a
    2.5 ms kernel) sat an order of magnitude below that noise, which handed
    the trigger to the dice in both directions: real progress that failed to
    out-roll a lucky early sample read as plateau (meta invoked while levers
    still paid), and a lucky sub-noise draw read as progress (a dead phase kept
    grinding).

    WP15 (contract C8's G8.2 sync obligation): the margin IS the watermark
    confirmation band — both call this one resolvable_diff, so the v2 layered
    estimator moves both judgments together by construction; the test suite
    pins that identity (two estimators that could drift apart were the bug
    class this is named for). Drifted rows (clock_drift=True) drop from the
    window while enough clean rows remain, like every other consumer.

    ``noise_floor`` stays the ceiling of the margin (its historical meaning);
    ``floor`` / ``cap`` forward to resolvable_diff when the caller carries
    config (loop.toml [loop] resolvable_floor / resolvable_cap); ``raw``
    forwards the caller's confirmation-ratio injection (see resolvable_diff).
    """
    evaluated = [r for r in records if gate_ms(r, gate_window) is not None]
    if len(evaluated) <= window:
        return False
    margin = min(noise_floor, resolvable_diff(records, gate_window, floor=(
        RESOLVABLE_DIFF_FLOOR if floor is None else floor),
        cap=noise_floor if cap is None else cap, raw=raw))
    evaluated = _drift_clean(evaluated, window + 1)
    prior_best = min(gate_ms(r, gate_window) for r in evaluated[:-window])
    recent_best = min(gate_ms(r, gate_window) for r in evaluated[-window:])
    return recent_best >= prior_best * (1.0 - margin)
