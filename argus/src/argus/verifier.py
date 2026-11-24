"""Board-faithful, reward-hack-resistant verification (ARGUS contribution C1).

CudaForge certifies a kernel with a single call::

    torch.allclose(ref_out, test_out, atol=tol, rtol=tol)

on **one** input, with ``tol`` defaulting to 1e-3/1e-4.  Three things are wrong with that
as a gate for SOL-ExecBench:

1. **Wrong rule.** SOL evaluates ``|out - ref| <= atol + rtol*|ref|`` per element with
   ``atol = rtol = 1e-2`` and requires a **matched ratio >= 0.99**, plus an optional
   ``max_error_cap`` -- not a plain ``allclose``.
2. **Wrong coverage.** The board scores a kernel over *every* workload of a definition; a
   kernel that fits the one sampled shape can still zero its kernel on another shape.  A
   one-shape check certifies nothing about the workload distribution.
3. **No hack resistance.** SOL itself screens submissions and reports ~14.5% flagged,
   dominated by precision downgrades.  ``allclose`` cannot see a precision downgrade, a
   memoisation, a returned-input, or a skippped-store kernel, because all of those *pass*
   ``allclose`` on the shape they were tuned for.

ARGUS replaces the gate with a layered verifier:

* **static scan** for the known cheat classes (delegate to the reference, identity/no-op
  stores, pointer-keyed caching, clock/L2 tampering, in-kernel dtype narrowing);
* **board-exact tolerance** over *all* workloads, with the matched-ratio rule;
* **differential adversarial probes** -- determinism, memoisation, identity, degenerate
  variance, shape/dtype/finiteness;
* **precision contract** against a float64 oracle: a candidate whose output passes the
  tolerance but whose error against fp64 is far larger than the *reference's own* error is
  an internal-precision downgrade -- the exact failure mode that silently inflated the
  attention-backward number in the ForgeKernel ablation, and the #1 flagged hack class on
  the board;
* **metamorphic probes** from :mod:`argus.probes`.

Every check is sober: hard checks gate acceptance, soft checks only raise flags.  The whole
module is pure numpy and runs on a CPU, which is what makes it locally falsifiable -- see
``experiments/exp_hack_detection.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .objective import SOL_ATOL, SOL_MATCHED_RATIO, SOL_RTOL

HARD = "hard"
SOFT = "soft"
INFO = "info"

#: A runner maps a dict of named inputs to a dict of named outputs.
Runner = Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]


@dataclass(frozen=True)
class Tolerance:
    """SOL-ExecBench per-element tolerance rule."""

    atol: float = SOL_ATOL
    rtol: float = SOL_RTOL
    matched_ratio: float = SOL_MATCHED_RATIO
    max_error_cap: Optional[float] = None

    def as_dict(self) -> dict:
        return dict(atol=self.atol, rtol=self.rtol, matched_ratio=self.matched_ratio,
                    max_error_cap=self.max_error_cap)


@dataclass(frozen=True)
class PrecisionContract:
    """The minimum numeric fidelity a candidate must preserve."""

    output_dtype: str = "float32"       # required output dtype
    min_accum_bits: int = 32            # minimum accumulation / working precision
    allow_downcast: bool = False        # whether a narrower compute dtype is acceptable
    oracle_error_slack: float = 4.0     # candidate may be this much worse than the reference


@dataclass
class CheckOutcome:
    name: str
    passed: bool
    severity: str
    detail: str
    data: Dict[str, float] = field(default_factory=dict)


@dataclass
class Verdict:
    accepted: bool
    outcomes: List[CheckOutcome] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    @property
    def hard_failures(self) -> List[CheckOutcome]:
        return [o for o in self.outcomes if o.severity == HARD and not o.passed]

    @property
    def soft_failures(self) -> List[CheckOutcome]:
        return [o for o in self.outcomes if o.severity == SOFT and not o.passed]

    def as_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "flags": self.flags,
            "outcomes": [dict(o.__dict__) for o in self.outcomes],
        }

    def render(self) -> str:
        lines = [f"Verdict: {'ACCEPT' if self.accepted else 'REJECT'}"]
        for o in self.outcomes:
            mark = "ok " if o.passed else ("FAIL" if o.severity == HARD else "warn")
            lines.append(f"  [{mark}] {o.name}: {o.detail}")
        if self.flags:
            lines.append("  flags: " + ", ".join(self.flags))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Numeric primitives
# --------------------------------------------------------------------------- #

def elementwise_pass_ratio(out: np.ndarray, ref: np.ndarray, tol: Tolerance):
    """Return ``(ratio, max_abs_err, max_rel_err)`` under the SOL element rule."""
    o = np.asarray(out, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if o.shape != r.shape:
        return 0.0, float("inf"), float("inf")
    abs_err = np.abs(o - r)
    allowed = tol.atol + tol.rtol * np.abs(r)
    ratio = float(np.mean(abs_err <= allowed)) if abs_err.size else 1.0
    denom = np.maximum(np.abs(r), 1e-12)
    return ratio, float(abs_err.max()) if abs_err.size else 0.0, float((abs_err / denom).max()) if abs_err.size else 0.0


def _iter_outputs(out, ref):
    """Yield ``(name, out_tensor, ref_tensor)`` for dict/tuple/array outputs."""
    if isinstance(ref, dict):
        for name in ref:
            yield name, out[name], ref[name]
    elif isinstance(ref, tuple):
        for i, r in enumerate(ref):
            yield str(i), out[i], r
    else:
        yield "out", out, ref


def check_tolerance(out, ref, tol: Tolerance, *, workload: str = "") -> CheckOutcome:
    worst_ratio = 1.0
    worst_abs = 0.0
    worst_rel = 0.0
    for name, o, r in _iter_outputs(out, ref):
        ratio, ma, mr = elementwise_pass_ratio(o, r, tol)
        worst_ratio = min(worst_ratio, ratio)
        worst_abs = max(worst_abs, ma)
        worst_rel = max(worst_rel, mr)
    passed = worst_ratio >= tol.matched_ratio
    detail = (f"matched_ratio={worst_ratio:.4f} (need >= {tol.matched_ratio}); "
              f"max_abs_err={worst_abs:.3e}, max_rel_err={worst_rel:.3e}")
    if tol.max_error_cap is not None and worst_abs > tol.max_error_cap:
        passed = False
        detail += f"; max_error_cap {tol.max_error_cap:.3e} exceeded"
    return CheckOutcome(f"tolerance{(':'+workload) if workload else ''}", passed, HARD, detail,
                        {"matched_ratio": worst_ratio, "max_abs_err": worst_abs})


def check_shape_dtype(out, ref) -> CheckOutcome:
    for name, o, r in _iter_outputs(out, ref):
        o = np.asarray(o)
        r = np.asarray(r)
        if o.shape != r.shape:
            return CheckOutcome("shape", False, HARD,
                                f"'{name}' shape {o.shape} != reference {r.shape}")
        if o.dtype != r.dtype:
            return CheckOutcome("dtype", False, HARD,
                                f"'{name}' dtype {o.dtype} != reference {r.dtype}")
    return CheckOutcome("shape_dtype", True, HARD, "shapes and dtypes match")


def check_finite(out) -> CheckOutcome:
    for name, o in (out.items() if isinstance(out, dict) else enumerate(out)):
        o = np.asarray(o)
        if not np.all(np.isfinite(o)):
            n = int(np.size(o) - np.count_nonzero(np.isfinite(o)))
            return CheckOutcome("finite", False, HARD, f"'{name}' has {n} non-finite values")
    return CheckOutcome("finite", True, HARD, "all outputs finite")


# --------------------------------------------------------------------------- #
# Differential adversarial probes
# --------------------------------------------------------------------------- #

def _flat(a) -> np.ndarray:
    if isinstance(a, dict):
        return np.concatenate([np.asarray(v).ravel().astype(np.float64) for v in a.values()])
    if isinstance(a, tuple):
        return np.concatenate([np.asarray(v).ravel().astype(np.float64) for v in a])
    return np.asarray(a).ravel().astype(np.float64)


def check_determinism(candidate: Runner, inputs: Dict[str, np.ndarray],
                      tol: Tolerance) -> CheckOutcome:
    a = _flat(candidate(_copy(inputs)))
    b = _flat(candidate(_copy(inputs)))
    same = a.shape == b.shape and bool(np.allclose(a, b, rtol=tol.rtol, atol=tol.atol))
    return CheckOutcome("determinism", same, SOFT,
                        "repeat run identical" if same else "nondeterministic output",
                        {})


def check_not_memoized(candidate: Runner, inputs_a, inputs_b, tol: Tolerance) -> CheckOutcome:
    """A candidate that ignores its input (memoisation / pointer cache) is rejected."""
    oa = _flat(candidate(_copy(inputs_a)))
    ob = _flat(candidate(_copy(inputs_b)))
    ia = _flat(inputs_a)
    ib = _flat(inputs_b)
    inputs_differ = bool(np.linalg.norm(ia - ib) > 1e-8)
    outputs_same = oa.shape == ob.shape and bool(np.allclose(oa, ob, rtol=1e-6, atol=1e-6))
    if inputs_differ and outputs_same:
        return CheckOutcome("no_memoization", False, HARD,
                            "different inputs produced identical outputs (input-ignoring / cached)")
    return CheckOutcome("no_memoization", True, HARD, "outputs respond to inputs")


def check_not_identity(candidate: Runner, inputs, ref, tol: Tolerance) -> CheckOutcome:
    """A candidate that returns one of its inputs is rejected (unless the reference agrees)."""
    out = candidate(_copy(inputs))
    of = _flat(out)
    rf = _flat(ref)
    if np.allclose(of, rf, rtol=tol.rtol, atol=tol.atol):
        return CheckOutcome("no_identity", True, HARD, "output differs from the reference inputs")
    for name, arr in inputs.items():
        af = _flat(arr)
        if af.shape == of.shape and np.allclose(of, af, rtol=tol.rtol, atol=tol.atol):
            return CheckOutcome("no_identity", False, HARD,
                                f"output equals input '{name}' while the reference does not")
    return CheckOutcome("no_identity", True, HARD, "output is not an input passthrough")


def check_degenerate_variance(candidate: Runner, inputs, ref, tol: Tolerance) -> CheckOutcome:
    """A constant / near-constant output where the reference varies is a work-skipping hack."""
    out = candidate(_copy(inputs))
    of, rf = _flat(out), _flat(ref)
    ref_var = float(np.var(rf))
    out_var = float(np.var(of))
    if ref_var > 1e-6 and out_var < 1e-9:
        return CheckOutcome("variance", False, HARD,
                            f"output variance {out_var:.2e} ~ 0 while reference varies "
                            f"({ref_var:.2e})")
    return CheckOutcome("variance", True, HARD, "output variance is non-degenerate")


# --------------------------------------------------------------------------- #
# Precision contract
# --------------------------------------------------------------------------- #

def check_output_dtype(out, contract: PrecisionContract) -> CheckOutcome:
    want = np.dtype(contract.output_dtype)
    for name, o in (out.items() if isinstance(out, dict) else enumerate(out)):
        if np.asarray(o).dtype != want:
            return CheckOutcome("output_dtype", False, HARD,
                                f"'{name}' dtype {np.asarray(o).dtype} != contract {want}")
    return CheckOutcome("output_dtype", True, HARD, f"output dtype == {want}")


def _rel_err(a, b) -> float:
    a, b = _flat(a), _flat(b)
    denom = max(float(np.linalg.norm(b)), 1e-12)
    return float(np.linalg.norm(a - b) / denom)


def check_precision_vs_oracle(candidate: Runner, reference: Runner,
                              reference_fp64: Runner, inputs,
                              tol: Tolerance, contract: PrecisionContract) -> CheckOutcome:
    """Reject a candidate that is materially *less precise* than the reference.

    Both the candidate and the (fp32-ish) reference are compared to the fp64 oracle.  If
    the candidate's error exceeds the reference's error by more than
    ``contract.oracle_error_slack``, the candidate has quietly narrowed precision -- it may
    still pass the tolerance, but it is not computing the reference operator.
    """
    out_c = candidate(_copy(inputs))
    out_r = reference(_copy(inputs))
    try:
        out_o = reference_fp64(_copy(inputs))
    except Exception as exc:  # oracle not available
        return CheckOutcome("precision_oracle", True, SOFT,
                            f"fp64 oracle unavailable ({exc.__class__.__name__}); skipped")
    err_c = max(_rel_err(v, (out_o[k] if isinstance(out_o, dict) else out_o)) 
                for k, v in (out_c.items() if isinstance(out_c, dict) else enumerate(out_c)))
    err_r = max(_rel_err(v, (out_o[k] if isinstance(out_o, dict) else out_o))
                for k, v in (out_r.items() if isinstance(out_r, dict) else enumerate(out_r)))
    # Expected error floor for a given accumulation width (~sqrt(n) * 2^-bits).
    n = max(_flat(out_c).size, 1)
    expected = np.sqrt(n) * (2.0 ** (-contract.min_accum_bits + 1))
    detail = (f"candidate rel-err {err_c:.3e}, reference rel-err {err_r:.3e}, "
              f"expected floor {expected:.3e}")
    if not contract.allow_downcast and err_c > max(contract.oracle_error_slack * err_r, 8 * expected):
        return CheckOutcome("precision_oracle", False, HARD,
                            detail + " -> internal precision below the reference")
    if err_c > 32 * expected:
        return CheckOutcome("precision_oracle", True, SOFT,
                            detail + " -> suspect, above the expected floor")
    return CheckOutcome("precision_oracle", True, HARD, detail)


# --------------------------------------------------------------------------- #
# Static reward-hack scan
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class HackPattern:
    name: str
    regex: str
    severity: str
    detail: str


HACK_PATTERNS: List[HackPattern] = [
    HackPattern("delegate_reference", r"\b(from|import)\s+ref\b|\bref_out\b|reference_model\b|"
                                     r"\bimport\s+reference\b", HARD,
                "candidate appears to import / call the reference implementation"),
    HackPattern("float8_narrow", r"float8|fp8|\.to\(\s*torch\.float8|__nv_fp8", HARD,
                "in-kernel fp8 narrowing (the #1 flagged SOL hack class)"),
    HackPattern("half_narrow", r"\.half\(\)|torch\.float16|__half\b|\.bfloat16\(\)", SOFT,
                "narrower compute dtype; verify the precision contract"),
    HackPattern("memoize", r"lru_cache|@cache\b|_cache\s*=|functools\.cache", HARD,
                "memoisation: output may not depend on the input"),
    HackPattern("id_keyed_cache", r"id\(\s*\w+\s*\)\s*\]|\{id\(", HARD,
                "pointer-keyed caching"),
    HackPattern("identity_return", r"return\s+inputs?\[|\breturn\s+x\b(?![_a-zA-Z0-9])", SOFT,
                "possible identity / no-op return"),
    HackPattern("clock_tamper", r"cudaDeviceSetLimit|cudaDeviceSetCacheConfig|clock\(\)|"
                                 r"__nanosleep|cudaDeviceSetSharedMemConfig", HARD,
                "clock / cache-config tampering"),
    HackPattern("self_allclose", r"allclose\s*\(", SOFT,
                "candidate runs its own allclose (check for a self-certifying cheat)"),
]


def scan_static(source: str, ref_symbols: Sequence[str] = ()) -> List[CheckOutcome]:
    outcomes: List[CheckOutcome] = []
    for pat in HACK_PATTERNS:
        if re.search(pat.regex, source):
            outcomes.append(CheckOutcome(f"static:{pat.name}", False, pat.severity, pat.detail))
    for sym in ref_symbols:
        if sym and re.search(rf"\b{re.escape(sym)}\b", source):
            outcomes.append(CheckOutcome(f"static:ref_symbol:{sym}", False, HARD,
                                         f"source mentions the reference symbol '{sym}'"))
    if not outcomes:
        outcomes.append(CheckOutcome("static", True, HARD, "no known reward-hack pattern found"))
    return outcomes


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

def _copy(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.array(v, copy=True) for k, v in inputs.items()}


@dataclass
class VerifyConfig:
    tolerance: Tolerance = Tolerance()
    contract: PrecisionContract = PrecisionContract()
    probes: Optional[List] = None            # argus.probes.Probe instances
    ref_symbols: Sequence[str] = ()
    require_determinism: bool = False        # determinism is a soft flag by default


def verify_kernel(
    *,
    candidate: Runner,
    reference: Runner,
    workloads: List[Dict[str, np.ndarray]],
    config: Optional[VerifyConfig] = None,
    source: Optional[str] = None,
    reference_fp64: Optional[Runner] = None,
    metamorphic_inputs: Optional[List[Dict[str, np.ndarray]]] = None,
) -> Verdict:
    """Run the full ARGUS verification stack and return a :class:`Verdict`.

    ``accepted`` is true iff every HARD check passes on every workload.
    """
    config = config or VerifyConfig()
    tol = config.tolerance
    outcomes: List[CheckOutcome] = []
    flags: List[str] = []

    if source is not None:
        stat = scan_static(source, config.ref_symbols)
        outcomes.extend(stat)
        flags.extend(o.name for o in stat if not o.passed)

    if not workloads:
        return Verdict(False, [CheckOutcome("workloads", False, HARD, "no workloads supplied")],
                       ["no_workloads"])

    # 1) Every workload: shape/dtype/finite + board tolerance.
    all_tol_ok = True
    for i, wl in enumerate(workloads):
        out = candidate(_copy(wl))
        ref = reference(_copy(wl))
        outcomes.append(check_shape_dtype(out, ref))
        outcomes.append(check_finite(out))
        oc = check_tolerance(out, ref, tol, workload=f"wl{i}")
        outcomes.append(oc)
        if not oc.passed:
            all_tol_ok = False
            flags.append(f"tolerance_fail_wl{i}")

    # 2) Adversarial probes on the primary workload.
    primary = workloads[0]
    ref0 = reference(_copy(primary))
    if config.require_determinism:
        outcomes.append(check_determinism(candidate, primary, tol))
    outcomes.append(check_not_identity(candidate, primary, ref0, tol))
    outcomes.append(check_degenerate_variance(candidate, primary, ref0, tol))

    # memoisation needs a second, different input
    alt = _perturb(primary)
    m = check_not_memoized(candidate, primary, alt, tol)
    outcomes.append(m)
    if not m.passed:
        flags.append("memoized")

    # 3) Precision contract.
    out0 = candidate(_copy(primary))
    outcomes.append(check_output_dtype(out0, config.contract))
    if reference_fp64 is not None:
        po = check_precision_vs_oracle(candidate, reference, reference_fp64, primary, tol,
                                       config.contract)
        outcomes.append(po)
        if not po.passed:
            flags.append("precision_downgrade" if po.severity == HARD else "precision_suspect")

    # 4) Metamorphic probes.
    if config.probes:
        for probe in config.probes:
            for mm_in in (metamorphic_inputs or [primary]):
                oc = probe.check(candidate, reference, mm_in, tol)
                outcomes.append(oc)
                if not oc.passed:
                    flags.append(f"probe:{probe.name}")

    accepted = all(o.passed for o in outcomes if o.severity == HARD)
    return Verdict(accepted, outcomes, sorted(set(flags)))


def _perturb(inputs: Dict[str, np.ndarray], seed: int = 1234) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    alt = {}
    for k, v in inputs.items():
        v = np.asarray(v)
        if np.issubdtype(v.dtype, np.floating):
            alt[k] = (v + rng.normal(0, 0.5, size=v.shape)).astype(v.dtype)
        elif np.issubdtype(v.dtype, np.integer):
            alt[k] = (v + rng.integers(1, 3, size=v.shape)).astype(v.dtype)
        else:
            alt[k] = np.array(v, copy=True)
    return alt
