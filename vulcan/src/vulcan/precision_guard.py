"""Correctness / precision guard: make accuracy a hard constraint, not a knob.

The upstream ablation is a warning worth taking seriously: an attention-backward
implementation **without the milestone template** passed the output-error check yet
"lowers internal precision" and scored +16.2% -- i.e. it paid accuracy for speed and
the surface check did not notice.  On the board that is not just unsporting, it is
fragile: SOL-ExecBench enforces a per-workload tolerance *and* a ``max_error_cap``,
and a kernel that fails one workload zeroes its kernel (geometric mean) and loses its
``Fast_1`` count.

VULCAN encodes the precision *contract* explicitly and gates on it:

* ``PrecisionContract`` -- the accumulation/output precision the operator's accuracy
  budget requires (derived from the tolerance and the op class).
* ``check_contract`` -- reject a candidate whose accumulation precision is below the
  contract, even if its elementwise error currently passes.
* ``adversarial_probes`` -- inputs that expose approximation shortcuts (correlated
  keys, long memory, near-cancellation, wide dynamic range) -- the paper adds these
  for GDN/KDA precisely because a fixed absolute tolerance passes trivially on small
  outputs.
* ``reward_hack_scan`` -- static heuristics for the classic cheats (calling the
  reference, returning the input, id-based caching, in-kernel precision downcast,
  tampering with the clock/allocator).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Precision ordering for *accumulation*: higher is more accurate.
ACCUM_RANK = {
    "fp8": 1, "int8": 1, "fp16": 2, "bf16": 2, "tf32": 3, "fp32": 4, "fp64": 5,
}
# Ordering for *storage* (bit-width, for exactness checks).
STORE_RANK = {
    "bool": 0, "int8": 1, "fp8": 2, "fp16": 3, "bf16": 3, "tf32": 4,
    "fp32": 5, "int32": 5, "fp64": 6,
}


def _rank(table: Dict[str, int], dtype: str) -> int:
    d = dtype.lower().replace("torch.", "")
    if d not in table:
        raise ValueError(f"unknown dtype {dtype!r}")
    return table[d]


@dataclass(frozen=True)
class Tolerance:
    atol: float = 1e-2
    rtol: float = 1e-2
    match_ratio: float = 1.0
    max_error_cap: Optional[float] = None

    def __post_init__(self) -> None:
        if self.atol < 0 or self.rtol < 0:
            raise ValueError("tolerances must be >= 0")
        if not (0.0 < self.match_ratio <= 1.0):
            raise ValueError("match_ratio must be in (0, 1]")


@dataclass(frozen=True)
class PrecisionContract:
    """What precision the operator requires to stay within tolerance."""

    input_dtype: str
    accum_dtype: str
    output_dtype: str
    exact: bool = False          # bit-exact required (pack/unpack, index ops)

    def describe(self) -> str:
        return (f"in={self.input_dtype} accum={self.accum_dtype} "
                f"out={self.output_dtype}" + (" [bit-exact]" if self.exact else ""))


@dataclass
class Finding:
    code: str
    severity: str                # "fail" | "warn" | "info"
    message: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity,
                "message": self.message, "detail": self.detail}


@dataclass
class ProbeSpec:
    name: str
    description: str
    catches: str


@dataclass
class Gate:
    verdict: str                 # "PASS" | "FAIL"
    findings: List[Finding] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"verdict": self.verdict,
                "findings": [f.as_dict() for f in self.findings]}


def default_contract(op_class: str) -> PrecisionContract:
    """A reasonable starting contract per operator class (override per task)."""
    table = {
        "norm": PrecisionContract("bf16", "fp32", "bf16"),
        "rmsnorm": PrecisionContract("bf16", "fp32", "bf16"),
        "layernorm": PrecisionContract("bf16", "fp32", "bf16"),
        "gemm": PrecisionContract("fp16", "fp32", "fp16"),
        "fp8_gemm": PrecisionContract("fp8", "fp32", "bf16"),
        "attention": PrecisionContract("bf16", "fp32", "bf16"),
        "attention_fp8": PrecisionContract("fp8", "fp32", "bf16"),
        "softmax": PrecisionContract("bf16", "fp32", "bf16"),
        "moe": PrecisionContract("fp8", "fp32", "bf16"),
        "pack": PrecisionContract("int8", "int32", "int8", exact=True),
        "unpack": PrecisionContract("int8", "int32", "int8", exact=True),
        "index": PrecisionContract("fp32", "fp32", "int32", exact=True),
        "linear_attention": PrecisionContract("bf16", "fp32", "bf16"),
    }
    return table.get(op_class, PrecisionContract("bf16", "fp32", "bf16"))


def check_contract(
    candidate: PrecisionContract,
    required: PrecisionContract,
) -> List[Finding]:
    """Flag precision downgrades relative to the contract."""
    findings: List[Finding] = []

    if required.exact:
        if candidate != required:
            findings.append(Finding(
                "exactness", "fail",
                "operator requires bit-exact arithmetic but candidate differs",
                f"required {required.describe()}, candidate {candidate.describe()}",
            ))
        return findings

    if _rank(ACCUM_RANK, candidate.accum_dtype) < _rank(ACCUM_RANK, required.accum_dtype):
        findings.append(Finding(
            "accum-precision", "fail",
            "accumulation precision below the operator's accuracy contract",
            f"accum {candidate.accum_dtype} < required {required.accum_dtype}; "
            "this is the 'passes output check, lowers internal precision' failure mode",
        ))
    if _rank(STORE_RANK, candidate.output_dtype) < _rank(STORE_RANK, required.output_dtype):
        findings.append(Finding(
            "output-precision", "fail",
            f"output dtype {candidate.output_dtype} narrower than required "
            f"{required.output_dtype}",
        ))
    if _rank(STORE_RANK, candidate.input_dtype) < _rank(STORE_RANK, required.input_dtype):
        findings.append(Finding(
            "input-precision", "warn",
            f"input dtype {candidate.input_dtype} narrower than declared "
            f"{required.input_dtype}",
        ))
    return findings


def adversarial_probes() -> List[ProbeSpec]:
    """Input families that expose approximation shortcuts a tolerance hides."""
    return [
        ProbeSpec("correlated_keys",
                  "keys/queries drawn from a low-rank subspace so a dropped term "
                  "changes the result stiffly",
                  "state-decay shortcuts, sparse-attention index pruning"),
        ProbeSpec("long_memory",
                  "very long sequence with slowly decaying state",
                  "truncated recurrent state / windowed approximations"),
        ProbeSpec("near_cancellation",
                  "inputs constructed so two large opposing terms nearly cancel",
                  "term-dropping, low-precision reordering, fast-math reassociation"),
        ProbeSpec("wide_dynamic_range",
                  "magnitudes spanning several decades in one tensor",
                  "fp16 overflow, fp8 saturation, scale mis-selection"),
        ProbeSpec("zero_output_guard",
                  "reference non-zero everywhere; a solution of all zeros is wrong",
                  "lazy/uninitialised outputs (a board rule, not a tolerance)"),
        ProbeSpec("negative_inf_parity",
                  "-inf must appear at the same positions as the reference",
                  "masked-softmax handling that silently changes -inf tiles"),
    ]


_DENY_PATTERNS: List[Tuple[str, str, str]] = [
    (r"\btorch\.nn\.functional\b|\bF\.(rms_norm|softmax|scaled_dot_product|layer_norm|gelu)\b",
     "calls-torch-op", "delegates to a torch reference/functional op"),
    (r"\bfrom\s+flashinfer\b|\bimport\s+flashinfer\b|\bflash_attn\b",
     "calls-flashinfer", "names an expert reference library"),
    (r"\bflash_mla\b|\bdeep_gemm\b|\bdeepep\b",
     "calls-expert-lib", "names an expert reference library"),
    (r"\.to\(\s*(?:torch\.)?(?:bfloat16|float16|half)\s*\)\s*#\s*accum",
     "accum-downcast", "explicit accumulation downcast"),
    (r"\breturn\s+\w*_?x\b\s*$", "returns-input", "may return the input unchanged"),
    (r"\bid\s*\(\s*[A-Za-z_]\w*\s*\)\s*(?:as|:)", "id-caching",
     "caches on Python object identity"),
    (r"\bos\.environ\b|\bctypes\b.*nvidia|\bnvidia-smi\b", "env-clock-tamper",
     "touches environment / clock state"),
    (r"\btorch\.empty\s*\(", "uninitialised-output", "torch.empty output"),
    (r"L2_CACHE|persisting_l2|access_policy_window", "l2-tamper",
     "manipulates the L2 persistence policy used by the stopwatch"),
]


def reward_hack_scan(source: str) -> List[Finding]:
    """Static heuristics for leaderboard cheats.  Advisory, not a proof."""
    findings: List[Finding] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        for pattern, code, message in _DENY_PATTERNS:
            if re.search(pattern, line):
                findings.append(Finding(code, "warn", message,
                                        f"line {lineno}: {line.strip()[:120]}"))
    return findings


def evaluate_gate(
    candidate: PrecisionContract,
    required: PrecisionContract,
    source: str = "",
    probe_failures: Optional[Sequence[str]] = None,
) -> Gate:
    """Combine contract, static scan and probe outcomes into one verdict.

    Failures are contract violations or failed probes.  Hack-scan hits are warnings
    that a human/agent must clear before submission.
    """
    findings: List[Finding] = []
    findings += [f for f in check_contract(candidate, required) if f.severity == "fail"]
    for name in probe_failures or []:
        findings.append(Finding("probe-failed", "fail", f"adversarial probe failed: {name}"))
    if source:
        findings += reward_hack_scan(source)
    verdict = "FAIL" if any(f.severity == "fail" for f in findings) else "PASS"
    return Gate(verdict, findings)
