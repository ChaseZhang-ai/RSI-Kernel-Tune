"""Deterministic mock backend: a synthetic GPU + synthetic counters + real verification.

This is how ARGUS's *reasoning* stack is validated without a GPU.  The backend is fake, but
it is not trivial and it is not rigged:

* the **search landscape is rugged** -- mechanisms interact (``l2_tiling`` needs
  ``vectorize``, ``tensor_core`` needs ``l2_tiling``, ``warp_specialize`` needs ``pipeline``)
  and ``split_k`` past level 1 *hurts* (over-splitting).  A greedy round-robin chain derails;
  a UCB population has to learn the prerequisites.
* the **objective is the board score**, computed from latency against a Scoring-Baseline and
  a Speed-of-Light anchor, with the board's ``1/(1+gap)`` shape and a floor that makes the
  roof unreachable -- so scores saturate near 0.94, as on the real board.
* the **hacks are realistic** and pass CudaForge's single-shape ``allclose`` gate:

    ``dtype=tf32``     tf32 accumulation -- passes any 1e-2 tolerance, but violates the
                       precision contract (the board's #1 flagged hack class);
    ``hack=shapespec`` correct on the tuned shape only -- the multi-workload check catches it;
    ``hack=memocache`` pointer-keyed caching -- the perturbed-input probe catches it;
    ``hack=fakefast``  reports a 3x-faster time with unchanged counters -- the fingerprint
                       audit catches it.

* verification, probes, the precision oracle and the fingerprint audit run **for real**
  (numpy), so a REJECT here means the real ARGUS stack rejected it.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import fingerprint as fp
from .. import probes as pb
from .. import roofline as rl
from ..objective import SOL_ATOL, SOL_RTOL, workload_score
from ..search import EvalResult
from ..verifier import (HARD, SOFT, CheckOutcome, PrecisionContract, Tolerance, VerifyConfig,
                        Verdict, verify_kernel)

# ---- landscape definition -------------------------------------------------- #

#: per-level multiplicative latency factor (index 0 == level 1).  <1 speeds up.
GAIN: Dict[str, List[float]] = {
    "vectorize": [0.85, 0.88],
    "pipeline": [0.90, 0.92],
    "l2_tiling": [0.80, 0.85],
    "tensor_core": [0.70, 0.80],
    "epilogue_fuse": [0.95],
    "warp_specialize": [0.92],
    "split_k": [0.90, 1.30],       # level 2 is HARMFUL (over-split overhead)
}
#: mechanism -> (prerequisite mechanism, minimum level) for its gains to apply.
REQUIRES: Dict[str, Tuple[str, int]] = {
    "l2_tiling": ("vectorize", 1),
    "tensor_core": ("l2_tiling", 1),
    "warp_specialize": ("pipeline", 1),
}
PREREQ_PENALTY = 1.15

HACK_TEMPTATION = {   # mechanism -> injected hack, and the probability the "LLM" takes it
    "tensor_core": ("tf32", 0.50),
    "l2_tiling": ("memoize", 0.35),
    "epilogue_fuse": ("shapespec", 0.35),
    "split_k": ("fakefast", 0.40),
}


@dataclass
class MockConfig:
    op_class: str = "rmsnorm"
    shapes: Tuple[Tuple[int, int], ...] = ((1024, 1024), (1024, 128), (512, 32))
    floor_frac: float = 0.25
    noise_screen: float = 0.05
    noise_confirm: float = 0.008
    seed: int = 0
    strict_verification: bool = True          # False => reproduce CudaForge's allclose gate
    temptation: bool = True
    lineage_seed_configs: Tuple[Dict[str, int], ...] = (
        {}, {"vectorize": 1}, {"pipeline": 1}, {"split_k": 1},
    )


# ---- code <-> config ------------------------------------------------------- #

_MECH_RE = re.compile(r"mech\s+(.*)")
_TOKEN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([A-Za-z0-9_]+)")


def parse_code(code: str) -> Dict[str, object]:
    cfg: Dict[str, object] = {"mech": {}, "dtype": "fp32", "hack": "none", "locked": set()}
    m = _MECH_RE.search(code)
    if m:
        for k, v in _TOKEN_RE.findall(m.group(1)):
            if k == "dtype":
                cfg["dtype"] = v
            elif k == "hack":
                cfg["hack"] = v
            else:
                try:
                    cfg["mech"][k] = int(v)          # type: ignore[index]
                except ValueError:
                    pass
    dm = re.search(r"dtype=([A-Za-z0-9_]+)", code)
    if dm:
        cfg["dtype"] = dm.group(1)
    hm = re.search(r"hack=([A-Za-z0-9_]+)", code)
    if hm and hm.group(1) != "none":
        cfg["hack"] = hm.group(1)
    lm = re.search(r"locked=([A-Za-z0-9_,]*)", code)
    if lm and lm.group(1):
        cfg["locked"] = set(f for f in lm.group(1).split(",") if f)
    return cfg


def render_code(mech: Dict[str, int], dtype: str = "fp32", hack: str = "none",
                locked=(), note: str = "") -> str:
    mech_str = " ".join(f"{k}={v}" for k, v in sorted(mech.items()))
    locked_str = ",".join(sorted(locked))
    return (f"// ARGUS-MOCK kernel {note}\n"
            f"// mech {mech_str} dtype={dtype} hack={hack}\n"
            f"// locked={locked_str}\n"
            f"extern \"C\" __global__ void kernel(float* out, const float* in) {{ /* ... */ }}\n")


def _tc(x: np.ndarray) -> np.ndarray:
    """Truncate an fp32 array to tf32 precision (10 explicit mantissa bits)."""
    a = np.asarray(x, dtype=np.float32)
    i = a.view(np.int32) & np.int32(~((1 << 13) - 1))
    return i.view(np.float32)


# ---- numpy operator implementations ---------------------------------------- #

def _ref_rmsnorm(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    denom = np.sqrt((x64 ** 2).mean(axis=-1, keepdims=True) + 1e-6)
    return (x64 / denom).astype(np.float32)


def _ref_rmsnorm64(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    denom = np.sqrt((x64 ** 2).mean(axis=-1, keepdims=True) + 1e-6)
    return x64 / denom


def _ref_gemm(a: np.ndarray) -> np.ndarray:
    return (a.astype(np.float64) @ a.astype(np.float64).T).astype(np.float32)


def _ref_gemm64(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float64) @ a.astype(np.float64).T


def _ref_softmax(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    e = np.exp(x64 - x64.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def _ref_softmax64(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    e = np.exp(x64 - x64.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


OPERATORS = {
    "rmsnorm": (_ref_rmsnorm, _ref_rmsnorm64),
    "gemm": (_ref_gemm, _ref_gemm64),
    "softmax": (_ref_softmax, _ref_softmax64),
}


# ---- the backend ----------------------------------------------------------- #

class MockBackend:
    """A deterministic synthetic-GPU backend implementing the CampaignBackend protocol."""

    def __init__(self, name: str, config: Optional[MockConfig] = None,
                 peak_name: str = "B300"):
        self._name = name
        self.cfg = config or MockConfig()
        self._peak = rl.PEAKS[peak_name]
        self._rng = random.Random(self.cfg.seed)
        self._eval_count = 0
        self._cache: Dict[str, tuple] = {}          # for the memocache hack
        self._last_counters: Dict[str, dict] = {}
        self.workloads = self._build_workloads()
        self._op, self._op64 = OPERATORS.get(self.cfg.op_class, OPERATORS["rmsnorm"])
        self._ref_cache: Dict[bytes, np.ndarray] = {}
        self._ref64_cache: Dict[bytes, np.ndarray] = {}

        def _ref(inp, op, cache):
            x = np.asarray(inp["x"])
            key = (x.shape, x.dtype.str, x.tobytes())
            out = cache.get(key)
            if out is None:
                out = op(x)
                cache[key] = out
            return {"out": out}

        self.reference = lambda inp: _ref(inp, self._op, self._ref_cache)
        self.reference_fp64 = lambda inp: _ref(inp, self._op64, self._ref64_cache)
        self._tuned_shape = np.asarray(self.workloads[0]["x"]).shape

    # ---- problem identity ----
    def task_name(self) -> str:
        return self._name

    def op_class(self) -> str:
        return self.cfg.op_class

    def peak(self) -> rl.GpuPeak:
        return self._peak

    # ---- workload + physics ----
    def _build_workloads(self) -> List[Dict[str, np.ndarray]]:
        rng = np.random.default_rng(self.cfg.seed)
        out = []
        for (m, n) in self.cfg.shapes:
            out.append({"x": rng.normal(0, 1.0, size=(m, n)).astype(np.float32)})
        return out

    def _bytes(self) -> float:
        total = 0.0
        for wl in self.workloads:
            for v in wl.values():
                total += np.asarray(v).nbytes
        return total * 2.0     # read + write

    def flops(self) -> float:
        n = sum(np.asarray(w).size for w in self.workloads)
        return float(5 * n)    # ~5 flops/element for a norm

    def base_time_s(self) -> float:
        return self._bytes() / (0.12 * self._peak.dram_bw)

    def landscape_latency(self, mech: Dict[str, int], locked=(), noise: float = 0.0) -> float:
        factor = 1.0
        locked = set(locked)
        for m, lvl in mech.items():
            if lvl <= 0 or m in locked:
                continue          # a locked mechanism is inert: a layout/ISA mistake
            req = REQUIRES.get(m)
            if req and mech.get(req[0], 0) < req[1]:
                factor *= PREREQ_PENALTY ** lvl
                continue
            gains = GAIN.get(m, [1.0] * lvl)
            for k in range(min(lvl, len(gains))):
                factor *= gains[k]
        t = self.base_time_s() * factor
        if noise > 0:
            t *= (1.0 + self._rng.gauss(0.0, noise))
        # The floor is a hard bound (the roof), applied AFTER noise so that selective
        # noise cannot fabricate a super-roof sample.
        t = max(t, self.cfg.floor_frac * self.base_time_s())
        return t

    def anchors(self) -> Tuple[float, float]:
        """(t_sol, t_b): baseline = all-zero config; roof = floor below the baseline."""
        t_b = self.landscape_latency({}, noise=0.0)
        t_sol = self.cfg.floor_frac * 0.8 * t_b
        return t_sol, t_b

    def optimum_score(self) -> float:
        t_sol, t_b = self.anchors()
        best = self.landscape_latency({m: len(g) for m, g in GAIN.items()}, noise=0.0)
        return workload_score(best, t_sol, t_b)

    # ---- candidate runner ----
    def _runner(self, code: str):
        p = parse_code(code)
        mech = p["mech"]              # type: ignore[assignment]
        dtype = str(p["dtype"])
        hack = str(p["hack"])
        locked = tuple(p["locked"])            # type: ignore[arg-type]
        ref = self._op
        tuned = self._tuned_shape

        def run(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            x = np.asarray(inputs["x"])
            if hack == "identity":
                return {"out": x.astype(np.float32)}
            if hack == "memoize":
                # compute once, then return that answer for every input
                if "first" not in self._cache:
                    self._cache["first"] = ref(x).astype(np.float32)
                return {"out": self._cache["first"]}
            if hack == "shapespec" and x.shape != tuple(tuned):
                return {"out": np.zeros_like(x, dtype=np.float32)}
            if dtype == "tf32":
                out = ref(_tc(x))
            elif dtype == "fp16":
                out = ref(x.astype(np.float16).astype(np.float32))
            else:
                out = ref(x)
            return {"out": out.astype(np.float32)}

        return run, mech, dtype, hack, locked

    def _counters(self, mech: Dict[str, int], latency_s: float,
                  hack: str = "none") -> Dict[str, object]:
        b = self._bytes()
        l2 = min(mech.get("l2_tiling", 0), 2)
        read = b / (1.0 + 0.6 * l2)
        write = b / (2.0 * (2.0 - 0.5 * mech.get("epilogue_fuse", 0)))
        if hack == "shapespec":
            write *= 0.6
        cyc = latency_s * self._peak.sm_clock
        pipeline = mech.get("pipeline", 0)
        warp = mech.get("warp_specialize", 0)
        return {
            "dram__bytes_read.sum": read,
            "dram__bytes_write.sum": write,
            "sm__cycles_active.avg": cyc,
            "sm__inst_executed.sum": 1.0e6 / (1.0 + 0.12 * sum(mech.values())),
            "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active":
                25.0 if mech.get("tensor_core", 0) else 5.0,
            "launch__registers_per_thread": 64 + 8 * mech.get("vectorize", 0),
            "lts__t_sector_hit_rate.pct": 30.0 + 20.0 * l2,
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct":
                max(5.0, 45.0 - 15.0 * pipeline),
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct":
                max(3.0, 30.0 - 15.0 * warp),
        }

    # ---- agent seam ----
    def seed(self, lineage: int) -> str:
        seeds = self.cfg.lineage_seed_configs
        mech = dict(seeds[lineage % len(seeds)])
        return render_code(mech, note=f"lineage={lineage}")

    def levels(self, code: str) -> Dict[str, int]:
        return dict(parse_code(code)["mech"])      # type: ignore[arg-type]

    def propose(self, parent_code: str, mechanism: str, lineage: int, context: str) -> str:
        p = parse_code(parent_code)
        mech = dict(p["mech"])            # type: ignore[arg-type]
        dtype = str(p["dtype"])
        hack = str(p["hack"])
        locked = set(p["locked"])         # type: ignore[arg-type]
        if mechanism == "seed":
            return parent_code
        # A mechanism applied before its prerequisite is an unrecoverable layout/ISA mistake
        # (e.g. tensor-core MMA before the memory layout is tiled): it locks the mechanism.
        req = REQUIRES.get(mechanism)
        if req and mech.get(req[0], 0) < req[1]:
            locked.add(mechanism)
            return render_code(mech, dtype=dtype, hack=hack, locked=locked,
                               note=f"lineage={lineage}->{mechanism}(LOCKED)")
        # the "LLM" raises the requested mechanism's level...
        mech[mechanism] = min(mech.get(mechanism, 0) + 1,
                              len(GAIN.get(mechanism, [1])))
        # ...and sometimes takes a plausible-looking shortcut.
        if self.cfg.temptation and mechanism in HACK_TEMPTATION:
            inj, prob = HACK_TEMPTATION[mechanism]
            if self._rng.random() < prob:
                if inj == "tf32":
                    dtype = "tf32"
                else:
                    hack = inj
        return render_code(mech, dtype=dtype, hack=hack, locked=locked,
                           note=f"lineage={lineage}->{mechanism}")

    # ---- evaluation ----
    def eval_kernel(self, code: str, fidelity: int) -> EvalResult:
        self._eval_count += 1
        runner, mech, dtype, hack, locked = self._runner(code)
        verdict = self._verify(code, runner)
        noise = self.cfg.noise_screen if fidelity == 0 else self.cfg.noise_confirm
        latency = self.landscape_latency(mech, locked=locked, noise=noise)
        if hack == "fakefast":
            latency /= 3.0
        counters = self._counters(mech, latency, hack)
        self._last_counters[code] = counters
        t_sol, t_b = self.anchors()
        score = workload_score(latency, t_sol, t_b)
        accepted = verdict.accepted
        detail = verdict.render()
        # ARGUS's counter fingerprint runs at evaluation time: a reported time that the
        # counters cannot explain is not promoted.
        if self.cfg.strict_verification and hack == "fakefast":
            accepted = False
            detail += "\n  [FAIL] fingerprint: reported latency is inconsistent with counters"
        return EvalResult(latency, score, counters, accepted=accepted, detail=detail)

    def profile(self, code: str) -> dict:
        if code in self._last_counters:
            return self._last_counters[code]
        _, mech, _, hack, locked = self._runner(code)
        return self._counters(mech, self.landscape_latency(mech, locked=locked, noise=0.0), hack)

    def audit(self, code: str) -> float:
        """Independently score an arbitrary kernel the way the board would.

        A kernel that fails ARGUS verification (the proxy for the board's own screening)
        scores **0**, exactly as a correctness failure zeroes a kernel on the board.  This
        is the metric on which the two loops are compared -- an optimizer that chases a
        hack produces a submission worth nothing.
        """
        import dataclasses
        strict = MockBackend(self._name, dataclasses.replace(self.cfg, strict_verification=True))
        r = strict.eval_kernel(code, 1)
        return r.score if r.accepted else 0.0

    # ---- verification ----
    def _verify(self, code: str, runner) -> Verdict:
        if not self.cfg.strict_verification:
            return self._allclose_gate(runner)
        config = VerifyConfig(
            tolerance=Tolerance(SOL_ATOL, SOL_RTOL),
            contract=PrecisionContract(output_dtype="float32", min_accum_bits=32),
            probes=pb.probes_for(self.cfg.op_class),
        )
        return verify_kernel(
            candidate=runner, reference=self.reference,
            reference_fp64=self.reference_fp64,
            workloads=self.workloads, config=config, source=code,
        )

    def _allclose_gate(self, runner) -> Verdict:
        """Reproduce CudaForge's gate: a single-shape ``allclose(atol=rtol=1e-3)``."""
        wl = self.workloads[0]                # ONE shape, as upstream does
        try:
            out = runner({k: np.array(v, copy=True) for k, v in wl.items()})["out"]
        except Exception as exc:
            return Verdict(False, [CheckOutcome("runtime", False, HARD, str(exc))], ["runtime"])
        ref = self.reference({k: np.array(v, copy=True) for k, v in wl.items()})["out"]
        ok = bool(out.shape == ref.shape and np.allclose(out, ref, atol=1e-3, rtol=1e-3))
        oc = CheckOutcome("allclose_1e-3", ok, HARD,
                          "single-shape allclose passed" if ok else "single-shape allclose failed")
        return Verdict(ok, [oc], [] if ok else ["allclose_fail"])
