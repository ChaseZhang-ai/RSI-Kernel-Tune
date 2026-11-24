"""Faithful reproduction of the CudaForge agent architecture, plus the ARGUS loop.

Upstream: ``OptimAI-Lab/CudaForge`` (MIT), pinned at
``eaba693fa01bdafc9848334cada2d789e240ce3f``.  CudaForge is a **training-free two-agent**
system: a *Coder* writes CUDA C++ for a KernelBench-style task and a *Judge* reads the
Nsight-Compute counters and prescribes the single highest-impact optimisation; the two
iterate.  This module reproduces that loop exactly enough to be a control, and adds the
ARGUS loop on the same seam (see ``ARCHITECTURE.md`` for the paper<->code map).

Reproduced, faithfully
----------------------
* the per-round state machine: round 0 seeds; a *non-runnable* candidate goes to a
  repair path (identify-the-problem -> rewrite), a runnable one goes to the optimisation
  path (profile -> Judge -> rewrite);
* the single greedy chain from :func:`argus.search.greedy_chain` (parent = latest, not
  best);
* the optimizer's objective, ``score = ref_latency / test_latency`` (raw speedup);
* the Nsight-Compute metric set CudaForge profiles with (:data:`CUDAFORGE_METRICS`);
* correctness as a plain ``allclose`` with a single tolerance (:data:`CUDAFORGE_DEFAULT_TOL`).

Changed (and flagged)
---------------------
* the loop is driven through the :class:`~argus.backends.base.CampaignBackend` seam so it
  can run against a mock on a CPU, and against the real CLI in production;
* prompt *text* is rewritten (upstream prompts are not copied verbatim); the *contracts*
  they carry are preserved and documented.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import attribution as at
from . import fingerprint as fp
from . import probes as pb
from . import roofline as rl
from .memory import MechanismMemory, MechanismRecord
from .objective import SOL_ATOL, SOL_RTOL, workload_score
from .search import (MECHANISMS, EvalResult, PopulationSearch, SearchConfig, SearchResult,
                     greedy_chain)
from .verifier import PrecisionContract, Tolerance, VerifyConfig, verify_kernel

CUDAFORGE_URL = "https://github.com/OptimAI-Lab/CudaForge"
CUDAFORGE_PIN = "eaba693fa01bdafc9848334cada2d789e240ce3f"
CUDAFORGE_LICENSE = "MIT (c) 2025 Zijian Zhang"
CUDAFORGE_DEFAULT_TOL = 1e-3
CUDAFORGE_DEFAULT_ROUNDS = 10
CUDAFORGE_DEFAULT_WARMUP = 5
CUDAFORGE_DEFAULT_REPEAT = 20

#: The exact metric list from upstream ``run_ncu.py`` (24 counters).
CUDAFORGE_METRICS: Tuple[str, ...] = (
    "sm__cycles_active.avg",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__registers_per_thread",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.per_second",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "lts__throughput.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
)


# --------------------------------------------------------------------------- #
# Prompt builders (contract-preserving; text rewritten)
# --------------------------------------------------------------------------- #

SEED_SYSTEM = (
    "You are a senior CUDA performance engineer. Write a complete, runnable CUDA C++ "
    "kernel for the given PyTorch module on the given GPU. Prefer correctness first, then "
    "the obvious memory/compute wins for the target architecture. Output one fenced block."
)

JUDGE_SYSTEM = (
    "You are a senior CUDA performance engineer. Read the target GPU, the reference "
    "module, the current candidate, and the profiling evidence. Identify EXACTLY ONE "
    "highest-impact bottleneck, propose EXACTLY ONE optimisation method, and give a "
    "minimal modification plan. Return strict JSON: "
    '{"bottleneck": str, "method": str, "mechanism": str, "plan": [str, ...], '
    '"expected_gain": float}. Output one JSON object, nothing else.'
)

REPAIR_SYSTEM = (
    "You are a CUDA debugging expert. Given an error log and the failing kernel, return "
    'strict JSON {"problem": str, "fix": str}, then rewrite the kernel.'
)


def build_seed_prompt(task_name: str, gpu_name: str, reference_source: str,
                      op_class: str) -> str:
    ref = reference_source if reference_source else "(reference hidden; infer from the task)"
    return (f"# Task: {task_name}\n# Target GPU: {gpu_name}\n# Operator class: {op_class}\n\n"
            f"## Reference PyTorch module\n```python\n{ref}\n```\n\n"
            "Write the optimized CUDA C++ implementation.")


def build_judge_optimization_prompt(task_name: str, gpu_name: str, metrics_block: str,
                                    cuda_code: str) -> str:
    return (f"# Task: {task_name}\n# Target GPU: {gpu_name}\n\n"
            f"## Profiling evidence\n{metrics_block}\n\n"
            f"## Current candidate\n```cuda\n{cuda_code}\n```\n\n"
            "Return the single highest-impact optimisation as strict JSON.")


def build_optimization_prompt(cuda_code: str, gpu_name: str, strategy: dict) -> str:
    return (f"# Target GPU: {gpu_name}\n## Current candidate\n```cuda\n{cuda_code}\n```\n\n"
            f"## Judge strategy\n```json\n{json.dumps(strategy, indent=2)}\n```\n\n"
            "Apply exactly this change and return the full updated kernel in one fenced block.")


def build_repair_prompts(error_log: str, reference_source: str, cuda_code: str) -> Tuple[str, str]:
    problem = (f"## Error log\n{error_log}\n\n## Reference\n```python\n{reference_source}\n```\n\n"
               f"## Failing kernel\n```cuda\n{cuda_code}\n```\n\n"
               'Return strict JSON {"problem": str, "fix": str}.')
    repair = (f"## Error log\n{error_log}\n\n## Failing kernel\n```cuda\n{cuda_code}\n```\n\n"
              "Rewrite the kernel so it compiles, runs, and passes tolerance.")
    return REPAIR_SYSTEM, problem, repair


# ---- parsers (reproduced behaviour) ----

def extract_code_block(text: str) -> str:
    m = re.search(r"```(?:cuda|cpp|c\+\+)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def extract_json(raw: str):
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    candidate = m.group(1) if m else raw
    try:
        return json.loads(candidate)
    except Exception:
        m2 = re.search(r"\{[\s\S]*\}", raw)
        if m2:
            try:
                return json.loads(m2.group(0))
            except Exception:
                pass
    return {}


def extract_cuda_kernel_names(source: str) -> List[str]:
    names = re.findall(r"__global__\s+void\s+([A-Za-z_]\w*)\s*\(", source)
    names += re.findall(r"__global__\s+__launch_bounds__\s*\([^)]*\)\s*void\s+([A-Za-z_]\w*)\s*\(", source)
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


# --------------------------------------------------------------------------- #
# Run configuration + result
# --------------------------------------------------------------------------- #

@dataclass
class RunConfig:
    rounds: int = CUDAFORGE_DEFAULT_ROUNDS
    tolerance: Tolerance = field(default_factory=Tolerance)     # board tolerance (ARGUS)
    cudaforge_tol: float = CUDAFORGE_DEFAULT_TOL                # upstream allclose tol
    precision: PrecisionContract = field(default_factory=PrecisionContract)
    gpu_name: str = "B300"
    use_probes: bool = True
    use_fingerprint: bool = True
    memory: Optional[MechanismMemory] = None


@dataclass
class RunResult:
    mode: str
    task: str
    best_code: str
    best_score: float
    best_latency_s: float
    n_evals: int
    curve: List[float]
    flags: List[str] = field(default_factory=list)
    search: Optional[SearchResult] = None


# --------------------------------------------------------------------------- #
# Loops
# --------------------------------------------------------------------------- #

def _verify_config(backend, config: RunConfig) -> VerifyConfig:
    probes = pb.probes_for(backend.op_class()) if config.use_probes else None
    return VerifyConfig(tolerance=config.tolerance, contract=config.precision, probes=probes)


def run_cudaforge_loop(backend, config: RunConfig) -> RunResult:
    """ARGUS *control*: CudaForge's loop as-is (single chain, raw-counter context, allclose).

    The verification gate is bypassed to upstream behaviour (a single tolerance check is
    delegated to the backend's ``eval_kernel`` in CudaForge mode), and the search is the
    faithful greedy chain.
    """
    ctx_holder: Dict[str, dict] = {"prev": None}

    def seed_fn(lineage: int) -> str:
        return backend.seed(lineage)

    def mutate_fn(parent_code: str, mechanism: str, lineage: int) -> str:
        # CudaForge's Judge reads the RAW counter dump, not a causal block.
        counters = ctx_holder.get("prev") or {}
        context = json.dumps({k: counters.get(k) for k in CUDAFORGE_METRICS if k in counters},
                             default=str)
        return backend.propose(parent_code, mechanism, lineage, context)

    def eval_fn(code: str, fidelity: int) -> EvalResult:
        res = backend.eval_kernel(code, fidelity)
        ctx_holder["prev"] = backend.profile(code)
        return res

    sr = greedy_chain(seed_fn, mutate_fn, eval_fn, mechanisms=MECHANISMS,
                      max_evals=config.rounds)
    best = sr.best
    return RunResult("cudaforge", backend.task_name(),
                     best.code if best else "", best.score if best else float("-inf"),
                     best.result.latency_s if best else float("inf"),
                     sr.n_evals, sr.curve, [], sr)


def run_argus_loop(backend, config: RunConfig, search_config: Optional[SearchConfig] = None) -> RunResult:
    """The ARGUS loop: the same two agents, wrapped in verification + attribution + search."""
    peak = backend.peak()
    flops = backend.flops()
    prev_holder: Dict[str, Optional[dict]] = {"counters": None, "latency": None}
    op_class = backend.op_class()

    prior = {}
    if config.memory is not None:
        prior = config.memory.prior_for(op_class, "balanced", MECHANISMS)

    def context_for(counters: Optional[dict]) -> str:
        if not counters:
            return "## No profile yet; this is the first evaluation of the lineage."
        return at.build_judge_block(counters, peak, flops,
                                    prev_counters=prev_holder["counters"], kernel_name="kernel")

    def seed_fn(lineage: int) -> str:
        return backend.seed(lineage)

    def mutate_fn(parent_code: str, mechanism: str, lineage: int) -> str:
        context = context_for(prev_holder["counters"])
        return backend.propose(parent_code, mechanism, lineage, context)

    def eval_fn(code: str, fidelity: int) -> EvalResult:
        res = backend.eval_kernel(code, fidelity)
        if res.counters:
            prev_holder["counters"] = res.counters
            prev_holder["latency"] = res.latency_s
        return res

    sc = search_config or SearchConfig(n_lineages=4, mechanisms=list(MECHANISMS),
                                       max_evals=config.rounds)
    # Model-based veto: never spend a GPU evaluation on a mechanism a lineage is not yet
    # ordered to apply (it would lock the kernel into a worse ceiling).
    if sc.arm_filter is None and hasattr(backend, "levels"):
        sc.arm_filter = lambda m, code: rl.prereq_satisfied(m, backend.levels(code))
    searcher = PopulationSearch(sc, seed_fn, mutate_fn, eval_fn, transfer_prior=prior)
    sr = searcher.run()

    flags: List[str] = []
    if config.use_fingerprint:
        for o in searcher.fingerprint_flags(peak, flops):
            if not o.passed:
                flags.append(o.name)
    best = sr.best
    if config.memory is not None and best is not None:
        bottleneck = "balanced"
        try:
            if best.result and best.result.counters:
                bottleneck = rl.counters_to_point(best.result.counters, peak, flops).bottleneck
        except Exception:
            pass
        config.memory.add(MechanismRecord(op_class, bottleneck, best.mechanism, 1.0,
                                          task=backend.task_name()))
    return RunResult("argus", backend.task_name(),
                     best.code if best else "", best.score if best else float("-inf"),
                     best.result.latency_s if best else float("inf"),
                     sr.n_evals, sr.curve, flags, sr)


def solution_score(latency_s: float, t_sol_s: float, t_b_s: float) -> float:
    """Board score of a candidate given its latency and the two anchors (seconds)."""
    return workload_score(latency_s, t_sol_s, t_b_s)
