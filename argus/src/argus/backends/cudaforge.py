"""Production adapter to the upstream CudaForge repository.

ARGUS does not fork CudaForge; it drives the documented entry points per candidate and reads
the artifacts they already write.  Because the driver seam is a protocol, the adapter can be
exercised in ``--dry-run`` (command construction + artifact parsing, no GPU), which is the
only mode validated locally -- the live path needs the GPU.

Upstream entry points used (pinned ``eaba693f``):

* ``python3 main.py <task.py> --gpu <name> --model_name <m> --round 1 ...``
  -- one candidate generation + compile + correctness + timing; writes
  ``<work_dir>/<batch>/<task>/`` with ``evaluation/eval_*.json`` (metrics) and ``code/``.
* ``run_ncu.profile_bench(bench_py, out_csv)`` then ``load_ncu_metrics``/``metrics_to_prompt``
  -- the counter collection and rendering.
* ``utils/compile_and_run.compare_and_bench`` -- correctness (``allclose``) and paired timing.

The adapter passes the *diagnosed context* (ARGUS's causal block, or CudaForge's raw counter
JSON) into the Judge prompt; everything else is upstream behaviour.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..cudaforge import (CUDAFORGE_METRICS, JUDGE_SYSTEM, SEED_SYSTEM,
                         build_judge_optimization_prompt, build_optimization_prompt,
                         build_seed_prompt, extract_code_block, extract_json)
from ..roofline import PEAKS, GpuPeak
from ..search import EvalResult
from .base import BackendError


@dataclass
class CudaForgeBackend:
    """Drive upstream CudaForge for one task.

    ``dry_run=True`` builds the exact commands and returns prompt text instead of code, so
    the orchestration can be smoke-tested without a GPU or an LLM.
    """

    repo_dir: Path
    task_path: Path
    label: str                       # task name (label avoids clashing with task_name())
    op: str = "generic"              # operator class
    gpu: str = "B300"
    model_name: str = "claude-opus-5-5"
    server_type: str = "openai"
    device: int = 0
    warmup: int = 5
    repeat: int = 20
    cudaforge_tol: float = 1e-3
    dry_run: bool = True
    peak_name: str = "B300"
    _reference_source: str = ""
    _python: str = "python3"

    def __post_init__(self) -> None:
        self.repo_dir = Path(self.repo_dir)
        self.task_path = Path(self.task_path)
        if not self.dry_run and not (self.repo_dir / "main.py").exists():
            raise BackendError(f"upstream CudaForge not found at {self.repo_dir}/main.py")

    # ---- identity ----
    def task_name(self) -> str:
        return self.label

    def op_class(self) -> str:
        return self.op

    def peak(self) -> GpuPeak:
        return PEAKS[self.peak_name]

    def flops(self) -> float:
        return 0.0    # supplied by the NCU card in a live run; the model tolerates 0

    # ---- agent seam ----
    def seed(self, lineage: int) -> str:
        prompt = build_seed_prompt(self.label, self.gpu, self._reference_source, self.op)
        if self.dry_run:
            return f"// DRY-RUN seed prompt (lineage {lineage})\n// {prompt[:120]}..."
        return self._run_agent(prompt, SEED_SYSTEM, "seed")

    def propose(self, parent_code: str, mechanism: str, lineage: int, context: str) -> str:
        judge_prompt = build_judge_optimization_prompt(self.label, self.gpu, context, parent_code)
        if self.dry_run:
            return f"// DRY-RUN propose '{mechanism}' (lineage {lineage})"
        raw = self._run_agent(judge_prompt, JUDGE_SYSTEM, "judge_optimization")
        strategy = extract_json(raw) or {"mechanism": mechanism}
        opt_prompt = build_optimization_prompt(parent_code, self.gpu, strategy)
        return extract_code_block(self._run_agent(opt_prompt, SEED_SYSTEM, "optimization"))

    def _run_agent(self, prompt: str, system: str, call_type: str) -> str:
        # upstream exposes query_server(...); the live path shells out to a tiny runner.
        runner = self.repo_dir / "argus_runner.py"
        if not runner.exists():
            raise BackendError(
                f"live mode needs {runner}; run scripts/bootstrap.sh then see docs/B300_RUNBOOK.md"
            )
        proc = subprocess.run(
            [self._python, str(runner), "--server_type", self.server_type,
             "--model_name", self.model_name, "--call_type", call_type],
            input=prompt, text=True, capture_output=True, cwd=str(self.repo_dir), timeout=7200,
        )
        if proc.returncode != 0:
            raise BackendError(f"agent call failed: {proc.stderr[-2000:]}")
        return proc.stdout

    # ---- hardware seam ----
    def eval_kernel(self, code: str, fidelity: int) -> EvalResult:
        if self.dry_run:
            raise BackendError("eval_kernel needs a GPU; use the mock backend for local runs")
        return self._eval_live(code, fidelity)

    def _eval_live(self, code: str, fidelity: int) -> EvalResult:
        """Run upstream compare_and_bench on the candidate and parse the metrics JSON."""
        work = self.repo_dir / "argus_work"
        work.mkdir(exist_ok=True)
        kernel_file = work / "candidate.py"
        kernel_file.write_text(code, encoding="utf-8")
        repeat = self.repeat if fidelity >= 1 else max(3, self.repeat // 5)
        cmd = [self._python, "-c", _COMPARE_SNIPPET,
               str(kernel_file), str(self.task_path), str(self.device),
               str(self.warmup), str(repeat), str(self.cudaforge_tol)]
        proc = subprocess.run(cmd, text=True, capture_output=True, cwd=str(self.repo_dir),
                              timeout=600)
        if proc.returncode != 0:
            return EvalResult(float("inf"), 0.0, {}, accepted=False,
                              detail=proc.stderr[-1000:])
        metrics = json.loads(proc.stdout.strip().splitlines()[-1])
        test_ms = metrics["test_latency_ms"]["avg"]
        latency_s = test_ms / 1e3
        return EvalResult(latency_s, metrics.get("speedup", 0.0), {}, accepted=True,
                          detail="compare_and_bench ok")

    def profile(self, code: str) -> dict:
        if self.dry_run:
            return {}
        return self._profile_live(code)

    def _profile_live(self, code: str) -> dict:
        """Run NCU via upstream run_ncu.py and return the counter dict for the hot kernel."""
        work = self.repo_dir / "argus_work"
        work.mkdir(exist_ok=True)
        kernel_file = work / "candidate.py"
        kernel_file.write_text(code, encoding="utf-8")
        csv_out = work / "ncu.csv"
        snippet = (
            "import json,sys;from pathlib import Path;"
            "from run_ncu import profile_bench, load_ncu_metrics;"
            f"p=profile_bench(bench_py='{kernel_file}', out_csv='{csv_out}');"
            "df=load_ncu_metrics(p);"
            "print(json.dumps({str(k): (None if v!=v else v) for k,v in "
            "df.iloc[0].to_dict().items()}))"
        )
        proc = subprocess.run([self._python, "-c", snippet], text=True, capture_output=True,
                              cwd=str(self.repo_dir), timeout=1800)
        if proc.returncode != 0:
            return {}
        try:
            return {k: v for k, v in json.loads(proc.stdout.strip().splitlines()[-1]).items()
                    if v is not None}
        except Exception:
            return {}


_COMPARE_SNIPPET = (
    "import json,sys;from pathlib import Path;"
    "from utils.compile_and_run import compare_and_bench;"
    "r=compare_and_bench(ref_py=Path(sys.argv[2]), test_py=Path(sys.argv[1]),"
    " device_idx=int(sys.argv[3]), warmup=int(sys.argv[4]), repeat=int(sys.argv[5]),"
    " tol=float(sys.argv[6]));"
    "spd=r['ref_latency_ms']['avg']/max(1e-9,r['test_latency_ms']['avg']);"
    "r['speedup']=spd;print(json.dumps(r))"
)


def parse_eval_metrics(path: Path) -> Dict[str, float]:
    """Parse an upstream ``evaluation/eval_*.json`` artifact into a flat dict."""
    data = json.loads(Path(path).read_text())
    out: Dict[str, float] = {}
    for key in ("ref_latency_ms", "test_latency_ms"):
        if key in data and isinstance(data[key], dict):
            out[f"{key}.avg"] = float(data[key].get("avg", float("nan")))
    for key in ("max_abs_err", "matched_ratio", "tolerance"):
        if key in data:
            out[key] = float(data[key])
    return out


def metrics_subset(counters: Dict[str, float]) -> Dict[str, float]:
    """Keep only the CudaForge metric keys (for the raw-dump context)."""
    return {k: counters[k] for k in CUDAFORGE_METRICS if k in counters}
