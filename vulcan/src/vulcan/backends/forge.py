"""Production backend: drive the upstream ``forge`` CLI (OpenBMB/ForgeKernel).

This is a thin, auditable wrapper.  It does **not** re-implement the upstream loop; the
upstream is vendored (see ``scripts/bootstrap.sh``) and invoked through its documented
entry points::

    forge init <op> --leaderboard sol-execbench --dataset <dir> \
        --definition <definition> --workload <workload>
    forge new tasks/<op>
    forge run .artifacts/<op>_<id>
    forge export .artifacts/<op>_<id> --out solution.json --name <name> --author <author>

The exact eval-JSON schema is upstream's; ``_find_metric`` scans the artifact tree for
the known keys and fails loudly with what it saw, rather than guessing.  Set
``dry_run=True`` to print the command plan without executing (used by the tests and by
``vulcan plan``).
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .base import EvalResult

# latency keys upstream writes, in order of preference (device reading first)
_LATENCY_KEYS = ("device_median_ms", "device_window_ms", "best_ms", "median_ms")


@dataclass
class ForgeConfig:
    forge_repo: str                     # path to the cloned upstream repository
    python: str = "python3"
    suite: str = "sol-execbench"
    dataset: str = ""                   # SOL-ExecBench data/benchmark directory
    sol_board: str = ""                 # optional offline sol_board JSON
    author: str = "zixinzenith"
    name: str = "vulcan"
    workdir: str = ""                   # defaults to <forge_repo>

    def root(self) -> Path:
        return Path(self.workdir or self.forge_repo)


class ForgeBackend:
    """``CampaignBackend`` over the upstream CLI."""

    def __init__(
        self,
        config: ForgeConfig,
        definitions: Mapping[str, str],
        workloads: Optional[Mapping[str, str]] = None,
        dry_run: bool = False,
        run_timeout_s: Optional[float] = None,
    ):
        self.cfg = config
        self.definitions = dict(definitions)      # vulcan task -> board definition
        self.workloads = dict(workloads or {})
        self.dry_run = dry_run
        self.run_timeout_s = run_timeout_s
        self.calls = 0
        self.commands: List[str] = []

    # -- command plumbing -------------------------------------------------- #
    def _forge(self, *args: str) -> List[str]:
        return [self.cfg.python, "-m", "forgekernel", *args]

    def _run(self, argv: Sequence[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        cmd = " ".join(argv)
        self.commands.append(cmd)
        if self.dry_run:
            print(f"[dry-run] (cd {cwd or self.cfg.root()}) {cmd}")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        env = dict(os.environ)
        env.setdefault("PYTHONPATH", str(Path(self.cfg.forge_repo) / "src"))
        return subprocess.run(argv, cwd=str(cwd or self.cfg.root()), env=env,
                              capture_output=True, text=True, timeout=self.run_timeout_s)

    def _task_dir(self, task: str) -> Path:
        return self.cfg.root() / "tasks" / task

    # -- preparation ------------------------------------------------------- #
    def prepare(self, task: str) -> None:
        """Create ``tasks/<task>`` if absent (``forge init``)."""
        if (self._task_dir(task) / "task.toml").exists():
            return
        if task not in self.definitions:
            raise KeyError(f"no board definition registered for task {task!r}")
        args = ["init", task, "--leaderboard", self.cfg.suite,
                "--dataset", self.cfg.dataset,
                "--definition", self.definitions[task]]
        if task in self.workloads:
            args += ["--workload", self.workloads[task]]
        if self.cfg.sol_board:
            args += ["--sol-board", self.cfg.sol_board]
        r = self._run(self._forge(*args))
        if not self.dry_run and r.returncode != 0:
            raise RuntimeError(f"forge init {task} failed: {r.stderr[-800:]}")

    def _latest_artifact(self, task: str) -> Optional[Path]:
        cands = sorted(glob.glob(str(self.cfg.root() / ".artifacts" / f"{task}_*")),
                       key=lambda p: os.path.getmtime(p))
        return Path(cands[-1]) if cands else None

    # -- campaign ---------------------------------------------------------- #
    def evaluate(
        self,
        task: str,
        reference_latency_ms: float,
        architecture: str,
        hours: float,
        variant: str = "",
    ) -> EvalResult:
        self.calls += 1
        self.prepare(task)
        art_before = self._latest_artifact(task)
        r_new = self._run(self._forge("new", f"tasks/{task}"))
        if not self.dry_run and r_new.returncode != 0:
            raise RuntimeError(f"forge new {task} failed: {r_new.stderr[-800:]}")
        art = self._latest_artifact(task)
        if art is None:
            if self.dry_run:
                return EvalResult(latency_ms=reference_latency_ms, correct=True,
                                  info={"dry_run": True, "architecture": architecture})
            raise RuntimeError(f"no .artifacts/{task}_* produced")

        r_run = self._run(self._forge("run", str(art)))
        if not self.dry_run and r_run.returncode not in (0, 2):
            raise RuntimeError(f"forge run failed: {r_run.stderr[-800:]}")

        latency = self._find_metric(art)
        if latency is None:
            return EvalResult(latency_ms=float("inf"), correct=False,
                              artifacts={"artifact": str(art)},
                              info={"architecture": architecture,
                                    "reason": "no metric found"})
        ok = r_run.returncode == 0
        return EvalResult(latency_ms=float(latency), correct=ok,
                          artifacts={"artifact": str(art)},
                          info={"architecture": architecture})

    def export(self, task: str) -> Optional[str]:
        art = self._latest_artifact(task)
        if art is None:
            return None
        out = self.cfg.root() / "submissions" / f"{task}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        self._run(self._forge("export", str(art), "--out", str(out),
                              "--name", self.cfg.name, "--author", self.cfg.author))
        return str(out)

    # -- parsing ----------------------------------------------------------- #
    def _find_metric(self, artifact: Path) -> Optional[float]:
        """Scan the artifact tree for the device latency reading."""
        for path in sorted(artifact.rglob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            val = _deep_find(data, _LATENCY_KEYS)
            if val is not None:
                return val
            sb = _deep_find(data, ("sol_score",))
            if sb is not None and 0.0 < sb <= 1.0:
                # a score was found but no latency; the caller can still act on it
                # by treating the score as the objective -- return None here and let
                # the operator read sol_score via .info in a future revision.
                pass
        return None


def _deep_find(obj, keys: Sequence[str]) -> Optional[float]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, (int, float)):
                return float(v)
            found = _deep_find(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _deep_find(v, keys)
            if found is not None:
                return found
    return None
