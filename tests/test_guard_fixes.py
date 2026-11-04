"""Regression tests ported from the ForgeKernel fix wave (guard / eval batch).

This lineage carries almost none of the fix wave's guard mechanisms; every drop
is recorded here with its reason and, where the old test exposed a live bug, in
the BUG-PORT ledger. The one mechanism that survives verbatim in this package is
the precision stress record: the card template's stress rows measure real draws
(one row per stress profile, never a "nothing to move" row).

Dropped from the source file, with reasons:

* harness: tree-root python modules are harness (sitecustomize.py / json.py at
  the tree root hijacking a harness that runs with the workspace on PYTHONPATH)
  -- loop/harness_guard.py here keys harness on HARNESS_PREFIXES ("eval/",
  "gpu_entry.sh", "ref_bin/", ".gitattributes") only; the root-module glob is
  not in this lineage (BUG-PORT: the hijack vector is open);
* goal-ref: every import name / importlib.import_module / __import__ literal
  scanned -- library_findings here is an earlier, deliberately narrow heuristic
  (first import name only, no dynamic-import literals; its docstring states the
  limits), so the old assertions pin neither its question nor its scope;
* leaderboard: load_entry appending the kernel's directory (so a reference's
  lazy imports are not shadowed) -- load_entry here PREPENDS (BUG-PORT); the
  FlashInfer-Bench judge (fib_judge, FIB_DEFAULTS, SUITE_TASK) does not exist in
  this package (the card judge is benchkit's task protocol now);
* leaderboard: a task card's stress rows drawing the task's own make_inputs
  distribution -- superseded by design: gen_inputs accepts custom_inputs_fn and
  ignores it ("the historical signature"), the stress rides plain random draws
  and the task's make_inputs seeds are drawn by task_inputs;
* timing: _geomean degrading to None on a missing reading -- here None readings
  are silently FILTERED (an aggregate over a subset) and workloads_performance
  CRASHES (round(None)) on a non-positive one (BUG-PORT);
* precision: a length-mismatched measure being unavailable -- measure() zips the
  three lists and silently truncates to the shortest (BUG-PORT);
* timing_audit: the host-path note naming the thicker direction -- both
  directions here print "thinner" (BUG-PORT).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "src" / "peregrine_kernel" / "card_template" / "eval"


def _load_eval(name: str):
    """The card-template eval module under its own name (precision.py is copied into
    cards and loaded by path there too)."""
    spec = importlib.util.spec_from_file_location(f"pk_guard_fixes_{name}",
                                                  EVAL_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lb = _load_eval("benchkit")
precision = _load_eval("precision")


# --- the precision stress record measures real draws -------------------------------

TASK_JUDGE = """import torch

def run(x, w):
    y = x.to(torch.float64)
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * w.to(torch.float64)
    return y.to(x.dtype)
"""


def _task_problem() -> dict:
    defn = {"name": "rms", "axes": {"m": {"type": "var"}, "n": {"type": "var"}},
            "inputs": {"x": {"shape": ["m", "n"], "dtype": "float32"},
                       "w": {"shape": ["n"], "dtype": "float32"}},
            "outputs": {"y": {"shape": ["m", "n"], "dtype": "float32"}},
            "reference": TASK_JUDGE}
    return {"definition": defn,
            "workload": {"uuid": "s", "axes": {"m": 4, "n": 16},
                         "input_groups": [{"name": "plain", "kind": "generated", "seed": 1},
                                          {"name": "spiky", "kind": "generated", "seed": 2}]},
            "dps": False, "_dir": "."}


def test_the_precision_stress_record_measures_real_draws():
    problem = _task_problem()

    def kernel(x, w):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * w

    prec = precision.PrecisionRecorder()
    lb._precision_stress(problem, kernel, False, "cpu", prec)
    rows = [r for r in prec.rows if r.get("case") == "stress"]
    assert {r.get("stress") for r in rows} == set(precision.STRESS_PROFILES)
    assert all(r["status"] in ("pass", "fail") for r in rows), prec.rows
    # the regression this pins: not one row may say the stress had nothing to move
    assert not any("no plain random float input" in str(r.get("why")) for r in prec.rows)
