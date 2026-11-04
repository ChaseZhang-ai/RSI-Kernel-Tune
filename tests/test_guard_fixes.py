"""Regression tests ported from the audit snapshot's guard-and-eval fix batch.

Only the assertions that hold in today's semantics are kept here; the rest of
the snapshot file is superseded:

* the tree-root shadow-module guard (harness_guard ROOT_PY_PATHSPEC) and the
  multi-name / importlib library scan are ledger BUG-SEED-07 / BUG-SEED-08,
  both alive here and owned by another fix line — their regression tests wait
  for those fixes;
* ``load_entry`` still prepends the kernel directory to sys.path
  (BUG-SEED-09, alive, owned elsewhere);
* there is no external-benchmark judge in this tree (the snapshot's fib
  judge assertions have no referent in the shipped card template);
* the WP20 rework has gen_inputs accept-and-ignore a custom make_inputs
  callable by documented design (the task's make_inputs channel is
  task_inputs), so the snapshot's "stress rows draw make_inputs" contract is
  gone; what remains testable — a task card's precision stress rows are
  measured, never dismissed as "no plain random float input" — is ported;
* timing's aggregate-degrade contract (a missing reading degrades the geomean
  to None instead of averaging a subset) is not what today's _geomean does,
  and the precision length-mismatch and host_path_note direction fixes
  (BUG-SEED-24 / BUG-SEED-23) are alive — those tests wait for their fixes.

Needs torch (skipped without it). The eval modules are loaded by path, the way
a generated card loads them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TEMPLATE_EVAL = REPO / "src" / "peregrine_kernel" / "card_template" / "eval"

torch = pytest.importorskip("torch")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("pk_guard_port_precision", TEMPLATE_EVAL / "precision.py")
bk = _load("pk_guard_port_benchkit", TEMPLATE_EVAL / "benchkit.py")


# --- a task card's stress rows are measured, not dismissed --------------------

TASK_JUDGE = """import torch

def run(x, w):
    y = x.to(torch.float64)
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * w.to(torch.float64)
    return y.to(x.dtype)

def make_inputs(axes, group, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(axes["m"], axes["n"], generator=g)
    if group == "spiky":
        x[:, 0] = 100.0
    return {"x": x.to(device), "w": torch.ones(axes["n"]).to(device)}
"""


def _task_problem() -> dict:
    defn = {"name": "rms", "axes": {"m": {"type": "var"}, "n": {"type": "var"}},
            "inputs": {"x": {"shape": ["m", "n"], "dtype": "float32"},
                       "w": {"shape": ["n"], "dtype": "float32"}},
            "outputs": {"y": {"shape": ["m", "n"], "dtype": "float32"}},
            "reference": TASK_JUDGE, "custom_inputs_entrypoint": "make_inputs"}
    return {"definition": defn,
            "workload": {"uuid": "s", "axes": {"m": 4, "n": 16},
                         "input_groups": [{"name": "plain", "kind": "generated", "seed": 1},
                                          {"name": "spiky", "kind": "generated", "seed": 2}]},
            "dps": False, "_dir": "."}


def test_the_precision_stress_record_measures_the_task_distribution():
    # A task card with plain float inputs gets one measured stress row per
    # profile (WP20's gen_inputs draws — the judged/timed groups keep the task's
    # own make_inputs channel via task_inputs). The regression this pins: a
    # card that has stressable inputs must never be filed as "no plain random
    # float input to stress", which used to hollow the whole precision record.
    problem = _task_problem()

    def kernel(x, w):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * w

    prec = P.PrecisionRecorder()
    bk._precision_stress(problem, kernel, False, "cpu", prec)
    rows = [r for r in prec.rows if r.get("case") == "stress"]
    assert {r.get("stress") for r in rows} == set(P.STRESS_PROFILES)
    assert all(r["status"] in ("pass", "fail") for r in rows), prec.rows
    assert not any("no plain random float input" in str(r.get("why")) for r in prec.rows)
