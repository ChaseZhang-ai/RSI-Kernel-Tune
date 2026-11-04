"""Regression tests for the audit-corps fix wave over the benchkit/eval surface.

Covered here:
- BUG-C5-02: a Python kernel that dies at import time lands a structured
  IMPORT_ERROR verdict in the eval JSON (it used to re-raise bare: the job
  burned with no ``=== JSON ===`` block and the parse could only report "no
  eval JSON block").
- BUG-C5-03: a workload with no input_groups is an explicit INPUT_ERROR fail
  (it used to judge zero checks and PASS), and ncu_window refuses empty pairs
  with a clear message instead of a ZeroDivisionError.
- BUG-T7C-01: the WP9 fresh-recheck child prints its ``pass=null`` verdict line
  even when the kernel fails to import (the WP29 replay-probe child's posture).
- BUG-SEED-24: the precision measure flags disagreeing output/ref/truth lengths
  as an unavailable row instead of silently truncating.

CPU only where possible; needs torch (skipped without it). The card helpers
come from test_fresh_recheck (the same minimal self-contained card the recheck
tests run).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from test_fresh_recheck import PROBLEM, _VERDICT, _card, _run_card

REPO = Path(__file__).resolve().parents[1]
EVAL_SRC = REPO / "src" / "peregrine_kernel" / "card_template" / "eval"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lb = _load("pk_benchkit_corps_fix", EVAL_SRC / "benchkit.py")
P = _load("pk_precision_corps_fix", EVAL_SRC / "precision.py")


def _json_payload(stdout: str) -> dict:
    """The eval JSON the bench emitted (the block after the last marker)."""
    return json.loads(stdout.split("=== JSON ===")[-1].strip())


# --- BUG-C5-02: a Python kernel's import-time failure is a structured verdict -------


def test_python_kernel_import_failure_lands_a_structured_verdict(tmp_path):
    """A syntax-error kernel.py used to re-raise out of run_bench: exit 1, zero
    JSON blocks, and the job log's only clue was a traceback (the parse reads it
    as "no eval JSON block"). Now the same capture that wraps the .cu build
    failure wraps the Python import: a FAIL verdict named IMPORT_ERROR, the
    original error text capped like COMPILE_ERROR's."""
    broken = "import torch\n\ndef run(:   # a syntax error, dead at import\n"
    card = _card(tmp_path / "broken", broken)
    r = _run_card(card, "--mode", "correctness")
    assert r.returncode == 0, r.stdout + r.stderr
    payload = _json_payload(r.stdout)
    assert payload["correctness"]["pass"] is False
    assert payload["correctness"]["status"] == "IMPORT_ERROR"
    assert "SyntaxError" in payload["correctness"]["error"]
    assert payload["performance"] is None


def test_cuda_kernel_build_failure_keeps_its_compile_error_name(tmp_path):
    """The capture is unified but the .cu build failure keeps the COMPILE_ERROR
    verdict the loop's parse reads its compiler context from."""
    broken = "this is not cuda c++\n"
    card = _card(tmp_path / "brokencu", broken, problem=dict(PROBLEM, kernel_file="kernel.cu"))
    r = _run_card(card, "--mode", "correctness")
    assert r.returncode == 0, r.stdout + r.stderr
    payload = _json_payload(r.stdout)
    assert payload["correctness"]["pass"] is False
    assert payload["correctness"]["status"] == "COMPILE_ERROR"
    assert payload["correctness"]["error"]


# --- BUG-C5-03: an empty input_groups workload fails, it does not fake a PASS --------

# The task suite's judge + its input generator (what a task card's definition
# reference carries; task_correctness execs it before anything else).
JUDGE = '''import torch

def run(hidden_states, weight):
    x = hidden_states.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float())

def make_inputs(axes_and_scalars, group=None, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    n = int(axes_and_scalars["batch_size"])
    dev = torch.device(device)
    return {"hidden_states": torch.randn(n, 64, generator=g, device=dev),
            "weight": (torch.rand(64, generator=g, device=dev) + 0.5)}
'''


def _task_problem(**workload) -> dict:
    wl = {"uuid": "w0", "axes": {"batch_size": 5, "hidden_size": 64}}
    wl.update(workload)
    return {"definition": {"name": "rmsnorm_h64", "op_type": "rmsnorm",
                           "inputs": {"hidden_states": {"shape": ["batch_size", "hidden_size"],
                                                        "dtype": "float32"},
                                      "weight": {"shape": ["hidden_size"], "dtype": "float32"}},
                           "outputs": {"output": {"shape": ["batch_size", "hidden_size"],
                                                  "dtype": "float32"}},
                           "reference": JUDGE},
            "workload": wl, "dps": True, "kernel_file": "kernel.py",
            "reference_impl": {"kind": "definition"}, "time_reference": False, "_dir": "."}


def _honest_kernel(hidden_states, weight, output):
    x = hidden_states.float()
    output.copy_(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float())


@pytest.mark.parametrize("wl", [{}, {"input_groups": []}],
                             ids=["no-key", "empty-list"])
def test_task_correctness_fails_a_workload_with_no_input_groups(wl):
    """Zero input groups judged zero checks and fell through to a PASS — a fake
    pass no check backed (and the timing / NCU stages after it crashed bare).
    It is an INPUT_ERROR fail, the same verdict every other malformed input of
    this function gets."""
    verdict, sets = lb.task_correctness(_task_problem(**wl), _honest_kernel, True, "cpu")
    assert verdict["pass"] is False
    assert verdict["status"] == "INPUT_ERROR"
    assert "input_groups" in verdict["error"]
    assert verdict["checks"] == [] and sets == []


def test_task_correctness_still_judges_a_populated_workload():
    """The guard rides before the loop; a workload with groups judges exactly as
    before (the honest kernel passes its judge)."""
    problem = _task_problem(input_groups=[{"name": "g0", "kind": "drawn", "seed": 0},
                                          {"name": "g1", "kind": "drawn", "seed": 1}])
    verdict, sets = lb.task_correctness(problem, _honest_kernel, True, "cpu")
    assert verdict["pass"] is True and verdict["status"] == "PASSED"
    assert len(sets) == 2
    assert {c["group"] for c in verdict["checks"]} == {"g0", "g1"}


def test_ncu_window_says_why_when_the_card_has_no_input_groups(tmp_path, monkeypatch):
    """Empty pairs used to reach `calls // len(pairs)` — a ZeroDivisionError with
    no hint. The window now refuses with a message naming the empty card."""
    pdir = tmp_path / "eval" / "problem"
    pdir.mkdir(parents=True)
    pjson = pdir / "problem.json"
    pjson.write_text(json.dumps(_task_problem()), encoding="utf-8")
    monkeypatch.setattr(lb, "_impl", lambda problem, which: (lambda *a: None, False))
    with pytest.raises(SystemExit, match="no input groups to profile"):
        lb.ncu_window(problem_path=str(pjson))


# --- BUG-T7C-01: the fresh-recheck child speaks even when the kernel cannot load -----


def test_fresh_recheck_child_prints_a_verdict_when_the_kernel_cannot_import(tmp_path):
    """The child re-imports the kernel in a fresh process; a module that dies at
    import (an allocation, a sys.exit, parent state it cannot see) used to kill
    the child bare — no verdict line, and the parent could only guess "the child
    printed no verdict". The load now sits inside the child's try: one
    pass=null verdict line, the WP29 replay-probe child's posture."""
    broken = ("import torch\n"
              "raise RuntimeError('boom at module import — the child cannot load me')\n"
              "\n"
              "def run(hidden_states, weight, output):\n"
              "    output.copy_(hidden_states)\n")
    card = _card(tmp_path / "dies", broken)
    r = _run_card(card, "--mode", "correctness", "--fresh-recheck")
    assert r.returncode == 0, r.stdout + r.stderr
    lines = [l for l in r.stdout.splitlines() if l.startswith(_VERDICT)]
    assert len(lines) == 1, r.stdout
    verdict = json.loads(lines[0][len(_VERDICT):])
    assert verdict["pass"] is None
    assert "RuntimeError" in verdict["detail"] and "boom at module import" in verdict["detail"]
    assert "Traceback" not in r.stdout


def test_fresh_recheck_child_still_rechecks_an_honest_kernel(tmp_path):
    """The moved call keeps the honest path byte-identical: the child loads the
    kernel, runs the protocol, and reports pass=True."""
    honest = ("import torch\n"
              "\n"
              "def run(hidden_states, weight, output):\n"
              "    x = hidden_states.float()\n"
              "    output.copy_(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)\n"
              "                * weight.float())\n")
    card = _card(tmp_path / "honest", honest)
    r = _run_card(card, "--mode", "correctness", "--fresh-recheck")
    assert r.returncode == 0, r.stdout + r.stderr
    lines = [l for l in r.stdout.splitlines() if l.startswith(_VERDICT)]
    assert len(lines) == 1, r.stdout
    assert json.loads(lines[0][len(_VERDICT):])["pass"] is True


# --- BUG-SEED-24: disagreeing measure lengths are a visible row, not a truncation ----


def test_precision_measure_flags_disagreeing_lengths_instead_of_truncating():
    """A hand-written bench whose outputs / refs / truths disagree in length used
    to zip silently short: the extra outputs got no row and no error — a false
    pass by omission. strict zip makes the mismatch a ValueError, and measure's
    own except turns it into one visible unavailable row."""
    y = torch.ones(4, dtype=torch.float32)
    rec = P.PrecisionRecorder()
    rec.measure("case", [y, y.clone(), y.clone()], [y.clone()], [y.clone()])
    assert len(rec.rows) == 2            # the one aligned row + the mismatch row
    last = rec.rows[-1]
    assert last["status"] == "unavailable"
    assert "ValueError" in last["why"] and "zip" in last["why"]


def test_precision_measure_still_rows_equal_lengths():
    """Aligned lists measure exactly as before (strict zip changes nothing): the
    candidate sits inside the reference's own error against the truth."""
    t = torch.ones(4, dtype=torch.float64)
    r = (t * 1.02).to(torch.float32)
    y = (t * 1.021).to(torch.float32)
    rec = P.PrecisionRecorder()
    rec.measure("case", [y], [r], [t])
    assert len(rec.rows) == 1
    row = rec.rows[0]
    assert row["status"] == "pass"       # a real measurement, not an unavailable row
    assert row["e_cand"] is not None and row["limit"] is not None
