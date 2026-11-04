"""Red-team matrix, the try/except fallback family (SOL categories: one-shot
correctness tricks; degraded precision dressed as a fast path).

Expected defense, written before the attacks: the correctness judge — a fast
path that returns wrong values fails INCORRECT_NUMERICAL on the judged groups,
and the WP20 corr_margin_ratio reading records how close to the judge's edge
the worst output sat (>=1.0 fails).

Known gap (TODO, recorded per the matrix contract): NO static layer scans the
kernel's try/except SHAPE. The L0 fence (eval/benchkit.py kernel_fence) scans
forbidden APIs, and kb_lint scans KB entries / exam surfaces — neither sees a
dead fast path wrapped in try/except (fixture 3b passes every defense while its
fast path never computes). A static scanner for the fallback shape is future
work; the WP20 package records the exposure instead of pretending a defense.

In-process on CPU (the parent bench's device is cuda by construction; the
correctness protocol itself runs anywhere — test_precision_measure's path).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from test_fresh_recheck import _card  # noqa: E402  (conftest path)

REPO = Path(__file__).resolve().parents[2]

# Attack 3a — the fallback masks a wrong fast path: try the cheap wrong answer
# first; if anything about it ever raises, the honest math follows. The judge
# never raises, so the wrong answer ships — unless correctness is judged.
FALLBACK_WRONG_FASTPATH = '''import torch

def run(hidden_states, weight, output):
    x = hidden_states.float()
    good = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float())
    try:
        output.copy_(torch.zeros_like(output))   # the "fast path": wrong
        return
    except Exception:
        output.copy_(good.to(hidden_states.dtype))
'''

# Attack 3b — the fallback masks a DEAD fast path (the known gap): the try
# always raises, the honest math always runs, every defense passes. Nothing
# today distinguishes this from an honest kernel — recorded, not defended.
FALLBACK_DEAD_FASTPATH = '''import torch

def run(hidden_states, weight, output):
    x = hidden_states.float()
    good = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float())
    try:
        raise RuntimeError   # the "fast path" never ships anything
    except Exception:
        output.copy_(good.to(hidden_states.dtype))
'''


def _bench_of(card: Path):
    spec = importlib.util.spec_from_file_location(
        f"redteam_benchkit_{card.name}", card / "eval" / "benchkit.py")
    lb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lb)
    return lb


def _run_correctness(card: Path) -> dict:
    """The production correctness protocol over the card's own kernel, in-process,
    on CPU: the exact code path a GPU job's correctness segment runs."""
    lb = _bench_of(card)
    problem = lb.load_problem(card / "eval" / "problem" / "problem.json")
    impl, dps = lb._impl(problem, "kernel")
    verdict, _sets = lb.task_correctness(problem, impl, dps, "cpu")
    return lb._correctness_json(verdict)


def test_attack3a_wrong_fastpath_fails_correctness(tmp_path):
    """Defense: correctness. The judged groups see the zeros the fast path
    serves; the verdict fails numerically, with the per-check error/tolerance
    rows the corr_margin_ratio reading is parsed from (>1 on this fail)."""
    card = _card(tmp_path, FALLBACK_WRONG_FASTPATH)
    out = _run_correctness(card)
    assert out["pass"] is False
    assert out["status"] == "INCORRECT_NUMERICAL"
    checks = out.get("checks") or []
    assert checks and checks[0]["error"] > checks[0]["tolerance"], (
        "the fail carries structured error/tolerance rows — the REVERT row's "
        "corr_margin_ratio (>1) is parsed from these")


def test_attack3b_dead_fastpath_passes_everything_known_gap(tmp_path):
    """Known gap (TODO): a dead fast path under try/except is invisible to the
    correctness judge (the honest path computes) AND to the static layers —
    recorded as the matrix's open row, never as covered."""
    card = _card(tmp_path, FALLBACK_DEAD_FASTPATH)
    out = _run_correctness(card)
    assert out["pass"] is True, (
        "the honest fallback path computes — correctness passes (this IS the gap)")
    # and the static layer stays silent on the shape, documenting the gap:
    lb = _bench_of(card)
    problem = lb.load_problem(card / "eval" / "problem" / "problem.json")
    assert lb.kernel_fence(problem) is None, (
        "the L0 fence flags APIs, not try/except shapes — the static layer has "
        "no coverage for this class (TODO, see module docstring)")
