"""WP9 task 4: the timing-fresh-input recheck (the agent-side cheat insurance).

The exposure it closes (verified against the eval flow first): correctness and
timing share ONE workload instance — the judge builds ``sets`` and the windows
time exactly those tensors — and nothing after the windows re-checks them. A
module-level stateful kernel can therefore compute the verdict on the judged
instance, serve the timed calls from a side buffer keyed on data pointers /
input content, and pass the whole eval: correctness replays its own answers,
timing reads like a no-op, and every existing defense (harness_guard: file
scans; prep snapshot: input mutation; host_bound_ratio: host-only paths) looks
elsewhere. The recheck re-runs the correctness protocol in a FRESH PROCESS after
all windows close — pointers move, module state resets — and a second layer
re-draws generated inputs at a different seed so content-keyed caches miss too.

Runs on CPU: the child device is FK_FRESH_RECHECK_DEVICE=cpu (the GPU job keeps
cuda); torch is required.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
EVAL_SRC = REPO / "src" / "forgekernel" / "card_template" / "eval"

# A "cache cheat", in the shape the recheck exists for: the module computes the
# correct answer ONCE (the correctness call — the first, stateless moment) and
# parks it in a side buffer; every call AFTER that artifact exists replays it by
# key — or, for any input the buffer does not cover, emits zeros. The cheater's
# bet: correctness and timing share one workload instance, so the buffer always
# covers the judged tensors in-process (correctness = the computing call, timing
# = replays), and nothing else ever asks. The recheck asks.
CHEAT_KERNEL = '''import torch
from pathlib import Path

_SIDE = Path(__file__).parent / ".side_buffer.pt"

def run(hidden_states, weight, output):
    x = hidden_states.float()
    good = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float())
    buf = torch.load(_SIDE) if _SIDE.is_file() else {}
    key = float(hidden_states[0, 0])
    if key in buf:                       # the timed calls replay the stored verdict
        output.copy_(buf[key])
        return
    if _SIDE.is_file():                  # post-correctness input the buffer misses:
        output.zero_()                   # the cheater's bet — nobody ever asks
        return
    output.copy_(good.to(hidden_states.dtype))   # the correctness call is honest
    buf[key] = good.to(hidden_states.dtype)
    torch.save(buf, _SIDE)
'''

# An honest kernel: computes every call — passes every layer by construction.
HONEST_KERNEL = '''import torch

def run(hidden_states, weight, output):
    x = hidden_states.float()
    output.copy_(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float())
'''

# The task suite's judge + its input generator (the card's own input code — what
# layer 2 re-draws with).
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

PROBLEM = {
    "schema": 1, "suite": "forgekernel",
    "definition": {"name": "rmsnorm_h64", "op_type": "rmsnorm",
                   "inputs": {"hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"},
                              "weight": {"shape": ["hidden_size"], "dtype": "float32"}},
                   "outputs": {"output": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"}},
                   "reference": JUDGE},
    "workload": {"uuid": "w0", "axes": {"batch_size": 5, "hidden_size": 64},
                 "input_groups": [{"name": "g0", "kind": "drawn", "seed": 0},
                                  {"name": "g1", "kind": "drawn", "seed": 1},
                                  {"name": "g2", "kind": "drawn", "seed": 2}]},
    "dps": True, "kernel_file": "kernel.py", "entry": "run",
    "reference_impl": {"kind": "definition"},
    "time_reference": False,
}

BENCH = '''from benchkit import run_bench

if __name__ == "__main__":
    raise SystemExit(run_bench())
'''


def _card(tmp_path: Path, kernel: str, problem: dict = PROBLEM) -> Path:
    """A minimal self-contained card: eval/benchkit.py + eval/bench.py + problem.json
    + kernel.py — what a generated card carries (no forgekernel import in the job)."""
    (tmp_path / "eval").mkdir(parents=True, exist_ok=True)
    shutil.copy2(EVAL_SRC / "benchkit.py", tmp_path / "eval" / "benchkit.py")
    shutil.copy2(EVAL_SRC / "timing.py", tmp_path / "eval" / "timing.py")
    shutil.copy2(EVAL_SRC / "precision.py", tmp_path / "eval" / "precision.py")
    for f in ("closed_ref.py", "ncu_probe.py"):
        src = EVAL_SRC / f
        if src.is_file():
            shutil.copy2(src, tmp_path / "eval" / f)
    (tmp_path / "eval" / "bench.py").write_text(BENCH, encoding="utf-8")
    (tmp_path / "eval" / "problem").mkdir(parents=True, exist_ok=True)
    (tmp_path / "eval" / "problem" / "problem.json").write_text(
        json.dumps(problem), encoding="utf-8")
    (tmp_path / "kernel.py").write_text(kernel, encoding="utf-8")
    return tmp_path


def _run_card(card: Path, *extra: str, **env_over) -> subprocess.CompletedProcess:
    import os
    env = {**os.environ, "PYTHONPATH": os_pathsep(card),
           "FK_FRESH_RECHECK_DEVICE": "cpu",
           "FK_FRESH_RECHECK": "1",   # what the production parent exports
           **env_over}
    return subprocess.run([sys.executable, str(card / "eval" / "bench.py"), *extra],
                          capture_output=True, text=True, env=env, cwd=str(card),
                          timeout=300)


def os_pathsep(card: Path) -> str:
    import os
    return os.pathsep.join([str(card), str(card / "eval")])


_VERDICT = "FRESH-RECHECK-VERDICT "


def test_child_mode_catches_the_pointer_keyed_cache_cheat(tmp_path):
    """Layer 1: the eval process first (correctness computes and parks the side
    buffer; the timed calls would replay it), THEN the same protocol in a fresh
    process — new tensors, new pointers, but the parked artifact survived, and
    the post-artifact path serves zeros. The honest kernel passes everywhere."""
    cheat = _card(tmp_path / "cheat", CHEAT_KERNEL)
    # the eval's own correctness call parks the side buffer (the same write the
    # cheat's run() performs on the judged instance — reproduced here because the
    # CPU box cannot run the parent bench's cuda path; the artifact, not the
    # process, is what carries the cheat into the recheck)
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(5, 64, generator=g)
    xf = x0.float()
    good = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)
    torch.save({float(x0[0, 0]): good}, cheat / ".side_buffer.pt")
    # the recheck child: a fresh process, freshly drawn tensors — the buffer does
    # not cover them and the post-artifact path serves zeros -> FAIL
    r = _run_card(cheat, "--mode", "correctness", "--fresh-recheck")
    verdict = json.loads([l for l in r.stdout.splitlines()
                          if l.startswith(_VERDICT)][-1][len(_VERDICT):])
    assert verdict["pass"] is False, "the parked-buffer cheat must fail the fresh process"

    # and the honest kernel, artifact never parked, re-passes everywhere
    honest = _card(tmp_path / "honest", HONEST_KERNEL)
    r = _run_card(honest, "--mode", "correctness", "--fresh-recheck")
    verdict = json.loads([l for l in r.stdout.splitlines()
                          if l.startswith(_VERDICT)][-1][len(_VERDICT):])
    assert verdict["pass"] is True, "an honest kernel re-passes everywhere"


def _park(card: Path, seeds) -> None:
    """Park the side buffer with judged instances' answers (what the cheat's
    correctness call wrote): one entry per input group's first element."""
    buf = {}
    for s in seeds:
        g = torch.Generator().manual_seed(s)
        x = torch.randn(5, 64, generator=g)
        w = torch.rand(64, generator=g) + 0.5
        xf = x.float()
        buf[float(x[0, 0])] = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    torch.save(buf, card / ".side_buffer.pt")


def test_child_regen_shifts_the_drawn_content(tmp_path):
    """Layer 2: a content-keyed cheat (replay on the judged values, zeros for
    anything else) survives a re-judge of the SAME content and fails the SHIFTED
    draw — the shift is what moves the content under the cache."""
    card_hit = _card(tmp_path / "hit", CHEAT_KERNEL)
    _park(card_hit, [s + 7919 for s in (0, 1, 2)])   # covers the shifted child's draws
    r = _run_card(card_hit, "--mode", "correctness", "--fresh-recheck")
    same = json.loads([l for l in r.stdout.splitlines()
                       if l.startswith(_VERDICT)][-1][len(_VERDICT):])
    assert same["pass"] is True, "content the buffer covers re-passes (the cheat survives it)"

    card_miss = _card(tmp_path / "miss", CHEAT_KERNEL)
    _park(card_miss, [0, 1, 2])                      # covers only the JUDGED instance
    r = _run_card(card_miss, "--mode", "correctness", "--fresh-recheck")
    shifted = json.loads([l for l in r.stdout.splitlines()
                          if l.startswith(_VERDICT)][-1][len(_VERDICT):])
    assert shifted["pass"] is False, "the regen draw must miss a judged-content cache"


def test_child_never_nests_and_prints_exactly_one_verdict(tmp_path):
    card = _card(tmp_path, HONEST_KERNEL)
    r = _run_card(card, "--mode", "correctness", "--fresh-recheck")
    assert len([l for l in r.stdout.splitlines() if l.startswith(_VERDICT)]) == 1
    assert r.stdout.count("=== JSON ===") == 0, "child mode emits no eval JSON"


def test_parse_carries_fresh_recheck_into_evalresult():
    from forgekernel.loop.parse import parse_eval
    ev = parse_eval("=== JSON ===\n"
                    + json.dumps({"correctness": {"pass": True},
                                  "performance": {"device_median_ms": 1.0},
                                  "fresh_recheck": {"mode": "regen", "pass": False,
                                                    "layers": [{"status": "INCORRECT"}]}})
                    + "\n")
    assert ev.fresh_recheck == {"mode": "regen", "pass": False,
                                "layers": [{"status": "INCORRECT"}]}
    ev2 = parse_eval("=== JSON ===\n"
                     + json.dumps({"correctness": {"pass": True},
                                   "performance": {"device_median_ms": 1.0}}) + "\n")
    assert ev2.fresh_recheck is None


def test_driver_reverts_and_marks_suspicious_on_a_fresh_recheck_fail(tmp_path):
    """The driver's consumption: a fail is a hard REVERT whose row carries
    suspicious=True — and the loop buys no further jobs on that tree."""
    from forgekernel.inner.trajectory import load_records
    from test_loop_driver import EVAL, NCU, REC, FakeBackend, _ctx
    from forgekernel.loop.driver import run_inner_loop

    fail_json = ("=== JSON ===\n"
                 + json.dumps({"correctness": {"pass": True},
                               "performance": {"device_median_ms": 1.0},
                               "fresh_recheck": {"mode": "regen", "pass": False,
                                                 "layers": [{"status": "INCORRECT_NUMERICAL"}]}})
                 + "\n")
    be = FakeBackend([REC("D1", 1.0)], [fail_json, NCU])
    ctx = _ctx(tmp_path, be)
    ctx.max_iters = 1
    run_inner_loop(ctx, start_iter=0)
    records = load_records(ctx.status_path)
    assert len(records) == 1 and records[0].verdict == "REVERT"
    assert records[0].suspicious is True
    assert "fresh_recheck FAIL" in records[0].error
