"""tests for the card's runtime integrity checks (A-D) and for the eval JSON's block.

The card is loaded the way the job loads it (a standalone module) and driven through
``run_leaderboard_bench`` -- the same entry ``eval/bench.py`` calls -- with the clocks
mocked, so the checks themselves run for real on CPU tensors.  The protocol-invariance
test is the one that pins the boundary: the same fake readings produce the same
``device_median_ms`` and the same SOL score with the checks on and off.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fk_template_leaderboard_integrity",
    REPO_ROOT / "src" / "forgekernel" / "card_template" / "eval" / "leaderboard.py")
lb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lb)

sys.path.insert(0, str(REPO_ROOT / "src"))
from forgekernel.integrity import report   # noqa: E402

RMS_REF = """import torch

def run(hidden_states, weight):
    x = hidden_states.to(torch.float32)
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)
"""
RMS_DEF = {
    "name": "rmsnorm_h64", "op_type": "rmsnorm",
    "axes": {"batch_size": {"type": "var"}, "hidden_size": {"type": "const", "value": 64}},
    "inputs": {"hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"},
               "weight": {"shape": ["hidden_size"], "dtype": "float32"}},
    "outputs": {"output": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"}},
    "reference": RMS_REF,
}


def _problem(suite: str = lb.SUITE_SOL, **wl) -> dict:
    workload = {"uuid": "w0", "axes": {"batch_size": 5},
                "inputs": {"hidden_states": {"type": "random"}, "weight": {"type": "random"}}}
    workload.update(wl)
    p = {"suite": suite, "definition": json.loads(json.dumps(RMS_DEF)),
         "workload": workload, "dps": True, "_dir": ".",
         "sol_board": {"kernel_id": 7, "workload_index": 3, "t_b_ms": 0.04, "t_sol_ms": 0.004,
                       "clock_mhz": {"sm_mhz": 1500, "mem_mhz": 3996}}}
    return p


def _group_problem(sizes=(3, 17, 64)) -> dict:
    p = _problem()
    p["group"] = []
    for i, b in enumerate(sizes):
        wl = {"uuid": f"w{i}", "axes": {"batch_size": b},
              "inputs": {"hidden_states": {"type": "random"}, "weight": {"type": "random"}}}
        p["group"].append({"index": 10 + i, "name": f"wl{10 + i}", "workload": wl,
                           "sol_board": {"kernel_id": 7, "workload_index": 10 + i,
                                         "t_b_ms": 0.04 * (i + 1), "t_sol_ms": 0.004 * (i + 1),
                                         "clock_mhz": {"sm_mhz": 1500, "mem_mhz": 3996}}})
    p["workload"] = p["group"][0]["workload"]
    p["sol_board"] = p["group"][0]["sol_board"]
    return p


def _rms(hidden_states, weight, output):
    x = hidden_states.float()
    output.copy_(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float())


CARD_KERNEL = ("import torch\n\ndef run(hidden_states, weight, output):\n"
               "    x = hidden_states.float()\n"
               "    output.copy_(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)\n"
               "                 * weight.float())\n")


def _card(tmp_path: Path, p: dict, kernel: str = CARD_KERNEL) -> Path:
    path = tmp_path / "card" / "eval" / "leaderboard" / "problem.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "card" / "kernel.py").write_text(kernel)
    p.pop("_dir", None)
    path.write_text(json.dumps(p))
    return path


def _fake_measure(kernel_ms=0.01):
    def measure(problem, sets, kernel, kernel_dps, ref=None, ref_dps=False, device="cuda", **kw):
        return {"suite": problem["suite"], "timer": "cupti", "basis": lb.BASIS_CUPTI,
                "protocol": {"warmup": 10, "iters": 50, "aggregate": "single", "seed": 200},
                "kernel_ms": kernel_ms, "kernel_per_set_ms": [kernel_ms],
                "kernel_readings_ms": [[kernel_ms, kernel_ms]],
                "ref_ms": None, "ref_per_set_ms": [], "ref_readings_ms": [],
                "ref_error": None, "unstable": {"kernel": [], "ref": []}}
    return measure


def _fake_windows(fn, dps, problem, inp, device):
    return {"median_ms": 0.012, "min_ms": 0.011, "host_enqueue_ms": 0.004}


def _fake_correctness(impl, dps, device, calls=1):
    """A correctness phase that judges the candidate on ONE draw and returns it.

    Stands in for a protocol that already validated the candidate (the fake's first call)
    so a failure later can only come from the checks: the real protocols would fail a
    candidate that is wrong on its second input at round 1, which is a different test.
    """
    def correctness(problem, impl_fn, dps_fn, device_fn):
        ref_run, _ = lb.reference_callables(problem)
        inp = lb.gen_inputs(problem, "cpu")
        refs = lb.call_impl(problem, ref_run, lb._clone(inp), False, "cpu", no_grad=True)
        outs = lb.call_impl(problem, impl_fn, inp, dps_fn, "cpu", no_grad=True)
        ok, worst, why = lb._integrity_judge(problem, outs, refs)
        verdict = {"pass": ok, "status": "PASSED" if ok else "INCORRECT_NUMERICAL",
                   "error": why, "threads_after_first_call": None}
        return verdict, [inp]
    return correctness


def _static_correctness(inp: list):
    """A correctness phase over an input set built here (no generator, no file access)."""
    def correctness(problem, impl_fn, dps_fn, device_fn):
        ref_run, _ = lb.reference_callables(problem)
        refs = lb.call_impl(problem, ref_run, lb._clone(inp), False, "cpu", no_grad=True)
        outs = lb.call_impl(problem, impl_fn, inp, dps_fn, "cpu", no_grad=True)
        ok, worst, why = lb._integrity_judge(problem, outs, refs)
        verdict = {"pass": ok, "status": "PASSED" if ok else "INCORRECT_NUMERICAL",
                   "error": why, "threads_after_first_call": None}
        return verdict, [inp]
    return correctness


def _real_correctness(problem, impl, dps, device):
    """The real SOL protocol, on CPU (the whole test file stays off the GPU)."""
    return lb.sol_correctness(problem, impl, dps, "cpu")


def _run(path: Path, monkeypatch, capsys, *, measure=None, correctness=None,
         integrity="1") -> dict:
    monkeypatch.setenv(lb.INTEGRITY_ENV_OFF, integrity)
    monkeypatch.setattr(lb, "measure", measure or _fake_measure())
    monkeypatch.setattr(lb, "_side_windows", _fake_windows)
    monkeypatch.setattr(lb, "_correctness", correctness or _real_correctness)
    assert lb.run_leaderboard_bench(["--mode", "all"], problem_path=path) == 0
    return json.loads(capsys.readouterr().out.split("=== JSON ===")[1])


# --- A: the fresh-input recheck ---------------------------------------------------------

def test_a_candidate_that_is_only_right_the_first_time_fails_the_recheck(
        tmp_path, monkeypatch, capsys) -> None:
    """Check A on its own: correctness passed, the redraw does not."""
    seen = {}

    def only_first_time(hidden_states, weight, output):
        if "first" not in seen:
            seen["first"] = hidden_states.clone()
            _rms(hidden_states, weight, output)
            return
        output.zero_()                       # a cached answer, then something cheap

    path = _card(tmp_path, _problem(),
                 "import torch\n\n_SEEN = None\n\ndef run(hidden_states, weight, output):\n"
                 "    global _SEEN\n"
                 "    if _SEEN is None:\n"
                 "        _SEEN = hidden_states.clone()\n"
                 "        x = hidden_states.float()\n"
                 "        output.copy_(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)\n"
                 "                     * weight.float())\n"
                 "    else:\n"
                 "        output.zero_()\n")
    out = _run(path, monkeypatch, capsys,
               correctness=_fake_correctness(None, None, None))
    assert out["correctness"]["pass"] is False
    assert out["correctness"]["status"] == "INTEGRITY_ERROR"
    assert out["correctness"]["error"].startswith("INTEGRITY: fresh-input recheck mismatch")
    assert "max relative error" in out["correctness"]["error"]
    assert out["integrity"]["fresh_input_recheck"]["status"] == "fail"
    # the reading itself is not touched: the fake clock's reading is what the block carries
    assert out["performance"]["device_median_ms"] == 0.01
    assert out["performance"]["leaderboard"]["sol_board"]["sol_score"] == \
        round(lb.sol_score(0.01, 0.04, 0.004), 4)
    assert report.validate(out["integrity"]) == []


def test_a_correct_candidate_passes_the_recheck_and_the_block_is_the_documented_one(
        tmp_path, monkeypatch, capsys) -> None:
    path = _card(tmp_path, _problem())
    out = _run(path, monkeypatch, capsys,
               correctness=_fake_correctness(None, None, None))
    block = out["integrity"]
    assert out["correctness"]["pass"] is True
    assert block["fresh_input_recheck"]["status"] == "pass"
    assert block["fresh_input_recheck"]["workloads"] == ["w0"]
    assert block["fresh_input_recheck"]["max_rel_err"] is not None
    assert block["global_state"]["status"] == "clean"
    # no captured reference (the fake correctness does not record one) -> skipped, not passed
    assert block["reference_stability"]["status"] == "skipped"
    assert "recorded no reference output" in block["reference_stability"]["reason"]
    assert block["timer_consistency"]["status"] == "unavailable"     # no CUDA here
    assert block["implausible_speedup"] is False
    assert report.validate(block) == []


def test_fixed_file_inputs_are_skipped_with_the_reason(tmp_path, monkeypatch, capsys) -> None:
    p = _problem()
    p["workload"]["inputs"]["hidden_states"] = {"type": "safetensors", "path": "x.safetensors",
                                                "tensor_key": "x"}
    path = _card(tmp_path, p)
    inp = [torch.randn(5, 64), torch.randn(64)]
    out = _run(path, monkeypatch, capsys, correctness=_static_correctness(inp))
    fresh = out["integrity"]["fresh_input_recheck"]
    assert fresh["status"] == "skipped" and "safetensors" in fresh["reason"]
    assert out["correctness"]["pass"] is True


def test_the_number_of_rechecked_workloads_comes_from_the_environment(
        tmp_path, monkeypatch, capsys) -> None:
    path = _card(tmp_path, _group_problem())
    monkeypatch.setenv(lb.INTEGRITY_FRESH_ENV, "1")
    one = _run(path, monkeypatch, capsys, correctness=_fake_correctness(None, None, None))
    assert len(one["integrity"]["fresh_input_recheck"]["workloads"]) == 1
    monkeypatch.setenv(lb.INTEGRITY_FRESH_ENV, "3")
    three = _run(path, monkeypatch, capsys, correctness=_fake_correctness(None, None, None))
    assert len(three["integrity"]["fresh_input_recheck"]["workloads"]) == 3
    monkeypatch.setenv(lb.INTEGRITY_FRESH_ENV, "0")
    off = _run(path, monkeypatch, capsys, correctness=_fake_correctness(None, None, None))
    fresh = off["integrity"]["fresh_input_recheck"]
    assert fresh["status"] == "skipped" and "FK_INTEGRITY_FRESH_N=0" in fresh["reason"]


def test_the_seed_environment_variable_changes_the_redraw(tmp_path, monkeypatch, capsys) -> None:
    """``FK_INTEGRITY_SEED`` is what forgekernel.integrity.multiseed drives."""
    p = _problem()
    seen = []
    real = lb._fresh_input_set

    def capture(problem, seed, device):
        inp = real(problem, seed, device)
        seen.append(float(inp[0].sum().item()))
        return inp

    monkeypatch.setattr(lb, "_fresh_input_set", capture)
    path = _card(tmp_path, p)
    monkeypatch.setenv(lb.INTEGRITY_SEED_ENV, "11")
    _run(path, monkeypatch, capsys, correctness=_fake_correctness(None, None, None))
    monkeypatch.setenv(lb.INTEGRITY_SEED_ENV, "12")
    _run(path, monkeypatch, capsys, correctness=_fake_correctness(None, None, None))
    assert len(seen) == 2 and seen[0] != seen[1]


# --- B: global state and the reference ---------------------------------------------------

def test_a_candidate_that_changes_a_global_flag_fails(tmp_path, monkeypatch, capsys) -> None:
    flag = "torch.backends.cudnn.benchmark"
    before = lb.torch.backends.cudnn.benchmark

    def flip(hidden_states, weight, output):
        _rms(hidden_states, weight, output)
        torch.backends.cudnn.benchmark = not before

    path = _card(tmp_path, _problem(),
                 "import torch\n\ndef run(hidden_states, weight, output):\n"
                 "    x = hidden_states.float()\n"
                 "    output.copy_(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)\n"
                 "                 * weight.float())\n"
                 "    torch.backends.cudnn.benchmark = not torch.backends.cudnn.benchmark\n")
    try:
        out = _run(path, monkeypatch, capsys,
                   correctness=_fake_correctness(None, None, None))
    finally:
        torch.backends.cudnn.benchmark = before
    assert out["correctness"]["pass"] is False
    assert out["correctness"]["error"].startswith(
        f"INTEGRITY: global state changed after the candidate ran")
    assert any("cudnn.benchmark" in line for line in out["integrity"]["global_state"]["diff"])
    assert out["integrity"]["global_state"]["status"] == "changed"


def test_a_thread_spawned_by_the_candidate_is_a_warning_not_a_failure(
        tmp_path, monkeypatch, capsys) -> None:
    stop = threading.Event()

    def spawn(hidden_states, weight, output):
        _rms(hidden_states, weight, output)
        if not any(t.name == "fk-integrity-test" for t in threading.enumerate()):
            threading.Thread(target=stop.wait, name="fk-integrity-test", daemon=True).start()

    path = _card(tmp_path, _problem(),
                 "import threading\nimport torch\n\n"
                 "def run(hidden_states, weight, output):\n"
                 "    x = hidden_states.float()\n"
                 "    output.copy_(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)\n"
                 "                 * weight.float())\n"
                 "    if not any(t.name == 'fk-integrity-test' for t in threading.enumerate()):\n"
                 "        threading.Thread(target=lambda: __import__('time').sleep(60),\n"
                 "                         name='fk-integrity-test', daemon=True).start()\n")
    try:
        out = _run(path, monkeypatch, capsys,
                   correctness=_fake_correctness(None, None, None))
    finally:
        stop.set()
        time.sleep(0.05)
    assert out["correctness"]["pass"] is True
    assert out["integrity"]["global_state"]["status"] == "clean"
    assert out["integrity"]["global_state"]["threads"]
    assert all(line.startswith("threads: ") for line
               in out["integrity"]["global_state"]["threads"])


def test_a_reference_that_changes_after_the_candidate_ran_fails(tmp_path, monkeypatch,
                                                                capsys) -> None:
    """Check B2: a candidate that cheapens the reference is caught by recomputing it."""
    calls = {"n": 0}

    def drifting(hidden_states, weight):
        calls["n"] += 1
        y = hidden_states.float() * weight.float()
        if calls["n"] > lb.SOL_ROUNDS:            # after the correctness rounds
            return (y + 1.0).to(hidden_states.dtype)
        return y.to(hidden_states.dtype)

    path = _card(tmp_path, _problem(),
                 "import torch\n\ndef run(hidden_states, weight, output):\n"
                 "    output.copy_(hidden_states * weight)\n")
    monkeypatch.setattr(lb, "reference_callables",
                        lambda problem: (drifting, None))
    monkeypatch.setenv(lb.INTEGRITY_FRESH_ENV, "0")     # isolate the stability check
    out = _run(path, monkeypatch, capsys)               # the REAL sol_correctness
    assert out["correctness"]["pass"] is False
    assert out["correctness"]["error"].startswith(
        "INTEGRITY: reference output changed after the candidate ran")
    assert out["integrity"]["reference_stability"]["status"] == "fail"
    assert out["integrity"]["reference_stability"]["max_rel_err"] > 0


def test_the_reference_stability_check_passes_on_a_deterministic_reference(
        tmp_path, monkeypatch, capsys) -> None:
    path = _card(tmp_path, _problem())
    monkeypatch.setenv(lb.INTEGRITY_FRESH_ENV, "0")
    out = _run(path, monkeypatch, capsys)               # the REAL sol_correctness
    assert out["correctness"]["pass"] is True
    assert out["integrity"]["reference_stability"]["status"] == "pass"
    assert out["integrity"]["reference_stability"]["max_rel_err"] == 0.0


# --- C / D -----------------------------------------------------------------------------

def test_the_timer_rule_is_the_documented_one() -> None:
    assert lb._timer_verdict(0.11, 0.10, 0.12)[0] == "ok"
    assert lb._timer_verdict(0.10, 0.01, 0.11)[0] == "suspicious"
    assert lb._timer_verdict(0.01, 0.10, 0.11)[0] == "suspicious"
    assert lb._timer_verdict(0.10, 0.10, 0.01)[0] == "suspicious"
    assert lb._timer_verdict(None, 0.1, 0.1)[0] == "unavailable"


def test_a_reading_below_the_speed_of_light_is_flagged_not_failed(
        tmp_path, monkeypatch, capsys) -> None:
    path = _card(tmp_path, _problem())                  # T_SOL 0.004 ms
    out = _run(path, monkeypatch, capsys, measure=_fake_measure(0.001),
               correctness=_fake_correctness(None, None, None))
    assert out["correctness"]["pass"] is True           # D is a warning
    assert out["integrity"]["implausible_speedup"] is True
    assert out["performance"]["leaderboard"]["sol_board"]["sol_score"] > 1.0


def test_a_normal_reading_is_not_flagged(tmp_path, monkeypatch, capsys) -> None:
    path = _card(tmp_path, _problem())
    out = _run(path, monkeypatch, capsys, measure=_fake_measure(0.03),
               correctness=_fake_correctness(None, None, None))
    assert out["integrity"]["implausible_speedup"] is False


# --- the boundary: the reading does not move -------------------------------------------

def test_the_protocol_reading_is_identical_with_the_checks_on_and_off(
        tmp_path, monkeypatch, capsys) -> None:
    """The acceptance proof: same fake readings, same gate number, same SOL score."""
    path = _card(tmp_path, _problem())
    off = _run(path, monkeypatch, capsys, measure=_fake_measure(0.013),
               correctness=_fake_correctness(None, None, None), integrity="0")
    on = _run(path, monkeypatch, capsys, measure=_fake_measure(0.013),
              correctness=_fake_correctness(None, None, None), integrity="1")
    assert "integrity" not in off and "integrity" in on
    for key in ("device_median_ms", "bwd_median_ms", "min_ms", "host_enqueue_ms"):
        assert off["performance"][key] == on["performance"][key], key
    assert off["performance"]["leaderboard"]["sol_board"] == \
        on["performance"]["leaderboard"]["sol_board"]
    assert off["performance"]["leaderboard"]["protocol"] == \
        on["performance"]["leaderboard"]["protocol"]
    assert off["correctness"]["pass"] == on["correctness"]["pass"] is True


def test_correctness_only_mode_still_snapshots_the_state_but_skips_the_redraw(
        tmp_path, monkeypatch, capsys) -> None:
    path = _card(tmp_path, _problem())
    monkeypatch.setenv(lb.INTEGRITY_ENV_OFF, "1")
    monkeypatch.setattr(lb, "_side_windows", _fake_windows)
    monkeypatch.setattr(lb, "_correctness", _fake_correctness(None, None, None))
    assert lb.run_leaderboard_bench(["--mode", "correctness"], problem_path=path) == 0
    out = json.loads(capsys.readouterr().out.split("=== JSON ===")[1])
    assert out["performance"] is None
    block = out["integrity"]
    assert block["global_state"]["status"] == "clean"
    fresh = block["fresh_input_recheck"]
    assert fresh["status"] == "skipped" and "--mode correctness" in fresh["reason"]
    assert block["timer_consistency"]["status"] == "unavailable"
    assert report.validate(block) == []
