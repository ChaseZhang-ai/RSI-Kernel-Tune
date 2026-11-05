# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""WP22: the declared compute precision, wired end to end.

``[precision] compute_dtype`` in task.toml is the user's precision declaration:
the kernel computes at exactly that dtype, the student's prompt carries the
one-sentence contract, the precision gate evaluates the reference at the
declaration (its measure unchanged) and convicts arithmetic BELOW it, the
launcher passes the declaration to the loop like every other task key, and the
digest stamps it. The default (no table) is fp32 and every surface it touches
is byte-identical to pre-WP22 -- including the golden records. Runs on CPU.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
EVAL_SRC = REPO / "src" / "rsi_kernel_tune" / "card_template" / "eval"

from rsi_kernel_tune.inner.trajectory import (  # noqa: E402
    IterationRecord, build_digest)
from rsi_kernel_tune.loop import driver as driver_mod  # noqa: E402
from rsi_kernel_tune.loop import precision_gate  # noqa: E402
from rsi_kernel_tune.task import COMPUTE_DTYPES, load_task  # noqa: E402

from test_task import MINIMAL, _kb, _task  # noqa: E402


# -- task.py: the declaration ---------------------------------------------------------

def test_task_precision_defaults_to_fp32_undeclared(tmp_path):
    t = load_task(_task(tmp_path, MINIMAL), kb_root=_kb(tmp_path))
    assert t.precision.compute_dtype == "fp32"
    assert t.precision.declared is False


def test_task_precision_declared_loads(tmp_path):
    t = load_task(_task(tmp_path, MINIMAL + '[precision]\ncompute_dtype = "fp16"\n'),
                  kb_root=_kb(tmp_path))
    assert t.precision.compute_dtype == "fp16"
    assert t.precision.declared is True


def test_task_precision_explicit_fp32_is_declared(tmp_path):
    t = load_task(_task(tmp_path, MINIMAL + '[precision]\ncompute_dtype = "fp32"\n'),
                  kb_root=_kb(tmp_path))
    assert t.precision.compute_dtype == "fp32" and t.precision.declared is True


def test_task_precision_illegal_dtype_is_a_problem(tmp_path):
    with pytest.raises(Exception, match="fp32, tf32, bf16, fp16, fp8"):
        load_task(_task(tmp_path, MINIMAL + '[precision]\ncompute_dtype = "fp64"\n'),
                  kb_root=_kb(tmp_path))


def test_every_declared_dtype_loads(tmp_path):
    for i, dt in enumerate(COMPUTE_DTYPES):
        t = load_task(_task(tmp_path, MINIMAL + f'[precision]\ncompute_dtype = "{dt}"\n',
                            name=f"op{i}"),
                      kb_root=_kb(tmp_path))
        assert t.precision.compute_dtype == dt


# -- loop cli: the declaration is validated, never silently defaulted -----------------


def test_loop_cli_computes_dtype_validates():
    from rsi_kernel_tune.loop import cli as loop_cli
    assert loop_cli._compute_dtype("fp16") == "fp16"
    with pytest.raises(SystemExit, match="compute-dtype must be one of"):
        loop_cli._compute_dtype("fp64")


# -- driver: the contract sentence ----------------------------------------------------


def _ctx(**kw):
    base = dict(memo_lane=False, compute_dtype="fp32")
    base.update(kw)
    return SimpleNamespace(**base)


def test_student_prompt_contract_sentence_appears_only_when_declared():
    assert driver_mod._compute_dtype_note(_ctx()) == ""
    note = driver_mod._compute_dtype_note(_ctx(compute_dtype="fp16"))
    assert ("This task's contract declares compute_dtype=fp16. Work within it; "
            "going below it is a contract violation (REVERT).") in note
    assert note.count("\n") <= 4          # one header + one line: the injection stays small


def test_every_declared_dtype_renders_the_contract():
    for dt in ("tf32", "bf16", "fp16", "fp8"):
        note = driver_mod._compute_dtype_note(_ctx(compute_dtype=dt))
        assert f"declares compute_dtype={dt}" in note and "REVERT" in note


# -- the gate: reference at the declaration, conviction below it ----------------------


_FAIL_SUMMARY = {"status": "fail", "k": 3, "n_checks": 4, "n_failed": 1,
                 "worst": {"case": "g0", "output": "y", "ratio": 12.5, "e_cand": "0.02",
                           "e_ref": "0.001", "limit": "0.004"}}


def test_gate_failure_default_text_is_byte_identical():
    text = precision_gate.gate_failure("strict", _FAIL_SUMMARY)
    assert text == precision_gate.gate_failure("strict", _FAIL_SUMMARY, compute_dtype="fp32")
    assert "keep the reference's precision" in text          # the historical line
    assert "compute_dtype=fp32" not in text


def test_gate_failure_declared_names_the_contract():
    text = precision_gate.gate_failure("strict", _FAIL_SUMMARY, compute_dtype="fp16")
    assert "compute_dtype=fp16" in text and "contract violation (REVERT)" in text
    assert "a reference evaluated at fp16" in text


def test_gate_failure_pass_and_non_strict_stay_silent():
    assert precision_gate.gate_failure("strict", {"status": "pass"}, compute_dtype="fp8") is None
    assert precision_gate.gate_failure("official", {"status": "fail", "worst": {}},
                                       compute_dtype="fp8") is None


# -- the digest: one declaration per comparison column --------------------------------


_FIELDS = {f.name for f in dataclasses.fields(IterationRecord)}


def _rec(**kw) -> IterationRecord:
    base = dict(iter=0, decision_id="D1", change="widen tiles", correctness=True,
                verdict="KEEP", base_sha="a", sha="b",
                bwd_median_ms=1.0, device_median_ms=1.0, memo_lane=False)
    base.update(kw)
    return IterationRecord(**{k: v for k, v in base.items() if k in _FIELDS})


def test_digest_stamps_a_non_default_declaration():
    d = build_digest([_rec()], {"value": 1.0}, compute_dtype="fp16")
    assert d["compute_dtype"] == "fp16"


def test_digest_default_omits_the_stamp_pre_wp22_byte_identical():
    assert "compute_dtype" not in build_digest([_rec()], {"value": 1.0})
    assert "compute_dtype" not in build_digest([_rec()], {"value": 1.0}, compute_dtype="fp32")


def test_digest_accepts_rows_under_a_declared_run():
    # the rows carry no dtype (the declaration is run-level, like the lane):
    # a declared run's digest builds and stamps
    d = build_digest([_rec()], {"value": 1.0}, compute_dtype="fp8")
    assert d["compute_dtype"] == "fp8"


# -- the carry: a price is a prior under one declaration only -------------------------


def test_carry_dtype_stamp_and_mismatch(tmp_path):
    """A carry whose levers.json names another compute_dtype is dropped, like a
    lane mismatch (the memo pattern); no stamp reads fp32 (pre-WP22 carries)."""
    d = tmp_path / "carry"
    d.mkdir()
    (d / "levers.json").write_text(json.dumps({"memo_lane": False}))
    ctx = _ctx(carry_dir=str(d), memo_lane=False)
    assert driver_mod._carry_dtype(ctx, d) == "fp32"
    assert driver_mod._carry_binding(ctx) != "(none)"      # same declaration: kept
    (d / "levers.json").write_text(json.dumps({"memo_lane": False,
                                               "compute_dtype": "fp16"}))
    assert driver_mod._carry_dtype(ctx, d) == "fp16"
    assert driver_mod._carry_binding(ctx) == "(none)"      # dropped outright


# -- taskcard: the declaration rides the card's jobs ----------------------------------

def test_submit_toml_carries_the_declaration(tmp_path):
    from rsi_kernel_tune import taskcard
    from rsi_kernel_tune.localenv import Machine
    t = load_task(_task(tmp_path, MINIMAL + '[precision]\ncompute_dtype = "bf16"\n'),
                  kb_root=_kb(tmp_path))
    (tmp_path / "tree").mkdir()
    p = taskcard.write_submit(tmp_path / "tree", t, Machine(delivery="local", gpu="0"))
    assert 'RKT_COMPUTE_DTYPE = "bf16"' in p.read_text()
    # and the default stays byte-identical: no line at all
    t32 = load_task(_task(tmp_path, MINIMAL, name="rmsnorm32"), kb_root=_kb(tmp_path))
    (tmp_path / "tree32").mkdir()
    p32 = taskcard.write_submit(tmp_path / "tree32", t32, Machine(delivery="local", gpu="0"))
    assert "RKT_COMPUTE_DTYPE" not in p32.read_text()


# -- benchkit: the reference evaluated at the declaration -----------------------------


def _benchkit(env_dtype: str = ""):
    """The template benchkit loaded fresh with RKT_COMPUTE_DTYPE set as asked (the
    card copies it verbatim; the env is read once at import)."""
    old = os.environ.pop("RKT_COMPUTE_DTYPE", None)
    if env_dtype:
        os.environ["RKT_COMPUTE_DTYPE"] = env_dtype
    try:
        spec = importlib.util.spec_from_file_location(
            f"wp22_benchkit_{env_dtype or 'fp32'}_{id(object())}", EVAL_SRC / "benchkit.py")
        bk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bk)
        return bk
    finally:
        os.environ.pop("RKT_COMPUTE_DTYPE", None)
        if old is not None:
            os.environ["RKT_COMPUTE_DTYPE"] = old


def test_benchkit_default_reads_fp32_and_adds_no_behavior():
    bk = _benchkit()
    assert bk.COMPUTE_DTYPE == "fp32"
    refs = [torch.tensor([1.0])]
    assert bk._refs_at_declared({}, lambda x: x, [], refs, "cpu") == refs


def test_benchkit_refuses_a_garbage_declaration():
    with pytest.raises(ValueError, match="declared compute precision"):
        _benchkit("fp64")


PROBLEM = {"definition": {"outputs": {"y": {"dtype": "float32"}}}, "dps": False}


def _ref_fn(x, w):
    return x @ w


def test_benchkit_evaluates_the_reference_at_fp16_on_cpu():
    bk = _benchkit("fp16")
    torch.manual_seed(0)
    inp = [torch.randn(32, 32), torch.randn(32, 32)]
    refs_native = bk.normalize_outputs(PROBLEM, _ref_fn(*inp), "cpu")
    refs = bk._refs_at_declared(PROBLEM, _ref_fn, inp, refs_native, "cpu")
    assert all(r.dtype == torch.float16 for r in refs)
    # the fp16 evaluation is a real computation at fp16, not just an output cast:
    # its error against the fp64 truth exceeds the pure output-rounding error
    truth = inp[0].double() @ inp[1].double()
    e_fp16 = (refs[0].double() - truth).norm() / truth.norm()
    e_cast = (refs_native[0].to(torch.float16).double() - truth).norm() / truth.norm()
    assert e_fp16 >= e_cast


def test_benchkit_tf32_keeps_fp32_storage():
    bk = _benchkit("tf32")
    torch.manual_seed(0)
    inp = [torch.randn(16, 16), torch.randn(16, 16)]
    refs_native = bk.normalize_outputs(PROBLEM, _ref_fn(*inp), "cpu")
    refs = bk._refs_at_declared(PROBLEM, _ref_fn, inp, refs_native, "cpu")
    assert all(r.dtype == torch.float32 for r in refs)


def test_benchkit_fp8_quantizes_the_reference_outputs():
    bk = _benchkit("fp8")
    torch.manual_seed(0)
    inp = [torch.randn(8, 8), torch.randn(8, 8)]
    refs_native = bk.normalize_outputs(PROBLEM, _ref_fn(*inp), "cpu")
    refs = bk._refs_at_declared(PROBLEM, _ref_fn, inp, refs_native, "cpu")
    assert all(r.dtype == torch.float8_e4m3fn for r in refs)


def test_benchkit_a_failing_declaration_side_run_falls_back_to_the_grid():
    bk = _benchkit("fp16")

    def broken(x, w):
        raise RuntimeError("no fp16 kernel for this op")

    torch.manual_seed(0)
    inp = [torch.randn(8, 8), torch.randn(8, 8)]
    refs_native = bk.normalize_outputs(PROBLEM, _ref_fn(*inp), "cpu")
    refs = bk._refs_at_declared(PROBLEM, broken, inp, refs_native, "cpu")
    # the fallback quantizes the native outputs to the declared grid (the row's
    # dtype then says what was measured)
    assert all(r.dtype == torch.float16 for r in refs)
