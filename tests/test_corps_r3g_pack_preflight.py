# Copyright 2026 The Peregrine-Kernel Authors
# SPDX-License-Identifier: Apache-2.0

"""FIX-R3G regression pins for BUG-CH-J-02 / BUG-CH-J-03
(``scripts/eval-export/pack.py``'s correctness preflight).

BUG-CH-J-02: the preflight's device-transfer / reference-call / kernel-call
segment ran unguarded — one RuntimeError (an fp8 reference on a non-sm90
card, an OOM on a big workload, a kernel that refuses to launch on this
node) escaped as a traceback and exit 1, breaking the tool's
``0 = package written / 2 = refused`` contract and the stage's own SKIP
charter ("anything less is SKIP+WARN — this stage never replaces the run's
full gate"). The pins: every escape point degrades to the stage's WARN, and
the report — not a traceback — is what the caller sees.

BUG-CH-J-03: ``zip(names, outs, refs)`` truncated to the shortest sequence,
so a kernel returning fewer outputs than its contract never had the missing
comparison run and the stage recorded PASS for an output it never looked at.
The pin: a short kernel return is a FAIL, and equal-length returns still
pass.

The stage is exercised directly against a fake ``torch`` (a thin shim over
tensor-like doubles), so the pins are hermetic: no GPU, no real torch.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PACK = REPO / "scripts" / "eval-export" / "pack.py"

FAKE_IO = "_r3g_pack_preflight_fake_io"


@pytest.fixture(scope="module")
def pack():
    spec = importlib.util.spec_from_file_location("pk_corps_r3g_pack", PACK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Max:
    """The tail of the comparison's abs().max().clamp(min=...).item() chain."""

    def __init__(self, v: float):
        self._v = float(v)

    def clamp(self, min=None):
        return _Max(max(self._v, min) if min is not None else self._v)

    def item(self):
        return self._v


class _Tensor:
    """Just what the comparison loop and the smoke generator touch."""

    def __init__(self, vals, dtype: str = "float32"):
        self._v = [float(v) for v in vals]
        self.shape = (len(self._v),)
        self.dtype = dtype

    def to(self, device):
        return self

    def clone(self):
        return _Tensor(list(self._v), self.dtype)

    def float(self):
        return self

    def abs(self):
        return _Tensor([abs(v) for v in self._v], self.dtype)

    def max(self):
        return _Max(max(self._v) if self._v else 0.0)

    def __sub__(self, other):
        return _Tensor([a - b for a, b in zip(self._v, other._v)], self.dtype)


class _OOMTensor(_Tensor):
    """A tensor that cannot move to the device (the OOM trigger)."""

    def to(self, device):
        raise RuntimeError("CUDA out of memory")


def _fake_torch(tensor_cls):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        device=lambda name: name,
        float32="float32",
        Generator=lambda device="cpu": types.SimpleNamespace(
            manual_seed=lambda seed: None),
        no_grad=contextlib.nullcontext,
        randn=lambda *shape, dtype=None, generator=None: tensor_cls(
            [0.25 * (i + 1) for i in range(shape[0] if shape else 1)]),
        randint=lambda low, high, shape, dtype=None, generator=None: tensor_cls(
            [low] * (shape[0] if shape else 1)),
    )


def _run_stage(pack, monkeypatch, tmp_path: Path, *, kernel_src: str,
               reference_src: str, problem: dict | None = None,
               tensor_cls=_Tensor, entry: str = "run", dps: bool = False):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(tensor_cls))
    monkeypatch.setitem(sys.modules, FAKE_IO, types.SimpleNamespace(
        out=lambda *vals: tensor_cls(list(vals))))
    files: list[tuple[str, Path]] = []
    if problem is not None:
        ppath = tmp_path / "problem.json"
        ppath.write_text(json.dumps(problem), encoding="utf-8")
        files = [("problem.json", ppath)]
    work = tmp_path / "work"
    work.mkdir()
    (work / "kernel.py").write_text(kernel_src, encoding="utf-8")
    contract = pack.Contract(definition="d", entry=entry, inputs=[], outputs=[],
                             dps=None, kernel_file="kernel.py",
                             reference_src=reference_src, tolerance={},
                             contract_files=files)
    report = pack.Report()
    pack.preflight_correctness(b"", "kernel.py", contract, dps, work, report)
    return report


def _one_input_problem(outputs: dict | None = None) -> dict:
    return {"definition": {"inputs": {"x": {"shape": ["4"], "dtype": "float32"}},
                           **({"outputs": outputs} if outputs else {})},
            "workload": {}}


def test_reference_runtimeerror_degrades_to_warn(pack, monkeypatch, tmp_path):
    """A reference that only runs on other hardware (the fp8-on-non-sm90
    trigger) must land in the report as a WARN, not escape."""
    ref = "def run():\n    raise RuntimeError('fp8 reference needs sm90')\n"
    kern = "def run():\n    return None\n"
    report = _run_stage(pack, monkeypatch, tmp_path,
                        kernel_src=kern, reference_src=ref)
    assert len(report.stages) == 1
    stage = report.stages[-1]
    assert stage["stage"] == "correctness"
    assert stage["status"] == "WARN"
    assert "could not run the reference or the kernel here" in stage["detail"]
    assert not report.failed


def test_device_transfer_runtimeerror_degrades_to_warn(pack, monkeypatch, tmp_path):
    """A workload too big for this card's memory (the .to(device) OOM trigger)
    must land in the report as a WARN, not escape."""
    ref = (f"import {FAKE_IO}\n"
           "def run(x):\n"
           f"    return {FAKE_IO}.out(1.0, 2.0, 3.0, 4.0)\n")
    kern = "def run(x):\n    return x\n"
    report = _run_stage(pack, monkeypatch, tmp_path,
                        kernel_src=kern, reference_src=ref,
                        problem=_one_input_problem(), tensor_cls=_OOMTensor)
    assert len(report.stages) == 1
    stage = report.stages[-1]
    assert stage["stage"] == "correctness"
    assert stage["status"] == "WARN"
    assert "could not run the reference or the kernel here" in stage["detail"]
    assert not report.failed


def test_kernel_runtimeerror_degrades_to_warn(pack, monkeypatch, tmp_path):
    """A kernel that will not run on this node must land in the report as a
    WARN, not escape."""
    ref = (f"import {FAKE_IO}\n"
           "def run(x):\n"
           f"    return {FAKE_IO}.out(1.0, 2.0, 3.0, 4.0)\n")
    kern = "def run(x):\n    raise RuntimeError('launch failed on this node')\n"
    report = _run_stage(pack, monkeypatch, tmp_path,
                        kernel_src=kern, reference_src=ref,
                        problem=_one_input_problem())
    assert len(report.stages) == 1
    stage = report.stages[-1]
    assert stage["stage"] == "correctness"
    assert stage["status"] == "WARN"
    assert "could not run the reference or the kernel here" in stage["detail"]
    assert not report.failed


def test_short_kernel_output_is_a_fail_not_a_pass(pack, monkeypatch, tmp_path):
    """A kernel that returns one output where the contract/reference has two
    must FAIL — the pre-fix zip truncated silently and the never-compared
    second output was recorded as PASS."""
    ref = (f"import {FAKE_IO}\n"
           "def run(x):\n"
           f"    return {FAKE_IO}.out(1.0, 2.0), {FAKE_IO}.out(3.0, 4.0)\n")
    kern = (f"import {FAKE_IO}\n"
            "def run(x):\n"
            f"    return {FAKE_IO}.out(1.0, 2.0)\n")
    report = _run_stage(pack, monkeypatch, tmp_path,
                        kernel_src=kern, reference_src=ref,
                        problem=_one_input_problem(outputs={"o0": {}, "o1": {}}))
    stage = report.stages[-1]
    assert stage["stage"] == "correctness"
    assert stage["status"] == "FAIL"
    assert "1 output(s)" in stage["detail"] and "2" in stage["detail"]
    assert report.failed


def test_matching_outputs_still_pass(pack, monkeypatch, tmp_path):
    """The length guard must not block an honest kernel: equal-length outputs
    within tolerance still record PASS."""
    ref = (f"import {FAKE_IO}\n"
           "def run(x):\n"
           f"    return {FAKE_IO}.out(1.0, 2.0), {FAKE_IO}.out(3.0, 4.0)\n")
    kern = (f"import {FAKE_IO}\n"
            "def run(x):\n"
            f"    return {FAKE_IO}.out(1.0, 2.0), {FAKE_IO}.out(3.0, 4.0)\n")
    report = _run_stage(pack, monkeypatch, tmp_path,
                        kernel_src=kern, reference_src=ref,
                        problem=_one_input_problem(outputs={"o0": {}, "o1": {}}))
    stage = report.stages[-1]
    assert stage["stage"] == "correctness"
    assert stage["status"] == "PASS"
    assert "1 contract workload(s)" in stage["detail"]
    assert not report.failed
