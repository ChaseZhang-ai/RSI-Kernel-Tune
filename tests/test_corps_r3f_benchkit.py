# Copyright 2026 The Peregrine-Kernel Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the audit-corps R3F fix wave over the benchkit/cardkit surface.

Covered here:
- BUG-CH-G-01: the kernel fence reads the kernel file with a fixed utf-8
  replace decode, so one non-UTF-8 byte (a latin-1 comment, a high byte in a
  .cu) cannot crash the bench before any verdict exists -- it used to raise
  UnicodeDecodeError straight out of run_bench / ncu_window: zero JSON, a
  burned GPU job the parse can only read as "no eval JSON block".
- BUG-CH-G-03: the WP29 replay-probe child applies the same workload scope as
  the parent bench, so on a scoped meta job the child's shape names match the
  parent's (the probe used to read unavailable on every scoped job, after
  burning the probe protocol on the whole card).
- BUG-CH-G-02: the gpu_entry LB_REFUSE list covers the WP11 merged allncu
  task, so a card that asks for a clock lock refuses it too when the lock did
  not hold (it used to run the full timing protocol unlocked).

Runs on CPU; no GPU is needed for any assertion here.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EVAL_SRC = REPO / "src" / "peregrine_kernel" / "card_template" / "eval"
CARDKIT = REPO / "src" / "peregrine_kernel" / "cardkit.py"


def _benchkit():
    """The template benchkit, loaded fresh (the card copies it verbatim; the
    tests drive the same code the production bench child runs)."""
    spec = importlib.util.spec_from_file_location("corps_r3f_benchkit", EVAL_SRC / "benchkit.py")
    bk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bk)
    return bk


def _fence_card(tmp_path: Path, kfile: str, body: bytes, module: str) -> dict:
    """A card whose kernel file carries a non-UTF-8 byte, with a sealed reference
    (the only kind the fence polices) naming a module the kernel imports."""
    (tmp_path / "eval" / "problem").mkdir(parents=True, exist_ok=True)
    problem = {
        "schema": 1, "suite": "peregrine_kernel",
        "definition": {"name": "rmsnorm_h64", "op_type": "rmsnorm",
                       "inputs": {}, "outputs": {}, "reference": "def run(**kw):\n    pass\n"},
        "workload": {"uuid": "w0", "axes": {}, "input_groups": []},
        "dps": True, "kernel_file": kfile, "entry": "run",
        "reference_impl": {"kind": "sealed", "module": module, "path": "reference/sealed.bin"},
        "_dir": str(tmp_path / "eval" / "problem"),
    }
    (tmp_path / kfile).write_bytes(body)
    return problem


# --- BUG-CH-G-01 -------------------------------------------------------------------

def test_kernel_fence_survives_a_non_utf8_python_kernel(tmp_path):
    """A latin-1 high byte in kernel.py cannot kill the fence (it used to raise
    UnicodeDecodeError through run_bench before any verdict existed)."""
    bk = _benchkit()
    problem = _fence_card(tmp_path, "kernel.py",
                          b"# -*- coding: latin-1 -*-\nimport r\xe9flib\nimport torch\n\n"
                          b"def run(x):\n    return x\n",
                          "reflib")
    why = bk.kernel_fence(problem)          # used to raise UnicodeDecodeError
    # the mangled byte reads as a replacement char (r\ufffdflib matches nothing):
    # a verdict -- here None -- instead of a crash; policing itself is pinned below
    assert why is None or isinstance(why, str)


def test_kernel_fence_survives_a_non_utf8_cuda_kernel(tmp_path):
    """The same byte in kernel.cu: the .cu fence path reads the file too."""
    bk = _benchkit()
    problem = _fence_card(tmp_path, "kernel.cu",
                          b"// a comment with a latin-1 na\xefeve byte\n"
                          b"#include <torch/extension.h>\n",
                          "reflib")
    why = bk.kernel_fence(problem)          # used to raise UnicodeDecodeError
    assert why is None or isinstance(why, str)


def test_bench_kernels_the_fence_reports_for_a_readable_kernel(tmp_path):
    """Parity guard: on plain utf-8 content the fence still reads and reports."""
    bk = _benchkit()
    problem = _fence_card(tmp_path, "kernel.py",
                          b"import reflib\n\ndef run(x):\n    return x\n", "reflib")
    why = bk.kernel_fence(problem)
    assert isinstance(why, str) and "reflib" in why
    (tmp_path / "kernel.py").write_bytes(b"import torch\n\ndef run(x):\n    return x\n")
    assert bk.kernel_fence(problem) is None


# --- BUG-CH-G-03 -------------------------------------------------------------------

_GROUP_PROBLEM = {
    "schema": 1, "suite": "peregrine_kernel",
    "definition": {"name": "rmsnorm_h64", "op_type": "rmsnorm", "inputs": {},
                   "outputs": {}, "reference": "def run(**kw):\n    pass\n"},
    "group": [
        {"name": "wl1", "index": 1,
         "workload": {"uuid": "uuid-AAA", "axes": {"batch_size": 5, "hidden_size": 64},
                      "input_groups": [{"name": "g0", "kind": "drawn", "seed": 0}]}},
        {"name": "wl5", "index": 5,
         "workload": {"uuid": "uuid-BBB", "axes": {"batch_size": 9, "hidden_size": 64},
                      "input_groups": [{"name": "g0", "kind": "drawn", "seed": 0}]}},
    ],
    "dps": True, "kernel_file": "kernel.py", "entry": "run",
    "reference_impl": {"kind": "definition"}, "time_reference": False,
}


def _group_card(tmp_path: Path) -> Path:
    (tmp_path / "eval" / "problem").mkdir(parents=True, exist_ok=True)
    (tmp_path / "eval" / "problem" / "problem.json").write_text(
        json.dumps(_GROUP_PROBLEM), encoding="utf-8")
    return tmp_path


def test_replay_probe_child_sees_the_workload_scope(tmp_path, monkeypatch):
    """The probe child applies the parent's workload scope: its shape list is the
    scoped single view named like the parent's timing block (the workload uuid),
    not the whole group under group-entry names nothing can match."""
    bk = _benchkit()
    card = _group_card(tmp_path)
    monkeypatch.setenv(bk.WORKLOAD_SCOPE_ENV, "wl1")
    seen: list[dict] = []
    monkeypatch.setattr(bk, "_run_replay_probe_child", lambda problem: seen.append(problem) or 0)
    rc = bk.run_bench(["--mode", "correctness", "--replay-probe",
                       str(card / "eval" / "problem" / "problem.json")])
    assert rc == 0 and len(seen) == 1
    # the parent, scoped, names its single timing block by the workload uuid
    parent = json.loads(json.dumps(_GROUP_PROBLEM))
    parent["_dir"] = str(card / "eval" / "problem")
    bk.apply_workload_scope(parent)
    parent_names = [str(v["_wl_name"]) for v in bk._task_shapes(parent)]
    # the child must hand the probe the SAME scoped problem (one group left)
    assert len(seen[0]["group"]) == 1
    child_names = [str(v["_wl_name"]) for v in bk._task_shapes(seen[0])]
    assert child_names == parent_names


def test_scoped_child_and_parent_shape_names_match(tmp_path, monkeypatch):
    """End to end over the real functions: with the scope env set, a freshly
    loaded child problem scoped the way run_bench does it names its shapes
    exactly as the parent's scoped timing blocks."""
    bk = _benchkit()
    monkeypatch.setenv(bk.WORKLOAD_SCOPE_ENV, "wl1")
    parent = json.loads(json.dumps(_GROUP_PROBLEM))
    bk.apply_workload_scope(parent)
    parent_names = {str(b.get("name")) for b in bk._task_shapes(parent)}
    child = json.loads(json.dumps(_GROUP_PROBLEM))
    bk.apply_workload_scope(child)
    child_names = {str(v["_wl_name"]) for v in bk._task_shapes(child)}
    assert child_names == parent_names
    assert len(child_names) == 1   # one scoped shape on both sides


# --- BUG-CH-G-02 -------------------------------------------------------------------

def _render_gpu_entry() -> str:
    spec = importlib.util.spec_from_file_location("corps_r3f_cardkit", CARDKIT)
    ck = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ck)
    return ck.render_gpu_entry(name="t", suite="peregrine_kernel", arch="sm100")


@pytest.mark.parametrize("task,refused", [
    ("all", True), ("perf", True), ("cardref", True), ("ref", True),
    ("allncu", True),          # the WP11 merged task: used to slip through unlocked
    ("ncu", False), ("help", False),
])
def test_clock_lock_refuses_allncu_too(task, refused):
    """A card that asks for a clock lock refuses every reading task when the lock
    did not hold -- the WP11 merged allncu task included (its first segment is
    the full timing protocol, exactly what an unlocked reading must not run)."""
    src = _render_gpu_entry()
    block = src[src.index("LB_REFUSE=0"):src.index('if [ "$LB_REFUSE" = "1" ]; then')]
    env = {"TASK": task, "LB_REQUIRE_CLOCK_LOCK": "1", "LB_CLOCK_STATE": "unlocked",
           "LB_SM_CLOCK_MHZ": "1410"}
    code = "\n".join(f"{k}={json.dumps(str(v))}" for k, v in env.items()) + "\n" + block \
        + f'\nprintf "%s" "$LB_REFUSE"\n'
    p = subprocess.run(["bash", "-c", code], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == ("1" if refused else "0"), (task, p.stdout, p.stderr)
