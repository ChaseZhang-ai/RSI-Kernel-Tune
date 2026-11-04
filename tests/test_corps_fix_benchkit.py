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
