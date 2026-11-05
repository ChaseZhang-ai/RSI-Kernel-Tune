# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""WP22 completion: the declaration travels the record and report faces.

3e152cb landed the compute-dtype core (task key, card env, benchkit seam, gate
text, digest stamp, carry guard, red-team fixture). This file pins the pieces
the corrected design adds on top of it (user clarification 2026-11-05):

- IterationRecord.compute_dtype: every row names the declaration its numbers
  were read under (default "fp32"; load_records drops the key on pre-WP22
  files, which then read fp32 like every row before the field existed);
- build_digest refuses a row set that MIXES declarations (the memo-lane
  mixing-defense pattern) — numbers read under different declared precisions
  are never one comparison;
- the teacher's diagnose / patch turns carry the one-sentence contract (the
  precision axis is the task's locked constant, not a lever family); a default
  (fp32) run renders zero new prompt bytes;
- the driver stamps the declaration beside the lane at every IterationRecord
  site (a source-level count pins the stamping completeness);
- the report face (report.md) names the declaration when it is not the
  default, and adds nothing on fp32 (pre-WP22 reports, byte for byte);
- benchkit's declaration read goes through the unified guard chain (pin >
  kernel-import snapshot > live): a kernel's post-import env write moves the
  declaration at neither layer.

Runs on CPU (torch required by the benchkit module import).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
EVAL_SRC = REPO / "src" / "rsi_kernel_tune" / "card_template" / "eval"
DRIVER_SRC = REPO / "src" / "rsi_kernel_tune" / "loop" / "driver.py"

from rsi_kernel_tune.inner.trajectory import (  # noqa: E402
    IterationRecord, build_digest, load_records)
from rsi_kernel_tune.loop import driver as driver_mod  # noqa: E402
from rsi_kernel_tune.task import COMPUTE_DTYPES, load_task  # noqa: E402

from test_task import MINIMAL, _kb, _task  # noqa: E402


# -- the record field ------------------------------------------------------------------

_FIELDS = {f.name for f in dataclasses.fields(IterationRecord)}


def _rec(**kw) -> IterationRecord:
    base = dict(iter=0, decision_id="D1", change="widen tiles", correctness=True,
                verdict="KEEP", base_sha="a", sha="b",
                bwd_median_ms=1.0, device_median_ms=1.0, memo_lane=False)
    base.update(kw)
    return IterationRecord(**{k: v for k, v in base.items() if k in _FIELDS})


def test_record_defaults_to_fp32():
    assert _rec().compute_dtype == "fp32"


def test_record_carries_the_declaration(tmp_path):
    line = json.dumps({**{f.name: getattr(_rec(compute_dtype="fp16"), f.name)
                          for f in dataclasses.fields(IterationRecord)}})
    p = tmp_path / "status.jsonl"
    p.write_text(line + "\n", encoding="utf-8")
    rows = load_records(p)
    assert [r.compute_dtype for r in rows] == ["fp16"]


def test_pre_wp22_rows_read_fp32(tmp_path):
    """The append-only rule: a row written before the field existed loads with
    the default (load_records drops unknown keys; absence reads fp32)."""
    row = _rec()
    old = {k: v for k, v in dataclasses.asdict(row).items() if k != "compute_dtype"}
    p = tmp_path / "status.jsonl"
    p.write_text(json.dumps(old) + "\n", encoding="utf-8")
    assert [r.compute_dtype for r in load_records(p)] == ["fp32"]


# -- the mixing defense (the memo-lane pattern) -----------------------------------------

def test_digest_refuses_mixed_declarations():
    rows = [_rec(iter=1), _rec(iter=2, compute_dtype="fp16")]
    with pytest.raises(ValueError, match="different compute_dtype"):
        build_digest(rows, {"value": 1.0})


def test_digest_refuses_mixed_declarations_either_direction():
    rows = [_rec(iter=1, compute_dtype="bf16"), _rec(iter=2, compute_dtype="fp8")]
    with pytest.raises(ValueError, match="bf16, fp8|fp8, bf16"):
        build_digest(rows, {"value": 1.0})


def test_digest_accepts_uniform_declared_rows():
    rows = [_rec(iter=1, compute_dtype="fp16"), _rec(iter=2, compute_dtype="fp16")]
    d = build_digest(rows, {"value": 1.0}, compute_dtype="fp16")
    assert d["compute_dtype"] == "fp16"


# -- the teacher's one-sentence contract (diagnose / patch) ------------------------------

def _ctx(**kw):
    base = dict(memo_lane=False, compute_dtype="fp32")
    base.update(kw)
    return SimpleNamespace(**base)


def test_teacher_note_silent_on_the_default():
    assert driver_mod._compute_dtype_teacher_note(_ctx()) == ""


def test_teacher_note_names_the_declaration_and_the_arbiter():
    for dt in ("tf32", "bf16", "fp16", "fp8"):
        note = driver_mod._compute_dtype_teacher_note(_ctx(compute_dtype=dt))
        assert f"compute_dtype={dt}" in note
        assert "not a lever" in note          # the axis is a locked constant
        assert "REVERT" in note               # the gate is still the arbiter
        assert note.count("\n") <= 4          # one header + one line: stays small


def test_every_record_site_stamps_the_declaration():
    """Every IterationRecord stamp of the lane is accompanied by the declaration
    stamp, and the two non-record consumers (gate_failure, build_digest) are the
    only extra reads of ctx.compute_dtype."""
    src = DRIVER_SRC.read_text(encoding="utf-8")
    stamps = src.count("memo_lane=ctx.memo_lane")
    reads = src.count("compute_dtype=ctx.compute_dtype")
    assert stamps >= 1 and reads == stamps + 2


# -- the report face ---------------------------------------------------------------------

def _report(tmp_path: Path, toml: str) -> str:
    from rsi_kernel_tune import launch
    d = _task(tmp_path, MINIMAL + toml)
    task = load_task(d, kb_root=_kb(tmp_path))
    ops = tmp_path / "ops"
    ops.mkdir(exist_ok=True)
    run = SimpleNamespace(name="r1", dir=tmp_path, ws=tmp_path,
                          operators_dir=ops, task_dir=d,
                          record=lambda: {"status": "done"})
    p = launch.write_report(run, task, 0)
    return p.read_text(encoding="utf-8")


def test_report_names_a_non_default_declaration(tmp_path):
    text = _report(tmp_path, '[precision]\ncompute_dtype = "fp16"\n')
    assert "- compute_dtype: fp16" in text
    assert "never comparable" in text


def test_report_is_byte_identical_on_the_default(tmp_path):
    text = _report(tmp_path, "")
    assert "compute_dtype" not in text


# -- benchkit: the declaration reads through the guard chain ------------------------------

def _benchkit(env_dtype: str | None, name: str):
    """The template benchkit loaded fresh, with RKT_COMPUTE_DTYPE set exactly as
    the import should see it (the card copies the file verbatim; the env is read
    once at import)."""
    import os
    old = os.environ.pop("RKT_COMPUTE_DTYPE", None)
    if env_dtype is not None:
        os.environ["RKT_COMPUTE_DTYPE"] = env_dtype
    try:
        spec = importlib.util.spec_from_file_location(name, EVAL_SRC / "benchkit.py")
        bk = importlib.util.module_from_spec(spec)
        sys_modules = __import__("sys").modules
        sys_modules[name] = bk
        spec.loader.exec_module(bk)
        return bk
    finally:
        import os as _os
        _os.environ.pop("RKT_COMPUTE_DTYPE", None)
        if old is not None:
            _os.environ["RKT_COMPUTE_DTYPE"] = old


def test_chain_reads_the_pinned_declaration_over_a_kernel_write():
    """The kernel flips RKT_COMPUTE_DTYPE after its import; the bench-start pin
    (the operator's configuration) is what the measure reads."""
    bk = _benchkit("fp16", "wp22_chain_pin")
    import os
    os.environ["RKT_COMPUTE_DTYPE"] = "fp16"   # the launch world, as at bench start
    bk._pin_guard_env()
    try:
        os.environ["RKT_COMPUTE_DTYPE"] = "fp8"      # the kernel's write
        assert bk._compute_dtype() == "fp16"
    finally:
        bk._unpin_guard_env()
        os.environ.pop("RKT_COMPUTE_DTYPE", None)


def test_chain_reads_the_snapshot_over_a_kernel_write():
    """No pin (a direct reader's posture), but the kernel imported: the
    pre-import capture is what the measure reads."""
    import os
    bk = _benchkit("bf16", "wp22_chain_snap")
    os.environ["RKT_COMPUTE_DTYPE"] = "bf16"   # the launch world, as at kernel import
    try:
        bk._capture_kernel_import_env()
        os.environ["RKT_COMPUTE_DTYPE"] = "fp8"          # the kernel's write
        assert bk._compute_dtype() == "bf16"
    finally:
        os.environ.pop("RKT_COMPUTE_DTYPE", None)


def test_chain_absence_falls_back_to_the_import_frozen_value():
    """Every trusted layer says absent (the tests' posture: the env unset after
    import — a state production never reaches, since the pin and the snapshot
    are taken in the same process after this module's import): the
    import-validated declaration stands."""
    import os
    bk = _benchkit("fp16", "wp22_chain_absent")
    assert bk._ENV_AT_KERNEL_IMPORT is None and bk._GUARD_ENV is None
    os.environ.pop("RKT_COMPUTE_DTYPE", None)
    assert bk._compute_dtype() == "fp16"


def test_chain_garbage_falls_back_to_the_frozen_value_never_crashes():
    """A value import never validated (a kernel's junk write in a bare process)
    falls back to the frozen declaration instead of crashing a precision row.
    The junk never outlives the test: a later fresh module load (a card copy)
    validates its import-time env and must not see it."""
    import os
    bk = _benchkit("fp16", "wp22_chain_junk")
    os.environ["RKT_COMPUTE_DTYPE"] = "int4"
    try:
        assert bk._compute_dtype() == "fp16"
    finally:
        os.environ.pop("RKT_COMPUTE_DTYPE", None)


def test_chain_fp32_absent_is_fp32():
    bk = _benchkit(None, "wp22_chain_fp32")
    assert bk._compute_dtype() == "fp32"
    assert bk.COMPUTE_DTYPE == "fp32"                # the import-frozen face


def test_the_declared_dtype_ladder_is_the_task_ladder():
    """One vocabulary everywhere: the task layer, the CLI, and the bench name the
    same five precisions with the same default."""
    from rsi_kernel_tune.task import COMPUTE_DTYPE_DEFAULT
    import os
    old = os.environ.pop("RKT_COMPUTE_DTYPE", None)
    try:
        spec = importlib.util.spec_from_file_location(
            "wp22_chain_vocab", EVAL_SRC / "benchkit.py")
        bk = importlib.util.module_from_spec(spec)
        __import__("sys").modules["wp22_chain_vocab"] = bk
        spec.loader.exec_module(bk)
        assert tuple(bk.COMPUTE_DTYPES) == tuple(COMPUTE_DTYPES)
        assert bk.COMPUTE_DTYPE_DEFAULT == COMPUTE_DTYPE_DEFAULT == "fp32"
    finally:
        os.environ.pop("RKT_COMPUTE_DTYPE", None)
        if old is not None:
            os.environ["RKT_COMPUTE_DTYPE"] = old
