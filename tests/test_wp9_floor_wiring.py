"""WP9 wiring: the analytic floor's teacher-side consumption points.

- The NCU ground-truth tail carries `floor=` and `MFU=` lines (teacher side,
  diagnose_extra); the student-visible digest carries NEITHER (both assertions
  are the acceptance requirement).
- [target] floor_margin derives a stop for a run with no measured reference
  target; a measured target always wins.
- [loop] floor_stop closes a run as "floor" only inside the band AND only after
  a verified-BLOCKED diagnosis explored the structural channel.
- C2: ref_host_enqueue_ms rides the KEEP row.
"""

from __future__ import annotations

import json

from rsi_kernel_tune.loop.driver import (_analytic_floor_ms, _floor_stop_check,
                                      _floor_teacher_line, _mfu_teacher_line,
                                      _resolve_target)
from rsi_kernel_tune.inner.trajectory import IterationRecord, build_digest
from rsi_kernel_tune.loop.parse import parse_eval

from test_loop_driver import FakeBackend, _ctx

_GEMM_CARD = {
    "definition": {"name": "gemm_fp16_8192", "op_type": "gemm",
                   "inputs": {"a": {"shape": ["m", "k"], "dtype": "bfloat16"},
                              "b": {"shape": ["k", "n"], "dtype": "bfloat16"}},
                   "outputs": {"c": {"shape": ["m", "n"], "dtype": "bfloat16"}}},
    "workload": {"uuid": "w0", "axes": {"m": 4096, "n": 4096, "k": 4096}},
}


def _ctx_with_card(tmp_path, be, **over):
    """A ctx whose operator card carries a gemm problem.json (sm90 peaks)."""
    ctx = _ctx(tmp_path, be, **over)
    card_dir = tmp_path / "eval" / "problem"
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / "problem.json").write_text(json.dumps(_GEMM_CARD), encoding="utf-8")
    ctx.kb_arch = "sm90"
    return ctx


def _rec(**over) -> IterationRecord:
    base = dict(iter=1, decision_id="D1", change="x", correctness=True, verdict="KEEP",
                device_median_ms=1.0, mfu=42.0)
    base.update(over)
    return IterationRecord(**base)


# --- the NCU ground-truth tail lines ---------------------------------------------------

def test_floor_and_mfu_lines_render_teacher_side(tmp_path):
    be = FakeBackend([], [])
    ctx = _ctx_with_card(tmp_path, be)
    records = [_rec()]
    floor_line = _floor_teacher_line(ctx)
    assert floor_line.startswith("floor=") and "bytes=" in floor_line
    assert "flops=" in floor_line and "family=gemm" in floor_line
    mfu_line = _mfu_teacher_line(records, ctx)
    assert mfu_line.startswith("MFU=42.0% (headroom ")


def test_student_digest_carries_neither_line(tmp_path):
    """The acceptance pair: build_digest (the student-visible Layer-1) must not
    contain the floor or the MFU headroom lines, whatever the teacher side shows."""
    be = FakeBackend([], [])
    ctx = _ctx_with_card(tmp_path, be)
    records = [_rec()]
    digest = json.dumps(build_digest(records, {"value": 1.0}))
    assert "headroom" not in digest and "floor=" not in digest
    # and the teacher lines do render (the fixture is sound)
    assert _floor_teacher_line(ctx) and _mfu_teacher_line(records, ctx)


def test_mfu_line_absent_without_a_reading(tmp_path):
    be = FakeBackend([], [])
    ctx = _ctx_with_card(tmp_path, be)
    assert _mfu_teacher_line([_rec(mfu=None)], ctx) == ""


def test_floor_absent_without_a_card_renders_nothing(tmp_path):
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    assert _floor_teacher_line(ctx) == ""
    assert _analytic_floor_ms(ctx) is None


# --- [target] floor_margin ---------------------------------------------------------------

def test_floor_margin_derives_a_target_when_no_measured_one_exists(tmp_path):
    be = FakeBackend([], [])
    ctx = _ctx_with_card(tmp_path, be, target_ms=None)
    ctx.target_floor_margin = 0.30
    floor = _analytic_floor_ms(ctx)
    assert floor is not None
    assert _resolve_target(ctx) == floor * 1.30


def test_floor_margin_off_keeps_the_historical_none(tmp_path):
    be = FakeBackend([], [])
    ctx = _ctx_with_card(tmp_path, be, target_ms=None)
    ctx.target_floor_margin = 0.0
    assert _resolve_target(ctx) is None


def test_a_measured_target_beats_the_floor_margin(tmp_path):
    be = FakeBackend([], [])
    ctx = _ctx_with_card(tmp_path, be, target_ms=1.9)
    ctx.target_floor_margin = 0.30
    assert _resolve_target(ctx) == 1.9


# --- [loop] floor_stop -----------------------------------------------------------------

def test_floor_stop_fires_only_inside_the_band_and_only_after_blocked(tmp_path):
    be = FakeBackend([], [])
    ctx = _ctx_with_card(tmp_path, be)
    floor = _analytic_floor_ms(ctx)
    best = floor * (1.0 + ctx.floor_stop_margin) * 0.999

    ctx.floor_stop = True
    assert _floor_stop_check(ctx, best, meta_steps=2) is None, \
        "inside the band but no structural exploration: not stopping"
    ctx.wp5_reach_blocked = True
    stop = _floor_stop_check(ctx, best, meta_steps=2)
    assert stop is not None and stop["stop"] == "floor" and stop["floor_ms"] == floor

    ctx.wp5_reach_blocked = False
    assert _floor_stop_check(ctx, floor * 1.5, meta_steps=2) is None, "above the band"
    ctx.floor_stop = False
    ctx.wp5_reach_blocked = True
    assert _floor_stop_check(ctx, best, meta_steps=2) is None, "opt-in off"


# --- C2: ref_host_enqueue_ms rides the KEEP row ------------------------------------------

def test_parse_and_record_carry_ref_host_enqueue_ms(tmp_path):
    ev = parse_eval(
        "=== JSON ===\n"
        + json.dumps({"correctness": {"pass": True},
                      "performance": {"device_median_ms": 1.0,
                                      "ref_device_ms": 2.0,
                                      "ref_host_enqueue_ms": 0.123}})
        + "\n")
    assert ev.ref_host_enqueue_ms == 0.123
    rec = _rec(ref_host_enqueue_ms=ev.ref_host_enqueue_ms)
    assert rec.ref_host_enqueue_ms == 0.123
