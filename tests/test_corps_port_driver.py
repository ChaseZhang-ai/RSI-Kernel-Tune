"""Ports of the audit-r2 archive's measurement regression tests (loop/driver).

Source: audit-r2-2026-11-05/forgekernel-untracked-tests/test_driver_measurement_
fixes.py, ported to the latest tree per the correction order of 2026-11-05
(``forgekernel`` -> ``peregrine_kernel``, coordinates re-anchored).

Subset rationale (full inventory in the porting report of CHANGES-corps-fix.md):
the archive's eleven other tests are covered by this session's wave or their
premises were replaced (WP15 A1 confirms every new best; SEED-05/06/29/31 and
SEED-01/02 each have dedicated corps tests). Only the confirmation-protocol
coverage below adds something the session tests do not pin:

* a confirmation draw that lands PAST the old best buys the budget's SECOND
  job (the tie-break), and the median of all three draws settles the watermark
  (the session tests run the budget-1 variant only);
* the draw-to-draw agreement band prices the SUSPECT draw (margin x ms), with
  the old best as a separate dissent check -- rewritten from the archive's
  "agreement is anchored to the old best" test, whose premise (a
  metric_before-anchored band) no longer holds after the dissent rule landed.
"""

from __future__ import annotations

import pytest

from peregrine_kernel.inner.trajectory import IterationRecord as IR, append_record
from peregrine_kernel.loop.driver import _confirm_suspicious_watermark
from peregrine_kernel.loop.parse import parse_eval

from test_loop_driver import EVAL_DW, FakeBackend, _ctx


def _ratio_row(it, ms, ratio):
    return IR(iter=it, decision_id=f"D{it}", change="c", correctness=True,
              verdict="KEEP", bwd_median_ms=ms, device_median_ms=ms,
              vs_ref_device=ratio)


def _seed_own_readings(ctx, n, ms=2.0):
    for i in range(n):
        append_record(ctx.status_path, _ratio_row(i, ms, 1.20))


def _wm_ctx(tmp_path, be, margin=0.03):
    ctx = _ctx(tmp_path, be)
    ctx.gate_window = "device"
    ctx.watermark_margin = margin
    return ctx


def test_confirmation_slower_than_the_old_best_buys_the_tie_break(tmp_path, capsys):
    """Ported from the archive (test_confirmation_slower_than_the_old_best_is_
    not_a_confirmed_gain): a band trigger (2% gain over a quiet 6-sample ratio
    history, so the band resolves well under the margin) whose confirmation
    reads 2.01 -- SLOWER than the old best 2.0 yet inside the draw-to-draw
    band. The draw did not replicate: the budget's second job is bought (the
    tie-break), and the median of all three draws settles the watermark."""
    be = FakeBackend([], [EVAL_DW(2.4, 2.01), EVAL_DW(2.4, 2.03)])
    ctx = _wm_ctx(tmp_path, be)
    ctx.confirm_budget_per_iter = 2
    _seed_own_readings(ctx, 6)
    ev = parse_eval(EVAL_DW(2.4, 1.96))
    out, info = _confirm_suspicious_watermark(ctx, be, "b", ev, 2.0, profiler={})
    assert out.device_median_ms == pytest.approx(2.01)  # median of 1.96/2.01/2.03
    assert info["confirm_jobs"] == 2
    assert [c[1] for c in be.calls if c[0] == "gpu"] == ["all", "all"]
    assert "did not replicate" in capsys.readouterr().out


def test_confirmation_agreement_band_prices_the_suspect_draw(tmp_path, capsys):
    """Rewritten from the archive's test_confirmation_agreement_is_anchored_to_
    the_old_best. That test pinned a metric_before-anchored agreement band
    (0.03 x 2.0); the dissent rule changed the protocol: draw-to-draw agreement
    is priced against the SUSPECT draw (margin x ms), and the old best is a
    separate dissent check. These numbers distinguish the two anchors: the
    confirmation's 98.9 sits 2.9 off the suspect 96.0 -- outside the
    suspect-anchored band (0.03 x 96.0 = 2.88) but inside a metric_before-
    anchored one (0.03 x 100 = 3.0) -- so the tie-break job must be bought and
    the jump must settle unconfirmed, exactly what the old anchor would not do."""
    be = FakeBackend([], [EVAL_DW(105.0, 98.9), EVAL_DW(105.0, 98.8)])
    ctx = _wm_ctx(tmp_path, be)
    ev = parse_eval(EVAL_DW(103.0, 96.0))          # a 4% jump: the margin trigger
    out, info = _confirm_suspicious_watermark(ctx, be, "b", ev, 100.0, profiler={})
    assert info["confirm_jobs"] == 2
    assert out.device_median_ms == pytest.approx(98.8)  # median of 96.0/98.9/98.8
    text = capsys.readouterr().out
    assert "did not confirm" in text
    assert "confirmed — recording" not in text
