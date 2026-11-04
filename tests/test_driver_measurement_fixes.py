"""Regression tests ported from the ForgeKernel fix wave (loop/driver measurement
batch).

Only the mechanisms that exist in this lineage are pinned; the one that survives
is the same-card paired re-anchor quoting the absolute gap it contradicts.

Dropped from the source file, with reasons (see the BUG-PORT ledger where the
old test exposed a live bug here):

* _paired_reanchor(gap=None) crashing on the final log line: the closing line
  interpolates ``gap`` unconditionally, so the workload-mismatch path (gap=None,
  which the function itself designs for and its note text handles) still raises
  TypeError after the account was persisted (BUG-PORT);
* unmeasured REVERT rows (unparseable / probe-blocked) carrying no verdict on
  green: run_inner_iteration files those rows with correctness=False here, so an
  exhausted student budget demotes a green lineage to scaffold on resume
  (BUG-PORT);
* the watermark band needing own-job samples / ignoring copied (reading_of)
  rows: superseded by the WP15 layered estimator -- the data-poor free pass is
  deliberately gone ("every new best confirms once") and the band's samples are
  this generation's own design, covered by this repo's WP15 watermark tests;
* _judge_patch's absolute branch reading the 6% cross-job band: superseded --
  the verdict currency is now the same-job ratio first, then
  _patch_improve_eps (the layered estimator with its own cap), covered by this
  repo's judge tests;
* the confirmation protocol pins (agree AND beat the old best, median of all
  draws): superseded by the WP15 protocol (a failed FIRST confirmation keeps
  the single reading; band triggers buy budget jobs without a tie-break),
  covered by this repo's watermark tests;
* the clean-stop safety net carrying the traceback: the handler here logs only
  ``{e!r}`` (BUG-PORT);
* the sitting best's gpu_state read same-cluster only: the drift check's
  _best_keep_gpu_state reads the global best with no cluster filter here
  (BUG-PORT).
"""

from __future__ import annotations

import pytest

import peregrine_kernel.loop.driver as drv
from peregrine_kernel.inner.trajectory import IterationRecord, append_record
from peregrine_kernel.loop.parse import parse_eval

from test_loop_driver import EVAL_DW, FakeBackend, _ctx


_CONFIRM = {"jobs": [{"ratio": 0.9}, {"ratio": 0.9}]}  # quiet re-measure, median 0.9


def test_paired_reanchor_still_quotes_the_absolute_gap_when_it_has_one(tmp_path,
                                                                       monkeypatch,
                                                                       capsys):
    # ref_in_job re-anchor with an absolute gap on file: the confirmed-gap account
    # still names what the absolute number said, so the diagnosis can see the two
    # estimators disagreed.
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    ctx.gate_window = "device"
    ctx.equivalence_ratio = 1.0
    ctx.reanchor_memo = None
    ctx.paired_memo = {}
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="c", correctness=True, verdict="KEEP",
        device_median_ms=1.0, sha="shabest"))
    monkeypatch.setattr(drv, "run_best_confirmation", lambda *a, **k: dict(_CONFIRM))
    assert drv._paired_reanchor(ctx, 1.0, 0.066) is None
    assert "absolute account said +6.6%" in capsys.readouterr().out
