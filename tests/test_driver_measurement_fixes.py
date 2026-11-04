"""Regression tests for the measurement-fix batch on loop/driver.py.

Ported from the ForgeKernel fix wave (audit-r2-2026-11-05): each test names the
fix it pins (same numbering as the fix list). The fixtures reuse
test_loop_driver.py's FakeBackend/_ctx, so the whole file runs without a GPU.
"""

from __future__ import annotations

import json

import pytest

import peregrine_kernel.loop.driver as drv
from peregrine_kernel.baseline import RefBaseline
from peregrine_kernel.inner.trajectory import IterationRecord, append_record, load_records
from peregrine_kernel.loop.parse import parse_eval

from test_loop_driver import EVAL_DW, FakeBackend, _ctx


# --- fix 1: _paired_reanchor(gap=None) must not crash on the final log line ----

_CONFIRM = {"jobs": [{"ratio": 0.9}, {"ratio": 0.9}]}  # quiet re-measure, median 0.9


def _paired_ctx(tmp_path, be):
    ctx = _ctx(tmp_path, be)
    ctx.gate_window = "device"
    ctx.equivalence_ratio = 1.0
    ctx.reanchor_memo = None
    ctx.paired_memo = {}
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="c", correctness=True, verdict="KEEP",
        device_median_ms=1.0, sha="shabest"))
    return ctx


def test_paired_reanchor_without_an_absolute_gap_does_not_crash(tmp_path, monkeypatch,
                                                                capsys):
    # The ref_in_job path is entered with gap=None when the absolute target
    # covers other workloads than the gate reading (resumed run + workload
    # mismatch). The closing log line used to interpolate gap unconditionally
    # and raised TypeError there — after the confirmed-gap account was already
    # computed and persisted, and after the two confirmation jobs were bought,
    # so a watchdog restart bought them again.
    be = FakeBackend([], [])
    ctx = _paired_ctx(tmp_path, be)
    monkeypatch.setattr(drv, "run_best_confirmation", lambda *a, **k: dict(_CONFIRM))
    assert drv._paired_reanchor(ctx, 1.0, None) is None
    out = capsys.readouterr().out
    assert "absolute account said" not in out
    assert "paired residual" in out          # the confirmed-gap account still logged


def test_paired_reanchor_still_quotes_the_absolute_gap_when_it_has_one(tmp_path,
                                                                       monkeypatch,
                                                                       capsys):
    be = FakeBackend([], [])
    ctx = _paired_ctx(tmp_path, be)
    monkeypatch.setattr(drv, "run_best_confirmation", lambda *a, **k: dict(_CONFIRM))
    assert drv._paired_reanchor(ctx, 1.0, 0.066) is None
    assert "absolute account said +6.6%" in capsys.readouterr().out


# --- BUG-SEED-02: the unmeasured REVERT rows carry no verdict on green ---------

def test_unparseable_revert_row_is_neutral_on_green(tmp_path):
    # The record-is-None branches (budget exhausted, nothing evaluated: gate
    # refusal / probe undelivered / probe blocked / unparseable) used to file
    # correctness=False: run_inner_loop read that as "not green" and dropped the
    # lineage into scaffold mode, and _green_on_resume kept the False across
    # restarts. Nothing was measured, so the row must move nothing (the same
    # neutrality _harness_edit_revert already documents for its un-evaluated
    # REVERTs).
    be = FakeBackend(["garbage"] * 4, [])
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 4
    rec = drv.run_inner_iteration(ctx, 0, metric_before=2.6, last_good_ref="sha-prev")
    assert rec.verdict == "REVERT" and rec.decision_id == "unparseable"
    assert rec.correctness is None
    # a lineage whose only row is unmeasured does not resume as under construction
    assert drv._green_on_resume(load_records(ctx.status_path), "sha0") is None


def test_probe_blocked_revert_row_is_neutral_on_green(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('rig')\n", encoding="utf-8")
    be = FakeBackend(["PROBE_REQUEST: notes/p.py"] * 2, ["PROBE FAIL max_diff=9"] * 2)
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 2
    rec = drv.run_inner_iteration(ctx, 0, metric_before=3.0, last_good_ref="sha-prev")
    assert rec.verdict == "REVERT" and rec.decision_id == "probe-blocked"
    assert rec.correctness is None


# --- BUG-C1-02: probe reads/writes survive non-UTF-8 bytes ----------------------

def test_bad_bytes_in_a_kept_probe_do_not_stop_the_run(tmp_path):
    # The kept-probe read sides used bare read_text(): one stray non-UTF-8 byte
    # in the probe file raised UnicodeDecodeError (not caught by the OSError
    # handlers) — the reset that ends the iteration never happened because the
    # keep blew up first, and the next iteration's re-materialisation blew up
    # the same way (it runs every iteration at run_inner_iteration's top).
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_bytes(b"print('rig \xff')\n")
    be = FakeBackend(["PROBE_REQUEST: notes/p.py"] * 2, ["PROBE FAIL max_diff=9"] * 2)
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 2
    rec = drv.run_inner_iteration(ctx, 0, metric_before=3.0, last_good_ref="sha-prev")
    assert rec.decision_id == "probe-blocked"      # the iteration ended normally
    assert "notes/p.py" in ctx.kept_probes         # and the keep survived the reset
    drv._restore_kept_probes(ctx)                  # the next iteration's restore
    assert (tmp_path / "notes" / "p.py").read_bytes().startswith(b"print('rig ")


# --- BUG-SEED-29: the clean-stop safety net carries the traceback ---------------

def test_unexpected_error_log_carries_the_traceback(tmp_path, monkeypatch, capsys):
    # The unattended-run safety net logged {e!r} only: for a bug reached through
    # a few call frames (a None riding into a formatter) the repr is one line
    # with no location, so the 2am operator had nothing to debug from and the
    # lost run could not be triaged. The exception's traceback rides the log now.
    def _boom(ctx, iter_no, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(drv, "run_inner_iteration", _boom)
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.max_iters = 1
    assert drv.run_inner_loop(ctx, start_iter=0, initial_green=True) == "error"
    out = capsys.readouterr().out
    assert "unexpected error, stopping cleanly: RuntimeError('boom')" in out
    assert "Traceback (most recent call last)" in out
    assert "_boom" in out   # the frames under the loop, not just the message


# --- BUG-SEED-31: the sitting best's gpu_state is read same-cluster only --------

def test_best_keep_gpu_state_reads_same_cluster_rows_only(tmp_path, monkeypatch):
    # Absolute comparisons are same-cluster (the ~6% node-draw band): the
    # telemetry behind the confirmation's clock-drift check must mirror
    # _best_ms(_on_cluster(...)), not the global minimum — a foreign machine's
    # faster best once supplied the reference clocks, flagging drift (or hiding
    # it) that this machine never measured. The signature stays (records,
    # gate_window); the CALLERS pass cluster-filtered records — pinned here on
    # the confirmation path, where the drift check reads the sitting best.
    be = FakeBackend([], [EVAL_DW(2.4, 1.97)])
    ctx = _ctx(tmp_path, be)
    ctx.gate_window = "device"
    ctx.watermark_margin = 0.03
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D0", change="c", correctness=True, verdict="KEEP",
        device_median_ms=1.0, sha="s0", cluster="box-b",
        gpu_state={"sm_clock_mhz": 2000.0}))
    seen = []
    real = drv._best_keep_gpu_state

    def spy(records, gate_window):
        seen.append(list(records))
        return real(records, gate_window)

    monkeypatch.setattr(drv, "_best_keep_gpu_state", spy)
    drv._confirm_suspicious_watermark(ctx, be, "b", parse_eval(EVAL_DW(2.4, 1.95)),
                                      2.0, profiler={})
    assert seen, "the drift check must read the sitting best's telemetry"
    assert all(getattr(r, "cluster", "") != "box-b" for r in seen[0])
    # the same-cluster row that later lands IS the sitting best it reads
    append_record(ctx.status_path, IterationRecord(
        iter=1, decision_id="D1", change="c", correctness=True, verdict="KEEP",
        device_median_ms=1.5, sha="s1", gpu_state={"sm_clock_mhz": 1900.0}))
    seen.clear()
    be.gpu_logs = [EVAL_DW(2.4, 1.44)]
    drv._confirm_suspicious_watermark(ctx, be, "b", parse_eval(EVAL_DW(2.4, 1.45)),
                                      1.9, profiler={})
    assert seen and any(getattr(r, "cluster", "") != "box-b" and r.sha == "s1"
                        for r in seen[0])


# --- BUG-SEED-34: "part 0 of n" is malformed, not a free scaffold pass ----------

def test_part_zero_of_n_is_not_a_construction_declaration():
    # The two "part k of n" readers disagreed on k=0: _declared_construction
    # scaffolded any k<n (a full eval skipped for free on a malformed step),
    # while _open_package_tail refuses k outside 1<=k<n (pricing rollback at
    # once). The package contract's steps count from 1, so a k=0 declaration is
    # malformed input, not a declared build step: it must not defer the eval.
    assert drv._declared_construction("part 0 of 3: open the ledger") is None
    assert drv._declared_construction("part 1 of 3: open the ledger") == "part 1 of 3"
    # unchanged shapes: finished and over-cap declarations never scaffolded
    assert drv._declared_construction("part 3 of 3: done") is None
    assert drv._declared_construction("[PLACEHOLDER: D0.9 -> register split]") is not None


# --- BUG-SEED-35: a meta verdict probe's GPU hiccup is feedback, not a crash ----

class _FailingCommitBackend(FakeBackend):
    """A node whose git side just failed (disk full, lock left behind) — the
    probe's commit/push/submit run must not terminate the whole meta step."""

    def commit(self, message: str) -> str:
        raise drv.BackendError("git lock held")


def _probe_ctx(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('experiment')\n", encoding="utf-8")
    be = _FailingCommitBackend([], [])
    ctx = _ctx(tmp_path, be)
    return ctx, be


def test_meta_verdict_probe_backend_error_is_feedback_not_a_crash(tmp_path):
    # The probe's commit/push/submit chain was bare: one BackendError from any
    # of the three propagated out of the meta loop to the CLI and terminated an
    # hours-long run over a transient GPU/node hiccup. The sibling ablation
    # probe has always caught it into "job failed (...)" feedback — same here.
    ctx, _be = _probe_ctx(tmp_path)
    pv, log, readings = drv._run_meta_verdict_probe(ctx, ctx.backend,
                                                    "fs-demo-metaprobe",
                                                    "notes/p.py", 1)
    assert pv.found is False and "job failed" in pv.reason
    assert log == "" and readings == []
    # the rendered feedback carries the failure to the teacher
    feedback = drv._verdict_probe_feedback("notes/p.py", pv, log, 1, 3)
    assert "job failed" in feedback


def test_probe_manifest_backend_error_is_feedback_not_a_crash(tmp_path, monkeypatch):
    # The manifest path's commit/push/batch chain was equally bare.
    ctx, _be = _probe_ctx(tmp_path)
    be_ok = FakeBackend([], [])
    ran = {"batch": False}

    def fake_batch(c, b, branch, paths):
        ran["batch"] = True
        return [["notes/p.py", "=== slice ==="]]

    monkeypatch.setattr(drv, "_run_probe_manifest_batch", fake_batch)
    batch, err = drv._run_probe_manifest_commit_batch(
        ctx, be_ok, "fs-demo-metaprobe", "notes/p.py", ["notes/p.py"])
    assert ran["batch"] and err == "" and batch == [["notes/p.py", "=== slice ==="]]

    monkeypatch.setattr(drv, "_run_probe_manifest_batch",
                        lambda *a, **k: (_ for _ in ()).throw(drv.BackendError("queue gone")))
    batch, err = drv._run_probe_manifest_commit_batch(
        ctx, be_ok, "fs-demo-metaprobe", "notes/p.py", ["notes/p.py"])
    assert batch is None and "job failed" in err


# --- BUG-SEED-49: quarantining two same-named entries keeps both ---------------

def test_quarantine_keeps_both_same_named_entries(tmp_path):
    # The quarantine moved entries to qdir / src.name: two trees' entries with
    # the same file name (sm90/leak.md and sm100/leak.md quarantined in one
    # harvest) renamed onto the same destination — the second silently
    # replaced the first and one leaky entry's evidence was lost. The
    # destination now carries the entry's tree in the name (rel's separators
    # flattened), and a residual collision is broken by a sequence number.
    kb = tmp_path / "repo" / "hw_kb"
    for arch in ("sm90", "sm100"):
        (kb / arch).mkdir(parents=True)
        (kb / arch / "leak.md").write_text(f"{arch} leaked content\n", encoding="utf-8")
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    ctx.kb_dir = kb
    drv._quarantine_leaky_kb(ctx, ["sm90/leak.md:1: oracle narrative",
                                   "sm100/leak.md:2: oracle narrative"])
    qdir = tmp_path / "kb_quarantine"
    sm90 = qdir / "sm90__leak.md"
    sm100 = qdir / "sm100__leak.md"
    assert sm90.is_file() and "sm90 leaked content" in sm90.read_text(encoding="utf-8")
    assert sm100.is_file() and "sm100 leaked content" in sm100.read_text(encoding="utf-8")
    assert not (kb / "sm90" / "leak.md").exists()
    assert not (kb / "sm100" / "leak.md").exists()


def test_quarantine_collision_gets_a_sequence_number(tmp_path):
    kb = tmp_path / "repo" / "hw_kb"
    (kb / "sm90").mkdir(parents=True)
    (kb / "sm90" / "leak.md").write_text("v1\n", encoding="utf-8")
    # a name that flattens onto the same destination as sm90/leak.md
    (kb / "sm90__leak.md").write_text("root-level lookalike\n", encoding="utf-8")
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    ctx.kb_dir = kb
    drv._quarantine_leaky_kb(ctx, ["sm90/leak.md:1: x", "sm90__leak.md:1: x"])
    qdir = tmp_path / "kb_quarantine"
    bodies = {p.name: p.read_text(encoding="utf-8")
              for p in qdir.iterdir() if p.is_file()}
    assert len(bodies) == 2
    assert sorted(bodies.values()) == ["root-level lookalike\n", "v1\n"]


# --- BUG-C1-03: a cluster anchor read is window-checked -------------------------

def test_cluster_scale_ignores_an_anchor_from_another_window(tmp_path):
    # The anchor file stores ref_ms together with the window it measured
    # (the writer stamps window+home_ref_ms), but the read side never checked
    # the window: a device-window anchor scaled an e2e target (or the reverse)
    # — numbers from different timing windows dividing each other. A
    # mismatched anchor is no anchor: the target stop falls to the existing
    # re-measure/suspend channel instead.
    be = FakeBackend([], [])
    be.last_cluster = "box-b"
    ctx = _ctx(tmp_path, be)
    ctx.home_cluster = "home"
    ctx.gate_window = "device"
    ctx.ref_baseline = RefBaseline(op="demo", ms=2.0, ms_device=2.0, job="j",
                                   ref_name="cuDNN")
    ctx.reanchor_ref = None   # the fresh-measurement channel is closed here
    (tmp_path / "cluster_anchors.json").write_text(json.dumps(
        {"box-b": {"attempts": 1, "home_ref_ms": 2.0,
                   "window": "device", "ref_ms": 2.4}}), encoding="utf-8")
    assert drv._cluster_scale(ctx) == pytest.approx(1.2)
    # the same anchor read from the other window must not scale the target
    ctx.gate_window = "e2e"
    assert drv._cluster_scale(ctx) is None


# --- C-T2-01: the student digest's header line carries the clock header ---------

def test_student_digest_header_carries_the_clock_header(tmp_path):
    # build_digest computes the WP15 clock header (the last window's SM clock
    # beside the job mean — the cross-job comparison's first reading), but the
    # digest writer's key list dropped it, so the file never carried it; the
    # curve/regressions trimming above the write was dead with it (nothing but
    # the header keys and decisions is serialized).
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    ctx.gate_window = "device"
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D0", change="c", correctness=True, verdict="KEEP",
        bwd_median_ms=1.8, device_median_ms=1.8, sha="s0",
        gpu_state={"sm_clock_mhz": 1500,
                   "window_readings": [{"sm_clock_mhz": 1450}]}))
    drv._write_student_digest(ctx)
    path = drv._student_digest_path(ctx)
    head = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert head.get("clock", {}).get("sm_clock_mhz") == 1450
    # the header line still carries everything it always did
    assert set(head) >= {"sol", "target", "achieved", "flags", "clock"}
