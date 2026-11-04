"""WP20 task 3+4 accounting: gate/judge bookkeeping rows, the edge-band judge
recheck, the nondeterminism fail form, the milestone tag-wipe WARN — and the
zero-verdict-control tests (every new reading must leave the baseline judgment
byte-identical when it does not fire).

The bench-side emission (precision_audit / gate_degraded / the top-level
precision block) is asserted live through run_bench --mode correctness; the
driver-side consumption through the scripted FakeBackend loop.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
EVAL_SRC = REPO / "src" / "peregrine_kernel" / "card_template" / "eval"

from test_fresh_recheck import HONEST_KERNEL  # noqa: E402  (conftest path)
from test_loop_driver import NCU, REC, FakeBackend, _ctx  # noqa: E402


def _eval(correctness: dict | None = None, **extra) -> str:
    obj = {"correctness": correctness if correctness is not None else {"pass": True},
           "performance": {"device_median_ms": 1.0, "bwd_median_ms": 1.0}}
    obj.update(extra)
    return "=== JSON ===\n" + json.dumps(obj) + "\n"


def _edge_fail(margin: float = 0.95) -> str:
    return _eval({"pass": False, "status": "INCORRECT_NUMERICAL",
                  "checks": [{"group": "g0", "output": "output",
                              "error": margin, "tolerance": 1.0}]})


# --- the WP24 regression fix + the audit row, live through the bench -------------

_DEVICE_AGNOSTIC_JUDGE = (
    "import torch\n\n"
    "def run(hidden_states, weight):\n"
    "    x = hidden_states.float()\n"
    "    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float())\n"
    "\n"
    "def make_inputs(axes_and_scalars, group=None, seed=0, device='cpu'):\n"
    "    g = torch.Generator().manual_seed(seed)\n"   # device ignored: the CPU
    "    n = int(axes_and_scalars['batch_size'])\n"    # rehearsal draws where it runs
    "    return {'hidden_states': torch.randn(n, 64, generator=g),\n"
    "            'weight': (torch.rand(64, generator=g) + 0.5)}\n"
)


def _card_with(problem_extras: dict | None = None, tmp_path=None) -> Path:
    from test_fresh_recheck import PROBLEM, BENCH, _card
    problem = json.loads(json.dumps(PROBLEM))
    problem["definition"]["reference"] = _DEVICE_AGNOSTIC_JUDGE
    problem.update(problem_extras or {})
    return _card(tmp_path, HONEST_KERNEL, problem=problem)


def _bench_module(card: Path):
    spec = importlib.util.spec_from_file_location(
        f"redteam_acct_{card.name}", card / "eval" / "benchkit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _force_cpu(lb, monkeypatch):
    """The parent bench hardcodes device="cuda"; the CPU rehearsal pins the
    tensor allocation / normalization to cpu (this file asserts the JSON
    assembly and the accounting, not the device)."""
    orig_alloc, orig_norm = lb.allocate_outputs, lb.normalize_outputs
    orig_inputs = lb.task_inputs
    monkeypatch.setattr(lb, "allocate_outputs", lambda p, d: orig_alloc(p, "cpu"))
    monkeypatch.setattr(lb, "normalize_outputs", lambda p, r, d: orig_norm(p, r, "cpu"))
    monkeypatch.setattr(lb, "task_inputs",
                        lambda p, g, d, m=None: orig_inputs(p, g, "cpu", m))


def test_precision_block_rides_the_top_level_and_names_its_contract(tmp_path, capsys, monkeypatch):
    """The WP24 regression fix + the WP20 audit row, live: the parent eval JSON
    carries the precision block at TOP level again (the rename dropped the
    re-attach; strict cards read `unreported` and every iteration REVERTed),
    beside `precision_audit` and `gate_degraded`."""
    card = _card_with(tmp_path=tmp_path)
    lb = _bench_module(card)
    _force_cpu(lb, monkeypatch)
    assert lb.run_bench(["--mode", "correctness"],
                        problem_path=card / "eval" / "problem" / "problem.json") == 0
    out = json.loads(capsys.readouterr().out.split("=== JSON ===")[1])
    assert isinstance(out.get("precision"), dict), (
        "the top-level precision block is back (WP24 rename regression fixed)")
    assert out["precision"]["status"] == "pass"
    assert out["precision_audit"] == "strict (default)"
    assert out["gate_degraded"] is False   # rows ran; nothing went blind


def test_precision_contract_off_names_itself(tmp_path, capsys, monkeypatch):
    card = _card_with({"precision_contract": "off"}, tmp_path=tmp_path)
    lb = _bench_module(card)
    _force_cpu(lb, monkeypatch)
    assert lb.run_bench(["--mode", "correctness"],
                        problem_path=card / "eval" / "problem" / "problem.json") == 0
    out = json.loads(capsys.readouterr().out.split("=== JSON ===")[1])
    assert out["precision_audit"] == "off (problem)"
    assert "precision" not in out          # the measure is off: no block at all
    assert "gate_degraded" not in out      # no rows to speak of: the row is absent


def test_gate_degraded_marks_an_all_unavailable_block(tmp_path, capsys, monkeypatch):
    """The gate's own visibility: a strict card whose precision rows all report
    `unavailable` is a gate that saw nothing — gate_failure reads only
    status=='fail', so this row is the only trace. Zero verdict weight."""
    card = _card_with(tmp_path=tmp_path)
    lb = _bench_module(card)
    _force_cpu(lb, monkeypatch)
    # the honest kernel, but the precision helper cannot measure (no float rows):
    # force every row unavailable through the recorder's own unavailable() path
    P = lb._precision()
    orig_unavail = P.PrecisionRecorder.unavailable
    P.PrecisionRecorder.measure = (
        lambda self, *a, **k: orig_unavail(self, "forced", "forced unavailable"))
    try:
        assert lb.run_bench(["--mode", "correctness"],
                            problem_path=card / "eval" / "problem" / "problem.json") == 0
    finally:
        del P.PrecisionRecorder.measure   # restore the class's own method
    out = json.loads(capsys.readouterr().out.split("=== JSON ===")[1])
    assert out["precision"]["status"] == "unavailable"
    assert out["gate_degraded"] is True, (
        "all-unavailable rows beyond the ratio mark the eval gate_degraded")
    # the verdict itself is untouched: accounting, never a conviction
    assert out["correctness"]["pass"] is True


def test_gate_degraded_ratio_env_overrides(tmp_path, monkeypatch):
    lb = _bench_module(_card_with(tmp_path=tmp_path))
    rows = {"rows": [{"status": "unavailable"}, {"status": "pass"}]}
    assert lb.gate_degraded(rows) is False          # 1/2 <= 0.5 default is not >
    monkeypatch.setenv("PK_GATE_DEGRADED_RATIO", "0.4")
    assert lb.gate_degraded(rows) is True           # 1/2 > 0.4
    monkeypatch.setenv("PK_GATE_DEGRADED_RATIO", "bogus")
    assert lb.gate_degraded(rows) is False          # bad value: the default stands


# --- corr_margin_ratio: parsed once, carried by the REVERT row --------------------

def test_corr_margin_ratio_parsed_from_the_checks():
    from peregrine_kernel.loop.parse import corr_margin_ratio, parse_eval
    ev = parse_eval(_eval({"pass": False, "checks": [
        {"output": "o", "error": 0.5, "tolerance": 1.0},
        {"output": "o", "error": 0.95, "tolerance": 1.0}]}))
    assert ev.corr_margin_ratio == pytest.approx(0.95)   # the WORST, not the first
    ev2 = parse_eval(_eval({"pass": False}))             # a harness-level fail
    assert ev2.corr_margin_ratio is None                 # absent, never a passing one


def test_revert_row_carries_the_judge_edge_it_sat_on(tmp_path):
    from peregrine_kernel.inner.trajectory import load_records
    from peregrine_kernel.loop.driver import run_inner_iteration
    be = FakeBackend([REC("D1", 1.0)], [_edge_fail(5.0), NCU])  # far outside the band
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec.verdict == "REVERT"
    assert rec.corr_margin_ratio == pytest.approx(5.0), (
        "the margin that was buried verbatim in the error text is now a field")
    assert rec.gate_degraded is None


# --- the edge-band judge recheck (one measurement, the unchanged rules) -----------

def test_edge_band_first_offender_buys_one_recheck_and_keeps(tmp_path):
    """0.9 < margin < 1.1 on the first offender: one fresh-draw re-eval; the
    fresh draw clears the judge — the KEEP row carries judge_edge_recheck=pass
    and the FRESH reading (the failed draw never becomes a row)."""
    from peregrine_kernel.inner.trajectory import load_records
    from peregrine_kernel.loop.driver import run_inner_iteration
    be = FakeBackend([REC("D1", 1.0)], [_edge_fail(0.95), NCU, _eval(), NCU])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec.verdict == "KEEP" and rec.judge_edge_recheck == "pass"
    modes = [c[1] for c in be.calls if c[0] == "gpu"]
    assert modes == ["all", "ncu", "all"], ("the profiler rode the failed attempt; the recheck buys the eval only")


def test_edge_recheck_that_fails_again_reverts_as_measured(tmp_path):
    from peregrine_kernel.inner.trajectory import load_records
    from peregrine_kernel.loop.driver import run_inner_iteration
    be = FakeBackend([REC("D1", 1.0)], [_edge_fail(0.95), NCU, _edge_fail(1.02), NCU])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec.verdict == "REVERT"
    assert rec.judge_edge_recheck is None
    assert "correctness FAIL" in rec.error


def test_edge_recheck_fires_once_per_run(tmp_path):
    from peregrine_kernel.inner.trajectory import load_records
    from peregrine_kernel.loop.driver import run_inner_loop
    be = FakeBackend([REC("D1", 1.0), REC("D2", 1.0)],
                     [_edge_fail(0.95), NCU, _edge_fail(0.97),          # iter 1: eval, profile, recheck
                      _edge_fail(0.93), NCU])                           # iter 2: judged as measured
    ctx = _ctx(tmp_path, be)
    ctx.max_iters = 2
    run_inner_loop(ctx, start_iter=0)
    records = load_records(ctx.status_path)
    assert [r.verdict for r in records] == ["REVERT", "REVERT"]
    modes = [c[1] for c in be.calls if c[0] == "gpu"]
    assert modes == ["all", "ncu", "all", "all", "ncu"], ("the recheck fired once (iter 1's third job); iter 2 was judged as measured")


def test_edge_recheck_off_or_out_of_band_judges_immediately(tmp_path):
    """The kill switch and the band's edges: an explicit false (the zero-verdict
    control) and a margin outside (0.9, 1.1) both keep the bare judgment."""
    from peregrine_kernel.inner.trajectory import load_records
    from peregrine_kernel.loop.driver import run_inner_iteration
    for ctx_flag, margin in ((False, 0.95), (None, 0.5), (None, 0.9), (None, 1.1)):
        be = FakeBackend([REC("D1", 1.0)], [_edge_fail(margin), NCU])
        d = tmp_path / f"c{ctx_flag}-{margin}"; d.mkdir(); ctx = _ctx(d, be)
        if ctx_flag is False:
            ctx.judge_edge_recheck = False
        rec = run_inner_iteration(ctx, 0, metric_before=3.0)
        assert rec.verdict == "REVERT"
        modes = [c[1] for c in be.calls if c[0] == "gpu"]
        assert modes == ["all", "ncu"], f"no recheck job for flag={ctx_flag}, margin={margin}"


# --- the fresh_recheck fail form (nondeterministic vs cache replay) ---------------

def test_nondeterministic_form_is_recorded_not_reweighted(tmp_path):
    """A fresh_recheck fail whose same-seed double run disagreed with itself is
    a DIFFERENT fail form than a cache replay. Both REVERT (a nondeterministic
    kernel is not deliverable); the row says which — suspicious still True (the
    WP9 marker stays; the form is a reading on the row)."""
    from peregrine_kernel.inner.trajectory import load_records
    from peregrine_kernel.loop.driver import run_inner_iteration
    fr = {"mode": "regen", "pass": False, "form": "nondeterministic",
          "nondeterminism": {"status": "ran", "max_diff": 0.5, "n_pairs": 6},
          "layers": [{"status": "INCORRECT_NUMERICAL"}]}
    be = FakeBackend([REC("D1", 1.0)], [_eval(fresh_recheck=fr), NCU])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec.verdict == "REVERT" and rec.suspicious is True
    assert rec.fresh_recheck_form == "nondeterministic"
    assert "form=nondeterministic" in rec.error
    # the plain cache-replay fail keeps the default form (None)
    be2 = FakeBackend([REC("D1", 1.0)],
                      [_eval(fresh_recheck={"mode": "regen", "pass": False,
                                            "layers": [{"status": "X"}]}), NCU])
    d3 = tmp_path / "plain"; d3.mkdir(); rec2 = run_inner_iteration(_ctx(d3, be2), 0, metric_before=3.0)
    assert rec2.verdict == "REVERT" and rec2.suspicious is True
    assert rec2.fresh_recheck_form is None


# --- board_span_unstable: the reference's own mirror pair is the noisy side -------

_WL_JSON = {
    "device_median_ms": 1.0, "bwd_median_ms": 1.0, "n_workloads": 1,
    "workloads": [{"name": "w0", "device_median_ms": 1.0,
                   "device_readings_ms": [1.0, 1.001],        # quiet: 0.1%
                   "ref_device_readings_ms": [1.0, 1.5]}],    # wild: 50%
}


def test_board_span_unstable_reported_never_judged(tmp_path, capsys):
    from peregrine_kernel.inner.trajectory import load_records
    from peregrine_kernel.loop.driver import run_inner_iteration
    be = FakeBackend([REC("D1", 1.0)], [_eval(performance=_WL_JSON), NCU])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec.verdict == "KEEP", "report-only: the guard never flips a verdict"
    assert rec.board_span_unstable is True
    assert "board_span_unstable" in capsys.readouterr().out, (
        "the fragment renders into the teacher's workload block")


def test_board_span_guard_off_or_quiet_is_silent(tmp_path):
    from peregrine_kernel.inner.trajectory import load_records
    from peregrine_kernel.loop.driver import _board_span_unstable, run_inner_iteration
    be = FakeBackend([REC("D1", 1.0)], [_eval(performance=_WL_JSON), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.board_span_guard = False
    rec = run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec.board_span_unstable is False
    # and a quiet pair (the reference as quiet as the kernel) raises nothing
    from peregrine_kernel.loop.parse import parse_eval
    ev = parse_eval(_eval(performance=_WL_JSON))
    ev.workloads[0]["ref_device_readings_ms"] = [1.0, 1.001]
    d2 = tmp_path / "quiet"; d2.mkdir(); ctx2 = _ctx(d2, FakeBackend([], []))
    assert _board_span_unstable(ctx2, ev) is None


# --- the zero-verdict-control: closed keys, baseline judgment ---------------------

def test_zero_verdict_control_all_rows_quiet_on_a_baseline_eval(tmp_path):
    """With the WP20 switches off (or untriggered), a plain eval's row carries
    none of the new readings and the loop buys the baseline two jobs — the
    judgment path is byte-identical to pre-WP20."""
    from peregrine_kernel.inner.trajectory import load_records
    from peregrine_kernel.loop.driver import run_inner_iteration
    be = FakeBackend([REC("D1", 1.0)], [_eval(), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.judge_edge_recheck = False
    ctx.board_span_guard = False
    rec = run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec.verdict == "KEEP" and rec.bwd_median_ms == 1.0
    assert rec.corr_margin_ratio is None and rec.gate_degraded is None
    assert rec.fresh_recheck_form is None and rec.judge_edge_recheck is None
    assert rec.board_span_unstable is False
    assert [c[1] for c in be.calls if c[0] == "gpu"] == ["all", "ncu"]
    # and with the defaults (None = spec defaults) the same eval judges the same:
    be2 = FakeBackend([REC("D1", 1.0)], [_eval(), NCU])
    d6 = tmp_path / "d"; d6.mkdir(); rec2 = run_inner_iteration(_ctx(d6, be2), 0, metric_before=3.0)
    assert rec2.verdict == "KEEP" and rec2.bwd_median_ms == 1.0
    assert [c[1] for c in be2.calls if c[0] == "gpu"] == ["all", "ncu"]


# --- config plumbing ---------------------------------------------------------------

def test_wp20_apply_cfg_resolves_and_protects(monkeypatch, tmp_path):
    from peregrine_kernel.loop import driver
    ctx = _ctx(tmp_path, FakeBackend([], []))
    monkeypatch.setattr(driver, "_wp5_cfg_sections", lambda c: {"loop": {
        "gate_degraded_ratio": 0.7, "judge_edge_recheck": False,
        "board_span_guard": "yes"}})          # a non-bool: the default stands
    driver._wp20_apply_cfg(ctx)
    assert ctx.gate_degraded_ratio == 0.7
    assert ctx.judge_edge_recheck is False
    assert ctx.board_span_guard is True
    # an explicitly set field (tests, callers) wins and is never clobbered
    d4 = tmp_path / "b"; d4.mkdir(); ctx2 = _ctx(d4, FakeBackend([], []))
    ctx2.gate_degraded_ratio = 0.3
    driver._wp20_apply_cfg(ctx2)
    assert ctx2.gate_degraded_ratio == 0.3
    # out-of-range / garbage ratios fall back to the spec default
    d5 = tmp_path / "c"; d5.mkdir(); ctx3 = _ctx(d5, FakeBackend([], []))
    monkeypatch.setattr(driver, "_wp5_cfg_sections", lambda c: {"loop": {
        "gate_degraded_ratio": 9.0}})
    driver._wp20_apply_cfg(ctx3)
    assert ctx3.gate_degraded_ratio == 0.5


# --- the milestone tag-wipe WARN (N-B16): silence made visible, nothing more -------

from test_loop_driver import _MILESTONE  # noqa: E402  (a schema-conformant baseline)

_TAGGED = _MILESTONE   # the conformant baseline is new-era (it carries tags)
_WIPED = (_MILESTONE.replace(" [SYNC: none]", "")
                    .replace(" [GATHER: none]", "")
                    .replace(" [PHASE: none]", ""))   # conformant, zero tags


def test_tag_wipe_warns_and_still_accepts(tmp_path):
    """The accepted residual hole stays accepted — the validator does not reject
    the wiped text; the WARN is the only trace (never a violation, never a
    re-prompt)."""
    from peregrine_kernel.milestone import tag_wipe_warnings, validate_milestone
    assert "[SYNC:" in _TAGGED, "the baseline must be new-era (tags present)"
    assert tag_wipe_warnings(_TAGGED, _WIPED) != []
    assert validate_milestone(_WIPED) == [], "the accepted ruling stands"
    assert tag_wipe_warnings(_TAGGED, _TAGGED) == []     # nothing wiped: silent
    assert tag_wipe_warnings(None, _WIPED) == []         # no baseline: silent


def test_driver_logs_the_tag_wipe(tmp_path, capsys):
    from peregrine_kernel.loop.driver import _milestone_tagwipe_warnings
    p = tmp_path / "m.md"
    p.write_text(_WIPED, encoding="utf-8")
    warns = _milestone_tagwipe_warnings(p, baseline=_TAGGED)
    assert warns and "every tag was deleted" in warns[0]
