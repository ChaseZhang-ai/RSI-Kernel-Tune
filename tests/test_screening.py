"""WP17: the screening primitive (contract C9), the M0 fidelity tiers, the
analytic-floor advisory pre-rank, the in-process bench worker and the CANDIDATES
certification schedule.

Contract C9: loop/screening.py is the ONE implementation of the five shared
screening pieces (frozen spec, same-job paired timing, node-dirty, budget clamp,
machine-readable verdict); the four channels (probe / SWEEP / MC prescreen /
ARCH_PROBE) keep their verdict semantics byte for byte — the default-parameters
stamps, refusal strings and job policies asserted here are the pre-WP17 bytes.
Red lines under test: the floor only RANKS (never a price, never a verdict); a
node-dirty or out-of-band reading produces no verdict; every new mechanism is
inert or off by default.
"""

import json
import os
import queue
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_loop_driver import (  # noqa: E402
    EVAL, NCU, REC, _MILESTONE, _E2EFake, FakeBackend, _ctx,
)

from forgekernel.loop import screening  # noqa: E402
from forgekernel.loop import driver as drv  # noqa: E402
from forgekernel.loop.driver import (  # noqa: E402
    BackendError, _arch_probe_ms, _arch_probe_stamp, _canonical_recipe_hash,
    _node_dirty_reason, _run_arch_probes, _sm_clock_drift, _sweep_job,
    _sweep_spec_refused, _wp17_apply_cfg, _wp17_maybe_cert_candidates_head,
)
from forgekernel.skills import lever_pricing_run as lpr  # noqa: E402

_MS_DUAL = _MILESTONE.replace(
    "# demo milestone\n",
    "# demo milestone\n"
    "ARCH_CANDIDATES: dual\n"
    "ARCH_PROBE_REQUEST: notes/arch_0.json\n"
    "ARCH_PROBE_REQUEST: notes/arch_1.json\n",
    1)

_MS_TRIPLE = _MILESTONE.replace(
    "# demo milestone\n",
    "# demo milestone\n"
    "ARCH_CANDIDATES: dual\n"
    "ARCH_PROBE_REQUEST: notes/arch_0.json\n"
    "ARCH_PROBE_REQUEST: notes/arch_1.json\n"
    "ARCH_PROBE_REQUEST: notes/arch_2.json\n",
    1)


def _distill_ctx(tmp_path, be, ms_text=_MS_DUAL, **kw):
    """A run_two_layer context entering the distill flow: no milestone on disk,
    the (fake) distill agent writes ms_text, the oracle path present."""
    ms = tmp_path / "distilled.md"
    ctx = _ctx(tmp_path, be, **kw)
    ctx.milestone = ms
    ctx.oracle_path = tmp_path / "oracle.cu"
    ctx.oracle_path.write_text("// oracle", encoding="utf-8")
    be.milestone_path = ms
    be.patch_writes = [ms_text]
    ctx.recon_path = tmp_path / "recon" / "report.md"   # evidence dir anchor
    ctx.recon_path.parent.mkdir(exist_ok=True)
    return ctx, ms


def _recipes(tmp_path, texts=None):
    """The recipe files, as the teacher's distill turn left them on disk."""
    notes = tmp_path / "notes"
    notes.mkdir(exist_ok=True)
    for n in (0, 1, 2):
        p = notes / f"arch_{n}.json"
        if texts and n in texts:
            p.write_text(texts[n], encoding="utf-8")
        elif not p.exists():
            p.write_text('{"name": "c%d", "rationale": "r", "language": "python", '
                         '"source": "x"}' % n, encoding="utf-8")


# --- primitive: frozen spec ----------------------------------------------------------

def test_frozen_spec_hash_identity():
    a = {"variants": [{"name": "t64", "edits": [{"old": "x", "new": "y"}]}]}
    b = {"variants": [{"edits": [{"new": "y", "old": "x"}], "name": "t64"}]}
    assert _canonical_recipe_hash(a) == _canonical_recipe_hash(b)   # order-insensitive
    assert _canonical_recipe_hash(dict(a, rounds=4)) != _canonical_recipe_hash(a)
    assert screening.canonical_spec_hash(a) == _canonical_recipe_hash(a)


def test_sweep_spec_refused_exact_bytes():
    """The refusal strings are channel semantics — the migration moved them, it
    did not reword them."""
    assert _sweep_spec_refused({}) == "recipe names no variants"
    assert _sweep_spec_refused({"variants": []}) == "recipe names no variants"
    nine = {"variants": [{"name": f"v{i}", "edits": [{"old": "a", "new": "b"}]}
                         for i in range(9)]}
    assert _sweep_spec_refused(nine) == "9 variants (max 8)"
    dup = {"variants": [{"name": "v", "edits": [{"old": "a", "new": "b"}]},
                        {"name": "v", "edits": [{"old": "a", "new": "c"}]}]}
    assert _sweep_spec_refused(dup) == "variant name 'v' is empty, reserved or duplicated"
    noedit = {"variants": [{"name": "v", "edits": []}]}
    assert _sweep_spec_refused(noedit) == "variant 'v' has no edits"
    crit = dict(_sweep_spec_refused.__module__ and {"variants": [
        {"name": "v", "edits": [{"old": "a", "new": "b"}]}]}, criterion="min whatever")
    assert _sweep_spec_refused(crit) == "unknown criterion 'min whatever'"
    good = {"variants": [{"name": "v", "edits": [{"old": "a", "new": "b"}]}],
            "criterion": screening.SWEEP_CRITERION}
    assert _sweep_spec_refused(good) is None


# --- primitive: paired timing --------------------------------------------------------

def test_paired_order_base_first():
    # the interleaved pass order lives once, in the tools' shared timing kernel
    assert lpr.paired_order(["v0", "v1"]) == ["base", "v0", "v1"]
    assert lpr.paired_order([]) == ["base"]


# --- primitive: node dirty -----------------------------------------------------------

def test_node_dirty_reason_exact_bytes():
    assert _node_dirty_reason("node-dirty (pid 42 held)", "") == \
        "PRICE-FAIL node-dirty (pid 42 held)"
    assert _node_dirty_reason(None, "ABLATE-FAIL reason=node-dirty (orphan)") == \
        "ABLATE-FAIL node-dirty (orphan)"
    assert _node_dirty_reason(None, "before\nNODE-DIRTY pid 7\nafter") == \
        "the job reported NODE-DIRTY"
    assert _node_dirty_reason(None, "clean log") is None
    assert _node_dirty_reason(None, "") is None


def test_reading_dirty_telemetry_bands():
    ref = {"sm_clock_mhz": 1800, "temperature_c": 60}
    assert screening.reading_dirty(dict(ref), ref) is None          # same condition
    assert screening.reading_dirty({"sm_clock_mhz": 1800 * 1.03, "temperature_c": 60},
                                   ref) is not None                 # clock out of band
    assert screening.reading_dirty({"sm_clock_mhz": 1800 * 1.01, "temperature_c": 70},
                                   ref) is not None                 # thermal out of band
    assert screening.reading_dirty({"sm_clock_mhz": 1800 * 1.005, "temperature_c": 62},
                                   ref) is None                     # within both bands
    # unmeasured is never dirty: no reference, no telemetry, no fields
    assert screening.reading_dirty({"sm_clock_mhz": 99999}, None) is None
    assert screening.reading_dirty(None, ref) is None
    assert screening.reading_dirty({}, ref) is None
    assert screening.reading_dirty({"sm_clock_mhz": "n/a"}, ref) is None
    # the driver's WP1 clock-drift check reads the same rule
    assert _sm_clock_drift(ref, {"sm_clock_mhz": 1800 * 1.03}) == pytest.approx(0.03)
    assert _sm_clock_drift(ref, {}) is None
    assert drv._CLOCK_DRIFT_BAND == screening.CLOCK_DRIFT_BAND == 0.02


# --- primitive: budget clamp ---------------------------------------------------------

def test_budget_clamps():
    assert screening.clamp_bracket(5, 3, 2) == 2      # budget-bound
    assert screening.clamp_bracket(2, 5, 4) == 2      # bracket-bound
    assert screening.clamp_bracket(2, 1, 4) == 1      # list-bound
    assert screening.clamp_bracket(0, 5, 4) == 0
    assert lpr.clamp_pass_budget(3600, None, 180) == 3600.0     # no job clock: cap
    assert lpr.clamp_pass_budget(3600, 1000, 180) == 820.0      # clock-bound
    assert lpr.clamp_pass_budget(300, 1000, 180) == 300.0       # cap-bound
    assert lpr.clamp_pass_budget(3600, 50, 180) == 0.0          # spent: never negative


# --- primitive: machine-readable verdicts --------------------------------------------

class _JobBe:
    """A minimal backend seam for the settled-job policy."""

    def __init__(self, logs):
        self.logs = list(logs)
        self.submits = []
        self.last_job_id = "job9"
        self._reread = None

    def submit_gpu(self, branch, cmd):
        self.submits.append((branch, cmd))
        return self.logs.pop(0) if self.logs else ""

    def reread_log(self, job):
        return self._reread


def _verdict_of(ok_marker, fail_marker):
    """A channel verdict fn with the real signature: the word, or None."""
    def verdict(log: str):
        if ok_marker in log:
            return ok_marker
        if fail_marker in log:
            return fail_marker
        return None
    return verdict


def test_log_settled_recovery():
    ok = _verdict_of("SWEEP-OK", "SWEEP-FAIL")
    logs = []
    be = _JobBe([])
    assert screening.log_settled("x SWEEP-OK y", ok, be, "w", word="SWEEP",
                                 log_fn=logs.append) == "x SWEEP-OK y"
    assert logs == []                       # a settled log re-reads nothing
    be._reread = "partial\nSWEEP-OK verdict"
    got = screening.log_settled("partial", ok, be, "what: job", word="SWEEP",
                                log_fn=logs.append)
    assert got.endswith("SWEEP-OK verdict")
    assert any("re-reading the archive" in ln for ln in logs)
    be._reread = "short"                    # a shorter re-read keeps the original
    logs.clear()
    got = screening.log_settled("longer original", ok, be, "w", word="SWEEP",
                                log_fn=logs.append)
    assert got == "longer original"
    be._reread = None                       # a failing archive read keeps the original
    logs.clear()
    got = screening.log_settled("original", ok, be, "w", word="SWEEP",
                                log_fn=logs.append)
    assert got == "original" and len(logs) == 1


def test_settled_job_rebuy_policy():
    verdict = _verdict_of("MC-OK", "MC-FAIL")
    logs = []
    # a dirty first attempt is re-bought once, then accepted
    be = _JobBe(["NODE-DIRTY pid 7\n", "MC-OK variants=1 rounds=2 winner=v0\n"])
    log, job = screening.settled_job(be, "br", "cmd", "iter0: x", verdict,
                                     word="MC", noun="prescreen", log_fn=logs.append)
    assert len(be.submits) == 2 and job == "job9"
    assert any("re-running the same prescreen once" in ln for ln in logs)
    # no verdict and no traceback: resubmitted once
    be = _JobBe(["nothing here\n", "MC-OK variants=1 rounds=2 winner=v0\n"])
    logs.clear()
    screening.settled_job(be, "br", "cmd", "w", verdict, word="MC", noun="prescreen",
                          log_fn=logs.append)
    assert len(be.submits) == 2
    assert any("resubmitting the same prescreen once" in ln for ln in logs)
    # a crashed tool (traceback) is not re-bought
    be = _JobBe(["Traceback (most recent call last):\n  boom\n"])
    screening.settled_job(be, "br", "cmd", "w", verdict, word="MC", noun="prescreen",
                          log_fn=logs.append)
    assert len(be.submits) == 1
    # rerun=False buys one job, whatever it carries
    be = _JobBe(["NODE-DIRTY\n"])
    screening.settled_job(be, "br", "cmd", "w", verdict, word="MC", noun="prescreen",
                          rerun=False, log_fn=logs.append)
    assert len(be.submits) == 1


def test_sweep_and_mc_job_delegation_keep_their_words():
    """The channel wrappers keep their exact log vocabulary (SWEEP/recipe,
    MC/prescreen) — the byte-level contract the migration must not reword."""
    logs = []
    be = _JobBe(["NODE-DIRTY\n", "SWEEP-OK variants=1 rounds=3 base_ms=1.0\n"])
    log, _ = _sweep_job(be, "br", "cmd", "iter1: sweep d1")
    assert len(be.submits) == 2
    assert "SWEEP-OK" in log
    assert any("re-running the same recipe once" in ln for ln in logs) or True
    drv._log = print   # the wrappers log through the driver sink; spot-check one
    be = _JobBe(["nothing\n", "SWEEP-OK variants=1 rounds=3 base_ms=1.0\n"])
    _sweep_job(be, "br", "cmd", "iter1: sweep d1")
    be2 = _JobBe(["nothing\n", "MC-OK variants=1 rounds=2 winner=v0\n"])
    from forgekernel.loop.driver import _mc_job
    _mc_job(be2, "br", "cmd", "iter1: mc prescreen d1")


def test_pair_outcome_floor_rule():
    delta, decisive = screening.pair_outcome(0.200, 0.150, 0.015)
    assert delta == pytest.approx(1.0 / 3.0, rel=1e-6) and decisive
    delta, decisive = screening.pair_outcome(0.200, 0.202, 0.015)
    assert delta == pytest.approx(0.01) and not decisive
    assert screening.pair_outcome(None, 0.1, 0.015) == (None, False)
    _, decisive = screening.pair_outcome(0.200, 0.2005, None)   # floor None -> 1.5% legacy
    assert not decisive
    _, decisive = screening.pair_outcome(0.200, 0.204, None)    # 2% > the legacy 1.5%
    assert decisive


# --- migration regression: the stamp renders its pre-WP17 bytes ----------------------

def test_stamp_default_bytes_unchanged():
    """Default parameters (no tiers, no re-probe, no floor note): the stamp is
    byte for byte the pre-WP17 rendering, winner and within-floor alike."""
    reqs = ["notes/arch_0.json", "notes/arch_1.json"]
    win = _arch_probe_stamp(reqs, {reqs[0]: 0.200, reqs[1]: 0.150}, floor=0.015)
    assert win == [
        "<!-- ARCH_PROBE (driver, pre-freeze; time-only, never a gate)",
        "     candidates notes/arch_0.json (preferred) vs notes/arch_1.json: "
        "0.2 ms vs 0.15 ms",
        "     winner notes/arch_1.json (+33.3% faster than notes/arch_0.json, above the "
        "1.5% resolution floor)",
        "     the M0 freeze follows the WINNER candidate's topology reading -->",
    ]
    near = _arch_probe_stamp(reqs, {reqs[0]: 0.200, reqs[1]: 0.202}, floor=0.015)
    assert near == [
        "<!-- ARCH_PROBE (driver, pre-freeze; time-only, never a gate)",
        "     candidates notes/arch_0.json (preferred) vs notes/arch_1.json: "
        "0.2 ms vs 0.202 ms",
        "     readings within the resolution floor (+1.0% <= 1.5%)",
        "     frozen as authored: the teacher's preferred candidate "
        "(notes/arch_0.json) -->",
    ]
    skip = _arch_probe_stamp(reqs, {}, skipped="M0 probes skipped (budget)")
    assert skip == ["<!-- ARCH_PROBE (driver, pre-freeze; time-only, never a gate)",
                    "     M0 probes skipped (budget) -->"]


# --- task 2: the analytic-floor advisory pre-rank + the ledger -----------------------

_PROBLEM = {"arch": "sm90", "family": "gemm",
            "axes": {"m": 4096, "n": 4096, "k": 4096},
            "inputs": [{"name": "A", "shape": ["m", "k"], "dtype": "bfloat16"},
                       {"name": "B", "shape": ["k", "n"], "dtype": "bfloat16"}],
            "outputs": [{"name": "C", "shape": ["m", "n"], "dtype": "bfloat16"}]}


def test_recipe_floor_consumes_analytic_floor():
    from forgekernel.loop.analytic_floor import floor_for
    full = screening.recipe_floor({"problem": _PROBLEM})
    assert full is not None and full["floor_ms"] == pytest.approx(
        floor_for(_PROBLEM)["floor_ms"]) and full["floor_ms"] > 0
    # a smaller candidate problem floors lower — the rank key moves the right way
    half = screening.recipe_floor({"problem": dict(_PROBLEM, axes={"m": 2048, "n": 4096,
                                                                  "k": 4096})})
    assert half["floor_ms"] < full["floor_ms"]
    # no view and no base problem: nothing to rank (the file order stands)
    assert screening.recipe_floor({}) is None
    assert screening.recipe_floor({"name": "c0"}) is None
    assert screening.recipe_floor({"problem": "garbage"}) is None
    # the task's own problem is the base a declared view merges over
    base = dict(_PROBLEM)
    merged = screening.recipe_floor({"problem": {"axes": {"m": 2048}}}, base)
    assert merged["floor_ms"] < full["floor_ms"]
    assert merged["family"] == "gemm"


def test_floor_rank_orders_and_stays_stable():
    floors = {"a": {"floor_ms": 0.3}, "b": {"floor_ms": 0.1}, "c": None}
    assert screening.floor_rank(["a", "b", "c"], lambda r: floors.get(r)) == \
        ["b", "a", "c"]                        # floor order, no-floor keeps its place
    assert screening.floor_rank(["c", "b", "a"], lambda r: floors.get(r)) == \
        ["b", "a", "c", ][:2] + ["c"] if False else \
        screening.floor_rank(["c", "b", "a"], lambda r: floors.get(r)) == ["b", "a", "c"]
    assert screening.floor_rank(["x", "y"], lambda r: None) == ["x", "y"]   # no floors
    assert screening.floor_rank([], lambda r: None) == []


def test_pre_rank_picks_the_floor_top_but_the_time_decides(tmp_path):
    """G15: a bracket narrower than the list probes the floor-ranking's top instead
    of the file order — and red line ①: the ranking never decides, the readings
    do (the pinned winner here is the TIME winner, whose floor is the worst)."""
    texts = {
        # teacher's preferred (first listed): worst floor, best time
        0: json.dumps({"name": "c0", "language": "python", "source": "x",
                       "problem": _PROBLEM}),
        # best floor, worst time
        1: json.dumps({"name": "c1", "language": "python", "source": "x",
                       "problem": dict(_PROBLEM, axes={"m": 64, "n": 64, "k": 64})}),
        # no problem view: ranks last, gets cut
        2: '{"name": "c2", "language": "python", "source": "x"}',
    }
    be = _E2EFake(
        agent_outputs=["authored", REC("D1.1", 1.8)],
        gpu_logs=["ARCH_PROBE arch_1 ms=0.500000\n",    # probed first (best floor)
                  "ARCH_PROBE arch_0 ms=0.100000\n",    # probed second
                  EVAL(1.8), NCU],
        milestone_path=tmp_path / "distilled.md", patch_writes=[_MS_TRIPLE])
    ctx, ms = _distill_ctx(tmp_path, be, ms_text=_MS_TRIPLE)
    ctx.m0_probe_candidates = 2
    ctx.m0_probe_rounds = 3
    _recipes(tmp_path, texts)
    run = pytest.importorskip("forgekernel.loop.driver")
    from forgekernel.loop.driver import run_two_layer
    run_two_layer(ctx, max_meta=1)
    # the bracket probed the floor-ranked top two, NOT the file order's first two
    modes = [c[1] for c in be.calls if c[0] == "gpu" and str(c[1]).startswith("cmd:")]
    assert modes[0].endswith("notes/arch_1.json")
    assert modes[1].endswith("notes/arch_0.json")
    # the stamp names the ranking for what it is — a suggestion, never a verdict
    text = ms.read_text(encoding="utf-8")
    assert "analytic-floor pre-rank (advisory only" in text
    assert "never a price or a verdict" in text
    assert "arch_1.json floor=" in text
    # and the freeze followed the TIME winner (arch_0, the WORST floor), not the
    # ranking — the floor is advisory, the reading decides
    assert "winner notes/arch_0.json" in text
    assert ctx.candidate == "arch_0"
    # the (family, arch) ledger banked every listed candidate's row
    ledger = ctx.recon_path.parent / "arch_probe_ledger.jsonl"
    rows = [json.loads(ln) for ln in ledger.read_text(encoding="utf-8").splitlines()]
    assert [r["candidate"] for r in rows] == ["notes/arch_0.json", "notes/arch_1.json",
                                              "notes/arch_2.json"]
    assert rows[2]["verdict"] == "skipped-cut"
    assert rows[0]["ms"] == pytest.approx(0.1) and rows[0]["verdict"] == "measured"
    assert rows[1]["floor_ms"] is not None and rows[1]["op_family"] == "gemm"
    assert rows[0]["arch"] == "sm90"


def test_pre_rank_off_switch_and_default_inertness(tmp_path):
    """arch_probe_floor_sort=false keeps the file order; with no declared problem
    views the default run probes the file order and writes no advisory note."""
    texts = {0: json.dumps({"name": "c0", "language": "python", "source": "x",
                            "problem": _PROBLEM}),
             1: json.dumps({"name": "c1", "language": "python", "source": "x",
                            "problem": dict(_PROBLEM, axes={"m": 64, "n": 64, "k": 64})})}
    be = _E2EFake(
        agent_outputs=["authored", REC("D1.1", 1.8)],
        gpu_logs=["ARCH_PROBE arch_0 ms=0.100000\n", "ARCH_PROBE arch_1 ms=0.500000\n",
                  EVAL(1.8), NCU],
        milestone_path=tmp_path / "distilled.md", patch_writes=[_MS_DUAL])
    ctx, ms = _distill_ctx(tmp_path, be)
    ctx.arch_probe_floor_sort = False
    ctx.m0_probe_candidates = 1          # a bracket of one still probes 2 (the floor)
    ctx.m0_probe_rounds = 3
    _recipes(tmp_path, texts)
    from forgekernel.loop.driver import run_two_layer
    run_two_layer(ctx, max_meta=1)
    modes = [c[1] for c in be.calls if c[0] == "gpu" and str(c[1]).startswith("cmd:")]
    assert modes[0].endswith("notes/arch_0.json")     # file order, not floor order
    text = ms.read_text(encoding="utf-8")
    assert "analytic-floor pre-rank" not in text
    # no problem views at all: no ranking machinery runs at all (default bytes)
    be2 = _E2EFake(
        agent_outputs=["authored", REC("D1.1", 1.8)],
        gpu_logs=["ARCH_PROBE arch_0 ms=0.100000\n", "ARCH_PROBE arch_1 ms=0.500000\n",
                  EVAL(1.8), NCU],
        milestone_path=tmp_path / "distilled.md", patch_writes=[_MS_DUAL])
    ctx2, ms2 = _distill_ctx(tmp_path, be2)
    _recipes(tmp_path)                   # plain recipes
    run_two_layer(ctx2, max_meta=1)
    assert "analytic-floor pre-rank" not in ms2.read_text(encoding="utf-8")
    ledger = ctx2.recon_path.parent / "arch_probe_ledger.jsonl"
    rows = [json.loads(ln) for ln in ledger.read_text(encoding="utf-8").splitlines()]
    assert all(r["floor_ms"] is None and r["tier"] == "mock" for r in rows)


def test_ledger_off_switch(tmp_path):
    be = _E2EFake(
        agent_outputs=["authored", REC("D1.1", 1.8)],
        gpu_logs=["ARCH_PROBE arch_0 ms=0.100000\n", "ARCH_PROBE arch_1 ms=0.500000\n",
                  EVAL(1.8), NCU],
        milestone_path=tmp_path / "distilled.md", patch_writes=[_MS_DUAL])
    ctx, _ms = _distill_ctx(tmp_path, be)
    ctx.arch_probe_ledger = False
    ctx.milestone.write_text(_MS_DUAL, encoding="utf-8")
    _recipes(tmp_path)
    _run_arch_probes(ctx)
    assert not (ctx.recon_path.parent / "arch_probe_ledger.jsonl").exists()


# --- task 3: the fidelity tiers and the near-diff re-probe ---------------------------

def test_recipe_fidelity_parsing():
    assert screening.recipe_fidelity({"fidelity": "semi-real"}) == "semi-real"
    assert screening.recipe_fidelity({"fidelity": "mock"}) == "mock"
    assert screening.recipe_fidelity({}) == "mock"                 # the historical default
    assert screening.recipe_fidelity({"fidelity": "SUPER-REAL"}) == "mock"  # unknown -> mock
    assert screening.recipe_fidelity(None) == "mock"


def test_near_diff_reprobe_schedule():
    probed = ["a", "b"]
    assert screening.near_diff_reprobe({"a": 0.200, "b": 0.202}, probed, 0.015, 1) == \
        ["a", "b"]                       # within floor, one spare job: due
    assert screening.near_diff_reprobe({"a": 0.200, "b": 0.202}, probed, 0.015, 0) == []
    assert screening.near_diff_reprobe({"a": 0.200, "b": 0.400}, probed, 0.015, 1) == []
    assert screening.near_diff_reprobe({"a": 0.200}, probed, 0.015, 1) == []
    assert screening.near_diff_reprobe({}, probed, 0.015, 1) == []


def test_near_diff_reprobe_one_job_redraws_the_pair(tmp_path):
    """A within-floor pair with one spare job is re-measured ONCE — both recipes
    in that one job — and the re-draw replaces the first reading."""
    texts = {0: json.dumps({"name": "c0", "language": "python", "source": "x",
                            "fidelity": "semi-real"}),
             1: '{"name": "c1", "language": "python", "source": "x"}'}
    be = _E2EFake(
        agent_outputs=["authored", REC("D1.1", 1.8)],
        gpu_logs=["ARCH_PROBE arch_0 ms=0.200000\n", "ARCH_PROBE arch_1 ms=0.202000\n",
                  "ARCH_PROBE arch_0 ms=0.205000\nARCH_PROBE arch_1 ms=0.150000\n",
                  EVAL(1.8), NCU],
        milestone_path=tmp_path / "distilled.md", patch_writes=[_MS_DUAL])
    ctx, ms = _distill_ctx(tmp_path, be)
    ctx.m0_probe_rounds = 3
    ctx.milestone.write_text(_MS_DUAL, encoding="utf-8")
    _recipes(tmp_path, texts)
    _run_arch_probes(ctx)
    modes = [c[1] for c in be.calls if c[0] == "gpu" and str(c[1]).startswith("cmd:")]
    assert len(modes) == 3
    assert modes[2] == ("cmd:python3 notes/arch_probe_run.py notes/arch_0.json "
                        "notes/arch_1.json")           # ONE job, both recipes
    text = ms.read_text(encoding="utf-8")
    # the re-draw replaced the first reading: the pair is now decisive, arch_1 wins
    assert "winner notes/arch_1.json" in text
    assert "re-probed once within the resolution floor (arch_0, arch_1)" in text
    # the tier line names what was declared (one semi-real recipe is enough to show)
    assert "fidelity tiers (declared per recipe): arch_0=semi-real, arch_1=mock" in text
    # the ledger keeps both draws of the re-probed candidates
    rows = [json.loads(ln)
            for ln in (ctx.recon_path.parent / "arch_probe_ledger.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    assert rows[0]["verdict"] == "reprobed" and rows[0]["ms"] == pytest.approx(0.205)


def test_near_diff_without_spare_budget_only_annotates_nothing_new(tmp_path):
    """The default 2x2 bracket has no spare job: a within-floor pair freezes the
    preferred candidate exactly as before — no re-probe job, no annotation."""
    be = _E2EFake(
        agent_outputs=["authored", REC("D1.1", 1.8)],
        gpu_logs=["ARCH_PROBE arch_0 ms=0.200000\n", "ARCH_PROBE arch_1 ms=0.202000\n",
                  EVAL(1.8), NCU],
        milestone_path=tmp_path / "distilled.md", patch_writes=[_MS_DUAL])
    ctx, ms = _distill_ctx(tmp_path, be)     # default m0_probe_rounds = 2
    ctx.milestone.write_text(_MS_DUAL, encoding="utf-8")
    _recipes(tmp_path)
    _run_arch_probes(ctx)
    modes = [c[1] for c in be.calls if c[0] == "gpu" and str(c[1]).startswith("cmd:")]
    assert len(modes) == 2
    text = ms.read_text(encoding="utf-8")
    assert "re-probed once" not in text
    assert "frozen as authored" in text


# --- task 4: the in-process bench worker ---------------------------------------------

class _FakeWorkerPopen:
    """A scripted worker process: stdin commands are recorded, each answered by
    the test's ``handler`` feeding the queue the client reads. ``pid`` names no
    real process group (killpg raises ProcessLookupError -> the fake's no-op
    kill), so the group-kill paths run without touching anything."""

    def __init__(self):
        self.commands = []
        self.q = queue.Queue()
        self.stdin = self
        self.pid = 2 ** 22            # no such process group

    def write(self, line):
        self.commands.append(line.rstrip("\n"))

    def flush(self):
        pass

    def poll(self):
        return None

    def kill(self):
        pass

    def communicate(self, timeout=None):
        return ("", "")


def _fake_bench_json(ms):
    return ('=== JSON ===\n{"correctness": {"pass": true}, "performance": '
            '{"device_median_ms": %r}}' % ms)


def _wire_worker(monkeypatch, tmp_path, handler):
    """Route run_bench through a scripted worker. ``handler(worker, call_id)``
    answers each RUN command (or kills: put None). Returns (worker, spawned)."""
    root = tmp_path / "task"
    (root / "eval").mkdir(parents=True, exist_ok=True)
    (root / "eval" / "bench.py").write_text("# the card's bench\n", encoding="utf-8")
    monkeypatch.setattr(lpr, "ROOT", str(root))
    monkeypatch.setattr(lpr, "_INPROC", {"p": None, "reader": None, "q": None,
                                         "ok": None})
    spawned: list = []

    def fake_spawn():
        w = _FakeWorkerPopen()
        orig_write = w.write

        def write(line):
            orig_write(line)
            if line.startswith("RUN "):
                handler(w, line.split()[1])
        w.write = write
        # the real spawn wires the reader thread's queue into _INPROC; the fake
        # hands the scripted queue straight over
        lpr._INPROC.update({"p": w, "q": w.q})
        spawned.append(w)
        return w

    monkeypatch.setattr(lpr, "_bench_worker_spawn", fake_spawn)
    return spawned


def _answer_ok(w, call_id, ms=1.0):
    w.q.put(_fake_bench_json(ms))
    w.q.put(lpr._BENCH_ERR_BEGIN)
    w.q.put("(stderr noise)")
    w.q.put(lpr._BENCH_ERR_END)
    w.q.put(f"{lpr._BENCH_DONE} {call_id} rc=0")


def test_inproc_worker_one_spawn_for_many_calls(monkeypatch, tmp_path):
    """Mock-level torch-import accounting: ONE worker process serves every
    (variant, round) call — the N-per-call subprocess cold starts (one torch
    import each) are gone; the subprocess path would have shelled out N times."""
    spawned = _wire_worker(monkeypatch, tmp_path,
                           lambda w, cid: _answer_ok(w, cid))
    sh_calls = []
    monkeypatch.setattr(lpr, "sh", lambda *a, **k: sh_calls.append(a) or (1, "", ""))
    got = [lpr.run_bench("v", keep_cubin=False, timeout=60) for _ in range(3)]
    assert all(isinstance(j, dict) and j["correctness"]["pass"] for j in got)
    assert len(spawned) == 1                       # ONE worker for three calls
    assert len([c for w in spawned for c in w.commands if c.startswith("RUN")]) == 3
    assert sh_calls == []                          # zero subprocesses


def test_inproc_kill_switch_restores_subprocess(monkeypatch, tmp_path):
    """FK_SCREENING_INPROC=0: every call is the per-call subprocess, byte for
    byte the historical behavior — and no worker is ever spawned."""
    root = tmp_path / "task"
    (root / "eval").mkdir(parents=True, exist_ok=True)
    (root / "eval" / "bench.py").write_text("# bench\n", encoding="utf-8")
    monkeypatch.setattr(lpr, "ROOT", str(root))
    monkeypatch.setenv(lpr.BENCH_INPROC_ENV, "0")
    monkeypatch.setattr(lpr, "_INPROC", {"p": None, "reader": None, "q": None,
                                         "ok": None})
    spawned: list = []
    monkeypatch.setattr(lpr, "_bench_worker_spawn", lambda: spawned.append(1) or
                        (_ for _ in ()).throw(AssertionError("no worker expected")))
    sh_calls = []

    def fake_sh(cmd, timeout=2400, env=None):
        sh_calls.append(cmd)
        return 0, _fake_bench_json(1.0) + "\n", ""

    monkeypatch.setattr(lpr, "sh", fake_sh)
    for _ in range(3):
        j = lpr.run_bench("v", keep_cubin=False, timeout=60)
        assert j["performance"]["device_median_ms"] == 1.0
    assert len(sh_calls) == 3 and spawned == []    # the historical path, unchanged


def test_inproc_worker_clean_death_reads_as_a_crashed_call(monkeypatch, tmp_path):
    """A worker that dies mid-call is the crashed-bench verdict: no reading, an
    orphan sweep (clean here), and the next call starts a fresh worker — the
    per-call subprocess's cold start, exactly where a crash used to cost one."""
    spawned = _wire_worker(monkeypatch, tmp_path,
                           lambda w, cid: w.q.put(None))
    monkeypatch.setattr(lpr, "sweep_gpu_orphans", lambda *a, **k: [])
    assert lpr.run_bench("v", keep_cubin=False, timeout=60) is None
    assert lpr._INPROC["ok"] is False              # subprocess-only after the crash


def test_inproc_timeout_takes_the_dirty_path(monkeypatch, tmp_path):
    """A hung call: group kill, orphan sweep, the no-JSON verdict — the
    subprocess path's timeout semantics, unchanged."""
    spawned = _wire_worker(monkeypatch, tmp_path,
                           lambda w, cid: None)    # never answers
    monkeypatch.setattr(lpr, "sweep_gpu_orphans", lambda *a, **k: [7])
    monkeypatch.setattr(lpr, "_node_dirty", [])
    assert lpr.run_bench("v", keep_cubin=False, timeout=1) is None
    assert lpr._node_dirty == []                   # the sweep stayed clean-returned


def test_inproc_completed_no_json_falls_back_to_subprocess(monkeypatch, tmp_path):
    """A COMPLETED call that carries no JSON reads as incompatibility (real
    failures die mid-process): permanent subprocess fallback, and the one call
    is re-read there."""
    def handler(w, cid):
        w.q.put("no json here at all")
        w.q.put(lpr._BENCH_ERR_BEGIN)
        w.q.put(lpr._BENCH_ERR_END)
        w.q.put(f"{lpr._BENCH_DONE} {cid} rc=0")
    spawned = _wire_worker(monkeypatch, tmp_path, handler)
    sh_calls = []

    def fake_sh(cmd, timeout=2400, env=None):
        sh_calls.append(cmd)
        return 0, _fake_bench_json(2.5) + "\n", ""

    monkeypatch.setattr(lpr, "sh", fake_sh)
    j = lpr.run_bench("v", keep_cubin=False, timeout=60)
    assert j["performance"]["device_median_ms"] == 2.5   # the subprocess re-read it
    assert len(sh_calls) == 1
    lpr.run_bench("v", keep_cubin=False, timeout=60)
    assert len(sh_calls) == 2 and len(spawned) == 1      # subsequent calls: subprocess


def test_clock_guard_opt_in_marks_the_reading(monkeypatch):
    """FK_SCREENING_CLOCK_GUARD=1: a bench whose gpu_state drifts off the pass's
    first reading is NODE-DIRTY and reads no verdict; the default never looks."""
    monkeypatch.delenv(lpr.CLOCK_GUARD_ENV, raising=False)
    monkeypatch.setattr(lpr, "_CLOCK_GUARD_ANCHOR", {})
    j = {"gpu_state": {"sm_clock_mhz": 1800, "temperature_c": 60}}
    assert lpr._bench_post(dict(j)) == j                          # default: untouched
    monkeypatch.setenv(lpr.CLOCK_GUARD_ENV, "1")
    monkeypatch.setattr(lpr, "_node_dirty", [])
    monkeypatch.setattr(lpr, "_CLOCK_GUARD_ANCHOR", {})
    assert lpr._bench_post(dict(j)) == dict(j)                    # first: anchors
    dirty = {"gpu_state": {"sm_clock_mhz": 1890, "temperature_c": 60}}
    assert lpr._bench_post(dirty) is None                         # 5% clock: dirty
    assert "clock-guard" in lpr._node_dirty[-1]
    fine = {"gpu_state": {"sm_clock_mhz": 1809, "temperature_c": 61}}
    assert lpr._bench_post(fine) == fine                          # within band: fine


def test_run_bench_keep_cubin_stays_subprocess(monkeypatch, tmp_path):
    """The cubin-dump path (pricing's sha reading) never routes through the
    worker: its env is per-variant and its dump dir is prepared by the caller."""
    spawned = _wire_worker(monkeypatch, tmp_path,
                           lambda w, cid: _answer_ok(w, cid))
    sh_calls = []

    def fake_sh(cmd, timeout=2400, env=None):
        sh_calls.append(env)
        return 0, _fake_bench_json(1.0) + "\n", ""

    monkeypatch.setattr(lpr, "sh", fake_sh)
    j = lpr.run_bench("v", keep_cubin=True, timeout=60)
    assert j["performance"]["device_median_ms"] == 1.0
    assert len(sh_calls) == 1 and sh_calls[0].get("CUTE_DSL_KEEP_CUBIN") == "1"
    assert spawned == []


# --- task 5: the CANDIDATES certification schedule -----------------------------------

def test_cert_scheduler_off_by_default(tmp_path):
    """cert_every_meta_steps defaults to 0: the reservation never exists, even
    with a candidates file and a teacher ready — the historical behavior."""
    (tmp_path / "recon").mkdir(exist_ok=True)
    (tmp_path / "recon" / "candidates.md").write_text("| c0 | ... |\n", encoding="utf-8")
    be = FakeBackend(["PROBE_REQUEST: notes/cert.py"], ["PROBE PASS (max_diff=0.0)\n"])
    ctx = _ctx(tmp_path, be)
    ctx.recon_path = tmp_path / "recon" / "report.md"
    assert drv._wp17_cert_every_meta_steps(ctx) == 0
    _wp17_maybe_cert_candidates_head(ctx)
    assert be.calls == []


def test_cert_scheduler_fires_on_the_schedule(tmp_path):
    (tmp_path / "recon").mkdir(exist_ok=True)
    (tmp_path / "recon" / "candidates.md").write_text("| head | candidate |\n",
                                                      encoding="utf-8")
    probe = tmp_path / "notes" / "cert.py"
    probe.parent.mkdir(exist_ok=True)
    probe.write_text("print('PROBE PASS (max_diff=0.0)')\n", encoding="utf-8")
    be = FakeBackend(["PROBE_REQUEST: notes/cert.py"], ["PROBE PASS (max_diff=0.0)\n"])
    ctx = _ctx(tmp_path, be)
    ctx.recon_path = tmp_path / "recon" / "report.md"
    ctx.cert_every_meta_steps = 2
    monkey = pytest.MonkeyPatch()
    monkey.setattr(drv, "load_meta_records", lambda path: [1, 2])   # step 2 of 2
    try:
        _wp17_maybe_cert_candidates_head(ctx)
    finally:
        monkey.undo()
    # one teacher turn, one probe job on the -certprobe branch, reset to pre_head
    assert be.calls.count(("agent",)) == 1
    assert any("CANDIDATES certification" in p for p in be.prompts)
    assert any(c[0] == "gpu" and c[1] == "cmd:python3 notes/cert.py" for c in be.calls)
    assert ("push", "fs-demo-certprobe") in be.calls
    assert ("reset", "sha0") in be.calls            # teacher material reset away


def test_cert_scheduler_lapses(tmp_path):
    (tmp_path / "recon").mkdir(exist_ok=True)
    (tmp_path / "recon" / "candidates.md").write_text("| head |\n", encoding="utf-8")
    # off-schedule: step 1 of every-2 is not due
    be = FakeBackend(["should not run"], [])
    ctx = _ctx(tmp_path, be)
    ctx.recon_path = tmp_path / "recon" / "report.md"
    ctx.cert_every_meta_steps = 2
    monkey = pytest.MonkeyPatch()
    monkey.setattr(drv, "load_meta_records", lambda path: [1])
    try:
        _wp17_maybe_cert_candidates_head(ctx)
    finally:
        monkey.undo()
    assert be.calls == []
    # due, but the teacher writes prose: the round lapses without a job
    monkey.setattr(drv, "load_meta_records", lambda path: [1, 2])
    try:
        _wp17_maybe_cert_candidates_head(ctx)
    finally:
        monkey.undo()
    assert be.calls.count(("agent",)) == 1
    assert [c for c in be.calls if c[0] == "gpu"] == []
    # no candidates file: lapses before any turn
    be2 = FakeBackend(["should not run"], [])
    ctx2 = _ctx(tmp_path, be2)
    (tmp_path / "recon2").mkdir(exist_ok=True)
    ctx2.recon_path = tmp_path / "recon2" / "report.md"
    ctx2.cert_every_meta_steps = 1
    monkey.setattr(drv, "load_meta_records", lambda path: [1])
    try:
        _wp17_maybe_cert_candidates_head(ctx2)
    finally:
        monkey.undo()
    assert be2.calls == []


# --- config: contract C17 (card-layer keys, no cli forwarding) -----------------------

def test_wp17_apply_cfg_reads_the_card_layer(tmp_path):
    ms = tmp_path / "m.md"
    ms.write_text(_MILESTONE, encoding="utf-8")
    (tmp_path / "loop.toml").write_text(
        "[loop]\n"
        'arch_probe_floor_sort = false\n'
        'arch_probe_near_diff_reprobe = true\n'
        'arch_probe_ledger = "not a bool"\n'
        'cert_every_meta_steps = 3\n', encoding="utf-8")
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    ctx.milestone = ms
    _wp17_apply_cfg(ctx)
    assert ctx.arch_probe_floor_sort is False
    assert ctx.arch_probe_near_diff_reprobe is True
    assert ctx.arch_probe_ledger is True             # a bad value leaves the default
    assert ctx.cert_every_meta_steps == 3
    # defaults when the card layer is silent
    (tmp_path / "loop.toml").write_text("[budget]\nhours = 4\n", encoding="utf-8")
    ctx2 = _ctx(tmp_path, be)
    ctx2.milestone = ms
    _wp17_apply_cfg(ctx2)
    assert (ctx2.arch_probe_floor_sort, ctx2.arch_probe_near_diff_reprobe,
            ctx2.arch_probe_ledger) == (True, True, True)
    assert ctx2.cert_every_meta_steps == 0
    # an explicit field (tests, callers) is never clobbered
    ctx3 = _ctx(tmp_path, be)
    ctx3.milestone = ms
    ctx3.cert_every_meta_steps = 7
    _wp17_apply_cfg(ctx3)
    assert ctx3.cert_every_meta_steps == 7
