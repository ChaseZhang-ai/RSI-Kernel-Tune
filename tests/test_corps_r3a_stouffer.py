"""Regression test for the R3A audit wave: WP15 Stouffer advisor's input rows.

BUG-CH-C-03 (P3): run_meta_step's per-workload combined test was fed the raw
load_records sequence — the only consumer in the function that skipped
_live_records — so readings from a rolled-back segment (REVERT_PATCH voids the
rows by index range) flowed into the "the recent workloads agree on a small
GAIN" teacher signal.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from test_loop_driver import EVAL, NCU, REC, FakeBackend, _ctx  # noqa: E402

from peregrine_kernel.loop import driver

_MILESTONE_PATCH = "\n### D0.9 loosened\nx\n"
_PATCH_REPLY = "patch_class: structural\nchanged_entries: [D0.9]\n"
_DIAG = ("cause_class: R2\ndeficient_locus: D6\npatch_direction: add decision\n"
         "reachability: floor_opt=1.20 ms vs target=1.40 ms -> OPEN")


def test_stouffer_advisor_reads_live_rows_only(tmp_path, monkeypatch):
    be = FakeBackend([REC("D2.3", 2.6), REC("D2.4", 2.6), _DIAG,
                      _stub_patch(tmp_path)],
                     [EVAL(2.6), NCU, EVAL(2.6, False), NCU])
    ctx = _ctx(tmp_path, be)
    driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    driver.run_inner_iteration(ctx, 1, metric_before=2.6)
    # the second segment was rolled back: its rows are void by index range
    ctx.voided_ranges = [(1, 2)]
    captured = []
    monkeypatch.setattr(driver, "combined_workload_test",
                        lambda records, *a, **k: captured.append(
                            [r.iter for r in records]) or None)
    driver.run_meta_step(ctx, best_before=2.66, milestone_before="OLD")
    assert captured, "the advisor ran"
    assert captured[-1] == [0]   # the voided row never reached the signal


def _stub_patch(tmp_path):
    def _run(_prompt):
        (tmp_path / "m.md").write_text(
            (tmp_path / "m.md").read_text(encoding="utf-8") + _MILESTONE_PATCH,
            encoding="utf-8")
        return _PATCH_REPLY
    return _run
