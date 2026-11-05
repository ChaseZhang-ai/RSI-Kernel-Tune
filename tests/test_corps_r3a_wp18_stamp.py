"""Regression tests for the R3A audit wave: WP18 PRICED_NULL stamping.

BUG-CH-C-01 (P1): the per-lever stamp loop stripped every PRICED_NULL marker
from the whole milestone before re-stamping only the current lever, so with two
cited null levers only the last kept its stamp; an uncited-but-decisioned null
lever erased the others' stamps while adding none of its own; and the citation
marker's replacement dropped the standing [RE-PRICED … SUSPENDED] verdict, so a
suspended rung could be landed again.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from test_loop_driver import FakeBackend, _ctx

from rsi_kernel_tune.loop import driver

_SUSPENDED = ("[RE-PRICED @abc1234567 job 9: 1.10 -> 0.80 ms — FADED; SUSPENDED by "
              "driver — the rung citing this price may not be landed until the "
              "teacher revises it]")


def _ctx_ms(tmp_path, text):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.milestone.write_text(text, encoding="utf-8")
    return ctx


def test_two_cited_null_levers_both_keep_their_stamp(tmp_path):
    """Two cited null levers: each citation carries its own lever's stamp (the
    old code let the last lever's strip erase every earlier stamp)."""
    ctx = _ctx_ms(tmp_path,
                  "### D6\nAction: try `lever:regsplit` [MEASURED, job 12]\n"
                  "### D7\nAction: try `lever:swizzle` [MEASURED, job 13]\n")
    table = [
        {"kind": "lever", "name": "regsplit", "cls": "null", "add_pct": 0.2,
         "remove_pct": -0.1, "decisions": ["D6"]},
        {"kind": "lever", "name": "swizzle", "cls": "noop", "add_pct": 0.05,
         "remove_pct": -0.04, "decisions": ["D7"]}]
    driver._wp18_stamp_priced_null(ctx, table)
    ms = ctx.milestone.read_text(encoding="utf-8")
    d6 = ms.split("### D6")[1].split("### D7")[0]
    d7 = ms.split("### D7")[1]
    assert "PRICED_NULL: measured null" in d6
    assert "PRICED_NULL: measured noop" in d7
    assert ms.count("PRICED_NULL: measured") == 2


def test_uncited_null_lever_does_not_erase_other_stamps(tmp_path):
    """A null lever named in the table but not cited by the milestone must not
    strip the stamps the cited levers just earned (old code ran its strip pass
    anyway and re-added nothing)."""
    ctx = _ctx_ms(tmp_path,
                  "### D6\nAction: try `lever:regsplit` [MEASURED, job 12]\n")
    table = [
        {"kind": "lever", "name": "regsplit", "cls": "null", "add_pct": 0.2,
         "remove_pct": -0.1, "decisions": ["D6"]},
        {"kind": "lever", "name": "uncited", "cls": "null", "add_pct": 0.1,
         "remove_pct": -0.05, "decisions": ["D6"]}]
    driver._wp18_stamp_priced_null(ctx, table)
    ms = ctx.milestone.read_text(encoding="utf-8")
    assert "PRICED_NULL: measured null (+0.20%)" in ms
    assert ms.count("PRICED_NULL") == 1


def test_stamp_does_not_eat_a_standing_suspended_verdict(tmp_path):
    """A citation already carrying [RE-PRICED … SUSPENDED] keeps the verdict
    when the null stamp lands next to it (SUSPENDED forbids landing the rung;
    PRICED_NULL only annotates the price)."""
    ctx = _ctx_ms(tmp_path,
                  f"### D6\nAction: try `lever:regsplit` [MEASURED, job 12] {_SUSPENDED}\n")
    table = [{"kind": "lever", "name": "regsplit", "cls": "null", "add_pct": 0.2,
              "remove_pct": -0.1, "decisions": ["D6"]}]
    driver._wp18_stamp_priced_null(ctx, table)
    ms = ctx.milestone.read_text(encoding="utf-8")
    assert "PRICED_NULL: measured null" in ms
    assert "SUSPENDED by driver" in ms and "RE-PRICED @abc1234567" in ms
    # the standing verdict survives verbatim
    assert _SUSPENDED in ms


def test_restamp_one_lever_keeps_the_other_levers_stamp(tmp_path):
    """A later table that re-stamps lever A must not touch lever B's standing
    stamp, and re-stamping A replaces only A's own marker (idempotent)."""
    ctx = _ctx_ms(tmp_path,
                  "### D6\nAction: try `lever:regsplit` [MEASURED, job 12]\n"
                  "### D7\nAction: try `lever:swizzle` [MEASURED, job 13]\n")
    table = [
        {"kind": "lever", "name": "regsplit", "cls": "null", "add_pct": 0.2,
         "remove_pct": -0.1, "decisions": ["D6"]},
        {"kind": "lever", "name": "swizzle", "cls": "noop", "add_pct": 0.05,
         "remove_pct": -0.04, "decisions": ["D7"]}]
    driver._wp18_stamp_priced_null(ctx, table)
    once = ctx.milestone.read_text(encoding="utf-8")
    driver._wp18_stamp_priced_null(ctx, table)   # same table again: idempotent
    assert ctx.milestone.read_text(encoding="utf-8") == once
    # a newer table prices only regsplit: swizzle's stamp stands, regsplit's is
    # refreshed (replaced, not stacked)
    driver._wp18_stamp_priced_null(ctx, [
        {"kind": "lever", "name": "regsplit", "cls": "null", "add_pct": 0.3,
         "remove_pct": -0.2, "decisions": ["D6"]}])
    ms = ctx.milestone.read_text(encoding="utf-8")
    assert "PRICED_NULL: measured null (+0.30%)" in ms
    assert "PRICED_NULL: measured noop" in ms
    assert ms.count("PRICED_NULL: measured") == 2


def test_stamp_log_counts_the_levers_actually_stamped(tmp_path, monkeypatch):
    """The closing log names the levers the pass actually stamped, not the whole
    null set (the old message over-reported when levers were skipped)."""
    logs = []
    monkeypatch.setattr(driver, "_log", lambda m: logs.append(m))
    ctx = _ctx_ms(tmp_path,
                  "### D6\nAction: try `lever:regsplit` [MEASURED, job 12]\n")
    table = [
        {"kind": "lever", "name": "regsplit", "cls": "null", "add_pct": 0.2,
         "remove_pct": -0.1, "decisions": ["D6"]},
        {"kind": "lever", "name": "uncited", "cls": "null", "add_pct": 0.1,
         "remove_pct": -0.05, "decisions": ["D6"]}]
    driver._wp18_stamp_priced_null(ctx, table)
    stamp_logs = [m for m in logs if "PRICED_NULL stamped" in m]
    assert stamp_logs and "regsplit" in stamp_logs[-1]
    assert "uncited" not in stamp_logs[-1]
