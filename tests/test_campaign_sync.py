"""`campaign sync`: the append-only ledger, the saved-page import, and the ratio.

`ratios.json` is the only place the B300/B200 question is answered with data, so these
tests pin its arithmetic by hand (ratio = board S / dev S, family and confidence means,
and `correction()` falling back to the overall mean only when the family has no pair).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import campaign_helpers as H
from forgekernel.campaign import solscore, sync, tools as tools_mod
from forgekernel.ext import campaign as cli


def _ledger(tmp_path) -> sync.Ledger:
    return sync.Ledger(tmp_path / "work" / sync.LEDGER_FILE)


def _reading(mean_s, *, correct=True, cold=5.0, conf=None) -> tools_mod.BenchReading:
    return tools_mod.BenchReading(
        correct=correct, mean_s=mean_s, cold_start_s=cold,
        workloads=[tools_mod.WorkloadReading(name="wl0", t_ms=0.01),
                   tools_mod.WorkloadReading(name="wl1", t_ms=0.02)],
        confidence=conf or ["dev-only"], candidate="c")


def test_ledger_is_append_only_and_validates_the_event(tmp_path):
    led = _ledger(tmp_path)
    led.append({"event": "note", "problem": "p", "note": "one"})
    led.append({"event": "board", "problem": "p", "mean_s": 0.5, "source": "board"})
    rows = led.entries()
    assert [r["event"] for r in rows] == ["note", "board"]
    assert all("ts" in r for r in rows)
    with pytest.raises(ValueError, match="unknown ledger event"):
        led.append({"event": "made-up"})
    # a corrupted line is reported, never fatal
    with open(led.path, "a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    rows = led.entries()
    assert rows[-1]["event"] == "error"


def test_record_reading_keeps_the_per_workload_detail(tmp_path):
    led = _ledger(tmp_path)
    led_ = sync.record_reading(led, problem="069_rms_norm", track="l1", kernel_id=69,
                               reading=_reading(0.7), candidate="run:it2", run="r1",
                               sha="abc", event="starter")
    assert led_["mean_s"] == 0.7 and led_["sha"] == "abc"
    assert led_["workloads"]["wl0"]["t_ms"] == 0.01
    assert led_["geomean_ms"] == pytest.approx(solscore.geomean([0.01, 0.02]))
    with pytest.raises(ValueError, match="source must be"):
        sync.record_reading(led, problem="p", track="l1", kernel_id=1,
                            reading=_reading(0.5), source="sometimes")


def test_record_results_is_the_entry_the_board_importer_calls(tmp_path):
    path = tmp_path / "ledger.jsonl"
    rows = [{"kernel_name": "069_rms_norm", "sol_score": 0.91, "track": "l1"},
            {"problem": "030_x", "mean_s": 0.4, "sha": "deadbeef"},
            {"nothing": "useful"}]
    assert sync.record_results(rows, path) == 2
    entries = sync.Ledger(path).entries()
    assert [e["problem"] for e in entries] == ["069_rms_norm", "030_x"]
    assert entries[0]["source"] == "board" and entries[0]["mean_s"] == 0.91
    assert entries[1]["sha"] == "deadbeef"


def test_a_saved_per_kernel_page_gives_one_team_s_rows():
    page = H.SNAPSHOT / "perkernel" / "69.json"
    everyone = sync.rows_from_saved(page)
    assert len(everyone) > 5
    assert {r["problem"] for r in everyone} == {H.L1_PROBLEM}
    mine = sync.rows_from_saved(page, username="yue shui")
    assert len(mine) == 1 and mine[0]["mean_s"] == pytest.approx(0.997818)
    # a name that is not on the page yields nothing (never a guess)
    assert sync.rows_from_saved(page, username="nobody") == []


def test_a_saved_result_page_is_recognised(tmp_path):
    p = tmp_path / "result.json"
    p.write_text(json.dumps({"kernel_name": H.L1_PROBLEM, "sol_score": 0.8,
                             "submission_id": 7, "track": "l1"}))
    rows = sync.rows_from_saved(p)
    assert len(rows) == 1 and rows[0]["mean_s"] == 0.8
    assert rows[0]["source"] == "board"


def test_import_saved_writes_the_ledger_and_warns_when_nothing_matched(tmp_path):
    led = tmp_path / "work" / sync.LEDGER_FILE
    n, warns = sync.import_saved(H.SNAPSHOT / "perkernel" / "69.json", led,
                                 username="yue shui")
    assert n == 1 and not warns
    assert sync.Ledger(led).entries()[0]["candidate"] == "yue shui"

    empty = tmp_path / "empty"
    empty.mkdir()
    n, warns = sync.import_saved(empty, led)
    assert n == 0 and warns and "no per-kernel page" in warns[0]


def test_ratio_arithmetic_by_hand(tmp_path):
    led = _ledger(tmp_path)
    for name, dev, board, conf in (("069_rms_norm", 0.80, 0.72, ["dev-only"]),
                                   ("030_attention_output_projection_with_residual",
                                    0.50, 0.40, ["dev-only", "sfu-bound"])):
        sync.record_reading(led, problem=name, track="l1", kernel_id=1,
                            reading=_reading(dev, conf=conf))
        led.append({"event": "board", "problem": name, "track": "l1", "source": "board",
                    "mean_s": board, "correct": True})
    doc = sync.ratios(led.entries())
    assert doc["n_pairs"] == 2
    assert doc["problems"]["069_rms_norm"]["ratio"] == pytest.approx(0.9)
    assert doc["problems"][
        "030_attention_output_projection_with_residual"]["ratio"] == pytest.approx(0.8)
    assert doc["overall"]["mean_ratio"] == pytest.approx(0.85)
    fam = sync.targets.family_of("069_rms_norm")
    assert doc["by_family"][fam]["mean_ratio"] == pytest.approx(0.9)
    assert doc["by_confidence"]["dev-only"]["n"] == 2
    assert doc["by_confidence"]["sfu-bound"]["mean_ratio"] == pytest.approx(0.8)
    # a family with no pair falls back to the overall mean; nothing at all is None
    assert sync.correction(fam, doc) == pytest.approx(0.9)
    assert sync.correction("no-such-family", doc) == pytest.approx(0.85)
    assert sync.corrected_estimate(0.5, fam, doc) == pytest.approx(0.45)
    assert sync.corrected_estimate(None, fam, doc) is None
    assert sync.correction("x", {"by_family": {}, "overall": {"mean_ratio": None}}) is None


def test_board_readings_need_no_correction_and_win_in_ours(tmp_path):
    led = _ledger(tmp_path)
    sync.record_reading(led, problem="p", track="l1", kernel_id=1, reading=_reading(0.9))
    led.append({"event": "board", "problem": "p", "source": "board", "mean_s": 0.7,
                "correct": True})
    led.append({"event": "board", "problem": "p", "source": "board", "mean_s": 0.8,
                "correct": True})
    assert sync.ours_from_ledger(led.entries())["p"] == pytest.approx(0.8)


def test_incorrect_readings_never_enter_the_ratio(tmp_path):
    led = _ledger(tmp_path)
    sync.record_reading(led, problem="p", track="l1", kernel_id=1,
                        reading=_reading(0.9, correct=False))
    led.append({"event": "board", "problem": "p", "source": "board", "mean_s": 0.5,
                "correct": True})
    assert sync.ratios(led.entries())["n_pairs"] == 0


def test_write_ratios_and_summary(tmp_path):
    led = _ledger(tmp_path)
    sync.record_reading(led, problem="069_rms_norm", track="l1", kernel_id=69,
                        reading=_reading(0.8))
    led.append({"event": "board", "problem": "069_rms_norm", "source": "board",
                "mean_s": 0.72, "correct": True, "track": "l1"})
    path = sync.write_ratios(tmp_path / "work", led.entries())
    doc = json.loads(path.read_text())
    assert doc["n_pairs"] == 1 and path.name == "ratios.json"
    text = sync.summary(led.entries())
    assert "board/dev overall 0.9000" in text
    assert "family" in text


def test_the_sync_command_imports_and_writes_ratios(tmp_path, capsys):
    work = tmp_path / "work"
    rc = cli.main(["sync", "--work-dir", str(work), "--from-saved",
                   str(H.SNAPSHOT / "perkernel"), "--username", "yue shui",
                   "--root", str(tmp_path)])
    assert rc == 0
    assert (work / "ratios.json").is_file()
    out = capsys.readouterr().out
    assert "imported 2 board reading(s)" in out


def test_sync_without_a_source_says_what_to_pass(tmp_path):
    with pytest.raises(SystemExit, match="--from-saved"):
        cli.main(["sync", "--work-dir", str(tmp_path / "w"), "--root", str(tmp_path)])


def test_a_track_and_problem_helper():
    snap = sync.targets.load_snapshot(H.SNAPSHOT)
    assert sync.track_of_problem(snap, H.L1_PROBLEM) == "l1"
    assert sync.track_of_problem(snap, "not-a-problem") is None


def test_the_board_readback_hook_writes_one_board_event_per_record(tmp_path):
    """`forge board import` calls this when campaign is installed (board/importer.py):
    the board's read-back records land in the same ledger as the local readings."""
    campaign = tmp_path / "camp"
    campaign.mkdir()
    sync.record_board_readback(campaign, [
        {"problem": H.L1_PROBLEM, "sol_score": 0.83, "kernel_id": 69,
         "submission_id": 4242, "fast_1_count": 16, "is_correct": True},
        {"problem": "", "sol_score": 0.1},                    # no problem: skipped
    ])
    entries = sync.Ledger(campaign / sync.LEDGER_FILE).entries()
    assert [e["event"] for e in entries] == ["board"]
    e = entries[0]
    assert (e["problem"], e["mean_s"], e["kernel_id"], e["submission_id"], e["fast1"]) == \
        (H.L1_PROBLEM, 0.83, 69, 4242, 16)
    assert e["source"] == sync.BOARD and e["correct"] is True
    # the pair a campaign report needs: one dev reading and this board reading
    sync.Ledger(campaign / sync.LEDGER_FILE).append(
        {"event": "bench", "problem": H.L1_PROBLEM, "mean_s": 0.8, "source": sync.DEV,
         "confidence": ["dev-only"]})
    doc = sync.ratios(sync.Ledger(campaign / sync.LEDGER_FILE).entries())
    assert doc["problems"][H.L1_PROBLEM]["ratio"] == pytest.approx(0.83 / 0.8)
    assert doc["n_pairs"] == 1
