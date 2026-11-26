"""rsi_kernel_tune.opkb.record: extraction from a run directory, field by field.

The fixture ``tests/fixtures/opkb_run`` is a synthesized run with the real layout and the
real field names of the artifacts a run writes (``run.json``, the task snapshot, the card's
``problem.json``, ``loop.toml``, ``student/status.jsonl``, ``meta_status.jsonl``,
``loop_summary.json``, ``best_confirmation.json``, the recon deliverables). Every expectation
below is the fixture's own literal value, not what the extractor computes, so the two
implementations disagree visibly.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import rsi_kernel_tune.ext.lessons as lessons_cli
from rsi_kernel_tune.opkb import record

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "opkb_run"


def _sol(t_k: float, t_b: float, t_sol: float) -> float:
    """The board's score formula, written out here so the extraction has an independent check."""
    return 1.0 / (1.0 + (t_k - t_sol) / (t_b - t_sol))


@pytest.fixture
def run(tmp_path) -> Path:
    """A writable copy of the fixture run (tests may delete or edit files)."""
    dest = tmp_path / "opkb_run"
    shutil.copytree(FIXTURE, dest)
    return dest


# --- the whole record, field by field --------------------------------------------------------


def test_extract_reproduces_the_fixture_field_by_field():
    rec = record.extract_run(FIXTURE).record
    assert rec["schema"] == 1
    assert rec["id"] == "069_rms_norm-251124-101500-ab12"
    assert rec["created"] == "2026-11-24T14:30:00+0800"          # run.json's "updated"
    assert rec["family"] == "rmsnorm"
    assert rec["problem"] == {
        "suite": "sol-execbench", "name": "069_rms_norm", "definition": "069_rms_norm",
        "track": "l1", "arch": "sm100", "language": "triton", "workloads": 2}
    assert rec["reference"] == {"kind": "eager"}                 # reference_impl kind=definition
    assert rec["outcome"]["window"] == "device"                  # loop.toml gate_window
    assert rec["outcome"]["metric_kind"] == "sol"
    expected_s = (_sol(0.0312, 0.0338, 0.011022) + _sol(0.0711, 0.0676, 0.022044)) / 2
    assert rec["outcome"]["value"] == pytest.approx(expected_s, rel=1e-12)
    assert rec["outcome"]["stop"] == "target"                    # loop_summary.json
    assert rec["outcome"]["hours"] == 4.25                        # 10:15 -> 14:30
    assert rec["outcome"]["student_iterations"] == 4             # four status.jsonl rows
    assert rec["source"] == {"run": "opkb_run", "best_sha": "4c1a7d9e2b3f"}
    assert rec["certified_kb"] == ["hw_kb/triton/sm100/rowwise_reduction.md"]
    assert rec["usage"] == {"teacher_tokens": 120000, "student_tokens": 340000,
                            "teacher_turns": 6, "student_turns": 37}


def test_kept_and_failed_rows_are_the_trajectory_rows_verbatim():
    rec = record.extract_run(FIXTURE).record
    assert rec["kept"] == [
        {"iter": 1, "change": "one fused RMSNorm kernel, one program per row",
         "verbatim": True, "before_ms": 0.2243, "after_ms": 0.0612},
        {"iter": 3, "change": "two rows per program with a strided weight load",
         "verbatim": True, "before_ms": 0.0612, "after_ms": 0.0473},
    ]
    assert rec["failed"] == [
        {"iter": 2, "change": "128-bit vector loads over the hidden dimension",
         "error": "output: max_diff=0.00390625 tol=0.001953125", "verbatim": True,
         "kind": "correctness"},
        {"iter": 4, "change": "persistent grid sized to the SM count",
         "error": "regression: 0.0499 ms vs 0.0473 ms best (device window)",
         "verbatim": True, "kind": "revert"},
    ]
    assert rec["meta"] == [{"round": 1, "outcome": "KEEP_PATCH"},
                           {"round": 2, "outcome": "REVERT_PATCH"}]


def test_the_after_reading_is_the_runs_gate_window(run):
    # the KEEP's device reading is 0.0612 / 0.0473; its e2e reading is 0.0651 / 0.0501.
    (run / "operators/069_rms_norm/loop.toml").write_text('[loop]\ngate_window = "e2e"\n')
    rec = record.extract_run(run).record
    assert rec["outcome"]["window"] == "e2e"
    assert [row["after_ms"] for row in rec["kept"]] == [0.0651, 0.0501]


def test_the_gate_window_falls_back_to_the_task_and_then_to_the_loop_default(run):
    (run / "operators/069_rms_norm/loop.toml").unlink()
    assert record.extract_run(run).record["outcome"]["window"] == "device"
    toml = (run / "task/task.toml").read_text(encoding="utf-8")
    (run / "task/task.toml").write_text(toml + '\n[target]\ntiming = "e2e"\n')
    rec = record.extract_run(run).record
    assert rec["outcome"]["window"] == "e2e"
    assert rec["kept"][0]["after_ms"] == 0.0651


# --- what the artifacts do not say is omitted, never defaulted --------------------------------


def test_absent_sources_drop_their_fields_and_say_which_file_was_missing(run):
    for name in ("operators/069_rms_norm/best_confirmation.json",
                 "operators/069_rms_norm/loop_summary.json", "agent_usage.jsonl"):
        (run / name).unlink()
    run_json = json.loads((run / "run.json").read_text(encoding="utf-8"))
    run_json.pop("stop")
    (run / "run.json").write_text(json.dumps(run_json), encoding="utf-8")

    ext = record.extract_run(run)
    rec = ext.record
    assert "metric_kind" not in rec["outcome"] and "value" not in rec["outcome"]
    assert "stop" not in rec["outcome"]
    assert "best_sha" not in rec["source"] and rec["source"] == {"run": "opkb_run"}
    assert "usage" not in rec
    assert any("best_confirmation.json" in n for n in ext.notes)
    assert any("loop_summary.json" in n for n in ext.notes)


def test_no_trajectory_file_means_no_iteration_fields(run):
    (run / "operators/069_rms_norm/student/status.jsonl").unlink()
    (run / "operators/069_rms_norm/meta_status.jsonl").unlink()
    ext = record.extract_run(run)
    assert "kept" not in ext.record and "failed" not in ext.record
    assert "meta" not in ext.record
    assert "student_iterations" not in ext.record["outcome"]
    assert any("status.jsonl" in n for n in ext.notes)


def test_an_unknown_usage_ledger_is_ignored_instead_of_guessed(run):
    (run / "agent_usage.jsonl").write_text('{"provider": "claude", "note": "no totals"}\n',
                                          encoding="utf-8")
    assert "usage" not in record.extract_run(run).record


def test_a_card_without_a_sol_board_scores_by_the_same_job_speedup(run):
    card = json.loads((run / "operators/069_rms_norm/eval/leaderboard/problem.json")
                      .read_text(encoding="utf-8"))
    card.pop("group")
    card.pop("sol_board", None)
    (run / "operators/069_rms_norm/eval/leaderboard/problem.json").write_text(
        json.dumps(card), encoding="utf-8")
    confirm = json.loads((run / "operators/069_rms_norm/best_confirmation.json")
                         .read_text(encoding="utf-8"))
    confirm["median_ratio"] = 1.31
    (run / "operators/069_rms_norm/best_confirmation.json").write_text(
        json.dumps(confirm), encoding="utf-8")

    rec = record.extract_run(run).record
    assert rec["outcome"]["metric_kind"] == "speedup"
    assert rec["outcome"]["value"] == 1.31
    assert rec["problem"]["workloads"] == 1


def test_a_plain_task_run_without_a_card_has_no_suite_and_no_definition(tmp_path):
    """A non-leaderboard run (the real harness's own shape: no eval/leaderboard/): the
    card-derived fields are omitted and the reference kind comes from [reference]."""
    run = tmp_path / "rmsnorm_251124-101500-ab12"
    (run / "task").mkdir(parents=True)
    (run / "task/task.toml").write_text(
        '[task]\nlanguage = "cutedsl"\narch = "sm100"\n\n'
        '[reference]\nrun = "reference/run.py:run"\nsource = "reference"\n',
        encoding="utf-8")
    op = run / "operators" / "rmsnorm"
    (op / "student").mkdir(parents=True)
    (op / "loop.toml").write_text('[loop]\ngate_window = "e2e"\n', encoding="utf-8")
    (op / "student/status.jsonl").write_text(json.dumps({
        "iter": 1, "decision_id": "D1", "change": "fuse the reduction",
        "correctness": True, "verdict": "KEEP", "bwd_median_ms": 0.9}) + "\n",
        encoding="utf-8")
    (op / "loop_summary.json").write_text('{"stop": "meta_budget", "meta_steps": 1}',
                                          encoding="utf-8")
    (op / "best_confirmation.json").write_text(json.dumps({
        "sha": "abcdef123456", "window": "e2e", "median_ms": 0.01, "median_ratio": 1.3}),
        encoding="utf-8")
    (run / "run.json").write_text(json.dumps({
        "task": "rmsnorm", "id": "251124-101500-ab12",
        "created": "2026-11-24T10:00:00+0800", "updated": "2026-11-24T12:00:00+0800"}),
        encoding="utf-8")

    rec = record.extract_run(run).record
    assert rec["family"] == "rmsnorm"
    assert rec["problem"] == {"name": "rmsnorm", "arch": "sm100", "language": "cutedsl"}
    assert rec["reference"] == {"kind": "eager"}
    assert rec["outcome"] == {"window": "e2e", "metric_kind": "speedup", "value": 1.3,
                             "stop": "meta_budget", "hours": 2.0, "student_iterations": 1}
    assert rec["kept"] == [{"iter": 1, "change": "fuse the reduction", "verbatim": True,
                            "after_ms": 0.9}]
    assert "certified_kb" not in rec and "usage" not in rec and "recon" not in rec


def test_flat_usage_columns_are_accepted_too(run):
    (run / "agent_usage.jsonl").write_text(
        '{"teacher_tokens": 5, "student_tokens": 7, "teacher_turns": 1, "student_turns": 2}\n',
        encoding="utf-8")
    assert record.extract_run(run).record["usage"] == {
        "teacher_tokens": 5, "student_tokens": 7, "teacher_turns": 1, "student_turns": 2}


# --- free text: verbatim, truncated, marked ---------------------------------------------------


def test_free_text_is_stored_verbatim_and_cut_with_a_marker(run):
    long_change = "x" * (record.MAX_CHANGE + 500)
    long_error = "e" * (record.MAX_ERROR + 500)
    status = run / "operators/069_rms_norm/student/status.jsonl"
    rows = [json.loads(line) for line in status.read_text(encoding="utf-8").splitlines()]
    rows[1]["change"], rows[1]["error"] = long_change, long_error
    status.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    failed = record.extract_run(run).record["failed"][0]
    assert failed["verbatim"] is True
    assert failed["change"] == long_change[: record.MAX_CHANGE - len(record.TRUNCATION_MARK)] \
        + record.TRUNCATION_MARK
    assert len(failed["change"]) == record.MAX_CHANGE
    assert failed["error"].startswith("e" * 100) and failed["error"].endswith("[truncated]")
    assert len(failed["error"]) == record.MAX_ERROR


def test_short_free_text_is_not_touched():
    failed = record.extract_run(FIXTURE).record["failed"][0]
    assert failed["change"] == "128-bit vector loads over the hidden dimension"
    assert failed["error"] == "output: max_diff=0.00390625 tol=0.001953125"


# --- the recon's structured entries only ------------------------------------------------------


def test_recon_prose_milestone_and_probe_notes_do_not_enter_the_record():
    rec = record.extract_run(FIXTURE).record
    text = record.serialize(rec)
    assert "DECOY_TOKEN_REFERENCE_PROSE_9f3a" not in text
    assert "DECOY_TOKEN_MILESTONE_7c1d" not in text
    assert "DECOY_TOKEN_PROBE_2b8e" not in text
    # ... while the fixed-format recon entries do:
    kinds = [v["kind"] for v in rec["recon"]["verdicts"]]
    assert kinds == ["LAUNCH-OK", "SASS-OK", "SCHED-OK", "LIVE-OK"]
    launch = rec["recon"]["verdicts"][0]
    assert launch["fields"]["grid"] == "(2,1,66)" and launch["fields"]["ctas"] == "132"
    assert launch["line"].startswith("LAUNCH-OK kernel=rmsnorm_kernel")
    assert rec["recon"]["candidates"] == [
        {"decision": "tile_M x tile_N", "value": "128x256", "verify": "ncu-derived",
         "verbatim": True},
        {"decision": "stage count", "value": "3", "verify": "sass-derived", "verbatim": True},
        {"decision": "register split", "value": "40/232", "verify": "ncu-direct",
         "verbatim": True},
        {"decision": "tile_M x tile_N", "value": "256x128", "verify": "inferred",
         "verbatim": True},
        {"decision": "stage count", "value": "4", "verify": "none", "verbatim": True},
    ]
    assert rec["recon"]["verbatim"] is True


def test_a_run_without_recon_artifacts_has_no_recon_block(run):
    shutil.rmtree(run / "operators/_recon")
    assert "recon" not in record.extract_run(run).record


def test_a_recon_report_with_prose_only_yields_no_entries(run):
    (run / "operators/_recon/candidates.md").unlink()
    (run / "operators/_recon/report.md").write_text(
        "# Report\n\nSomething was probably fast, but nothing was measured.\n",
        encoding="utf-8")
    assert "recon" not in record.extract_run(run).record


def test_candidate_rows_need_a_table_row_not_a_sentence():
    text = ("The tile is probably 128x256 (not measured).\n\n"
            "| decision | value |\n|---|---|\n| stage count | 3 |\n")
    assert record.candidate_rows(text) == [
        {"decision": "stage count", "value": "3", "verbatim": True}]


def test_verdict_lines_are_parsed_with_their_fields():
    text = "noise\n  SCHED-OK loop=1 mma_runs=4 handoff=[A B]\nnot a verdict: OK\n"
    rows = record.verdict_lines(text)
    assert [r["kind"] for r in rows] == ["SCHED-OK"]
    assert rows[0]["fields"]["mma_runs"] == "4"


# --- refusals ----------------------------------------------------------------------------------


def test_a_directory_that_is_not_a_run_is_refused(tmp_path):
    with pytest.raises(record.RecordError, match="not a run directory"):
        record.extract_run(tmp_path)


def test_a_run_without_a_task_name_is_refused(tmp_path):
    (tmp_path / "run.json").write_text("{}", encoding="utf-8")
    with pytest.raises(record.RecordError, match="names no task"):
        record.extract_run(tmp_path)


def test_a_run_without_its_operator_card_is_refused(tmp_path):
    (tmp_path / "run.json").write_text('{"task": "069_rms_norm"}', encoding="utf-8")
    with pytest.raises(record.RecordError, match="no operator card"):
        record.extract_run(tmp_path)


def test_a_corrupt_trajectory_is_an_error_not_a_silent_empty_list(run):
    (run / "operators/069_rms_norm/student/status.jsonl").write_text("{not json}\n",
                                                                   encoding="utf-8")
    with pytest.raises(record.RecordError, match="status.jsonl"):
        record.extract_run(run)


# --- determinism and the CLI ------------------------------------------------------------------


def test_serialize_is_deterministic():
    a = record.serialize(record.extract_run(FIXTURE).record)
    b = record.serialize(record.extract_run(FIXTURE).record)
    assert a == b and a.endswith("\n")
    assert json.loads(a)["id"] == "069_rms_norm-251124-101500-ab12"


def test_forge_lessons_help_and_every_subcommands_help(capsys):
    for argv in (["--help"], ["harvest", "--help"], ["query", "--help"],
                 ["render", "--help"], ["lint", "--help"], ["stats", "--help"]):
        with pytest.raises(SystemExit) as e:
            lessons_cli._parser().parse_args(argv)
        assert e.value.code == 0
        assert "forge lessons" in capsys.readouterr().out


def test_the_forge_extension_entry_point_reaches_the_command(capsys):
    from rsi_kernel_tune import __main__ as forge_main

    with pytest.raises(SystemExit) as e:
        forge_main.main(["lessons", "--help"])            # `forge lessons --help`
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "harvest" in out and "render" in out
    assert forge_main.main(["--help"]) == 0
    assert "forge lessons ..." in capsys.readouterr().out


def test_harvest_prints_a_summary_and_the_record_is_on_disk(tmp_path, capsys):
    root = tmp_path / "op_kb"
    assert lessons_cli.main(["harvest", str(FIXTURE), "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "[forge lessons] wrote" in out
    assert "sol=0.505883" in out and "2 kept / 2 failed" in out and "stop target" in out
    filed = root / "rmsnorm" / "069_rms_norm-251124-101500-ab12.json"
    assert filed.is_file()
    assert json.loads(filed.read_text(encoding="utf-8"))["family"] == "rmsnorm"
    assert (root / "index.json").is_file()
    assert (root / "README.md").is_file()


def test_harvest_is_idempotent(tmp_path):
    root = tmp_path / "op_kb"
    lessons_cli.main(["harvest", str(FIXTURE), "--root", str(root)])
    first = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
    bytes_first = (root / "rmsnorm" / "069_rms_norm-251124-101500-ab12.json").read_bytes()
    assert lessons_cli.main(["harvest", str(FIXTURE), "--root", str(root)]) == 0
    second = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
    assert first == second
    assert (root / "rmsnorm" / "069_rms_norm-251124-101500-ab12.json").read_bytes() == bytes_first


def test_harvest_refuses_a_directory_that_is_not_a_run(tmp_path, capsys):
    assert lessons_cli.main(["harvest", str(tmp_path), "--root", str(tmp_path / "kb")]) == 2
    assert "not a run directory" in capsys.readouterr().err
