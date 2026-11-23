"""`campaign plan`: the pilot selection, the argv it records, and the language mix."""

from __future__ import annotations

import json

import pytest

import campaign_helpers as H
from forgekernel.campaign import plan as P
from forgekernel.campaign import solscore
from forgekernel.campaign import targets as T
from forgekernel.ext import campaign as cli


def _plan(tmp_path, **kw):
    snap = T.load_snapshot(H.SNAPSHOT)
    kw.setdefault("dataset", str(H.DATASET))
    kw.setdefault("work_dir", tmp_path / "work")
    return P.build_plan(snap, selection=kw.pop("selection", None), root=tmp_path, **kw)


def test_default_pilot_is_the_documented_three(tmp_path):
    plan = _plan(tmp_path)
    assert [p.name for p in plan.problems] == list(P.DEFAULT_PILOT)
    assert {p.track for p in plan.problems} == {"l1", "quant"}
    assert any("documented pilot" in w for w in plan.warnings)


def test_problems_selector_explicit_and_unknown(tmp_path):
    plan = _plan(tmp_path, selection="069_rms_norm")
    assert [p.name for p in plan.problems] == [H.L1_PROBLEM]
    with pytest.raises(ValueError, match="not a problem of this snapshot"):
        _plan(tmp_path, selection="does_not_exist")
    plan = _plan(tmp_path, selection="quant:001_fp8_attention_output_projection")
    assert [p.track for p in plan.problems] == ["quant"]
    with pytest.raises(ValueError, match="ambiguous|not a problem"):
        _plan(tmp_path, selection="l1:001_fp8_attention_output_projection")


def test_pilot_auto_takes_the_priority_order(tmp_path):
    plan = _plan(tmp_path, selection="auto", count=2)
    snap = T.load_snapshot(H.SNAPSHOT)
    rows, _, _ = T.rows(snap, starters=T.starter_situation(snap, tmp_path))
    want = [r.name for r in T.ranked(rows, by="priority")[:2]]
    assert [p.name for p in plan.problems] == want
    assert any("gap x feasibility" in w for w in plan.warnings)


def test_a_bare_count_takes_the_gap_order(tmp_path):
    plan = _plan(tmp_path, selection="1")
    snap = T.load_snapshot(H.SNAPSHOT)
    rows, _, _ = T.rows(snap, starters=T.starter_situation(snap, tmp_path))
    assert [p.name for p in plan.problems] == [T.ranked(rows)[0].name]


def test_plan_roundtrip_and_init_argv(tmp_path):
    plan = _plan(tmp_path, hours=3.5, parallel=4, step=1.05, workloads="0-2")
    path = P.write_plan(plan, tmp_path / "work" / "plan.json")
    doc = P.load_plan(path)
    assert doc["schema"] == P.PLAN_SCHEMA
    assert doc["hours_per_problem"] == 3.5 and doc["parallel"] == 4 and doc["step"] == 1.05
    entry = doc["problems"][0]
    assert entry["init_argv"][0] == entry["op"]
    joined = " ".join(entry["init_argv"])
    assert "--leaderboard sol-execbench" in joined
    assert f"--definition {entry['subset']}/{entry['name']}" in joined
    assert "--no-sol-board" in joined and "--workload 0-2" in joined
    assert entry["workloads"] == "0-2"
    # a second write is a load, not an append
    assert P.load_plan(tmp_path / "work")["problems"] == doc["problems"]
    with pytest.raises(ValueError, match="schema"):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"schema": 99}))
        P.load_plan(bad)
    with pytest.raises(FileNotFoundError):
        P.load_plan(tmp_path / "nothing.json")


def test_init_argv_carries_the_optional_flags(tmp_path):
    entry = _plan(tmp_path, selection="069_rms_norm").problems[0]
    argv = P.init_argv(entry, dataset="/d", sol_board="/r/69.json",
                       init_kernel="/s/kernel.py", kernel_file="kernel.cu")
    joined = " ".join(argv)
    assert "--sol-board /r/69.json" in joined and "--no-sol-board" not in joined
    assert "--init-kernel /s/kernel.py" in joined and "--kernel-file kernel.cu" in joined


def test_languages_reflect_the_current_checkout(monkeypatch):
    langs = {e["route"]: e for e in P.languages_now()}
    assert langs["cutedsl"]["available"] is True
    assert langs["cuda"]["export"] == "needs klib: cuda_to_inline"
    assert langs["torch"]["available"] is True

    import forgekernel.task as task_mod
    monkeypatch.setattr(task_mod, "LANGUAGES", ("cutedsl", "cuda", "triton"))
    langs = {e["route"]: e for e in P.languages_now()}
    assert langs["triton"]["available"] is True
    monkeypatch.setattr(task_mod, "LANGUAGES", ("cutedsl",))
    langs = {e["route"]: e for e in P.languages_now()}
    assert langs["cuda"]["available"] is False
    assert "not in forgekernel.task.LANGUAGES" in langs["cuda"]["export"]


def test_op_name_disambiguates_a_repeated_problem():
    class Snap:
        kernel_ids = [1, 2, 3]

        def name_of(self, kid):
            return {1: "same_name", 2: "same_name", 3: "unique"}[kid]

    rows = [T.TargetRow(kernel_id=1, name="same_name", track="l1", leader=0.5,
                        best_known=0.6, best_known_by="x"),
            T.TargetRow(kernel_id=3, name="unique", track="quant", leader=0.5,
                        best_known=0.6, best_known_by="x")]
    assert P.op_name(Snap(), rows[0]) == "same_name__l1"
    assert P.op_name(Snap(), rows[1]) == "unique"


def test_missing_dataset_is_a_warning_not_a_failure(tmp_path):
    plan = _plan(tmp_path, dataset=str(tmp_path / "no-dataset"))
    assert any("does not exist yet" in w for w in plan.warnings)
    assert len(plan.problems) == 3


def test_dry_run_through_the_cli_writes_nothing(tmp_path, capsys):
    work = tmp_path / "work"
    rc = cli.main(["plan", "--dry-run", "--snapshot", str(H.SNAPSHOT),
                   "--dataset", str(H.DATASET), "--work-dir", str(work),
                   "--root", str(tmp_path)])
    assert rc == 0
    assert not (work / "plan.json").exists()
    out = capsys.readouterr().out
    assert "forge init 069_rms_norm --leaderboard sol-execbench" in out
    assert "--dry-run: nothing written" in out


def test_cli_plan_writes_the_file_and_reports_tools(tmp_path, capsys):
    work = tmp_path / "work"
    rc = cli.main(["plan", "--snapshot", str(H.SNAPSHOT), "--dataset", str(H.DATASET),
                   "--work-dir", str(work), "--root", str(tmp_path)])
    assert rc == 0
    doc = P.load_plan(work / "plan.json")
    assert len(doc["problems"]) == 3
    out = capsys.readouterr().out
    assert "wrote" in out and "tool:" in out


def test_dry_run_commands_mention_evolve_and_export(tmp_path):
    plan = _plan(tmp_path, selection="069_rms_norm")
    without = "\n".join(P.dry_run_commands(plan, evolve_available=False))
    with_evolve = "\n".join(P.dry_run_commands(plan, evolve_available=True))
    assert "forge evolve" not in without and "forge new" in without
    assert "forge evolve tasks/069_rms_norm --lineages 3" in with_evolve
    assert "campaign export" in with_evolve
    assert f"step: {plan.step:g}" in P.format_plan(plan) or f"{plan.step:g}" in P.format_plan(plan)


def test_track_size_and_subset_are_consistent():
    assert solscore.TRACK_SUBSET["l1"] == "L1"
    assert solscore.TRACK_SUBSET["flashinfer-bench"] == "FlashInfer-Bench"
    assert all(solscore.track_size(t) > 0 for t in solscore.TRACKS)
