"""The agent usage ledger (`forgekernel.usage`) and where a run shows it.

The ledger is written by the launcher scripts (scripts/agent_trailer.py) and read here;
there is no budget and no cap anywhere in it -- the numbers are for the run's report and
for `forge status`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from forgekernel import launch, usage
from forgekernel import runs as runs_mod


def _record(**kw) -> dict:
    base = {"ts": "2026-11-24T00:00:00Z", "role": "teacher", "backend": "claude",
            "model_requested": "claude-opus-5-5", "model_used": "claude-opus-5-5-20260101",
            "session_id": "s", "rc": 0, "input_tokens": 100, "output_tokens": 10,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
            "total_tokens": 110, "cost_usd": 0.5, "duration_s": 60.0,
            "refusal_check": "clean"}
    base.update(kw)
    return base


def _ledger(tmp_path, records: list[dict]) -> Path:
    p = tmp_path / usage.LEDGER_NAME
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return p


def test_a_missing_ledger_is_no_records(tmp_path):
    assert usage.read_ledger(tmp_path / "nope.jsonl") == []


def test_read_ledger_skips_what_is_not_a_record(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text("\n".join([json.dumps(_record()),
                            "",
                            "not json at all",
                            json.dumps([1, 2]),            # a list, not a record
                            json.dumps(_record(role="student"))]) + "\n")
    records = usage.read_ledger(p)
    assert [r["role"] for r in records] == ["teacher", "student"]


def test_summarize_sums_per_role_and_per_model():
    records = [
        _record(role="teacher", total_tokens=1000, input_tokens=700, output_tokens=300,
                cost_usd=2.5, duration_s=120.0),
        _record(role="teacher", total_tokens=500, input_tokens=400, output_tokens=100,
                cost_usd=1.0, duration_s=60.0),
        _record(role="student", total_tokens=200, input_tokens=150, output_tokens=50,
                model_used="claude-sonnet-4-5", model_requested="claude-sonnet-4-5",
                cost_usd=0.1, duration_s=30.0),
    ]
    s = usage.summarize(records)
    assert s["turns"] == 3
    assert s["totals"]["total_tokens"] == 1700
    assert s["totals"]["input_tokens"] == 1250 and s["totals"]["output_tokens"] == 450
    assert s["totals"]["cost_usd"] == 3.6 and s["totals"]["duration_s"] == 210.0
    assert s["by_role"]["teacher"]["turns"] == 2 and s["by_role"]["teacher"]["total_tokens"] == 1500
    assert s["by_role"]["student"]["turns"] == 1
    assert s["by_model"]["claude-opus-5-5-20260101"]["turns"] == 2
    assert s["by_model"]["claude-sonnet-4-5"]["total_tokens"] == 200


def test_a_missing_total_is_unknown_not_zero():
    s = usage.summarize([_record(total_tokens=None), _record(total_tokens=10)])
    assert s["totals"]["total_tokens"] == 10
    assert s["totals"]["unknown_usage"] == 1
    assert "1 without usage" in usage.render_markdown(s)
    assert "1 turn(s) without usage" in usage.summary_line(s)


def test_a_turn_without_a_role_or_a_model_still_counts():
    s = usage.summarize([{"total_tokens": 5}])
    assert s["by_role"]["unknown"]["turns"] == 1
    assert s["by_model"]["unknown"]["total_tokens"] == 5


def test_render_markdown_lists_totals_roles_models_and_warnings():
    s = usage.summarize([_record(),
                         _record(role="student", warning="refusal_fallback", total_tokens=1)])
    text = usage.render_markdown(s)
    assert "| **all turns** | 2 |" in text
    assert "### By role" in text and "| teacher |" in text and "| student |" in text
    assert "### By model" in text and "claude-opus-5-5-20260101" in text
    assert "refusal_fallback x1" in text
    assert "no budget, no cap" in text
    # every table row has the same number of cells as the header
    for line in text.splitlines():
        if line.startswith("|"):
            assert line.count("|") == 10, line


def test_render_markdown_says_so_when_there_is_nothing():
    assert "No agent turns" in usage.render_markdown(usage.summarize([]))
    assert usage.summary_line(usage.summarize([])) == ""


# -- where a run shows it ---------------------------------------------------------------


def _run(tmp_path, ledger: list[dict] | None) -> runs_mod.Run:
    d = tmp_path / ".artifacts" / "op_1"
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"status": "done", "updated": "now"}))
    if ledger is not None:
        usage.ledger_path(d).write_text("\n".join(json.dumps(r) for r in ledger) + "\n")
    return runs_mod.Run(dir=d, task_name="op", id="1")


def test_status_text_shows_the_usage_total_when_the_ledger_exists(tmp_path):
    run = _run(tmp_path, [_record(total_tokens=1000, cost_usd=1.5, duration_s=90.0)])
    text = launch.status_text(run)
    assert "agent usage: 1 turn(s), 1,000 tokens, $1.50, 1m30s agent time" in text


def test_status_text_leaves_the_line_out_without_a_ledger(tmp_path):
    assert "agent usage" not in launch.status_text(_run(tmp_path, None))


def test_the_run_report_carries_the_usage_section(tmp_path):
    run = _run(tmp_path, [_record(role="teacher", total_tokens=1000),
                          _record(role="student", total_tokens=200)])
    task = SimpleNamespace(name="op", mode="source")
    path = launch.write_report(run, task, 0)
    text = path.read_text()
    assert "## Agent usage" in text
    assert "agent_usage.jsonl" in text and "3 turn(s)" not in text   # 2 turns, not more
    assert "| **all turns** | 2 |" in text
    assert "| teacher |" in text and "| student |" in text


def test_the_run_report_has_no_usage_section_without_a_ledger(tmp_path):
    run = _run(tmp_path, None)
    task = SimpleNamespace(name="op", mode="source")
    assert "## Agent usage" not in launch.write_report(run, task, 0).read_text()


def test_a_driven_run_points_every_agent_turn_at_its_own_ledger(tmp_path):
    """launch._entered: teacher, student and judge turns of one run share one file, and
    the previous value comes back when the run is no longer driven."""
    import os

    from forgekernel.localenv import Machine
    run = _run(tmp_path, None)
    machine = Machine(delivery="local", gpu="0")
    before = os.environ.get("FK_AGENT_USAGE_FILE")
    with launch._entered(run, machine):
        assert os.environ["FK_AGENT_USAGE_FILE"] == str(run.dir / "agent_usage.jsonl")
        assert os.environ[launch.USAGE_ENV] == str(run.dir / "agent_usage.jsonl")
    assert os.environ.get("FK_AGENT_USAGE_FILE") == before
