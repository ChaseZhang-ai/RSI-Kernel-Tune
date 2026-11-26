"""``forge board import`` tests: read results back into a campaign ledger."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rsi_kernel_tune.board import importer as bi
from rsi_kernel_tune.board.client import BoardError

UTC = timezone.utc
NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)


def _sub(sid, name, *, status="COMPLETED", score=0.6, submitted="2026-11-24T01:00:00Z",
         collection_id=None, parent=False, child_of=None):
    row = {"id": sid, "kernel_name": name, "kernel_title": name, "status": status,
           "sol_score": score, "is_correct": status == "COMPLETED",
           "is_disqualified": False, "gpu_type": "B200", "latency_ms": 0.01,
           "fast_1_count": 1, "fast_1_total": 2, "avg_speedup": 1.5,
           "submitted_at": submitted, "finished_at": submitted,
           "evaluation_stack_version": "v1.1"}
    if parent:
        row["type"] = "collection"
        row["collection_id"] = collection_id
        row["collection_title"] = "L1 - Single Operations"
    if child_of is not None:
        row["collection_submission_id"] = child_of
        row["collection_id"] = collection_id
    return row


class FakeClient:
    token = "sekret-token"

    def __init__(self, submissions):
        self.submissions = list(submissions)
        self.filters = None

    def me(self):
        return {"authenticated": True, "user": {"id": 7, "username": "u", "email": "u@x"}}

    def iter_submissions(self, **filters):
        self.filters = filters
        return iter(self.submissions)


def _ledger(tmp_path, rows=()):
    p = tmp_path / "ledger.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return p


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_import_writes_completed_kernel_results_with_ratio(tmp_path):
    _ledger(tmp_path, [{"kind": "eval", "problem": "p1", "local_s": 0.5}])
    fake = FakeClient([
        _sub(101, "p1"),
        _sub(102, "p2", score=0.4, submitted="2026-11-24T02:00:00Z"),
        _sub(103, "p3", status="FAILED", score=None),
        _sub(104, "p1", parent=True, collection_id=1),
        _sub(105, "p1", child_of=104, collection_id=1),
    ])
    summary = bi.import_board_results(tmp_path, client=fake, now=NOW, sync_fn=None)
    assert summary.imported == 2
    assert summary.skipped == 3
    assert summary.duplicates == 0
    assert fake.filters["user_id"] == 7
    assert fake.filters["gpu_type"] == "B200"

    rows = _read(tmp_path / "ledger.jsonl")
    board = [r for r in rows if r["kind"] == "board"]
    assert {r["submission_id"] for r in board} == {101, 102}
    p1 = next(r for r in board if r["problem"] == "p1")
    assert p1["local_s"] == 0.5
    assert p1["ratio"] == pytest.approx(1.2)
    p2 = next(r for r in board if r["problem"] == "p2")
    assert "ratio" not in p2 and "local_s" not in p2
    assert "sekret-token" not in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")


def test_import_is_idempotent(tmp_path):
    _ledger(tmp_path)
    fake = FakeClient([_sub(101, "p1")])
    assert bi.import_board_results(tmp_path, client=fake, now=NOW, sync_fn=None).imported == 1
    second = bi.import_board_results(tmp_path, client=fake, now=NOW, sync_fn=None)
    assert second.imported == 0 and second.duplicates == 1
    assert len([r for r in _read(tmp_path / "ledger.jsonl") if r["kind"] == "board"]) == 1


def test_import_since_filters_by_submitted_time(tmp_path):
    _ledger(tmp_path)
    fake = FakeClient([_sub(101, "p1", submitted="2026-11-24T01:00:00Z"),
                       _sub(102, "p2", submitted="2026-11-24T06:00:00Z")])
    since = datetime(2026, 11, 24, 3, 0, tzinfo=UTC)
    summary = bi.import_board_results(tmp_path, client=fake, since=since, now=NOW,
                                      sync_fn=None)
    assert summary.imported == 1
    assert summary.records[0]["problem"] == "p2"


def test_import_dry_run_changes_nothing(tmp_path):
    ledger = _ledger(tmp_path)
    before = ledger.read_text(encoding="utf-8")
    summary = bi.import_board_results(tmp_path, client=FakeClient([_sub(101, "p1")]),
                                      dry_run=True, now=NOW, sync_fn=None)
    assert summary.dry_run and summary.imported == 0
    assert len(summary.records) == 1
    assert ledger.read_text(encoding="utf-8") == before


def test_import_uses_campaign_sync_when_available(tmp_path):
    ledger = _ledger(tmp_path)
    before = ledger.read_text(encoding="utf-8")
    calls = []

    def record(campaign, records):
        calls.append((campaign, records))

    summary = bi.import_board_results(tmp_path, client=FakeClient([_sub(101, "p1")]),
                                      now=NOW, sync_fn=record)
    assert summary.used_campaign_sync
    assert len(calls) == 1 and calls[0][0] == tmp_path
    assert calls[0][1][0]["submission_id"] == 101
    assert ledger.read_text(encoding="utf-8") == before


def test_import_requires_an_existing_campaign_dir(tmp_path):
    with pytest.raises(BoardError):
        bi.import_board_results(tmp_path / "nope", client=FakeClient([]), now=NOW)


def test_parse_since_accepts_iso_and_relative():
    assert bi.parse_since("2026-11-24") == datetime(2026, 11, 24, tzinfo=UTC)
    assert bi.parse_since("2026-11-24T08:00:00Z") == datetime(2026, 11, 24, 8, tzinfo=UTC)
    assert bi.parse_since("7d", now=NOW) == NOW - timedelta(days=7)
    assert bi.parse_since("36h", now=NOW) == NOW - timedelta(hours=36)
    assert bi.parse_since(None) is None
    with pytest.raises(BoardError):
        bi.parse_since("yesterday-ish")


def test_read_ledger_tolerates_a_partial_last_line(tmp_path):
    p = tmp_path / "ledger.jsonl"
    p.write_text('{"kind": "eval"}\nnot json\n\n{"kind": "board"}\n', encoding="utf-8")
    rows = bi.read_ledger(p)
    assert [r["kind"] for r in rows] == ["eval", "board"]
    assert bi.read_ledger(tmp_path / "missing.jsonl") == []


def test_build_record_keeps_missing_numbers_as_none():
    record = bi.build_record({"id": "5", "kernel_name": "p", "sol_score": "0.75"},
                             local_s=None, now_iso="t")
    assert record["submission_id"] == 5
    assert record["sol_score"] == 0.75
    assert record["latency_ms"] is None
    assert "ratio" not in record
