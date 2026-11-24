"""Rate-limit and status policy tests (pure functions)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from forgekernel.board import policy

UTC = timezone.utc
NOW = datetime(2026, 11, 24, 8, 30, tzinfo=UTC)


def test_standalone_delay_matches_the_guide_table():
    # 1-5 -> immediate; 6th -> 10; 7th -> 20; 8th -> 30; 11+ -> 60.
    assert [policy.standalone_delay_minutes(n) for n in range(0, 5)] == [0, 0, 0, 0, 0]
    assert policy.standalone_delay_minutes(5) == 10
    assert policy.standalone_delay_minutes(6) == 20
    assert policy.standalone_delay_minutes(7) == 30
    assert policy.standalone_delay_minutes(10) == 60
    assert policy.standalone_delay_minutes(11) == 60
    assert policy.standalone_delay_minutes(500) == 60


def test_standalone_delay_respects_account_cap_and_zero():
    assert policy.standalone_delay_minutes(9, max_delay_minutes=30) == 30
    assert policy.standalone_delay_minutes(9, max_delay_minutes=0) == 0


def test_concurrency_blocks_the_sixth_in_flight():
    usage = policy.Usage(active_submissions=4)
    assert policy.concurrency_block(usage, policy.Limits(5)) is None
    usage = policy.Usage(active_submissions=5)
    msg = policy.concurrency_block(usage, policy.Limits(5))
    assert msg and "5/5" in msg
    assert policy.concurrency_block(policy.Usage(active_submissions=99),
                                    policy.Limits(None)) is None


def test_policy_from_me_reads_submission_access():
    data = {"authenticated": True, "user": {
        "account_group": "member",
        "submission_access": {
            "limits": {"max_concurrent_submissions": 8},
            "usage": {"active_submissions": 2, "kernel_submissions_today": 6},
            "result_delay": {"max_delay_minutes": 40, "collection_delay_hours": 6},
        }}}
    pol = policy.policy_from_me(data)
    assert pol.limits.max_concurrent_submissions == 8
    assert pol.usage.active_submissions == 2
    assert pol.usage.kernel_submissions_today == 6
    assert pol.result_delay.max_delay_minutes == 40
    assert pol.account_group == "member"


def test_policy_from_me_accepts_admin_shape_and_defaults():
    data = {"user": {"submission_policy": {"limits": {"max_concurrent_submissions": 5},
                                           "result_delay": {"max_delay_minutes": 60,
                                                            "collection_delay_hours": 6}},
                     "usage": {"active_submissions": 1, "kernel_submissions_today": 2}}}
    pol = policy.policy_from_me(data)
    assert pol.usage.kernel_submissions_today == 2
    assert pol.usage.active_submissions == 1
    # Missing everything -> the guide's defaults, never an exception.
    default = policy.policy_from_me({})
    assert default.limits.max_concurrent_submissions == 5
    assert default.result_delay.max_delay_minutes == 60
    assert default.result_delay.collection_delay_hours == 6


def test_daily_notice_appears_from_the_sixth_submission():
    pol = policy.SubmissionPolicy()
    assert policy.daily_delay_notice(4, pol) is None
    notice = policy.daily_delay_notice(5, pol)
    assert notice and "第 6 次" in notice and "10 分钟" in notice


def _sub(sid, *, status="COMPLETED", collection_id=None, parent=False,
         submitted_at="2026-11-24T07:00:00Z", collection_submission_id=None,
         kind=None):
    row = {"id": sid, "status": status, "submitted_at": submitted_at,
           "collection_id": collection_id}
    if kind is not None:
        row["type"] = kind
    if parent:
        row["type"] = "collection"
    if collection_submission_id is not None:
        row["collection_submission_id"] = collection_submission_id
    return row


def test_collection_daily_count_only_counts_today_and_parents():
    rows = [
        _sub(1, collection_id=3, parent=True, submitted_at="2026-11-24T01:00:00Z"),
        _sub(2, collection_id=3, parent=True, submitted_at="2026-11-24T02:00:00Z"),
        _sub(3, collection_id=3, parent=True, submitted_at="2026-11-23T23:59:00Z"),
        _sub(4, collection_id=3, collection_submission_id=1, kind="kernel"),  # child
        _sub(5, collection_id=9, parent=True),                                # other track
        _sub(6, collection_id=3, kind="kernel"),                              # kernel sub
    ]
    assert policy.collection_submissions_today(rows, 3, NOW) == 2
    block = policy.collection_quota_block(rows, 3, NOW)
    assert block and "2/2" in block
    assert "2026-11-25 00:00:00 UTC" in block
    assert policy.collection_quota_block(rows, 9, NOW) is None


def test_collection_quota_allows_the_second_of_the_day():
    rows = [_sub(1, collection_id=3, parent=True)]
    assert policy.collection_quota_block(rows, 3, NOW) is None


def test_result_available_at_is_the_authority():
    sub = {"result_available_at": (NOW + timedelta(minutes=9)).isoformat()}
    assert policy.wait_seconds_until_available(sub, NOW) == pytest.approx(540)
    past = {"result_available_at": (NOW - timedelta(minutes=1)).isoformat()}
    assert policy.wait_seconds_until_available(past, NOW) == 0
    assert policy.wait_seconds_until_available({}, NOW) == 0


def test_wait_state_keeps_pending_result_waiting():
    sub = {"status": "PENDING_RESULT",
           "result_available_at": (NOW + timedelta(minutes=3)).isoformat()}
    status, wait = policy.wait_state(sub, NOW)
    assert status == "PENDING_RESULT" and wait == pytest.approx(180)
    done = {"status": "COMPLETED", "result_available_at": None}
    assert policy.wait_state(done, NOW) == ("COMPLETED", 0.0)


def test_parse_ts_variants():
    assert policy.parse_ts("2026-11-24T08:00:00Z") == datetime(2026, 11, 24, 8, tzinfo=UTC)
    assert policy.parse_ts("2026-11-24T16:00:00+08:00") == datetime(2026, 11, 24, 8,
                                                                  tzinfo=UTC)
    assert policy.parse_ts("2026-11-24T08:00:00") == datetime(2026, 11, 24, 8, tzinfo=UTC)
    assert policy.parse_ts("not a date") is None
    assert policy.parse_ts(None) is None


def test_status_vocabulary():
    assert policy.is_terminal("COMPLETED")
    assert policy.is_terminal("FAILED")
    assert policy.is_terminal("SYSTEM_ERROR")
    assert not policy.is_terminal("PENDING_RESULT")
    assert policy.is_in_flight("PENDING_RESULT")
    assert not policy.is_in_flight("COMPLETED")
    assert policy.status_label("PENDING_RESULT") == "结果待公布(PENDING_RESULT)"
    assert policy.status_label(None) == "-"


def test_format_duration():
    assert policy.format_duration(45) == "45 秒"
    assert policy.format_duration(600) == "10 分"
    assert policy.format_duration(3700) == "1 小时 1 分"
