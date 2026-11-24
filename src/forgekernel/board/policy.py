"""The leaderboard's submission policy, as pure functions.

Two sources, both quoted in ``docs/board.md``:

* the Submission Guide's "Submission Policy" section -- first 5 single-kernel
  submissions of the UTC day are immediate, then 10/20/30… minutes capped at 60,
  at most 5 in flight, no daily cap; collections: fixed 6 h delay, 2 per UTC day;
* ``GET /api/me``'s ``submission_access`` block (``limits``, ``usage``,
  ``result_delay``), which is the per-account truth when it disagrees with the
  guide's defaults.

Waiting is decided by the server's ``result_available_at`` when a submission
carries one; the arithmetic here only *estimates* before a submission exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

UTC = timezone.utc

STANDALONE_FREE_SUBMISSIONS = 5
STANDALONE_DELAY_STEP_MINUTES = 10
STANDALONE_DELAY_CAP_MINUTES = 60
COLLECTION_DAILY_LIMIT = 2
COLLECTION_DELAY_HOURS = 6
MAX_CONCURRENT_SUBMISSIONS = 5

# The bundle's status vocabulary; the in-flight subset is its `fN` array, so a
# submission in one of these is not finished even though it may already have a score.
TERMINAL_STATUSES = ("COMPLETED", "FAILED", "SYSTEM_ERROR")
IN_FLIGHT_STATUSES = ("QUEUED", "PENDING", "RUNNING", "EVALUATING", "PENDING_RESULT")
ALL_STATUSES = IN_FLIGHT_STATUSES + TERMINAL_STATUSES

STATUS_LABELS = {
    "QUEUED": "排队",
    "PENDING": "已受理",
    "RUNNING": "运行中",
    "EVALUATING": "评测中",
    "PENDING_RESULT": "结果待公布",
    "COMPLETED": "完成",
    "FAILED": "失败",
    "SYSTEM_ERROR": "系统错误",
}


def is_terminal(status: str | None) -> bool:
    return (status or "").upper() in TERMINAL_STATUSES


def is_in_flight(status: str | None) -> bool:
    return (status or "").upper() in IN_FLIGHT_STATUSES


def status_label(status: str | None) -> str:
    if not status:
        return "-"
    label = STATUS_LABELS.get(status.upper())
    return f"{label}({status})" if label else status


# --------------------------------------------------------------------------- timestamps


def parse_ts(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp from the board; naive values are read as UTC."""

    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def now_utc() -> datetime:
    return datetime.now(UTC)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} 秒"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} 分 {sec} 秒" if sec else f"{minutes} 分"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} 小时 {minutes} 分" if minutes else f"{hours} 小时"


def format_utc(dt: datetime | None) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC") if dt else "-"


def next_utc_midnight(now: datetime) -> datetime:
    n = now.astimezone(UTC)
    return (n + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


# --------------------------------------------------------------------------- /api/me


@dataclass(frozen=True)
class Limits:
    """``submission_access.limits``. ``None`` means "not reported / unlimited"."""

    max_concurrent_submissions: int | None = MAX_CONCURRENT_SUBMISSIONS


@dataclass(frozen=True)
class ResultDelay:
    """``submission_access.result_delay``."""

    max_delay_minutes: int = STANDALONE_DELAY_CAP_MINUTES
    collection_delay_hours: int = COLLECTION_DELAY_HOURS


@dataclass(frozen=True)
class Usage:
    """``submission_access.usage``."""

    active_submissions: int = 0
    kernel_submissions_today: int = 0


@dataclass(frozen=True)
class SubmissionPolicy:
    limits: Limits = field(default_factory=Limits)
    result_delay: ResultDelay = field(default_factory=ResultDelay)
    usage: Usage = field(default_factory=Usage)
    account_group: str = ""


def _int_or(value: Any, default: int | None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def policy_from_me(data: dict) -> SubmissionPolicy:
    """``GET /api/me``'s payload (its ``data``) -> :class:`SubmissionPolicy`.

    ``/api/me`` nests everything under ``user.submission_access``; the admin view
    exposes the same numbers as ``user.submission_policy`` + ``user.usage``. Both are
    accepted so a change of shape degrades to the guide's defaults instead of
    crashing.
    """

    user = (data or {}).get("user") or {}
    access = user.get("submission_access") or user.get("submission_policy") or {}
    limits_raw = access.get("limits") or {}
    delay_raw = access.get("result_delay") or {}
    usage_raw = access.get("usage") or user.get("usage") or {}

    maxc = limits_raw.get("max_concurrent_submissions", MAX_CONCURRENT_SUBMISSIONS)
    limits = Limits(max_concurrent_submissions=None if maxc is None else _int_or(maxc, MAX_CONCURRENT_SUBMISSIONS))
    delay = ResultDelay(
        max_delay_minutes=_int_or(delay_raw.get("max_delay_minutes"), STANDALONE_DELAY_CAP_MINUTES),
        collection_delay_hours=_int_or(delay_raw.get("collection_delay_hours"), COLLECTION_DELAY_HOURS),
    )
    usage = Usage(
        active_submissions=_int_or(usage_raw.get("active_submissions"), 0) or 0,
        kernel_submissions_today=_int_or(usage_raw.get("kernel_submissions_today"), 0) or 0,
    )
    return SubmissionPolicy(limits=limits, result_delay=delay, usage=usage,
                            account_group=str(user.get("account_group") or ""))


# --------------------------------------------------------------------------- limits


def standalone_delay_minutes(today_count: int, *,
                             free: int = STANDALONE_FREE_SUBMISSIONS,
                             step_minutes: int = STANDALONE_DELAY_STEP_MINUTES,
                             max_delay_minutes: int | None = STANDALONE_DELAY_CAP_MINUTES) -> int:
    """The result-pending delay of the *next* submission of the UTC day.

    ``today_count`` is how many single-kernel submissions the account already made
    today (``usage.kernel_submissions_today``). 1-5 -> 0 minutes; the 6th -> 10;
    the 7th -> 20; … capped at ``max_delay_minutes`` (the account's
    ``result_delay.max_delay_minutes``; 0 means the account has no delay at all).
    """

    if today_count < free:
        return 0
    if max_delay_minutes is not None and max_delay_minutes <= 0:
        return 0
    delay = step_minutes * (today_count - free + 1)
    if max_delay_minutes is not None:
        delay = min(delay, max_delay_minutes)
    return delay


def concurrency_block(usage: Usage, limits: Limits) -> str | None:
    """Why another single-kernel submission is refused now, or ``None``."""

    maxc = limits.max_concurrent_submissions
    if maxc is None:
        return None
    if usage.active_submissions >= maxc:
        return (f"在途提交 {usage.active_submissions}/{maxc} 已达上限"
                f"（榜单规则：最多 {MAX_CONCURRENT_SUBMISSIONS} 个在途 = 排队+运行+待公布）。"
                "先等结果公布（forge board status / result），"
                "或等某个在途提交结束再交。")
    return None


def daily_delay_notice(today_count: int, policy: SubmissionPolicy) -> str | None:
    """A warning to print before submitting when the result will be held back."""

    delay = standalone_delay_minutes(today_count,
                                     max_delay_minutes=policy.result_delay.max_delay_minutes)
    if delay <= 0:
        return None
    return (f"这是今天第 {today_count + 1} 次单题提交：结果立即算，但要等约 {delay} 分钟才公布"
            f"（服务端 result_available_at 为准；上限 {policy.result_delay.max_delay_minutes} 分钟）。")


def _is_collection_parent(sub: dict, collection_id: int) -> bool:
    kind = sub.get("type")
    if kind is not None:
        # The status page renders `type` as "kernel" / "collection"; only the
        # collection upload itself counts against the daily quota.
        return kind == "collection" and _int_or(sub.get("collection_id"), None) == collection_id
    # Older payloads carry no type: a child submission has collection_submission_id.
    return (_int_or(sub.get("collection_id"), None) == collection_id
            and sub.get("collection_submission_id") is None)


def collection_submissions_today(submissions: Iterable[dict], collection_id: int,
                                 now: datetime) -> int:
    """How many times this account already submitted to ``collection_id`` today (UTC)."""

    day = now.astimezone(UTC).date()
    count = 0
    for sub in submissions:
        if not _is_collection_parent(sub, collection_id):
            continue
        ts = parse_ts(sub.get("submitted_at"))
        if ts is not None and ts.date() == day:
            count += 1
    return count


def collection_quota_block(submissions: Iterable[dict], collection_id: int, now: datetime, *,
                           limit: int = COLLECTION_DAILY_LIMIT) -> str | None:
    """Why a collection upload is refused now (2 per UTC day), or ``None``."""

    count = collection_submissions_today(submissions, collection_id, now)
    if count < limit:
        return None
    nxt = next_utc_midnight(now)
    return (f"该赛道（collection {collection_id}）今天已提交 {count}/{limit} 次"
            f"（每个 UTC 日最多 {limit} 次）。下一次可提交时间：{format_utc(nxt)}"
            f"（约 {format_duration((nxt - now.astimezone(UTC)).total_seconds())} 后）。")


# --------------------------------------------------------------------------- waiting


def result_available_at(sub: dict) -> datetime | None:
    return parse_ts(sub.get("result_available_at"))


def wait_seconds_until_available(sub: dict, now: datetime) -> float:
    """Seconds still to wait before the result may be revealed (0 when it is due)."""

    ts = result_available_at(sub)
    if ts is None:
        return 0.0
    return max(0.0, (ts - now.astimezone(UTC)).total_seconds())


def wait_state(sub: dict, now: datetime) -> tuple[str, float]:
    """(status, seconds to wait) for polling: terminal + due -> no wait."""

    status = str(sub.get("status") or "").upper()
    return status, wait_seconds_until_available(sub, now)
