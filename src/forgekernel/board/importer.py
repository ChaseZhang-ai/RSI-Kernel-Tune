"""``forge board import <campaign>``: read back our own board results into a ledger.

The campaign ledger is an append-only ``ledger.jsonl`` (one JSON object per line;
background §8 of the prompts). This module writes one line per finished board
submission, in the same file, with ``kind": "board"`` so campaign's ``sync`` can
tell board readbacks from local evaluations:

```json
{"ts": "...", "kind": "board", "source": "board", "problem": "069_rms_norm",
 "submission_id": 123, "kernel_id": 45, "status": "COMPLETED", "gpu_type": "B200",
 "sol_score": 0.83, "local_s": 0.79, "ratio": 1.05, "is_correct": true,
 "is_disqualified": false, "latency_ms": 0.0123, "fast_1_count": 16,
 "fast_1_total": 16, "avg_speedup": 3.2, "submitted_at": "...",
 "finished_at": "...", "evaluation_stack_version": "v1.1"}
```

``local_s`` is the best local score already present for the same problem name in
the ledger (fields ``local_s`` / ``mean_s`` / ``est_s`` / a non-board ``s``);
``ratio`` is board / local, the number the pilot needs. Lines are deduplicated by
``submission_id``, so re-running ``import`` is a no-op for already-imported results.

If the campaign module exposes ``forgekernel.campaign.sync.record_board_readback(
campaign_dir, records)`` it is called instead (soft dependency); campaign then owns
the write and the ratio bookkeeping. The interface is repeated in
``reports/board-report.md``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .client import DEFAULT_GPU, BoardClient, BoardError
from .policy import format_utc, parse_ts

UTC = timezone.utc
LEDGER_NAME = "ledger.jsonl"

# Local-score fields campaign may use; `s` only counts for non-board lines.
_LOCAL_SCORE_FIELDS = ("local_s", "mean_s", "est_s", "s", "sol_score")


@dataclass
class ImportSummary:
    ledger: str
    imported: int = 0
    duplicates: int = 0
    skipped: int = 0
    dry_run: bool = False
    used_campaign_sync: bool = False
    records: list[dict] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [f"台账：{self.ledger}",
               f"导入 {self.imported} 条，跳过重复 {self.duplicates} 条，"
               f"未完成/无关 {self.skipped} 条"]
        if self.dry_run:
            out.append("（--dry-run：没有写任何文件）")
        if self.used_campaign_sync:
            out.append("已交给 campaign 的 record_board_readback 落账。")
        return out


def parse_since(text: str | None, *, now: datetime | None = None) -> datetime | None:
    """``--since``: ISO-8601 (``2026-11-24``, ``2026-11-24T08:00:00Z``) or ``7d``/``36h``."""

    if not text:
        return None
    raw = str(text).strip()
    now = now or datetime.now(UTC)
    match = re.fullmatch(r"(\d+)\s*([dh])", raw, re.IGNORECASE)
    if match:
        amount = int(match.group(1))
        delta = timedelta(hours=amount) if match.group(2).lower() == "h" else timedelta(days=amount)
        return now - delta
    dt = parse_ts(raw)
    if dt is None:
        raise BoardError(f"--since 解析不了：{text!r}（用 ISO 时间或 7d / 36h）")
    return dt


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_ledger(path: str | Path) -> list[dict]:
    """Parse a jsonl ledger, tolerating an empty or partially written last line."""

    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def imported_submission_ids(rows: Iterable[dict]) -> set[int]:
    ids: set[int] = set()
    for row in rows:
        if str(row.get("kind", "")).lower() != "board":
            continue
        sid = _int_or_none(row.get("submission_id"))
        if sid is not None:
            ids.add(sid)
    return ids


def local_best_scores(rows: Iterable[dict]) -> dict[str, float]:
    """Best local score per problem name, from non-board ledger lines."""

    best: dict[str, float] = {}
    for row in rows:
        if str(row.get("kind", "")).lower() == "board":
            continue
        problem = row.get("problem") or row.get("name") or row.get("kernel")
        if not problem:
            continue
        for field_name in _LOCAL_SCORE_FIELDS:
            value = _float_or_none(row.get(field_name))
            if value is None:
                continue
            current = best.get(str(problem))
            if current is None or value > current:
                best[str(problem)] = value
            break
    return best


def build_record(sub: dict, *, local_s: float | None, now_iso: str) -> dict:
    """One ledger line for a finished board submission."""

    problem = sub.get("kernel_name") or sub.get("kernel_title") or ""
    sol_score = _float_or_none(sub.get("sol_score"))
    record: dict[str, Any] = {
        "ts": now_iso,
        "kind": "board",
        "source": "board",
        "problem": problem,
        "submission_id": _int_or_none(sub.get("id")),
        "kernel_id": _int_or_none(sub.get("kernel_id")),
        "status": sub.get("status"),
        "gpu_type": sub.get("gpu_type"),
        "sol_score": sol_score,
        "is_correct": sub.get("is_correct"),
        "is_disqualified": sub.get("is_disqualified"),
        "latency_ms": _float_or_none(sub.get("latency_ms")),
        "fast_1_count": _int_or_none(sub.get("fast_1_count")),
        "fast_1_total": _int_or_none(sub.get("fast_1_total")),
        "avg_speedup": _float_or_none(sub.get("avg_speedup")),
        "submitted_at": sub.get("submitted_at"),
        "finished_at": sub.get("finished_at"),
        "evaluation_stack_version": sub.get("evaluation_stack_version"),
    }
    if local_s is not None:
        record["local_s"] = local_s
        if sol_score is not None and local_s > 0:
            record["ratio"] = round(sol_score / local_s, 6)
    return record


def _discover_campaign_sync():
    try:
        from forgekernel.campaign import sync as campaign_sync  # type: ignore
    except Exception:  # noqa: BLE001 - campaign may not be merged yet
        return None
    fn = getattr(campaign_sync, "record_board_readback", None)
    return fn if callable(fn) else None


def import_board_results(campaign_dir: str | Path, *, client: BoardClient,
                         since: datetime | None = None, gpu_type: str | None = DEFAULT_GPU,
                         dry_run: bool = False, now: datetime | None = None,
                         max_items: int | None = None,
                         sync_fn: Any = "auto") -> ImportSummary:
    """Import this account's finished single-kernel submissions into ``campaign_dir``.

    Only ``status == "COMPLETED"`` single-kernel submissions are imported; collection
    parents and their children are skipped (a collection's score is the collection
    mean, which campaign computes from the children anyway).
    """

    campaign = Path(campaign_dir)
    if not campaign.exists():
        raise BoardError(f"campaign 目录不存在：{campaign}")
    if not campaign.is_dir():
        raise BoardError(f"不是目录：{campaign}")

    ledger_path = campaign / LEDGER_NAME
    rows = read_ledger(ledger_path)
    already = imported_submission_ids(rows)
    local_scores = local_best_scores(rows)

    me = client.me()
    user = me.get("user") or {}
    user_id = _int_or_none(user.get("id"))
    username = user.get("username")
    if user_id is None and not username:
        raise BoardError("读取账号信息失败：既没有 user.id 也没有 user.username，"
                         "拒绝在“全部用户”的提交里导入。")

    records: list[dict] = []
    duplicates = skipped = 0
    already_seen = set(already)
    now = now or datetime.now(UTC)
    now_iso = now.astimezone(UTC).isoformat().replace("+00:00", "Z")

    for sub in client.iter_submissions(user_id=user_id, username=None if user_id else username,
                                       gpu_type=gpu_type, max_items=max_items):
        if sub.get("type") == "collection" or sub.get("collection_submission_id") is not None:
            skipped += 1
            continue
        if str(sub.get("status") or "").upper() != "COMPLETED":
            skipped += 1
            continue
        sid = _int_or_none(sub.get("id"))
        if sid is not None and sid in already_seen:
            duplicates += 1
            continue
        if since is not None:
            when = parse_ts(sub.get("finished_at")) or parse_ts(sub.get("submitted_at"))
            if when is None or when < since:
                skipped += 1
                continue
        problem = sub.get("kernel_name") or ""
        records.append(build_record(sub, local_s=local_scores.get(str(problem)), now_iso=now_iso))
        if sid is not None:
            already_seen.add(sid)

    summary = ImportSummary(ledger=str(ledger_path), imported=len(records),
                            duplicates=duplicates, skipped=skipped, dry_run=dry_run,
                            records=records)

    if not records:
        return summary
    if dry_run:
        # Nothing is written and no campaign hook is called: dry-run is read-only.
        summary.imported = 0
        return summary

    if sync_fn == "auto":
        sync_fn = _discover_campaign_sync()
    if callable(sync_fn):
        sync_fn(campaign, records)
        summary.used_campaign_sync = True
        return summary

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return summary


def summarize_for_display(summary: ImportSummary) -> str:
    lines = list(summary.lines())
    for record in summary.records[:20]:
        ratio = record.get("ratio")
        ratio_text = f"，比值 {ratio:.3f}" if isinstance(ratio, (int, float)) else ""
        score = record.get("sol_score")
        score_text = f"S={score:.4f}" if isinstance(score, (int, float)) else "S=-"
        lines.append(f"  #{record.get('submission_id')} {record.get('problem')} "
                     f"{score_text}{ratio_text} "
                     f"({format_utc(parse_ts(record.get('submitted_at')))})")
    if len(summary.records) > 20:
        lines.append(f"  … 其余 {len(summary.records) - 20} 条见台账")
    return "\n".join(lines)
