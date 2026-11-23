"""``forge board`` -- the SOL-ExecBench leaderboard client.

Commands (all need ``SOL_EXECBENCH_TOKEN``; it is read from the environment only):

  forge board whoami                      account, group, policy (concurrency, delays), usage
  forge board submit <problem> <file>     one kernel submission (waits inside the immediate quota)
  forge board status [id]                 my recent submissions, or one submission
  forge board result <id>                 correctness, per-workload S, disqualification, stack
  forge board publish <id> [--yes]        make a submission public (confirmation required)
  forge board submit-collection <track> <zip> [--yes]
                                          upload a track archive (confirmation required)
  forge board import <campaign>           read finished results back into the campaign ledger

Behaviour is deliberately bounded: it only performs operations the web UI performs,
it honours the published rate limits on the client side (never bypasses them), it
refuses to upload anything that fails the local self-check, and every real call is
appended to ``.artifacts/board/log.jsonl`` with the token scrubbed.

This module is an ``ext`` command: ``COMMAND = "board"`` and ``main(argv) -> int``.
Until the step-0 extension registry dispatches ``forge board``, it is runnable as
``python -m forgekernel.ext.board``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forgekernel.board import policy
from forgekernel.board.client import (
    API_BASE,
    DEFAULT_GPU,
    GPU_TYPES,
    SUBMISSION_MODES,
    APIError,
    AuthError,
    BoardClient,
    BoardError,
    NetworkError,
    NotFound,
    ProtocolError,
    RateLimited,
    ServerError,
    TokenMissing,
    redact,
)
from forgekernel.board.importer import import_board_results, parse_since, summarize_for_display
from forgekernel.board.validate import (
    MAX_UPLOAD_BYTES,
    CheckResult,
    validate_collection_archive,
    validate_kernel_file,
)

COMMAND = "board"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2          # bad arguments / self-check failed
EXIT_BLOCKED = 3        # rate limit / policy / wait timeout
EXIT_NETWORK = 4        # connection / 5xx / protocol
EXIT_AUTH = 5           # 401 / 403

DEFAULT_LOG = ".artifacts/board/log.jsonl"
DEFAULT_POLL_S = 5.0


# --------------------------------------------------------------------------- output


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class AuditLog:
    """Append-only jsonl of real calls. The token never reaches the file."""

    def __init__(self, path: str | Path, token: str):
        self.path = Path(path)
        self.token = token

    def write(self, **fields: Any) -> None:
        record = {"ts": _iso_now()}
        record.update({k: v for k, v in fields.items() if v is not None})
        text = redact(json.dumps(record, ensure_ascii=False), self.token)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except OSError:
            # An unwritable audit log must not hide a successful submission.
            pass


def _print_json(obj: Any, out) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str), file=out)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_findings(result: CheckResult, out, err) -> None:
    for finding in result.warnings:
        print(f"警告：[{finding.code}] {finding.message}", file=err)
    for finding in result.errors:
        print(f"错误：[{finding.code}] {finding.message}", file=err)


def _confirm(question: str, *, yes: bool, out, err) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        print("非交互环境（stdin 不是 TTY）：确认操作必须加 --yes。", file=err)
        return False
    print(f"{question} [y/N] ", end="", file=out)
    out.flush()
    answer = sys.stdin.readline().strip().lower()
    return answer in ("y", "yes")


# --------------------------------------------------------------------------- waiting


def _wait_for_result(client: BoardClient, submission_id: int, *,
                     timeout_minutes: float, poll_s: float, out, err) -> dict:
    """Poll until terminal *and* ``result_available_at`` has passed (server clock)."""

    deadline = time.monotonic() + max(0.0, timeout_minutes) * 60.0
    last_status = None
    while True:
        sub = client.get_submission(submission_id)
        status = str(sub.get("status") or "").upper()
        now = policy.now_utc()
        wait_s = policy.wait_seconds_until_available(sub, now)
        if status != last_status:
            tail = ""
            if wait_s > 0:
                tail = f"，约 {policy.format_duration(wait_s)} 后公布"
            print(f"  [{_iso_now()}] {policy.status_label(status)}{tail}", file=err)
            last_status = status
        if policy.is_terminal(status) and wait_s <= 0:
            return sub
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"等待超时（{timeout_minutes:.0f} 分钟）：提交 #{submission_id} 仍是 "
                f"{policy.status_label(status)}。用 forge board result {submission_id} 稍后回读。")
        if wait_s > 0 and policy.is_terminal(status):
            pause = min(wait_s, remaining, 300.0)
        else:
            pause = min(max(1.0, poll_s), remaining)
        time.sleep(pause)


# --------------------------------------------------------------------------- rendering


def _short_result(sub: dict) -> tuple[str, str]:
    status = sub.get("status")
    if sub.get("is_disqualified"):
        result = "DQ"
    elif sub.get("is_correct") is True:
        result = "AC"
    elif sub.get("is_correct") is False:
        result = "WA"
    else:
        result = "-"
    return policy.status_label(status), result


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _print_submission(sub: dict, out, *, with_workloads: bool = True) -> None:
    label, result = _short_result(sub)
    ident = f"#{sub.get('id')}"
    if sub.get("type") == "collection":
        ident += f"（collection：{sub.get('collection_title') or sub.get('collection_id')}）"
    print(f"{ident}  {sub.get('kernel_title') or sub.get('kernel_name') or ''}", file=out)
    print(f"  状态：{label}    结果：{result}    device：{sub.get('gpu_type') or '-'}", file=out)
    if sub.get("sol_score") is not None:
        print(f"  SOL 分数：{_fmt_float(sub.get('sol_score'), 6)}", file=out)
    bits = []
    if sub.get("latency_ms") is not None:
        bits.append(f"延迟 {_fmt_float(sub.get('latency_ms'))} ms")
    if sub.get("fast_1_total"):
        bits.append(f"fast_1 {sub.get('fast_1_count')}/{sub.get('fast_1_total')}")
    if sub.get("avg_speedup") is not None:
        bits.append(f"平均加速 {_fmt_float(sub.get('avg_speedup'), 2)}x")
    if bits:
        print("  " + "    ".join(bits), file=out)
    print(f"  提交时间：{sub.get('submitted_at') or '-'}    "
          f"评测栈：{sub.get('evaluation_stack_version') or '-'}", file=out)
    if sub.get("result_available_at"):
        print(f"  结果公布：{sub.get('result_available_at')}", file=out)
    if sub.get("is_disqualified"):
        source = sub.get("disqualification_source") or "?"
        reason = sub.get("disqualification_reason") or ""
        print(f"  失格：{source} {reason}", file=out)
    if sub.get("error_log"):
        print("  error_log（末尾）：", file=out)
        for line in str(sub["error_log"]).splitlines()[-10:]:
            print(f"    {line}", file=out)

    if with_workloads:
        rows = sub.get("benchmark_results") or []
        if rows:
            print("", file=out)
            print(f"  {'#':>3}  {'workload':<44} {'延迟(ms)':>10} {'基线(ms)':>10} "
                  f"{'加速':>7} {'S':>8}", file=out)
            for idx, row in enumerate(rows):
                axes = row.get("workload_axes") or row.get("spec") or {}
                if isinstance(axes, dict):
                    name = ", ".join(f"{k}={v}" for k, v in axes.items())
                else:
                    name = str(axes)
                if len(name) > 44:
                    name = name[:41] + "..."
                print(f"  {idx:>3}  {name:<44} {_fmt_float(row.get('latency_ms')):>10} "
                      f"{_fmt_float(row.get('baseline_latency_ms')):>10} "
                      f"{_fmt_float(row.get('speedup_factor'), 2):>7} "
                      f"{_fmt_float(row.get('sol_score'), 6):>8}", file=out)
        elif sub.get("type") == "collection" and sub.get("kernel_submissions"):
            print("", file=out)
            print(f"  {'kernel':<44} {'状态':<16} {'S':>8}", file=out)
            for child in sub["kernel_submissions"]:
                print(f"  {str(child.get('kernel_title') or child.get('kernel_name'))[:44]:<44} "
                      f"{policy.status_label(child.get('status')):<16} "
                      f"{_fmt_float(child.get('sol_score'), 6):>8}", file=out)


def _dry_run_request(method: str, url: str, fields: dict,
                     file_path: str | Path | None, out) -> None:
    print("[DRY-RUN] 不会发送请求：", file=out)
    print(f"  {method} {url}", file=out)
    for key, value in fields.items():
        print(f"  {key}: {value}", file=out)
    if file_path is None:
        return
    p = Path(file_path)
    try:
        size = p.stat().st_size
        digest = _sha256(p)
    except OSError as exc:  # pragma: no cover - validated earlier
        size, digest = -1, f"(读不到：{exc})"
    print(f"  file: {p.name}（{size} 字节，sha256={digest}）", file=out)


# --------------------------------------------------------------------------- commands


def cmd_whoami(ns, client: BoardClient, out, err) -> int:
    data = client.me()
    user = data.get("user") or {}
    pol = policy.policy_from_me(data)
    if ns.json:
        _print_json({"authenticated": data.get("authenticated"), "user": user,
                     "policy": {"limits": pol.limits.__dict__,
                                "result_delay": pol.result_delay.__dict__,
                                "usage": pol.usage.__dict__}}, out)
        return EXIT_OK
    print(f"账号：{user.get('username') or '-'}"
          f"（{user.get('display_name') or '-'}）"
          f" <{user.get('email') or '-'}>  id={user.get('id')}", file=out)
    print(f"账号组：{user.get('account_group') or '-'}", file=out)
    maxc = pol.limits.max_concurrent_submissions
    print(f"并发上限：{'不限' if maxc is None else maxc}"
          f"（在途 {pol.usage.active_submissions}）", file=out)
    print(f"今日单题提交：{pol.usage.kernel_submissions_today}", file=out)
    print(f"结果延迟：单题最长 {pol.result_delay.max_delay_minutes} 分钟；"
          f"赛道 {pol.result_delay.collection_delay_hours} 小时", file=out)
    return EXIT_OK


def cmd_status(ns, client: BoardClient, out, err) -> int:
    if ns.id is not None:
        sub = client.get_submission(ns.id)
        if ns.json:
            _print_json(sub, out)
        else:
            _print_submission(sub, out)
        return EXIT_OK

    data = client.me()
    user = data.get("user") or {}
    uid, uname = user.get("id"), user.get("username")
    if uid is None and not uname:
        print("读取账号信息失败：没有 user.id / user.username，拒绝列出全部用户的提交。",
              file=err)
        return EXIT_ERROR
    gpu = None if (ns.all_gpus or ns.gpu_type is None) else ns.gpu_type
    subs, total = client.list_submissions(user_id=uid,
                                          username=None if uid is not None else uname,
                                          gpu_type=gpu, limit=ns.limit)
    if ns.json:
        _print_json({"total": total, "submissions": subs}, out)
        return EXIT_OK
    if not subs:
        print("还没有提交。", file=out)
        return EXIT_OK
    print(f"最近 {len(subs)}/{total} 次提交：", file=out)
    print(f"  {'#':>7}  {'状态':<18} {'结果':<5} {'S':>9}  {'提交时间':<20} 题目", file=out)
    for sub in subs:
        label, result = _short_result(sub)
        score = "-" if sub.get("sol_score") is None else f"{float(sub['sol_score']):.6f}"
        name = str(sub.get("kernel_title") or sub.get("kernel_name")
                   or sub.get("collection_title") or "-")
        if len(name) > 40:
            name = name[:37] + "..."
        print(f"  {str(sub.get('id')):>7}  {label:<18} {result:<5} {score:>9}  "
              f"{str(sub.get('submitted_at') or '-')[:19]:<20} {name}", file=out)
    return EXIT_OK


def cmd_result(ns, client: BoardClient, out, err) -> int:
    sub = client.get_submission(ns.id)
    if ns.json:
        _print_json(sub, out)
    else:
        _print_submission(sub, out)
    status = str(sub.get("status") or "").upper()
    wait_s = policy.wait_seconds_until_available(sub, policy.now_utc())
    if not policy.is_terminal(status):
        print(f"结果尚未产生（{policy.status_label(status)}）；稍后再运行 "
              f"forge board result {ns.id}。", file=err)
        return EXIT_BLOCKED
    if wait_s > 0:
        print(f"结果已算出但按榜单策略尚未公布：约 {policy.format_duration(wait_s)} 后可见。",
              file=err)
        return EXIT_BLOCKED
    return EXIT_OK


def cmd_submit(ns, client: BoardClient, out, err) -> int:
    check = validate_kernel_file(ns.file)
    _print_findings(check, out, err)
    if not check.ok:
        print("自检未通过，没有上传。", file=err)
        return EXIT_USAGE

    me = client.me()
    pol = policy.policy_from_me(me)
    block = policy.concurrency_block(pol.usage, pol.limits)
    if block:
        print(block, file=err)
        return EXIT_BLOCKED

    kernel = client.resolve_kernel(ns.problem, ns.subset)
    gpu = ns.gpu_type or DEFAULT_GPU
    supported = kernel.get("gpu_types") or list(GPU_TYPES)
    if supported and gpu not in supported:
        print(f"题目 {kernel.get('name')} 不支持 {gpu}；可选：{', '.join(supported)}", file=err)
        return EXIT_USAGE
    mode = ns.mode or "private"

    today = pol.usage.kernel_submissions_today
    delay = policy.standalone_delay_minutes(
        today, max_delay_minutes=pol.result_delay.max_delay_minutes)
    notice = policy.daily_delay_notice(today, pol)
    if notice:
        print(notice, file=err)
    wait = ns.wait if ns.wait is not None else (delay == 0)
    if wait and delay > ns.timeout:
        print(f"预计要等约 {delay} 分钟才公布，但 --timeout 只有 {ns.timeout:.0f} 分钟。"
              f"提交会成功但本命令不会等结果；可提高 --timeout 或改用 --no-wait。",
              file=err)
        if ns.wait:
            # Explicit --wait: do not submit rather than surprise the caller.
            return EXIT_BLOCKED
        wait = False

    url = f"{client.api_base}/submissions/upload"
    fields = {"kernel_id": kernel["id"], "gpu_type": gpu, "submission_mode": mode}
    if ns.dry_run:
        _dry_run_request("POST", url, fields, ns.file, out)
        return EXIT_OK

    audit = AuditLog(ns.log, client.token)
    try:
        resp = client.submit_kernel(int(kernel["id"]), ns.file, gpu_type=gpu, mode=mode)
    except BoardError as exc:
        audit.write(command="submit", problem=kernel.get("name"), ok=False, error=str(exc))
        raise
    sid = resp.get("submission_id") or resp.get("id")
    audit.write(command="submit", problem=kernel.get("name"), submission_id=sid,
                gpu_type=gpu, mode=mode, ok=True)
    if ns.json:
        _print_json({"ok": True, "submission_id": sid, "problem": kernel.get("name"),
                     "gpu_type": gpu, "mode": mode, "waited": bool(wait),
                     "estimated_delay_minutes": delay}, out)
    else:
        print(f"已提交 #{sid}：{kernel.get('name')}（{gpu}，{mode}）", file=out)
    if sid is None:
        print("服务端没有返回提交 id；用 forge board status 查看。", file=err)
        return EXIT_OK
    if not wait:
        if not ns.json:
            if delay > 0:
                print(f"结果预计 {delay} 分钟后公布；用 forge board result {sid} 回读。", file=out)
            else:
                print(f"用 forge board result {sid} 回读结果。", file=out)
        return EXIT_OK

    try:
        sub = _wait_for_result(client, int(sid), timeout_minutes=ns.timeout,
                               poll_s=ns.poll_interval, out=out, err=err)
    except TimeoutError as exc:
        print(str(exc), file=err)
        return EXIT_BLOCKED
    if ns.json:
        _print_json(sub, out)
    else:
        _print_submission(sub, out)
    return EXIT_OK


def cmd_publish(ns, client: BoardClient, out, err) -> int:
    sub = client.get_submission(ns.id)
    mode = sub.get("submission_mode")
    if mode == "release":
        print(f"提交 #{ns.id} 已经是公开（release），无需操作。", file=out)
        return EXIT_OK
    print(f"提交 #{ns.id}：{sub.get('kernel_title') or sub.get('kernel_name') or ''}  "
          f"S={_fmt_float(sub.get('sol_score'), 6)}  当前 mode={mode}", file=out)
    if ns.dry_run:
        _dry_run_request("POST", f"{client.api_base}/submissions/{int(ns.id)}/make-public",
                         {}, None, out)
        return EXIT_OK
    if not _confirm(f"确认把提交 #{ns.id} 设为公开？此操作不可撤销"
                    "（赛道提交会连同全部子提交公开）。", yes=ns.yes, out=out, err=err):
        print("已取消。", file=err)
        return EXIT_ERROR
    audit = AuditLog(ns.log, client.token)
    try:
        client.make_public(ns.id)
    except BoardError as exc:
        audit.write(command="publish", submission_id=ns.id, ok=False, error=str(exc))
        raise
    audit.write(command="publish", submission_id=ns.id, ok=True)
    print(f"已设为公开：提交 #{ns.id}。", file=out)
    return EXIT_OK


def cmd_submit_collection(ns, client: BoardClient, out, err) -> int:
    coll = client.resolve_collection(ns.track)
    cid = int(coll.get("id"))
    kernels = coll.get("kernels") or []
    expected = [str(k.get("name")) for k in kernels if k.get("name")]
    if not expected:
        print("警告：榜单没有给出该赛道的 kernel 列表，跳过目录名核对。", file=err)

    check = validate_collection_archive(ns.archive, expected_names=expected or None,
                                        require_all=ns.require_all)
    _print_findings(check, out, err)
    if not check.ok:
        print("自检未通过，没有上传。", file=err)
        return EXIT_USAGE
    if expected and check.details.get("missing_kernels"):
        print(f"注意：缺少 {len(check.details['missing_kernels'])} 道题，这些题记 0 分。",
              file=err)

    me = client.me()
    user = me.get("user") or {}
    uid, uname = user.get("id"), user.get("username")
    if uid is None and not uname:
        print("读取账号信息失败：没有 user.id / user.username，无法核对赛道每日限额。",
              file=err)
        return EXIT_ERROR
    gpu = ns.gpu_type
    supported = coll.get("gpu_types") or list(GPU_TYPES)
    if gpu is None:
        gpu = supported[0] if supported else DEFAULT_GPU
    if supported and gpu not in supported:
        print(f"赛道 {coll.get('name')} 不支持 {gpu}；可选：{', '.join(supported)}", file=err)
        return EXIT_USAGE
    mode = ns.mode or "private"

    now = policy.now_utc()
    subs, _ = client.list_submissions(user_id=uid,
                                      username=None if uid is not None else uname,
                                      collection_id=cid, limit=100)
    quota = policy.collection_quota_block(subs, cid, now)
    if quota:
        print(quota, file=err)
        return EXIT_BLOCKED

    print(f"赛道：{coll.get('name')}（{coll.get('title')}，id={cid}，{gpu}，{mode}）", file=out)
    print(f"压缩包：{ns.archive}  "
          f"（{check.details.get('file_bytes', 0)} 字节，"
          f"{check.details.get('kernel_count', 0)}/{len(expected) or '?'} 道题）", file=out)
    print("提醒：赛道结果固定延迟约 "
          f"{policy.COLLECTION_DELAY_HOURS} 小时；每个 UTC 日最多 "
          f"{policy.COLLECTION_DAILY_LIMIT} 次。", file=out)

    url = f"{client.api_base}/collections/{cid}/submit"
    fields = {"gpu_type": gpu, "submission_mode": mode}
    if ns.dry_run:
        _dry_run_request("POST", url, fields, ns.archive, out)
        return EXIT_OK
    if not _confirm(f"确认上传赛道 {coll.get('name')} 的压缩包？", yes=ns.yes, out=out, err=err):
        print("已取消。", file=err)
        return EXIT_ERROR

    audit = AuditLog(ns.log, client.token)
    try:
        resp = client.submit_collection(cid, ns.archive, gpu_type=gpu, mode=mode)
    except BoardError as exc:
        audit.write(command="submit-collection", track=coll.get("name"), ok=False,
                    error=str(exc))
        raise
    sid = resp.get("submission_id") or resp.get("id")
    audit.write(command="submit-collection", track=coll.get("name"), submission_id=sid,
                gpu_type=gpu, mode=mode, ok=True,
                kernel_count=check.details.get("kernel_count"))
    if ns.json:
        _print_json({"ok": True, "submission_id": sid, "track": coll.get("name"),
                     "gpu_type": gpu, "mode": mode,
                     "kernel_count": check.details.get("kernel_count"),
                     "collection_delay_hours": policy.COLLECTION_DELAY_HOURS}, out)
    else:
        print(f"已上传赛道提交 #{sid}。结果约 {policy.COLLECTION_DELAY_HOURS} 小时后公布；"
              f"用 forge board result {sid} 回读。", file=out)
    if not ns.wait or sid is None:
        return EXIT_OK
    try:
        sub = _wait_for_result(client, int(sid), timeout_minutes=ns.timeout,
                               poll_s=ns.poll_interval, out=out, err=err)
    except TimeoutError as exc:
        print(str(exc), file=err)
        return EXIT_BLOCKED
    if ns.json:
        _print_json(sub, out)
    else:
        _print_submission(sub, out)
    return EXIT_OK


def cmd_import(ns, client: BoardClient, out, err) -> int:
    since = parse_since(ns.since)
    gpu = None if ns.all_gpus else ns.gpu_type
    summary = import_board_results(
        ns.campaign, client=client, since=since, gpu_type=gpu,
        dry_run=ns.dry_run, sync_fn="auto")
    if ns.json:
        _print_json({"ledger": summary.ledger, "imported": summary.imported,
                     "duplicates": summary.duplicates, "skipped": summary.skipped,
                     "dry_run": summary.dry_run,
                     "used_campaign_sync": summary.used_campaign_sync,
                     "records": summary.records}, out)
        return EXIT_OK
    print(summarize_for_display(summary), file=out)
    return EXIT_OK


# --------------------------------------------------------------------------- parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-base", default=os.environ.get("SOL_EXECBENCH_API") or API_BASE,
                        help="API 前缀（默认 %(default)s）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（令牌永不出现）")
    parser.add_argument("--log", default=DEFAULT_LOG,
                        help="审计日志（默认 %(default)s，每次真实调用一行）")
    parser.add_argument("--http-timeout", type=float, default=60.0,
                        help="单次 HTTP 超时秒数（默认 %(default)s）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forge board",
        description="SOL-ExecBench 榜单客户端：提交、回读、设公开、赛道上传、导入台账。"
                    "令牌只从 SOL_EXECBENCH_TOKEN 读；客户端遵守榜单限速。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""例子：
  forge board whoami
  forge board submit 069_rms_norm out/069_rms_norm/submission.py --wait
  forge board status
  forge board result 12345 --json
  forge board publish 12345 --yes
  forge board submit-collection l1 out/l1-<UTC timestamp>.zip --yes
  forge board import .artifacts/campaign --since 7d

没有令牌时的等价手工步骤见 docs/board.md。""")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("whoami", help="账号、账号组、提交策略、今日用量")
    _add_common(p)
    p.set_defaults(func=cmd_whoami)

    p = sub.add_parser("submit", help="单题提交（默认在即时额度内等结果）")
    p.add_argument("problem", help="题目名（GET /api/kernels 的 name，如 069_rms_norm）")
    p.add_argument("file", help="submission.py")
    p.add_argument("--gpu-type", default=DEFAULT_GPU, choices=list(GPU_TYPES))
    p.add_argument("--mode", default="private", choices=list(SUBMISSION_MODES))
    p.add_argument("--subset", default=None,
                   help="按题目 tags 限定的赛道名（如 L1 / Quant），重名时用")
    wait_group = p.add_mutually_exclusive_group()
    wait_group.add_argument("--wait", dest="wait", action="store_true", default=None,
                            help="等结果（默认：当天前 5 次即时额度内等，之后不等）")
    wait_group.add_argument("--no-wait", dest="wait", action="store_false",
                            help="提交后立即返回")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="等待超时分钟（默认 %(default)s）")
    p.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_S,
                   help="轮询间隔秒（默认 %(default)s）")
    p.add_argument("--dry-run", action="store_true", help="只打印请求，不上传")
    _add_common(p)
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", help="我的最近提交，或某次提交的状态")
    p.add_argument("id", nargs="?", type=int, default=None)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--gpu-type", default=DEFAULT_GPU, choices=list(GPU_TYPES))
    p.add_argument("--all-gpus", action="store_true", help="不过滤 device")
    _add_common(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("result", help="回读结果：正确性、逐 workload S、失格")
    p.add_argument("id", type=int)
    _add_common(p)
    p.set_defaults(func=cmd_result)

    p = sub.add_parser("publish", help="设为公开（默认要确认）")
    p.add_argument("id", type=int)
    p.add_argument("--yes", action="store_true", help="跳过确认")
    p.add_argument("--dry-run", action="store_true", help="只打印请求")
    _add_common(p)
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser("submit-collection", help="上传赛道压缩包（默认要确认）")
    p.add_argument("track", help="l1 / l2 / quant / flashinfer-bench（或 id）")
    p.add_argument("archive", help="zip / tar.gz / tgz")
    p.add_argument("--yes", action="store_true", help="跳过确认")
    p.add_argument("--dry-run", action="store_true", help="只打印请求")
    p.add_argument("--gpu-type", default=None, choices=list(GPU_TYPES))
    p.add_argument("--mode", default="private", choices=list(SUBMISSION_MODES))
    p.add_argument("--require-all", action="store_true",
                   help="缺少赛道里的任何一题就拒绝（默认只警告，缺题记 0 分）")
    p.add_argument("--wait", action="store_true", help="等结果（默认不等，6 小时太久）")
    p.add_argument("--timeout", type=float, default=420.0,
                   help="--wait 时的超时分钟（默认 %(default)s）")
    p.add_argument("--poll-interval", type=float, default=60.0)
    _add_common(p)
    p.set_defaults(func=cmd_submit_collection)

    p = sub.add_parser("import", help="把自己已完成的提交导进 campaign 台账")
    p.add_argument("campaign", help="campaign 目录（写入它的 ledger.jsonl）")
    p.add_argument("--since", default=None, help="ISO 时间，或 7d / 36h")
    p.add_argument("--gpu-type", default=DEFAULT_GPU, choices=list(GPU_TYPES))
    p.add_argument("--all-gpus", action="store_true", help="导入所有 device")
    p.add_argument("--dry-run", action="store_true", help="只显示将导入的记录")
    _add_common(p)
    p.set_defaults(func=cmd_import)

    return parser


# --------------------------------------------------------------------------- entry


def main(argv: list[str] | None = None, *, client: BoardClient | None = None,
         out=None, err=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    ns = parser.parse_args(argv)
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    if not getattr(ns, "func", None):
        parser.print_help(out)
        return EXIT_OK

    try:
        if client is None:
            client = BoardClient(api_base=ns.api_base, timeout=ns.http_timeout)
        client.require_token()
    except TokenMissing as exc:
        print(str(exc), file=err)
        return EXIT_ERROR

    try:
        return int(ns.func(ns, client, out, err) or EXIT_OK)
    except TokenMissing as exc:
        print(str(exc), file=err)
        return EXIT_ERROR
    except AuthError as exc:
        print(str(exc), file=err)
        return EXIT_AUTH
    except RateLimited as exc:
        print(str(exc), file=err)
        return EXIT_BLOCKED
    except (NetworkError, ServerError, ProtocolError) as exc:
        print(str(exc), file=err)
        return EXIT_NETWORK
    except NotFound as exc:
        print(str(exc), file=err)
        return EXIT_ERROR
    except APIError as exc:
        print(str(exc), file=err)
        return EXIT_NETWORK
    except BoardError as exc:
        print(str(exc), file=err)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("已中断。", file=err)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
