#!/usr/bin/env python3
"""energize_ab.py -- 通电验收 A/B 矩阵的参数化骨架（WP21 / 甲 G25；执行归 WP23）。

本脚本只做三件事：按 docs/energize-protocol.md §1 的矩阵项组装命令、收集证据、
输出对照报告。**它绝不翻默认**——翻默认是 WP23 集成步的手工动作，一次一翻，
本脚本的存在是为了让每轮 A/B 的命令与证据格式一致、可复核。

用法（真跑一律包 GPU 锁）：
    ~/tools/bin/with-lock gpu 1800 -- \\
        python scripts/energize_ab.py --item A4 --runs-dir .artifacts \\
        --task tasks/wp21mini --mode task-set

    python scripts/energize_ab.py --item A4 --runs-dir .artifacts --report-only

骨架说明：`--mode task-set` 的跑批段以注释占位（WP23 按当时的 runner 接上，
通常即 `peregrine run` 两轮：off 一轮 on 一轮）；`--report-only` 从已完成的 run 归档
提取墙钟构成表与 verdict 对照，产出 markdown 报告。3060 上产出的数字只作链路
证据，禁止写性能结论（红线 11）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# docs/energize-protocol.md §1 的矩阵项：id -> (标题, 默认开关, 回退开关, 依赖)
ITEMS: dict[str, dict] = {
    "A1": {"title": "普通任务集执行（对照组基线）", "switch": None, "rollback": None,
            "depends": []},
    "A2": {"title": "NCU flags A/B（WP9 ncu_flag_ab 方向确认）",
            "switch": "ncu regime flags（生成点开关，WP9 原生）", "rollback": "关 flags 即回旧 regime",
            "depends": ["scripts/ncu_flag_ab.py", "CUDA torch venv"]},
    "A3": {"title": "SPRT 早停默认", "switch": "[loop] sprt_early_stop",
            "rollback": "--sprt-early-stop=false（默认 false 即回退）", "depends": ["WP15/WP19 合入"]},
    "A4": {"title": "build cache 默认", "switch": "PK_BUILD_CACHE",
            "rollback": "PK_BUILD_CACHE=0（一行）",
            "depends": ["同内核跨 job 重复构建日志", "key 覆盖用例（改头文件必 miss/改注释必命中）"]},
    "A5": {"title": "floor_stop / floor_margin", "switch": "[target] floor_margin / [loop] floor_stop",
            "rollback": "键默认关即回退",
            "depends": ["floor_backtest.json 回填（--from-run 自动化缺位——WP21 冒烟 defect D-9："
                        "backtest_floors.py 无该旗标，best_ms 需按脚本自述人工从 run 归档填，"
                        "或集成步先落自动化）", "WP15 resolvable v2"]},
    "A6": {"title": "survey 通道（teacher_prior_survey）", "switch": "[loop] teacher_prior_survey",
            "rollback": "键默认 false 即回退", "depends": ["机械存在检查（F18）"]},
    "A7": {"title": "MC-DELIVERY 激活", "switch": "教师里程碑 multi_candidate 块",
            "rollback": "教师不写块即关闭；[loop] multi_candidate_max 转发（dcfdcb5）",
            "depends": ["WP16 池合入"]},
    "A8": {"title": "stagnation cli 转发复核（无翻转）", "switch": None,
            "rollback": "cli 不传参即默认", "depends": ["dcfdcb5 接缝"]},
}

# 翻默认顺序（协议 §1 汇总行）；本骨架只提示，不执行。
FLIP_ORDER = ["A8", "A2", "A4", "A3", "A5", "A6", "A7"]

# 墙钟构成表行（driver 的 [wp11] 输出）
_WALLCLOCK_RE = re.compile(r"^\[wp11\]\s+(.+)$", re.M)
# verdict 抽取（status.jsonl 的 verdict 字段）
_STOP_RE = re.compile(r'"stop"\s*:\s*"([^"]+)"')


def run_item(args: argparse.Namespace) -> int:
    """按矩阵项组装并（在 --execute 时）执行 A/B 两轮命令。骨架默认 dry。"""
    item = ITEMS[args.item]
    print(f"[energize-ab] item {args.item}: {item['title']}")
    print(f"[energize-ab] switch: {item['switch'] or '（基线态，无开关）'}")
    print(f"[energize-ab] rollback: {item['rollback'] or '—'}")
    if item["depends"]:
        print("[energize-ab] 前置依赖（未齐不许跑）:")
        for d in item["depends"]:
            print(f"  - {d}")
    order = FLIP_ORDER.index(args.item) if args.item in FLIP_ORDER else "-"
    print(f"[energize-ab] 建议翻默认顺序位: {order}（协议 §1：一次一翻）")

    if args.item in ("A8", "A2"):
        print("[energize-ab] 本项无默认翻转：纯复核/立证，跑批段见协议 §1 对应行。")
        return 0

    if not args.execute:
        print("[energize-ab] dry 骨架（--execute 才组装真跑；真跑须 with-lock gpu 包裹）：")
    for side, flag in (("A(off/基线)", None), ("B(on/翻转)", item["switch"])):
        cmd = (f"peregrine run <runs-dir>/<{'off' if side.startswith('A') else 'on'}>-run  "
               f"# {side}"
               + (f"  env/键: {flag}" if flag else ""))
        print(f"  {cmd}")
    print("[energize-ab] 两轮完成后: --report-only 汇总墙钟构成与 verdict 对照。")
    print("[energize-ab] 红线 11：3060 数字只作链路证据，不写性能结论。")
    return 0


def _wallclock_of(run_dir: Path) -> list[str]:
    """run.log 里的 [wp11] 墙钟构成表行。"""
    log = run_dir / "run.log"
    if not log.is_file():
        return []
    return _WALLCLOCK_RE.findall(log.read_text(errors="replace"))


def _verdicts_of(run_dir: Path) -> list[str]:
    """student/status.jsonl 的逐迭代 verdict 序列。"""
    sj = run_dir / "ws" / "student" / "status.jsonl"
    if not sj.is_file():
        return []
    out = []
    for line in sj.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line).get("verdict", ""))
        except ValueError:
            continue
    return out


def report(args: argparse.Namespace) -> int:
    """--report-only：从已完成 run 归档提取证据，产 markdown 对照报告。"""
    runs = sorted(Path(args.runs_dir).glob("*")) if Path(args.runs_dir).is_dir() else []
    lines = ["# energize-ab report（链路证据；3060 数字非性能结论）", ""]
    for run in runs:
        if not run.is_dir():
            continue
        wc = _wallclock_of(run)
        vd = _verdicts_of(run)
        lines.append(f"## {run.name}")
        lines.append(f"- verdicts: {vd or '（无 status.jsonl）'}")
        if wc:
            lines.append("- wall-clock composition:")
            lines.extend(f"  - {w}" for w in wc[-3:])
        lines.append("")
    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text)
        print(f"[energize-ab] report -> {args.out}")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="energize_ab.py",
        description="WP21 通电验收 A/B 矩阵骨架：组装命令、收集证据、出报告；绝不翻默认。")
    p.add_argument("--item", choices=sorted(ITEMS), default="A1",
                   help="协议 §1 的矩阵项（A1..A8）")
    p.add_argument("--mode", choices=("task-set", "same-job", "ncu-flag-ab"),
                   default="task-set",
                   help="A/B 形态：task-set=两轮完整 run；same-job=JustGoal 同 job 对照；"
                        "ncu-flag-ab=同内核新旧 regime flags 各采一次")
    p.add_argument("--task", default="tasks/wp21mini", help="A/B 用的任务卡目录")
    p.add_argument("--runs-dir", default=".artifacts", help="run 归档根")
    p.add_argument("--execute", action="store_true",
                   help="真跑（仍须调用方以 with-lock gpu 包裹；骨架默认 dry 只打印计划）")
    p.add_argument("--report-only", action="store_true",
                   help="从 --runs-dir 已有归档提取墙钟构成与 verdict 对照出报告")
    p.add_argument("--out", default=None, help="报告写文件（缺省打 stdout）")
    args = p.parse_args(argv)

    if args.report_only:
        return report(args)
    return run_item(args)


if __name__ == "__main__":
    sys.exit(main())
