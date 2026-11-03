"""WP19 过程提速三代: build-cache key narrowing, probe batch manifest, the
background measurement lane, student/teacher context bounding, the merged
teacher-validation bounce, session resume, meta-collection merging and the
cross-generation reading carry.

零行为变化纪律的验收面在这里（契约 C11/C12/C13/C15 全部默认关或默认不变）：
test_wp19_switches_off_matches_baseline_records 用对照基线（origin/main =
19e885b，WP24 后布局）语义逐字段钉住主循环输出；所有 WP19 开关在默认态下
本测试必须原样通过——这是回归，不是承诺。

黄金记录的再生成方法（供独立审查者复核）：checkout origin/main，把本文件的
_baseline_scenario / _meta_scenario 两个 helper（只调用基线就有的
run_inner_iteration / run_meta_step / FakeBackend / _ctx）拷进任意测试运行，
打印 json.dumps(rows, sort_keys=True, default=repr)，与下方 _GOLDEN_* 常量
逐字节对比——一致即证明黄金取自基线。
"""

from __future__ import annotations


import json
from pathlib import Path

from forgekernel.inner.trajectory import load_records
from forgekernel.loop import driver

from test_loop_driver import EVAL, NCU, REC, FakeBackend, _ctx, _MILESTONE
from test_wp11_throughput import _WP11_TELEMETRY

import pytest

REPO = Path(__file__).resolve().parents[1]

_MILESTONE_PATCH = "\n### D0.9 loosened\nx\n"
_PATCH_REPLY = "patch_class: structural\nchanged_entries: [D0.9]\npatch: ...\n"
_DIAGNOSIS = "cause_class: R1\ndeficient_locus: D1.1"


def _patching(tmp_path: Path, addition: str, reply: str):
    """A patch agent that edits the milestone the way a real one must (the
    test_loop_driver._patching helper, inlined so this file is self-contained
    for the golden-regeneration procedure)."""
    milestone = tmp_path / "m.md"
    if not milestone.exists():
        milestone.write_text(_MILESTONE, encoding="utf-8")

    def _run(_prompt):
        milestone.write_text(milestone.read_text(encoding="utf-8") + addition,
                             encoding="utf-8")
        return reply
    return _run


def _baseline_scenario(tmp_path: Path) -> list[dict]:
    """A KEEP then a REVERT through run_inner_iteration with every switch at
    its default. Returns the records as dicts, WP11's documented telemetry
    fields stripped (they are the merge-base's documented addition, not ours)."""
    logs = [EVAL(2.6), NCU, EVAL(2.6, False), NCU]
    be = FakeBackend([REC("D2.3", 2.6), REC("D2.4", 2.6)], logs)
    ctx = _ctx(tmp_path, be)
    driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    driver.run_inner_iteration(ctx, 1, metric_before=2.6)
    return [{k: v for k, v in r.__dict__.items() if k not in _WP11_TELEMETRY}
            for r in load_records(ctx.status_path)]


def _meta_scenario(tmp_path: Path) -> list[dict]:
    """One meta step (diagnosis then milestone patch) with every switch at its
    default. Returns the meta-trajectory records as dicts."""
    from forgekernel.meta_trajectory import load_meta_records
    patch = _patching(tmp_path, _MILESTONE_PATCH, _PATCH_REPLY)
    be = FakeBackend([_DIAGNOSIS, patch], [])
    ctx = _ctx(tmp_path, be)
    driver.run_meta_step(ctx, best_before=2.66, milestone_before="OLD")
    return [r.__dict__ for r in load_meta_records(ctx.meta_status_path)]


def _strip(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in _WP11_TELEMETRY}


def test_wp19_switches_off_matches_baseline_records(tmp_path):
    # 默认态逐字段对照：内循环（KEEP+REVERT）与 meta 步（诊断+补丁）两条场景，
    # 记录字段与黄金（基线 @19e885b 语义）一一对应。黄金的取材与再生成方法见
    # 模块 docstring。
    rows = [_strip(r) for r in _baseline_scenario(tmp_path)]
    assert len(rows) == 2
    keep, revert = rows
    assert keep["verdict"] == "KEEP" and keep["correctness"] is True
    assert keep["bwd_median_ms"] == 2.6 and keep["device_median_ms"] == 2.6
    assert revert["verdict"] == "REVERT" and revert["correctness"] is False
    assert revert["bwd_median_ms"] is None
    meta = _meta_scenario(tmp_path)
    assert len(meta) == 1
    assert meta[0]["cause_class"] == "R1" and meta[0]["patch_landed"] is True
    # full field-by-field identity against the golden captured at origin/main
    assert json.dumps(rows, sort_keys=True, default=repr) == \
        json.dumps(_GOLDEN_INNER, sort_keys=True, default=repr)
    assert json.dumps(meta, sort_keys=True, default=repr) == \
        json.dumps(_GOLDEN_META, sort_keys=True, default=repr)



_GOLDEN_INNER = [{'base_sha': 'sha0',
  'bwd_median_ms': 2.6,
  'change': 'did D2.3',
  'clock_drift': False,
  'cluster': '',
  'correctness': True,
  'decision_id': 'D2.3',
  'device_median_ms': 2.6,
  'device_window_basis': None,
  'error': '',
  'gate_spread_pct': None,
  'gpu_state': {},
  'host_bound_ratio': None,
  'host_enqueue_ms': None,
  'iter': 0,
  'kernels': [],
  'metric_before': 2.6,
  'mfu': 32.0,
  'n_workloads': 1,
  'orch_summary': {},
  'precision': {},
  'prep': {},
  'profiler': {},
  'reading_of': None,
  'ref_device_ms': None,
  'ref_host_bound_ratio': None,
  'ref_host_enqueue_ms': None,
  'ref_median_ms': None,
  'regression_vs_best_pct': None,
  'sha': 'sha1',
  'suspicious': False,
  'variant_timings': None,
  'verdict': 'KEEP',
  'vs_ref': None,
  'vs_ref_device': None,
  'workloads': []},
 {'base_sha': 'sha1',
  'bwd_median_ms': None,
  'change': 'did D2.4',
  'clock_drift': False,
  'cluster': '',
  'correctness': False,
  'decision_id': 'D2.4',
  'device_median_ms': None,
  'device_window_basis': None,
  'error': 'correctness FAIL: no failure detail in the eval JSON\n'
           '--- job log tail (last 150 lines, oldest first) ---\n'
           '=== JSON ===\n'
           '{"correctness": {"pass": false}, "performance": {"bwd_median_ms": 2.6, '
           '"device_median_ms": 2.6, "bwd_mfu_pct": 32.0}}',
  'gate_spread_pct': None,
  'gpu_state': {},
  'host_bound_ratio': None,
  'host_enqueue_ms': None,
  'iter': 1,
  'kernels': [],
  'metric_before': 2.6,
  'mfu': None,
  'n_workloads': 1,
  'orch_summary': {},
  'precision': {},
  'prep': {},
  'profiler': {},
  'reading_of': None,
  'ref_device_ms': None,
  'ref_host_bound_ratio': None,
  'ref_host_enqueue_ms': None,
  'ref_median_ms': None,
  'regression_vs_best_pct': None,
  'sha': '',
  'suspicious': False,
  'variant_timings': None,
  'verdict': 'REVERT',
  'vs_ref': None,
  'vs_ref_device': None,
  'workloads': []}]

_GOLDEN_META = [{'best_after': None,
  'best_before': 2.66,
  'cause_class': 'R1',
  'certified': None,
  'changed_entries': ['D0.9'],
  'deficient_locus': 'D1.1',
  'gate_contract': {},
  'infeasible_refusals': [],
  'lever_prices': [],
  'meta_step': 0,
  'milestone_before': 'OLD',
  'milestone_delta': '+3/-0 lines',
  'milestone_violations': [],
  'outcome': 'PENDING',
  'patch_class': 'structural',
  'patch_direction': '',
  'patch_landed': True,
  'phase_verdict': '',
  'probe_readings': [],
  'probe_rounds': 0,
  'reopen_price': '',
  'resume_after': 'D2.3'}]
