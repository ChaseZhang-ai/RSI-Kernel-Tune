#!/usr/bin/env python3
"""WP16 dry-run: render every injected block the package adds, with fixture
data, so the acceptance wording (COMBO-POOL 样例 / pattern search 读数表 /
重启靶子行 / EXPIRED 行 / 耦合图边表) has concrete artifacts to eyeball.
Run: python dry_run_wp16.py (torch-free, no backend, no GPU)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from forgekernel.loop import driver as drv
from forgekernel.loop import lever_pool
from forgekernel.inner.trajectory import IterationRecord


def main() -> int:
    print("=" * 72)
    print("WP16 dry-run — all injected blocks, fixture data, deterministic")
    print("=" * 72)

    # --- 1. the COMBO-POOL block (contract C10 position, <= 10 lines) --------
    rows = [lever_pool.admission(
        {"kind": "lever", "name": name, "cls": "null", "delta_pct": 0.1,
         "priced_on": "0123456789abcdef", "job": "812345",
         "spec": {"name": name, "edits": [{"old": "a", "new": name}]},
         "decisions": ["D1.1"]}, [], ts=1000.0, step=2, run="demo-run")
        for name in ("tilenh8", "kvstage2", "regsplit_184_120")]
    print("\n### 1. COMBO-POOL block (empty pool renders NOTHING — byte-identity)")
    print(lever_pool.combo_pool_block(rows))
    print("### with a null bundle reading:")
    print(lever_pool.combo_pool_block(rows, bundle={"all_pct": 0.2, "noise_pct": 0.5,
                                                    "null": True}))

    # --- 2. the pattern search readings table (乙A3) -------------------------
    res = {"trigger": "stagnation", "decision_id": "D2.3", "best_sha": "0123456789abcdef",
           "stages_run": 3, "jobs": 3,
           "rows": [
               {"name": "tile_16", "ratio": 0.912, "delta_pct": -1.85, "ms": 3.611,
                "floor_pct": 0.51, "ok": True},
               {"name": "tile_32", "ratio": 0.948, "delta_pct": 0.42, "ms": 3.385,
                "floor_pct": 0.51, "ok": True},
               {"name": "tile_48", "ratio": 0.939, "delta_pct": 1.96, "ms": 3.412,
                "floor_pct": 0.63, "ok": True},
               {"name": "tile_64", "ratio": 0.953, "delta_pct": 2.31, "ms": 3.361,
                "floor_pct": 0.63, "ok": True},
               {"name": "tile_96", "ratio": 0.951, "delta_pct": 1.02, "ms": 3.390,
                "floor_pct": 0.58, "ok": True},
               {"name": "tile_128", "ratio": 0.941, "delta_pct": -0.25, "ms": 3.455,
                "floor_pct": 0.58, "ok": True}],
           "winner": {"name": "tile_64", "axis": "tile", "value": 64.0,
                      "delta_pct": 2.31}}
    print("\n### 2. Pattern search readings table (stagnation trigger)")
    print(drv._pattern_search_block(res))

    # --- 3. the STAGNATION restart-target lines (contract C10, <= 6 lines) ---
    recs = [IterationRecord(iter=1, decision_id="D1.2", change="c", correctness=True,
                            verdict="KEEP", device_median_ms=2.00, sha="a" * 40),
            IterationRecord(iter=2, decision_id="D2.3", change="c", correctness=True,
                            verdict="KEEP", device_median_ms=2.03, sha="b" * 40)]
    class _Ctx:
        gate_window = "device"
        runner_up_pool_k = 3
    print("\n### 3. STAGNATION restart target (appended AFTER the WP10 block; "
          "the block itself is untouched)")
    ctx = _Ctx()
    print(drv._restart_target_lines(ctx, recs))
    print("### (no #2 -> nothing; pool empty -> nothing)")

    # --- 4. the EXPIRED price-tag review lines (甲G12) ------------------------
    rv = {"ts": 1762100000.0, "iter": 5, "lever": "win",
          "priced_on_before": "0123456789abcdef", "job_before": "812345",
          "before": {"add_pct": 2.0, "remove_pct": None, "delta_pct": None,
                     "cls": "independent"},
          "priced_on_now": "fedcba9876543210", "job_now": "", "now": None,
          "verdict": "expired",
          "note": ("add +2.00% > noise 0.50% on the old tree; the milestone does not "
                   "cite it and no fresh reading covers it on the new best — re-price "
                   "it before citing it there"),
          "retired": False}
    print("\n### 4. EXPIRED row (rides the existing Price-tag review block)")
    print(drv._price_revision_teacher_block([rv]))

    # --- 5. the coupling-graph edges inside the price block (甲G17) ----------
    table = [
        {"kind": "lever", "name": "tilenh8", "add_pct": 1.2, "remove_pct": -0.4,
         "cls": "independent", "spin_add_ns": 5.0, "spin_all_minus_ns": 1.0,
         "decisions": ["D1.1"]},
        {"kind": "lever", "name": "kvstage2", "add_pct": 0.8, "remove_pct": -0.2,
         "cls": "independent", "spin_add_ns": 3.0, "spin_all_minus_ns": 0.5,
         "decisions": ["D1.2"]},
        {"kind": "lever", "name": "regsplit_184_120", "add_pct": 0.1, "remove_pct": 0.1,
         "cls": "null", "spin_add_ns": 0.0, "spin_all_minus_ns": 0.0,
         "decisions": ["D2.1"]},
        {"kind": "lever", "name": "lanepct", "add_pct": -1.1, "remove_pct": 0.0,
         "cls": "harmful", "spin_add_ns": -4.0, "spin_all_minus_ns": 0.0,
         "decisions": ["D2.2"]},
        {"kind": "stack", "all_pct": 0.9, "sum_singles_pct": 1.9,
         "coupling_pp": -1.0, "all_pos_pct": 2.1,
         "all_pos_levers": ["tilenh8", "kvstage2"], "dropped": ["lanepct"],
         "noise_pct": 0.5, "base_ms": 3.42, "all_ms": 3.40},
    ]
    specs = [
        {"name": "tilenh8", "edits": [{"old": "a", "new": "b"}]},
        {"name": "kvstage2", "edits": [{"old": "c", "new": "d"}],
         "requires": ["tilenh8"], "regime": "prefill"},
        {"name": "regsplit_184_120", "edits": [{"old": "e", "new": "f"}],
         "conflicts": ["lanepct"]},
        {"name": "lanepct", "edits": [{"old": "g", "new": "h"}]},
    ]
    print("\n### 5. Lever coupling graph (rendered inside the existing price block)")
    print(drv._lever_price_block(table, specs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
