#!/usr/bin/env python3
"""Fetch the public SOL-ExecBench leaderboard snapshot ARGUS plans against.

Only the endpoints that are actually public are used (``/stats``, ``/collections``,
``/leaderboard/collections``, ``/kernels``).  The **per-workload anchors** (``T_SOL``, ``T_b``)
are *not* exposed by these routes: they are published inside the harness dataset
(``<collection>/<kernel>/workload.jsonl`` in ``NVIDIA/SOL-ExecBench``), which is the
authoritative source the board itself scores with.  So this snapshot records the board
structure and the rank table, and ``docs/LEADERBOARD_ANALYSIS.md`` explains how to compute
targets once the dataset is vendored.

Usage:  python3 scripts/fetch_board.py [--out data/board_snapshot.json]
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone

BASE = "https://research.nvidia.com/benchmarks/sol-execbench/api"


def _get(path: str, timeout: int = 40):
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "argus/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch() -> dict:
    snap = {"fetched_at": datetime.now(timezone.utc).isoformat(), "source": BASE,
            "stats": None, "collections": None, "rankings": None, "kernels": None,
            "notes": [
                "per-workload T_SOL / T_b are published in the SOL-ExecBench dataset, not here",
                "kernel rows carry id/name/tags/submission_count only",
            ]}
    for key, path in (("stats", "/stats"), ("collections", "/collections"),
                      ("rankings", "/leaderboard/collections"), ("kernels", "/kernels")):
        try:
            snap[key] = _get(path)
        except Exception as exc:
            print(f"[fetch_board] {path} failed: {exc}")
    return snap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/board_snapshot.json")
    args = ap.parse_args()
    snap = fetch()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)
    kernels = ((snap.get("kernels") or {}).get("data") or {}).get("kernels") or []
    print(f"[fetch_board] wrote {args.out}: stats={'ok' if snap['stats'] else 'missing'}, "
          f"{len(kernels)} kernel rows")
    return 0 if snap["stats"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
