#!/usr/bin/env python3
"""Refresh ``data/sol_execbench_b200_flashinfer.json`` from the public board API.

The SOL-ExecBench API publishes, per kernel: the per-workload ``baseline_latency_ms``
(T_b) and ``sol_ms`` (T_SOL), plus the per-kernel leaderboard.  That is everything
VULCAN needs to reproduce the scoring rule and set latency targets.

Usage::

    python scripts/fetch_board.py --out data/sol_execbench_b200_flashinfer.json

Only the standard library is used.  The script is idempotent; re-run it to track the
live board (competitor scores move).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://research.nvidia.com/benchmarks/sol-execbench/api"
COLLECTION_ID = 4          # FlashInfer-Bench
GPU = "B200"

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
from vulcan.dataset import infer_op_class  # noqa: E402


def _get(url: str, retries: int = 3) -> dict:
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return json.load(r)
        except Exception as e:            # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def fetch(out_path: str) -> dict:
    kernels = _get(f"{API}/kernels")["data"]["kernels"]
    fbi = sorted([k for k in kernels if "FlashInfer-Bench" in k.get("tags", [])],
                 key=lambda k: k["name"])
    if not fbi:
        raise RuntimeError("no FlashInfer-Bench kernels returned")

    out = {"meta": {
        "source": f"{API}",
        "collection_id": COLLECTION_ID,
        "gpu": GPU,
        "suite": "flashinfer-bench",
        "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_kernels": len(fbi),
    }, "kernels": {}}

    n_workloads = 0
    for k in fbi:
        kid = k["id"]
        det = _get(f"{API}/kernels/{kid}")["data"]
        lb = _get(f"{API}/leaderboard/kernel/{kid}/{GPU}")["data"]
        entries = lb.get("rankings") or lb.get("reference_entries") or []

        def score_of(user):
            for e in entries:
                if e.get("username") == user:
                    return e.get("sol_score"), e.get("fast_1_count"), e.get("fast_1_total")
            return None, None, None

        fk, fk_f, fk_t = score_of("ForgeKernel")
        gb, gb_f, gb_t = score_of("Golden Bridge")
        cands = [e for e in entries
                 if not e.get("is_reference") and e.get("sol_score") is not None]
        best = max(cands, key=lambda e: e["sol_score"]) if cands else None
        best_s = best["sol_score"] if best else None
        best_holder = best["username"] if best else None

        wls = []
        for w in det.get("workloads", []):
            t_b = w.get("baseline_latency_ms")
            t_sol = w.get("sol_ms")
            if t_b is None or t_sol is None:
                continue
            wls.append({"axes": w.get("axes", {}), "t_b_ms": float(t_b),
                        "t_sol_ms": float(t_sol)})
        n_workloads += len(wls)

        out["kernels"][k["name"]] = {
            "id": kid,
            "tags": list(k.get("tags", [])),
            "description": det.get("description", ""),
            "op_class": infer_op_class(k["name"]),
            "baseline_latency_ms": det.get("baseline_latency_ms"),
            "n_workloads": len(wls),
            "scores": {
                "forgekernel": fk,
                "golden_bridge": gb,
                "best": best_s,
                "best_holder": best_holder,
            },
            "fast1": {
                "forgekernel_missing": (fk_t - fk_f) if (fk_t and fk_f is not None) else None,
                "golden_bridge_missing": (gb_t - gb_f) if (gb_t and gb_f is not None) else None,
                "total": fk_t,
            },
            "workloads": wls,
        }
        time.sleep(0.1)

    out["meta"]["n_workloads"] = n_workloads

    # published collection scores, and the mean of per-kernel best-known
    def mean_score(key):
        vals = [v["scores"].get(key) for v in out["kernels"].values()]
        vals = [x for x in vals if x is not None]
        return sum(vals) / len(vals) if vals else None

    out["meta"]["published"] = {
        "forgekernel": mean_score("forgekernel"),
        "golden_bridge": mean_score("golden_bridge"),
        "best_known_mean": mean_score("best"),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parents[1] / "data"
                                         / "sol_execbench_b200_flashinfer.json"))
    args = ap.parse_args(argv)
    ds = fetch(args.out)
    print(f"wrote {args.out}: {ds['meta']['n_kernels']} kernels, "
          f"{ds['meta']['n_workloads']} workloads")
    p = ds["meta"]["published"]
    print(f"  published: FK={p['forgekernel']:.6f} "
          f"GB={p['golden_bridge']:.6f} best-known-mean={p['best_known_mean']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
