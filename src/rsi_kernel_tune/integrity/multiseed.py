"""Drop-precision kernels are only legitimate while they pass on *other* seeds too.

A kernel that accumulates in bf16 (or runs the matmul as TF32, or quantises to FP8) passes
the board's fixed seed and can miss a nearby one: the error distribution has a tail and one
draw is not a sample.  The card's fresh-input recheck compares one redraw per workload, and
its base seed is ``FK_INTEGRITY_SEED`` -- so this module drives the same card under five
seeds and requires every one of them to pass:

    from rsi_kernel_tune.integrity.multiseed import check
    verdict = check(card_dir, "submission.py")          # needs a GPU (a real Bench)
    verdict["status"], verdict["seeds"], verdict["failures"]

Each seed is one ordinary bench job with ``FK_INTEGRITY_SEED`` set in the environment (the
card reads it for the redraw), so nothing here re-implements an evaluator: the verdict per
seed is the bench's own ``correct`` and the card's own ``integrity.fresh_input_recheck``.
The environment variable is put back afterwards, whatever happens.

On a machine with no GPU this is not runnable; the orchestration (how many seeds, what the
verdict is) is what the CPU tests pin, with ``FakeBench`` standing in for the bench.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

SEED_ENV = "FK_INTEGRITY_SEED"
SEEDS = (0, 1, 7, 42, 1234)      # >= 5 draws; nobody should ship on one


def _default_bench():
    from rsi_kernel_tune.benchx.api import CardBench        # noqa: PLC0415 -- optional at import
    return CardBench()


@contextlib.contextmanager
def _env(name: str, value: str | None):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def plan(seeds=None) -> list[int]:
    """The seeds to draw with (at least five by default; a caller may name its own)."""
    out = [int(s) for s in (seeds if seeds is not None else SEEDS)]
    if not out:
        raise ValueError("multiseed.check needs at least one seed")
    return out


def _one(result) -> dict:
    """What one bench result says about the redraw, in this module's own words."""
    integrity = getattr(result, "integrity", None)
    if not isinstance(integrity, dict):
        integrity = {}
    fresh = integrity.get("fresh_input_recheck") or {}
    status = str(fresh.get("status") or "")
    correct = bool(getattr(result, "correct", False))
    if not correct:
        return {"ok": False, "fresh": status or "unreported",
                "why": str(getattr(result, "error", "") or "the bench reported correctness false")[:600]}
    if status == "fail":
        return {"ok": False, "fresh": status,
                "why": str(fresh.get("reason") or "the fresh-input recheck failed")[:600]}
    if status == "pass":
        return {"ok": True, "fresh": status, "why": ""}
    # A bench that reports no integrity block cannot testify: say so instead of counting
    # a missing measurement as a pass.
    return {"ok": True, "fresh": status or "unreported",
            "why": str(fresh.get("reason") or "the eval JSON carried no fresh-input recheck"
                       "(a card that predates the check)")[:600], "unverified": True}


def check(card_dir, candidate, *, seeds=None, bench=None, mode: str = "dev",
          workloads=None) -> dict:
    """Run the card once per seed and require every draw to pass.

    ``status`` is ``"pass"`` (every seed ran and passed), ``"fail"`` (a seed failed the
    reference comparison or the candidate raised), or ``"unavailable"`` (no bench).
    ``unverified`` lists the seeds whose eval JSON carried no recheck at all.
    """
    seeds = plan(seeds)
    if bench is None:
        try:
            bench = _default_bench()
        except Exception as exc:      # noqa: BLE001 -- no benchx, or no GPU for it
            return {"status": "unavailable", "seeds": seeds, "failures": [],
                    "why": f"no bench: {type(exc).__name__}: {exc}"}
    rows, failures, unverified = [], [], []
    for seed in seeds:
        with _env(SEED_ENV, str(seed)):
            try:
                result = bench.bench(Path(card_dir), candidate, workloads=workloads,
                                     mode=mode)
            except Exception as exc:  # noqa: BLE001 -- one bad job must not stop the rest
                rows.append({"seed": seed, "ok": False, "fresh": "error",
                             "why": f"{type(exc).__name__}: {exc}"[:600]})
                failures.append(seed)
                continue
        row = {"seed": seed, **_one(result)}
        rows.append(row)
        if not row["ok"]:
            failures.append(seed)
        elif row.get("unverified"):
            unverified.append(seed)
    if failures:
        status = "fail"
    else:
        status = "pass"
    return {"status": status, "seeds": seeds, "failures": failures,
            "unverified": unverified, "rows": rows,
            "why": ("; ".join(f"seed {r['seed']}: {r['why']}" for r in rows
                              if not r["ok"]))[:2000] if failures else ""}


def summarize(verdict: dict) -> str:
    bits = [f"multiseed {verdict.get('status')} over {len(verdict.get('seeds') or [])} seed(s)"]
    if verdict.get("failures"):
        bits.append("failed: " + ", ".join(str(s) for s in verdict["failures"]))
    if verdict.get("unverified"):
        bits.append("no recheck reported: " + ", ".join(str(s) for s in verdict["unverified"]))
    if verdict.get("why"):
        bits.append(verdict["why"])
    return "; ".join(bits)
