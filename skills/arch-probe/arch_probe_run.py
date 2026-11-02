#!/usr/bin/env python3
"""ARCH_PROBE runner (driver-frozen; seeded into notes/ at the distill step).

Before M0 freezes the architecture, the teacher may attach top-2 skeleton recipes
for a divided key-topology reading (``ARCH_PROBE_REQUEST: notes/arch_<n>.json`` in
the milestone). This runner executes one recipe per invocation:

    python3 notes/arch_probe_run.py notes/arch_0.json

and prints exactly one verdict line per recipe:

    ARCH_PROBE <file stem> ms=<float>     (the mean per-run time)
    ARCH_PROBE <file stem> status=FAIL reason=<first line>

**Wrong results are allowed** — what the skeleton computes does not matter, only
the time (the verdict-probe philosophy: a probe is a measuring instrument). The
skeleton is a topology mock, not a kernel implementation: it compiles, launches
and runs the candidate's main-loop shape; the driver resets it away after the
probes, so it never reaches the student.

Recipe schema (JSON):
    {"name":      "<short label, decorative>",
     "rationale": "<one sentence: why this reading>",
     "language":  "cuda" | "python",          (default "cuda")
     "rounds":    <timed repeats, default 20>,
     "source":    "<cuda: CUDA C++ defining void arch_probe_run(); "
                  "python: a snippet defining def arch_probe_run():>"}

The cuda source is built with torch's cpp_extension in the job (the same way a
CUDA-mode card builds the student kernel), so nvcc + the card's headers must be
on the image. Timing is CUDA events around the repeat loop, one warmup call
first; the printed value is the mean ms per call.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def _build_fn(recipe: dict, stem: str):
    """The recipe's ``arch_probe_run`` callable, compiled or exec'd per language."""
    language = str(recipe.get("language") or "cuda").lower()
    source = recipe.get("source")
    if not source or not isinstance(source, str):
        raise ValueError("recipe carries no 'source' (the skeleton)")
    if language == "python":
        mod: dict = {}
        exec(compile(source, f"<arch_probe:{stem}>", "exec"), mod)  # noqa: S102 — teacher's probe body, frozen runner
        fn = mod.get("arch_probe_run")
        if not callable(fn):
            raise ValueError("the python source defines no callable arch_probe_run()")
        return fn
    if language in ("cuda", "cu", "c++"):
        from torch.utils.cpp_extension import load_inline
        # One-off job artifact: the build directory lives in /tmp and dies with the job.
        import os
        os.environ.setdefault("TORCH_EXTENSIONS_DIR",
                              tempfile.mkdtemp(prefix="arch_probe_ext_"))
        mod = load_inline(name=f"arch_probe_{stem}", cpp_sources="",
                          cuda_sources=source, functions=["arch_probe_run"],
                          verbose=False)
        return mod.arch_probe_run
    raise ValueError(f"unknown language {language!r} (cuda | python)")


def probe_one(path: str) -> None:
    """Run one recipe file and print its verdict line. Never raises."""
    stem = Path(path).stem
    try:
        recipe = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(recipe, dict):
            raise ValueError("recipe is not a JSON object")
        import torch
        fn = _build_fn(recipe, stem)
        rounds = max(1, int(recipe.get("rounds") or 20))
        torch.cuda.synchronize()
        fn()                      # warmup: first launch / JIT touch
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rounds):
            fn()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / rounds
        print(f"ARCH_PROBE {stem} ms={ms:.6f}", flush=True)
    except Exception as e:  # noqa: BLE001 — a FAIL line is the protocol, not a crash
        print(f"ARCH_PROBE {stem} status=FAIL reason={type(e).__name__}: {e}",
              flush=True)


def main(argv: list[str]) -> int:
    if not argv:
        print("ARCH_PROBE status=FAIL reason=no recipe path given", flush=True)
        return 1
    for path in argv:
        probe_one(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
