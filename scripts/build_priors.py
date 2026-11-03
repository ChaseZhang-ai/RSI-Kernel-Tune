#!/usr/bin/env python3
"""Build the repo prior library: one architecture-decision-map skeleton per
published kernel under ``kernels/``.

    python3 scripts/build_priors.py            # dry run: list what would be written
    python3 scripts/build_priors.py --write    # write skeletons for kernels lacking one
    python3 scripts/build_priors.py --write --force   # refresh auto skeletons (never filled files)

``priors/`` is the teacher's few-shot library (WP8): per kernel, the decisions
that define its architecture — persistent scheduling or not, warp-role split,
pipeline depth, data residency, epilogue path, the shape domain it wins in —
10–20 lines each, distilled into prose. The script writes the SKELETON only
(metadata from ``kernels/README.md`` + empty decision sections); the content is
filled by hand or by a later run's harvest. Three samples ship filled (GEMM /
attention / conv) as the quality bar.

Red lines (enforced here, checked in tests):
- a summary states decisions and values, never an implementation body — the
  script emits prose stubs only, and a filled file with a ``{`` density
  resembling code is a review reject;
- ``priors/`` is teacher read-set: nothing in it names a run's sealed reference
  or carries experiment narrative (dates, job ids) — it is a library, not a lab
  notebook.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KERNELS = REPO / "kernels"
README = KERNELS / "README.md"
PRIORS = REPO / "priors"

# A kernels/README.md table row: | operator | gpu | vs | speedup | source | [`dir`](dir/) |
# (the directory cell is a markdown link — the optional `[` before the backtick)
_ROW_RE = re.compile(
    r"^\|(?P<op>[^|]+)\|(?P<gpu>[^|]+)\|(?P<vs>[^|]+)\|(?P<speedup>[^|]+)\|"
    r"(?P<source>[^|]+)\|\s*\[?`(?P<dir>[^`]+)`", re.M)

# A filled file declares itself; --force refreshes only auto skeletons.
_FILLED_MARK = "Status: filled"
_SKELETON_MARK = "Status: skeleton"

_SECTIONS = [
    ("Persistent scheduling",
     "grid == SM count with a software tile loop, or one CTA per output tile? "
     "How are tail tiles absorbed?"),
    ("Warp-role split",
     "which warps produce (TMA/cp.async), which consume (MMA), which own the "
     "epilogue; the register split between them"),
    ("Pipeline depth",
     "stages per ring buffer, who arrives/waits on what, the in-flight depth "
     "the schedule sustains"),
    ("Data residency",
     "what lives in smem vs registers vs L2 across the hot loop; what is "
     "multicast/shared inside a cluster"),
    ("Epilogue path",
     "accumulator to smem to global spelling (R2S/STSM/TMA store), swizzle, "
     "whether a second kernel or an atomic ever touches the output"),
    ("Shape domain",
     "the shapes this architecture wins at (decode vs prefill, tall vs square "
     "GEMM, grouped vs dense) and where it degrades"),
]


def _rows() -> dict[str, dict[str, str]]:
    """``kernels/README.md`` table rows keyed the same way ``_kernels()`` spells
    a kernel directory (``kernels/<family>/<op>/<gpu>``) — the two must meet,
    or every skeleton would fall back to bare names."""
    out: dict[str, dict[str, str]] = {}
    if not README.is_file():
        return out
    for m in _ROW_RE.finditer(README.read_text(encoding="utf-8")):
        d = m.group("dir").strip().rstrip("/")
        out[f"kernels/{d}"] = {k: m.group(k).strip() for k in
                               ("op", "gpu", "vs", "speedup", "source")}
    return out


def _kernels() -> list[str]:
    """Published kernel directories (relative to the repo root): leaves holding
    kernel.py. The CUDA-era kernel ships kernel.py alongside its .cu sources,
    so kernel.py is the marker — a .cu-only leaf would be missed, but there is
    none in the tree."""
    out = []
    for f in sorted(KERNELS.rglob("kernel.py")):
        out.append(str(f.parent.relative_to(REPO)))
    return out


def _priors_rel(kernel_dir: str) -> str:
    """kernels/<family>/<op>/<gpu>[/<gen>] -> priors/<family>/<op>_<gpu>[_<gen>].md"""
    parts = Path(kernel_dir).parts[1:]          # drop "kernels"
    return str(Path("priors", parts[0], "_".join(parts[1:]) + ".md"))


def _skeleton(kernel_dir: str, meta: dict[str, str]) -> str:
    d = REPO / kernel_dir
    lang = ("CUDA C++" if any(d.glob("*.cu")) else "CuTe DSL (nvidia-cutlass-dsl)")
    parts = Path(kernel_dir).parts          # ("kernels", family, operator, gpu[, gen])
    # the README row's operator name when it has one; else the OPERATOR directory
    # segment — never the leaf (a GPU codename like "b200" names no kernel)
    title = meta.get("op") or (parts[2] if len(parts) > 2 else parts[-1])
    lines = [
        f"# prior: {title}",
        "",
        f"- Kernel: `{kernel_dir}/` ({lang})",
        f"- Published: {meta.get('speedup', '?')} vs {meta.get('vs', '?')} "
        f"({meta.get('source', 'kernels/README.md')})",
        f"- {_SKELETON_MARK} (auto-generated; fill the sections below — decisions and "
        "values in prose, never an implementation body)",
        "",
    ]
    for name, hint in _SECTIONS:
        lines += [f"## {name}", "", f"TODO: {hint}.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="build_priors.py", description=__doc__.split("\n\n")[0],
        epilog="Filled samples ship in priors/ (gemm, attention, conv); the script "
               "never overwrites a file whose Status: filled line is present.")
    p.add_argument("--write", action="store_true",
                   help="write skeletons (default: dry run, list only)")
    p.add_argument("--force", action="store_true",
                   help="with --write: refresh auto skeletons too (a filled file is "
                        "never overwritten)")
    p.add_argument("--repo", default=None, type=Path,
                   help="the repository root (default: this script's repo; a test "
                        "or a tool points it at another checkout)")
    args = p.parse_args(argv)
    if args.repo is not None:
        # retarget the module-level roots (tests / another checkout)
        globals()["REPO"] = args.repo.resolve()
        globals()["KERNELS"] = REPO / "kernels"
        globals()["README"] = KERNELS / "README.md"
        globals()["PRIORS"] = REPO / "priors"

    rows = _rows()
    kernels = _kernels()
    written, skipped_filled, have = [], [], 0
    for k in kernels:
        rel = _priors_rel(k)
        dst = REPO / rel
        if dst.is_file():
            have += 1
            if _FILLED_MARK in dst.read_text(encoding="utf-8"):
                skipped_filled.append(rel)
                continue
            if not args.force:
                continue
        if args.write:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(_skeleton(k, rows.get(k, {})), encoding="utf-8")
            written.append(rel)
        else:
            print(f"would write {rel}"
                  + (" (refresh)" if dst.is_file() else ""))
    print(f"{len(kernels)} published kernel(s); {have} summary(ies) on file "
          f"({len(skipped_filled)} filled and protected)")
    if args.write:
        for w in written:
            print(f"wrote {w}")
        print(f"{len(written)} skeleton(s) written")
    else:
        print("dry run — pass --write to create the missing skeletons")
    return 0


if __name__ == "__main__":
    sys.exit(main())
