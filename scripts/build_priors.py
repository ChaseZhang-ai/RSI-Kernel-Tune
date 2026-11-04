#!/usr/bin/env python3
"""Build the repo prior library: one architecture-decision-map skeleton per
published kernel under ``kernels/``.

    python3 scripts/build_priors.py            # dry run: list what would be written
    python3 scripts/build_priors.py --write    # write skeletons for kernels lacking one
    python3 scripts/build_priors.py --write --force   # refresh auto skeletons (never filled files)
    python3 scripts/build_priors.py --from-artifacts  # fill skeletons from measured artifacts

``priors/`` is the teacher's few-shot library (WP8): per kernel, the decisions
that define its architecture — persistent scheduling or not, warp-role split,
pipeline depth, data residency, epilogue path, the shape domain it wins in —
10–20 lines each, distilled into prose. The script writes the SKELETON only
(metadata from ``kernels/README.md`` + empty decision sections); the content is
filled by hand or by a later run's harvest. Three samples ship filled (GEMM /
attention / conv) as the quality bar.

``--from-artifacts`` (WP18) fills the skeletons' decision sections from the
MEASURED artifacts ``scripts/collect_kernel_artifacts.py`` lands under
``priors/artifacts/`` (NCU key metrics + SASS census). Every line it writes is
a ``Measured anchor:`` fact — a number or count straight from the artifacts,
with its source named — never a narrative decision; a section with no artifact
face keeps its TODO. When every section carries an anchor the file flips to
``Status: filled (auto from measured artifacts …)``; partial anchors keep a
protective ``Status: auto-anchored`` (``--force`` never refreshes either one).

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
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KERNELS = REPO / "kernels"
README = KERNELS / "README.md"
PRIORS = REPO / "priors"
ARTIFACTS = REPO / "priors" / "artifacts"

# A kernels/README.md table row: | operator | gpu | vs | speedup | source | [`dir`](dir/) |
# (the directory cell is a markdown link — the optional `[` before the backtick)
_ROW_RE = re.compile(
    r"^\|(?P<op>[^|]+)\|(?P<gpu>[^|]+)\|(?P<vs>[^|]+)\|(?P<speedup>[^|]+)\|"
    r"(?P<source>[^|]+)\|\s*\[?`(?P<dir>[^`]+)`", re.M)

# A filled file declares itself; --force refreshes only auto skeletons.
_FILLED_MARK = "Status: filled"
_SKELETON_MARK = "Status: skeleton"
_ANCHORED_MARK = "Status: auto-anchored"

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


def _artifact_dir(kernel_dir: str) -> Path:
    """The matching artifacts directory (same naming rule, under priors/artifacts/)."""
    parts = Path(kernel_dir).parts[1:]          # drop "kernels"
    return ARTIFACTS / parts[0] / "_".join(parts[1:])


def _load_artifacts(kernel_dir: str) -> dict:
    """The measured faces this kernel's artifacts carry: the manifest (regime),
    the NCU key metrics and the SASS census histograms. {} when none landed."""
    d = _artifact_dir(kernel_dir)
    out: dict = {}
    try:
        for name, key in (("manifest.json", "manifest"), ("ncu.json", "ncu"),
                          ("sass_census.json", "sass")):
            p = d / name
            if p.is_file():
                out[key] = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return out


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _measured_anchors(arts: dict) -> dict[str, list[str]]:
    """{section name: [Measured anchor lines]} — every line one machine fact
    straight from the artifacts, source named, no interpretation. The census
    counts bound the role/pipeline/residency/epilogue sections; the NCU
    readings bound the schedule/residency sections; the manifest's regime line
    bounds the shape domain."""
    ncu = arts.get("ncu") or {}
    sass_fams: dict[str, int] = {}
    for census in (arts.get("sass") or {}).values():
        for fam, n in (census.get("families") or {}).items():
            sass_fams[fam] = sass_fams.get(fam, 0) + int(n)
    regime = str((arts.get("manifest") or {}).get("regime") or "")
    anchors: dict[str, list[str]] = {}

    def has(*keys):
        return all(k in ncu for k in keys)

    sched = []
    if has("grid_size", "block_size"):
        sched.append(f"Measured anchor (ncu): the published launch is grid={_fmt(ncu['grid_size'])} "
                     f"CTAs of {_fmt(ncu['block_size'])} threads")
    if has("occupancy_pct"):
        sched.append(f"Measured anchor (ncu): achieved occupancy {_fmt(ncu['occupancy_pct'])}% "
                     "of peak active warps")
    if sass_fams.get("reg_role"):
        sched.append(f"Measured anchor (sass census): {sass_fams['reg_role']} register-role "
                     "instruction(s) (setmaxnreg family) in the shipped cubin")
    if sched:
        anchors["Persistent scheduling"] = sched

    roles = []
    if sass_fams.get("reg_role"):
        roles.append(f"Measured anchor (sass census): {sass_fams['reg_role']} role-split "
                     "instruction(s) — the register budget is set per role in the shipped cubin")
    if sass_fams.get("named_sync"):
        roles.append(f"Measured anchor (sass census): {sass_fams['named_sync']} "
                     "grid-level/named sync instruction(s)")
    if roles:
        anchors["Warp-role split"] = roles

    depth = []
    if sass_fams.get("mbarrier"):
        depth.append(f"Measured anchor (sass census): {sass_fams['mbarrier']} "
                     "mbarrier/fence instruction(s) in the hot loop's cubin")
    if sass_fams.get("cp_async"):
        depth.append(f"Measured anchor (sass census): {sass_fams['cp_async']} "
                     "async-copy instruction(s) (the ring's supply side)")
    if depth:
        anchors["Pipeline depth"] = depth

    residency = []
    if has("smem_per_block"):
        residency.append(f"Measured anchor (ncu): {_fmt(ncu['smem_per_block'])} B shared memory "
                         "per block at the published launch")
    if sass_fams.get("tma"):
        residency.append(f"Measured anchor (sass census): {sass_fams['tma']} TMA instruction(s)")
    if residency:
        anchors["Data residency"] = residency

    epilogue = []
    if sass_fams.get("st_matrix"):
        epilogue.append(f"Measured anchor (sass census): {sass_fams['st_matrix']} "
                        "matrix store/load instruction(s) (STSM/LDSM family)")
    if sass_fams.get("st_global"):
        epilogue.append(f"Measured anchor (sass census): {sass_fams['st_global']} "
                        "global store instruction(s) in the shipped cubin")
    if epilogue:
        anchors["Epilogue path"] = epilogue

    if regime:
        anchors["Shape domain"] = [f"Measured anchor (published): {regime} — the shape domain "
                                   "the kernel was shipped for"]
    return anchors


_ANCHOR_RE = re.compile(r"^Measured anchor:")


def _fill_from_anchors(text: str, anchors: dict[str, list[str]]) -> tuple[str, int]:
    """Replace each anchored section's TODO stub with its anchor lines (the
    stub's hint stays as a comment for the human pass). Returns (text, how
    many sections gained anchors)."""
    filled = 0
    for name, lines in anchors.items():
        pat = re.compile(rf"(## {re.escape(name)}\n\n)TODO: [^\n]*\n", re.M)
        body = "\n".join(f"- {ln}" for ln in lines) + "\n"
        new, n = pat.subn(lambda m: m.group(1) + body, text)
        if n:
            text, filled = new, filled + 1
    return text, filled


def _fill_status(text: str, anchored: int, total: int) -> str:
    """The honest status line: filled only when EVERY section carries an
    anchor; partial anchors get the protective auto-anchored mark."""
    if anchored >= total:
        return text.replace(_SKELETON_MARK,
                            "Status: filled (auto from measured artifacts; machine-read "
                            "anchors — the decision narrative still awaits a harvest/human "
                            "pass)", 1)
    return text.replace(_SKELETON_MARK,
                        "Status: auto-anchored (from measured artifacts; sections still "
                        "TODO await a harvest/human pass)", 1)


def _run_from_artifacts(write: bool = False) -> int:
    """The WP18 fill pass: every skeleton whose artifacts landed gains its
    measured anchors. Filled / human files are never touched. Write gate =
    the skeleton pass's own discipline (WP23, 95-审查报告-WP18 F-3): the
    default is a dry run that lists what would be anchored; --write lands
    the diff."""
    kernels = _kernels()
    touched, full, partial = 0, 0, 0
    for k in kernels:
        rel = _priors_rel(k)
        dst = REPO / rel
        if not dst.is_file():
            continue
        text = dst.read_text(encoding="utf-8")
        if _FILLED_MARK in text or _ANCHORED_MARK in text:
            continue          # a filled / already-anchored file is protected
        arts = _load_artifacts(k)
        if not arts:
            continue
        anchors = _measured_anchors(arts)
        if not anchors:
            continue
        new, n = _fill_from_anchors(text, anchors)
        if n == 0:
            continue
        new = _fill_status(new, n, len(_SECTIONS))
        if write:
            dst.write_text(new, encoding="utf-8")
            touched += 1
            full += 1 if n >= len(_SECTIONS) else 0
            partial += 1 if n < len(_SECTIONS) else 0
            print(f"anchored {rel} ({n}/{len(_SECTIONS)} sections)"
                  + (" — FILLED" if n >= len(_SECTIONS) else " — partial"))
        else:
            print(f"would anchor {rel} ({n}/{len(_SECTIONS)} sections)"
                  + (" — FILLED" if n >= len(_SECTIONS) else " — partial"))
    if write:
        print(f"{len(kernels)} published kernel(s); {touched} skeleton(s) gained measured anchors "
              f"({full} now filled, {partial} partial)")
    else:
        print("dry run — pass --write to anchor the listed skeletons")
    return 0


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
                   help="write skeletons / land the --from-artifacts anchors "
                        "(default: dry run, list only)")
    p.add_argument("--from-artifacts", action="store_true", dest="from_artifacts",
                   help="fill skeleton sections from the measured artifacts under "
                        "priors/artifacts/ (scripts/collect_kernel_artifacts.py lands "
                        "them); default dry run, --write to anchor")
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
        globals()["ARTIFACTS"] = REPO / "priors" / "artifacts"

    if args.from_artifacts:
        return _run_from_artifacts(write=args.write)

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
