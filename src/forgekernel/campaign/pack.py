"""A track's upload archive: every problem of the track, one `submission.py` each.

A collection submission is scored as ``sum(each kernel's score) / the track's kernel
count``, and a missing kernel scores 0 (background §3.1) -- so the archive has to be
*complete*, and it must fail loudly rather than upload a track with holes. That is the
one rule this module is built around:

    a problem is packed only from (a) this campaign's own export, (b) the reference
    library's cover tree, or (c) the definition's own reference -- and (c) only when
    `--allow-reference-fallback` is passed, because a reference scores the 0.5-ish
    "reference implementation" row, not a kernel. Anything else fails the pack and names
    the problems it could not fill.

The archive is `out/<track>-<UTC stamp>.zip`, laid out ``<kernel_name>/submission.py``
with the archive's own directory names equal to the board's kernel names. Beside it,
``manifest.json`` records per problem which version went in, its local score, whether it
is a fallback, its licence provenance and any risk flag (an inline-CUDA conversion, a
candidate whose cold start was never measured). The self-check runs on the tree *before*
the zip and on the zip *after*: directory names, exactly one `submission.py` per
directory, a top-level ``run`` in each, the size limits (10 MB per file, 50 MB archive,
100 MB uncompressed), no symlinks, no path traversal.

Nothing here uploads. The printed reminder -- two submissions per track per UTC day and
a six-hour result delay -- is what `forge board submit-collection` would need to respect;
the suggested slots (00:30 and 12:30 UTC) put both of a day's chances to work.
"""

from __future__ import annotations

import ast
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from forgekernel.campaign import compose, solscore, sync
from forgekernel.campaign import targets as targets_mod
from forgekernel.campaign import tools as tools_mod

MAX_FILE_BYTES = 10 * 1024 * 1024          # one submission.py
MAX_ZIP_BYTES = 50 * 1024 * 1024           # the uploaded archive
MAX_UNPACKED_BYTES = 100 * 1024 * 1024     # everything inside it
PACK_DIR = "out"
UPLOAD_REMINDER = ("A track takes at most 2 collection submissions per UTC day and a "
                   "result appears about 6 h after upload; suggested slots 00:30 and "
                   "12:30 UTC put both of a day's chances to work.")


@dataclass
class PackProblem:
    kernel_id: int
    name: str
    subset: str
    source: str = ""                # our export | cover | reference
    kind: str = ""
    mean_s: float | None = None
    fallback: bool = False
    cold_start: dict = field(default_factory=dict)
    risk: list[str] = field(default_factory=list)
    licence: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class PackResult:
    track: str
    tree: Path
    zip_path: Path | None
    manifest: Path | None
    problems: list[PackProblem]
    expected: int = 0
    missing: list[str] = field(default_factory=list)
    packed: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and not self.missing and self.zip_path is not None


def track_problems(snap: targets_mod.Snapshot, track: str) -> list[tuple[int, str, str]]:
    """``[(kernel_id, name, subset)]`` for every problem of one track, in id order."""
    if track not in solscore.TRACK_RANGES:
        raise ValueError(f"unknown track {track!r}; one of {', '.join(solscore.TRACKS)}")
    lo, hi = solscore.TRACK_RANGES[track]
    return [(k, snap.name_of(k), solscore.TRACK_SUBSET[track])
            for k in range(lo, hi + 1) if k in snap.records]


def _export_path(work_dir: Path, op: str) -> Path:
    return work_dir / "exports" / op / "submission.py"


def _state_export(work_dir: Path, op: str) -> dict:
    p = work_dir / "state" / f"{op}.json"
    if not p.is_file():
        return {}
    try:
        return (json.loads(p.read_text(encoding="utf-8")).get("export") or {})
    except (OSError, ValueError):
        return {}


def _op_for(snap: targets_mod.Snapshot, name: str) -> str:
    same = [k for k in snap.kernel_ids if snap.name_of(k) == name]
    return name if len(same) == 1 else f"{name}__{solscore.track_of(same[0]) or 'x'}"


def reference_problem(dataset: str | Path, subset: str, name: str) -> dict | None:
    """The dataset's own definition as a problem dict (for the reference fallback)."""
    p = Path(dataset) / subset / name / "definition.json"
    if not p.is_file():
        return None
    defn = json.loads(p.read_text(encoding="utf-8"))
    if not defn.get("reference"):
        ref = Path(dataset) / subset / name / "reference.py"
        if ref.is_file():
            defn["reference"] = ref.read_text(encoding="utf-8")
    return {"definition": defn, "dps": True}


def collect(track: str, snap: targets_mod.Snapshot, *, work_dir: str | Path,
            dataset: str | Path, allow_reference_fallback: bool = False,
            cover_dir: str | Path | None = None) -> tuple[list[PackProblem], list[str]]:
    """Resolve each problem's source; ``(problems, missing names)``."""
    work = Path(work_dir)
    problems, missing = [], []
    cover = Path(cover_dir) if cover_dir else None
    for kid, name, subset in track_problems(snap, track):
        op = _op_for(snap, name)
        entry = PackProblem(kernel_id=kid, name=name, subset=subset)
        ours = _export_path(work, op)
        st = _state_export(work, op)
        if ours.is_file():
            entry.source, entry.kind = str(ours), "campaign"
            entry.mean_s = st.get("mean_s")
            entry.cold_start = st.get("cold_start") or {}
            entry.risk = list(st.get("risk") or [])
            entry.licence = dict(st.get("licence") or {})
            entry.notes = list(st.get("notes") or [])
        elif cover is not None and (cover / name / "submission.py").is_file():
            entry.source, entry.kind = str(cover / name / "submission.py"), "cover"
            entry.fallback = True
        elif allow_reference_fallback:
            prob = reference_problem(dataset, subset, name)
            if prob is None:
                missing.append(f"{name} (no definition under {Path(dataset) / subset})")
                continue
            entry.source, entry.kind = "", "reference"
            entry.fallback = True
            entry.notes = ["the definition's reference, generated by campaign"]
        else:
            missing.append(name)
            continue
        problems.append(entry)
    return problems, missing


def _reference_source(entry: PackProblem, dataset: str | Path) -> str:
    prob = reference_problem(dataset, entry.subset, entry.name)
    if prob is None:
        raise FileNotFoundError(f"{entry.name}: no definition under {Path(dataset) / entry.subset}")
    header = (f'"""{entry.name} -- campaign pack fallback (ForgeKernel).\n\n'
              f'Nothing else of this problem was produced: this is the board definition\'s\n'
              f'own eager reference wrapped as run(), the always-correct floor.\n"""\n')
    return compose.reference_only_submission(prob, header=header)


def build_tree(track: str, problems: list[PackProblem], *, out_dir: str | Path,
               dataset: str | Path) -> Path:
    """``out/<track>/<name>/submission.py`` for each problem (a fresh tree)."""
    tree = Path(out_dir) / track
    if tree.exists():
        shutil.rmtree(tree)
    for entry in problems:
        d = tree / entry.name
        d.mkdir(parents=True, exist_ok=True)
        if entry.source:
            (d / "submission.py").write_text(Path(entry.source).read_text(encoding="utf-8"),
                                             encoding="utf-8")
        else:
            (d / "submission.py").write_text(_reference_source(entry, dataset),
                                             encoding="utf-8")
    return tree


# --- the self-check ---------------------------------------------------------------------------

def _has_top_level_run(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run":
            return True
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "run"
                                                for t in node.targets):
            return True
    return False


def check_tree(tree: Path, expected: list[str]) -> list[str]:
    """Every rule the pack must satisfy before it becomes a zip. [] means clean."""
    found = []
    if not tree.is_dir():
        return [f"{tree}: no pack tree"]
    dirs = sorted(p.name for p in tree.iterdir() if p.is_dir())
    files = sorted(p.name for p in tree.iterdir() if p.is_file())
    if files:
        found.append(f"stray files at the archive root: {', '.join(files)}")
    extra = sorted(set(dirs) - set(expected))
    gone = sorted(set(expected) - set(dirs))
    if extra:
        found.append(f"directories that are not problems of the track: {', '.join(extra)}")
    if gone:
        found.append(f"missing problem directories: {', '.join(gone)}")
    total = 0
    for name in dirs:
        d = tree / name
        if d.is_symlink():
            found.append(f"{name}: a symlink")
            continue
        inner = sorted(p.name for p in d.iterdir())
        if inner != ["submission.py"]:
            found.append(f"{name}: must hold exactly submission.py, holds {inner}")
        f = d / "submission.py"
        if f.is_file():
            size = f.stat().st_size
            total += size
            if size > MAX_FILE_BYTES:
                found.append(f"{name}: submission.py is {size} B > {MAX_FILE_BYTES}")
            if not _has_top_level_run(f):
                found.append(f"{name}: submission.py has no top-level run")
    if total > MAX_UNPACKED_BYTES:
        found.append(f"uncompressed total {total} B > {MAX_UNPACKED_BYTES}")
    return found


def check_zip(zip_path: Path, expected: list[str]) -> list[str]:
    """The archive-level rules: entry names, no traversal, no symlinks, the size law."""
    found = []
    size = zip_path.stat().st_size
    if size > MAX_ZIP_BYTES:
        found.append(f"the archive is {size} B > {MAX_ZIP_BYTES}")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        total = sum(i.file_size for i in zf.infolist())
        for info in zf.infolist():
            n = info.filename
            if n.startswith("/") or ".." in Path(n).parts:
                found.append(f"{n}: path traversal")
            if info.external_attr >> 16 & 0o170000 == 0o120000:
                found.append(f"{n}: a symlink")
        want = {f"{name}/submission.py" for name in expected}
        got = {n for n in names if not n.endswith("/")}
        if got != want:
            if got - want:
                found.append(f"unexpected entries: {sorted(got - want)[:5]}")
            if want - got:
                found.append(f"missing entries: {sorted(want - got)[:5]}")
        if total > MAX_UNPACKED_BYTES:
            found.append(f"uncompressed total {total} B > {MAX_UNPACKED_BYTES}")
    return found


def make_zip(tree: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(tree.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(tree).as_posix())
    return zip_path


# --- pack ------------------------------------------------------------------------------------

def pack(track: str, snap: targets_mod.Snapshot, *, work_dir: str | Path,
         out_dir: str | Path, dataset: str | Path,
         tools: tools_mod.Tools | None = None, cover_dir: str | Path | None = None,
         allow_reference_fallback: bool = False, ledger_path: str | Path | None = None
         ) -> PackResult:
    """Build, check and zip one track; ``ok`` is False when a problem could not be filled."""
    work = Path(work_dir)
    problems, missing = collect(track, snap, work_dir=work, dataset=dataset,
                               allow_reference_fallback=allow_reference_fallback,
                               cover_dir=cover_dir)
    tree = build_tree(track, problems, out_dir=out_dir, dataset=dataset)
    expected = [name for _, name, _ in track_problems(snap, track)]
    found = check_tree(tree, expected)
    res = PackResult(track=track, tree=tree, zip_path=None, manifest=None,
                     problems=problems, expected=len(expected), missing=missing,
                     packed=[p.name for p in problems])
    if missing:
        res.error = (f"{len(missing)} problem(s) of {track} have no source and no "
                     f"cover: {', '.join(missing[:10])}"
                     + (" ..." if len(missing) > 10 else "")
                     + "  (a track with a missing kernel keeps 0 for it)")
        return res
    if found:
        res.error = "self-check failed: " + "; ".join(found[:8])
        return res
    if tools is not None and getattr(tools, "collection_validate", None) is not None:
        try:
            verdict = tools.collection_validate(tree)
        except Exception as e:                                  # noqa: BLE001
            verdict = {"error": f"{type(e).__name__}: {e}"}
        if isinstance(verdict, dict) and verdict.get("problems"):
            res.error = "the board task's validator refused the tree: " + "; ".join(
                str(x) for x in verdict["problems"][:6])
            return res
        if res.problems:
            res.problems[0].notes.append(f"board validator: {verdict}")
    stamp = compose.now_stamp()
    zip_path = Path(out_dir) / f"{track}-{stamp}.zip"
    make_zip(tree, zip_path)
    zfound = check_zip(zip_path, expected)
    if zfound:
        res.error = "the archive's own check failed: " + "; ".join(zfound[:8])
        return res
    manifest = Path(out_dir) / f"manifest-{track}-{stamp}.json"
    doc = {"schema": 1, "track": track, "created": stamp, "archive": zip_path.name,
           "problems_total": len(expected), "problems_packed": len(problems),
           "fallbacks": sum(1 for p in problems if p.fallback),
           "mean_s_local": solscore.mean_s_of_scores(p.mean_s for p in problems),
           "reminder": UPLOAD_REMINDER, "suggested_utc": ["00:30", "12:30"],
           "problems": [{"kernel_id": p.kernel_id, "name": p.name, "source": p.source,
                         "kind": p.kind, "mean_s": p.mean_s, "fallback": p.fallback,
                         "cold_start": p.cold_start, "risk": p.risk,
                         "licence": p.licence, "notes": p.notes} for p in problems]}
    manifest.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    res.zip_path, res.manifest = zip_path, manifest
    if ledger_path is not None:
        sync.Ledger(ledger_path).append({
            "event": "pack", "source": sync.DEV, "track": track,
            "note": f"{zip_path.name}: {len(problems)} kernel(s), "
                    f"{doc['fallbacks']} fallback(s)",
            "mean_s": doc["mean_s_local"]})
    return res


def summary(res: PackResult) -> str:
    lines = [f"pack {res.track}: {len(res.problems)}/{res.expected} kernel(s), "
             f"{sum(1 for p in res.problems if p.fallback)} fallback"]
    if res.error:
        lines.append(f"  FAILED: {res.error}")
    if res.zip_path:
        lines.append(f"  archive: {res.zip_path} ({res.zip_path.stat().st_size} B)")
        lines.append(f"  manifest: {res.manifest}")
    lines.append(f"  {UPLOAD_REMINDER}")
    return "\n".join(lines)
