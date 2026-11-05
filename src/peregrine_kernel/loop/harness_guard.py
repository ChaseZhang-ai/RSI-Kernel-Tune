"""The evaluation harness is the card's, not the student's.

A job judges whatever ``gpu_entry.sh`` and ``eval/`` its tree carries, and the tree a
job runs is the student's lineage: every student turn ends in ``git add -A``, so an edit
to ``eval/bench.py`` (a looser tolerance, a correctness block that always passes, a
benchkit.py whose timer skips the kernel) used to reach the next eval and be judged
by itself. Nothing the launch preflights checked could see it -- they read the CARD's
copies, before the run.

Two layers, neither relying on the agent sandbox:

* **driver path check** (``[loop] harness_guard``, default ``"revert"``): a delivered
  commit whose diff from the tree the iteration started on touches a harness path is
  REVERTed before any job is bought, with the paths named in the record; and at the
  start of every iteration the lineage is put back to the launch tree's harness files
  (a probe commit left behind by an iteration that ended without a record, a file a
  teacher turn left in the shared workspace) in a commit of its own.
  ``"warn"`` logs instead of reverting; ``"off"`` disables both.
* **job overlay** (``RealBackend._with_job_env``): every pushed job tree carries the
  card's tracked harness files byte for byte (binary blobs and the sealed reference
  included), and harness files the card does not have are removed from it. The
  lineage and the recorded shas are unchanged -- the overlay is a child commit, the
  same channel ``[loop] job_env_files`` always was.

What counts as harness: ``HARNESS_PREFIXES`` below, minus the kernel file and
``notes/`` (the student's tree), plus the tree-root shadow modules -- a root
``sitecustomize.py`` or a root file named like a module the judge imports (``json.py``,
``os.py``, ``torch.py``, ...) would be imported INSTEAD of the real one, so it is
harness wherever the tree root is on sys.path ahead of it. ``window.toml`` and README
are export-ignored and the driver reads the card's copies, so they are not listed;
``.gitattributes`` is, because ``git archive`` reads its export rules from the
archived tree.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# A prefix ending in "/" names a directory; the others name a file.
HARNESS_PREFIXES = ("eval/", "gpu_entry.sh", "ref_bin/", ".gitattributes")
# The same set as git pathspecs (directories without the slash), plus the tree-root
# *.py pathspec: git diff / git ls-files match the wildcard at every depth, and
# ``is_harness_path`` decides which of the surfaced .py files count as harness.
ROOT_PY_PATHSPEC = "*.py"
HARNESS_PATHSPECS = tuple(p.rstrip("/") for p in HARNESS_PREFIXES) + (ROOT_PY_PATHSPEC,)
# git ls-tree matches literal paths and directory prefixes only -- no wildcards -- so
# the recursive harness listing takes the prefix pathspecs, and the tree-root .py
# files surface from the top-level (non-recursive) listing instead.
HARNESS_TREE_PATHSPECS = tuple(p.rstrip("/") for p in HARNESS_PREFIXES)

# The shadow-module names: what Python itself imports at start-up (the site hooks) and
# what the harness and the judge import by name -- the standard library (any top-level
# name of it) and the third-party modules the evaluation runs on. A tree-root file with
# one of these names is imported instead of the real module by every process whose
# sys.path reaches the tree root first (gpu_entry.sh puts $WORKSPACE on PYTHONPATH;
# ``python3 -c`` starts on the working directory), so it can silence the correctness
# monitor's loader or forge what the judge reads. Other root .py files stay the
# student's: the kernel may import root siblings by name, and the kernel file itself is
# excluded in ``is_harness_path``.
_SITE_HOOKS = ("sitecustomize", "usercustomize")
_JUDGE_IMPORTS = ("torch", "numpy", "triton", "cuda", "cutlass", "flashinfer")
_ROOT_PY_NAMES = frozenset(_SITE_HOOKS + _JUDGE_IMPORTS) | frozenset(sys.stdlib_module_names)


def _root_py_shadows(rel: str) -> bool:
    """``rel`` is a tree-root ``*.py`` whose name Python or the judge imports."""
    if not rel.endswith(".py") or "/" in rel:
        return False
    return rel[:-3] in _ROOT_PY_NAMES

GUARD_REVERT = "revert"
GUARD_WARN = "warn"
GUARD_OFF = "off"
GUARD_MODES = (GUARD_REVERT, GUARD_WARN, GUARD_OFF)


def resolve_guard_mode(cfg: dict) -> str:
    """[loop] harness_guard: "revert" (default) | "warn" | "off" (true / false read as
    "revert" / "off"). Anything else raises ValueError -- a typo must not read as off."""
    raw = (cfg.get("loop", {}) or {}).get("harness_guard", GUARD_REVERT)
    if isinstance(raw, bool):
        return GUARD_REVERT if raw else GUARD_OFF
    mode = str(raw).strip().lower()
    if mode not in GUARD_MODES:
        raise ValueError(f"[loop] harness_guard must be one of {', '.join(GUARD_MODES)}, "
                         f"got {raw!r}")
    return mode


def _root_py_hijack(stem: str) -> bool:
    """Whether a tree-root ``<stem>.py`` is executed by a harness process without the
    tree naming it: an interpreter startup hook, or a stdlib top-level name the
    workspace's PYTHONPATH placement shadows (BUG-SEED-07: a root sitecustomize.py or
    json.py hijacked every job; the monitor's sitecustomize even chain-imports it)."""
    return stem in ROOT_PY_HOOKS or stem in getattr(sys, "stdlib_module_names", ())


def is_harness_path(path: str, kernel_file: str = "") -> bool:
    """Whether ``path`` (tree-relative, posix) is card-owned evaluation harness."""
    p = str(path).strip()
    if p.startswith("./"):
        p = p[2:]
    if not p or p == str(kernel_file) or p.split("/")[0] == "notes":
        return False
    if _root_py_shadows(p):
        return True
    return any(p == pre.rstrip("/") or (pre.endswith("/") and p.startswith(pre))
               for pre in HARNESS_PREFIXES)


def harness_only(paths, kernel_file: str = "") -> list[str]:
    """The harness paths among ``paths``, sorted and de-duplicated."""
    return sorted({p for p in (paths or ()) if is_harness_path(p, kernel_file)})


def describe(paths, limit: int = 6) -> str:
    paths = list(paths)
    return ", ".join(paths[:limit]) + (f" (+{len(paths) - limit} more)" if len(paths) > limit else "")


def _git(repo: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text)


def is_git_repo(path: Path) -> bool:
    """``path`` is the top of its own git repository. A card that was never ``git init``-ed
    but sits inside another repository (operators/<op> under Peregrine-Kernel, gitignored
    there) is not: its tracked set would read as empty, and an empty card harness would
    strip gpu_entry.sh and eval/ from every job tree."""
    try:
        out = _git(Path(path), "rev-parse", "--show-toplevel")
    except OSError:
        return False
    if out.returncode != 0:
        return False
    try:
        return Path(out.stdout.strip()).resolve() == Path(path).resolve()
    except OSError:
        return False


def card_harness_files(card_dir: Path, kernel_file: str = "") -> dict[str, Path] | None:
    """The card's TRACKED harness files: {tree path: the card's working copy}. Tracked
    only -- a gitignored file next to them (a sealed reference's key) must never ride a
    job tree. None when the card is not a git repository (nothing to read the tracked
    set from) or when it tracks none -- a card without a judge is a listing gone wrong
    (nothing committed yet), and reading it as empty would strip gpu_entry.sh and eval/
    from every job tree; files tracked but missing on disk are left out."""
    card_dir = Path(card_dir)
    if not is_git_repo(card_dir):
        return None
    out = _git(card_dir, "ls-files", "-z", "--", *HARNESS_PATHSPECS)
    if out.returncode != 0:
        return None
    files = {}
    for rel in out.stdout.split("\0"):
        if rel and is_harness_path(rel, kernel_file) and (card_dir / rel).is_file():
            files[rel] = card_dir / rel
    return files or None


def tree_harness_files(repo: Path, ref: str, kernel_file: str = "") -> dict[str, tuple[str, str]] | None:
    """{path: (mode, blob sha)} of the harness files in commit ``ref`` of ``repo``."""
    repo = Path(repo)
    out = _git(repo, "ls-tree", "-r", "-z", ref, "--", *HARNESS_TREE_PATHSPECS)
    if out.returncode != 0:
        return None
    top = _git(repo, "ls-tree", "-z", ref)
    if top.returncode != 0:
        return None
    files = {}
    for entry in out.stdout.split("\0") + top.stdout.split("\0"):
        if not entry or "\t" not in entry:
            continue
        meta, rel = entry.split("\t", 1)
        mode, _kind, sha = meta.split()
        if mode.startswith("04"):        # a directory: the top-level listing names them
            continue
        if is_harness_path(rel, kernel_file):
            files[rel] = (mode, sha)
    return files


def blob_sha(repo: Path, data: bytes, write: bool = False) -> str:
    args = ["hash-object", "--stdin"] + (["-w"] if write else [])
    out = subprocess.run(["git", "-C", str(repo), *args], input=data, capture_output=True)
    if out.returncode != 0:
        raise OSError(f"git hash-object failed: {out.stderr.decode(errors='replace')[:200]}")
    return out.stdout.decode().strip()


def harness_drift(card_dir: Path, workspace: Path, ref: str = "HEAD",
                  kernel_file: str = "") -> list[str] | None:
    """How the harness in ``workspace``'s ``ref`` differs from the card's: ``~path``
    (content), ``+path`` (in the card, missing from the tree), ``-path`` (in the tree, not
    in the card). [] = identical; None = cannot say (the card or the tree is not a git
    repository)."""
    card = card_harness_files(card_dir, kernel_file)
    tree = tree_harness_files(workspace, ref, kernel_file)
    if card is None or tree is None:
        return None
    out = []
    for rel, src in sorted(card.items()):
        if rel not in tree:
            out.append(f"+{rel}")
            continue
        try:
            if blob_sha(workspace, src.read_bytes()) != tree[rel][1]:
                out.append(f"~{rel}")
        except OSError:
            out.append(f"~{rel}")
    out += [f"-{rel}" for rel in sorted(tree) if rel not in card]
    return out
