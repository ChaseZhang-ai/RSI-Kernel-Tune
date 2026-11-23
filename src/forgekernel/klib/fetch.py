"""``forge lib fetch <name>``: clone an external kernel library at a pinned commit (opt-in).

The larger public kernel libraries (FlashInfer, FlashMLA, DeepGEMM, FlagGems, vLLM, SGLang, …)
are **not** cloned by ``forge lib build``. ``fetch`` takes one ``kind = "git"`` source of
``kernel_library/sources.toml``, shallow-clones it into
``kernel_library/_sources/<name>/`` (never committed — see ``.gitignore``), checks out the
pinned commit, and registers what it found: the commit actually checked out and the license
its ``LICENSE`` file states.

Only cloning and registration. Normalization rules are per library and are later work: a
fetched tree is a source of raw material, not something the index already knows.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from forgekernel.klib import ingest
from forgekernel.klib import normalize as nz

SOURCES_SUBDIR = "_sources"
REGISTRATION_FILE = ".klib-fetch.json"
LICENSE_NAMES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "LICENSE-APACHE", "COPYING")


class FetchError(RuntimeError):
    """The fetch could not be made (say why; nothing half-cloned is left behind)."""


@dataclass
class FetchResult:
    name: str
    url: str
    requested_commit: str
    commit: str = ""
    path: str = ""
    license: str = ""
    license_expected: str = ""
    license_file: str = ""
    files: int = 0
    already: bool = False
    pins: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "url": self.url, "requested_commit": self.requested_commit,
            "commit": self.commit, "path": self.path, "license": self.license,
            "license_expected": self.license_expected, "license_file": self.license_file,
            "files": self.files, "already": self.already, "pins": self.pins,
            "notes": self.notes,
        }

    def summary(self) -> str:
        where = "already present" if self.already else "fetched"
        return (f"[lib fetch] {self.name}: {where} at {self.path}\n"
                f"  commit: {self.commit or self.requested_commit}\n"
                f"  license: {self.license or 'not found'}"
                + (f" (expected {self.license_expected})" if self.license_expected else "")
                + f"\n  files: {self.files}"
                + ("\n  " + "\n  ".join(self.notes) if self.notes else ""))


def _git(args: list[str], cwd: Path, *, timeout: int = 900) -> str:
    proc = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True,
                          timeout=timeout, check=False)
    if proc.returncode != 0:
        raise FetchError(f"git {' '.join(args)} failed ({proc.returncode}): "
                         f"{proc.stderr.strip()[:400]}")
    return proc.stdout.strip()


def _find_license(root: Path) -> tuple[str, str]:
    """``(license, path)`` from the tree's own LICENSE file / SPDX-tagged headers."""
    for name in LICENSE_NAMES:
        candidate = root / name
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8", errors="replace")
            detected = nz.license_of_text(text, max_lines=120)
            return detected or "unknown", name
    return "", ""


def fetch(name: str, library: str | Path | None = None, *,
          sources_path: str | Path | None = None, allow_unpinned: bool = False,
          force: bool = False, into: str | Path | None = None) -> FetchResult:
    """Clone the named ``kind = "git"`` source at its pinned commit into ``_sources/<name>``."""
    library = ingest.library_root(library)
    sources_path = Path(sources_path) if sources_path is not None else library / ingest.SOURCES_FILE
    config = ingest.load_sources(sources_path)
    spec = config.by_name(name)
    if spec is None:
        raise FetchError(f"{name!r} is not a source in {sources_path}")
    if spec.kind != "git" or not spec.url:
        raise FetchError(f"{name!r} is not a git source (kind={spec.kind!r})")
    target = Path(into) if into else library / SOURCES_SUBDIR / name
    if target.exists() and any(target.iterdir()) and not force:
        registration = target / REGISTRATION_FILE
        result = FetchResult(name=name, url=spec.url, requested_commit=spec.commit,
                             path=str(target), already=True,
                             license_expected=spec.license,
                             pins=[{"name": spec.name, "commit": spec.commit,
                                    "license": spec.license}])
        if registration.is_file():
            data = json.loads(registration.read_text(encoding="utf-8"))
            result.commit = str(data.get("commit") or "")
            result.license = str(data.get("license") or "")
            result.license_file = str(data.get("license_file") or "")
            result.files = int(data.get("files") or 0)
        result.notes.append("pass --force to re-clone")
        return result    # already fetched: a resume, not a re-download

    if not spec.commit and not allow_unpinned:
        raise FetchError(
            f"{name!r} has no pinned commit in sources.toml. Pin one (the upstream SHA) or "
            f"pass --allow-unpinned to take the default branch — the default is unpinned "
            f"*by design*: a moving target is not a reproducible reference.")
    ref = spec.commit or "HEAD"
    target.parent.mkdir(parents=True, exist_ok=True)
    fresh = not (target / ".git").exists()
    try:
        if fresh:
            target.mkdir(parents=True, exist_ok=True)
            _git(["init", "-q"], target)
            _git(["remote", "add", "origin", spec.url], target)
        _git(["fetch", "-q", "--depth", "1", "origin", ref], target)
        _git(["checkout", "-q", "--detach", "FETCH_HEAD"], target)
    except (FetchError, OSError) as exc:
        raise FetchError(f"{name!r}: {exc}") from exc
    commit = _git(["rev-parse", "HEAD"], target)
    license_name, license_file = _find_license(target)
    files = sum(1 for p in target.rglob("*") if p.is_file() and ".git" not in p.parts)
    result = FetchResult(name=name, url=spec.url, requested_commit=spec.commit, commit=commit,
                         path=str(target), license=license_name,
                         license_expected=spec.license, license_file=license_file, files=files,
                         pins=[{"name": spec.name, "commit": spec.commit, "license": spec.license}])
    if spec.license and license_name and not _same_license(spec.license, license_name):
        result.notes.append(f"license mismatch: sources.toml says {spec.license!r}, the tree's "
                            f"{license_file} says {license_name!r}")
    elif spec.license and not license_name:
        result.notes.append("no LICENSE file found in the tree")
    (target / REGISTRATION_FILE).write_text(
        json.dumps({**result.to_dict(),
                    "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def _same_license(expected: str, found: str) -> bool:
    return re.sub(r"[^a-z0-9]", "", expected.lower()) == re.sub(r"[^a-z0-9]", "", found.lower())


def fetchable(library: str | Path | None = None, *,
              sources_path: str | Path | None = None) -> list[dict]:
    """The registered git sources with whether they are pinned and already fetched."""
    library = ingest.library_root(library)
    sources_path = Path(sources_path) if sources_path is not None else library / ingest.SOURCES_FILE
    config = ingest.load_sources(sources_path)
    out = []
    for spec in config.sources:
        if spec.kind != "git":
            continue
        present = (library / SOURCES_SUBDIR / spec.name).exists()
        out.append({"name": spec.name, "url": spec.url, "commit": spec.commit,
                    "pinned": bool(spec.commit), "license": spec.license,
                    "fetched": present, "notes": spec.notes})
    return out


def put_on_path(library: str | Path | None = None) -> Path:
    """The ``_sources`` directory (created on demand) — for a caller that wants to look."""
    path = ingest.library_root(library) / SOURCES_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path
