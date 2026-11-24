"""Seeding ``prior/``: the existing implementations a run starts from.

Whatever ``--init-kernel`` / ``--prior`` name is copied into the card's ``prior/``
directory, so the paths land INSIDE the student's workspace with the card's initial
commit: the file tools and the Bash sandbox read the workspace by default, the policy
denies nothing under it, and -- because the copy is part of the initial commit rather
than something the student wrote -- it is not part of the student's diff. The student's
own changes stay one file, whatever the starting point was.

``prior/SOURCES.json`` records where every file came from (origin, sha256, license), so
the manifest ``forge prior build`` writes is reproducible from the card alone and a run
carried to another machine still knows what it started from. The license is detected
mechanically -- from a LICENSE / COPYING / NOTICE file next to the source -- because the
submission rules only allow code under Apache-2.0 / MIT / BSD to be incorporated
(00-background §5, §8); "unknown" is not "allowed".

Nothing here is agent-facing prose: the manifest builder reads this record and renders it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

PRIOR_DIRNAME = "prior"
SOURCES_NAME = "SOURCES.json"

# The licenses a submission may incorporate (00-background §8). A separate licence
# spelling qualifies through _LICENSE_PATTERNS below, never by fuzzy matching.
ALLOWED_LICENSES = ("Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause")

_LICENSE_FILES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "LICENCE", "LICENCE.txt",
                  "COPYING", "COPYING.txt", "NOTICE", "NOTICE.txt", "LICENSE-APACHE",
                  "LICENSE-MIT")
# Ordered: the first pattern that matches decides. Longest / most specific first, so a
# repository holding both an MIT and a BSD notice is not read as whichever came first.
_LICENSE_PATTERNS = (
    ("Apache-2.0", ("apache license", "version 2.0, january 2004")),
    ("Apache-2.0", ("apache license", "version 2.0")),
    ("BSD-3-Clause", ("bsd 3-clause",)),
    ("BSD-2-Clause", ("bsd 2-clause",)),
    ("BSD-3-Clause", ("redistribution and use in source and binary forms",
                      "neither the name")),
    ("BSD-2-Clause", ("redistribution and use in source and binary forms",)),
    ("MIT", ("mit license",)),
    ("MIT", ("permission is hereby granted, free of charge",)),
    ("GPL-3.0", ("gnu general public license", "version 3")),
    ("GPL-2.0", ("gnu general public license", "version 2")),
    ("LGPL", ("lesser general public license",)),
    ("MPL-2.0", ("mozilla public license",)),
    ("AGPL-3.0", ("affero general public license",)),
    ("Unlicense", ("this is free and unencumbered software released into the public domain",)),
    ("CC0-1.0", ("cc0 1.0 universal",)),
)


def licenses_allow_reuse(license_id: str | None) -> bool:
    """May code under this licence be incorporated into a submission?"""
    return bool(license_id) and license_id in ALLOWED_LICENSES


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_license(source: Path, *, levels: int = 3) -> str:
    """The licence of ``source`` (a file or directory), from the nearest licence file at
    it or up to ``levels`` parent directories up. "unknown" when none is found."""
    start = source if source.is_dir() else source.parent
    seen = 0
    for directory in (start, *start.parents):
        if seen > levels:
            break
        seen += 1
        for name in _LICENSE_FILES:
            candidate = directory / name
            if candidate.is_file():
                found = _license_of_text(_read_text(candidate))
                if found:
                    return found
    return "unknown"


def _read_text(path: Path, limit: int = 200_000) -> str:
    try:
        with open(path, "rb") as fh:
            return fh.read(limit).decode("utf-8", errors="replace").lower()
    except OSError:
        return ""


def _license_of_text(text: str) -> str | None:
    for license_id, needles in _LICENSE_PATTERNS:
        if all(n in text for n in needles):
            return license_id
    return None


def _dest_name(path: Path, taken: set[str]) -> str:
    name = path.name or path.parent.name
    if name not in taken:
        return name
    stem, suffix = Path(name).stem, Path(name).suffix
    for i in range(2, 100):
        candidate = f"{stem}-{i}{suffix}"
        if candidate not in taken:
            return candidate
    raise ValueError(f"{path}: too many sources named {name}")


def seed_prior(op_dir: str | Path, *, init_kernel: tuple[bytes, Path] | None = None,
               priors: list[tuple[bytes, Path]] | None = None) -> list[Path]:
    """Copy ``init_kernel`` and every ``priors`` entry into ``op_dir/prior/``.

    Both arguments are ``(content, origin_path)`` pairs: ``content`` is the file's bytes as
    they are written (the init kernel's bytes are read once and written both as the kernel
    file and here, so the two are byte-identical), ``origin_path`` is where it came from
    (recorded, and the source of the licence lookup). A directory is copied as a tree.
    Returns the files written."""
    op_dir = Path(op_dir)
    entries: list[dict] = []
    written: list[Path] = []
    items: list[tuple[str | None, bytes, Path]] = []
    if init_kernel is not None:
        items.append(("init-kernel", init_kernel[0], init_kernel[1]))
    for content, origin in priors or []:
        items.append((None, content, origin))
    if not items:
        return written
    prior = op_dir / PRIOR_DIRNAME
    prior.mkdir(parents=True, exist_ok=True)
    taken = {p.name for p in prior.iterdir()}
    for role, content, origin in items:
        origin = Path(origin)
        if origin.is_dir():
            dest = prior / _dest_name(origin, taken)
            shutil.copytree(origin, dest, symlinks=False,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
            files = sorted(p for p in dest.rglob("*") if p.is_file())
            taken.add(dest.name)
            entries.append({"path": dest.name, "kind": "tree", "origin": str(origin),
                            "license": detect_license(origin), "role": role,
                            "files": [{"path": str(f.relative_to(prior)),
                                       "sha256": sha256(f),
                                       "bytes": f.stat().st_size} for f in files]})
            written += files
        else:
            dest = prior / _dest_name(origin, taken)
            dest.write_bytes(content)
            taken.add(dest.name)
            entries.append({"path": dest.name, "kind": "file", "origin": str(origin),
                            "license": detect_license(origin), "role": role,
                            "files": [{"path": dest.name, "sha256": sha256(dest),
                                       "bytes": dest.stat().st_size}]})
            written.append(dest)
    (prior / SOURCES_NAME).write_text(
        json.dumps({"schema": 1, "sources": entries}, indent=2) + "\n", encoding="utf-8")
    written.append(prior / SOURCES_NAME)
    return written


def read_sources(op_dir: str | Path) -> list[dict]:
    """The records ``seed_prior`` wrote, or [] when the card has no prior/."""
    path = Path(op_dir) / PRIOR_DIRNAME / SOURCES_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return list(data.get("sources") or [])


def prior_files(op_dir: str | Path) -> list[Path]:
    """Every file under ``prior/`` except the record itself, in a stable order."""
    prior = Path(op_dir) / PRIOR_DIRNAME
    if not prior.is_dir():
        return []
    return sorted(p for p in prior.rglob("*")
                  if p.is_file() and p.name != SOURCES_NAME and "__pycache__" not in p.parts)


def record_for(op_dir: str | Path, rel: str) -> dict:
    """The SOURCES.json record whose files include ``rel`` (relative to prior/)."""
    for entry in read_sources(op_dir):
        for f in entry.get("files") or []:
            if f.get("path") == rel:
                return entry
    return {}
