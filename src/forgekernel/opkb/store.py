"""Where the records live: ``op_kb/``, its index, its README, its quarantine.

    op_kb/
      README.md            what this directory is (written once, by the first harvest)
      index.json           every record's summary, rebuilt by harvest and lint
      <family>/<id>.json   one record per run (forgekernel.opkb.record)
      _quarantine/         records that failed lint, moved here with their file name

The directory is a project asset: it is **tracked by git** (the user made the lesson base
part of the repository). Run directories are not -- ``.artifacts/`` stays ignored -- so what
is committed is the extracted records only, never a run's tree.

The root is ``--root``, else ``FK_OPKB_ROOT``, else ``<repository>/op_kb``. ``_quarantine``
and every other ``_``-prefixed directory are not record directories: scanning skips them, so
a quarantined record is invisible to query / render / stats until a human fixes and moves it
back.

::

    from forgekernel.opkb import store
    root = store.root_dir()
    path = store.write_record(root, record)
    store.load_records(root)          # every record that parses, quarantine excluded
    store.rebuild_index(root)         # index.json from the records on disk
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from forgekernel.paths import repo_root

ROOT_ENV = "FK_OPKB_ROOT"
ROOT_NAME = "op_kb"
INDEX_NAME = "index.json"
QUARANTINE_NAME = "_quarantine"
README_NAME = "README.md"
SCHEMA = 1

README_TEXT = """\
# op_kb -- operator-family lessons

This directory holds one JSON record per finished run: the objective facts the run left on
disk, extracted mechanically (no agent wrote any of it) and filed under the run's operator
family. `forge lessons render tasks/<op> --out lessons.md` turns the records of one family
into the file a run's teacher reads as `{LESSONS_PATH}`;

    FK_LESSONS_PATH=<abs path>/lessons.md

It is part of the project: versions of these files are tracked, so what one run measured on
an operator is available to the next run of the same family. Records are produced by
`forge lessons harvest <run dir>` and checked by `forge lessons lint`; a record that fails
lint is moved to `_quarantine/` and stays there until a human fixes it. The format and the
extraction rules are documented in `docs/opkb.md`.

`index.json` is rebuilt by harvest and lint; it is a convenience view, the per-record files
are the source of truth.
"""


def root_dir(root: str | Path | None = None) -> Path:
    """The lesson base's root: ``root``, else ``FK_OPKB_ROOT``, else ``<repo>/op_kb``."""
    if root is not None:
        return Path(root).expanduser().resolve()
    env = os.environ.get(ROOT_ENV, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / ROOT_NAME


def index_path(root: Path) -> Path:
    return root / INDEX_NAME


def quarantine_dir(root: Path) -> Path:
    return root / QUARANTINE_NAME


def record_path(root: Path, family: str, rid: str) -> Path:
    return root / family / f"{rid}.json"


def is_record_dir(path: Path) -> bool:
    return path.is_dir() and not path.name.startswith("_")


def write_record(root: str | Path, record: dict) -> Path:
    """Write ``record`` to ``<root>/<family>/<id>.json`` atomically; touch README and index.

    Harvesting the same run twice writes the same path with the same bytes (the id is
    derived from the problem and the run's own id), so a re-harvest is idempotent."""
    base = Path(root)
    path = record_path(base, str(record["family"]), str(record["id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_readme(base)
    _atomic_write(path, json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    rebuild_index(base, write=True)
    return path


def write_readme(root: str | Path) -> Path:
    """The directory's own explanation; written only when it is not there yet."""
    base = Path(root)
    path = base / README_NAME
    if not path.is_file():
        base.mkdir(parents=True, exist_ok=True)
        path.write_text(README_TEXT, encoding="utf-8")
    return path


def load_records(root: str | Path) -> list[dict]:
    """Every record under ``root`` (family directories only; ``_quarantine`` excluded).

    A file that is not readable JSON, or whose JSON is not an object, is skipped -- lint is
    where a broken record is reported; a reader must not die on one."""
    base = Path(root)
    out: list[dict] = []
    if not base.is_dir():
        return out
    for family_dir in sorted(p for p in base.iterdir() if is_record_dir(p)):
        for path in sorted(family_dir.glob("*.json")):
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def load_records_with_paths(root: str | Path) -> list[tuple[Path, dict]]:
    """``[(path, record)]`` for every readable record -- lint and stats want the path too."""
    base = Path(root)
    out: list[tuple[Path, dict]] = []
    if not base.is_dir():
        return out
    for family_dir in sorted(p for p in base.iterdir() if is_record_dir(p)):
        for path in sorted(family_dir.glob("*.json")):
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(obj, dict):
                out.append((path, obj))
    return out


def index_entry(record: dict, path: Path, root: Path) -> dict:
    """The summary one record contributes to ``index.json`` (enough to list, not to render)."""
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = path.name
    problem = record.get("problem") if isinstance(record.get("problem"), dict) else {}
    outcome = record.get("outcome") if isinstance(record.get("outcome"), dict) else {}
    entry = {
        "id": record.get("id"),
        "family": record.get("family"),
        "created": record.get("created"),
        "problem": {k: problem.get(k) for k in ("suite", "name", "definition", "arch",
                                                "language", "workloads") if k in problem},
        "outcome": {k: outcome.get(k) for k in ("metric_kind", "value", "stop")
                    if k in outcome},
        "path": rel,
    }
    return entry


def rebuild_index(root: str | Path, *, write: bool = True, now: str | None = None) -> dict:
    """``index.json`` from the records on disk: one summary row per record, in path order."""
    base = Path(root)
    entries = [index_entry(rec, path, base) for path, rec in load_records_with_paths(base)]
    entries.sort(key=lambda e: (str(e.get("family")), str(e.get("id")), str(e.get("path"))))
    index = {
        "schema": SCHEMA,
        "updated": now or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "count": len(entries),
        "records": entries,
    }
    if write:
        base.mkdir(parents=True, exist_ok=True)
        _atomic_write(base / INDEX_NAME, json.dumps(index, indent=1, ensure_ascii=False) + "\n")
    return index


def quarantine(root: str | Path, path: Path, reason: str) -> Path:
    """Move ``path`` into ``_quarantine`` keeping its file name; return where it went.

    Nothing is deleted: a record that fails lint is evidence of what a run wrote, and a human
    decides whether to fix it and move it back. A name already taken in quarantine gets a
    numeric suffix (``x.json`` -> ``x.2.json``) rather than overwriting the earlier one."""
    base = Path(root)
    dest_dir = quarantine_dir(base)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    n = 2
    while dest.exists():
        dest = dest_dir / f"{path.stem}.{n}{path.suffix}"
        n += 1
    path.replace(dest)
    try:
        (dest_dir / f"{dest.stem}.reason.txt").write_text(
            f"moved from {path.relative_to(base) if path.is_relative_to(base) else path}\n"
            f"reason(s):\n" + "\n".join(f"- {r}" for r in reason.splitlines()) + "\n",
            encoding="utf-8")
    except OSError:
        pass
    return dest


def write_quarantined(root: str | Path, record: dict, reason: str) -> Path:
    """Keep a record that failed lint out of the base: write it straight to ``_quarantine``.

    Harvest uses this when the record it just extracted does not pass the lint (a path in an
    error message, say). The facts are preserved for a human to repair, and no reader ever
    sees the file."""
    base = Path(root)
    dest_dir = quarantine_dir(base)
    dest_dir.mkdir(parents=True, exist_ok=True)
    rid = str(record.get("id") or "record")
    dest = dest_dir / f"{rid}.json"
    n = 2
    while dest.exists():
        dest = dest_dir / f"{rid}.{n}.json"
        n += 1
    _atomic_write(dest, json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    try:
        (dest_dir / f"{dest.stem}.reason.txt").write_text(
            "harvest refused to file this record\nreason(s):\n"
            + "\n".join(f"- {r}" for r in reason.splitlines()) + "\n", encoding="utf-8")
    except OSError:
        pass
    return dest


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
