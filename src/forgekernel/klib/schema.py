"""klib: the reference library's index -- what an entry is and how it is checked.

The reference library (``kernel_library/``) holds kernels that already exist for problems of
a board, each normalized to ONE file ``kernel.py`` with a top-level ``run`` taking positional
arguments. Its index lists them; anything that looks for a starting point -- a campaign, an
evolve lineage, ``forge lib adapt`` -- reads the index and never the sources directly.

::

    kernel_library/
      index.json              # {"version": 1, "entries": [ <entry>, ... ]}   -- this schema
      kernels/<id>/kernel.py  # the single file
      kernels/<id>/meta.json  # where it came from (the index is rebuilt from these)

    {
      "id": "an-entry",
      "problem_keys": ["018_mla_paged_decode_h16_ckv512_kpe64_ps1", "attention"],
      "family": "attention",
      "path": "kernels/an-entry/kernel.py",
      "language": "cuda-inline",
      "entry": "run",
      "source": {"repo": "<repository>", "commit": "<commit>", "path": "<file in it>"},
      "license": "Apache-2.0",
      "reported": {"score": 0.9, "board": "<which board>", "url": "<where it was read>"},
      "verified_local": null,
      "usable_in_submission": true
    }

    from forgekernel.klib.schema import KernelEntry, load_index, save_index, submittable, validate
    problems = validate(entry)                  # [] when the entry follows the schema
    entries = load_index(path)                  # raises SchemaError naming every problem

``problem_keys`` are what the entry can be used for: exact problem names of the board and
family keys (``forgekernel.family``). ``reported`` is what a board said about it and
``verified_local`` what a ``Bench`` measured here; each is ``null`` when nothing is known.
``reported`` and ``source`` are copied from where the kernel came, never invented.

**Only an entry whose ``license`` is Apache-2.0, MIT or BSD may be merged into a submission**
(:func:`submittable`). An entry under any other license, or under none that could be read, may
stay in the library to be read and adapted from, provided it says ``usable_in_submission:
false`` -- ``validate`` refuses it otherwise, so an unusable entry cannot slip in unnoticed.

This module is the schema only: building the index from sources (``forge lib build``), finding
and scoring entries are task seedlib's.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath

INDEX_VERSION = 1

# The languages a normalized kernel is written in.
LANGUAGES = ("triton", "cutedsl", "pytorch", "cuda-inline")
# Every kernel is called the same way: `run(*args)`, positional, outputs written in place.
ENTRY = "run"
# The licenses whose code may be merged into a submission (spelled the SPDX way, in any case).
ALLOWED_LICENSES = ("Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause")

FIELDS = ("id", "problem_keys", "family", "path", "language", "entry", "source", "license",
          "reported", "verified_local", "usable_in_submission")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class SchemaError(ValueError):
    """An index or an entry does not follow this schema; ``problems`` lists every reason found
    (all of them, so one edit can fix them all)."""

    def __init__(self, where, problems: list[str]):
        self.where = str(where)
        self.problems = list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"{self.where}: not a valid kernel library index\n{lines}")


@dataclass
class KernelEntry:
    """One kernel of the library, as the index lists it (the module docstring has each field).
    ``usable_in_submission`` is ``None`` while the entry does not say."""
    id: str
    problem_keys: list[str]
    family: str
    path: str
    language: str
    source: dict
    license: str
    entry: str = ENTRY
    reported: dict | None = None
    verified_local: dict | None = None
    usable_in_submission: bool | None = None

    @classmethod
    def from_dict(cls, data: Mapping) -> KernelEntry:
        """The entry of a mapping as read from the index (keys outside the schema are dropped;
        ``validate`` is what reports them)."""
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})

    def to_dict(self) -> dict:
        """The entry as the index writes it: what is not known (``None``) is left out."""
        return {f.name: v for f in fields(self) if (v := getattr(self, f.name)) is not None}


def license_allowed(license: str) -> bool:
    """Whether ``license`` is one of ``ALLOWED_LICENSES`` (case, surrounding space and ``_`` for
    ``-`` do not matter; any other spelling is not allowed)."""
    name = str(license).strip().casefold().replace("_", "-")
    name = re.sub(r"\s+", "-", name)
    return name in {x.casefold() for x in ALLOWED_LICENSES}


def submittable(entry) -> bool:
    """Whether the entry's kernel may be merged into a submission: its license is one of the
    allowed ones and the entry has not ruled itself out (``usable_in_submission: false``).
    Anything that is not an entry is not submittable."""
    if isinstance(entry, Mapping):
        data = entry
    elif dataclasses.is_dataclass(entry) and not isinstance(entry, type):
        data = asdict(entry)
    else:
        return False
    lic = data.get("license")
    return isinstance(lic, str) and license_allowed(lic) and data.get("usable_in_submission") is not False


def validate(entry) -> list[str]:
    """Why ``entry`` (a mapping as read from the index, or a ``KernelEntry``) does not follow the
    schema: every reason found; ``[]`` when it does."""
    data = entry if isinstance(entry, Mapping) else asdict(entry)
    out: list[str] = []
    unknown = sorted(repr(k) for k in data if k not in FIELDS)
    if unknown:
        out.append(f"unknown key(s) {', '.join(unknown)}; the schema has {', '.join(FIELDS)}")

    def text(key: str) -> str | None:
        v = data.get(key)
        return v if isinstance(v, str) and v.strip() else None

    for key in ("id", "family", "path", "language", "entry", "license"):
        if text(key) is None:
            out.append(f"{key} must be a non-empty string, got {data.get(key)!r}")
    if (e_id := text("id")) and not _ID_RE.match(e_id):
        out.append(f"id {e_id!r} is not an identifier (letters, digits, '_', '.', '-'; not "
                   f"starting with a separator)")
    keys = data.get("problem_keys")
    if not isinstance(keys, (list, tuple)) or not keys:
        out.append("problem_keys must be a non-empty list of problem names and family keys, got "
                   f"{keys!r}")
        keys = []
    else:
        out += [f"problem_keys: {k!r} is not a name" for k in keys
                if not (isinstance(k, str) and k.strip())]
    if (fam := text("family")):
        from forgekernel.family import families
        known = families()
        if fam not in known:
            out.append(f"family {fam!r} is not one family.py knows ({', '.join(known)})")
        elif keys and fam not in keys:
            out.append(f"family {fam!r} is not among problem_keys {list(keys)!r}: an entry is "
                       f"found through its family, so the family must be one of its keys")
    if (rel := text("path")):
        p = PurePosixPath(rel)
        if p.is_absolute() or ".." in p.parts or "\\" in rel:
            out.append(f"path {rel!r} must be relative to kernel_library/ and stay inside it")
        elif p.suffix != ".py":
            out.append(f"path {rel!r} must name a Python file (a kernel is one kernel.py)")
    if (lang := text("language")) and lang not in LANGUAGES:
        out.append(f"language must be one of {list(LANGUAGES)}, got {lang!r}")
    if (fn := text("entry")) and fn != ENTRY:
        out.append(f"entry must be {ENTRY!r} (positional arguments, outputs written in place), "
                   f"got {fn!r}")
    source = data.get("source")
    if not isinstance(source, Mapping):
        out.append(f"source must be a table with repo, commit and path, got {source!r}")
    else:
        out += [f"source.{k} must be a non-empty string, got {source.get(k)!r}"
                for k in ("repo", "commit", "path")
                if not (isinstance(source.get(k), str) and source.get(k).strip())]
    for key in ("reported", "verified_local"):
        v = data.get(key)
        if v is not None and not isinstance(v, Mapping):
            out.append(f"{key} must be a table of what is known, or null when nothing is, got "
                       f"{type(v).__name__}")
    usable = data.get("usable_in_submission")
    if usable is not None and not isinstance(usable, bool):
        out.append(f"usable_in_submission must be true or false, got {usable!r}")
    if (lic := text("license")) and not license_allowed(lic) and usable is not False:
        out.append(f"license {lic!r} is not one of {', '.join(ALLOWED_LICENSES)}: the kernel may "
                   f"stay in the library to be read and adapted, but then the entry must say "
                   f"usable_in_submission: false (got {usable!r})")
    return out


def _index(data, where) -> list[dict]:
    """The entry mappings of an index object, after checking the whole of it."""
    problems: list[str] = []
    if not isinstance(data, dict):
        raise SchemaError(where, [f"the index must be an object with version and entries, got "
                                  f"{type(data).__name__}"])
    if extra := sorted(repr(k) for k in data if k not in ("version", "entries")):
        problems.append(f"unknown key(s) {', '.join(extra)}; the index has version and entries")
    if data.get("version") != INDEX_VERSION:
        problems.append(f"version must be {INDEX_VERSION}, got {data.get('version')!r}")
    rows = data.get("entries")
    if not isinstance(rows, list):
        problems.append(f"entries must be a list, got {rows!r}")
        rows = []
    seen: dict[str, int] = {}
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            problems.append(f"entries[{i}]: must be an object, got {type(row).__name__}")
            continue
        at = f"entries[{i}]" + (f" ({row['id']})" if isinstance(row.get("id"), str) else "")
        problems += [f"{at}: {p}" for p in validate(row)]
        e_id = row.get("id")
        if isinstance(e_id, str):
            if e_id in seen:
                problems.append(f"{at}: id is already used by entries[{seen[e_id]}]")
            else:
                seen[e_id] = i
    if problems:
        raise SchemaError(where, problems)
    return list(rows)


def load_index(path: str | Path) -> list[KernelEntry]:
    """The entries of the index at ``path``, in the file's order. Raise SchemaError naming
    every problem found (unreadable, not JSON, a wrong envelope, an entry off the schema, an
    id used twice)."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise SchemaError(p, [f"cannot read it: {e}"]) from None
    except ValueError as e:
        raise SchemaError(p, [f"does not parse as JSON: {e}"]) from None
    return [KernelEntry.from_dict(row) for row in _index(data, p)]


def save_index(entries, path: str | Path) -> Path:
    """Write ``entries`` (``KernelEntry`` or mapping) as the index at ``path``. An entry that
    does not follow the schema, or an id used twice, is refused with SchemaError and nothing
    is written."""
    rows: list[dict] = []
    for e in entries:
        if isinstance(e, KernelEntry):
            rows.append(e.to_dict())
        elif isinstance(e, Mapping):
            rows.append(dict(e))
        else:
            raise SchemaError(path, [f"entries[{len(rows)}]: must be an entry (a KernelEntry or a "
                                     f"mapping), got {type(e).__name__}"])
    _index({"version": INDEX_VERSION, "entries": rows}, where=path)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps({"version": INDEX_VERSION, "entries": rows}, indent=2,
                              ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p
