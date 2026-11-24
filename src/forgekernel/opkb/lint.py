"""Lint of the lesson base: every record is checked before a teacher may read it.

The lesson base is read by later runs' teachers as ``{LESSONS_PATH}``, so a record is an exam
surface in the same sense ``hw_kb/`` entries are (``docs/kb-provenance/README.md``): a machine
path, a credential or a hand-written narrative in one record reaches every later run of that
family. ``loop/kb_lint.py`` enforces that discipline for ``hw_kb/``; this module does the same
job for ``op_kb/`` with the checks the record format needs:

1. **schema** -- ``schema`` is the integer this build reads (``1``).
2. **required fields** -- every field a reader relies on is there.
3. **types** -- each field is the type the schema says (numbers are not bools, ids are strings,
   lists are lists).
4. **unknown keys** -- a typo in a key is a finding, not a silently ignored field.
5. **family** -- the key is one ``families.toml`` declares (``forgekernel.family.is_family``),
   so no record is filed where no query will look for it.
6. **length caps** -- per string, per list and per file; the caps are
   ``forgekernel.opkb.record``'s own constants, so what harvest writes always passes.
7. **path leakage** -- absolute machine paths (``/home/...``, ``C:\\...``), run-internal
   relative paths (``.artifacts/``, ``operators/``), ``..`` traversal, ``local.env`` /
   ``submit.local.toml``. A sha-shaped token is fine (``source.best_sha`` is one); a path into
   this machine is not.
8. **secret shapes** -- key material and token assignments.
9. **verbatim** -- an entry carrying free text (``kept[*].change``, ``failed[*].change`` /
   ``.error``, the recon values) says ``verbatim: true``: the text is the source's own.
10. **id** -- a safe file-name stem that matches the file's name.
11. **created** -- an ISO 8601 timestamp.
12. **outcome / reference / certified_kb values** -- the enums and shapes the readers assume;
    ``certified_kb`` entries are ``hw_kb/``-relative paths.

A failing record is **moved to** ``op_kb/_quarantine/`` -- file name preserved -- never
deleted: it is evidence of what a run wrote, and a human decides whether to repair it. After
the sweep the index is rebuilt from what remains.

::

    forge lessons lint [--root DIR]
    from forgekernel.opkb import lint
    lint.lint_record(data, label="rmsnorm/x.json", stem="x")   # ["...: kept[0].change: ..."]
    report = lint.lint_root(root)                               # quarantine + index rebuild
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from forgekernel.family import is_family
from forgekernel.inner.trajectory import GATE_WINDOWS
from forgekernel.opkb import record as record_mod
from forgekernel.opkb import store

TOP_KEYS = frozenset({"schema", "id", "created", "family", "problem", "reference", "outcome",
                      "kept", "failed", "meta", "certified_kb", "usage", "recon", "source"})
REQUIRED_KEYS = ("schema", "id", "created", "family", "problem", "reference", "outcome",
                 "source")
PROBLEM_KEYS = frozenset({"suite", "name", "definition", "track", "arch", "language",
                          "workloads"})
OUTCOME_KEYS = frozenset({"window", "metric_kind", "value", "stop", "hours",
                          "student_iterations"})
KEPT_KEYS = frozenset({"iter", "change", "before_ms", "after_ms", "verbatim"})
FAILED_KEYS = frozenset({"iter", "change", "error", "kind", "verbatim"})
META_KEYS = frozenset({"round", "outcome"})
RECON_KEYS = frozenset({"verbatim", "verdicts", "candidates"})
VERDICT_KEYS = frozenset({"kind", "line", "fields", "verbatim"})
CANDIDATE_KEYS = frozenset({"decision", "value", "verify", "verbatim"})
USAGE_KEYS = frozenset({"teacher_tokens", "student_tokens", "teacher_turns", "student_turns"})
SOURCE_KEYS = frozenset({"run", "best_sha"})

MAX_FILE_BYTES = 512 * 1024
MAX_STRING = {"id": record_mod.MAX_ID, "created": 64, "family": 64}
MAX_PROBLEM_STRING = {"suite": 64, "name": 256, "definition": 200, "track": 40, "arch": 32,
                      "language": 32}
MAX_LIST = {"kept": record_mod.MAX_ENTRIES, "failed": record_mod.MAX_ENTRIES,
            "meta": record_mod.MAX_ENTRIES, "certified_kb": record_mod.MAX_KB_ENTRIES}

# Path leakage. A record describes what a kernel did, never where this machine keeps it.
_ABS_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?:/(?:home|Users|root|tmp|mnt|opt|var|usr|workspace|data|scratch|gpfs|shared)/"
    r"|[A-Za-z]:[\\/]"
    r"|\\\\wsl)")
_LOCAL_PATH = re.compile(r"(?<![A-Za-z0-9_.-])(?:\.artifacts/|operators/)")
_TRAVERSAL = re.compile(r"(?:^|[/\\])\.\.(?:[/\\]|$)")
_LOCAL_ENV = re.compile(r"(?<![A-Za-z0-9_.-])(?:local\.env|submit\.local\.toml)"
                        r"(?![A-Za-z0-9_.-])")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_HW_KB_PATH = re.compile(r"^hw_kb/[A-Za-z0-9_./-]+\.md$")

SECRET_SHAPES = (
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"), "private key block"),
    (re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"), "API key (sk-...)"),
    (re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}"), "GitHub token"),
    (re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}"), "Hugging Face token"),
    (re.compile(r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    (re.compile(r"SOL_EXECBENCH_TOKEN\s*=\s*\S+"), "board token assignment"),
    # An assignment whose NAME is credential-shaped. A sentence that merely uses the word
    # ("secret: the weight row stays resident") is not a credential and must not be flagged.
    (re.compile(r"(?i)(?<![A-Za-z0-9_])(?:[A-Z0-9_]*(?:TOKEN|APIKEY|API_KEY|SECRET|PASSWORD|"
                r"PASSWD|CREDENTIALS?))\s*=\s*\S{4,}"), "credential assignment"),
)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass
class LintReport:
    """What one sweep found and did: findings, moved files, the rebuilt index."""
    findings: list[str] = field(default_factory=list)
    quarantined: list[tuple[str, str]] = field(default_factory=list)  # (from, to)
    checked: int = 0
    valid: int = 0
    index: dict | None = None

    @property
    def ok(self) -> bool:
        return not self.findings


# -- individual rules -------------------------------------------------------------------------


def _unknown_keys(data: dict, allowed: frozenset, where: str) -> list[str]:
    def _label(key: str) -> str:
        return f"{where}.{key}" if where else key

    return [f"{_label(k)}: unknown key (allowed: {', '.join(sorted(allowed))})"
            for k in sorted(set(data) - set(allowed))]


def _scan_paths(where: str, text: str) -> list[str]:
    out = []
    for pat, why in ((_ABS_PATH, "absolute machine path"),
                     (_LOCAL_PATH, "run-internal path"),
                     (_TRAVERSAL, "path traversal"),
                     (_LOCAL_ENV, "local credentials/config file name")):
        m = pat.search(text)
        if m:
            out.append(f"{where}: {why} ({m.group(0)!r})")
    return out


def _scan_secrets(where: str, text: str) -> list[str]:
    return [f"{where}: secret-shaped string ({why})" for pat, why in SECRET_SHAPES
            if pat.search(text)]


def _free_text_items(data: dict):
    """``(where, text)`` for every string a run's own words land in (the lint's scan set)."""
    problem = data.get("problem") if isinstance(data.get("problem"), dict) else {}
    for key in ("suite", "name", "definition", "track", "arch", "language"):
        if isinstance(problem.get(key), str):
            yield f"problem.{key}", problem[key]
    outcome = data.get("outcome") if isinstance(data.get("outcome"), dict) else {}
    for key in ("stop", "window", "metric_kind"):
        if isinstance(outcome.get(key), str):
            yield f"outcome.{key}", outcome[key]
    for key in ("kept", "failed"):
        rows = data.get(key)
        if not isinstance(rows, list):
            continue
        for i, item in enumerate(rows):
            if not isinstance(item, dict):
                continue
            for f in ("change", "error", "kind"):
                if isinstance(item.get(f), str):
                    yield f"{key}[{i}].{f}", item[f]
    for i, item in enumerate(data.get("meta") or ()):
        if isinstance(item, dict) and isinstance(item.get("outcome"), str):
            yield f"meta[{i}].outcome", item["outcome"]
    recon = data.get("recon") if isinstance(data.get("recon"), dict) else {}
    for i, item in enumerate(recon.get("verdicts") or ()):
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("line"), str):
            yield f"recon.verdicts[{i}].line", item["line"]
        fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
        for k, v in fields.items():
            if isinstance(v, str):
                yield f"recon.verdicts[{i}].fields.{k}", v
    for i, item in enumerate(recon.get("candidates") or ()):
        if not isinstance(item, dict):
            continue
        for f in ("decision", "value", "verify"):
            if isinstance(item.get(f), str):
                yield f"recon.candidates[{i}].{f}", item[f]
    source = data.get("source") if isinstance(data.get("source"), dict) else {}
    if isinstance(source.get("run"), str):
        yield "source.run", source["run"]


def check_types(data: dict) -> list[str]:
    out: list[str] = []
    if not _is_int(data.get("schema")):
        out.append("schema: must be an integer")
    for key in ("id", "created", "family"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            out.append(f"{key}: must be a non-empty string")
    for key in ("problem", "reference", "outcome", "source"):
        if not isinstance(data.get(key), dict):
            out.append(f"{key}: must be an object")
    for key in ("kept", "failed", "meta", "certified_kb"):
        if key in data and not isinstance(data[key], list):
            out.append(f"{key}: must be a list")
    if "usage" in data and not isinstance(data["usage"], dict):
        out.append("usage: must be an object")
    if "recon" in data and not isinstance(data["recon"], dict):
        out.append("recon: must be an object")

    problem = data.get("problem") if isinstance(data.get("problem"), dict) else {}
    out += _unknown_keys(problem, PROBLEM_KEYS, "problem")
    for key in ("suite", "name", "definition", "track", "arch", "language"):
        if key in problem and not isinstance(problem[key], str):
            out.append(f"problem.{key}: must be a string")
    if "workloads" in problem and not (_is_int(problem["workloads"])
                                       and problem["workloads"] >= 1):
        out.append("problem.workloads: must be a positive integer")
    if not isinstance(problem.get("name"), str) or not problem["name"].strip():
        out.append("problem.name: must be a non-empty string")

    reference = data.get("reference") if isinstance(data.get("reference"), dict) else {}
    out += _unknown_keys(reference, frozenset({"kind"}), "reference")
    if reference.get("kind") not in record_mod.REFERENCE_KINDS:
        out.append("reference.kind: must be one of " + ", ".join(record_mod.REFERENCE_KINDS))

    outcome = data.get("outcome") if isinstance(data.get("outcome"), dict) else {}
    out += _unknown_keys(outcome, OUTCOME_KEYS, "outcome")
    if "window" in outcome and outcome["window"] not in GATE_WINDOWS:
        out.append("outcome.window: must be one of " + ", ".join(GATE_WINDOWS))
    if "metric_kind" in outcome and outcome["metric_kind"] not in record_mod.METRIC_KINDS:
        out.append("outcome.metric_kind: must be one of "
                   + ", ".join(record_mod.METRIC_KINDS))
    if "metric_kind" in outcome and "value" not in outcome:
        out.append("outcome.value: missing although metric_kind is set")
    if "value" in outcome and not _is_number(outcome["value"]):
        out.append("outcome.value: must be a number")
    if "stop" in outcome and not isinstance(outcome["stop"], str):
        out.append("outcome.stop: must be a string")
    if "hours" in outcome and not (_is_number(outcome["hours"]) and outcome["hours"] >= 0):
        out.append("outcome.hours: must be a non-negative number")
    if "student_iterations" in outcome and not (_is_int(outcome["student_iterations"])
                                                and outcome["student_iterations"] >= 0):
        out.append("outcome.student_iterations: must be a non-negative integer")

    for i, item in enumerate(data.get("kept") or ()):
        where = f"kept[{i}]"
        if not isinstance(item, dict):
            out.append(f"{where}: must be an object")
            continue
        out += _unknown_keys(item, KEPT_KEYS, where)
        if not _is_int(item.get("iter")):
            out.append(f"{where}.iter: must be an integer")
        for key in ("before_ms", "after_ms"):
            if key in item and not _is_number(item[key]):
                out.append(f"{where}.{key}: must be a number")
        if "change" in item and not isinstance(item["change"], str):
            out.append(f"{where}.change: must be a string")
        if "verbatim" in item and item["verbatim"] is not True:
            out.append(f"{where}.verbatim: must be true (free text is the source's own)")
    for i, item in enumerate(data.get("failed") or ()):
        where = f"failed[{i}]"
        if not isinstance(item, dict):
            out.append(f"{where}: must be an object")
            continue
        out += _unknown_keys(item, FAILED_KEYS, where)
        if not _is_int(item.get("iter")):
            out.append(f"{where}.iter: must be an integer")
        for key in ("change", "error"):
            if key in item and not isinstance(item[key], str):
                out.append(f"{where}.{key}: must be a string")
        if "kind" in item and item["kind"] not in ("revert", "correctness"):
            out.append(f"{where}.kind: must be 'revert' or 'correctness'")
        if "verbatim" in item and item["verbatim"] is not True:
            out.append(f"{where}.verbatim: must be true (free text is the source's own)")
    for i, item in enumerate(data.get("meta") or ()):
        where = f"meta[{i}]"
        if not isinstance(item, dict):
            out.append(f"{where}: must be an object")
            continue
        out += _unknown_keys(item, META_KEYS, where)
        if not _is_int(item.get("round")):
            out.append(f"{where}.round: must be an integer")
        if not isinstance(item.get("outcome"), str) or not item["outcome"]:
            out.append(f"{where}.outcome: must be a non-empty string")

    if "usage" in data and isinstance(data["usage"], dict):
        out += _unknown_keys(data["usage"], USAGE_KEYS, "usage")
        for key, value in data["usage"].items():
            if key in USAGE_KEYS and not (_is_int(value) and value >= 0):
                out.append(f"usage.{key}: must be a non-negative integer")

    if "recon" in data and isinstance(data["recon"], dict):
        recon = data["recon"]
        out += _unknown_keys(recon, RECON_KEYS, "recon")
        if recon.get("verbatim") is not True:
            out.append("recon.verbatim: must be true (the entries are the files' own text)")
        for i, item in enumerate(recon.get("verdicts") or ()):
            where = f"recon.verdicts[{i}]"
            if not isinstance(item, dict):
                out.append(f"{where}: must be an object")
                continue
            out += _unknown_keys(item, VERDICT_KEYS, where)
            for key in ("kind", "line"):
                if not isinstance(item.get(key), str) or not item[key]:
                    out.append(f"{where}.{key}: must be a non-empty string")
            if not isinstance(item.get("fields"), dict):
                out.append(f"{where}.fields: must be an object")
            if item.get("verbatim") is not True:
                out.append(f"{where}.verbatim: must be true")
        for i, item in enumerate(recon.get("candidates") or ()):
            where = f"recon.candidates[{i}]"
            if not isinstance(item, dict):
                out.append(f"{where}: must be an object")
                continue
            out += _unknown_keys(item, CANDIDATE_KEYS, where)
            if not isinstance(item.get("decision"), str) or not item["decision"]:
                out.append(f"{where}.decision: must be a non-empty string")
            if not isinstance(item.get("value"), str):
                out.append(f"{where}.value: must be a string")
            if "verify" in item and not isinstance(item["verify"], str):
                out.append(f"{where}.verify: must be a string")
            if item.get("verbatim") is not True:
                out.append(f"{where}.verbatim: must be true")

    source = data.get("source") if isinstance(data.get("source"), dict) else {}
    out += _unknown_keys(source, SOURCE_KEYS, "source")
    for key in ("run", "best_sha"):
        if key in source and not isinstance(source[key], str):
            out.append(f"source.{key}: must be a string")
    if not isinstance(source.get("run"), str) or not source["run"].strip():
        out.append("source.run: must be a non-empty string")
    return out


def check_verbatim(data: dict) -> list[str]:
    """Every entry that carries free text says ``verbatim: true``."""
    out: list[str] = []
    for key in ("kept", "failed"):
        for i, item in enumerate(data.get(key) or ()):
            if not isinstance(item, dict):
                continue
            has_text = any(isinstance(item.get(f), str) and item[f]
                           for f in ("change", "error"))
            if has_text and item.get("verbatim") is not True:
                out.append(f"{key}[{i}].verbatim: missing although the entry carries free text")
    recon = data.get("recon")
    if isinstance(recon, dict):
        if (recon.get("verdicts") or recon.get("candidates")) and recon.get("verbatim") is not True:
            out.append("recon.verbatim: missing although the block carries recon text")
        for i, item in enumerate(recon.get("verdicts") or ()):
            if isinstance(item, dict) and item.get("verbatim") is not True:
                out.append(f"recon.verdicts[{i}].verbatim: missing")
        for i, item in enumerate(recon.get("candidates") or ()):
            if isinstance(item, dict) and item.get("verbatim") is not True:
                out.append(f"recon.candidates[{i}].verbatim: missing")
    return out


def check_limits(data: dict) -> list[str]:
    out: list[str] = []
    for key, limit in MAX_STRING.items():
        value = data.get(key)
        if isinstance(value, str) and len(value) > limit:
            out.append(f"{key}: {len(value)} characters, the cap is {limit}")
    problem = data.get("problem") if isinstance(data.get("problem"), dict) else {}
    for key, limit in MAX_PROBLEM_STRING.items():
        value = problem.get(key)
        if isinstance(value, str) and len(value) > limit:
            out.append(f"problem.{key}: {len(value)} characters, the cap is {limit}")
    for key, limit in MAX_LIST.items():
        rows = data.get(key)
        if isinstance(rows, list) and len(rows) > limit:
            out.append(f"{key}: {len(rows)} entries, the cap is {limit}")
    for key in ("kept", "failed"):
        for i, item in enumerate(data.get(key) or ()):
            if not isinstance(item, dict):
                continue
            for f, limit in (("change", record_mod.MAX_CHANGE),
                             ("error", record_mod.MAX_ERROR)):
                value = item.get(f)
                if isinstance(value, str) and len(value) > limit:
                    out.append(f"{key}[{i}].{f}: {len(value)} characters, the cap is {limit}")
    recon = data.get("recon") if isinstance(data.get("recon"), dict) else {}
    if isinstance(recon.get("verdicts"), list) and len(recon["verdicts"]) > \
            record_mod.MAX_RECON_VERDICTS:
        out.append(f"recon.verdicts: {len(recon['verdicts'])} entries, the cap is "
                   f"{record_mod.MAX_RECON_VERDICTS}")
    if isinstance(recon.get("candidates"), list) and len(recon["candidates"]) > \
            record_mod.MAX_RECON_CANDIDATES:
        out.append(f"recon.candidates: {len(recon['candidates'])} entries, the cap is "
                   f"{record_mod.MAX_RECON_CANDIDATES}")
    return out


def check_certified_kb(data: dict) -> list[str]:
    out: list[str] = []
    for i, entry in enumerate(data.get("certified_kb") or ()):
        if not isinstance(entry, str):
            out.append(f"certified_kb[{i}]: must be a string")
        elif not _HW_KB_PATH.match(entry):
            out.append(f"certified_kb[{i}]: {entry!r} is not a hw_kb/... .md path")
    return out


def lint_record(data, *, label: str = "", stem: str | None = None) -> list[str]:
    """Every finding for one record (``label`` is what a reader sees; ``stem`` is the file name)."""
    if not isinstance(data, dict):
        return [f"{label}: not a JSON object"]
    raw: list[str] = []
    raw += _unknown_keys(data, TOP_KEYS, "")
    for key in REQUIRED_KEYS:
        if key not in data:
            raw.append(f"{key}: required field is missing")
    if _is_int(data.get("schema")) and data["schema"] != record_mod.SCHEMA:
        raw.append(f"schema: {data['schema']} is not the version this build reads "
                   f"({record_mod.SCHEMA})")
    raw += check_types(data)
    if isinstance(data.get("family"), str) and not is_family(data["family"]):
        raw.append(f"family: {data['family']!r} is not a family families.toml declares")
    if isinstance(data.get("id"), str):
        rid = data["id"]
        if not _SAFE_ID.match(rid):
            raw.append(f"id: {rid!r} is not a safe file-name stem")
        if ".." in rid:
            raw.append("id: contains '..'")
        if stem is not None and rid != stem:
            raw.append(f"id: {rid!r} does not match the file name {stem!r}")
    if isinstance(data.get("created"), str) and record_mod.parse_time(data["created"]) is None:
        raw.append(f"created: {data['created']!r} is not an ISO 8601 timestamp")
    raw += check_limits(data)
    raw += check_certified_kb(data)
    raw += check_verbatim(data)
    for where, text in _free_text_items(data):
        raw += _scan_paths(where, text)
        raw += _scan_secrets(where, text)
    return [f"{label}: {finding}" if label else finding for finding in raw]


# -- the sweep --------------------------------------------------------------------------------


def lint_root(root: str | Path, *, quarantine_bad: bool = True,
              now: str | None = None) -> LintReport:
    """Lint every record under ``root``; quarantine the failures; rebuild the index.

    A missing root is a finding (fail closed: a mispointed root must not look clean), and
    nothing is created. A record that cannot be read at all is quarantined as well -- it is a
    file somebody put in the lesson base, and it must not sit there unread."""
    base = store.root_dir(root) if root is None else Path(root).expanduser()
    report = LintReport()
    if not base.is_dir():
        report.findings.append(f"{base}: lesson base does not exist -- nothing was scanned")
        return report
    for family_dir in sorted(p for p in base.iterdir() if store.is_record_dir(p)):
        for path in sorted(family_dir.glob("*.json")):
            report.checked += 1
            label = str(path.relative_to(base))
            findings: list[str] = []
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError as e:
                findings = [f"{label}: unreadable ({e})"]
            else:
                size = len(raw.encode("utf-8"))
                if size > MAX_FILE_BYTES:
                    findings = [f"{label}: {size} bytes, the cap is {MAX_FILE_BYTES}"]
                else:
                    try:
                        data = json.loads(raw)
                    except ValueError as e:
                        findings = [f"{label}: does not parse ({e})"]
                    else:
                        findings = lint_record(data, label=label, stem=path.stem)
            if not findings:
                report.valid += 1
                continue
            report.findings.extend(findings)
            if quarantine_bad:
                dest = store.quarantine(base, path, "\n".join(findings))
                report.quarantined.append((str(path), str(dest)))
    report.index = store.rebuild_index(base, write=True, now=now)
    return report


def stats(root: str | Path) -> dict:
    """Record counts per family, plus how many files sit in quarantine."""
    base = store.root_dir(root) if root is None else Path(root).expanduser()
    counts: dict[str, int] = {}
    for record in store.load_records(base):
        family = record.get("family")
        if isinstance(family, str):
            counts[family] = counts.get(family, 0) + 1
    quarantine = store.quarantine_dir(base)
    return {"root": str(base),
            "total": sum(counts.values()),
            "families": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "quarantined": len(list(quarantine.glob("*.json"))) if quarantine.is_dir() else 0}
