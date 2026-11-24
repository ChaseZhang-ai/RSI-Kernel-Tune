"""The licence gate on a submission that derives from someone else's kernel.

The board's collection submission is one ``submission.py`` per kernel, and the team is
explicitly allowed to start from an existing kernel (another repository, a reference
library entry, one of our own older solutions).  What is not allowed is shipping code
whose licence forbids it, or shipping copied code with its licence notice stripped -- the
second is how a permissively-licensed kernel becomes a legal problem, and it is invisible
in a diff.

``origins`` is what campaign passes in (the reference library entry's ``meta.json``
fields): a list of ``{"license", "source", "notice"}``.  For every origin:

* ``license`` must be one of :data:`ALLOWED_LICENSES`, else ``block``
  (``license-not-allowed``);
* ``notice`` -- the original file's licence header or copyright line -- must appear in the
  submission's source, else ``block`` (``license-notice-missing``), with the text to add
  as the finding's detail ("add this declaration at the top of the file");
* an origin with no ``notice`` at all falls back to requiring the licence *name* in the
  source (a banner line is enough) -- there is nothing else to anchor on, and a
  submission with no trace of the origin's licence is exactly what this gate exists for.

``notices.banner(origins)`` renders the comment block that satisfies the second rule; the
export path (`campaign.compose.export`, `forge export`) puts it at the top of the file.

Two call shapes, on purpose:

* ``license_check(submission_src, origins)`` -- the API this task specifies; returns a
  ``list[Finding]`` (empty = clean).
* ``license_check(meta)`` -- campaign's ``tools.resolve`` slot calls it with one candidate
  ``meta.json`` dict and reads a dict back (``{"ok", "problems", "findings"}``).  Keeping
  that shape here rather than in campaign means the gate is one function, not two.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .lint import BLOCK, Finding, blocks

ALLOWED_LICENSES = ("Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "BSD")
# Spellings that mean one of the above (SPDX ids are not always used in practice).
_NORMALIZE = {
    "apache-2.0": "Apache-2.0", "apache 2.0": "Apache-2.0", "apache2": "Apache-2.0",
    "apache-2": "Apache-2.0", "apache license 2.0": "Apache-2.0",
    "mit": "MIT", "mit license": "MIT", "expat": "MIT",
    "bsd": "BSD", "bsd-3-clause": "BSD-3-Clause", "bsd 3-clause": "BSD-3-Clause",
    "bsd-3": "BSD-3-Clause", "3-clause bsd": "BSD-3-Clause",
    "bsd-2-clause": "BSD-2-Clause", "bsd 2-clause": "BSD-2-Clause",
    "bsd-2": "BSD-2-Clause", "2-clause bsd": "BSD-2-Clause",
}

LICENSE_NOT_ALLOWED = "license-not-allowed"
LICENSE_NOTICE_MISSING = "license-notice-missing"
LICENSE_UNKNOWN = "license-unknown"

# A notice is compared with whitespace normalised (a copy may re-wrap a comment block).
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Origin:
    license: str = ""
    source: str = ""
    notice: str = ""

    @classmethod
    def of(cls, item) -> "Origin":
        if isinstance(item, Origin):
            return item
        if isinstance(item, dict):
            return cls(license=str(item.get("license") or ""),
                       source=str(item.get("source") or item.get("path") or ""),
                       notice=str(item.get("notice") or ""))
        return cls(license=str(item or ""))


def allowed_license(name: str) -> bool:
    """Whether a licence name is one a submission may be derived from."""
    return normalize_license(name) is not None


def normalize_license(name: str) -> str | None:
    """The canonical allowed licence name, or None when it is not allowed."""
    raw = (name or "").strip()
    if not raw:
        return None
    key = raw.lower()
    if key in _NORMALIZE:
        return _NORMALIZE[key]
    for token in re.split(r"[,;/]|\s+or\s+|\s+and\s+", key):
        token = token.strip()
        if token in _NORMALIZE:
            return _NORMALIZE[token]
    return None


def _normal(text: str) -> str:
    return _WS.sub(" ", (text or "").strip())


def _contains(source: str, text: str) -> bool:
    return bool(text) and _normal(text) in _normal(source)


def license_check(submission_src, origins=None):
    """The licence gate.

    ``license_check(source, origins)`` -> ``list[Finding]`` (empty means clean).
    ``license_check(meta_dict)`` -> ``{"ok", "problems", "findings"}`` (campaign's slot).
    """
    if origins is None and not isinstance(submission_src, str):
        return _campaign_shape(submission_src)
    source = submission_src if isinstance(submission_src, str) else ""
    findings: list[Finding] = []
    for item in origins or ():
        origin = Origin.of(item)
        label = origin.source or "(no source)"
        canonical = normalize_license(origin.license)
        if canonical is None:
            findings.append(Finding(
                rule=LICENSE_NOT_ALLOWED, severity=BLOCK, file="", line=0,
                snippet=f"{label}: {origin.license or '(no licence stated)'}",
                detail=(f"{label} is not licensed under one of {', '.join(ALLOWED_LICENSES)}; "
                        f"code from it cannot enter a submission")))
            continue
        anchor = origin.notice.strip() or canonical
        if not _contains(source, anchor):
            findings.append(Finding(
                rule=LICENSE_NOTICE_MISSING, severity=BLOCK, file="", line=0,
                snippet=label,
                detail=(f"add this declaration at the top of the file (see "
                        f"forgekernel.integrity.notices.banner): {anchor}")))
    return findings


def _campaign_shape(meta) -> dict:
    """campaign's slot: one ``meta.json`` dict in, a verdict dict out.

    campaign resolves this slot from a candidate's *meta* before the file is composed, so
    there is no source to look for a notice in.  Without a source this verdict is the
    allowlist only (``notes`` says the notice is checked on the composed file); with one
    (``submission_src`` / ``source_text`` in the meta) it is the full gate.
    """
    meta = dict(meta or {}) if isinstance(meta, dict) else {}
    origins = meta.get("origins")
    if origins is None:
        origins = [meta] if meta.get("license") or meta.get("notice") else []
    elif isinstance(origins, dict):
        origins = [origins]
    source = str(meta.get("submission_src") or meta.get("source_text") or "")
    notes = []
    findings = license_check(source, list(origins))
    if not source:
        findings = [f for f in findings if f.rule != LICENSE_NOTICE_MISSING]
        if any(f.rule == LICENSE_NOTICE_MISSING for f in license_check("", list(origins))):
            notes.append("notice not checked: campaign checks licences before the file "
                         "exists; the notice is verified on the composed submission "
                         "(forgekernel.integrity.license_check(source, origins))")
    problems = []
    for f in blocks(findings):
        problems.append(f"{f.rule}: {f.snippet} ({f.detail})" if f.detail
                        else f"{f.rule}: {f.snippet}")
    return {"ok": not problems, "problems": problems,
            "findings": [f.to_dict() for f in findings],
            "notes": notes,
            "licenses": sorted({str(Origin.of(o).license) for o in origins if Origin.of(o).license})}


def license_findings_for_source(submission_src: str, origins) -> list[Finding]:
    """``license_check`` with the source shape spelled out (used by the export path)."""
    return license_check(submission_src, list(origins or ()))


def read_origins(path) -> list[dict]:
    """The origins a sidecar file names: a JSON list, or a dict with ``origins``."""
    import json
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("origins") or [data]
    return [dict(O if isinstance(O, dict) else {"license": O}) for O in data]
