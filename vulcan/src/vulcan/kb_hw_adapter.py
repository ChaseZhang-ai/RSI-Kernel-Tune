"""Emit VULCAN knowledge entries as upstream ``hw_kb/<lang>/<arch>/*.md`` files.

ForgeKernel's agents read hardware mechanisms from ``hw_kb/<language>/<arch>/`` markdown
files, one per interface, with fixed sections (see upstream ``hw_kb/*/TEMPLATE.md``).
This adapter renders a :class:`vulcan.kb_graph.KnowledgeEntry` into that format so the
mechanisms VULCAN certifies are actually available to the agents during a run — the one
bridge that was missing for improvement item 6.

Two upstream rules are honoured:

* **Only certified entries are emitted** into the knowledge base.  Candidates are
  hypotheses and refuted entries are negative knowledge; both go elsewhere.
* **No provenance tokens.**  Upstream's ``kb_lint`` bans ``oracle`` / ``teacher`` /
  ``reference implementation`` anywhere in a `hw_kb` entry.  The evidence text is routed
  to a companion provenance file (for humans), and the entry keeps a one-line pointer.
  :func:`lint_text` reproduces the banned-word check so a violation is caught here first.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import kb_graph
from .kb_graph import KnowledgeEntry, KnowledgeGraph

BANNED_WORDS = ("oracle", "teacher", "reference implementation")

_LANG_FENCE = {"cutedsl": "python", "triton": "python", "cuda": "cpp", "cuda-sol": "cpp"}


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "entry"


def lint_text(text: str) -> List[str]:
    """Return the banned provenance tokens present (case-insensitive)."""
    low = text.lower()
    return [w for w in BANNED_WORDS if w in low]


def _predicate_lines(entry: KnowledgeEntry) -> str:
    if not entry.predicates:
        return "- (no measured trigger recorded — probe before prescribing)"
    lines = []
    for name, (lo, hi) in sorted(entry.predicates.items()):
        if lo == hi:
            lines.append(f"- `{name}` == {lo:g}")
        else:
            lines.append(f"- `{name}` in [{lo:g}, {hi:g}]")
    return "\n".join(lines)


def entry_to_markdown(
    entry: KnowledgeEntry,
    lang: str = "cutedsl",
    provenance_rel: str = "",
) -> str:
    """Render one certified entry as an upstream-format ``hw_kb`` markdown file."""
    fence = _LANG_FENCE.get(lang, "python")
    prov_line = (f"- Detailed provenance: {provenance_rel} "
                 f"(for human review, not in the agent read set).") if provenance_rel else \
        "- Detailed provenance: not recorded."

    recipe = entry.recipe.strip() or (
        "// No minimal example recorded. Probe the interface on the target GPU before\n"
        "// prescribing this mechanism.")
    gotchas = []
    if entry.cost:
        gotchas.append(f"- Cost / side effects: {entry.cost}")
    if entry.limitations:
        gotchas.append(f"- Limitations: {entry.limitations}")
    if not gotchas:
        gotchas.append("- None recorded.")

    return f"""# {_slug(entry.id)} ({entry.mechanism})

> Certified on {entry.arch} (VULCAN knowledge base; expected effect {entry.effect:+.3f}
> relative). One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

{entry.mechanism}

## When needed (trigger)

{_predicate_lines(entry)}

## Minimal runnable example

```{fence}
{recipe}
```

## Gotchas

{chr(10).join(gotchas)}

## Sources

- GPU-certified by the VULCAN probe on {entry.arch} (PROBE PASS).
{prov_line}
"""


def provenance_markdown(entry: KnowledgeEntry) -> str:
    """Human-facing provenance note (may mention the reference; kept out of `hw_kb`)."""
    return (f"# Provenance: {entry.id}\n\n"
            f"- mechanism: {entry.mechanism}\n"
            f"- arch: {entry.arch}\n"
            f"- status: {entry.status}\n"
            f"- evidence: {entry.evidence or '(none)'}\n"
            f"- analog of: {entry.analog_of or '(n/a)'}\n"
            f"- negative of: {entry.negative_of or '(n/a)'}\n")


@dataclass
class EmitResult:
    written: List[str]
    provenance: List[str]
    negatives: List[str]
    lint_hits: Dict[str, List[str]]

    @property
    def ok(self) -> bool:
        return not self.lint_hits


def write_hw_kb(
    graph: KnowledgeGraph,
    out_dir: str,
    arch: str,
    lang: str = "cutedsl",
    provenance_dir: Optional[str] = None,
    include_negatives: bool = False,
) -> EmitResult:
    """Write the graph's certified entries for ``arch`` into ``out_dir``.

    ``out_dir`` should be the upstream tree ``hw_kb/<lang>/<arch>``.  Refuted entries are
    written to ``out_dir/negatives.md`` only when ``include_negatives`` is set, so the
    agent read set stays positive-prescriptive by default.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prov_dir = Path(provenance_dir) if provenance_dir else None
    if prov_dir:
        prov_dir.mkdir(parents=True, exist_ok=True)

    written, prov, negs, hits = [], [], [], {}

    def _emit(entry: KnowledgeEntry) -> None:
        prov_rel = ""
        if prov_dir:
            pr = prov_dir / f"{_slug(entry.id)}.md"
            text = provenance_markdown(entry)
            pr.write_text(text)
            prov.append(str(pr))
            try:
                prov_rel = str(pr.relative_to(out))
            except ValueError:
                prov_rel = str(pr)
        md = entry_to_markdown(entry, lang, prov_rel)
        bad = lint_text(md)
        if bad:
            hits[entry.id] = bad
        fp = out / f"{_slug(entry.id)}.md"
        fp.write_text(md)
        written.append(str(fp))

    for entry in graph:
        if entry.status == kb_graph.STATUS_CERTIFIED and entry.arch == arch:
            _emit(entry)

    if include_negatives:
        neg_lines = []
        for entry in graph:
            if entry.status == kb_graph.STATUS_REFUTED and entry.arch == arch:
                text = (entry_to_markdown(entry, lang)
                        + "\n> **NEGATIVE KNOWLEDGE: this was tried and regressed under the "
                          "trigger above. Do not prescribe it.**\n")
                neg_lines.append(text)
        if neg_lines:
            fp = out / "negatives.md"
            fp.write_text("\n\n---\n\n".join(neg_lines))
            negs.append(str(fp))

    return EmitResult(written=written, provenance=prov, negatives=negs, lint_hits=hits)
