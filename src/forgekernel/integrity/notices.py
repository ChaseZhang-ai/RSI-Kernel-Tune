"""The comment block a derived submission must carry, and the helpers that keep it first.

A submission that started from an existing kernel has to say so: which file, under which
licence, with whose copyright line.  ``banner(origins)`` renders exactly that as a comment
block; ``forge export`` and ``campaign.compose.export`` put it at the top of the file, and
``forgekernel.integrity.license.license_check`` refuses a file where the declarations it
names cannot be found.

The output is deterministic: the origins are printed in the order given (campaign passes
the entry's own list), duplicates collapse, and nothing depends on a set's iteration
order.  An empty origin list gives an empty string -- a submission that derives from
nothing needs no banner.
"""

from __future__ import annotations

from .license import Origin, normalize_license


def banner(origins) -> str:
    """The notice block for ``origins`` (``source``, ``license``, ``notice``), or ``""``."""
    lines = ["# NOTICE: sources this file derives from"]
    seen: set[tuple[str, str, str]] = set()
    count = 0
    for item in origins or ():
        origin = Origin.of(item)
        source = origin.source.strip() or "(source not named)"
        license_name = normalize_license(origin.license) or origin.license.strip() or \
            "(licence not stated)"
        notice = origin.notice.strip()
        key = (source, license_name, notice)
        if key in seen:
            continue
        seen.add(key)
        count += 1
        lines.append(f"# - {source}: {license_name}")
        if notice:
            for line in notice.splitlines():
                lines.append(f"#   {line.strip()}" if line.strip() else "#")
    return "\n".join(lines) if count else ""


def strip_banner(source: str) -> str:
    """``source`` without a leading notice banner (the block ``banner`` writes)."""
    lines = source.splitlines(True)
    i = 0
    if lines and lines[0].startswith("# NOTICE:"):
        i = 1
        while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
            i += 1
    return "".join(lines[i:])


def has_banner(source: str) -> bool:
    return source.lstrip().startswith("# NOTICE:")


__all__ = ["banner", "strip_banner", "has_banner"]
