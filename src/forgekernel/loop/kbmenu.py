"""Deterministic KB-menu synthesizer (WP12; contract C7: the single
implementation of the injection synth — pure functions, no model calls, same
input -> same output).

The teacher's "diagnose -> decide" chain had two places that ran on memory:
(1) which hw_kb index row a reading points at, and (2) which lever families
the design round should consider, with which measured exclusions. This module
turns both into a lookup: a fixed rule table maps digest readings /
resusage / falsified-ledger rows / meta-status tail to a <=30-line ``KB-MENU``
block (contract C3) with three sections:

- ``KB-HINTS:`` symptom readings -> index row POINTERS (filename + a verbatim
  quote of the matched row — never the answer itself; the row lives in the
  tree the teacher already reads).
- ``LEVER-MENU:`` symptom -> lever FORMS to consider for the design round,
  plus the falsified-ledger rows of this arch as measured exclusions ("do not
  re-propose"). The teacher keeps selection and pricing: nothing here carries
  an unmeasured number.
- ``CALIBRATION:`` the teacher's own recent record, aggregated from
  meta_status.jsonl (reopen payback, lever reversed/faded rate, the
  consecutive-REVERT_PATCH direction list) plus stale-KB-entry warnings.

Robustness contract: every section is rendered only when its inputs yield
lines — a run with no ledger rows, a short meta history, or an index with no
matching row gets shorter output, never an empty-shell heading; with nothing
at all the block is ``""`` and the driver injects nothing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

__all__ = [
    "RULES", "Rule", "build_kb_menu", "parse_symptom_rows", "stale_kb_entries",
    "kb_cite_name", "SymptomRow",
]

# --- the fixed rule table -----------------------------------------------------
#
# Each rule: fires on readings (digest SOL header + WP2 profiler keys) and/or
# resusage (regs / smem / cstack), points at the FIRST index row carrying all
# of ``keywords`` (case-insensitive, absent keywords simply never match so a
# tree without the row shrinks the hint list), and contributes one lever FORM
# to LEVER-MENU. ``keywords=None`` = lever-menu-only (no index pointer is
# claimed). Extensible by appending — order is the render order.

# Thresholds (documented so the table stays reviewable; deliberately coarse —
# the menu triages, the teacher judges):
L2_LOW_PCT = 40.0        # l2_hit_rate_pct below this on a working set that should fit L2
WAVES_LOW = 1.0          # waves_per_sm below 1 = most SMs never saw work (grid/tail)
SOL_WARM_PCT = 70.0      # both SOL sides below this = no obvious roof; attribution needs decomposition
SMEM_CLIFF_BYTES = 200_000  # per-CTA smem near the pipeline cap (the KB entry reads 232448 B)


def _num(readings: dict, key: str):
    v = readings.get(key)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


@dataclass(frozen=True)
class Rule:
    name: str
    # (readings, resusage) -> bool; pure
    when: Callable[[dict, dict], bool]
    # trigger line for KB-HINTS / LEVER-MENU; pure
    label: Callable[[dict, dict], str]
    # all keywords must appear (ci) in a row for the pointer; None = no pointer
    keywords: tuple[str, ...] | None
    # the lever FORM suggested to the design round (no numbers)
    lever_hint: str


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _first(*vals):
    return next((v for v in vals if v is not None), None)


RULES: list[Rule] = [
    Rule(
        "spill",
        lambda r, ru: (_first(_num(r, "spill_st_sectors"), _num(r, "spill_ld_sectors")) or 0) > 0
        or (_num(ru, "cstack") or 0) > 0,
        lambda r, ru: (f"spill/cstack>0 (spill_st={_fmt(_num(r, 'spill_st_sectors'))}, "
                       f"spill_ld={_fmt(_num(r, 'spill_ld_sectors'))}, cstack={_fmt(_num(ru, 'cstack'))})"),
        ("STL",),
        "register-frame levers (setmaxnreg / maxrregcount / cursor-vs-array / role-local scalars)",
    ),
    Rule(
        "regs-at-cap",
        lambda r, ru: (_num(r, "regs_per_thread") or 0) >= 128
        and (_first(_num(r, "spill_st_sectors"), _num(r, "spill_ld_sectors")) or 0) > 0,
        lambda r, ru: f"spilling at high regs (regs_per_thread={_fmt(_num(r, 'regs_per_thread'))})",
        ("setmaxnreg",),
        "register-tier levers (warp_specialization caps / two-pass readout)",
    ),
    Rule(
        "bankconf",
        lambda r, ru: (_num(r, "bankconf_st") or 0) > 0 or (_num(r, "bankconf_ld") or 0) > 0,
        lambda r, ru: (f"bank conflicts (bankconf_st={_fmt(_num(r, 'bankconf_st'))}, "
                       f"bankconf_ld={_fmt(_num(r, 'bankconf_ld'))})"),
        ("bankconf",),
        "smem layout / face width / read-mapping levers",
    ),
    Rule(
        "l2-low",
        lambda r, ru: (_num(r, "l2_hit_rate_pct") or 100.0) < L2_LOW_PCT,
        lambda r, ru: f"l2_hit_rate_pct={_fmt(_num(r, 'l2_hit_rate_pct'))} < {L2_LOW_PCT:g}",
        ("swizzle", "reference"),
        "tiling-order / swizzle / raster levers (check the working set against L2 first)",
    ),
    Rule(
        "waves-low",
        lambda r, ru: (_num(r, "waves_per_sm") or 1.0) < WAVES_LOW,
        lambda r, ru: f"waves_per_sm={_fmt(_num(r, 'waves_per_sm'))} < {WAVES_LOW:g}",
        ("persistent",),
        "grid-shape / tail / persistent-scheduling levers (settle the grid before inner-loop tuning)",
    ),
    Rule(
        "smem-cliff",
        lambda r, ru: (_num(ru, "smem") or 0) >= SMEM_CLIFF_BYTES,
        lambda r, ru: f"smem={_fmt(_num(ru, 'smem'))} B >= {SMEM_CLIFF_BYTES} (stage-count cliff)",
        ("ring depth",),
        "stage_count / ring-depth levers (depth pays only to saturation; smem is the cap)",
    ),
    Rule(
        "latency-bound",
        lambda r, ru: (_num(r, "compute_sol_pct") or 100.0) < SOL_WARM_PCT
        and (_num(r, "memory_sol_pct") or 100.0) < SOL_WARM_PCT,
        lambda r, ru: (f"both SOL low (compute={_fmt(_num(r, 'compute_sol_pct'))}%, "
                       f"memory={_fmt(_num(r, 'memory_sol_pct'))}%)"),
        None,
        "latency-bound reading: price per-instruction stall-ledger rows, not aggregate ratios",
    ),
    Rule(
        "unbound-attribution",
        lambda r, ru: (_num(r, "compute_sol_pct") or 100.0) < SOL_WARM_PCT
        and (_num(r, "memory_sol_pct") or 100.0) < SOL_WARM_PCT
        and (_num(r, "tensor_pct") is not None and _num(r, "tensor_pct") < SOL_WARM_PCT),
        lambda r, ru: f"tensor_pct={_fmt(_num(r, 'tensor_pct'))}% with both SOL low",
        None,
        "attribution recipe: skills/probe-recipes/phase_ablation.py (read/compute split)",
    ),
    Rule(
        "wave-quantization",
        lambda r, ru: 0.0 < (_num(r, "waves_per_sm") or 0.0) < 8.0,
        lambda r, ru: f"waves_per_sm={_fmt(_num(r, 'waves_per_sm'))} (few waves: check the tail)",
        None,
        "curve recipe: skills/probe-recipes/occupancy_scaling.py (flat/linear/step reading)",
    ),
    Rule(
        "dram-high",
        lambda r, ru: (_num(r, "dram_pct") or 0.0) >= 85.0,
        lambda r, ru: f"dram_pct={_fmt(_num(r, 'dram_pct'))}% (SOL denominator is theoretical peak)",
        None,
        "bandwidth check recipe: skills/probe-recipes/effective_bw.py (re-base SOL on measured node BW)",
    ),
]

# --- index row parsing ---------------------------------------------------------


@dataclass(frozen=True)
class SymptomRow:
    text: str      # the raw row line, verbatim (substring source for quotes)
    entry: str     # primary entry file name ("x.md"), "" when the row carries no link
    snippet: str   # verbatim quote (a substring of text) — the row's symptom/purpose cell, bounded
    probe: str = ""  # the UNBOUNDED snippet cell — what keywords match against

    def __post_init__(self):
        if not self.probe:
            object.__setattr__(self, "probe", self.snippet)

    def __str__(self) -> str:  # pragma: no cover - repr helper
        return f"{self.entry}: {self.snippet}"


_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+\.md)\)")
_SYMPTOM_HEADING_RE = re.compile(r"^#+\s*.*symptom\s+first", re.IGNORECASE)
_HEADING_RE = re.compile(r"^#+\s")
_SNIPPET_MAX = 96


def _cells(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [c.strip() for c in body.split("|")]


def _has_link(cell: str) -> bool:
    return _LINK_RE.search(cell) is not None


def _snippet_cell(cells: list[str]) -> str:
    """The row's symptom/purpose cell: the first cell that carries prose (a
    symptom-table row keeps it in cell 1; an interface-list row like sm90's
    puts the link in cell 1 and the purpose in cell 2). Verbatim — quotes must
    stay substrings of the raw line."""
    for c in cells[:-1] if len(cells) > 2 else cells:
        if not _has_link(c) and len(c) >= 24:
            return c
    return (cells[1] if len(cells) > 1 else (cells[0] if cells else ""))[:_SNIPPET_MAX]


def _entry_cell(cells: list[str], skip: str) -> str:
    """First markdown link outside the snippet cell, scanning cells in order."""
    for c in cells:
        if c == skip:
            continue
        m = _LINK_RE.search(c)
        if m:
            return m.group(2)
    return ""


def parse_symptom_rows(index_text: str) -> list[SymptomRow]:
    """The rows a pointer may quote. Trees organised symptom-first contribute
    only the symptom section's rows; a flat interface-list tree (sm90 style,
    no symptom heading) contributes every table row. Header / separator rows
    are skipped. Order = file order (the pointer takes the FIRST match)."""
    lines = index_text.splitlines()
    start = next((i for i, ln in enumerate(lines) if _SYMPTOM_HEADING_RE.match(ln)), None)
    rows: list[SymptomRow] = []
    if start is not None:
        scan = lines[start + 1:]
        end = next((i for i, ln in enumerate(scan) if _HEADING_RE.match(ln)), len(scan))
        scan = scan[:end]
    else:
        scan = lines
    for ln in scan:
        s = ln.strip()
        if not (s.startswith("|") and s.endswith("|")) or re.fullmatch(r"[|\-:\s]+", s):
            continue
        cells = _cells(s)
        if len(cells) < 2 or not any(cells):
            continue
        probe = _snippet_cell(cells)
        entry = _entry_cell(cells, probe)
        rows.append(SymptomRow(text=s, entry=entry, snippet=probe[:_SNIPPET_MAX], probe=probe))
    return rows


def _match_row(rows: Iterable[SymptomRow], keywords: tuple[str, ...]) -> SymptomRow | None:
    """First row (file order) whose snippet CELL, unbounded (the symptom /
    purpose text the pointer quotes), contains every keyword, ci — matching
    where the reader will look, not in some incidental word deep in a row's
    description."""
    for row in rows:
        low = row.probe.lower()
        if all(k.lower() in low for k in keywords):
            return row
    return None


# --- meta-record accessors (dicts or MetaRecord-like objects) ------------------

def _get(rec, name, default=None):
    if isinstance(rec, dict):
        v = rec.get(name, default)
    else:
        v = getattr(rec, name, default)
    return v


_KB_CITE_RE = re.compile(r"\bkb\s*[:：]\s*([^\s,;，；()（）]+)", re.IGNORECASE)


def kb_cite_name(cite: str) -> str:
    """The KB entry name a ``reopen_price: ... via KB:<name>`` line cites
    (path/extension stripped, the same normalisation the price gate applies)."""
    m = _KB_CITE_RE.search(cite or "")
    if not m:
        return ""
    return re.sub(r"\.md$", "", m.group(1).strip("`'\"").split("/")[-1])


def stale_kb_entries(meta_records: Iterable, streak: int = 2) -> list[str]:
    """KB entries whose last ``streak`` reopen citations (in meta-step order)
    all ended REVERT_PATCH — the entry backed the reopen and the reopen twice
    failed to pay back, so the entry's reading is suspect (demotion is the
    harvest round's job; this only names the suspects)."""
    cites: dict[str, list[str]] = {}
    for rec in sorted(meta_records, key=lambda r: _get(r, "meta_step", 0) or 0):
        name = kb_cite_name(_get(rec, "reopen_price", "") or "")
        if name:
            cites.setdefault(name, []).append(_get(rec, "outcome", "") or "")
    return [name for name, outs in cites.items()
            if len(outs) >= streak and all(o == "REVERT_PATCH" for o in outs[-streak:])]


# --- calibration aggregation ---------------------------------------------------

_REOPEN_TAIL = 5          # the payback window (brief: N=5)
_LEDGER_TAIL = 10         # the reversed/faded window (brief: last 10 lever rows)
_STREAK_TAIL = 3          # directions quoted from a consecutive-REVERT_PATCH run
_STALE_CAP = 2            # stale-entry warnings rendered per menu

_REVERSED_NOTE = "the sign flipped"


def _reopens(meta_records: list) -> list:
    """Records that reopened a delivered decision (they carry a price line)."""
    return [r for r in meta_records if (_get(r, "reopen_price", "") or "").strip()]


def _calibration_lines(meta_records: list, ledger_rows: list) -> list[str]:
    lines: list[str] = []
    reopens = _reopens(meta_records)[-_REOPEN_TAIL:]
    if reopens:
        paid = sum(1 for r in reopens if _get(r, "outcome") == "KEEP_PATCH")
        lines.append(f"- reopens (last {len(reopens)}): {paid}/{len(reopens)} paid back "
                     f"(KEEP_PATCH); use a cheap probe on the next one first")
    tail = [r for r in ledger_rows[-_LEDGER_TAIL:] if isinstance(r, dict)]
    if tail:
        flip = sum(1 for rec in tail if str(rec.get("class")) == "faded"
                   or _REVERSED_NOTE in str(rec.get("condition", "")))
        lines.append(f"- ledger tail (last {len(tail)}): {flip}/{len(tail)} rows faded/reversed "
                     f"— old prices do not carry across trees; re-price before citing")
    streak: list[str] = []
    for rec in reversed(meta_records):
        if _get(rec, "outcome") != "REVERT_PATCH":
            break
        d = str(_get(rec, "patch_direction", "") or "").strip()
        if d:
            streak.append(d[:70] + ("…" if len(d) > 70 else ""))
        if len(streak) >= _STREAK_TAIL:
            break
    if streak:
        lines.append(f"- REVERT streak ({len(streak)}): " + "; ".join(f'"{d}"' for d in streak)
                     + " — dead ends; do not retry as-is")
    for name in stale_kb_entries(meta_records)[:_STALE_CAP]:
        lines.append(f"- KB entry {name} possibly stale (last {streak_n(meta_records, name)} "
                     f"reopens citing it REVERTed) — schedule a harvest review")
    return lines


def streak_n(meta_records: list, name: str, streak: int = 2) -> int:
    """How many of the entry's last reopen citations are read when it goes
    stale (the streak length, for the rendered warning)."""
    outs = [_get(r, "outcome") for r in sorted(meta_records, key=lambda r: _get(r, "meta_step", 0) or 0)
            if kb_cite_name(_get(r, "reopen_price", "") or "") == name]
    return min(streak, len(outs))


# --- ledger filter -------------------------------------------------------------

_LEDGER_LINES_MAX = 4     # of the <=10 LEVER-MENU lines


def _ledger_lines(rows: list, arch: str, op: str) -> list[str]:
    """This arch's dead levers, same-op findings first — rendered as measured
    exclusions (the one place the menu carries numbers, and they are measured).
    Non-dict rows (a corrupt ledger line read tolerantly) contribute nothing."""
    same, other = [], []
    for rec in rows:
        if not isinstance(rec, dict) or str(rec.get("arch") or "") != arch:
            continue
        (same if str(rec.get("op_family") or "") == op else other).append(rec)
    out = []
    for rec in (same + other)[:_LEDGER_LINES_MAX]:
        d = rec.get("delta_pct")
        out.append(f"- LEDGER {rec.get('lever')} {rec.get('class')} "
                   f"Δ{'n/a' if d is None else format(d, '+.2f')}% — do not re-propose "
                   f"unless you name what changed on the current tree")
    return out


# --- block assembly ------------------------------------------------------------

_HEADER = ("## KB-MENU (deterministic retrieval aid — a pointer is not an answer, "
           "the menu is not a price)")


def build_kb_menu(*, readings: dict, resusage: dict | None = None,
                  ledger_rows: list | None = None, meta_records: list | None = None,
                  index_text: str = "", kb_label: str = "", op: str = "",
                  recipe_dir: str = "") -> str:
    """The KB-MENU block, or ``""`` when no section yields a line (the driver
    injects nothing rather than an empty shell). Pure: the same inputs render
    the same text, byte for byte.

    ``recipe_dir`` (absolute) rewrites the recipe-library pointers — the
    teacher runs with cwd = the student workspace, so a bare repo-relative
    ``skills/probe-recipes/...`` would not resolve; the driver passes the
    repo's absolute recipe directory and the menu names the scripts there."""
    readings = readings or {}
    resusage = resusage or {}
    rows = parse_symptom_rows(index_text) if index_text else []

    hints: list[str] = []
    lever_forms: list[str] = []
    shown: set[str] = set()
    for rule in RULES:
        if not rule.when(readings, resusage):
            continue
        trigger = rule.label(readings, resusage)
        entry = snippet = ""
        if rule.keywords:
            hit = _match_row(rows, rule.keywords)
            if hit is not None and hit.entry:
                entry = hit.entry
                # one pointer per row: a second rule matching the same line
                # wastes the hint budget on a duplicate
                if hit.text not in shown:
                    shown.add(hit.text)
                    snippet = hit.snippet
        if entry and snippet:
            loc = f"hw_kb/{kb_label}/{entry}" if kb_label else entry
            hints.append(f"- {trigger}: {loc} — \"{snippet}\"")
        lever_forms.append((rule, trigger, entry))

    lever_lines: list[str] = []
    seen_forms: set[str] = set()
    for rule, trigger, entry in lever_forms:
        if rule.lever_hint in seen_forms:
            continue
        seen_forms.add(rule.lever_hint)
        kb = f"; KB: {entry}" if entry else ""
        hint = rule.lever_hint
        if recipe_dir:
            hint = hint.replace("skills/probe-recipes/", recipe_dir.rstrip("/") + "/")
        lever_lines.append(f"- {trigger} -> {hint}{kb}")
    lever_lines += _ledger_lines(ledger_rows or [], arch=str(arch_of(kb_label)),
                                 op=op) if ledger_rows else []
    lever_lines = lever_lines[:10]

    cal_lines = _calibration_lines(meta_records or [], ledger_rows or [])[:5]

    sections: list[list[str]] = []
    if hints:
        sections.append(["KB-HINTS:"] + hints[:5])
    if lever_lines:
        sections.append(["LEVER-MENU:"] + lever_lines)
    if cal_lines:
        sections.append(["CALIBRATION:"] + cal_lines)
    if not sections:
        return ""
    out = [_HEADER]
    for sec in sections:
        out += sec
    return "\n".join(out)


def arch_of(kb_label: str) -> str:
    """``cuda/sm100`` -> ``sm100`` (the ledger rows' ``arch`` field)."""
    return kb_label.split("/")[-1]
