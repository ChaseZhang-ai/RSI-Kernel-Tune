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
    "kb_cite_name", "SymptomRow", "fired_rule_names", "hit_record",
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


def _num_or(readings: dict, key: str, absent):
    """The reading, or ``absent`` only when the key carries no number at all. A
    measured 0.0 is a measurement -- for l2_hit_rate_pct / compute_sol_pct /
    waves_per_sm it is the strongest form of the very symptoms these rules
    catch -- so it must compare as itself, never be swapped for the
    absent-value sentinel (the tensor_pct rule's ``is not None`` is the same
    reading, spelled out)."""
    v = _num(readings, key)
    return absent if v is None else v


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
        lambda r, ru: _num_or(r, "l2_hit_rate_pct", 100.0) < L2_LOW_PCT,
        lambda r, ru: f"l2_hit_rate_pct={_fmt(_num(r, 'l2_hit_rate_pct'))} < {L2_LOW_PCT:g}",
        ("swizzle", "reference"),
        "tiling-order / swizzle / raster levers (check the working set against L2 first)",
    ),
    Rule(
        "waves-low",
        lambda r, ru: _num_or(r, "waves_per_sm", 1.0) < WAVES_LOW,
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
        lambda r, ru: _num_or(r, "compute_sol_pct", 100.0) < SOL_WARM_PCT
        and _num_or(r, "memory_sol_pct", 100.0) < SOL_WARM_PCT,
        lambda r, ru: (f"both SOL low (compute={_fmt(_num(r, 'compute_sol_pct'))}%, "
                       f"memory={_fmt(_num(r, 'memory_sol_pct'))}%)"),
        None,
        "latency-bound reading: price per-instruction stall-ledger rows, not aggregate ratios",
    ),
    Rule(
        "unbound-attribution",
        lambda r, ru: _num_or(r, "compute_sol_pct", 100.0) < SOL_WARM_PCT
        and _num_or(r, "memory_sol_pct", 100.0) < SOL_WARM_PCT
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
    # --- WP18: symptom keys for the beyond-reference supply (hw_kb/CANDIDATES.md
    # second batch). Appended only (contract C10): the pointers name the nearest
    # existing index row where the candidate is not yet certified, so a tree
    # without the row simply renders the LEVER-MENU form — never a made-up
    # pointer. Thresholds stay coarse: the menu triages, the teacher judges.
    Rule(
        "wave-tail",
        lambda r, ru: (lambda w: w is not None and w >= 1.0 and (w % 1.0) >= 0.05)(
            _num(r, "waves_per_sm")),
        lambda r, ru: f"waves_per_sm={_fmt(_num(r, 'waves_per_sm'))} (a non-integer wave count: "
                      f"the fractional part is a tail wave)",
        ("persistent",),
        "tail-wave governance levers (persistent residency to absorb the tail / tile height vs the "
        "SM count)",
    ),
    Rule(
        "tma-multicast",
        lambda r, ru: (_num(r, "memory_sol_pct") or 0.0) >= 80.0,
        lambda r, ru: f"memory_sol_pct={_fmt(_num(r, 'memory_sol_pct'))}% (at the bandwidth roof)",
        ("multicast",),
        "cluster multicast / pair-load levers (one gmem read fanned out to many SMEMs)",
    ),
    Rule(
        "warp-spec-pipeline",
        lambda r, ru: (_num(r, "compute_sol_pct") or 0.0) >= 70.0
        and (_num(r, "memory_sol_pct") or 0.0) >= 70.0,
        lambda r, ru: (f"both SOL high (compute={_fmt(_num(r, 'compute_sol_pct'))}%, "
                       f"memory={_fmt(_num(r, 'memory_sol_pct'))}%)"),
        ("producer",),
        "producer/consumer warp-specialization pipeline levers (stage the roles so supply and "
        "compute overlap)",
    ),
    Rule(
        "prefetch-lead",
        lambda r, ru: _num_or(r, "l2_hit_rate_pct", 100.0) < 20.0,
        lambda r, ru: f"l2_hit_rate_pct={_fmt(_num(r, 'l2_hit_rate_pct'))} < 20 (almost no L2 reuse)",
        ("prefetch",),
        "bulk-prefetch lead levers (start the next tile's L2 fill early; read the L2 hit rate so "
        "the prefetch does not evict data with real reuse)",
    ),
    Rule(
        "launch-bounds",
        lambda r, ru: (_num(r, "regs_per_thread") or 0) >= 96
        and (_first(_num(r, "spill_st_sectors"), _num(r, "spill_ld_sectors")) or 0) == 0
        and (_num(ru, "cstack") or 0) == 0,
        lambda r, ru: f"high regs with no spill (regs_per_thread={_fmt(_num(r, 'regs_per_thread'))})",
        ("setmaxnreg",),
        "launch-bounds / controlled-spill variants (a measured spill-for-registers trade can win; "
        "price it, do not assume)",
    ),
    Rule(
        "l2-policy-window",
        lambda r, ru: (_num(r, "dram_pct") or 0.0) >= 80.0
        and (_num(r, "l2_hit_rate_pct") or 0.0) >= 45.0,
        lambda r, ru: (f"dram_pct={_fmt(_num(r, 'dram_pct'))}% with l2_hit_rate_pct="
                       f"{_fmt(_num(r, 'l2_hit_rate_pct'))}% (streaming and reused data competing "
                       f"for L2)"),
        ("L2",),
        "L2 access-policy window levers (only where streaming and persistent data compete for L2; "
        "near the roof without competition it reads as no-op)",
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


def _prev_resusage(readings: dict) -> dict:
    """The build-time resource fields of a raw profiler dict, namespaced the
    same way the driver delivers the newest reading's (``resusage_*`` keys
    stripped of their prefix; ``resusage_*_kernel`` excluded)."""
    return {k.removeprefix("resusage_"): v for k, v in readings.items()
            if k.startswith("resusage_") and not k.endswith("_kernel")}


# --- block assembly ------------------------------------------------------------

_HEADER = ("## KB-MENU (deterministic retrieval aid — a pointer is not an answer, "
           "the menu is not a price)")


def build_kb_menu(*, readings: dict, resusage: dict | None = None,
                  ledger_rows: list | None = None, meta_records: list | None = None,
                  index_text: str = "", kb_label: str = "", op: str = "",
                  recipe_dir: str = "", prev_readings: dict | None = None) -> str:
    """The KB-MENU block, or ``""`` when no section yields a line (the driver
    injects nothing rather than an empty shell). Pure: the same inputs render
    the same text, byte for byte.

    ``prev_readings`` (WP18) is the second-newest profiler reading. A rule that
    fired THERE but not on the newest reading renders once with a
    ``[prev reading]`` marker — the bottleneck changed phase mid-run, and a menu
    that triages on the newest reading alone would quietly drop the phase the
    teacher may still be mid-decision on. A rule firing on both renders as
    before (persistent symptom), and ``None`` (one reading on file, or the
    driver's default) renders byte-identical output to the pre-WP18 form.

    ``recipe_dir`` (absolute) rewrites the recipe-library pointers — the
    teacher runs with cwd = the student workspace, so a bare repo-relative
    ``skills/probe-recipes/...`` would not resolve; the driver passes the
    repo's absolute recipe directory and the menu names the scripts there."""
    readings = readings or {}
    resusage = resusage or {}
    prev = prev_readings or {}
    rows = parse_symptom_rows(index_text) if index_text else []

    hints: list[str] = []
    lever_forms: list[str] = []
    shown: set[str] = set()
    for rule in RULES:
        fires = rule.when(readings, resusage)
        fires_prev = bool(prev) and rule.when(prev, _prev_resusage(prev))
        if not fires and not fires_prev:
            continue
        # the trigger line quotes the reading that fired it (newest first) and
        # marks a prev-only fire: the symptom moved, it did not vanish
        face, ru_face = (readings, resusage) if fires else (prev, _prev_resusage(prev))
        trigger = rule.label(face, ru_face)
        if fires_prev and not fires:
            trigger += " [prev reading]"
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


# --- hit telemetry (WP18; task 5) ----------------------------------------------
#
# kbmenu stays pure (contract C7): the counters below only EVALUATE the rule
# table against the readings the driver is about to render — the driver owns
# the file, the timestamps and the run-end aggregation.

def fired_rule_names(readings: dict, resusage: dict | None = None,
                     prev_readings: dict | None = None) -> list[str]:
    """The rule names that fire for these readings, table order — the pure
    half of the hit telemetry: the driver renders a menu, evaluates this on the
    same inputs, and appends one jsonl row. ``prev_readings`` counts a rule
    that fires only there as a hit too (the prev-marker line the teacher
    reads)."""
    resusage = resusage or {}
    prev = prev_readings or {}
    prev_ru = _prev_resusage(prev) if prev else {}
    return [r.name for r in RULES
            if r.when(readings, resusage) or (prev and r.when(prev, prev_ru))]


def hit_record(*, kb_label: str, op: str, rules: list[str], n_hints: int,
               n_lever_lines: int, n_calibration: int) -> dict:
    """One ``evidence/kbmenu_hits.jsonl`` row: which rules fired for this
    render and how many lines each menu section carried. Objective telemetry
    about the DRIVER's own rendering — nothing here is a claim about the
    kernel, and nothing reads it back into a decision; the run-end aggregation
    is the first-hand data on whether KB-MENU pulls its weight (A/B note in
    CHANGES-WP18)."""
    return {"kb_label": kb_label, "op": op, "rules": list(rules),
            "hints": int(n_hints), "lever_lines": int(n_lever_lines),
            "calibration_lines": int(n_calibration)}
