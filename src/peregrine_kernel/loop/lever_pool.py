"""WP16: the sub-noise compounding pool — the ONE implementation, pure functions.

A lever priced ``null`` is not a dead end. Alone it reads ≈ 0 inside the run's
noise band; what was never measured is its combination with a lever priced in a
LATER batch (the cross-batch compounding leak: a +0.7% lever from batch one and
a +0.7% lever from batch two never met, because the first batch's ``null``
verdicts read as "do not propose again"). The pool records those levers; when
the pool holds ≥ 2 entries and the run sits at a plateau (or the pool has aged
past ``pool_age_trigger`` meta steps), the driver prices the all-on bundle —
through the pricing job's own STACK pass, never an assumed sum.

Discipline (red lines ③-② / ③-⑤):
  * the pool is a teacher-side information and budget channel: it never
    machine-changes a verdict. KEEP / REVERT / watermark / override logic is
    untouched;
  * an empty pool regresses to the pre-WP16 behavior byte for byte (every
    consumer gates on ``pending(rows)`` being non-empty);
  * a bundle that also reads null closes the combination: its members move to
    the cross-run falsified ledger under ``class: combo-null`` and the pool
    clears. A bundle that pays is a LANDABLE-style reading for the teacher —
    landing still goes teacher → student patch → full gate.

Row schema (``notes/lever_pool.jsonl``, one JSON object per line; rows are
rewritten in place when a status changes):

    {"lever": str, "cls": "null", "delta_pct": float | null,
     "priced_on": sha, "job": str, "edits": [...], "decisions": [str],
     "regime": str | null,      # recorded verbatim, never judged (WP14's axis)
     "ts": float, "step": int, "run": str,
     "status": "pending"|"combo-null"|"priced"|"landed"}
"""

from __future__ import annotations

import json

POOL_STATUS_PENDING = "pending"
POOL_STATUS_COMBO_NULL = "combo-null"
POOL_STATUS_LANDED = "landed"

# the injected COMBO-POOL block shows at most this many pending rows (contract
# C10: the whole block stays ≤ 10 lines)
BLOCK_MAX_ROWS = 5


def parse_pool(text: str) -> list:
    """The pool rows off disk, bad lines skipped (best-effort by contract: the
    pool is a bypass — a corrupt row must never break the loop)."""
    rows: list = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("lever"):
            rows.append(rec)
    return rows


def serialize_pool(rows: list) -> str:
    """The pool's deterministic on-disk form (sorted keys, no spaces): the same
    rows always render the same bytes, so a snapshot test can pin them."""
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":")) + "\n" for r in rows)


def pending(rows: list) -> list:
    """The rows still awaiting a bundle measurement, oldest first."""
    return [r for r in rows if r.get("status") == POOL_STATUS_PENDING]


def admission(row: dict, rows: list, *, ts: float, step: int, run: str) -> dict | None:
    """The pool row a priced table's ``row`` admits, or None.

    A priced row enters the pool when it is a lever (not a subset), the tool's
    own class ladder read ``null`` (≈ 0 both ways — a dead end in isolation,
    unmeasured in combination), a banked edit recipe exists (the bundle job has
    to be able to apply it), and no pending entry of the same name is already
    in the pool. ``harmful`` / ``noop`` / ``faded`` are NOT pool material: the
    first is a backwards direction, the last two say the spelling (or the
    reading) never lived — none of them is a "null alone, untested together".
    """
    if row.get("kind") not in (None, "lever") or str(row.get("cls") or "") != "null":
        return None
    name = str(row.get("name") or "")
    edits = (row.get("spec") or {}).get("edits") if isinstance(row.get("spec"), dict) else None
    if not name or not edits:
        return None
    if any(p.get("lever") == name and p.get("status") == POOL_STATUS_PENDING for p in rows):
        return None
    return {"lever": name, "cls": "null",
            "delta_pct": row.get("delta_pct"),
            "priced_on": str(row.get("priced_on") or ""),
            "job": str(row.get("job") or ""),
            "edits": edits,
            "decisions": list(row.get("decisions") or []),
            "regime": row.get("regime"),
            "ts": float(ts), "step": int(step), "run": str(run),
            "status": POOL_STATUS_PENDING}


def pool_age_steps(rows: list, current_step: int) -> int:
    """How many meta steps the oldest pending row has waited. 0 when the pool
    is empty (an empty pool never triggers, never defers, never renders)."""
    pend = pending(rows)
    if not pend:
        return 0
    return max(0, int(current_step) - min(int(r.get("step") or 0) for r in pend))


def should_trigger(rows: list, *, plateau: bool, pool_age_trigger: int,
                   current_step: int) -> bool:
    """The bundle-pricing trigger: ≥ 2 pending rows AND (the run is at a
    plateau OR the pool has aged past ``pool_age_trigger`` steps).
    ``pool_age_trigger`` ≤ 0 disables the age path; fewer than two pending
    rows never triggers (an empty pool regresses to the pre-WP16 behavior)."""
    pend = pending(rows)
    if len(pend) < 2:
        return False
    if plateau:
        return True
    trigger = int(pool_age_trigger or 0)
    return trigger > 0 and pool_age_steps(rows, current_step) >= trigger


def consume(rows: list, names: list, *, outcome: str) -> list:
    """``rows`` with every pending row named in ``names`` marked ``outcome``
    (``combo-null`` when the bundle read null, ``landed`` when the teacher had
    the bundle delivered through the gate). Unmentioned rows pass through."""
    named = set(names)
    return [dict(r, status=outcome)
            if (r.get("status") == POOL_STATUS_PENDING and r.get("lever") in named)
            else r for r in rows]


def combo_null_ledger_rows(rows: list, *, bundle_delta_pct: float | None,
                           noise_pct: float, op_family: str, arch: str, run: str,
                           date: str) -> list:
    """The falsified-ledger rows a null BUNDLE books: one per pending row,
    ``class: combo-null`` — the combination (not the lever alone) is what the
    measurement closed. Same shape ``_append_falsified_ledger`` writes, so the
    diagnose-prompt injection reads them like any other negative reading."""
    pend = pending(rows)
    members = ", ".join(str(r.get("lever")) for r in pend)
    cond = (f"all-on bundle with the other pooled null levers ({members}) also read "
            f"≈ 0 within the {noise_pct:.2f}% floor — the combination is closed, "
            f"not just the single lever")
    if bundle_delta_pct is not None:
        cond += f"; bundle delta {bundle_delta_pct:+.2f}%"
    return [{"op_family": op_family, "arch": arch,
             "lever": str(r.get("lever")), "class": "combo-null",
             "delta_pct": r.get("delta_pct"),
             "condition": cond, "run": run, "date": date}
            for r in pend]


def bundle_levers(rows: list) -> list:
    """The pricing-job lever specs for the pool's all-on bundle: the pending
    rows' banked recipes, in pool order. The pricing tool's own STACK pass
    times the all-on — the pool adds no second stacking implementation."""
    out = []
    for r in pending(rows):
        out.append({"name": str(r.get("lever")),
                    "edits": list(r.get("edits") or []),
                    "in_stack": True})
    return out


def combo_pool_block(rows: list, *, bundle: dict | None = None,
                     trigger: bool = False, cap: int = BLOCK_MAX_ROWS) -> str:
    """The injected COMBO-POOL block (contract C10 position, ≤ 10 lines), or
    "" when the pool is empty (byte-for-byte the pre-WP16 prompt). ``bundle``
    is the STACK reading when a bundle job ran this step."""
    pend = pending(rows)
    if not pend:
        return ""
    shown = pend[:max(1, int(cap))]
    lines = [f"- `{r['lever']}`: null @`{str(r.get('priced_on'))[:10]}`"
             + (f" (regime {r['regime']})" if r.get("regime") else "")
             for r in shown]
    hidden = len(pend) - len(shown)
    if hidden > 0:
        lines.append(f"- (+{hidden} more pooled)")
    head = ("\n## Combo pool (driver-side bypass: null-priced levers whose combination "
            "with later levers was never measured)\n")
    body = "\n".join(lines) + "\n"
    rule = ("These levers are NOT dead ends: each read ≈ 0 alone, so the null rule does "
            "not close them — their combinations stay open. When the pool holds ≥ 2 entries "
            "at a plateau (or past its age trigger), the driver prices the all-on bundle "
            "automatically (one pricing job, STACK pass); a bundle that also reads null "
            "closes the combination (combo-null) and clears the pool. You may name a subset "
            "of these in a design turn earlier — measured subset readings supersede the "
            "pool's.\n")
    tail = ""
    if bundle is not None:
        tail = (f"Bundle reading (this step): all-on {bundle.get('all_pct')}, "
                f"noise {bundle.get('noise_pct')}% — "
                + ("null: the pool's combination is closed (combo-null rows booked).\n"
                   if bundle.get("null") else
                   "it pays: cite it with `priced_by=` and land it through the normal "
                   "teacher → student patch → gate channel.\n"))
    elif trigger:
        tail = "The trigger fired this step: the bundle pricing job is being run.\n"
    block = head + body + rule + tail
    if len(block.splitlines()) > 10:      # contract C10: the block stays ≤ 10 lines
        keep = block.splitlines()[:10]
        block = "\n".join(keep) + "\n"
    return block
