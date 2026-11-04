========================================================================
WP16 dry-run — all injected blocks, fixture data, deterministic
========================================================================

### 1. COMBO-POOL block (empty pool renders NOTHING — byte-identity)

## Combo pool (driver-side bypass: null-priced levers whose combination with later levers was never measured)
- `tilenh8`: null @`0123456789`
- `kvstage2`: null @`0123456789`
- `regsplit_184_120`: null @`0123456789`
These levers are NOT dead ends: each read ≈ 0 alone, so the null rule does not close them — their combinations stay open. When the pool holds ≥ 2 entries at a plateau (or past its age trigger), the driver prices the all-on bundle automatically (one pricing job, STACK pass); a bundle that also reads null closes the combination (combo-null) and clears the pool. You may name a subset of these in a design turn earlier — measured subset readings supersede the pool's.

### with a null bundle reading:

## Combo pool (driver-side bypass: null-priced levers whose combination with later levers was never measured)
- `tilenh8`: null @`0123456789`
- `kvstage2`: null @`0123456789`
- `regsplit_184_120`: null @`0123456789`
These levers are NOT dead ends: each read ≈ 0 alone, so the null rule does not close them — their combinations stay open. When the pool holds ≥ 2 entries at a plateau (or past its age trigger), the driver prices the all-on bundle automatically (one pricing job, STACK pass); a bundle that also reads null closes the combination (combo-null) and clears the pool. You may name a subset of these in a design turn earlier — measured subset readings supersede the pool's.
Bundle reading (this step): all-on 0.2, noise 0.5% — null: the pool's combination is closed (combo-null rows booked).


### 2. Pattern search readings table (stagnation trigger)

## Pattern search (driver-run shrink grid on best `0123456789` — readings only)

| point | same-job ratio | Δ vs base | ms | floor | ok |
|---|---|---|---|---|---|
| `tile_16` | 0.912 | -1.85% | 3.611 | +0.51% | pass |
| `tile_32` | 0.948 | +0.42% | 3.385 | +0.51% | pass |
| `tile_48` | 0.939 | +1.96% | 3.412 | +0.63% | pass |
| `tile_64` | 0.953 | +2.31% | 3.361 | +0.63% | pass |
| `tile_96` | 0.951 | +1.02% | 3.39 | +0.58% | pass |
| `tile_128` | 0.941 | -0.25% | 3.455 | +0.58% | pass |

**LANDABLE: pattern-search `tile_64` (axis tile = 64) expected=+2.31%** over 3 contraction stage(s), 3 job(s), armed by stagnation. Teacher: cite it in a milestone patch (the winner's edit is the recipe's `old` → the winning value); the student delivers it as an ordinary decision and the standard eval gate judges the landing — the driver never applies kernel edits itself.


### 3. STAGNATION restart target (appended AFTER the WP10 block; the block itself is untouched)

# --- WP16 --- restart target (driver-read: an optional route, never a machine switch)
- runner-up #2: `D2.3` tree `bbbbbbbbbb` gate 2.03 ms (+1.50% vs the best — its eval and NCU are already paid).
- Re-basing the run on that tree is a legal structural answer here: point the milestone patch at it and the student delivers on top of it.
- Revival always re-verifies (red line ③-②): the revived node re-passes the full suite with its next gated delivery — watermark / clock-drift / fresh-recheck apply as to any delivery — before any of its readings re-enter best, plateau or pricing.

### (no #2 -> nothing; pool empty -> nothing)

### 4. EXPIRED row (rides the existing Price-tag review block)

## Price-tag review (measured by the driver in the inner loop: when the best tree changes, the levers the milestone cites are re-measured)
These levers' price tags were written into the milestone (`lever:<name>` citations); below are their re-measurements on the later best trees. **Citations that came back REVERSED / FADED have been marked `SUSPENDED` in the milestone by the driver, and the student is forbidden to land them** — this diagnosis / patch must dispose of every one: retract the rung, rewrite it at the new price, or reopen at the new price; no SUSPENDED mark may be left in the milestone.
- `win`: was add +2.00% / remove n/a @`0123456789` (job 812345) → now n/a @`fedcba9876` → **EXPIRED** (add +2.00% > noise 0.50% on the old tree; the milestone does not cite it and no fresh reading covers it on the new best — re-price it before citing it there)


### 5. Lever coupling graph (rendered inside the existing price block)

## Pricing table (measured by the driver on the best tree: add-one / remove-one / all-on, same-job paired ratios)
LANDABLE: all-pos levers=tilenh8,kvstage2 expected=+2.10% (floor=0.50%) priced_on= — the teacher can land these levers as one merged decision line with a single milestone patch; the student delivers once per rung (ordinary bundle delivery, judged by the standard eval gate).
| lever | add-one | remove-one | class | Δspin(ns, +L) | Δspin(ns, all−L) | decisions |
|---|---|---|---|---|---|---|
| `tilenh8` | +1.20% | -0.40% | **independent** | 5.0 | 1.0 | D1.1 |
| `kvstage2` | +0.80% | -0.20% | **independent** | 3.0 | 0.5 | D1.2 |
| `regsplit_184_120` | +0.10% | +0.10% | **null** | 0.0 | 0.0 | D2.1 |
| `lanepct` | -1.10% | +0.00% | **harmful** | -4.0 | 0.0 | D2.2 |

all-on = +0.90%; sum of add-ones = +1.90%; coupling = -1.00% (negative = they eat each other, positive = they enable each other). base 3.42 ms → all 3.4 ms; spin exposure None → None ns; tensor None → None%.
**Harmful levers `lanepct` (negative add-one, and all-on does not depend on them) went into all-on**, and they skew every `remove` reading (remove is measured on the all-on that contains them) — all-on without them, `all-pos` = +2.10% (members tilenh8, kvstage2); all-on and the class column go by it.

**How to read it (hard rules)**: `independent` = this group pays on its own; fix in add order. `coupled` = ≈ 0 alone but taking it out of all-on loses — you split a group that must go in together: it is the companion of another candidate in the table, and the diagnosis **must merge it with the candidate that depends on it into one ⊗ package** (price the package with −remove); it must not be judged useless on its own. `dependent` = the add-one pays but taking it out of all-on loses nothing — someone else swallowed the gain; rank it last. `null` = ≈ 0 both ways; drop it. `harmful` = negative add-one **and** taking it out of all-on loses nothing — the direction is backwards and must not be proposed as a direction again (base never had it; its reverse is not a gain). A negative add-one whose removal from all-on does lose is `coupled`, not harmful — it costs on its own but all-on cannot do without it; treat it as a ⊗ package. `noop` = the edit did not change the cubin; the spelling is ineffective. `broken` = the variants containing it failed (did not compile / were incorrect): this lever **was not priced**, and the all-on and remove-one readings that contain it are n/a too — the direction is not falsified, the spelling is rejected; either rewrite it and propose it again, or mark it unpriced in the diagnosis. `unstacked` = the add-one has a reading but the all-on variant failed correctness, so its coupling with the others is unmeasured. **Every price the diagnosis cites must be a measured value from this table** (add for an independent item; −remove, a subset delta or an all-on difference for a package); write `priced_by=<lever or subset name from the table>` to point at it; a quantity the table does not have may only be marked `est:<x>%` and must not be cited as measured. The class is derived mechanically by the driver from add-one / remove-one: `coupled` only says that **some** lever depends on it, not which one — the subset deltas you specified are the judge; when a subset and the class disagree, the subset wins. The ceiling of the pricing table: the spin exposure that all-on still leaves belongs to something outside this batch of candidates; the diagnosis must say whose it is.

**Lever coupling graph (driver-read from the recipes)**:
- `tilenh8` → `kvstage2` (requires): price and land `tilenh8` first — a `kvstage2` citation with `tilenh8` unlanded is unbacked.
- `lanepct` >< `regsplit_184_120` (mutually exclusive): separate deliveries, never one bundle; an all-on above that contains both says this edge went stale.

