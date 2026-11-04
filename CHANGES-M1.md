# CHANGES-M1 — the industrial memo lane design document (2026-11-04)

Branch `feat/m1-memo-lane-design` (off `origin/main` = `5347952`). Pure documentation
package — **no code, no `meta/`, no behavior change**. The only deliverable is
`docs/industrial-lane.md`, the design specification for the user-adjudicated
two-lane contract (2026-11-04): industrial production may use content-addressed
memoization (the MEMO lane, opt-in per task), academic benchmarking stays STRICT
(memoization forbidden — the numbers measure compute). The M3 wiring package
implements from it.

## What was added

- **`docs/industrial-lane.md`** — the two-lane contract specification, covering:
  1. Motivation (industrial input repetition vs benchmark "memorizing the exam";
     the G1 replay-cheat tension this adjudication resolves);
  2. Lane definitions — `STRICT` (default; today's behavior byte for byte, all
     anti-cheat checks kept, the only source of paper capability numbers) and
     `MEMO` (opt-in via `task.toml [protocol] memo_lane = true`; content-keyed
     caching legal; fresh_recheck and the future WP29 replay probe off; all
     correctness kept plus a new hard requirement: the cache's miss path must
     answer a fresh draw correctly — the recheck-child machinery is re-purposed as
     the `memo_fallback` block, FAIL = hard correctness REVERT);
  3. Lane attribute semantics — the lane is a task-level property; a run's records
     are single-lane by construction; cross-lane numbers banned from one comparison
     column; lane labeling mandatory in reports and paper;
  4. What "keep optimizing" means in MEMO — cache management + miss-path compute;
     new lever-family candidates (host-side content-keyed wrapper, device-side
     persistent cache, partial memoization) all going through the ordinary lever
     pricing / SWEEP / gate / falsified-ledger discipline;
  5. Honesty boundaries — MEMO numbers mean something only when production inputs
     repeat by value; never phrased as a compute-speedup claim ("kernel N times
     faster than X" is banned); STRICT-only paper capability numbers; the
     file-based-card fallback limitation (the G1 note) carried forward; the WP9
     no-reading posture kept verbatim;
  6. The M3 wiring specification, per file with acceptance assertions:
     `task.py` (`_KEYS["protocol"]` + `Protocol.memo_lane`), `taskcard.py`
     (problem.json stamp + `PK_MEMO_LANE` env line, STRICT submit.toml
     byte-identical), `card_template/eval/benchkit.py` (lane-gated recheck branch,
     `memo_fallback` block, MEMO-only `lane` eval stamp, sentinels unchanged, WP29
     conditional touchpoint), `loop/parse.py` (`EvalResult.lane` /
     `.memo_fallback`), `loop/driver.py` (context field, record stamping, the MEMO
     REVERT path parallel to the WP9 fresh-recheck FAIL path, one lane-gated brief
     paragraph), `inner/trajectory.py` (`IterationRecord.memo_lane`, digest
     `lane` key, report lane-column rule), `scripts/self_evolve.py` (lane
     inheritance + guard, carry lane stamp in `levers.json`, carry consumption
     drops cross-lane carry with a log line);
  7. Red lines — `measure_paired` byte-frozen; STRICT path byte-identical (default
     false, no pin modified); anti-fraud discipline text untouched; correctness
     never negotiable in either lane; no cross-lane contamination; this document is
     not an implementation license.
- **`docs/README.md`** — one index line for the new document.
- **`CHANGES-M1.md`** — this file.

## Not done here (deliberate)

- No code change anywhere (`src/`, `scripts/`, `tests/`, `card_template/`), no
  `meta/` touch, no `docs/backlog.md` edit (the G1 status update belongs to the
  chief engineer).
- No push, no merge to main; the branch tip carries the three files above and a
  clean tree.

## Verification

- `python -m pytest tests/test_branding.py -q` (the framework venv) — **8 passed**
  (one interim failure was self-inflicted and fixed: this file's first draft quoted
  the venv's on-disk name in a paste, hitting the lowercase module-path residual
  gate; rephrased, all green).
- Full-tree CJK grep (`grep -rlP '[\x{4e00}-\x{9fff}]' . --exclude-dir=.git`) —
  zero hits.
- New files carry the "The Peregrine-Kernel Authors" SPDX headers (Apache-2.0).
- Note: the design document lands at 298 lines, somewhat above the task pack's
  ~150-250 guidance; the per-file wiring section (7 files, each with change points
  and acceptance assertions) is kept complete at the cost of the line budget —
  completeness is the package's hard requirement.

<!-- SPDX-FileCopyrightText: Copyright 2026 The Peregrine-Kernel Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
