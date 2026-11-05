# CHANGES-P22 — the declared compute precision (WP22)

Package: the declared compute precision, delivered per the corrected design
(user clarification, 2026-11-05). Supersedes the rejected "precision lane"
concept entirely.

## Approval record

- **2026-11-04 — rejected on principle by the user** (backlog C1): an
  optimizer-chosen "spendable precision budget" is forbidden — "whatever
  precision the user demands must be written, not one bit less."
- **2026-11-05 — approved and opened by the user ("user approved")**, with the
  design inverted: the precision is **the task publisher's declaration**, one
  frozen value — the system optimizes at exactly the declared dtype and never
  goes below it; the optimizer never picks, prices, sweeps, or trades a
  precision. The prior rejection stands for what it named; this package is the
  inverted design landing as WP22.

## The design, per the corrected contract

1. `[precision]` in `task.toml` declares ONE value: `compute_dtype = "fp32"`
   (default) | `tf32` | `bf16` | `fp16` | `fp8`. Not a ladder, not a lever
   family. A typo is a load-time refusal (`TaskError`), never a silent fp32.
2. Semantics: the student implements at the declared dtype; the precision
   gate's reference side is evaluated AT the declaration (`e_ref` = the
   reference's own error there, against the unchanged float64 truth); the
   floor is the declared output dtype's ulp; the relative bound
   `3*e_ref + floor` is unchanged. eval/precision.py's measure, the three
   stress profiles, `timing.py`'s `measure_paired` and the meta discipline
   text: zero diff.
3. Contract text: arithmetic below the declaration is a contract violation —
   the strict gate REVERTs it, naming the declaration; within-grade reduced
   precision is legitimate and the gate still judges the output. The red-team
   fixture pins a detectable below-declaration scene (an fp8 kernel on an
   fp16 declaration) and its at-declaration control.
4. No precision-axis exploration: the teacher's diagnose / patch prompts carry
   one sentence ("the precision axis is the task's locked constant, not a
   lever"); the student's prompt carries the one-sentence contract. A default
   (fp32) run renders zero new prompt bytes anywhere.
5. README states it verbatim, above Quick start: "The precision of the
   optimized kernel is YOUR decision. Declare `compute_dtype` in task.toml —
   the system optimizes at exactly the declared precision and never goes below
   it. Default fp32."
6. Untouched: the gate formulas, the stress profiles, `measure_paired`,
   `timing.py`. Default fp32 is byte-for-byte the prior behavior (golden
   records included). The trajectory record carries `compute_dtype` (renamed
   from the abandoned `precision_lane` spelling); the mixing defense stays —
   numbers read under different declarations are never one comparison.
7. Where two declarations could meet, they cannot mix: the carry guard drops
   a `levers.json` priced under another declaration; self_evolve refuses a
   chain that changes dtype and stamps `compute_dtype` into `levers.json`.

## What landed, by surface

- `src/rsi_kernel_tune/task.py` — `[precision] compute_dtype` (`_KEYS`,
  `COMPUTE_DTYPES`, `Precision`, `_read_precision`, `dump_task_toml`).
- `src/rsi_kernel_tune/runconfig.py` / `scripts/` + `parallelrun.py` — the
  run-level `rsi-run.toml [precision]` declaration, its merge rule (fills in
  an undeclared task; a conflict with the task's own declaration is refused).
- `src/rsi_kernel_tune/taskcard.py` — `RKT_COMPUTE_DTYPE` on the card's jobs
  (submit.toml `[env]`); fp32 adds no line.
- `src/rsi_kernel_tune/card_template/eval/benchkit.py` — the declaration is a
  guard knob: read once at import (garbage refused outright), consumed only
  through the unified defensive chain (`_compute_dtype()` over
  `_guard_env_get`: bench-start pin > kernel-import snapshot > live), pinned
  at `run_bench` entry (`GUARD_ENV_KEYS`), and carried into child spawns via
  `_guard_env_apply`. `_refs_at_declared` evaluates the reference at the
  declaration (tf32 matmuls on; bf16/fp16 via precision.py's own `_Upcast`
  lift; fp8 grid quantization; a failing declaration-side run falls back to
  the grid and the row says what ran).
- `src/rsi_kernel_tune/loop/precision_gate.py` — `gate_failure` names the
  declaration's contract when one is declared; the fp32 text is byte-identical.
- `src/rsi_kernel_tune/loop/driver.py` — `LoopContext.compute_dtype`; the
  student contract sentence; the teacher diagnose/patch contract sentence;
  every `IterationRecord` site stamps `compute_dtype` beside the lane; the
  carry guard; `_precision_reading` passes the declaration to `gate_failure`.
- `src/rsi_kernel_tune/inner/trajectory.py` — `IterationRecord.compute_dtype`
  (append-only; pre-WP22 files read fp32); `build_digest` stamps the
  declaration (omitted on fp32) and REFUSES a row set that mixes declarations.
- `src/rsi_kernel_tune/loop/cli.py` — `--compute-dtype` validated (a typo is a
  launch refusal); the run-end summary stamps the declaration.
- `src/rsi_kernel_tune/launch.py` — `--compute-dtype` pass-through (the task
  snapshot pins it); `report.md` names the declaration; the export carries it
  in its provenance note and `--out` confirmation. fp32 adds nothing.
- `scripts/self_evolve.py` — a chain that changes the declaration is refused;
  `levers.json` stamps `compute_dtype`.
- Docs: `docs/compute-dtype.md` (the task publisher's view), the README
  quote + highlight, `docs/run-configuration.md` (the run-level table),
  `docs/backlog.md` C1 reversed: rejected (2026-11-04) -> user approved
  (2026-11-05).

Also in this package (the defensive-env unification the declaration's reads
ride): benchkit's defensive env reads went from two rival mechanisms (the R3
bench-start pin vs the hunt4 kernel-import snapshot) to ONE trust chain —
`_guard_env_get` / `_defensive_env`: pin (None = authoritative absence) >
kernel-import snapshot (whole-world capture; a key it lacks reads absent) >
live. Every guard knob read — memo lane, recheck mode / seed shift / timeout,
replay probe switch / ratio / timeout / child marker, gate-degraded ratio,
clock state, the declared compute precision — goes through it; child spawns
build from the snapshot when one exists and layer the pin on top.

## Tests

- `tests/test_compute_dtype.py` (the landed core), `tests/redteam/
  test_redteam_compute_dtype.py` (below-declaration convicted by name,
  at-declaration passes, the floor is the declared ulp, fp32 unmarked),
  `tests/test_compute_dtype_records.py` (the record field round-trip, the
  mixing refusal, the teacher sentence, the record-site stamping count, the
  report face, and the chain-routed declaration read: pin/snapshot each
  defeat a kernel's post-import write; absence falls back to the frozen
  value; garbage falls back instead of crashing a row).
- The env-unification contract surfaces: `tests/test_guard_env_pin.py`,
  `tests/test_defense_env_snapshot.py`, `tests/test_memo_lane.py`,
  `tests/test_replay_probe.py`, `tests/redteam/` — all green; full suite
  green at delivery (count in the delivery note).

## Invariants kept

- `timing.py` byte-identical; `measure_paired` untouched; eval/precision.py's
  measure untouched (K_RATIO / floor / relative-L2 / stress / float64 truth).
- meta/ discipline text untouched; LICENSE/NOTICE untouched.
- STRICT default (no declaration) byte-for-byte the prior behavior: no env
  line, no prompt bytes, no stamps, the historical gate text, golden records.
- English output; one concern per commit; nothing pushed, nothing merged to
  a remote main by this package.

<!-- SPDX-FileCopyrightText: Copyright 2026 The RSI-Kernel-Tune Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
