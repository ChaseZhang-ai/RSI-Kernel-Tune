# The declared compute precision (`[precision] compute_dtype`)

**The precision of the optimized kernel is YOUR decision.** Declare
`compute_dtype` in `task.toml` — the system optimizes at exactly the declared
precision and never goes below it. Default fp32.

This page is the task publisher's view of WP22. (The run-level
`rsi-run.toml` `[precision]` table and its one-precision-per-task merge rule
live in [run-configuration.md](run-configuration.md); everything below applies
to both spellings of the same declaration.)

## What you declare

```toml
[precision]
compute_dtype = "fp32"     # fp32 (default) | tf32 | bf16 | fp16 | fp8
```

One frozen value, chosen once per task:

- `fp32` — the default. Every task without a `[precision]` table runs exactly
  the pre-WP22 behavior, byte for byte: no prompt text, no submit.toml line,
  the historical precision-gate text, no digest stamp.
- `tf32` — tf32 matmuls are the declared arithmetic; the reference side of the
  gate runs with tf32 on (fp32 storage).
- `bf16` / `fp16` — bfloat16 / half is the declared compute dtype; inputs are
  lifted to it and the reference's float computation runs there.
- `fp8` — the declared output grid; the reference's outputs are quantized to
  `float8_e4m3fn` (fp8 has no general compute path; the row records what ran).

A typo (`"fp64"`) is a load-time refusal, never a silent fp32.

## What the system does with it

1. **The student implements at the declared dtype.** Its brief carries one
   contract sentence: *"This task's contract declares compute_dtype=<X>. Work
   within it; going below it is a contract violation (REVERT)."*
2. **The teacher designs inside it.** The same sentence reaches the teacher's
   diagnose / patch turns: the precision axis is the task's locked constant,
   **not a lever family** — it is never priced, swept, or traded for speed.
   The optimizer does not choose a precision; you did.
3. **The gate judges at the declaration.** The precision gate's reference side
   is evaluated AT the declared dtype (`e_ref` = the reference's own error at
   the declaration, against the unchanged float64 truth), and the floor is the
   declared output dtype's ulp. The relative bound — `e_cand >
   3*e_ref + floor`, the three stress profiles, the float64 truth — is
   eval/precision.py's measure, byte for byte; the declaration only moves what
   the reference side computes at.
4. **Below the declaration is a conviction, not a style point.** A kernel that
   quietly computes one grade coarser than the declaration lands past the
   bound and the strict gate REVERTs it, naming the declaration
   (`tests/redteam/test_redteam_compute_dtype.py`'s fixture pins the scene).
   Within-grade arithmetic (another reduction order, a fused multiply-add) is
   legitimate — the gate still judges the output.

## The records and the mixing ban

- Every iteration record carries `compute_dtype` (default `"fp32"`; rows from
  before WP22 read fp32), the digest is stamped with the declaration (omitted
  on fp32), and so are the run-end summary, `report.md` and the export note.
- **Numbers never mix across declarations** (the memo-lane rule): a digest
  built from rows read under different declarations is refused, not rendered;
  a carried lever price stamped with another declaration is dropped, not
  juxtaposed; `report.md` and the export say so next to their numbers.
- fp32 remains the only source of paper capability numbers; a bf16-declared
  run's speedup is a claim about bf16 arithmetic.

## Boundaries (unchanged by this feature)

- `eval/precision.py`'s measure (K_RATIO, the floor, relative-L2, the stress
  profiles, the float64 truth), `timing.py`'s `measure_paired`, and the meta
  discipline text are untouched — zero diff.
- The gate never loosens for a declaration: the declaration moves the
  reference side, and the same relative bound convicts anything past it.

<!-- SPDX-FileCopyrightText: Copyright 2026 The RSI-Kernel-Tune Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
