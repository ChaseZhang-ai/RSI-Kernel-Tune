# WP20R known gaps — the red-team matrix's open rows

The red-team matrix (contract C16, `tests/redteam/`) exists to prove the
anti-fraud defenses catch what they claim to catch. Where a cheat form is NOT
caught, the posture is: record it as an open row (a fixture asserts the cheat's
documented behavior, so the hole stays regression-visible), never claim a
defense, touch no frozen surface (`eval/timing.py`, `eval/benchkit.py`, gate
judgment logic), and route the fix through an adjudicated package. This page is
that record for the WP20R delivery (2026-11-04). It feeds the paper's
measurement-integrity section via the attack -> defense -> status table in
`CHANGES-WP20R.md`.

## G1 — the replay cheat whose miss path recomputes honestly

- **Status**: CLOSED (implemented 2026-11-04, WP29, by explicit user
  adjudication — "fill the gap directly, add a switch"; the WP20R proposal
  below was the adjudicated shape). Defense: the **replay probe**
  (`src/peregrine_kernel/card_template/eval/benchkit.py`, the WP29 section;
  package record `CHANGES-WP29.md`). Fixture:
  `tests/redteam/test_redteam_recon.py` — the value-blindness assertions
  stay (they document why the values cannot see this form), and the WP29
  tests flip the fixture from "documented pass" to "caught": the probe reads
  the honest miss path on fresh draws and the assembly lands `suspected`
  (x570 vs threshold 3.0 on the CPU rehearsal), which the driver turns into
  a fresh_recheck-same suspicious REVERT.
- **What it was**: a content-keyed replay cheat that answers any input it
  does not recognize with an honest computation passed every value check the
  eval has: the judged correctness call, and the WP9 fresh-recheck child at
  its shifted seed (the re-drawn inputs are unrecognized, and the honest
  answer is correct by construction). The timed windows still measured a
  replay of parked answers, not the compute — a fake speedup shipped.
- **Why the value layers stayed quiet**: the value layers are blind to the
  miss path by construction (honest is correct anywhere); the stream sentinel
  reads the replay as it reads an honest fast call (no side-stream span); the
  process census and the timer sentinel are orthogonal (no threads, no clock
  rebinding). The probe reads the one channel the cheat cannot fake without
  being honest: its own miss path's SPEED.
- **The defense (WP29)**: after ALL timing windows close (and only on a
  passing kernel with a performance block), the bench spawns a child process
  (the fresh-recheck child's spawn shape) that draws an independent THIRD
  input set per timed group — seed shift 104729, disjoint from the judged
  draw (+0) and the recheck's (+7919) — burns in (mandatory: unfamiliar-input
  cold effects are the false-positive surface), and measures the
  honest-compute latency in the main timing's device-event protocol (one
  event pair around back-to-back fresh-draw calls, divided by the calls;
  median over the windows). Every timed probe call draws fresh content, so a
  content-keyed store recognizes nothing and every probe call sits on the
  honest path. The parent compares per shape:
  `probe_median > PK_REPLAY_PROBE_RATIO x judged_median` (default 3.0,
  conservative — better to miss a cheat than to convict an honest kernel;
  `PK_REPLAY_PROBE_RATIO` overrides) -> `replay_probe: suspected` on the eval
  JSON, and the driver turns it into a fresh_recheck-same suspicious REVERT
  carrying the block with both latency numbers. Switch: `PK_REPLAY_PROBE`
  (default ON per the adjudication; `=0` fully off — the bench is
  byte-identical to pre-WP29). Unavailable path (the fresh_recheck posture):
  file-based inputs (no content re-draw), a non-finite probe draw, or any
  probe failure — including a cheat that crashes on unrecognized input —
  reads `unavailable`, never a FAIL. timing.py and the window protocol are
  untouched; the probe never runs inside a timing window.
- **Residual false-positive surface, by design**: an honest kernel whose
  latency depends on the input VALUES (a sparse-adaptive shape) with a
  natural >3x spread between draws can read suspected; a cheat that crashes
  or hangs on unrecognized input degrades the probe to `unavailable`. The
  conservative threshold, the mandatory burn-in and the `unavailable` escape
  hatch are the guardrails; `CHANGES-WP29.md` carries the full discussion.
- **Note**: for file-based input cards (recheck mode `process`, no re-draw)
  any same-content replay is value-invisible by design; the probe reads
  `unavailable` there and closes the drawn-input case, which is the dominant
  card shape.
- **Verification**: CPU rehearsal — real probe child on the recon fixture,
  judged(replay) 0.0028 ms vs probe(honest) 1.6003 ms -> x570.5 suspected;
  honest and bounded data-sensitive honest kernels pass; crash-on-miss and
  file-based cards read unavailable; switch-off byte-golden. Real-GPU
  end-to-end (3060): see `CHANGES-WP29.md`.

## Open rows carried by the matrix (pre-WP20R, unchanged)

- **try/except dead fast path** (`tests/redteam/test_redteam_fallback.py`): no
  static layer scans the fallback shape — the honest path computes, every
  defense passes. A static shape scanner is future work (frozen: new-mechanism
  freeze order).
- **process census is accounting-only**
  (`tests/redteam/test_redteam_timer_proc.py`): a thread/child anomaly never
  flips a verdict (convicting needs a per-job-image baseline the loop keys do
  not carry); recorded posture.
