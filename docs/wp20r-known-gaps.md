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

- **Status**: OPEN (verified and recorded 2026-11-04, WP20R). Fixture:
  `tests/redteam/test_redteam_recon.py` — asserts the cheat's documented
  behavior (it re-passes the value recheck), following the matrix's
  open-row precedent (the fallback family's dead fast path).
- **What**: a content-keyed replay cheat that answers any input it does not
  recognize with an honest computation passes every value check the eval has:
  the judged correctness call, and the WP9 fresh-recheck child at its shifted
  seed (the re-drawn inputs are unrecognized, and the honest answer is correct
  by construction). The timed windows still measured a replay of parked
  answers, not the compute — a fake speedup ships.
- **Why the existing readings stay quiet**: the value layers are blind to the
  miss path by construction (honest is correct anywhere); the stream sentinel
  reads the replay as it reads an honest fast call (no side-stream span); the
  process census and the timer sentinel are orthogonal (no threads, no clock
  rebinding). Verified empirically against the production bench child on the
  CPU rehearsal path (PK_FRESH_RECHECK_DEVICE=cpu).
- **Minimal proposal (NOT implemented — needs adjudication)**: give the
  fresh-recheck child one cheap latency probe per timed group — run the same
  call twice on the shifted instance and compare the pair against the parent's
  reported device median for that shape. An honest kernel's two calls agree;
  a replay-faked parent number sits far below the child's honest re-time, and
  the child's own first-to-second drop (compute then replay) is a shape no
  honest kernel shows. A smaller alternative: append a latency-delta
  fingerprint beside the existing same-seed nondeterminism fingerprint (which
  only fires on a fresh-recheck FAIL today, so it never sees this form). Both
  variants are measurement-protocol changes on frozen surfaces (benchkit /
  timing protocol), so per the red lines they are a chief-engineer decision in
  a separate package — nothing was changed in WP20R.
- **Note**: for file-based input cards (recheck mode `process`, no re-draw)
  any same-content replay is value-invisible by design; G1's probe closes the
  drawn-input case, which is the dominant card shape.
- **Re-estimation trigger**: together with the backlog's global trigger (the
  first real-run A/B numbers), or immediately if the paper's measurement
  section needs this row closed.

## Open rows carried by the matrix (pre-WP20R, unchanged)

- **try/except dead fast path** (`tests/redteam/test_redteam_fallback.py`): no
  static layer scans the fallback shape — the honest path computes, every
  defense passes. A static shape scanner is future work (frozen: new-mechanism
  freeze order).
- **process census is accounting-only**
  (`tests/redteam/test_redteam_timer_proc.py`): a thread/child anomaly never
  flips a verdict (convicting needs a per-job-image baseline the loop keys do
  not carry); recorded posture.
