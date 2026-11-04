# CHANGES-M2 — MEMO-lane memoization demo fixture (2026-11-04)

Branch `feat/m2-memo-fixture` (off `origin/main` = `5347952`). Scope per the M2
task pack (handoff 141; design context: handoff 140, the MEMO industrial lane):
pre-build a unit-testable, self-contained demo asset for the MEMO lane — a
content-addressed memoization wrapper and its pure unit tests — so the lane's
delivery shape is pinned and reusable as an integration fixture when the lane
wiring lands in M3. This package is asset pre-positioning, not lane
implementation: no driver, benchkit, timing, meta, task, hw_kb, or kernels code
was touched, and nothing is wired into any measured path.

## Results

- **Baseline:** 2622 passed (recorded before any change, `tests/ -q`).
- **Final:** 2638 passed (2622 + 16 new), 0 failed, 0 skipped — see
  "Verification" below.
- No GPU used (CPU-only torch); the fixture and its tests are CPU-only by
  construction.

## Files added

| Path | What |
|---|---|
| `tests/fixtures/memo_demo.py` | the demo asset: `MemoKernel`, a content-addressed memoization wrapper with an injectable fallback kernel (`__call__(*inputs) -> output`), plus `buffer_digest`/`content_key` (shape + dtype + streamed single-pass sha256 over each buffer's bytes) |
| `tests/test_memo_demo.py` | 16 pure unit tests pinning the contract (below) |
| `CHANGES-M2.md` | this file |

Nothing else was touched: no `src/`, no `benchkit`, no driver, no `timing`,
no `meta/`, no `task.py`, no `hw_kb/`, no `kernels/`, no other tests.

## What the fixture demonstrates

`MemoKernel` is the reference delivery shape of the MEMO lane:

- **Content-addressed key**: per input buffer, sha256 over shape + dtype +
  raw bytes, streamed in fixed 1 Mi chunks (O(data), single pass, no full
  buffer copy for the hash); per-buffer digests combine into one key. Equal
  values at different memory addresses share a key; any value, dtype, shape,
  or input-order change produces a different key.
- **Hit path computes nothing**: a content-equal repeat serves the cached
  output (returned as a fresh clone so callers cannot poison the cache); the
  wrapped kernel is not invoked — observable through the wrapper's
  `compute_calls` counter and the test's injected counting kernel.
- **Miss path is honest**: an unseen input is computed by the injected
  fallback kernel and cached. Correctness on any input is the iron rule.
- **Cache management surface**: optional FIFO-bounded `max_entries`,
  `clear()`, `stats()` — the levers the MEMO lane optimizes after
  memoization lands (hash overhead, hit policy, footprint).

The module docstring states the contract boundary explicitly: this shape is
the MEMO lane's legitimate deliverable; the same shape under the STRICT lane
is answer memorization and is exactly what the WP29 replay probe exists to
capture. The full two-lane contract is `docs/industrial-lane.md` (M1
deliverable; the path is referenced before that document lands). Stated
limitations: single-threaded by contract (the measured MEMO regime is a
sequential repeated-input stream, so the cache has no locking), tensor-only
inputs (non-tensor arguments are rejected rather than weakly keyed).

## Test inventory (16)

- Hit correctness: cached output allclose on repeat, injected compute counter
  frozen (3 tests: single hit, repeated hits + stats snapshot, hit-output
  independence from caller mutation).
- Fallback correctness: novel inputs (changed seed / one-element change /
  changed operand) match the unwrapped kernel's honest output with the
  compute counter incremented (4 tests).
- Latency signature: a deliberately slow toy kernel (sleep 60 ms) gives
  first-call wall >= 50 ms and best-of-5 hit wall < 5 ms — coarse CPU
  wall-clock assertions, no nanosecond-grade fragility (1 test).
- Key stability / collision resistance: equal values at different addresses
  share one key end-to-end; different values, dtype reinterpretation of
  identical bytes, reshapes, and input permutation all produce distinct keys;
  non-tensor input fails loudly (6 tests).
- Cache management: FIFO eviction recomputes honestly, `clear()` resets state,
  constructor rejects non-callable kernels and non-positive bounds (3 tests).

No test is skipped or xfailed; all are deterministic and single-threaded.

## Verification

(both commands run with the repo virtualenv's python, as in every package)

```
$ python -m pytest tests/test_memo_demo.py tests/test_branding.py -q
........................  [100%]  24 passed (16 new + 8 branding)
$ python -m pytest -q
2638 passed
```

Branding gate green (all outputs English; zero CJK, zero forbidden-brand
residuals in the new files). Working tree clean after commit; branch
`feat/m2-memo-fixture` is local only — not pushed, not merged (integration
and lane wiring are collected separately).

<!-- SPDX-FileCopyrightText: Copyright 2026 The Peregrine-Kernel Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
