# CHANGES-WP29 — the replay probe (the G1 fix)

Branch `feat/wp29-replay-probe`, cut from origin/main `692b4e4` (WP20R
merged). Package brief: handoff/139 (the replay probe, the G1 fix).

**User approval record**: the user explicitly approved this package on
2026-11-04 ("fill the gap directly, add a switch"). It is the ONLY
measurement-surface package authorized under the 2026-11-04 freeze order;
everything else in the measurement zone stays frozen.

## 1. What and why

G1 (docs/wp20r-known-gaps.md, verified and recorded in WP20R): a
content-keyed replay cheat that answers any input it does NOT recognize with
an honest computation passes every VALUE check the eval has — the judged
correctness call, and the WP9 fresh-recheck child at its shifted seed (the
miss path recomputes, correct by construction) — while the timed windows
measured replays of parked answers. A fake speedup ships.

The probe closes it on the channel the cheat cannot fake without being
honest: the SPEED of its own miss path. After ALL timing windows close, the
bench measures the honest-compute latency on fresh input draws and compares
it against the judged median the windows reported:

    probe_median > PK_REPLAY_PROBE_RATIO x judged_median  ->  suspected

- an honest kernel answers "same shape, same distribution, fresh values only"
  inputs at the same latency magnitude as its judged inputs;
- a replay cheat's honest miss path sits far above its judged median (which a
  replay produced).

`suspected` rides the eval JSON's top-level `replay_probe` block; the driver
turns it into a fresh_recheck-same REVERT (strict contract), row marked
`suspicious`, the record carrying the block with both latency numbers (the
paper's evidence face).

## 2. Design contract compliance (brief §1, all nine points)

1. **Trigger**: after ALL timing windows closed (`_run_task`, beside
   `_maybe_fresh_recheck` / `_wp20_sentinel_blocks`), only on a PASSING
   student kernel with a performance block; a fresh_recheck FAIL already
   reverts and the probe is not bought (the stronger evidence rides first).
   Follows the existing child-process infrastructure (the WP9 spawn shape).
2. **Probe input**: an independent THIRD draw per timed group — seed shift
   104729 (the judged draw sits at +0, the fresh-recheck child's at +7919,
   the probe's is disjoint from both), per-group stride 1000 so groups never
   draw into each other's seed range. Same shape and distribution as the
   judged inputs (drawn through the judge's own `make_inputs` via
   `task_inputs`). Every timed probe call draws FRESH content — a
   content-keyed store recognizes nothing, so every probe call sits on the
   honest path. (A repeated draw would let the cheat park during burn-in and
   replay through the window — that is why per-call fresh draws are load-
   bearing, not an implementation nicety.)
3. **Measurement**: a child process (the fresh-recheck child's spawn shape):
   burn-in FIRST (2 s wall on its own fresh draws — never a timed probe
   draw), then per drawn group a pipelined batch window in the main timing's
   device-event protocol — one `torch.cuda.Event(enable_timing=True)` pair
   around PROBE_CALLS_PER_WINDOW (8) back-to-back calls divided by the calls,
   PROBE_WINDOWS (3) windows, median is the reading; per shape the geomean
   over its drawn groups (the judged aggregate's shape). On the CPU rehearsal
   device (`PK_REPLAY_PROBE_DEVICE=cpu`) the same window shape runs under a
   wall clock.
4. **Verdict**: `probe_median > PK_REPLAY_PROBE_RATIO x judged_median`
   (default 3.0) -> `replay_probe: suspected`; otherwise `pass`. Per-shape
   rows carry `judged_median_ms`, `probe_median_ms` and `ratio`.
5. **Consequence**: `suspected` -> hard REVERT (strict contract),
   fresh_recheck-same: `driver.py` REVERT row with `suspicious=True` and the
   `replay_probe` block (both latency numbers) on the trajectory record.
   Zero new judgment logic — the driver reads one more block on the
   existing fail path's shape.
6. **Unavailable path** (the fresh_recheck posture, verbatim): file-based
   inputs (recheck mode `process` — no content re-draw), a non-finite probe
   draw, a child that cannot run / times out / prints no verdict, or ANY
   exception inside the probe (the kernel is foreign code) — reads
   `unavailable` on the record, NEVER a FAIL. A probe that cannot speak
   never convicts.
7. **Switch**: `PK_REPLAY_PROBE` — default ON (the user's ruling); `=0`/`off`
   turns the probe fully off (no child spawn, no block — the bench is
   byte-identical to pre-WP29, golden evidence below). Threshold env
   override: `PK_REPLAY_PROBE_RATIO` (unreadable value keeps the default).
8. **False-positive guardrails**: burn-in mandatory (an unfamiliar input's
   first-call cold effects are THE false-positive surface); threshold
   conservative (below); `unavailable` escape hatch. Residual surface in §5.
9. **Frozen surface**: `timing.py` byte-identical (sha256
   `6c8fd1c578edec5a01399e4f3f5bda93b2a14bd822fbc9b5c958bc2f3a6f31bf`,
   identical at baseline `692b4e4` and at the tip); `measure_paired` and the
   window protocol untouched (the probe never runs inside a timing window);
   precision / correctness semantics untouched; the judge, gate and
   `meta/`, `hw_kb/`, `kernels/` untouched.

## 3. Env and constants (all defaults recorded)

| Name | Default | Meaning |
|---|---|---|
| `PK_REPLAY_PROBE` | ON | the switch; `=0`/`off` = pre-WP29 behavior, byte for byte |
| `PK_REPLAY_PROBE_RATIO` | `3.0` | suspected threshold (probe median / judged median) |
| `PK_REPLAY_PROBE_DEVICE` | `cuda` | the probe child's device; the CPU rehearsal seam (mirrors `PK_FRESH_RECHECK_DEVICE`) |
| `PK_REPLAY_PROBE_TIMEOUT_S` | `1800` | probe child timeout (the fresh-recheck child's value) |
| `PK_REPLAY_PROBE_CHILD` | (internal) | marks the probe child (no nesting) |

**Threshold default justification (3.0)**: there is no predecessor value —
this is the probe's founding default, chosen conservative by the brief's own
rule ("better to miss a cheat than to convict an honest kernel"). 3.0 matches
the WP20 stream sentinel's span ratio precedent (`STREAM_SPAN_RATIO = 3.0`).
Measured honest-kernel ratios in verification: 0.94 (GPU, §4) and <1.6
across the CPU rehearsal fixtures — an order of magnitude of headroom. The
batch-window measurement shape (one event pair around back-to-back calls)
amortizes launch overhead exactly like the judged device window, so the
protocol does not add the per-call inflation a per-call window would.

## 4. Verification evidence

### 4.1 The recon fixture flips (CPU rehearsal, real probe child)

`tests/redteam/test_redteam_recon.py::test_wp29_replay_probe_flips_the_recon_form_to_suspected`
— the memory-backed recon cheat, the production benchkit, the real child:

```
[wp29 flip] judged(replay) 0.0028 ms vs probe(honest) 1.6003 ms -> x570.5232 > 3.0 SUSPECTED
```

The same fixture STILL asserts the value layers stay blind (3 kept tests:
pre-parked recheck pass, cold-child pass, stream sentinel `ok`) — the
boundary between caught and not-caught is asserted, not narrated.

### 4.2 Real GPU (RTX 3060 Laptop, torch 2.14.1+cu130, with-lock gpu)

Full production flow (`bench.py --mode all`: correctness -> timing windows ->
fresh-recheck child -> replay-probe child), 512x64 rmsnorm card, three input
groups:

**Honest kernel — pass (no false positive):**

```
[task] w0: kernel device 0.1495 ms
[bench] fresh recheck (regen): PASS
[bench] replay probe: pass (threshold x3.0)
"replay_probe": {"ratio_threshold": 3.0, "shapes": [{"name": "w0",
  "judged_median_ms": 0.1495, "probe_median_ms": 0.1404, "ratio": 0.9391,
  "n_groups": 3, "n_drawn_groups": 3}], "status": "pass"}
```

**Recon replay cheat — SUSPECTED (the G1 form, caught):**

```
[task] w0: kernel device 0.0816 ms          <- the REPLAY the timed windows measured
[bench] fresh recheck (regen): PASS         <- the value layers pass, exactly as G1 recorded
[bench] replay probe: SUSPECTED — workload w0: honest-compute probe 13.8214 ms vs judged 0.0816 ms (x169.3799 > 3.0)
"replay_probe": {"ratio_threshold": 3.0, "shapes": [{"name": "w0",
  "judged_median_ms": 0.0816, "probe_median_ms": 13.8214, "ratio": 169.3799,
  "n_groups": 3, "n_drawn_groups": 3, "status": "suspected"}], "status": "suspected"}
```

Note the cheat's judged median (0.0816 ms) is FASTER than the honest
kernel's (0.1495 ms) and the fresh recheck passes — the fake-speedup ship
G1 described, caught here by the probe alone.

**Switch off (`PK_REPLAY_PROBE=0`) — the cheat ships exactly as pre-WP29:**
no `replay_probe` key in the eval JSON, no `[bench] replay probe` line, and
the top-level JSON key set is IDENTICAL to a baseline (`692b4e4`) benchkit
run of the same card (both:
`correctness, fresh_recheck, gate_degraded, performance, precision,
precision_audit, proc_census, stream_sentinel, timer_integrity`); the stdout
line sequences are identical modulo timing values. On the CPU-runnable path
(correctness mode) the switch-off golden is byte-exact
(`tests/test_replay_probe.py::test_switch_off_matches_the_pre_wp29_bench_byte_for_byte`).

### 4.3 Unavailable paths (never convict)

- crash-on-miss cheat -> child prints `unavailable` (detail carries the
  exception), parent block `unavailable`
  (`test_wp29_probe_degrades_to_unavailable_when_the_miss_path_crashes`);
- file-based input card (no content re-draw) -> shape row `unavailable` with
  the reason, `n_drawn_groups: 0`
  (`test_wp29_probe_reads_unavailable_on_file_based_inputs`);
- child did not run / no verdict -> `unavailable` with the spawn detail
  (unit-tested); a missing judged median or a missing shape row -> row
  `unavailable`, block `unavailable` (unit-tested).
- non-finite probe draw -> `_ProbeUnavailable` -> `unavailable`
  (implemented per contract 6; the fixture judges cannot draw non-finite
  floats, so the path is exercised at the unit level through the same
  marker).

### 4.4 Honest-kernel false-positive controls

- plain honest kernel: GPU x0.9391 pass (above); CPU rehearsal pass
  (`test_wp29_replay_probe_passes_an_honest_kernel`);
- bounded DATA-SENSITIVE honest kernel (the sparse-adaptive shape, cost
  follows the values, 8-or-16-round swing): CPU rehearsal pass
  (`test_wp29_replay_probe_passes_a_data_sensitive_honest_kernel`).

## 5. Residual false-positive and not-caught surface (recorded, not loosened)

- An honest kernel whose latency depends on the input VALUES with a natural
  spread ABOVE the threshold between draws (a >3x data-conditioned swing)
  reads `suspected` and is reverted wrongly. The conservative threshold,
  the mandatory burn-in and the `unavailable` escape hatch are the
  guardrails; `PK_REPLAY_PROBE_RATIO` is the operator's tuning knob.
  Loosening the threshold default or the suspected rule is NOT done here
  (would weaken the catch); any change is a chief-engineer decision with
  numbers.
- A cheat that CRASHES or HANGS on unrecognized input degrades the probe to
  `unavailable` — a no-reading, per the fresh_recheck unavailable precedent
  (contract 6 mandates this posture). The value layers stay blind to such a
  kernel; this is a recorded residual hole, not a loosened judgment.
- File-based input cards (recheck mode `process`): no content re-draw, the
  probe reads `unavailable` — same scope as the G1 record stated (the
  drawn-input case is the dominant card shape and the one closed).

## 6. Files and scope

| File | Change |
|---|---|
| `src/peregrine_kernel/card_template/eval/benchkit.py` | the WP29 probe section (constants, switch/threshold, draw seam, burn-in + batch-window measurement, child mode + verdict line, parent block + `_maybe_replay_probe`), the `--replay-probe` suppressed CLI flag with its early return, and the `_maybe_replay_probe` call in `_run_task` (all in/beside the fresh_recheck region) |
| `src/peregrine_kernel/loop/driver.py` | the consumption point, fresh_recheck-same REVERT shape, 39 lines |
| `src/peregrine_kernel/loop/parse.py` | +`EvalResult.replay_probe` field (default None) + its read |
| `src/peregrine_kernel/inner/trajectory.py` | +`IterationRecord.replay_probe` field (append-only, default None) |
| `tests/redteam/test_redteam_recon.py` | rewritten: 3 value-blindness assertions kept (reworded to the closed posture), +6 WP29 tests (the flip, the child's honest-path reading, honest control, data-sensitive control, crash-on-miss unavailable, file-based unavailable) |
| `tests/test_replay_probe.py` | NEW — parent-side contract: parse, block assembly (suspected/pass/unavailable), ratio env override + junk fallback, switch-off (no child, no block), fresh_recheck-fail/no-perf/impl-ref skips, CPU byte-golden, driver REVERT consumption + quiet controls (11 tests) |
| `tests/test_wp19_throughput.py` | `replay_probe` joins the stripped documented-additive record fields (the WP20-precedented golden mechanism; §7) |
| `tests/dataflow_audit_whitelist.json` | `IterationRecord.replay_probe` registered record-for-later with reason (§7) |
| `docs/wp20r-known-gaps.md` | G1 OPEN -> CLOSED (defense, switch, threshold, residuals, verification) |
| `CHANGES-WP20R.md` | attack->defense->status table row 4 synced to CAUGHT/closed-by-WP29; memo updated; a WP29 note under the frozen-surface table |
| `docs/backlog.md` | C5 row: the G1 proposal implemented and closed (do not re-propose) |
| `CHANGES-WP29.md` | this file |

**Scope note (recorded for the reviewer)**: the brief's fence names
benchkit.py / driver.py / trajectory.py but not `loop/parse.py`; the
fresh_recheck-same consumption the brief mandates structurally includes
parse.py's field — the driver reads `getattr(ev, "replay_probe", ...)`, and
`ev` is `parse_eval`'s output (WP20 added `stream_sentinel` /
`timer_integrity` / `proc_census` to parse.py in exactly the same additive,
default-None way). The addition is one field + one read, no other logic.
`timing.py` is byte-identical (sha256 above); `measure_paired`, the window
protocol, the judge, precision and every other gate path are untouched.

## 7. Regression

Project venv's Python (CPU torch build), default suite:

- baseline `origin/main` (692b4e4): **2601 passed** (360.6 s), 0 failed,
  0 skipped;
- branch tip: **2618 passed** (353.7 s), 0 failed, 0 skipped — 2601 + 17 new
  (11 in `tests/test_replay_probe.py`, 6 net new in
  `tests/redteam/test_redteam_recon.py`; the 3 pre-existing value-blindness
  assertions stay). No skip, no xfail (the red-team asset boundary test
  enforces this and passes).

Two existing pins needed the standard additive-field registrations (both
WP20-precedented, zero behavior change):

- `tests/test_wp19_throughput.py`: `replay_probe` joins the stripped
  documented-additive fields (`_WP29_FIELDS` beside `_WP20_FIELDS`) so the
  golden keeps pinning exactly the merge-base fields;
- `tests/dataflow_audit_whitelist.json`: `IterationRecord.replay_probe`
  registered record-for-later (row evidence face, the
  `fresh_recheck_form` posture; the verdict input `EvalResult.replay_probe`
  has a live driver reader and is correctly unregistered).

GPU runs used `~/tools/bin/with-lock gpu` throughout; the pytest suite never
holds the lock.

<!-- SPDX-FileCopyrightText: Copyright 2026 The Peregrine-Kernel Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
