# sweep-decision

Batch measurement as a first-class decision step: a **teacher-authored**
parameter sweep (tile × stage × cluster style grids and any other knob set the
milestone pins down) timed in ONE GPU job — base and every variant interleaved,
same-job paired ratios, the pricing tool's noise floor.

The sweep ban in the student prompt ("tunable" is not an invitation to sweep
parameters ad hoc) stays: the recipe is written into the milestone decision
line by the teacher, the student only triggers and lands it.

## Protocol

1. The teacher pre-writes the recipe into the milestone decision line as an
   optional `sweep:` field (≤ 8 variants, exact-text edits, criterion fixed to
   "min same-job ratio above floor").
2. On reaching that decision the student copies the recipe verbatim to
   `notes/<decision_id>.json` and ends the turn with
   `SWEEP_REQUEST: notes/<decision_id>.json`.
3. The driver validates the file against the milestone recipe by canonical-JSON
   hash (a mismatch is refused without a job — no smuggling), seeds this
   script, and submits `cmd:python3 notes/sweep_run.py notes/<decision_id>.json`.
4. The feedback carries the `SWEEP` table + `SWEEP-WINNER`; the student lands
   ONLY the winner as an ordinary decision record — the standard eval gate
   judges it, like any other delivery.

`sweep_run.py` reuses `lever_pricing_run`'s interleaved-timing kernel (bench
calls, node hygiene, noise floor, ratio reading) over an explicit variant
list; `lever_pricing_run.py` is unchanged. Timeouts follow the pricing tool's
precedents: one bench ≤ `SWEEP_BENCH_TIMEOUT` (900 s), the whole pass ≤
`SWEEP_BUDGET_S` (3600 s), both clamped to the job clock's remaining time.

Each `SWEEP_REQUEST` consumes one probe round of the iteration's
`max_probe_rounds` budget (the verdict-probe budget class; no extra dimension).

## Flags are sweepable parameters (WP10)

A compile flag on the `// fk-cuda-cflags:` line is kernel source text, so a flag
axis is a legitimate sweep recipe — the lever discipline is unchanged (exact-text
edits, the whitelist holds; the card refuses anything outside it before a build).
The canonical recipe — a kernel pinned at a register tier with spill stack
(resusage `regs_per_thread` at ≥ 0.9× of an architecture tier, `STACK > 0`) —
sweeps the tier, not the tile:

```json
{"kernel_file": "kernel.cu", "rounds": 3,
 "criterion": "min same-job ratio above floor",
 "variants": [
   {"name": "regs128", "edits": [{"old": "// fk-cuda-cflags: -lineinfo",
                                  "new": "// fk-cuda-cflags: -lineinfo --maxrregcount=128"}]},
   {"name": "regs168", "edits": [{"old": "// fk-cuda-cflags: -lineinfo",
                                  "new": "// fk-cuda-cflags: -lineinfo --maxrregcount=168"}]},
   {"name": "regs224", "edits": [{"old": "// fk-cuda-cflags: -lineinfo",
                                  "new": "// fk-cuda-cflags: -lineinfo --maxrregcount=224"}]},
   {"name": "regs255", "edits": [{"old": "// fk-cuda-cflags: -lineinfo",
                                  "new": "// fk-cuda-cflags: -lineinfo --maxrregcount=255"}]}]}
```

Reading: the tier that clears the spill (`STACK` back to 0 in the RES-USAGE
reading) and beats base above the floor is the winner — occupancy paid for the
pressure relief. A tier that clears the spill but reads flat means the spill was
not the binding constraint (occupancy-bound: check `waves_per_sm` for the tier
jump before re-sweeping). Whitelist members worth the same treatment: `--ftz`,
`--fmad`, `--prec-div` / `--prec-sqrt`, `--extra-device-vectorization`;
`--use_fast_math` may be swept too, but its numerics are arbitrated by the
precision gate.

## WP17: in-process bench switching (default on, `FK_SCREENING_INPROC=0` to kill)

The sweep's timed readings run through one long-lived bench worker process (the pricing
tool's `lever_pricing_run`): torch and the card's eval modules import once, and every
(variant, round) call re-executes `eval/bench.py` in that process — fresh inputs, correctness
and the paired windows per call, exactly the subprocess's protocol, minus the per-call cold
start (~30-50% of the job's clock on a full grid). A hung or crashed call is killed as a
process group and takes the same NODE-DIRTY path as before; the driver's re-buy-once policy is
untouched. Set `FK_SCREENING_INPROC=0` in the job env to put every bench call back on the
per-call subprocess (the historical behavior, byte for byte).
