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
