---
name: roofline
description: Use when deciding whether a kernel still has room to improve, when a reading is close to (or suspiciously below) what the hardware allows, or when reading Nsight Compute's Speed-of-Light and stall numbers and turning them into one bottleneck class and a short list of next actions. Three pure-Python tools — `classify` (SOL percentages + stall evidence → memory / compute / latency / balanced), `suggest` (an NCU signal → cause → candidate action table, hardware-level only) and `distance` (a cost model + measured peaks → the floor a call cannot beat and how far a reading is from it). First-order advice: it never measures and it never replaces a profile or `forge calibrate`.
---

# Skill: roofline — where a kernel stands against the hardware

**When to use:** before choosing what to optimize next, when a kernel is faster or slower
than expected and a profile's Speed-of-Light numbers are in hand, when a reading is within
a few percent of a limit and you need to know whether that limit is real, or when someone
asks "how much is left here" / "why is this slow" after an `ncu` run.

**What it is:** three readings of measurements you already took.

| subcommand | input | output |
|---|---|---|
| `classify` | NCU metrics (a `name=number` list, an `ncu --import ... --csv` export, or a JSON object) | one bottleneck class + every number the verdict used |
| `suggest` | a signal id (or nothing, for the whole table) | `signal → likely cause → candidate action` rows, hardware-level only |
| `distance` | a calibration file, FLOPs and bytes per call | the floor `t_floor = max(t_mem, t_tensor, t_launch)`, its class, and a reading's distance from it |

**What it is not:** a measurement, or a substitute for one. `classify` maps numbers to a
class; `distance` divides counted work by measured peaks. If the numbers are wrong the
verdict is wrong, and no tool here can tell. Nothing in this skill may be copied into
`hw_kb/` — entries there are certified by a probe on the architecture they describe, and a
heuristic with thresholds is not a certification.

---

## Running it

A GPU job has the file at `notes/roofline_tools.py` (the harness seeds every skill's
scripts there); a ForgeKernel checkout has it at `skills/roofline/roofline_tools.py`. It is
pure stdlib Python — no GPU, no torch, no ForgeKernel import.

```bash
# what is this kernel bound by? (new metric names; older spellings are accepted too)
python3 roofline_tools.py classify --metrics "\
sm__throughput.avg.pct_of_peak_sustained_elapsed=41.2,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed=93.4,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active=0"

# from a saved report's CSV export, one kernel of several launches
python3 roofline_tools.py classify --csv report.csv --kernel my_kernel

# the whole playbook, or one row
python3 roofline_tools.py suggest
python3 roofline_tools.py suggest --signal long_scoreboard
python3 roofline_tools.py suggest --list

# how far is 0.42 ms from what the hardware allows?
python3 roofline_tools.py distance --calibration calibration/<gpu>-<date>.json \
    --flops 2.1e9 --bytes-read 8.4e7 --bytes-written 1.1e7 --peak bf16 --current-ms 0.42
```

Output lines are machine-readable (`CLASS ...`, `EVIDENCE ...`, `SUGGEST signal=...`,
`DISTANCE ...`) so a script can read them back; the human-readable part of each line says
which numbers were used. Exit status: `0` ok, `2` a usage error (an unreadable file, a
missing metric pair, an unknown signal).

---

## `classify`: four classes, and the evidence

`compute_sol_pct` (`sm__throughput.avg.pct_of_peak_sustained_elapsed`) and
`memory_sol_pct` (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` — or
the DRAM-only `dram__throughput...` when that is all the report has) are the two
Speed-of-Light headlines: how close each pipeline is to its own peak.

- **both under 60%** → neither resource is saturated. Either the kernel is
  **underutilized** (launch geometry, occupancy, dependency chains — look at those next)
  or, when DRAM throughput is under 10% *and* the stall samples are dominated by
  long-scoreboard waits (over 40% of the samples, or a per-issue ratio over 3), it is
  **latency**-bound on memory: more requests in flight is the fix, not fewer bytes;
- **one at 60% or more, the other within 10% of it** → **balanced**: the two pipelines
  bind together;
- **memory higher** → **memory**-bound: bytes are the cost;
- **compute higher** → **compute**-bound: arithmetic is the cost.

The thresholds are KernelAgent's `_classify_bottleneck` (60% = underutilized) plus the
ncu-report-skill's stall dimension (the latency distinction). The verdict prints
`compute_sol_pct`, `memory_sol_pct`, `efficiency_pct` (the larger), `gap_pct`, `dram_pct`
and whichever stall numbers it used — check them before acting; a metric that is absent
is read as 0 and listed under `WARNING missing`, never silently.

## `suggest`: signal → cause → action

Every row is hardware-level, so the table is operator-agnostic: it names a resource
(tensor pipe, scalar pipe, special-function pipe, L1/L2/DRAM, shared-memory banks,
registers), a scheduling property (grid size, waves, barriers, pipeline stages) or an
instruction pattern (coalescing, store width, FP64 promotion), never what the kernel
computes. Ported from ncu-report-skill's `reference/06-diagnosis-playbook.md` (MIT; see
`NOTICE`) and from KernelAgent's roofline thresholds (Apache-2.0).

Read the profile first, then match **one or two** rows: a kernel with several matches is
ranked by the magnitude the profile shows (NCU's own `Est. Speedup` fields are a starting
estimate, not a measurement — check each against the counters it rests on).

## `distance`: the floor and the distance

`distance` is the same arithmetic `forge roofline` uses (`t_mem`, `t_tensor`,
`t_floor = max(...)`, the bottleneck class, `current / t_floor`), so a number printed here
and a row in the roofline context file agree. The peak set comes from a
`forge calibrate` document (measured on that machine); `--hbm-gb-s` / `--tflops` can
override a peak on purpose, and an unknown peak is an error, never a default.

Use it to answer one question: **is this reading near the floor?** Under 1.1x, the model
says there is little left; between 1.1 and 2x there is room; over 2x the kernel is not
close to any limit this model knows about. Then check the model itself against a profile:
a reading *below* the floor means the cost model over-counts or the peaks are wrong, not
that the hardware broke.

---

## Boundaries

- This skill never starts a GPU job, never runs `ncu`, and never edits a kernel.
- It reads no network and writes no file.
- It is a reading **about** a reading: the loop's gate (`device_median_ms`) stays the
  only number a KEEP/REVERT decision rests on.
