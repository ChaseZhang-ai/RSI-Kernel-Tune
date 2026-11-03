# arch-probe — pre-freeze architecture skeletons, timed by the driver (teacher-side)

**Owner: the teacher (meta agent) at the distill step. A student never uses this skill** — the
student line implements only the milestone's decisions; if you are the student, stop reading here.

## What it is

The M0 freeze is the one bet the loop cannot cheaply re-open, so it is measured BEFORE it is
placed. When a key resource-topology axis (cluster size, tile scheme, warp-role split, smem
buffer topology) still has more than one defensible reading at authoring time, the teacher
writes the top-2 candidates each as a **skeleton recipe** and lists them in the milestone:

    ARCH_CANDIDATES: dual
    ARCH_PROBE_REQUEST: notes/arch_0.json   # first listed = the teacher's preferred reading
    ARCH_PROBE_REQUEST: notes/arch_1.json

No divergence on any key axis → `ARCH_CANDIDATES: single`, no recipes, no probe jobs.

## The recipe

```json
{"name": "<label>", "rationale": "<one sentence: why this reading>",
 "language": "cuda" | "python",
 "fidelity": "mock" | "semi-real",          (optional, default "mock" — see below)
 "problem": {"axes": {...}, "inputs": [...], "outputs": [...]},   (optional — see below)
 "rounds": <timed repeats, default 20>,
 "source": "<the skeleton>"}
```

- `cuda`: CUDA C++ defining `void arch_probe_run()` that sets up this candidate's launch
  geometry (grid / block / cluster via `cudaLaunchKernelEx` / smem carveout / warp roles as
  no-op roles) and runs the main loop's shape once.
- `python`: a snippet defining `def arch_probe_run():` to the same effect.

The skeleton is a **topology mock, not a kernel implementation**: minimal, dummy data,
correctness irrelevant — **wrong results are allowed, only the time is read**. The recipes never
enter the milestone beyond the request lines, and the driver resets the probe workspace away
after the jobs, so the skeletons never reach the student.

## Fidelity tiers (WP17): `mock` vs `semi-real`

A recipe declares its own fidelity. The driver cannot verify data sizes (correctness is never
read here), so the declaration is a discipline you own, and the stamp names the tiers either
way:

- **`mock`** (the default) — the historical skeleton: launch geometry and loop shape, dummy
  data. Fine when the candidates differ on an axis the mock already exercises (launch overhead,
  occupancy at the topology's block shape, smem carveout).
- **`semi-real`** — the skeleton must let the **dominant data axis flow through real-size
  data**: the memory axis reads/writes the real byte volume (allocate and stream the actual
  tensor shapes, even if the values are garbage), the compute axis issues the real MMA volume;
  **only the compute LOGIC may be a no-op**. Write this tier when the candidates' readings
  could plausibly be decided by data movement or issue rate — anything where a dummy-data
  skeleton can mis-order the pair.

**Choosing rule**: if the two candidates differ mainly in *how data moves* (tile scheme,
vectorization width, staging topology, multicast) or *how much compute issues per byte* —
write both recipes `semi-real`. If they differ in launch/occupancy/scheduling topology that a
dummy body exercises the same — `mock` is honest. When unsure, `semi-real`: it costs the same
rounds.

**Pre-rank (advisory)**: a recipe may carry a `problem` view (analytic_floor's schema: `axes` /
`inputs` / `outputs`; arch and family default from the task's own problem). The driver computes
each candidate's zero-GPU-cost bytes/FLOPs floor and — when the bracket cannot probe every
listed candidate — probes the top of that ranking instead of the file order. It is a SORT
SUGGESTION only: never a price, never a verdict, and the readings (not the floor) decide.

**Near-tie rule (driver-enforced)**: when the two decisive readings sit within the run's
resolution floor AND the probe budget still has a job, the driver re-measures the pair once
(the re-draw replaces the first reading) — a within-floor winner is one noise draw until a
second draw says otherwise. The readings only ever freeze a candidate; **they are never a
gate** (`meta_milestone.md`'s time-only rule), and a re-probe costs budget only when the
bracket left it unspent.

## Execution and decision

The driver seeds the frozen runner (`arch_probe_run.py`, this directory) into the tree's
`notes/`, runs one GPU job per candidate (`cmd:python3 notes/arch_probe_run.py
notes/arch_<n>.json`; budget `m0_probe_rounds`, default 2), and compares the two
`ARCH_PROBE <stem> ms=<mean>` readings against the run's resolution floor (`resolvable_diff`):
a clear winner is stamped into the milestone head and **the freeze follows the winner**; an
inconclusive pair freezes the teacher's preferred candidate with the readings attached. Skips
(budget, a missing recipe, a failed job) are recorded in the milestone head too, so an unprobed
freeze is visible to the M0 audit. The readings inform the freeze decision only — **they are
never a gate**; everything downstream still goes through the standard correctness + precision
gates.
