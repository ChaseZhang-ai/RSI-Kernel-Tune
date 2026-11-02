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
