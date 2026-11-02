## Verdict experiments (optional, budget {MAX_VERDICT_PROBE_ROUNDS} GPU-job rounds)

**Before** giving the final diagnosis, you may run bounded experiments on real hardware to decide between hypotheses. Once a diagnosis lands, the granularity at which it
gets tested is a whole segment of student iterations (hours); a verdict experiment costs only one GPU job (minutes).
When the evidence is not enough to decide, experiment first, then diagnose.

**Protocol**: write a self-contained python script to `notes/<name>.py` (inside the student workspace; it travels to the node with the job tarball
and runs there), then make **this one turn end with a single line on its own, without writing cause_class**:

```
PROBE_REQUEST: notes/<name>.py
```

The driver commits, submits the GPU job, and feeds back to you the log tail (your experiment's readings) and the verdict-line audit; you can then
send the next experiment or give the final diagnosis. Once the budget is used up you must diagnose from the evidence you have. After the diagnosis the driver resets the
student workspace to its pre-probe state (the scripts are archived on the `-metaprobe` branch) — probe files are **not** left to the
student, so do not let the diagnosis or the patch cite their paths.

**When a job is worth spending**:
- two or more cause hypotheses are compatible with the existing trajectory/NCU evidence, and choosing the wrong direction costs a whole segment of iterations;
- you are about to propose a **structural/foundation** patch (invalidating many completed decisions) — falsify the cheaper explanations first;
- the trajectory shows several REVERT_PATCH / "patch had no gain" in a row — first suspect that **the best watermark is poisoned by an outlier**:
  re-measure the current best commit in the same job before concluding "the direction is wrong";
- you suspect some milestone contract **never really landed** (e.g. an exclusive region / hiding relation rewritten by the compiler) — verify it down to SASS.

**Experiment design rules**:
1. **Pair within one job, read only ratios**: node bimodality makes absolute values drift ±6% across jobs, wider than most single-round gains. All
   variants (control included) must be measured alternately in the same job, and conclusions are read only from ratios.
2. **One variable per variant, pre-registered criteria**: for each hypothesis, have the script itself print one line
   `PROBE PASS/FAIL max_diff=<reading>` (write `max_diff=n/a` for a non-numerical contract) — the verdict is put in the
   log by the script, not by your after-the-fact interpretation. **One contract per line**: the driver records line by line (verdict + reading verbatim + end-of-line comment);
   a reading can be a number, `n/a` or a `k=v,...` string; a control contract that fails to run affects only its own line
   and does not wipe a decisive contract's PASS into a FAIL. In the end-of-line comment, name the contract and whether it is control or decisive.
3. **PASS and FAIL are both evidence**: a FAIL in a verdict experiment is not a failed experiment, it is a falsified hypothesis — often
   the most valuable result. Write the verdict into the final diagnosis's `evidence` field (cite the probe name + reading).
4. **Measure only, do not build**: a probe is a measuring instrument — add switches/instrumentation/re-measurements on the existing student code; do not deliver
   a new implementation, let alone include any ORACLE implementation-body code (the script goes into the probe job's snapshot of the student tree).
   **Measured numbers** of the reference may be used (legitimate anchors).
5. The script may call tools inside the job image (ncu, SASS dump, etc.) for attribution, not only for timing.

**Verdict experiments without a script (ablation)**: when the hypothesis is "some mechanism already on the tree (a barrier layer, a pipeline segment, a
stage) is currently costing how much", you need not write a measurement script yourself — write the ablation as JSON:

```
PROBE_REQUEST: notes/<name>.json
```

The file format is the same as the pricing round's `ablations` (`{"ablations": [{"name", "decisions", "hypothesis",
"edits": [{"old", "new"}]}]}`; each entry removes that mechanism or degrades it to a naive form, `old` occurs exactly once
in the kernel, at most 4 entries). The driver runs it with a frozen recipe: base and each ablation alternate in the same job, reading the NCU kernel clock, and **wrong
results are allowed** (what the ablated kernel computes does not matter, only the time). ncu runs first; a timeout marks it `hung` and it is not touched again (no
bench, no later rounds), and the ablation segment has its own total budget; **an ablation that deadlocks still costs one ncu timeout**, so if it can be written as a degraded form
(a coarser synchronization that still runs), do not write it as a deletion that hangs. A reading Δ that is positive and above the noise floor = faster without it = an upper bound on that mechanism's current
cost (hypothesis holds); within the band = not a major item; Δ negative beyond noise = slower without it = it is helping (hypothesis falsified). Only a positive,
above-noise Δ can price an R1 (`reopen_price: <Δ> via ablation:<name>`), and it is not the gain of any fix. The hypothesis "the reference has no such wait layer"
cannot be tested against the reference in a blackbox run; ablation is the only way to answer it on the student side.

**(Not this channel: sweep decisions)** a `SWEEP_REQUEST: notes/<decision_id>.json` from the
student in the INNER loop is a different mechanism — a teacher-authored parameter sweep
pre-written on that milestone decision line, hash-checked and run by the driver (`notes/sweep_run.py`,
the same interleaved single-job discipline, `SWEEP`/`SWEEP-WINNER` lines instead of `PROBE`),
spending the iteration's probe-round budget. It measures a grid the milestone already fixed; it
neither replaces a verdict experiment here nor loosens the student's sweep ban — only the winner
is landed, through the standard eval gate.

## ARCH_PROBE (the distill step's pre-freeze cousin of this channel)

The same "measure, don't argue" philosophy, applied at the one bet that cannot be re-opened cheaply: **the M0 freeze**.
At authoring time (the distill step, not a diagnosis), when a key resource-topology axis — cluster size, tile scheme,
warp-role split, smem buffer topology — still has more than one defensible reading, the teacher writes the **top-2
candidates** each as a skeleton recipe (`ARCH_PROBE_REQUEST: notes/arch_<n>.json` in the milestone; first listed =
preferred) instead of freezing one silently. The driver freezes the execution: one GPU job per candidate (budget
`m0_probe_rounds`, default 2), the frozen runner `notes/arch_probe_run.py` builds the recipe's skeleton
(`"language": "cuda"` → `void arch_probe_run()` via torch cpp_extension, the student kernel's own in-job toolchain;
`"python"` → `def arch_probe_run():`) and times repeats with CUDA events, printing
`ARCH_PROBE <stem> ms=<mean>` per candidate. **Wrong results are allowed — only the time is read**; a recipe that
fails to compile or launch reads as "no reading", not as evidence about the other candidate. The two times are
compared against the run's resolution floor (`resolvable_diff`): a clear winner is stamped into the milestone head and
**the freeze follows the winner**; an inconclusive pair freezes the teacher's preferred candidate with the readings
attached. Ownership mirrors this channel exactly: the skeletons are **teacher-written, driver-executed** probe
material — never delivered to the student (the probe workspace is reset away after the jobs), and the readings are
never a gate: they inform the freeze decision only, and everything downstream still goes through the standard
correctness + precision gates. Skips are recorded in the milestone head ("M0 probes skipped (budget)" and alike), so
an unprobed freeze is visible to the M0 audit, which names it as a risk item.
