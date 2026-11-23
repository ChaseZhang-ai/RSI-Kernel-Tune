# The Triton probe ladder: how a mechanism gets certified in this tree

> **Certification source: [PENDING-CERT] methodology, written for this repository from the
> discipline the other two knowledge-base trees use; no GPU probe has exercised it yet.**
> Detailed provenance archive: docs/kb-provenance/triton/gpu_probe_ladder.md (for human review; not in any agent read set).

## Purpose
A Triton entry may only carry readings that a probe earned. This is the ladder: which reading a
probe can take with no launch, with one launch, or only with a measurement — so an entry's
`Certification source` line means something.

## When needed (trigger)
- An entry says `[PENDING-CERT]` and a run wants to earn it.
- A mechanism's claim is about *what the compiler emits* and it is unclear whether a GPU is needed.
- A recipe is about to be written into the tree (`RECIPE_CERTIFIED:`), which requires a verdict.

## Minimal runnable example
```python
# Rung 0 -- no GPU, no launch: what did the compiler emit for the target?
#   python -m forgekernel.triton_check kernel.py --target 10.0 --kernel <name> \
#       --signature '{"x_ptr": "*bf16", "n": "i32"}' --constexprs '{"BLOCK": 1024}'
# -> a TRITON-COMPILE verdict line; a compiled kernel's PTX is readable from the same call.

# Rung 1 -- one launch, one contract: does the kernel launch and compute what the contract says?
#   a standalone script in notes/<name>.py that builds known inputs, runs ONLY the mechanism,
#   compares against a torch/numpy rearrangement and prints exactly one line:
#   PROBE PASS max_diff=0.0        (or PROBE FAIL max_diff=<number>)

# Rung 2 -- measured: what does the mechanism cost in the whole kernel?
#   a job that reads the card's own clock (device_median_ms), A/B against the same tree with the
#   mechanism removed. A gain smaller than the run's noise floor is not a certification.
```
- Rung 0 certifies **structure** (which MMA form, whether an accumulator is in TMEM, whether a
  pattern compiles for the target at all). It is the only rung that needs no GPU.
- Rung 1 certifies a **contract** (a layout mapping, a mask's coverage, a concat's order).
- Rung 2 certifies a **price**, and only the card's own clock may be cited for it.

## Gotchas
- **Rung 0's PTX is not the final code.** `ptxas` schedules afterwards, so a claim about issue
  order or stall reasons needs rung 2 (or NCU) — do not certify a tuning claim from PTX alone.
- **A Triton probe must run in the same process shape as the job.** A kernel that imports cleanly
  under `python -c` but not under the card's `python3 eval/bench.py` (a different interpreter, a
  different `triton` install) has certified nothing about the job.
- **A `PROBE PASS` line is a claim about one contract.** A probe that prints several PASS lines
  certifies several; a probe whose input is a scaled copy of the real shape (same tile boundaries,
  smaller batch) may skip the boundary mask the real shape exercises.
- **The compile artefacts rung 0 reads are the compiler's cache.** Read them from a probe script;
  the kernel file itself must not name that cache (the card's fence refuses it for a sealed
  reference, and naming it is not a kernel's business in any case).
- **`[PENDING-CERT]` is not a weaker ✅.** It tells the reader the number is unverified so that a
  milestone cannot cite it as measured evidence; certifying it means replacing the line, not
  adding a second one.

## Sources
- The loop's probe protocol (the `PROBE PASS` / `PROBE FAIL` / `NO VERDICT` disciplines the driver
  audits) is the shared prompt surface every run reads.
- API symbols: `triton.compile`, `triton.backends.compiler.GPUTarget`, `triton.compiler.compiler.ASTSource`.
- Detailed provenance archive: docs/kb-provenance/triton/gpu_probe_ladder.md (for human review; not in any agent read set).
