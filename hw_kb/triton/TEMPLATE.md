# <mechanism name> (e.g. the MMA tile picker / masked vector load / two-pass reduce)

> One file per mechanism. Operator-agnostic: mechanism only. Triton mode: examples are Python
> modules of `@triton.jit` kernels launched from a `run()` (the leaderboard's one-file format).

## Purpose
What compiler / hardware problem this mechanism answers (one sentence).

## When needed (trigger)
Which student need / symptom calls for it (compiler error text, a PTX mnemonic, an NCU stall
reason, a shape regime).

## Minimal runnable example
```python
# Minimal, self-contained, runnable against a real Triton install. From a verifiable source;
# nothing invented. Shows only "how to use this mechanism"; embeds no operator's algorithm.
# State the API level it needs when it is not in every Triton release (e.g. `tl.make_tensor_descriptor`).
```

## Gotchas
- What a misuse does silently (a different MMA selected, a spurious recompile, a pipeliner that
  does not carry across a loop, an autotune that re-times inside the job's clock, an accumulator
  that moves from registers to TMEM).

## Sources

**The student reads this section; only three kinds of citation are allowed** (a tool's symbols may be
cited, a reference solution's may not):

1. Official Triton documentation / release notes (triton-lang.org, the `triton-lang/triton`
   release notes for the version in use);
2. **API symbols** of the public Triton surface (`triton.language.*`, `triton.compile`,
   `triton.backends.compiler.GPUTarget`, `triton.runtime.jit.JITFunction`, the documented knobs) —
   i.e. what the student has to call when writing the kernel anyway;
3. The path and verdict (`PROBE PASS ...`) of a GPU-certified probe in this repo.

**Must not appear**: the reference implementation's (whether open- or closed-source) file names,
function names, line numbers, paraphrased content or measured footprints, nor the words
"oracle / teacher / reference implementation" — all such narrative evidence chains go to
`docs/kb-provenance/triton/<this entry name>.md` (for human review; never in any agent read set),
and the entry keeps only one line:

```
- Detailed provenance archive: docs/kb-provenance/triton/<entry>.md (for human review; not in any agent read set).
```

Every entry opens with a **`Certification source`** line: `✅ Verified` plus the probe verdict, or
`[PENDING-CERT]` plus where the mechanism was written from (see `hw_kb/triton/README.md`).
The mechanical gate `src/forgekernel/loop/kb_lint.py` scans for banned tokens at three points — loop
start, recipe certification into the KB, and publish: on a hit at certification the entry is first
bounced back for revision and quarantined if it still fails; a hit at harvest refuses the publish.
