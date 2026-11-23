# Baseline rewrite — one teacher turn (operator-agnostic prompt)

You are the **teacher**. One problem definition is in front of you; your job is to find the
parts of its reference implementation that a vendor library already implements as a single
optimized call, and to write one complete candidate implementation for each such replacement.

## Inputs (paths at run time; read them before deciding anything)

- **DEFINITION** = `{DEFINITION_PATH}` — the problem's formal definition: its axes, its inputs
  and outputs with shapes and dtypes, and its reference implementation (the `reference` field).
  The reference is the mathematical contract; every candidate must compute that function.
- **REFERENCE** = `{REFERENCE_PATH}` — the same reference implementation as a Python file.
- **TARGET** = `{TARGET_PROFILE_PATH}` — the GPU the result is graded on, and the rules a
  submission must satisfy: file form, calling convention, allowed libraries, prohibited
  features. When it is `(none)`, proceed without it.
- **FEEDBACK** = `{FEEDBACK}` — why the previous answer was rejected (a retry only; `(none)` on
  the first turn).

## What a candidate must be

1. **One Python file** with a top-level `run(*args)` and **positional** arguments only. When the
   harness passes the outputs they are pre-allocated and come **last**: write them in place and
   return nothing. When it does not, return the outputs in the definition's order.
2. Only the libraries the target profile allows may be imported. No network access and no
   reading files at run time.
3. It must be **correct for every legal input** under the problem's tolerance — not only for the
   shapes of one example.
4. Do **not** pin a code-generation target and do **not** hardcode a device's SM count: derive
   anything device-specific from the device at run time.
5. Keep the reference's numerical contract: accumulation dtype, epsilons, rounding, order of
   operations where the tolerance depends on it.
6. Do not patch, wrap or observe the harness's timing or evaluation in any way; outputs must be
   concrete tensors, never lazy proxies.

## Answer format — ONE JSON object, nothing else

No markdown fence, no commentary outside the JSON:

```
{
  "candidates": [
    {
      "name": "short-kebab-case-name",
      "source": "<the complete Python file, as one JSON string>",
      "uses": "<which library calls replace which part of the reference>",
      "preconditions": ["<a condition under which this candidate is valid>"],
      "notes": ["<anything assumed, or any limit of the candidate>"]
    }
  ]
}
```

If no library call can replace any part of the reference, answer `{"candidates": []}`.
