# Cost model for one card's workloads

You are estimating the **minimum work one call must do** for each workload of a kernel
problem, so that a first-order hardware bound can be computed from it. Nothing is run:
this is a counting exercise over the problem definition below.

The problem is `{PROBLEM_NAME}`. It has {N_WORKLOADS} workload(s).

## The definition

```json
{DEFINITION}
```

`inputs` and `outputs` are the tensors the call receives and must produce (shapes are in
the problem's axes, dtypes are exact); `axes` defines every axis name (a variable one
takes its size from the workload, a constant carries its own); `reference` is the
executable meaning of the problem — count what it computes, on every element of every
tensor it touches.

## The workloads

```json
{WORKLOADS}
```

Each entry is a workload name and the axis sizes that apply to it. These are the exact
names to use in the answer.

## What to count, per workload

For each workload, give three numbers:

- `bytes_read`: bytes that must be read from device memory during one call. Count every
  input tensor's elements once, at the byte width of its dtype. Do not assume any cache
  reuse, do not count a tensor twice because two stages read it, and do not count
  constant memory or tiny metadata.
- `bytes_written`: bytes that must be written to device memory during one call. Count
  every output tensor's elements once. If the definition also materializes a temporary
  tensor in device memory, count its write and its later read too — that is real traffic
  and a fused kernel may avoid it, which the bound should show.
- `flops`: floating-point operations one call must perform, counting one multiply-add as
  2. Count the arithmetic the definition actually performs per element (a sum of `k`
  products is `2k` per result element; a transcendental operation counts as 1). This is a
  **lower bound**: count what cannot be avoided, not an upper bound of a naive schedule.

And two labels:

- `peak`: which hardware pipe the arithmetic runs on. Use `bf16`, `fp16`, `fp8` or `fp4`
  when the arithmetic is a dense product that a tensor-core path can do at that
  precision; `fp32` when it must be scalar arithmetic; `sfu` when it is dominated by
  transcendental operations (exponential, reciprocal square root, sigmoid and the like);
  `none` if the call performs no arithmetic at all. This decides which hardware peak the
  FLOPs are divided by, so pick the one a good kernel would actually use.
- `basis`: an object with one short string per number, saying which term of the
  definition it counts — for example which tensor and which axis product. A reader uses
  it to check the arithmetic, so be specific.

## Rules

1. Answer with **one JSON object and nothing else**. No prose, no code fence.
2. Every workload of the list above appears **exactly once**, with its name spelled
   exactly as given.
3. Every number is a plain JSON number (no expressions, no strings, no scientific
   notation with a trailing unit). Zero is a legitimate number.
4. Do not use any latency, throughput or score from a leaderboard, a previous run or your
   memory of this problem: the estimate must follow from the definition alone.
5. Do not infer the answer from the problem's name. If the definition is ambiguous, count
   the lower bound and say what you assumed in `basis`.
6. FLOPs and bytes are per **one call** at the workload's axis sizes, not per second and
   not summed over calls.

The shape of the answer:

```json
{
  "workloads": [
    {
      "name": "wl0",
      "flops": 0,
      "bytes_read": 0,
      "bytes_written": 0,
      "peak": "bf16",
      "basis": {
        "flops": "which term of the definition it counts",
        "bytes_read": "which tensors and axis products it counts",
        "bytes_written": "which tensors and axis products it counts"
      }
    }
  ]
}
```

## Feedback on the previous answer

{FEEDBACK}

If the feedback above reports a problem, fix exactly that problem and answer again with
the whole object. If it says `(none)`, this is the first answer.

The machine the result is graded on is described in `{TARGET_PROFILE_PATH}`; it is
context for which precision is realistic, not an input to the arithmetic. Do not read a
latency out of it.
