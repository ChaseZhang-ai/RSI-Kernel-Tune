# Persistent-loop forms the Triton compiler accepts

> **Certification source: [PENDING-CERT] written from the public Triton documentation and a
> public MIT-licensed Triton guide; not certified by a GPU probe in this repository.**
> Detailed provenance archive: docs/kb-provenance/triton/persistent_loop_forms.md (for human review; not in any agent read set).

## Purpose
A persistent kernel (one CTA per SM, a loop that pulls work items) is expressible in Triton, but
three forms of the textbook skeleton do not compile or do not pay — and each fails in a way that
looks like a different problem.

## When needed (trigger)
- `Cannot have return statements inside while or for statements in triton` — the standard
  `while True: tid = atomic_add(...); if tid >= N: return` skeleton.
- A persistent grid built as `grid = (num_SMs,)` measures *slower* than `grid = (N_TASKS,)` with
  `program_id` dispatch, at small task counts.
- A spin-wait on a global counter with one CTA per SM appears to hang the whole device.

## Minimal runnable example
```python
import triton
import triton.language as tl


@triton.jit
def _persistent(counter_ptr, x_ptr, n, BLOCK: tl.constexpr):
    # No `return` inside the loop: a sentinel flag ends it instead.
    active = True
    while active:
        tid = tl.atomic_add(counter_ptr, 1, sem="relaxed", scope="gpu")
        if tid < n:
            offs = tid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < (tid + 1) * BLOCK
            tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)
        else:
            active = False
```
Launch it **oversubscribed** when warps have to make progress while others wait:
```python
grid = (4 * sm_count,)      # sm_count = torch.cuda.get_device_properties(0).multi_processor_count
```
- `sm_count` is queried at run time, never written down: the target and the development GPU differ.

## Gotchas
- **`return` is rejected inside `while` / `for`** (the compiler needs single-exit control flow for
  the pipeliner); the sentinel-flag form above is the replacement.
- **The pipeliner does not carry across outer `while` iterations**: each task pays the full
  `num_stages` prologue/epilogue. For tasks shorter than about `2 × num_stages` tile iterations the
  per-task prologue dominates and the plain `grid=(N_TASKS,)` + `program_id` dispatch is faster.
- **One CTA per SM spin-waiting blocks that SM**: with `grid = (num_SMs,)` every spinner idles its
  whole SM on the spin branch (all its warps wait in lockstep). Oversubscribe the grid so producer
  warps can co-run on the spinner's SM, or replace the spin with a cache-modifier load
  (`tl.load(..., cache_modifier=".cv")`) where the semantics allow it.
- **`tl.atomic_add`'s `sem` / `scope` are load-bearing**: `sem="relaxed"` for a work counter is
  fine, but a hand-off of *data* needs `acquire`/`release` (or a barrier) as well, or the consumer
  can read a stale tile.
- A persistent kernel's launch shape is a *policy*, not an optimization detail: it decides the
  dispatch order, so it interacts with `grid_axis_dispatch_order.md`'s L2 reasoning.

## Sources
- Triton documentation — `tl.atomic_add` (`sem`, `scope`), `tl.range`, control-flow restrictions:
  https://triton-lang.org/main/python-api/
- API symbols: `tl.atomic_add`, `tl.program_id`, `tl.num_programs`, `num_stages`.
- Detailed provenance archive: docs/kb-provenance/triton/persistent_loop_forms.md (for human review; not in any agent read set).
