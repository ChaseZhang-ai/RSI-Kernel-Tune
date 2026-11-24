# sm86 is a development / smoke target, not a production target

> **Certification source: [PENDING-CERT] a policy entry written for this repository; no readings
> are claimed.**
> Detailed provenance archive: docs/kb-provenance/triton/dev_target_contract.md (for human review; not in any agent read set).

## Purpose
What this tree is for: letting the whole loop — the card's build, its correctness check and its
clock — run end to end on a small Ampere card, so that a broken call path is found in minutes
instead of on the board. It is **not** a place to tune for.

## When needed (trigger)
- Bringing up a new task or a new kernel file where the only question is "does the call path work".
- A local run on the RTX 3060 (6 GB, CC 8.6) whose numbers someone is tempted to compare with the
  target's.
- Choosing `arch` for a task you intend to smoke locally.

## Minimal runnable example
```toml
# tasks/<op>/task.toml — the default is the production target,
[task]
language = "triton"
arch     = "sm100"          # the target the readings are about

# ... switched to the local card for a smoke run only:
# arch = "sm86"             # a 6 GB Ampere laptop part: DEVELOPMENT / SMOKE only
```
```python
# The kernel itself must not know which of the two it runs on: query, never write down.
import torch
sm = torch.cuda.get_device_properties(0).multi_processor_count     # grid sizing
cap = torch.cuda.get_device_capability(0)                          # ("10", "0") on the target
```
- A smoke run's verdict is **PASS / FAIL on correctness and "it launched"** — never a performance
  conclusion.
- A kernel that needs a Blackwell-only instruction (`tcgen05`, TMA descriptors, clusters) cannot be
  smoked here at all; smoke its fallback path or the shape-only skeleton instead.

## Gotchas
- **The SM count differs** (an Ampere laptop part has a fraction of the target's), so any grid
  sized as a multiple of the SM count behaves differently: a persistent kernel that is fine here
  can be under-filled or over-subscribed there. The code must query the count; the *table* of
  values may not encode a count.
- **No TMA, no tcgen05, no clusters on this part.** A kernel written around them must have a path
  this card can take, or it cannot be smoked here.
- **The bandwidth and the clock are different**, so "bandwidth-bound here" says nothing about the
  target's balance point: a tile that is bandwidth-bound here can be compute-bound there, and the
  other way round.
- **6 GB is the real limit** of the local card: a shape that allocates the target's working set does
  not fit. Smoke the smallest shape of the group and say so when recording the reading.
- **A number taken here is not a reading of the target.** Do not write it into an entry's
  `Certification source` line; that line is for probes on the architecture the entry belongs to.

## Sources
- CUDA C++ Programming Guide — compute capability and the Ampere / Hopper / Blackwell feature split:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- API symbols: `torch.cuda.get_device_properties`, `torch.cuda.get_device_capability`,
  `triton.compile(..., target=triton.backends.compiler.GPUTarget("cuda", arch, 32))`.
- Detailed provenance archive: docs/kb-provenance/triton/dev_target_contract.md (for human review; not in any agent read set).
