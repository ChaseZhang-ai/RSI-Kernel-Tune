# hw_kb/triton/sm103 — Blackwell Ultra (B300), the development GPU

> The development GPU is not the compile target. A kernel is developed and measured here, and its
> reading counts only after the target's own run. A kernel must never be compiled *for* this part
> (the loop's target-compile check refuses an architecture pinned in the source).

## Look up by symptom

| symptom | entry |
|---|---|
| "which `arch` during development, and how do I get back to the target's reading?" | [`dev_target_contract`](dev_target_contract.md) |
| "should I tune `num_warps` / `num_stages` here?" / `@triton.autotune` in the file | [`num_warps_num_stages_table`](num_warps_num_stages_table.md) |

## Completion status

| entry | covers | status |
|---|---|---|
| [`dev_target_contract`](dev_target_contract.md) | same family, different variant; never pin the arch; how to read a dev-GPU number | [PENDING-CERT] |
| [`num_warps_num_stages_table`](num_warps_num_stages_table.md) | shape-keyed knob table; a value read here is a starting point for the target's | [PENDING-CERT] |

## To be filled (demand-driven)

- The parts of the target tree's mechanism set this GPU's runs actually need (the MMA/TMEM entry
  first, when a run here starts diagnosing accumulator placement), each copied with its
  `Certification source` line naming this tree.
- The dev-GPU-vs-target ratio the campaign records per problem, once there are readings to
  compare: softmax-heavy and dense low-precision kernels are the two families expected to read
  optimistically here.
