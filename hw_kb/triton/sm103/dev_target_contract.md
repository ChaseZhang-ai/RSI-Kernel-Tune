# sm103 (B300) is the development GPU, not the compile target

> **Certification source: [PENDING-CERT] a policy entry written for this repository; no readings
> are claimed.**
> Detailed provenance archive: docs/kb-provenance/triton/dev_target_contract.md (for human review; not in any agent read set).

## Purpose
What this tree is for: developing and measuring on the large development GPU (Blackwell Ultra,
CC 10.3) while every number that counts is still the target's (B200, CC 10.0). It says what is the
same, what differs, and what a kernel must not do about it.

## When needed (trigger)
- A task whose `arch` is the development GPU (`sm103`) during development and the target
  (`sm100`) for the reading.
- A kernel whose dev-GPU reading is *better* than the target's and someone wants to know why.
- A kernel file that names an architecture (`sm_103`, `compute_103`, `TRITON_OVERRIDE_ARCH`) —
  the loop's target-compile check refuses it by name.

## Minimal runnable example
```toml
# tasks/<op>/task.toml: the development tree,
[task]
language = "triton"
arch     = "sm103"          # the development GPU; switch to "sm100" for the target's reading
```
```bash
# Back to the target, without running anything: does the file compile for CC 10.0?
python -m forgekernel.triton_check kernel.py --target 10.0
# ... and for the development GPU, when the question is "does it run here at all":
python -m forgekernel.triton_check kernel.py --target 10.3
```
- The same source must lower for both; that is what "not arch-pinned" buys.
- The kernel's launch parameters may be chosen from queried device properties (SM count), never from
  the architecture's name.

## Gotchas
- **Same family, different variant.** The two parts share the Blackwell instruction set, so a
  kernel compiles for both far more often than across families — but the parts are not identical
  (SM count, dense low-precision throughput, the special-function throughput that softmax-heavy
  kernels pay). A reading here is close, not equal.
- **A softmax- or low-precision-heavy kernel reads optimistically here**: those are exactly the
  units that differ. A dev-GPU win smaller than the run's noise floor may be no win on the target.
- **`sm_103a`'s cubin does not run on the target and the target's does not run here.** Triton
  compiles at run time for the device it is on, which is why the source must stay arch-neutral —
  a pinned gencode string is a kernel that runs on exactly one of the two.
- **Do not file this tree's numbers as certification.** An entry here says "measured on the
  development GPU"; the target's tree is where the target's claim belongs.

## Sources
- CUDA C++ Programming Guide — compute capability, and family compatibility of the architecture
  targets: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Triton documentation — `triton.compile` with an explicit `GPUTarget`:
  https://triton-lang.org/main/python-api/
- API symbols: `triton.backends.compiler.GPUTarget`, `torch.cuda.get_device_capability`.
- Detailed provenance archive: docs/kb-provenance/triton/dev_target_contract.md (for human review; not in any agent read set).
