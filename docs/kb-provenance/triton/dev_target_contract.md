# Provenance — `dev_target_contract` (hw_kb/triton/sm86/ and hw_kb/triton/sm103/)

## Where these entries came from
Written for this repository as **policy entries**, from two sources:
- the device facts in the project's own target picture (the evaluation GPU and the development GPU:
  their compute capabilities, SM counts and the feature split — Ampere has no TMA / tcgen05 /
  clusters; the two Blackwell parts share the instruction set but not every unit's throughput);
- the CUDA C++ Programming Guide's compute-capability material for the architecture targets.

No reading is claimed by either entry; both exist to stop a development-tree number from being read
as a production one, and to say what a kernel must query instead of writing down.

## The mechanism entries in those trees
`hw_kb/triton/sm86/smoke_kernel_shape.md` is written for this repository (queried launch
parameters + arch-neutral math), citing `torch.cuda.get_device_properties` and the Triton launch
surface. The two transcribed neutral entries (`grid_axis_dispatch_order`, and
`masked_vector_load_store` in sm86, `num_warps_num_stages_table` in sm103) point at their own
provenance files, each of which names the source architecture.
