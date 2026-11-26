# Provenance — `smoke_kernel_shape` (hw_kb/triton/sm86/)

## Where it came from
Written for this repository: the shape of a kernel file whose first run can happen on a small
Ampere card and whose tuned run happens on the target. It combines
- the Triton launch surface (`triton.cdiv`, `num_warps` / `num_stages` / `num_ctas`, `tl.program_id`),
- PyTorch's device-property API (`torch.cuda.get_device_properties`,
  `torch.cuda.get_device_capability`), and
- this project's rule that a kernel must query the device instead of writing an SM count or an
  architecture into the source.

No reading is claimed.

## Status
`[PENDING-CERT]`. To certify: one smoke run on the Ampere card that records "launched and passed"
plus the same file's compile-only check for the target (`python -m rsi_kernel_tune.triton_check
--target 10.0`).
