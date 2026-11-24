# hw_kb/triton/sm100 — Blackwell (B200, CC 10.0), the board's target

Entry point for a Triton run on the target architecture. Read the symptom column first: it is
written the way the evidence reads (an error text, a PTX mnemonic, an NCU reading).

## Look up by symptom

| symptom | entry |
|---|---|
| writing the first version; the card refuses the entry's signature | [`jit_kernel_shape`](jit_kernel_shape.md) |
| "should I use `@triton.autotune`?" / a knob sweep inside the job | [`num_warps_num_stages_table`](num_warps_num_stages_table.md) |
| raising `num_warps` made it slower; `OutOfResources: tensor memory`; which MMA did it pick? | [`dot_tile_and_warp_count`](dot_tile_and_warp_count.md) |
| an "L2-friendly" grid reorder changed nothing | [`grid_axis_dispatch_order`](grid_axis_dispatch_order.md) |
| `Cannot have return statements inside while or for`; a persistent grid slower than `program_id` dispatch; a spin-wait that hangs the device | [`persistent_loop_forms`](persistent_loop_forms.md) |
| two tiles must be assembled in a known element order | [`tl_join_permute_concat`](tl_join_permute_concat.md) |
| a streaming kernel far below the bandwidth roofline with no stall reason to explain it | [`masked_vector_load_store`](masked_vector_load_store.md) |
| a reduction split across CTAs; a merge epilogue of many scalar loads; run-to-run differences | [`reduce_two_pass_and_atomics`](reduce_two_pass_and_atomics.md) |
| tolerance passes on one seed and fails on another; is this fp32 path really exact? | [`fp32_accum_and_precision`](fp32_accum_and_precision.md) |
| `PassManager::run failed` with `warp_specialize=True`; TMA slower than plain loads | [`blackwell_tma_and_warp_specialize`](blackwell_tma_and_warp_specialize.md) |
| "is this entry's number real?" / about to write a recipe into this tree | [`gpu_probe_ladder`](gpu_probe_ladder.md) |

## Completion status

| entry | covers | status |
|---|---|---|
| [`jit_kernel_shape`](jit_kernel_shape.md) | the one-file contract: `@triton.jit` kernels + `run`, grid, constexpr keywords | [PENDING-CERT] |
| [`num_warps_num_stages_table`](num_warps_num_stages_table.md) | shape-keyed `num_warps` / `num_stages` / `BLOCK`; why not autotune | [PENDING-CERT] |
| [`dot_tile_and_warp_count`](dot_tile_and_warp_count.md) | the MMA tile picker, accumulator placement (registers vs TMEM), reading the PTX | [PENDING-CERT] |
| [`grid_axis_dispatch_order`](grid_axis_dispatch_order.md) | `program_id(0)` varies fastest; reorders that are no-ops | [PENDING-CERT] |
| [`persistent_loop_forms`](persistent_loop_forms.md) | sentinel-flag loop, pipeliner across `while`, spin-wait, oversubscription | [PENDING-CERT] |
| [`tl_join_permute_concat`](tl_join_permute_concat.md) | deterministic hcat/vcat via `tl.join` + permute + reshape | [PENDING-CERT] |
| [`masked_vector_load_store`](masked_vector_load_store.md) | mask cost, alignment/contiguity hints, cache modifiers, coalescing floor | [PENDING-CERT] |
| [`reduce_two_pass_and_atomics`](reduce_two_pass_and_atomics.md) | 2D-tile merge vs scalar loop, atomic add's determinism and zeroing | [PENDING-CERT] |
| [`fp32_accum_and_precision`](fp32_accum_and_precision.md) | `out_dtype`, `input_precision`, rounding once, reduction order | [PENDING-CERT] |
| [`blackwell_tma_and_warp_specialize`](blackwell_tma_and_warp_specialize.md) | TMA descriptor cost model, `warp_specialize` restrictions | [PENDING-CERT] |
| [`gpu_probe_ladder`](gpu_probe_ladder.md) | how a claim here gets certified (compile-only / launch / measured) | [PENDING-CERT] |

Every row is `[PENDING-CERT]`: written from public Triton documentation and a public
MIT-licensed Triton guide, not yet earned by a probe of this repository. Certifying one means
running the ladder in `gpu_probe_ladder.md` and replacing its line with the probe's verdict.

## To be filled (demand-driven)

Not written yet; the run that needs one of these writes it and registers it here:

- `tcgen05`-specific accumulator handling through Triton's own surfaces (what a kernel can and
  cannot control of TMEM allocation).
- Cluster launch from Triton (`num_ctas`) and its effect on L2 / multicast.
- Program-dependent launch (`gdc_launch_dependents` / `gdc_wait` in `triton.language.extra.cuda`)
  and the `launch_pdl=True` launch keyword.
- `tl.dot_scaled` / block-scaled MMA surfaces for the low-precision formats.
- fp8 distance: which operand dtypes pair in one `tl.dot`, and the K-granularity each needs.
