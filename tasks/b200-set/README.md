# The B200 audit task set (`tasks/b200-*/`)

Ten operator task cards for the audit paper's B200 corpus (sm100, ~180 GB HBM). Every card
follows the formal anatomy (`tasks/_template/`, filled in by `tasks/rmsnorm/`):
`task.toml` + `description.md` (Signature / Semantics / Legal inputs) + `shapes.json` +
`reference/run.py` (plain PyTorch, source mode). Each card was validated offline with

```bash
PYTHONPATH=src ~/venvs/forgekernel/bin/python -m rsi_kernel_tune new tasks/<op> --dry-run
```

(exit 0, no GPU touched; all ten exit 0). The references were additionally smoke-run on
CPU against independent
formulations (all match; output dtypes as declared). **No performance conclusions are
drawn from this corpus until the B200 arrives and the runs execute** — the cards carry
targets (the global 1.2x ratio, device timing), not numbers.

## The set

| task | family | shapes (axes per shape identical within a task) | dtype declaration | tolerance policy |
|---|---|---|---|---|
| b200-rmsnorm | normalization, memory-bound | `{m,n}`: 128x4096, 8192x4096, 32768x8192 (bf16 I/O) | `fp32` (stats in fp32) | gate's relative bound vs the reference at the declaration; another fp32 reduction order / FMA legitimate |
| b200-layernorm | normalization, memory-bound | `{m,n}`: 64x5120, 8192x5120, 16384x8192 (bf16 I/O) | `fp32` (stats in fp32, biased variance) | same as rmsnorm |
| b200-softmax | normalization, memory-bound, numerics-hazard | `{m,n}`: 2048x1024, 16384x4096, 65536x8192 (bf16 I/O) | `fp32` (softmax arithmetic in fp32) | relative bound as above; the max shift is part of the contract, not a style |
| b200-gelu | elementwise activation, memory-bound | `{n}`: 2^20, 2^24, 2^27 (bf16 I/O) | `fp32` (erf evaluation in fp32) | relative bound as above; the exact erf form is the contract (tanh form = different function) |
| b200-add-gelu | fused elementwise, memory-bound | `{n}`: 2^20, 2^24, 2^26 (fp32 I/O) | `fp32` (end to end) | relative bound as above; IEEE fp32 add inside |
| b200-transpose | data movement, bandwidth-bound (no arithmetic) | `{m,n}`: 1024x4096, 8192x8192, 16384x32768 (fp32) | `fp32` (storage only) | **exact**: a bit permutation of the input; any mismatch fails |
| b200-logsumexp | reduction, precision-sensitive | `{m,n}`: 4096x1024, 8192x8192, 8192x65536 (bf16 in, fp32 out) | `fp32` (reduction arithmetic; output never rounded through bf16) | relative bound as above; max-shifted form is the contract |
| b200-embedding | gather, memory-bound (random access) | `{v,d,m}`: 32768x4096x128, 128256x4096x4096, 128256x8192x16384 (bf16 table, int64 ids) | `fp32` (no arithmetic) | **exact**: row copies; any mismatch fails |
| b200-rope | positional encoding, memory-bound + transcendentals | `{m,d}`: 32768x64, 1048576x128, 4194304x128 (bf16 I/O, int64 pos) | `fp32` (cos/sin in fp32) | relative bound as above; rotate-half (NeoX) convention, two outputs (`q_out`, `k_out`) in order |
| b200-gemm-fp16 | matmul, compute-bound | `{m,n,k}`: 2048^3, 4096^3, 8192^3 (fp16 I/O) | `fp16` (the set's only non-fp32 declaration: tensor-core fp16, fp32 accumulation) | relative bound vs the reference at fp16; another fp32 accumulation order (split-k, tile order) legitimate |

## Why each op is in the audit set

The audit needs coverage across regimes — the ten deliberately span memory-bound,
compute-bound, and precision-sensitive behavior, plus the two exactness corners:

- **Memory-bound streaming** (the majority of transformer overhead kernels):
  `b200-rmsnorm`, `b200-layernorm`, `b200-softmax`, `b200-gelu`, `b200-add-gelu`,
  `b200-transpose`, `b200-logsumexp`, `b200-rope`, `b200-embedding` all read/write
  far more bytes than they flop. Within the family they vary what the roofline test
  actually probes: pure bytes moved and fusion opportunity (add-gelu, gelu), access
  pattern (transpose: strided writes; embedding: random gathers into a multi-GB table),
  reduction depth (rmsnorm/layernorm/logsumexp), transcendentals (softmax, gelu, rope),
  and fan-out (rope: two tensors, one op).
- **Compute-bound**: `b200-gemm-fp16` is the tensor-core point (2048^3 .. 8192^3 —
  small for a B200, deliberately well inside its HBM), and the only card whose declared
  arithmetic is not fp32.
- **Precision-sensitive**: the reduction cards (rmsnorm, layernorm, softmax, logsumexp)
  declare fp32 arithmetic over bf16 storage — the WP22 declaration makes any quiet
  downgrade a REVERT, which is exactly the axis the audit examines. `b200-gemm-fp16`
  exercises the declaration machinery itself at fp16. `b200-transpose` and
  `b200-embedding` pin the other end of the spectrum: exactness, where no tolerance is
  owed at all.

Shape ranges per card exercise small / medium / large (decode-sized rows through
long-context tensors; the largest moves a few GB per tensor — far inside B200 HBM).

## Conventions every card of the set shares

- `[task] language = "cutedsl"`, `arch = "sm100"`; `[reference] source = "reference"`
  (source mode, like the formal rmsnorm card).
- `[precision] compute_dtype` declared **explicitly on every card** (fp32 where the
  semantics allow, fp16 on the GEMM) — the publisher's decision, never the optimizer's
  (docs/compute-dtype.md).
- No `[protocol]` table: cards run at the GPU's own clocks and the eval JSON records
  which — a pinned clock (`sm_clock_mhz`) is permission-gated GPU state that makes a
  card refuse to time when the lock does not hold, and the rented B200's lock behavior
  is unverified. If the audit later wants pinned clocks, that is a one-table edit per
  card (`[protocol] sm_clock_mhz = ...`), decided on the machine, not before it.
- No `[target]` of their own: the global default (ratio 1.2, device timing) stands.
- No `inputs/` groups: every shape starts with zero input groups and the teacher adds
  the three it needs when writing the judge (the formal card's own pattern; the dry-run
  notes say so per shape).
- Budgets: 2 h / 1 teacher round on the memory-bound cards; 4 h / 2 rounds on the GEMM.

## Red line

No performance conclusions are drawn from this set until the B200 run happens. The
cards, their shapes and their declarations are auditable now; every speedup, roofline or
regime claim waits for real device measurements on the arriving machine.
