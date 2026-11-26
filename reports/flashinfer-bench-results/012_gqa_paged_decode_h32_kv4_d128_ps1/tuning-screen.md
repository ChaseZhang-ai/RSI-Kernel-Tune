# Tuning screen

These are exploratory paired measurements under the same evaluator CUPTI protocol. The selectors use runtime batch and total token count. Full 48-workload verification is reported separately.

| Screen | Passes with timing | Build failures | Numerical failures | Timing failures |
|---|---:|---:|---:|---:|
| Triton tile sizes and CUDA cluster | 268 | 17 | 0 | 0 |
| Gluon layouts and absolute-index reduction | 236 | 0 | 0 | 0 |
| CUDA TMA contiguous and gather4 | 100 | 0 | 0 | 0 |
| CUDA split-KV final-CTA merge | 80 | 0 | 0 | 0 |

The first sweep compared split chunks of 32, 64, 128, 256, and 512 tokens, two to eight warps, and both MMA orientations. Oversized one-shot tiles or split chunks exceeded register or shared-memory limits in 17 configurations. The TMA screen compared separately encoded contiguous and gather4 paths with current input pointers. The final-CTA route included the counter clear and merge in the measured call.

The B=1 paths compared single-CTA online softmax, a direct warp reduction, and a transposed eight-head tensor-core route. For B=16/64, compact split slots derive from the current CSR pointers. Each tile loads one gathered K/V pair per KV head and shares it across eight query heads. The merge combines FP32 maxima, normalizers and weighted sums.

The manual asynchronous gather and thread-block cluster experiment was inspired by the [CompoundingKernel FlashInfer 012 archive](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/092577af37b8431d8dc16f6e90ec3d2ba1451aaa/kernels/FlashInfer/012_gqa_paged_decode_h32_kv4_d128_ps1), attributed there to Claude via Amazon Bedrock. The paged online softmax baseline follows the [RSI H100 seed](https://github.com/ChaseZhang-ai/RSI-Kernel-Tune/blob/main/kernels/attention/gqa_paged_decode/h100/kernel.py).
