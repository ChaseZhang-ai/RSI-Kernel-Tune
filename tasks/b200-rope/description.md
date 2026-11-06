# b200-rope

Rotary position embedding (RoPE, the rotate-half / GPT-NeoX formulation used by the open
transformer stacks) applied to the attention query and key tensors in one kernel.

## Signature

- inputs, in order:
  - `q`: `[m, d]`, bfloat16 -- the query rows (`m` = tokens x heads flattened, `d` = head dim)
  - `k`: `[m, d]`, bfloat16 -- the key rows, same layout
  - `pos`: `[m]`, int64 -- the position id of each row (rows of the same token's
    different heads share one position)
- outputs, in order:
  - `q_out`: `[m, d]`, bfloat16
  - `k_out`: `[m, d]`, bfloat16

## Semantics

With `theta = 10000.0` and `half = d / 2`, the inverse frequencies are
`inv_freq[j] = theta^(-2j / d)` for `j` in `[0, half)`. Each row `i` gets angles
`ang[i, j] = pos[i] * inv_freq[j]`. In the rotate-half convention, writing
`x1 = x[..., :half]`, `x2 = x[..., half:]`, every input row `x` (of `q` and of `k`
alike) maps to `y1 = x1 * cos(ang) - x2 * sin(ang)`, `y2 = x2 * cos(ang) + x1 * sin(ang)`,
and `y = concat(y1, y2)` on the same axis. The angles, cos and sin are evaluated in
float32 (at least); each output is rounded to bfloat16 once, at the end. `q` and `k` are
transformed independently by the same rule and returned as two outputs, `q_out` then
`k_out`. This task's contract declares compute_dtype=fp32: every arithmetic step below
fp32 is a contract violation (the tolerance policy is the gate's relative bound against
this reference's own error at the declaration).

## Legal inputs

`d` is even (it is split into two equal halves); realistic `d` is 64 or 128. `pos[i]` is
an integer in `[0, 131072)` (a 128k context); positions may repeat. `q`, `k` hold any
finite bfloat16 values.
