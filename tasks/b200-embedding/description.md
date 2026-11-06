# b200-embedding

An embedding-table lookup: for a batch of token ids, copy each id's row out of a dense
vocabulary table (PyTorch's `torch.nn.functional.embedding` without padding/scale
arguments).

## Signature

- inputs, in order:
  - `table`: `[v, d]`, bfloat16 -- the dense embedding table, `v` rows of width `d`
  - `ids`: `[m]`, int64 -- the token ids to look up
- output:
  - `y`: `[m, d]`, bfloat16

## Semantics

For every lookup: `y[i, :] = table[ids[i], :]`. No arithmetic: every output row is
exactly one table row. Duplicate ids are legal and each occurrence copies the same row;
the order of `y`'s rows follows the order of `ids`. This task's contract declares
compute_dtype=fp32 (no arithmetic happens; the copy is bitwise). The tolerance policy is
exact: any mismatch in any element is a failure, independent of the gate's relative
bound. Out-of-range ids are undefined behavior by this contract -- a kernel may assume
every id it receives is in range (see Legal inputs).

## Legal inputs

Every id is an integer in `[0, v)`; ids may repeat and may arrive in any order (a
realistic batch is Zipf-distributed over the vocabulary, hot rows looked up many times).
`table` holds any finite bfloat16 values.
