"""Direct-launch BF16 paged causal GQA."""
import triton
import triton.language as tl


@triton.jit
def _request_for_tile(QO, tile, B: tl.constexpr, T: tl.constexpr):
    lo = 0
    hi = B
    for _ in tl.static_range(B.bit_length()):
        mid = (lo + hi + 1) // 2
        begins = tl.load(QO + mid) // T + mid
        take = (mid < B) & (begins <= tile)
        lo, hi = tl.where(take, mid, lo), tl.where(take, hi, mid - 1)
    return lo


@triton.jit
def _attention(Q, K, V, QO, KO, IDX, O, L, SCALE,
               B: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    tile = tl.program_id(0)
    group = tl.program_id(1)
    T: tl.constexpr = M // 8
    req = _request_for_tile(QO, tile, B, T)
    qs = tl.load(QO + req)
    qe = tl.load(QO + req + 1)
    ks = tl.load(KO + req)
    ke = tl.load(KO + req + 1)
    first = qs + (tile - (qs // T + req)) * T
    if first < qe:
        r = tl.arange(0, M)
        qpos = first + r // 8
        head = group * 8 + r % 8
        hidx = qpos * 32 + head
        d = tl.arange(0, 128)
        prefix = qpos + 1 + ke - ks - qe
        stop = tl.minimum(ke - ks, first + T + ke - ks - qe)
        acc = tl.full((M, 128), 0, tl.float32)
        normalizer = tl.full((M,), 0, tl.float32)
        maximum = tl.full((M,), -float("inf"), tl.float32)
        if stop > 0:
            q = tl.load(Q + hidx[:, None] * 128 + d[None, :],
                        (qpos < qe)[:, None] & (prefix > 0)[:, None], 0)
            n = tl.arange(0, N)
            for start in range(0, stop, N):
                key = start + n
                page = tl.load(IDX + ks + key, key < stop, 0)
                off = page[:, None] * 512 + group * 128 + d[None, :]
                k = tl.load(K + off, (key < stop)[:, None], 0)
                v = tl.load(V + off, (key < stop)[:, None], 0)
                scores = tl.dot(q, tl.trans(k)) * (SCALE * 1.4426950408889634)
                scores = tl.where(key[None, :] < prefix[:, None], scores, -float("inf"))
                next_max = tl.maximum(maximum, tl.max(scores, 1))
                safe_max = tl.where(next_max == -float("inf"), 0., next_max)
                correction = tl.exp2(maximum - safe_max)
                weights = tl.exp2(scores - safe_max[:, None])
                normalizer = normalizer * correction + tl.sum(weights, 1)
                acc = tl.dot(weights.to(tl.bfloat16), v, acc * correction[:, None])
                maximum = next_max
        out = acc / tl.where(normalizer > 0, normalizer, 1.)[:, None]
        lse = maximum + tl.log2(tl.where(normalizer > 0, normalizer, 1.))
        tl.store(O + hidx[:, None] * 128 + d[None, :], out, (qpos < qe)[:, None])
        tl.store(L + hidx, lse, qpos < qe)


def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale, output, lse):
    batch = qo_indptr.numel() - 1
    total = q.shape[0]
    if total == 0:
        return
    m, n = (64, 64) if kv_indices.numel() >= 256 else (32, 32)
    tiles = triton.cdiv(total, m // 8) + batch
    _attention[(tiles, 4)](
        q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, output, lse,
        sm_scale, batch, m, n, num_warps=4,
    )
