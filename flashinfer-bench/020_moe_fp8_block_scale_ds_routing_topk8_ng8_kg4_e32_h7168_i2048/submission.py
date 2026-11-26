"""DeepSeek grouped FP8 MoE, with destination passing and FP32 accumulation.

The indirect GEMM and block-scale promotion follow the RSI-Kernel-Tune
ako4x fused_routing_v2 reference. The three-term FP8 SwiGLU expansion and
expert tile scheduling draw on QianQu/CompoundingKernel-Explorer, FlashInfer 020.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _clear(Counts, Acc, SIZE, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(Counts + tl.arange(0, 32), 0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(Acc + offset, 0, offset < SIZE)


@triton.jit
def _take_k(scores, K: tl.constexpr):
    values = tl.topk(scores, K)
    threshold = tl.min(values, 0)
    higher = scores > threshold
    tied = scores == threshold
    slots = K - tl.sum(higher.to(tl.int32), 0)
    return higher | (tied & (tl.cumsum(tied.to(tl.int32), 0) <= slots))


@triton.jit
def _route(Logits, Bias, Counts, Tokens, Mixture, T, LocalStart, RoutingScale):
    token = tl.program_id(0)
    experts = tl.arange(0, 256)
    logits = tl.load(Logits + token * 256 + experts)
    scores = tl.div_rn(1.0, 1.0 + libdevice.exp(-logits))
    biased = scores + tl.load(Bias + experts).to(tl.float32)
    top2 = tl.topk(tl.reshape(biased, (8, 32)), 2)
    groups = _take_k(tl.sum(top2, 1), 4)
    kept = tl.reshape(tl.broadcast_to(groups[:, None], (8, 32)), (256,))
    selected = _take_k(tl.where(kept, biased, float("-inf")), 8)
    normalizer = tl.sum(tl.where(selected, scores, 0.0), 0) + 1.0e-20
    local_experts = experts - LocalStart
    valid = selected & (local_experts >= 0) & (local_experts < 32)
    slot = tl.atomic_add(Counts + local_experts, 1, valid, sem="relaxed")
    index = local_experts * T + slot
    tl.store(Tokens + index, token, valid)
    tl.store(Mixture + index, (scores / normalizer) * RoutingScale, valid)


@triton.jit
def _plan(Counts, Plan, Tiles, M: tl.constexpr, PER_EXPERT: tl.constexpr):
    expert = tl.program_id(0)
    e = tl.arange(0, 32)
    counts = tl.load(Counts + e)
    blocks = tl.cdiv(counts, M)
    tiles = tl.sum(blocks, 0)
    if expert == 0:
        tl.store(Plan + 64, tiles)
        tl.store(Plan + 65, tl.sum(counts, 0))
    row_start = tl.sum(tl.where(e < expert, counts, 0), 0)
    tile_start = tl.sum(tl.where(e < expert, blocks, 0), 0)
    block_count = tl.sum(tl.where(e == expert, blocks, 0), 0)
    tl.store(Plan + expert, row_start)
    tl.store(Plan + 32 + expert, tile_start)
    lane = tl.arange(0, PER_EXPERT)
    tl.store(Tiles + tile_start + lane, expert, lane < block_count)


@triton.jit
def _gemm(A, AS, B, BS, Counts, Plan, Tiles, Tokens, Mixture, Out,
          T, CAP, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          DOWN: tl.constexpr, STAGES: tl.constexpr):
    tile = tl.program_id(0)
    if tile >= tl.load(Plan + 64):
        return
    col_tile = tl.program_id(1)
    expert = tl.load(Tiles + tile)
    tile_start = tl.load(Plan + 32 + expert)
    count = tl.load(Counts + expert)
    row_start = tl.load(Plan + expert)
    local_row = (tile - tile_start) * M + tl.arange(0, M)
    valid = local_row < count
    row = row_start + local_row
    route_slot = expert * T + local_row
    token = tl.load(Tokens + route_slot, valid, 0)
    col = col_tile * 128 + tl.arange(0, 128)
    k = tl.arange(0, 128)
    weight = B + (expert * N + col[:, None]) * K + k[None, :]
    weight_scale = BS + (expert * (N // 128) + col_tile) * (K // 128)
    acc = tl.zeros((M, 128), tl.float32)

    for block in tl.range(K // 128, num_stages=STAGES):
        w = tl.load(weight + block * 128)
        ws = tl.load(weight_scale + block)
        if DOWN:
            a = A + row[:, None] * (3 * K) + block * 384 + k[None, :]
            a0 = tl.load(a, valid[:, None], 0.0)
            a1 = tl.load(a + 128, valid[:, None], 0.0)
            a2 = tl.load(a + 256, valid[:, None], 0.0)
            partial = tl.dot(a0, tl.trans(w))
            partial = tl.dot(a1, tl.trans(w), partial)
            partial = tl.dot(a2, tl.trans(w), partial)
            a_scale = tl.load(AS + block * CAP + row, valid, 0.0)
        else:
            a0 = tl.load(A + token[:, None] * K + block * 128 + k[None, :],
                         valid[:, None], 0.0)
            partial = tl.dot(a0, tl.trans(w))
            a_scale = tl.load(AS + block * T + token, valid, 0.0)
        acc += partial * (a_scale[:, None] * ws)

    if DOWN:
        mixture = tl.load(Mixture + route_slot, valid, 0.0)
        tl.atomic_add(Out + token[:, None] * N + col[None, :],
                      acc * mixture[:, None], valid[:, None], sem="relaxed")
    else:
        tl.store(Out + row[:, None] * N + col[None, :], acc, valid[:, None])


@triton.jit
def _quantize_e4m3(value):
    bits = value.to(tl.uint32, bitcast=True)
    magnitude = tl.minimum(tl.abs(value), 448.0)
    exponent = tl.maximum(magnitude.to(tl.uint32, bitcast=True) >> 23, 121)
    inverse_step = ((257 - exponent) << 23).to(tl.float32, bitcast=True)
    rounded = tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;", constraints="=r,f",
        args=[magnitude * inverse_step], dtype=tl.int32, is_pure=True, pack=1)
    encoding = (exponent - 121) * 8 + rounded
    sign = (bits >> 24) & 128
    packed = (encoding | sign).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
    step = ((exponent - 3) << 23).to(tl.float32, bitcast=True)
    unpacked = tl.where(sign != 0, -rounded.to(tl.float32), rounded.to(tl.float32)) * step
    return packed, unpacked


@triton.jit
def _swiglu(GateUp, Activation, Scale, Plan, CAP, ROWS: tl.constexpr):
    first = tl.program_id(0) * ROWS
    count = tl.load(Plan + 65)
    if first >= count:
        return
    row = first + tl.arange(0, ROWS)
    block = tl.program_id(1)
    lane = tl.arange(0, 128)
    col = block * 128 + lane
    x1 = tl.load(GateUp + row[:, None] * 4096 + col[None, :],
                 row[:, None] < count, 0.0)
    x2 = tl.load(GateUp + row[:, None] * 4096 + col[None, :] + 2048,
                 row[:, None] < count, 0.0)
    # Evaluate the logistic with an exponent in [0, 1] even for large
    # negative projections. Keep the FP8 normalization in the same range.
    exp_neg_abs = tl.exp(-tl.abs(x2))
    sigmoid = tl.div_rn(tl.where(x2 >= 0, 1.0, exp_neg_abs), 1.0 + exp_neg_abs)
    value = x1 * (x2 * sigmoid)
    scale = tl.maximum(tl.max(tl.abs(value), 1) * (1.0 / 448.0), 1.0e-30)
    scaled = value * tl.div_rn(1.0, scale[:, None])
    first_fp8, first_value = _quantize_e4m3(scaled)
    rest = scaled - first_value
    second_fp8, second_value = _quantize_e4m3(rest)
    third_fp8, _ = _quantize_e4m3(rest - second_value)
    destination = Activation + row[:, None] * 6144 + block * 384 + lane[None, :]
    tl.store(destination, first_fp8, row[:, None] < count)
    tl.store(destination + 128, second_fp8, row[:, None] < count)
    tl.store(destination + 256, third_fp8, row[:, None] < count)
    tl.store(Scale + block * CAP + row, scale, row < count)


@triton.jit
def _finish(Acc, Output, SIZE, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Acc + idx, idx < SIZE, 0.0)
    tl.store(Output + idx, value, idx < SIZE)


def run(routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor, output):
    t = routing_logits.shape[0]
    if t == 0:
        return
    device = hidden_states.device
    capacity = 8 * t
    block_m = 32
    stages = 3
    tile_capacity = triton.cdiv(capacity, block_m) + 32
    counts = torch.empty((32,), device=device, dtype=torch.int32)
    plan = torch.empty((66,), device=device, dtype=torch.int32)
    tiles = torch.empty((tile_capacity,), device=device, dtype=torch.int32)
    tokens = torch.empty((32, t), device=device, dtype=torch.int32)
    mixture = torch.empty((32, t), device=device, dtype=torch.float32)
    gate_up = torch.empty((capacity, 4096), device=device, dtype=torch.float32)
    activation = torch.empty((capacity, 6144), device=device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((16, capacity), device=device, dtype=torch.float32)
    acc = torch.empty((t, 7168), device=device, dtype=torch.float32)

    flat_size = t * 7168
    output_grid = (triton.cdiv(flat_size, 2048),)
    _clear[output_grid](counts, acc, flat_size, 2048)
    _route[(t,)](routing_logits, routing_bias, counts, tokens, mixture, t,
                 int(local_expert_offset), float(routed_scaling_factor), num_warps=4)
    _plan[(32,)](counts, plan, tiles, block_m,
                 triton.next_power_of_2(triton.cdiv(t, block_m)), num_warps=1)
    _gemm[(tile_capacity, 32)](
        hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale,
        counts, plan, tiles, tokens, mixture, gate_up, t, capacity,
        block_m, 4096, 7168, False, stages, num_warps=4)
    _swiglu[(triton.cdiv(capacity, 4), 16)](
        gate_up, activation, scale, plan, capacity, 4, num_warps=4)
    _gemm[(tile_capacity, 56)](
        activation, scale, gemm2_weights, gemm2_weights_scale,
        counts, plan, tiles, tokens, mixture, acc, t, capacity,
        block_m, 7168, 2048, True, stages, num_warps=4)
    _finish[output_grid](acc, output, flat_size, 2048)
