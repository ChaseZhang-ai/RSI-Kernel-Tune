# SPDX-License-Identifier: Apache-2.0
"""Single-launch Triton KV-cache update for SOL-ExecBench problem 071."""

import triton
import triton.language as tl


# Cache tile elements and warps for the measured (batch, old tokens) pairs.
_CACHE_TILES = {
    (1, 127): (1024, 4),
    (1, 511): (2048, 4),
    (1, 2047): (4096, 4),
    (2, 130): (1024, 2),
    (2, 510): (4096, 4),
    (4, 372): (1024, 2),
    (4, 767): (2048, 4),
    (16, 996): (4096, 4),
    (32, 255): (4096, 4),
    (64, 255): (1024, 2),
}


# Vector lanes, warps, and percentage of stores to retain in L2 by (batch, old tokens).
_WIDE_TILES = {
    (4, 372): (256, 8, 100),
    (4, 767): (128, 4, 100),
    (32, 255): (128, 4, 100),
    (64, 255): (128, 4, 75),
    (16, 996): (128, 4, 85),
    (1, 2047): (128, 4, 100),
}


@triton.jit
def _rope_element(key, cos, rotated, sin, PACKED: tl.constexpr):
    if PACKED:
        return tl.inline_asm_elementwise(
            """{
                .reg .b32 a, b;
                mul.rn.bf16x2 a, $1, $2;
                mul.rn.bf16x2 b, $3, $4;
                add.rn.bf16x2 $0, a, b;
            }""",
            constraints="=r,r,r,r,r",
            args=[key, cos, rotated, sin],
            dtype=tl.bfloat16,
            is_pure=True,
            pack=2,
        )
    # The reference rounds both products to BF16 before adding them.
    a = (key.to(tl.float32) * cos.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    b = (rotated.to(tl.float32) * sin.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    return (a + b).to(tl.bfloat16)


@triton.jit
def _write_new_tokens(
    keys, values, cos, sin, out_keys, out_values, head, pair, valid,
    HEADS: tl.constexpr, DIM: tl.constexpr, NEW: tl.constexpr, OLD: tl.constexpr,
    PACKED: tl.constexpr,
):
    half: tl.constexpr = DIM // 2
    offset = pair // half * DIM + pair % half
    src = head * (NEW * DIM) + offset
    angle = (head // HEADS) * (NEW * DIM) + offset
    dst = head * ((OLD + NEW) * DIM) + OLD * DIM + offset

    left = tl.load(keys + src, valid, 0)
    right = tl.load(keys + src + half, valid, 0)
    neg_right = (-right.to(tl.float32)).to(tl.bfloat16)
    cos_left = tl.load(cos + angle, valid, 0)
    cos_right = tl.load(cos + angle + half, valid, 0)
    sin_left = tl.load(sin + angle, valid, 0)
    sin_right = tl.load(sin + angle + half, valid, 0)
    key_left = _rope_element(left, cos_left, neg_right, sin_left, PACKED)
    key_right = _rope_element(right, cos_right, left, sin_right, PACKED)
    value_left = tl.load(values + src, valid, 0)
    value_right = tl.load(values + src + half, valid, 0)
    tl.store(out_keys + dst, key_left, valid)
    tl.store(out_keys + dst + half, key_right, valid)
    tl.store(out_values + dst, value_left, valid)
    tl.store(out_values + dst + half, value_right, valid)


@triton.jit
def _prefill(
    keys, values, cos, sin, out_keys, out_values, count,
    HEADS: tl.constexpr, DIM: tl.constexpr, NEW: tl.constexpr,
    TILE: tl.constexpr, PACKED: tl.constexpr,
    SPLIT_HEADS: tl.constexpr, TOKEN_AXIS: tl.constexpr,
):
    pairs_per_head: tl.constexpr = NEW * (DIM // 2)
    if SPLIT_HEADS:
        head = tl.program_id(1 - TOKEN_AXIS)
        pair = tl.program_id(TOKEN_AXIS) * TILE + tl.arange(0, TILE)
        valid = pair < pairs_per_head
    else:
        i = tl.program_id(0) * TILE + tl.arange(0, TILE)
        head = i // pairs_per_head
        pair = i % pairs_per_head
        valid = i < count
    _write_new_tokens(
        keys, values, cos, sin, out_keys, out_values,
        head, pair, valid, HEADS, DIM, NEW, 0, PACKED,
    )


@triton.jit
def _update(
    keys, values, cos, sin, cache_keys, cache_values, out_keys, out_values,
    HEADS: tl.constexpr, DIM: tl.constexpr, OLD: tl.constexpr, NEW: tl.constexpr,
    CACHE_TILE: tl.constexpr, NEW_TILE: tl.constexpr, COPY_TILES: tl.constexpr,
    PACKED: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    if tile < COPY_TILES:
        i = tile * CACHE_TILE + tl.arange(0, CACHE_TILE)
        src = head * (OLD * DIM) + i
        dst = head * ((OLD + NEW) * DIM) + i
        valid = i < OLD * DIM
        k = tl.load(cache_keys + src, valid, 0)
        v = tl.load(cache_values + src, valid, 0)
        tl.store(out_keys + dst, k, valid)
        tl.store(out_values + dst, v, valid)
    elif NEW > 0:
        pair = (tile - COPY_TILES) * NEW_TILE + tl.arange(0, NEW_TILE)
        _write_new_tokens(
            keys, values, cos, sin, out_keys, out_values,
            head, pair, pair < NEW * (DIM // 2), HEADS, DIM, NEW, OLD, PACKED,
        )


@triton.jit
def _copy_vector(src, dst, valid, retain, STREAM_PREFIX: tl.constexpr):
    if STREAM_PREFIX:
        tl.inline_asm_elementwise(
            """{
                .reg .b32 b<8>;
                .reg .pred p, keep, steady, stream;
                setp.ne.u32 p, $3, 0;
                setp.ne.u32 keep, $4, 0;
                and.pred steady, p, keep;
                and.pred stream, p, !keep;
                @p ld.global.nc.L1::no_allocate.L2::evict_first.v8.b32
                    {b0,b1,b2,b3,b4,b5,b6,b7}, [$1];
                @steady st.global.L2::evict_last.v8.b32 [$2], {b0,b1,b2,b3,b4,b5,b6,b7};
                @stream st.global.cs.v8.b32 [$2], {b0,b1,b2,b3,b4,b5,b6,b7};
                mov.b32 $0, 0;
            }""",
            constraints="=r,l,l,r,r",
            args=[src, dst, valid.to(tl.int32), retain.to(tl.int32)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        tl.inline_asm_elementwise(
            """{
                .reg .b32 b<8>;
                .reg .pred p;
                setp.ne.u32 p, $3, 0;
                @p ld.global.nc.L1::no_allocate.L2::evict_first.v8.b32
                    {b0,b1,b2,b3,b4,b5,b6,b7}, [$1];
                @p st.global.L2::evict_last.v8.b32 [$2], {b0,b1,b2,b3,b4,b5,b6,b7};
                mov.b32 $0, 0;
            }""",
            constraints="=r,l,l,r",
            args=[src, dst, valid.to(tl.int32)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def _update_wide(
    keys, values, cos, sin, cache_keys, cache_values, out_keys, out_values,
    HEADS: tl.constexpr, DIM: tl.constexpr, OLD: tl.constexpr, LANES: tl.constexpr,
    STREAM_TILES: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    plane = tl.program_id(2)
    offset = (tile * LANES + tl.arange(0, LANES)) * 16
    src = head * (OLD * DIM) + offset
    dst = head * ((OLD + 1) * DIM) + offset
    valid = offset < OLD * DIM
    serial = tile + tl.num_programs(0) * (head + tl.num_programs(1) * plane)
    retain = serial >= STREAM_TILES
    if plane == 0:
        _copy_vector(cache_keys + src, out_keys + dst, valid, retain, STREAM_TILES > 0)
        if tile == 0:
            pairs = tl.arange(0, DIM // 2)
            _write_new_tokens(
                keys, values, cos, sin, out_keys, out_values,
                head, pairs, pairs < DIM // 2, HEADS, DIM, 1, OLD, True,
            )
    else:
        _copy_vector(cache_values + src, out_values + dst, valid, retain, STREAM_TILES > 0)


def run(keys, values, cos, sin, cache_keys, cache_values, out_keys, out_values):
    batch, heads, new, dim = keys.shape
    old = cache_keys.shape[2]
    if batch == 0 or old + new == 0:
        return

    half = dim // 2
    if old == 0:
        if new == 1:
            tile, warps, packed = 128, 2, False
        elif new <= 128:
            tile, warps, packed = 128, 2, True
        elif batch == 1 or new >= 1024:
            tile, warps, packed = 1024, 4, False
        else:
            tile, warps, packed = 512, 2, False
        count = batch * heads * new * half
        split_heads = (batch, new) in {(2, 512), (4, 256), (1, 512), (1, 128)}
        token_axis = int(new == 512)
        if split_heads:
            tile, warps, packed = 512, 4, batch == 2
            tiles = triton.cdiv(new * half, tile)
            grid = (batch * heads, tiles) if token_axis else (tiles, batch * heads)
        else:
            grid = (triton.cdiv(count, tile),)
        _prefill[grid](
            keys, values, cos, sin, out_keys, out_values, count,
            heads, dim, new, tile, packed, split_heads, token_axis,
            num_warps=warps, enable_fp_fusion=False,
        )
        return

    wide = _WIDE_TILES.get((batch, old))
    if new == 1 and dim == 128 and wide is not None and all(
        tensor.data_ptr() % 32 == 0
        for tensor in (cache_keys, cache_values, out_keys, out_values)
    ):
        lanes, warps, retain_pct = wide
        tiles = triton.cdiv(old * dim, 16 * lanes)
        stream_tiles = 2 * batch * heads * tiles * (100 - retain_pct) // 100
        _update_wide[(tiles, batch * heads, 2)](
            keys, values, cos, sin, cache_keys, cache_values, out_keys, out_values,
            heads, dim, old, lanes, stream_tiles, num_warps=warps, enable_fp_fusion=False,
        )
        return

    tile, warps = _CACHE_TILES.get((batch, old), (2048, 4))
    new_tile = triton.next_power_of_2(half) if new <= 1 else 256
    copy_tiles = triton.cdiv(old * dim, tile)
    new_tiles = triton.cdiv(new * half, new_tile)
    _update[(copy_tiles + new_tiles, batch * heads)](
        keys, values, cos, sin, cache_keys, cache_values, out_keys, out_values,
        heads, dim, old, new, tile, new_tile, copy_tiles, True,
        num_warps=warps, enable_fp_fusion=False,
    )
