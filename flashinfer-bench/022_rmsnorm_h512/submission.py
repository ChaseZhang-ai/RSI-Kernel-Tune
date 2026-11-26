"""Destination-passing BF16 RMSNorm with a fixed hidden size of 512."""

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


# Reduction adapted from kernel_library/kernels/sol-example-triton-rmsnorm/kernel.py.
@gluon.jit
def _blocked(hidden_states, weight, output, batch_size, ROWS: gl.constexpr, WARPS: gl.constexpr,
             WARP_COLS: gl.constexpr, VECTOR: gl.constexpr):
    layout: gl.constexpr = BlockedLayout([1, VECTOR], [1, 32], [WARPS // WARP_COLS, WARP_COLS], [1, 0])
    rows = gl.program_id(0) * ROWS + gl.arange(0, ROWS, layout=SliceLayout(1, layout))
    cols = gl.arange(0, 512, layout=SliceLayout(0, layout))
    offsets = rows[:, None] * 512 + cols[None, :]
    valid = rows[:, None] < batch_size
    x = gl.load(hidden_states + offsets, valid, 0).to(gl.float32)
    w = gl.load(weight + cols, eviction_policy="evict_last").to(gl.float32)
    inv_rms = gl.rsqrt(gl.sum(x * x, axis=1) * (1.0 / 512) + 1.0e-6)
    result = ((x * inv_rms[:, None]) * w[None, :]).to(gl.bfloat16)
    gl.store(output + offsets, result, valid)

# Warp-per-row 256-bit memory operations are inspired by QianQu's
# CompoundingKernel-Explorer FlashInfer/022_rmsnorm_h512/src/main.py.
# https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/022_rmsnorm_h512
# Each register holds two BF16 values. Bit shifts widen to two FP32 values,
# which can be multiplied and accumulated in packed FP32 registers.
def _read_activation_ptx(cache_modifier):
    words = ", ".join(f"${i}" for i in range(8))
    lines = [
        "{",
        ".reg .b32 lo, hi;",
        ".reg .b64 pair, a, b;",
        ".reg .f32 l, h;",
        f"ld.global{cache_modifier}.v8.u32 {{{words}}}, [$9];",
    ]
    for i in range(8):
        lines.extend([f"shl.b32 lo, ${i}, 16;", f"and.b32 hi, ${i}, 0xffff0000;",
                      "mov.b64 pair, {lo, hi};"])
        acc = "a" if i % 2 == 0 else "b"
        if i < 2:
            lines.append(f"mul.rn.f32x2 {acc}, pair, pair;")
        else:
            lines.append(f"fma.rn.f32x2 {acc}, pair, pair, {acc};")
    lines += ["add.rn.f32x2 a, a, b;", "mov.b64 {l, h}, a;", "add.rn.f32 $8, l, h;", "}"]
    return "\n".join(lines)


def _store_ptx():
    words = ", ".join(f"out{i}" for i in range(8))
    lines = [
        "{",
        ".reg .pred valid;",
        ".reg .b32 lo, hi, out<8>;",
        ".reg .b64 x, w, inv, y;",
        ".reg .f32 l, h;",
        "mov.b32 $0, 0;",
        "setp.ne.u32 valid, $19, 0;",
        "mov.b64 inv, {$17, $17};",
    ]
    for i in range(8):
        lines += [
            f"shl.b32 lo, ${i + 1}, 16;",
            f"and.b32 hi, ${i + 1}, 0xffff0000;",
            "mov.b64 x, {lo, hi};",
            f"shl.b32 lo, ${i + 9}, 16;",
            f"and.b32 hi, ${i + 9}, 0xffff0000;",
            "mov.b64 w, {lo, hi};",
            "mul.rn.f32x2 y, x, inv;",
            "mul.rn.f32x2 y, y, w;",
            "mov.b64 {l, h}, y;",
            f"cvt.rn.bf16x2.f32 out{i}, h, l;",
        ]
    lines += [f"@valid st.global.v8.u32 [$18], {{{words}}};", "}"]
    return "\n".join(lines)


_READ_DEFAULT = gl.constexpr(_read_activation_ptx(""))
_READ_STREAMING = gl.constexpr(_read_activation_ptx(".nc.L1::no_allocate"))
_STORE = gl.constexpr(_store_ptx())
_WORDS = gl.constexpr((gl.uint32,) * 8)
_ACTIVATION_TYPES = gl.constexpr((gl.uint32,) * 8 + (gl.float32,))
_LOAD_CONSTRAINTS = gl.constexpr("=r," * 8 + "l")
_STORE_CONSTRAINTS = gl.constexpr("=r," + "r," * 16 + "f,l,r")


@gluon.jit
def _activation(pointer, STREAM: gl.constexpr):
    if STREAM:
        asm: gl.constexpr = _READ_STREAMING
    else:
        asm: gl.constexpr = _READ_DEFAULT
    return gl.inline_asm_elementwise(
        asm, constraints="=r," * 8 + "=f,l", args=[pointer],
        dtype=_ACTIVATION_TYPES, is_pure=True, pack=1,
    )


@gluon.jit
def _weight(pointer):
    return gl.inline_asm_elementwise(
        "ld.global.nc.v8.u32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];",
        constraints=_LOAD_CONSTRAINTS, args=[pointer], dtype=_WORDS, is_pure=True, pack=1,
    )


@gluon.jit
def _write(x, w, inv, pointer, valid):
    gl.inline_asm_elementwise(
        _STORE, constraints=_STORE_CONSTRAINTS,
        args=[x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
              w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7],
              inv, pointer, valid.to(gl.uint32)],
        dtype=gl.uint32, is_pure=False, pack=1,
    )


@gluon.jit
def _packed(hidden_states, weight, output, batch_size, ROWS: gl.constexpr, WARPS: gl.constexpr,
           READ_STREAM: gl.constexpr):
    layout: gl.constexpr = BlockedLayout([1, 1], [1, 32], [WARPS, 1], [1, 0])
    rows = gl.program_id(0) * ROWS + gl.arange(0, ROWS, layout=SliceLayout(1, layout))
    cols = gl.arange(0, 32, layout=SliceLayout(0, layout)) * 16
    input_offsets = gl.minimum(rows[:, None], batch_size - 1) * 512 + cols[None, :]
    a, b, c, d, e, f, g, h, lane_sums = _activation(hidden_states + input_offsets, READ_STREAM)
    w = _weight(weight + cols)
    inv = gl.rsqrt(gl.sum(lane_sums, axis=1) * (1.0 / 512) + 1.0e-6)
    output_offsets = rows[:, None] * 512 + cols[None, :]
    ww = (w[0][None, :], w[1][None, :], w[2][None, :], w[3][None, :],
          w[4][None, :], w[5][None, :], w[6][None, :], w[7][None, :])
    _write((a, b, c, d, e, f, g, h), ww, inv[:, None], output + output_offsets,
           rows[:, None] < batch_size)



def run(hidden_states, weight, output):
    batch = hidden_states.shape[0]
    if batch == 0:
        return
    if batch <= 64:
        _blocked[(batch,)](hidden_states, weight, output, batch, 1, 4, 4, 8, num_warps=4)
        return

    # The 256-bit instructions require 32-byte aligned pointers.
    if (hidden_states.data_ptr() | weight.data_ptr() | output.data_ptr()) & 31:
        _blocked[(triton.cdiv(batch, 4),)](
            hidden_states, weight, output, batch, 4, 4, 1, 8, num_warps=4,
        )
        return

    # At medium batch sizes four rows expose more CTAs. At larger sizes,
    # each warp handles two rows and reuses its weight fragment.
    rows = 4 if batch <= 1024 else 8
    _packed[(triton.cdiv(batch, rows),)](
        hidden_states, weight, output, batch, rows, 4, batch <= 1024, num_warps=4,
    )
