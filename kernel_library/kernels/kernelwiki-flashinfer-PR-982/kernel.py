"""KernelWiki artifact PR-982 (flashinfer-ai/flashinfer)

Reference bundle — a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission"""

_SOURCES = {
    'key-files/flashinfer/triton/__init__.py': r"""from . import cascade  # noqa: F401
from . import sm_constraint_gemm  # noqa: F401
""",
    'key-files/flashinfer/triton/kernels/sm_constraint_gemm.py': r"""import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]


def matmul_get_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
            },
            num_stages=s,
            num_warps=w,
        )
        for BM in [128]
        for BN in [128]
        for BK in [64]
        for s in ([3])
        for w in [4]
    ]


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def gemm_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # NOTE: There is currently a bug in blackwell pipelining that means it can't handle a value being
    # used in both the prologue and epilogue, so we duplicate the counters as a work-around.
    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )

            a = tl.load(
                a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        c = accumulator.to(c_ptr.dtype.element_ty)

        c = tl.fma(c, alpha, beta * tl.load(c_ptrs, mask=c_mask))
        tl.store(c_ptrs, c, mask=c_mask)


@triton.jit(launch_metadata=_matmul_launch_metadata)
def gemm_kernel_descriptor_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    M,
    N,
    K,  #
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,
):  #
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    # tile_id_c is used in the epilogue to break the dependency between
    # the prologue and the epilogue
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            acc0 = tl.fma(acc0, alpha, beta * c_desc.load([offs_cm, offs_cn]))
            acc1 = tl.fma(
                acc1, alpha, beta * c_desc.load([offs_cm, offs_cn + BLOCK_SIZE_N // 2])
            )
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        else:
            accumulator = tl.fma(
                accumulator, alpha, beta * c_desc.load([offs_cm, offs_cn])
            )
            c = accumulator.to(dtype)
            c_desc.store([offs_cm, offs_cn], c)


# only for testing
@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,  #
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c = tl.fma(c, alpha, beta * tl.load(c_ptrs, mask=c_mask))
    tl.store(c_ptrs, c, mask=c_mask)
""",
    'key-files/flashinfer/triton/sm_constraint_gemm.py': r'''from typing import Optional

import torch
import triton

from .kernels.sm_constraint_gemm import (
    gemm_kernel,
    gemm_kernel_descriptor_persistent,
    gemm_kernel_persistent,
)
from .utils import check_device, check_dim, check_input


def gemm_persistent(a, b, c=None, alpha=1.0, beta=0.0, out_dtype=None, num_sms=None):
    """
    GEMM operation with SM constraint by Triton.
    C = alpha * (a @ b.T) + beta * C

    Args:
        a: The first input matrix. Shape: (M, K)
        b: The second input matrix. Shape: (K, N)
        c: The output matrix. Shape: (M, N). In-place epilogue is supported. Expected to be out_dtype (if not specified, same as a.dtype, but fp8 --> bf16).
        alpha: The scaling factor for the product of a and b.
        beta: The scaling factor for the output matrix c.
        out_dtype: The dtype of the output matrix. Default: fp8 --> bf16. Otherwise, same as a.dtype.
        num_sms: The number of SMs to use for the computation.
    """

    # Check inputs.
    check_input(a)
    # b can be non-contiguous
    check_device([a, b])
    check_dim(2, a)
    check_dim(2, b)

    if c is not None:
        check_input(c)
        check_device([c])
        check_dim(2, c)

    assert a.shape[1] == b.shape[0], "Incompatible dimensions between a and b"
    assert a.dtype == b.dtype, "Incompatible dtypes between a and b"

    if c is not None:
        assert a.shape[0] == c.shape[0], "Incompatible dimensions between a and c"
        assert b.shape[1] == c.shape[1], "Incompatible dimensions between b and c"

    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    out_dtype = (
        out_dtype
        if out_dtype
        else dtype if dtype != torch.float8_e4m3fn else torch.bfloat16
    )

    assert (
        c is None or c.dtype == out_dtype
    ), "Incompatible dtypes between c and out_dtype"

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=out_dtype) if c is None else c

    # Set num_sms to be 100% of the available SMs
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_sms = NUM_SMS if num_sms is None else min(NUM_SMS, num_sms)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        min(
            num_sms,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )

    gemm_kernel_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        alpha=alpha,
        beta=beta,
        NUM_SMS=num_sms,
    )
    return c


def gemm(a, b, c=None, alpha=1.0, beta=0.0, out_dtype=None):
    """
    GEMM operation without SM constraint by Triton.
    C = alpha * (a @ b.T) + beta * C

    Args:
        a: The first input matrix. Shape: (M, K)
        b: The second input matrix. Shape: (K, N)
        c: The output matrix. Shape: (M, N). In-place epilogue is supported. Expected to be out_dtype (if not specified, same as a.dtype, but fp8 --> bf16).
        alpha: The scaling factor for the product of a and b.
        beta: The scaling factor for the output matrix c.
        out_dtype: The dtype of the output matrix. Default: fp8 --> bf16. Otherwise, same as a.dtype.
        num_sms: The number of SMs to use for the computation.
    """
    # Check inputs.
    check_input(a)
    # b can be non-contiguous
    check_device([a, b])
    check_dim(2, a)
    check_dim(2, b)

    if c is not None:
        check_input(c)
        check_device([c])
        check_dim(2, c)

    assert a.shape[1] == b.shape[0], "Incompatible dimensions between a and b"
    assert a.dtype == b.dtype, "Incompatible dtypes between a and b"

    if c is not None:
        assert a.shape[0] == c.shape[0], "Incompatible dimensions between a and c"
        assert b.shape[1] == c.shape[1], "Incompatible dimensions between b and c"

    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    out_dtype = (
        out_dtype
        if out_dtype
        else dtype if dtype != torch.float8_e4m3fn else torch.bfloat16
    )

    assert (
        c is None or c.dtype == out_dtype
    ), "Incompatible dtypes between c and out_dtype"

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=out_dtype) if c is None else c

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        alpha=alpha,
        beta=beta,
    )
    return c


def gemm_descriptor_persistent(
    a,
    b,
    c=None,
    alpha=1.0,
    beta=0.0,
    out_dtype=None,
    num_sms=None,
    EPILOGUE_SUBTILE=False,
):
    """
    GEMM operation with SM constraint by Triton.
    Requires TMA support and descriptor creation.
    C = alpha * (a @ b.T) + beta * C

    Note:
        - K and N must be greater than 16B.
        - Support float16, float8_e4m3fn, bfloat16.
        - float32 is not supported due to performance issues.

    Args:
        a: The first input matrix. Shape: (M, K)
        b: The second input matrix. Shape: (N, K)
        c: The output matrix. Shape: (M, N). In-place epilogue is supported. Expected to be out_dtype (if not specified, same as a.dtype, but fp8 --> bf16).
        alpha: The scaling factor for the product of a and b.
        beta: The scaling factor for the output matrix c.
        out_dtype: The dtype of the output matrix. Default: fp8 --> bf16. Otherwise, same as a.dtype.
        num_sms: The number of SMs to use for the computation.
        EPILOGUE_SUBTILE: Whether to use the epilogue subtile optimization.
    """
    # Check inputs.
    check_input(a)
    check_input(b)
    check_device([a, b])
    check_dim(2, a)
    check_dim(2, b)

    if c is not None:
        check_input(c)
        check_device([c])
        check_dim(2, c)

    assert a.shape[1] == b.shape[1], "Incompatible dimensions between a and b"
    assert a.dtype == b.dtype, "Incompatible dtypes between a and b"

    if c is not None:
        assert a.shape[0] == c.shape[0], "Incompatible dimensions between a and c"
        assert b.shape[0] == c.shape[1], "Incompatible dimensions between b and c"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    out_dtype = (
        out_dtype
        if out_dtype
        else dtype if dtype != torch.float8_e4m3fn else torch.bfloat16
    )

    # check on TMA tensor map swizzling granularity
    # Swizzle 16B chunks within at least 32B span
    if dtype == torch.float8_e4m3fn:
        assert K >= 16, "Least chunk size must be 16B"
        assert N >= 16, "Least chunk size must be 16B"
    else:
        assert K >= 8, "Least chunk size must be 16B"
        assert N >= 8, "Least chunk size must be 16B"

    c = torch.empty((M, N), device=a.device, dtype=out_dtype) if c is None else c

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_sms = NUM_SMS if num_sms is None else min(NUM_SMS, num_sms)

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (
        min(
            num_sms,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )

    gemm_kernel_descriptor_persistent[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        alpha,
        beta,
        NUM_SMS=num_sms,  #
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128 if dtype != torch.float32 else 64,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        num_stages=3,
        num_warps=8,
        EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
    )
    return c
''',
}



def run(*args):
    """Not a submission: this artifact is reference material (see `_SOURCES`)."""
    raise NotImplementedError(
        "reference bundle (a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission): adapt it first (`forge lib adapt`)")
