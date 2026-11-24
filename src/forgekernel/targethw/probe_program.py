"""The target probe program: collect one machine's GPU, stack and smoke verdicts.

This file is deliberately standalone -- stdlib, plus the GPU stack imported lazily inside the
functions -- because it is also *shipped verbatim* to a remote machine that has no
forgekernel: ``forge target probe --remote`` puts it in a job tree and runs it with the same
ssh delivery a GPU job uses. It runs on the machine it describes and prints one JSON
document; the parsing and the comparison live in ``probe`` / ``parity`` on the control side.

Two things it never does:

* it never names an architecture -- the inline-CUDA smoke derives its ``-gencode`` from
  ``torch.cuda.get_device_capability()``, which is also the pattern every submission must
  follow (rule 2 of the target profile);
* it never runs a program it built at run time. The three smokes are ordinary functions,
  called in this process, each in its own ``try``: a smoke that fails produces a verdict, not
  a dead probe. (A smoke that segfaults would take the process with it -- ``--no-smokes``
  exists for that, and for machines where only the stack report matters.)

Every external command is a literal argv list written at its call site (``nvidia-smi``,
``nvcc``): nothing here builds or forwards a command line.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys

SCHEMA = "forgekernel.target-probe/1"
PAYLOAD_MARKER = "PROBE_JSON "
SMOKE_MARKER = "SMOKE_JSON "


def _error(rc: int, err: str) -> dict:
    return {"rc": rc, "out": "", "err": err}


def nvidia_smi_query() -> dict:
    """Name, CC, memory, driver and the three clock columns, as CSV."""
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,compute_cap,memory.total,driver_version,"
             "clocks.current.sm,clocks.current.memory,clocks.applications.sm,"
             "clocks.applications.memory,clocks.max.sm,clocks.max.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60)
        return {"rc": proc.returncode, "out": proc.stdout, "err": proc.stderr[-4000:]}
    except FileNotFoundError:
        return _error(127, "command not found")
    except subprocess.TimeoutExpired:
        return _error(124, "timeout")
    except Exception as e:  # noqa: BLE001 -- the report says why, the probe goes on
        return _error(1, f"{type(e).__name__}: {e}")


def nvidia_smi_q() -> dict:
    """The full ``-q`` report (the fallback for clocks on older drivers)."""
    try:
        proc = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True,
                              timeout=120)
        return {"rc": proc.returncode, "out": proc.stdout, "err": proc.stderr[-4000:]}
    except FileNotFoundError:
        return _error(127, "command not found")
    except subprocess.TimeoutExpired:
        return _error(124, "timeout")
    except Exception as e:  # noqa: BLE001
        return _error(1, f"{type(e).__name__}: {e}")


def nvcc_version() -> dict:
    """``nvcc --version`` of the toolkit on PATH."""
    try:
        proc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True,
                              timeout=60)
        return {"rc": proc.returncode, "out": proc.stdout, "err": proc.stderr[-4000:]}
    except FileNotFoundError:
        return _error(127, "command not found")
    except subprocess.TimeoutExpired:
        return _error(124, "timeout")
    except Exception as e:  # noqa: BLE001
        return _error(1, f"{type(e).__name__}: {e}")


def _dist(*names):
    """The installed version of the first of ``names`` that is present, else None."""
    import importlib.metadata as md

    for name in names:
        try:
            return md.version(name)
        except Exception:  # noqa: BLE001
            continue
    return None


def _verdict(**values) -> dict:
    out = {"compile": False, "run": False, "ms": None, "error": "", "detail": {}}
    out.update(values)
    return out


def _time_calls(call, warmup: int = 5, timed: int = 50) -> float:
    """Milliseconds per call, the card's way: warm up, then time, on this device."""
    import torch

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(timed):
        call()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / timed)


# -- the three smokes --------------------------------------------------------------------

def smoke_triton() -> dict:
    """``@triton.jit``: does Triton compile and run a kernel on this GPU?"""
    try:
        import torch
        import triton
        import triton.language as tl
    except Exception as e:  # noqa: BLE001
        return _verdict(error=f"import: {type(e).__name__}: {e}")

    @triton.jit
    def _add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        a = tl.load(x_ptr + off, mask=mask)
        b = tl.load(y_ptr + off, mask=mask)
        tl.store(out_ptr + off, a + b, mask=mask)

    detail = {"triton": getattr(triton, "__version__", "?"),
              "torch": getattr(torch, "__version__", "?")}
    try:
        n, block = 1 << 20, 1024
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        grid = (triton.cdiv(n, block),)
        _add[grid](x, y, out, n, BLOCK=block)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        return _verdict(error=f"launch: {type(e).__name__}: {e}", detail=detail)
    result = _verdict(compile=True, detail=detail)
    result["run"] = bool(torch.allclose(out, x + y, atol=1e-5, rtol=1e-5))
    if not result["run"]:
        result["error"] = "numerical mismatch"
    result["detail"]["device"] = torch.cuda.get_device_name(0)
    try:
        result["ms"] = _time_calls(lambda: _add[grid](x, y, out, n, BLOCK=block))
    except Exception as e:  # noqa: BLE001
        result["error"] = f"timing: {type(e).__name__}: {e}"
    return result


def smoke_cutedsl() -> dict:
    """``@cute.kernel`` + ``@cute.jit``: does the CuTe DSL compile and run here?"""
    try:
        import torch
        import cutlass  # noqa: F401 -- the package the DSL lives in
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
    except Exception as e:  # noqa: BLE001
        return _verdict(error=f"import: {type(e).__name__}: {e}")

    block = 128

    @cute.kernel
    def _scale_kernel(src: cute.Tensor, dst: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        index = bid * block + tid
        n = cute.size(src, mode=[0])
        if index < n:
            dst[index] = src[index] * 2.0 + 1.0

    @cute.jit
    def _scale(src: cute.Tensor, dst: cute.Tensor):
        n = cute.size(src, mode=[0])
        _scale_kernel(src, dst).launch(grid=((n + block - 1) // block, 1, 1),
                                       block=(block, 1, 1))

    detail = {"cutlass_dsl": _dist("nvidia-cutlass-dsl", "cutlass-dsl"),
              "torch": getattr(torch, "__version__", "?"),
              "cute_dsl_arch": os.environ.get("CUTE_DSL_ARCH", "")}
    try:
        n = 1 << 16
        src = torch.arange(n, dtype=torch.float32, device="cuda")
        dst = torch.zeros_like(src)
        _scale(from_dlpack(src), from_dlpack(dst))
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        return _verdict(error=f"launch: {type(e).__name__}: {e}", detail=detail)
    result = _verdict(compile=True, detail=detail)
    result["run"] = bool(torch.allclose(dst, src * 2.0 + 1.0, atol=1e-5, rtol=1e-5))
    if not result["run"]:
        result["error"] = "numerical mismatch"
    result["detail"]["device"] = torch.cuda.get_device_name(0)
    try:
        result["ms"] = _time_calls(lambda: _scale(from_dlpack(src), from_dlpack(dst)))
    except Exception as e:  # noqa: BLE001
        result["error"] = f"timing: {type(e).__name__}: {e}"
    return result


# One warp, one m16n8k8 TF32 MMA (a wmma-class instruction), D[16, 8] = A[16, 8] @ B[8, 8].
_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mma_smoke_kernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ D) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    const unsigned lane = threadIdx.x & 31u;
    const unsigned group = lane >> 2;
    const unsigned tig = lane & 3u;
    const unsigned a0 = __float_as_uint(A[group * 8 + tig]);
    const unsigned a1 = __float_as_uint(A[(group + 8) * 8 + tig]);
    const unsigned a2 = __float_as_uint(A[group * 8 + tig + 4]);
    const unsigned a3 = __float_as_uint(A[(group + 8) * 8 + tig + 4]);
    const unsigned b0 = __float_as_uint(B[tig * 8 + group]);
    const unsigned b1 = __float_as_uint(B[(tig + 4) * 8 + group]);
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    D[group * 8 + 2 * tig] = c0;
    D[group * 8 + 2 * tig + 1] = c1;
    D[(group + 8) * 8 + 2 * tig] = c2;
    D[(group + 8) * 8 + 2 * tig + 1] = c3;
#else
    for (int i = threadIdx.x; i < 128; i += blockDim.x) D[i] = 0.f;
#endif
}

torch::Tensor mma_smoke(torch::Tensor A, torch::Tensor B) {
    auto D = torch::empty({16, 8}, A.options());
    mma_smoke_kernel<<<1, 32>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                D.data_ptr<float>());
    return D;
}
"""


def device_gencode() -> str:
    """The nvcc gencode for the running device (never a literal architecture)."""
    import torch

    major, minor = torch.cuda.get_device_capability(0)
    accelerated = {(9, 0), (10, 0), (10, 3), (11, 0), (12, 0), (12, 1)}
    suffix = "a" if (major, minor) in accelerated else ""
    cc = f"{major}{minor}{suffix}"
    return f"-gencode=arch=compute_{cc},code=sm_{cc}"


def smoke_cuda_inline() -> dict:
    """``torch.utils.cpp_extension``: can a submission embed and build CUDA here?"""
    try:
        import torch
        from torch.utils.cpp_extension import load_inline
    except Exception as e:  # noqa: BLE001
        return _verdict(error=f"import: {type(e).__name__}: {e}")
    try:
        gencode = device_gencode()
    except Exception as e:  # noqa: BLE001
        return _verdict(error=f"device: {type(e).__name__}: {e}")
    detail = {"gencode": gencode, "torch": getattr(torch, "__version__", "?")}
    try:
        module = load_inline(name="fk_smoke_mma",
                             cpp_sources="torch::Tensor mma_smoke(torch::Tensor A, "
                                         "torch::Tensor B);",
                             cuda_sources=_CUDA_SOURCE, functions=["mma_smoke"],
                             extra_cuda_cflags=[gencode, "-O3"], verbose=False)
    except Exception as e:  # noqa: BLE001
        return _verdict(error=f"load_inline: {type(e).__name__}: {e}", detail=detail)
    try:
        torch.manual_seed(0)
        a = torch.randint(-4, 5, (16, 8), device="cuda", dtype=torch.float32)
        b = torch.randint(-4, 5, (8, 8), device="cuda", dtype=torch.float32)
        d = module.mma_smoke(a, b)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        return _verdict(compile=True, error=f"call: {type(e).__name__}: {e}", detail=detail)
    result = _verdict(compile=True, detail=detail)
    result["run"] = bool(torch.allclose(d, a @ b, atol=1e-3, rtol=1e-3))
    if not result["run"]:
        result["error"] = "numerical mismatch"
    result["detail"]["device"] = torch.cuda.get_device_name(0)
    try:
        result["ms"] = _time_calls(lambda: module.mma_smoke(a, b))
    except Exception as e:  # noqa: BLE001
        result["error"] = f"timing: {type(e).__name__}: {e}"
    return result


SMOKES = {
    "triton": smoke_triton,
    "cutedsl": smoke_cutedsl,
    "cuda_inline": smoke_cuda_inline,
}


# -- the report --------------------------------------------------------------------------

def _host() -> dict:
    return {"hostname": platform.node(), "platform": platform.platform(),
            "python": platform.python_version(), "executable": sys.executable}


def _python_info() -> dict:
    info = {"python": platform.python_version()}
    try:
        import torch
        info["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                         "available": bool(torch.cuda.is_available())}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["torch_device"] = {
                "index": 0, "name": props.name,
                "compute_capability": [props.major, props.minor],
                "multi_processor_count": props.multi_processor_count,
                "memory_total_mib": int(props.total_memory // (1 << 20)),
                "l2_cache_mib": int(getattr(props, "L2_cache_size", 0) // (1 << 20)),
            }
    except Exception as e:  # noqa: BLE001
        info["torch"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        import triton
        info["triton"] = {"version": triton.__version__,
                          "distributions": [n for n in ("fbtriton", "triton") if _dist(n)]}
    except Exception as e:  # noqa: BLE001
        info["triton"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        import cutlass.cute  # noqa: F401 -- importing it is the check
        info["cutlass_dsl"] = {"version": _dist("nvidia-cutlass-dsl", "cutlass-dsl"),
                               "import": "ok"}
    except Exception as e:  # noqa: BLE001
        info["cutlass_dsl"] = {"version": _dist("nvidia-cutlass-dsl", "cutlass-dsl"),
                               "error": f"{type(e).__name__}: {e}"}
    info["cupti"] = {"version": _dist("nvidia-cupti", "cupti-python")}
    info["cuda_toolkit"] = {"version": _dist("nvidia-cuda-nvcc-cu13", "cuda-toolkit")}
    return info


def _env() -> dict:
    names = ("CUDA_HOME", "CUDA_PATH", "CUTE_DSL_ARCH", "FK_CUDA_ARCH", "FK_TARGET_SMS",
             "CUTLASS_DIR", "LD_LIBRARY_PATH")
    return {k: os.environ[k] for k in names if os.environ.get(k)}


def collect(*, include_smokes: bool = True) -> dict:
    """The machine as one JSON-able document (the schema ``probe.parse_report`` reads)."""
    report = {
        "schema": SCHEMA,
        "host": _host(),
        "nvidia_smi_query": nvidia_smi_query(),
        "nvidia_smi_q": nvidia_smi_q(),
        "nvcc": nvcc_version(),
        "python_info": _python_info(),
        "smokes": {},
        "env": _env(),
    }
    if include_smokes:
        for name, smoke in SMOKES.items():
            try:
                report["smokes"][name] = smoke()
            except Exception as e:  # noqa: BLE001 -- one smoke cannot take the probe down
                report["smokes"][name] = _verdict(error=f"{type(e).__name__}: {e}")
    return report


def payload_line(report: dict) -> str:
    return PAYLOAD_MARKER + json.dumps(report)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    report = collect(include_smokes="--no-smokes" not in argv)
    print(payload_line(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
