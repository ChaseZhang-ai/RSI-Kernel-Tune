"""The three smokes the probe runs: Triton, CuTe DSL, inline CUDA -- as inspectable source.

The implementations are real functions in ``probe_program`` (they are called in that
process; nothing is built and executed at run time), and this module hands out their source
text so a test -- or a person -- can check that each one *parses* before it is ever sent
anywhere, and can read what "the smoke" means without opening the program:

    smoke_sources.source("triton")      the Triton smoke's source (a @triton.jit add)
    smoke_sources.source("cutedsl")     the CuTe DSL smoke's source (@cute.kernel / @cute.jit)
    smoke_sources.source("cuda_inline") the inline-CUDA smoke's function (a TF32 mma.sync
                                        built by torch.utils.cpp_extension for the running
                                        device)
    smoke_sources.cuda_source()         the CUDA C++ that smoke compiles

Each smoke answers three questions with ``compile`` / ``run`` / ``ms``: can this stack build a
kernel here, does the kernel produce the right numbers here, and what does one call read.
"""

from __future__ import annotations

import inspect

from rsi_kernel_tune.targethw import probe_program

NAMES = ("triton", "cutedsl", "cuda_inline")

_FUNCTIONS = {
    "triton": probe_program.smoke_triton,
    "cutedsl": probe_program.smoke_cutedsl,
    "cuda_inline": probe_program.smoke_cuda_inline,
}


def source(name: str) -> str:
    """The smoke ``name``'s source text (its function, plus the kernels it defines)."""
    try:
        fn = _FUNCTIONS[name]
    except KeyError:
        raise KeyError(f"unknown smoke {name!r}; known: {list(NAMES)}") from None
    return inspect.getsource(fn)


def all_sources() -> dict[str, str]:
    return {name: source(name) for name in NAMES}


def cuda_source() -> str:
    """The CUDA C++ source the inline-CUDA smoke builds."""
    return probe_program._CUDA_SOURCE


def compiles(name: str) -> bool:
    """Does the smoke's source parse as Python (before it is sent to another machine)?"""
    text = source(name)
    try:
        compile(text, f"<smoke {name}>", "exec")
    except SyntaxError:
        return False
    return True


def cuda_source_compiles() -> bool:
    """A cheap shape check on the CUDA source: the binding and the kernel are both there."""
    text = cuda_source()
    return "__global__ void mma_smoke_kernel" in text and "mma_smoke(torch::Tensor" in text


__all__ = ["NAMES", "all_sources", "compiles", "cuda_source", "cuda_source_compiles", "source"]
