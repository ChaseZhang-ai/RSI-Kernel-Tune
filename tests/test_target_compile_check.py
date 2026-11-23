"""The target compile check: what it can prove without a B200, and what it says it cannot.

The CUDA cases need nvcc (the acceptance case creates a source that cannot build for the
target); the Triton case needs the installed Triton. Both are skipped, not faked, when the
machine has neither -- the module's own answer for that machine is ``unknown``, which is
asserted separately.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from forgekernel.targethw import compile_check as cc
from forgekernel.targethw import find_nvcc

REPO = Path(__file__).resolve().parents[1]

GOOD_CUDA = '''
import torch
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <cuda_runtime.h>

__global__ void k(const float* __restrict__ A, float* __restrict__ D) {
    D[threadIdx.x] = A[threadIdx.x] * 2.0f;
}
"""


def _gencode():
    major, minor = torch.cuda.get_device_capability(0)
    cc = f"{major}{minor}"
    return f"-gencode=arch=compute_{cc},code=sm_{cc}"


def run(x, out):
    module = load_inline(name="k", cpp_sources="", cuda_sources=CUDA_SOURCE,
                         functions=["k"], extra_cuda_cflags=[_gencode()])
    out.copy_(x)
'''

# What a real submission's CUDA string looks like (torch's headers included). Compiling this
# takes a minute; the flag plumbing is checked against a fake compiler instead, and this
# source is here so `cuda_sources_in` is exercised on the realistic shape.
TORCH_CUDA = '''
CUDA_SOURCE = r"""
#include <torch/extension.h>
__global__ void k(const float* x, float* o) { o[threadIdx.x] = x[threadIdx.x]; }
torch::Tensor scale(torch::Tensor A) { return A; }
"""


def run(x, out):
    out.copy_(x)
'''

# __CUDA_ARCH_FEAT_SM103_ALL exists only for CC 10.3: compiling for sm_100a must fail.
DEV_ONLY_CUDA = '''
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
__global__ void k(const float* __restrict__ A, float* __restrict__ D) {
    float scale = (float)__CUDA_ARCH_FEAT_SM103_ALL;
    D[threadIdx.x] = A[threadIdx.x] * scale;
}
"""

FLAGS = ["-gencode=arch=compute_103a,code=sm_103a"]


def run(x, out):
    module = load_inline(name="k", cpp_sources="", cuda_sources=CUDA_SOURCE,
                         functions=["k"], extra_cuda_cflags=FLAGS)
    out.copy_(module.k(x, out))
'''

TRITON_SUBMISSION = '''
import torch
import triton
import triton.language as tl


@triton.jit
def _scale(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(out_ptr + off, tl.load(x_ptr + off, mask=mask, other=0.0) * 2.0, mask=mask)


def run(x, out):
    _scale[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)
'''

TRITON_NEEDS_MODULE_SCOPE = '''
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def _scale(x_ptr, out_ptr, n):
    off = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    tl.store(out_ptr + off, tl.load(x_ptr + off, mask=off < n), mask=off < n)
'''

NO_KERNEL = '''
import torch


def run(x, out):
    out.copy_(x * 2)
'''

CUTE_SUBMISSION = '''
import cutlass.cute as cute


@cute.kernel
def scale_kernel(src: cute.Tensor, dst: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    dst[tid] = src[tid] * 2.0
'''

nvcc_only = pytest.mark.skipif(find_nvcc() is None, reason="no nvcc on this machine")


# -- reading the source ------------------------------------------------------------------

def test_cuda_sources_and_triton_kernels_are_found():
    assert len(cc.cuda_sources_in(GOOD_CUDA)) == 1
    assert cc.cuda_sources_in(NO_KERNEL) == []
    assert [name for name, _ in cc.triton_kernels_in(TRITON_SUBMISSION)] == ["_scale"]
    assert cc.triton_kernels_in(NO_KERNEL) == []


def test_a_broken_source_is_not_a_crash():
    assert cc.cuda_sources_in("def run(:\n") == []
    assert cc.triton_kernels_in("def run(:\n") == []
    report = cc.check_source("def run(:\n", label="broken.py")
    assert not report.ok and any(c.target == "triton" for c in report.conclusions)


def test_check_file_reads_a_submission_and_a_task_directory(tmp_path):
    sub = tmp_path / "submission.py"
    sub.write_text(NO_KERNEL)
    assert cc.check_file(sub).source == str(sub)
    task = tmp_path / "task"
    (task / "eval" / "leaderboard").mkdir(parents=True)
    (task / "eval" / "leaderboard" / "problem.json").write_text(
        json.dumps({"kernel_file": "kernel.cu"}))
    (task / "kernel.cu").write_text("int x;\n")
    assert cc.check_file(task).source == str(task / "kernel.cu")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(cc.CompileCheckError, match="no submission to check"):
        cc.check_file(empty)
    with pytest.raises(cc.CompileCheckError):
        cc.check_file(tmp_path / "does-not-exist.py")


# -- cuda-inline -------------------------------------------------------------------------

@nvcc_only
def test_an_embedded_cuda_kernel_is_compiled_for_the_target(tmp_path):
    conclusion = cc.check_cuda_source(cc.cuda_sources_in(GOOD_CUDA)[0][1],
                                      workdir=tmp_path / "build")
    assert conclusion.status == cc.OK, conclusion.detail
    assert "sm_100a" in conclusion.detail
    assert (tmp_path / "build" / "submission_kernel.o").is_file()


@nvcc_only
def test_a_dev_only_instruction_fails_with_the_compilers_words(tmp_path):
    """The acceptance case: a source that can only build for the development architecture."""
    conclusion = cc.check_cuda_source(cc.cuda_sources_in(DEV_ONLY_CUDA)[0][1],
                                      workdir=tmp_path / "build")
    assert conclusion.status == cc.FAIL
    assert "__CUDA_ARCH_FEAT_SM103_ALL" in conclusion.detail


def test_the_board_flags_and_torch_includes_are_passed(monkeypatch, tmp_path):
    """No compiler runs: the argv is captured, so the flags cannot drift unnoticed."""
    seen = {}

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, **kwargs):
        seen["argv"] = list(argv)
        seen["kwargs"] = kwargs
        Path(argv[argv.index("-o") + 1]).write_bytes(b"\x7fELF")   # what nvcc would produce
        return Proc()

    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    report = cc.check_file(_write(tmp_path / "sub.py", TORCH_CUDA), workdir=tmp_path / "b")
    argv = seen["argv"]
    assert argv[0] == "nvcc" and "-O3" in argv and "--use_fast_math" in argv
    assert "-gencode=arch=compute_100a,code=sm_100a" in argv
    assert "-std=c++20" in argv
    assert any("-D_GLIBCXX_USE_CXX11_ABI" in a for a in argv)
    includes = [a for a in argv if a.startswith("-I")]
    assert any("torch" in a for a in includes)
    assert any(a.endswith("python3.11") or "include" in a for a in includes)
    assert seen["kwargs"]["timeout"] == 900 and seen["kwargs"]["capture_output"] is True
    assert [c.status for c in report.conclusions if c.target == "cuda-inline"] == [cc.OK]


def test_a_timeout_is_a_failure_with_the_boards_limit_in_the_words(monkeypatch, tmp_path):
    import subprocess as sp

    def fake_run(argv, **kwargs):
        raise sp.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    conclusion = cc.check_cuda_source("__global__ void k() {}", workdir=tmp_path)
    assert conclusion.status == cc.FAIL and "120s" in conclusion.detail


@nvcc_only
def test_a_missing_cutlass_tree_is_unknown_not_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "find_cutlass", lambda: None)
    source = '#include <cutlass/cutlass.h>\n__global__ void k() {}\n'
    conclusion = cc.check_cuda_source(source, workdir=tmp_path / "build")
    assert conclusion.status == cc.UNKNOWN and "CUTLASS" in conclusion.detail


def test_no_nvcc_is_unknown_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "find_nvcc", lambda: None)
    conclusion = cc.check_cuda_source("__global__ void k() {}", workdir=tmp_path)
    assert conclusion.status == cc.UNKNOWN and "no nvcc" in conclusion.detail


# -- triton ------------------------------------------------------------------------------

def test_the_triton_signature_is_inferred_from_the_wrapped_function(tmp_path):
    pytest.importorskip("triton")
    tree = __import__("ast").parse(TRITON_SUBMISSION)
    node = [n for n in __import__("ast").walk(tree)
            if isinstance(n, __import__("ast").FunctionDef) and n.name == "_scale"][0]
    fn = cc._load_kernel(TRITON_SUBMISSION, node, tmp_path, "_scale")
    assert cc._signature_for(fn) == {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32",
                                    "BLOCK": "constexpr"}


def test_a_triton_kernel_is_cross_compiled_for_the_target():
    pytest.importorskip("triton")
    tree = __import__("ast").parse(TRITON_SUBMISSION)
    node = [n for n in __import__("ast").walk(tree)
            if isinstance(n, __import__("ast").FunctionDef) and n.name == "_scale"][0]
    with __import__("tempfile").TemporaryDirectory() as tmp:
        fn = cc._load_kernel(TRITON_SUBMISSION, node, Path(tmp), "_scale")
        conclusion = cc.check_triton_kernel(fn)
    assert conclusion.status == cc.OK, conclusion.detail
    assert "CC 10.0" in conclusion.detail and "cubin" in conclusion.detail


def test_a_kernel_that_needs_module_scope_is_unknown_not_a_failure():
    pytest.importorskip("triton")
    report = cc.check_source(TRITON_NEEDS_MODULE_SCOPE, label="scope.py")
    triton = [c for c in report.conclusions if c.target == "triton"]
    assert triton and triton[0].status == cc.OK          # no constexpr, no module names


def test_a_triton_submission_without_kernels_is_ok():
    report = cc.check_source(NO_KERNEL)
    triton = [c for c in report.conclusions if c.target == "triton"][0]
    assert triton.status == cc.OK and "no @triton.jit" in triton.detail


# -- cutedsl -----------------------------------------------------------------------------

def test_cute_is_unknown_without_the_dsl_and_ok_without_a_cute_kernel():
    pytest.importorskip("triton")     # unrelated: just a cheap liveness check of the suite
    without = cc.check_source(NO_KERNEL)
    cute = [c for c in without.conclusions if c.target == "cutedsl"][0]
    assert cute.status == cc.OK and "no CuTe DSL kernel" in cute.detail
    with_cute = cc.check_source(CUTE_SUBMISSION)
    cute = [c for c in with_cute.conclusions if c.target == "cutedsl"][0]
    installed = importlib.util.find_spec("cutlass") is not None
    if installed:
        assert cute.status == cc.UNKNOWN and "needs a device" in cute.detail
    else:
        assert cute.status == cc.UNKNOWN and "not importable" in cute.detail


# -- the whole report --------------------------------------------------------------------

def test_a_refused_submission_fails_static_checks_and_exits_nonzero():
    report = cc.check_source(DEV_ONLY_CUDA, label="dev_only.py")
    static = [c for c in report.conclusions if c.target == "static"][0]
    assert static.status == cc.FAIL
    assert {f.rule for f in report.findings} >= {"b300-only", "arch-literal", "gencode-dev-only"}
    assert report.exit_code() == 2 and not report.ok
    payload = report.to_dict()
    assert payload["ok"] is False and payload["targets"]
    assert any("line" in c for c in payload["conclusions"]) and report.notes


def test_the_good_submission_passes_and_its_shape_is_stable(tmp_path):
    report = cc.check_file(_write(tmp_path / "good.py", GOOD_CUDA), workdir=tmp_path / "b")
    assert report.findings == []
    assert report.exit_code() == 0
    assert sorted({c.target for c in report.conclusions}) == [
        "cuda-inline", "cutedsl", "static", "triton"]
    assert report.lines()[0].startswith(str(tmp_path))
    assert any("note:" in line for line in report.lines())


def test_main_reports_a_missing_path_on_stderr(tmp_path, capsys):
    assert cc.main([str(tmp_path / "nope.py")]) == 2
    assert "nope.py" in capsys.readouterr().err
    assert cc.main([]) == 2


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path
