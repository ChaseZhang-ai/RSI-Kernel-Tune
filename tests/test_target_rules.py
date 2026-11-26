"""The static rules a submission must pass: one positive and one negative per rule, plus the
three normal styles that must never be flagged."""

from __future__ import annotations

import pytest

from rsi_kernel_tune.targethw import rules

# -- the normal styles: zero findings -------------------------------------------------------

DEVICE_DERIVED_GENCODE = '''
import torch
from torch.utils.cpp_extension import load_inline

CUDA = r"""
#include <torch/extension.h>
__global__ void k(const float* x, float* o) { o[threadIdx.x] = x[threadIdx.x]; }
"""

def _gencode():
    major, minor = torch.cuda.get_device_capability(0)
    suffix = "a" if (major, minor) in {(9, 0), (10, 0), (10, 3), (12, 0)} else ""
    cc = f"{major}{minor}{suffix}"
    return f"-gencode=arch=compute_{cc},code=sm_{cc}"

def _sms():
    import os
    value = os.environ.get("FK_TARGET_SMS")
    if value:
        return int(value)
    return torch.cuda.get_device_properties(0).multi_processor_count

def run(x, out):
    module = load_inline(name="k", cpp_sources="", cuda_sources=CUDA,
                         functions=["k"], extra_cuda_cflags=[_gencode(), "-O3"])
    module.k(x, out)
'''

TRITON_KERNEL = '''
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    v = tl.load(x_ptr + off, mask=mask, other=-float("inf"))
    e = tl.math.exp(v - tl.max(v, axis=0))
    tl.store(out_ptr + off, e / tl.sum(e, axis=0), mask=mask)

def run(x, out):
    softmax_kernel[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)
'''

CUTE_KERNEL = '''
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def scale_kernel(src: cute.Tensor, dst: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    n = cute.size(src, mode=[0])
    if tid < n:
        dst[tid] = src[tid] * 2.0

@cute.jit
def scale(src: cute.Tensor, dst: cute.Tensor):
    scale_kernel(src, dst).launch(grid=(1, 1, 1), block=(128, 1, 1))

def run(x, out):
    scale(from_dlpack(x), from_dlpack(out))
'''

NORMAL = {
    "device-derived gencode": DEVICE_DERIVED_GENCODE,
    "triton kernel": TRITON_KERNEL,
    "cute kernel": CUTE_KERNEL,
}


@pytest.mark.parametrize("name", sorted(NORMAL))
def test_the_normal_styles_are_not_flagged(name):
    findings = rules.check_submission(NORMAL[name], rules.language_of(NORMAL[name]))
    assert findings == [], [str(f) for f in findings]


def test_language_is_recognised():
    assert rules.language_of(DEVICE_DERIVED_GENCODE) == "cuda-inline"
    assert rules.language_of(TRITON_KERNEL) == "triton"
    assert rules.language_of(CUTE_KERNEL) == "cutedsl"


# -- arch-literal ---------------------------------------------------------------------------

def test_an_architecture_in_a_compile_option_is_a_block():
    src = ('FLAGS = ["-gencode=arch=compute_100a,code=sm_100a"]\n'
           'def run(x, out):\n    pass\n')
    found = rules.check_submission(src)
    assert [f.rule for f in found] == [rules.ARCH_LITERAL]
    assert found[0].severity == rules.BLOCK and found[0].line == 1


def test_an_architecture_in_plain_code_is_still_a_block():
    src = 'ARCH = "sm_90a"\ndef run(x, out):\n    pass\n'
    found = rules.check_submission(src)
    assert [f.rule for f in found] == [rules.ARCH_LITERAL]


def test_an_architecture_in_a_comment_or_docstring_is_fine():
    src = ('"""Graded on sm_100a; this file derives its own gencode."""\n'
           '# the board builds sm_100a, we never name it\n'
           'def run(x, out):\n    pass\n')
    assert rules.check_submission(src) == []


def test_an_architecture_in_an_embedded_cuda_comment_is_fine():
    src = ('CUDA = """\n'
           '// the board builds sm_100a; no literal here\n'
           'extern "C" __global__ void k(float* x) { x[0] = 1.0f; }\n'
           '"""\n'
           'def run(x, out):\n    pass\n')
    assert rules.check_submission(src) == []


# -- b300-only ------------------------------------------------------------------------------

@pytest.mark.parametrize("src", [
    '#ifdef __CUDA_ARCH_FEAT_SM103_ALL\n    use_it();\n#endif\n',
    'flags = ["-gencode=arch=compute_103a,code=sm_103a"]\n',
    'if cc == (10, 3):\n    fast_path()\n',
    'if major == 10 and minor == 3:\n    fast_path()\n',
])
def test_a_cc_103_only_construct_is_a_block(src):
    found = rules.check_submission(src)
    assert rules.B300_ONLY in [f.rule for f in found], [str(f) for f in found]


def test_a_capability_ladder_is_not_a_cc_103_branch():
    src = ('supported = {(9, 0), (10, 0), (10, 3), (12, 0)}\n'
           'def run(x, out):\n'
           '    major, minor = 10, 3\n'
           '    return (major, minor) in supported\n')
    found = rules.check_submission(src)
    assert [f.rule for f in found] == [], [str(f) for f in found]


# -- hardcoded-sms --------------------------------------------------------------------------

def test_an_sm_count_in_a_grid_is_a_warning():
    src = 'n_sms = 148\ngrid = (n_sms * 4, 1, 1)\ndef run(x, out):\n    pass\n'
    found = rules.check_submission(src)
    assert [f.rule for f in found] == [rules.HARDCODED_SMS]
    assert found[0].severity == rules.WARN


def test_an_sm_count_in_a_launch_is_a_warning_too():
    src = 'def _grid(n_sms=160):\n    return (n_sms * 8, 1, 1)\n'
    found = rules.check_submission(src)
    assert [f.rule for f in found] == [rules.HARDCODED_SMS]


def test_a_number_148_outside_a_grid_is_not_an_sm_count():
    src = 'tol = 148\ndef run(x, out):\n    return x + tol\n'
    assert rules.check_submission(src) == []


# -- fk-target-sms-no-fallback ---------------------------------------------------------------

def test_fk_target_sms_without_a_fallback_is_a_block():
    src = ('import os\n'
           'def run(x, out):\n'
           '    n = int(os.environ["FK_TARGET_SMS"])\n'
           '    grid = (n, 1, 1)\n')
    found = rules.check_submission(src)
    assert [f.rule for f in found] == [rules.TARGET_SMS_NO_FALLBACK]
    assert found[0].severity == rules.BLOCK


@pytest.mark.parametrize("src", [
    'import os\nsms = int(os.environ.get("FK_TARGET_SMS", "0")) or 148\n',
    'import os\nraw = os.environ.get("FK_TARGET_SMS")\nsms = int(raw) if raw else 148\n',
    'import os\nsms = 0\nif "FK_TARGET_SMS" in os.environ:\n    sms = int(os.environ["FK_TARGET_SMS"])\n',
])
def test_fk_target_sms_with_a_fallback_is_not_flagged(src):
    found = [f for f in rules.check_submission(src) if f.rule == rules.TARGET_SMS_NO_FALLBACK]
    assert found == []


# -- gencode-dev-only -----------------------------------------------------------------------

def test_a_gencode_naming_only_the_development_arch_is_a_block():
    src = 'FLAGS = "-gencode=arch=compute_103a,code=sm_103a"\ndef run(x, out):\n    pass\n'
    found = rules.check_submission(src)
    assert rules.GENCODE_DEV_ONLY in [f.rule for f in found]


def test_a_gencode_naming_both_architectures_is_not_the_gencode_rule():
    src = ('FLAGS = ["-gencode=arch=compute_100a,code=sm_100a",\n'
           '         "-gencode=arch=compute_103a,code=sm_103a"]\n'
           'def run(x, out):\n    pass\n')
    found = [f for f in rules.check_submission(src) if f.rule == rules.GENCODE_DEV_ONLY]
    assert found == []


# -- the info rules -------------------------------------------------------------------------

def test_an_fp4_mma_kind_is_an_info():
    src = 'def run(x, out):\n    out = tl.dot_scaled(x, x, kind="mxf4")\n'
    found = rules.check_submission(src)
    assert [f.rule for f in found] == [rules.FP4_MMA] and found[0].severity == rules.INFO


def test_exp_inside_a_loop_is_an_sfu_info():
    src = ('@triton.jit\n'
           'def k(x_ptr, out_ptr, n):\n'
           '    for i in range(0, n):\n'
           '        v = tl.exp(tl.load(x_ptr + i))\n'
           '        tl.store(out_ptr + i, v)\n')
    found = rules.check_submission(src)
    assert [f.rule for f in found] == [rules.SFU_LOOP] and found[0].severity == rules.INFO


def test_a_single_exp_is_not_an_sfu_kernel():
    src = ('@triton.jit\n'
           'def k(x_ptr, out_ptr):\n'
           '    tl.store(out_ptr, tl.exp(tl.load(x_ptr)))\n')
    assert rules.check_submission(src) == []


# -- the finding object -----------------------------------------------------------------------

def test_findings_carry_rule_severity_line_and_snippet():
    src = '\n\nFLAGS = ["-gencode=arch=compute_100a,code=sm_100a"]\n'
    found = rules.check_submission(src)[0]
    assert found.line == 3 and "compute_100a" in found.snippet
    assert found.to_dict() == {"rule": rules.ARCH_LITERAL, "severity": rules.BLOCK,
                              "line": 3, "snippet": found.snippet,
                              "detail": found.detail}
    assert "block" in rules.summarize([found])
    assert rules.blocks([found]) == [found]
    assert rules.warn_or_worse([found]) == [found]


def test_the_source_may_be_unparseable():
    found = rules.check_submission('def run(:\n  pass\n')
    assert isinstance(found, list)          # the rules must not raise on a broken file


# -- the credibility labels the profile derives -------------------------------------------

def test_confidence_hints_add_the_reading_labels():
    from rsi_kernel_tune import target_profile

    assert target_profile.confidence_hints("def run(x, out):\n    pass\n") == ["dev-only"]
    fp4 = target_profile.confidence_hints('out = tl.dot_scaled(a, b, kind="mxf4")\n')
    assert "fp4-bound" in fp4 and fp4[0] == "dev-only"
    sfu = target_profile.confidence_hints(
        '@triton.jit\ndef k(x_ptr, n):\n    for i in range(n):\n'
        '        a = tl.exp(tl.load(x_ptr + i))\n        b = tl.exp(a)\n')
    assert "sfu-bound" in sfu
