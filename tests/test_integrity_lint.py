"""tests for forgekernel.integrity.lint: every rule a positive and a negative.

The rules are the board's Prohibited Behavior list in source form (docs/integrity.md), and
they are worth only as much as their precision: a rule that fires on an ordinary kernel is
one the team learns to ignore.  So every case here comes in pairs, and the negative corpus
is the repo's own kernels/ tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from forgekernel.integrity import lint as L   # noqa: E402

CLEAN = '''import torch


def run(x, y, out):
    out.copy_(x * y)
'''


def _rules(src: str):
    return [(f.rule, f.severity, f.line) for f in L.lint_source(src, "t.py")]


def _has(src: str, rule: str, severity: str) -> bool:
    return any(r == rule and s == severity for r, s, _ in _rules(src))


pytestmark = pytest.mark.skipif(not L.MAX_SUBMISSION_BYTES, reason="lint always present")


# --- block rules ------------------------------------------------------------------------

def test_env_assign_is_a_block_and_reading_it_is_not() -> None:
    assert _has("import os\nos.environ['TF32'] = '0'\ndef run(x, out):\n    out.copy_(x)\n",
                L.ENV_ASSIGN, L.BLOCK)
    assert _has("import os\nos.environ.update({'A': '1'})\ndef run(x, out):\n    out.copy_(x)\n",
                L.ENV_ASSIGN, L.BLOCK)
    assert _has("import os\nos.putenv('A', '1')\ndef run(x, out):\n    out.copy_(x)\n",
                L.ENV_ASSIGN, L.BLOCK)
    # a read is an info, never a block (FK_TARGET_SMS is read by every submission)
    assert not _has("import os\nS = os.environ.get('FK_TARGET_SMS')\n"
                    "def run(x, out):\n    out.copy_(x)\n", L.ENV_ASSIGN, L.BLOCK)


def test_torch_flag_assignment_needs_its_restore() -> None:
    once = ("import torch\ndef run(x, out):\n"
            "    torch.backends.cuda.matmul.allow_tf32 = True\n    out.copy_(x)\n")
    assert _has(once, L.FLAG_ASSIGN, L.BLOCK)
    twice = ("import torch\ndef run(x, out):\n"
             "    torch.backends.cuda.matmul.allow_tf32 = True\n    try:\n"
             "        out.copy_(x)\n    finally:\n"
             "        torch.backends.cuda.matmul.allow_tf32 = False\n")
    rules = dict((r, s) for r, s, _ in _rules(twice))
    assert L.FLAG_ASSIGN not in rules and rules.get(L.FLAG_RESTORED) == L.INFO
    at_import = ("import torch\ntorch.backends.cudnn.benchmark = True\n"
                 "def run(x, out):\n    out.copy_(x)\n")
    assert _has(at_import, L.FLAG_ASSIGN, L.BLOCK)
    prec = ("import torch\ndef run(x, out):\n"
            "    torch.set_float32_matmul_precision('high')\n    out.copy_(x)\n")
    assert _has(prec, L.FLAG_ASSIGN, L.BLOCK)


def test_patching_a_timing_function_is_a_block() -> None:
    for src in (
        "import torch\ntorch.cuda.Event.elapsed_time = lambda s, o: 0.0\n"
        "def run(x, out):\n    out.copy_(x)\n",
        "import time\ntime.perf_counter = lambda: 0.0\ndef run(x, out):\n    out.copy_(x)\n",
        "import torch\nsetattr(torch.cuda.Event, 'elapsed_time', lambda s, o: 0.0)\n"
        "def run(x, out):\n    out.copy_(x)\n",
        "import torch.cuda as tc\ntc.Event.elapsed_time = None\ndef run(x, out):\n"
        "    out.copy_(x)\n",
    ):
        assert _has(src, L.TIMING_PATCH, L.BLOCK), src
    # rebinding a from-imported name patches nothing
    assert not _has("from time import perf_counter\nperf_counter = 1\n"
                    "def run(x, out):\n    out.copy_(x)\n", L.TIMING_PATCH, L.BLOCK)


def test_monkeypatch_on_an_imported_module_and_not_on_a_local_name() -> None:
    assert _has("import numpy\nnumpy.pi = 3\ndef run(x, out):\n    out.copy_(x)\n",
                L.MONKEYPATCH, L.BLOCK)
    assert _has("import sys\nsys.modules['x'] = None\ndef run(x, out):\n    out.copy_(x)\n",
                L.MONKEYPATCH, L.BLOCK)
    # `from m import n` then `n = x` is ordinary code (the kernels/ corpus does it)
    assert not _has("from cutlass._mlir import extras as _extras\n_T = _extras.types\n"
                    "def run(x, out):\n    out.copy_(x)\n", L.MONKEYPATCH, L.BLOCK)
    assert not _has("from cutlass.cutlass_dsl import T\ndef run(q, out):\n"
                    "    _, T, H = q.shape\n    out.copy_(q)\n", L.MONKEYPATCH, L.BLOCK)
    # an instance attribute is not a patch
    assert not _has("class C:\n    pass\nc = C()\nc.BLOCK = 3\n"
                    "def run(x, out):\n    out.copy_(x)\n", L.MONKEYPATCH, L.BLOCK)


def test_threads_and_processes_are_blocks() -> None:
    for src in (
        "import threading\ndef run(x, out):\n"
        "    threading.Thread(target=lambda: None).start()\n    out.copy_(x)\n",
        "from concurrent.futures import ThreadPoolExecutor\ndef run(x, out):\n"
        "    with ThreadPoolExecutor(2) as ex:\n        ex.submit(lambda: None)\n"
        "    out.copy_(x)\n",
        "import multiprocessing\ndef run(x, out):\n"
        "    multiprocessing.Process(target=lambda: None).start()\n    out.copy_(x)\n",
        "import os\ndef run(x, out):\n    os.fork()\n    out.copy_(x)\n",
        "import subprocess\ndef run(x, out):\n    subprocess.Popen(['true'])\n"
        "    out.copy_(x)\n",
    ):
        assert _has(src, L.THREADS, L.BLOCK), src
    assert not _has("import threading\nLOCK = threading.Lock()\n"
                    "def run(x, out):\n    out.copy_(x)\n", L.THREADS, L.BLOCK)


def test_pointer_keyed_and_call_count_caches_are_blocks() -> None:
    assert _has("def run(x, out):\n    cache = {}\n    out.copy_(cache[x.data_ptr()])\n",
                L.POINTER_CACHE, L.BLOCK)
    assert _has("def run(x, out):\n    cache = {}\n    out.copy_(cache[id(x)])\n",
                L.POINTER_CACHE, L.BLOCK)
    # data_ptr() used for alignment is not a cache
    assert not _has("def run(x, out):\n    assert x.data_ptr() % 16 == 0\n    out.copy_(x)\n",
                    L.POINTER_CACHE, L.BLOCK)
    counter = ("N = 0\ndef run(x, out):\n    global N\n    N += 1\n    if N > 1:\n"
               "        out.zero_()\n    else:\n        out.copy_(x)\n")
    assert _has(counter, L.CALL_COUNT, L.BLOCK)
    # a module-level table that is only read is not a call counter
    assert not _has("CACHE = {}\ndef run(x, out):\n    out.copy_(x)\n",
                    L.CALL_COUNT, L.BLOCK)


def test_eval_paths_absolute_paths_and_traversal_are_blocks() -> None:
    assert _has("def run(x, out):\n    open('eval/leaderboard/problem.json').read()\n"
                "    out.copy_(x)\n", L.EVAL_PATHS, L.BLOCK)
    assert _has("P = 'ref_bin/ref.bin'\ndef run(x, out):\n    out.copy_(x)\n",
                L.EVAL_PATHS, L.BLOCK)
    assert _has("P = '/home/user/secret.bin'\ndef run(x, out):\n    out.copy_(x)\n",
                L.ABSOLUTE_PATH, L.BLOCK)
    assert _has("def run(x, out):\n    open('../outside').read()\n    out.copy_(x)\n",
                L.PATH_TRAVERSAL, L.BLOCK)
    # an inlined CUDA build may point at the image's toolchain, and a docstring may say
    # anything (it documents, it does not read)
    assert not _has('"""reads eval/leaderboard/problem.json"""\nimport torch\n'
                    "def run(x, out):\n    out.copy_(x)\n", L.EVAL_PATHS, L.BLOCK)
    assert not _has("CFLAGS = ['-I/usr/local/cutlass/include']\n"
                    "def run(x, out):\n    out.copy_(x)\n", L.ABSOLUTE_PATH, L.BLOCK)


def test_credentials_are_blocks() -> None:
    assert _has("import os\nT = os.environ.get('SOL_EXECBENCH_TOKEN')\n"
                "def run(x, out):\n    out.copy_(x)\n", L.SECRETS, L.BLOCK)
    assert _has("T = open('/home/u/.ssh/id_rsa').read()\ndef run(x, out):\n    out.copy_(x)\n",
                L.SECRETS, L.BLOCK)
    # an attention kernel's `token` tensor is not a credential
    assert not _has("import torch\ndef run(token, out):\n    out.copy_(token)\n",
                    L.SECRETS, L.BLOCK)


def test_a_lazy_proxy_returned_from_run_is_a_block() -> None:
    proxy = ("import torch\nclass Wrap:\n    def __init__(self, t):\n        self.t = t\n"
             "    def __torch_function__(self, *a, **k):\n        return self.t\n"
             "def run(x, out):\n    return Wrap(out)\n")
    assert _has(proxy, L.PROXY_RETURN, L.BLOCK)
    # a Tensor subclass returned from run is what the board takes
    tensor_sub = ("import torch\nclass MyTensor(torch.Tensor):\n    pass\n"
                  "def run(x, out):\n    return out\n")
    assert not _has(tensor_sub, L.PROXY_RETURN, L.BLOCK)


# --- warn / info rules ------------------------------------------------------------------

def test_graphs_streams_events_and_clocks_warn() -> None:
    graph = "import torch\ndef run(x, out):\n    g = torch.cuda.CUDAGraph()\n    out.copy_(x)\n"
    assert _has(graph, L.CUDA_GRAPH, L.WARN)
    stream = "import torch\ndef run(x, out):\n    s = torch.cuda.Stream()\n    out.copy_(x)\n"
    assert _has(stream, L.NON_DEFAULT_STREAM, L.WARN)
    event = "import torch\ndef run(x, out):\n    e = torch.cuda.Event()\n    out.copy_(x)\n"
    assert _has(event, L.EVENT_CONSTRUCT, L.WARN)
    clock = "import time\ndef run(x, out):\n    t = time.perf_counter()\n    out.copy_(x)\n"
    assert _has(clock, L.WALL_CLOCK, L.WARN)
    ctypes = "import ctypes\ndef run(x, out):\n    out.copy_(x)\n"
    assert _has(ctypes, L.CTYPES, L.WARN)
    # reading the current stream to hand its handle to a driver call is not a hidden stream
    assert not _has("import torch\ndef run(x, out):\n"
                    "    s = torch.cuda.current_stream().cuda_stream\n    out.copy_(x)\n",
                    L.NON_DEFAULT_STREAM, L.WARN)


def test_inline_cuda_and_torch_compile_are_info() -> None:
    inline = ("import torch\nfrom torch.utils.cpp_extension import load_inline\n"
              "SRC = '__global__ void k() {}'\ndef run(x, out):\n"
              "    mod = load_inline(name='k', cpp_sources='', cuda_sources=SRC, "
              "functions=['k'])\n    out.copy_(x)\n")
    rules = dict((r, s) for r, s, _ in _rules(inline))
    assert rules.get(L.INLINE_CUDA) == L.INFO and L.THREADS not in rules
    compiled = ("import torch\ndef run(x, out):\n    f = torch.compile(lambda t: t * 2)\n"
                "    out.copy_(f(x))\n")
    assert _has(compiled, L.TORCH_COMPILE, L.INFO)


# --- shape rules ------------------------------------------------------------------------

def test_shape_rules_only_apply_to_a_submission() -> None:
    no_run = "def main(x, out):\n    out.copy_(x)\n"
    assert _has(no_run, L.NO_RUN, L.BLOCK)
    assert L.lint_source(no_run, "kernel.py", submission=False) == []
    bad = "def run(:\n"
    assert _has(bad, L.SYNTAX_ERROR, L.BLOCK)


def test_the_size_limit_is_the_boards(monkeypatch) -> None:
    assert L.MAX_SUBMISSION_BYTES == 10 * 1024 * 1024
    monkeypatch.setattr(L, "MAX_SUBMISSION_BYTES", 40)
    assert _has(CLEAN, L.TOO_LARGE, L.BLOCK)
    monkeypatch.setattr(L, "MAX_SUBMISSION_BYTES", 10 * 1024 * 1024)
    assert not _has(CLEAN, L.TOO_LARGE, L.BLOCK)


def test_findings_are_unique_per_rule_and_line_and_sorted() -> None:
    src = ("import os\nos.environ['A'] = '1'\nos.environ['B'] = '2'\n"
           "def run(x, out):\n    out.copy_(x)\n")
    findings = L.lint_source(src, "t.py")
    keys = [(f.rule, f.line) for f in findings]
    assert keys == sorted(keys) and len(keys) == len(set(keys))
    assert L.summarize(findings).startswith("2 block")
    assert [f.line for f in L.blocks(findings)] == [2, 3]


# --- the negatives that matter: ordinary kernels ----------------------------------------

TRITON = '''import torch
import triton
import triton.language as tl


@triton.jit
def _copy(x, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + i, tl.load(x + i, mask=i < n), mask=i < n)


def run(x, out):
    n = x.numel()
    _copy[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
'''

CUTEDSL = '''import torch
import cutlass
import cutlass.cute as cute


class Kernel:
    @cute.kernel
    def kernel(self, gx):
        pass


@cute.jit
def run(x, out):
    cutlass.cuda.initialize_cuda_context()
    return out
'''

CONSTANTS = '''import torch

CONFIG = {"BLOCK": 128, "STAGES": 3, "dtype": torch.bfloat16}


def run(x, out):
    out.copy_(x)
'''


@pytest.mark.parametrize("src", [CLEAN, TRITON, CUTEDSL, CONSTANTS])
def test_normal_kernels_have_no_findings(src: str) -> None:
    assert L.lint_source(src, "kernel.py") == []


def test_the_repo_corpus_has_no_false_blocks_on_ordinary_code() -> None:
    """The calibration that matters: the repo's own kernels must not read as cheats.

    ``kernels/`` are ForgeKernel ``kernel.py`` files (``submission=False``: their entry
    point and tree are the loop's).  The two blocks that remain are real -- an inline-CUDA
    kernel that writes ``os.environ`` -- and are recorded in the report, not asserted away.
    """
    findings = L.lint_paths([REPO_ROOT / "kernels"])
    assert findings, "the corpus should produce at least the known findings"
    rules = {(f.rule, f.severity) for f in findings}
    assert (L.MONKEYPATCH, L.BLOCK) not in rules       # from-import rebinds are not patches
    assert (L.SYNTAX_ERROR, L.BLOCK) not in rules      # .cu files are not parsed as Python
    assert (L.NO_RUN, L.BLOCK) not in rules            # submission=False
    assert {r for r, s in rules if s == L.BLOCK} <= {L.ENV_ASSIGN}
