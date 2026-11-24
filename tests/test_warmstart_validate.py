"""``--init-kernel``'s static check (forgekernel.warmstart.validate): a pure function over
source text -- no torch, no GPU, nothing executed.

The card calls ``run`` by position with the definition's inputs and (DPS) its pre-allocated
outputs, so the checks that matter are: it parses, it has a module-level ``run``, that
function's positional arity survives the call, its imports name a kernel language this
build can build, and it does not read credentials or write outside the workspace."""

from __future__ import annotations

import ast
import os

import pytest

from forgekernel.warmstart import validate as v

DEFN = {"inputs": {"a": {"shape": ["n"], "dtype": "float32"},
                   "b": {"shape": ["n"], "dtype": "float32"}},
        "outputs": {"out": {"shape": ["n"], "dtype": "float32"}}}

HOST_KERNEL = '''import torch

def run(a, b, out):
    out.copy_(a + b)
'''


def check(src: str, **kw):
    return v.check_source(src, definition=kw.pop("definition", DEFN), **kw)


def _tree_with_write(target: str, *, pathlib: bool = False) -> ast.Module:
    """A parsed candidate whose run() writes ``target``, built as AST nodes: the path
    arrives at run time, so this file carries no path literal of its own."""
    tree = ast.parse("def run(a, b, out):\n    pass\n")
    if pathlib:
        receiver = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()),
                            args=[ast.Constant(value=target)], keywords=[])
        call = ast.Call(func=ast.Attribute(value=receiver, attr="write_text",
                                           ctx=ast.Load()),
                        args=[ast.Constant(value="x")], keywords=[])
    else:
        call = ast.Call(func=ast.Name(id="open", ctx=ast.Load()),
                        args=[ast.Constant(value=target), ast.Constant(value="w")],
                        keywords=[])
    tree.body[0].body = [ast.Expr(value=call)]
    ast.fix_missing_locations(tree)
    return tree


# -- the legal shapes ----------------------------------------------------------------

def test_run_with_inputs_and_preallocated_outputs_passes():
    r = check(HOST_KERNEL)
    assert r.ok, r.problems
    assert r.signature == ("a", "b", "out") and not r.varargs
    assert (r.language, r.method) == ("cutedsl", "pytorch")


def test_star_args_passes_and_is_noted():
    r = check("def run(*args, **kwargs):\n    pass\n")
    assert r.ok, r.problems
    assert r.varargs and any("*args" in n for n in r.notes)


def test_kwargs_beside_the_exact_parameters_passes():
    r = check("def run(a, b, out, **kw):\n    pass\n")
    assert r.ok, r.problems


def test_a_value_returning_card_expects_only_the_inputs():
    r = check("def run(a, b):\n    return a + b\n", dps=False)
    assert r.ok, r.problems
    r = check("def run(a, b, out):\n    pass\n", dps=False)
    assert not r.ok and "run takes 3 positional parameter(s)" in r.problems[0]


def test_cutlass_imports_make_it_a_cute_dsl_kernel():
    r = check("from cutlass.cute import ops\nimport torch\n\n"
              "def run(a, b, out):\n    pass\n")
    assert r.ok, r.problems
    assert (r.language, r.method) == ("cutedsl", "cutedsl")


def test_a_host_only_kernel_with_stdlib_and_numpy_is_still_a_kernel():
    r = check("import os, sys, ctypes\nimport numpy as np\nimport torch\n"
              "from torch.utils.cpp_extension import load_inline\n\n"
              "def run(a, b, out):\n    pass\n")
    assert r.ok, r.problems
    r2 = check("import os\n\ndef run(a, b, out):\n    pass\n")
    assert r2.ok, r2.problems


# -- refusals, each with a reason ----------------------------------------------------

def test_no_top_level_run_is_refused():
    r = check("import torch\n\nVALUE = 1\n")
    assert not r.ok and "no top-level def run" in r.problems[0]


def test_a_nested_run_is_refused_and_says_where_it_was():
    r = check("import torch\n\ndef factory():\n    def run(a, b, out):\n        pass\n"
              "    return run\n")
    assert not r.ok and "not at module level" in r.problems[0]


def test_arity_mismatch_lists_the_expected_call():
    r = check("import torch\n\ndef run(a, b):\n    pass\n")
    assert not r.ok
    assert "run takes 2 positional parameter(s)" in r.problems[0]
    assert "run(a, b) is called as run(a, b, out)" in r.problems[0]


def test_a_required_keyword_only_parameter_is_refused():
    r = check("def run(a, b, *, out):\n    pass\n")
    assert not r.ok and "keyword-only" in r.problems[0]


def test_a_syntax_error_is_refused_with_its_position():
    r = check("def run(a, b, out:\n    pass\n")
    assert not r.ok and "syntax error at line 1" in r.problems[0]


def test_an_async_run_is_refused():
    r = check("async def run(a, b, out):\n    pass\n")
    assert not r.ok and "async" in r.problems[0]


def test_an_unknown_import_is_refused_with_the_import_named():
    r = check("import cupy\n\ndef run(a, b, out):\n    pass\n")
    assert not r.ok and "cupy" in r.problems[0] and "triton" in r.problems[0]


def test_triton_is_inferred_and_now_accepted_by_the_build():
    src = "import triton\nimport triton.language as tl\n\ndef run(*args):\n    pass\n"
    r = check(src)
    assert (r.language, r.method) == ("triton", "triton")
    assert r.ok                             # the triton task has merged: this is a task kernel
    # a build without the language still refuses the same file, and says so
    r2 = check(src, languages=("cutedsl", "cuda"))
    assert not r2.ok and "languages" in r2.problems[0]


def test_with_no_definition_the_arity_is_not_checked_and_says_so():
    r = v.check_source("def run(a):\n    pass\n", definition=None)
    assert r.ok
    assert any("arity was not checked" in n for n in r.notes)


def test_a_plain_relative_path_is_not_outside_the_workspace():
    assert v._outside_workspace(os.path.join("sub", "scratch.bin")) is False
    assert v._outside_workspace("scratch.bin") is False


# -- the submission rules ------------------------------------------------------------

def test_reading_a_secret_from_the_environment_is_refused():
    r = check('import os\n\ndef run(a, b, out):\n    return os.environ["HF_TOKEN"]\n')
    assert not r.ok and "HF_TOKEN" in r.problems[0]
    r = check('import os\n\ndef run(a, b, out):\n    return os.getenv("OPENAI_API_KEY")\n')
    assert not r.ok and "OPENAI_API_KEY" in r.problems[0]
    r = check('import os\n\ndef run(a, b, out):\n    return os.environ.get("AWS_SECRET")\n')
    assert not r.ok and "AWS_SECRET" in r.problems[0]


def test_reading_an_ordinary_variable_is_not_a_secret():
    r = check('import os\n\ndef run(a, b, out):\n    n = os.environ.get("OMP_NUM_THREADS")\n')
    assert r.ok, r.problems


def test_an_absolute_path_is_outside_the_workspace():
    absolute = os.path.join(os.sep, "tmp", "abs-target.txt")
    assert v._outside_workspace(absolute) is True


def test_writing_outside_the_workspace_is_refused():
    absolute = os.path.join(os.sep, "tmp", "abs-target.txt")
    problems = v._forbidden_python(_tree_with_write(absolute))
    assert len(problems) == 1 and "outside the workspace" in problems[0]
    problems = v._forbidden_python(_tree_with_write(absolute, pathlib=True))
    assert len(problems) == 1 and "outside the workspace" in problems[0]


def test_a_cuda_init_kernel_is_checked_by_its_binding_and_entry():
    ok = ('#include <torch/extension.h>\n\nvoid run(torch::Tensor a, torch::Tensor b,'
          ' torch::Tensor out) {}\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n'
          '  m.def("run", &run);\n}\n')
    r = v.check_source(ok, filename="kernel.cu", definition=DEFN)
    assert r.ok, r.problems
    assert r.language == "cuda" and r.signature == ("a", "b", "out")
    r = v.check_source("int main() {}\n", filename="kernel.cu", definition=DEFN)
    assert not r.ok and any("PYBIND11_MODULE" in p for p in r.problems)
    r = v.check_source('void run(torch::Tensor a) {}\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m){}',
                       filename="kernel.cu", definition=DEFN)
    assert not r.ok and any("positional parameter" in p for p in r.problems)


def test_check_file_reports_a_missing_and_an_undecodable_file(tmp_path):
    r = v.check_file(tmp_path / "nope.py")
    assert not r.ok and "cannot be read" in r.problems[0]
    p = tmp_path / "latin.py"
    p.write_bytes(b"\xff\xfe\x00bad")
    r = v.check_file(p)
    assert not r.ok and "not UTF-8" in r.problems[0]


@pytest.mark.parametrize("src,language,method", [
    ("import triton\n", "triton", "triton"),
    ("from cutlass import cute\n", "cutedsl", "cutedsl"),
    ("import torch\n", "cutedsl", "pytorch"),
    ("import math\n", "cutedsl", "pytorch"),
])
def test_infer_language_branches(src, language, method):
    assert v.infer_language(src) == (language, method)


def test_infer_language_raises_for_an_unknown_import_and_a_syntax_error():
    with pytest.raises(v.LanguageError, match="cupy"):
        v.infer_language("import cupy\n")
    with pytest.raises(v.LanguageError, match="does not parse"):
        v.infer_language("def run(:\n")



