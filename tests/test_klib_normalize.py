"""rsi_kernel_tune.klib.normalize: the pure functions that bring a source into the one-file shape."""

from __future__ import annotations

import ast

import pytest

from rsi_kernel_tune.klib import normalize as nz


def _run(source: str) -> set[str]:
    return set(nz.top_level_functions(source))


# --- single files --------------------------------------------------------------------------


def test_a_single_file_whose_entry_has_another_name_gets_a_run_alias():
    source = "import torch\n\n\ndef gemm(a, b):\n    return a @ b\n"
    result = nz.single_python(source)
    assert result.ok and result.entry == "run" and result.language == "pytorch"
    assert "run = gemm" in result.text()
    assert result.text().startswith("import torch")          # the source is untouched otherwise
    assert "def gemm(a, b):" in result.text()


def test_a_single_file_that_already_defines_run_is_kept_verbatim():
    source = "def run(*args):\n    return args\n"
    result = nz.single_python(source)
    assert result.ok and result.text() == source and result.entry == "run"


def test_the_public_entry_is_the_last_one_and_a_private_one_is_a_fallback():
    source = ("def _helper():\n    return 1\n\n\ndef op(a):\n    return a\n\n\n"
              "def _after():\n    return 2\n")
    assert "run = op" in nz.single_python(source).text()
    assert nz.single_python(source).entry == "run"
    # no public function at all: the last definition is still an entry (loudly aliased)
    assert "run = _b" in nz.single_python("def _a():\n    pass\n\n\ndef _b():\n    pass\n").text()


def test_an_explicit_entry_must_exist():
    with pytest.raises(nz.NormalizeError):
        nz.ensure_entry("def a():\n    pass\n", "b")
    out, entry = nz.ensure_entry("def b():\n    pass\n", "b")
    assert entry == "run" and "run = b" in out


def test_a_source_with_no_top_level_function_is_refused():
    result = nz.single_python("X = 1\n")
    assert not result.ok and "no top-level function" in result.reason


def test_a_syntax_error_is_refused_not_raised_by_the_public_helper():
    result = nz.single_python("def broken(:\n")
    assert not result.ok and "not valid Python" in result.reason


# --- language detection --------------------------------------------------------------------


@pytest.mark.parametrize("source,language", [
    ("import triton\nimport triton.language as tl\n", "triton"),
    ("from triton.language import core\n", "triton"),
    ("import cutlass\nfrom cutlass import cute\n", "cutedsl"),
    ("from torch.utils.cpp_extension import load_inline\n", "cuda-inline"),
    ("import torch\n", "pytorch"),
])
def test_detect_language(source, language):
    assert nz.detect_language(source) == language


# --- renaming ------------------------------------------------------------------------------


def test_rename_symbol_touches_only_the_name_and_keeps_every_comment():
    source = ('"""Doc says run here."""\n\n# a comment about run\n\n\n'
              "def run(x):\n    return run(x - 1) if x else x\n\n\n"
              "class run_like:\n    pass\n")
    out = nz.rename_symbol(source, "run", "_klib_ref")
    assert '"""Doc says run here."""' in out            # docstrings survive
    assert "# a comment about run" in out               # comments survive
    assert "def _klib_ref(x):" in out
    assert "return _klib_ref(x - 1) if x else x" in out  # the recursive reference too
    assert "class run_like:" in out                     # a longer name is not a match
    ast.parse(out)


def test_rename_symbol_refuses_a_name_that_is_not_there():
    with pytest.raises(nz.NormalizeError):
        nz.rename_symbol("x = 1\n", "run", "_klib_ref")


# --- multi-file merge ----------------------------------------------------------------------


def test_multi_file_merge_inlines_in_dependency_order_with_prefixed_names():
    sources = {
        "kernel.py": ("import b\n\n\ndef run(x):\n    return b.foo(x) + b.CONST\n"),
        "b.py": ("CONST = 3\n\n\ndef foo(x):\n    return _scale(x) * _helper()\n\n\n"
                 "def _helper():\n    return 2\n\n\ndef _scale(x):\n    return x\n"),
    }
    result = nz.merge_python_sources(sources)
    assert result.ok, result.reason
    text = result.text()
    # dependency first, entry last
    assert text.index("inlined: b.py") < text.index("entry: kernel.py")
    # every top-level binding of the inlined module is prefixed, references included
    assert "def _klib_b__foo(" in text
    assert "def _klib_b___helper(" in text
    assert "_klib_b__CONST = 3" in text
    assert "_klib_b___scale(x) * _klib_b___helper()" in text
    # the entry's import and attribute access are rewritten
    assert "import b" not in text.split("entry: kernel.py")[1]
    assert "_klib_b__foo(x) + _klib_b__CONST" in text
    assert "run" in _run(text)
    ast.parse(text)


def test_multi_file_merge_keeps_the_import_of_a_module_it_did_not_inline():
    sources = {"kernel.py": "import math\n\n\ndef run(x):\n    return math.sqrt(x)\n",
               "helper.py": "def h():\n    return 1\n"}
    result = nz.merge_python_sources(sources, entry_path="kernel.py")
    assert result.ok, result.reason
    assert "import math" in result.text()
    assert "def _klib_helper__h(" in result.text()


def test_a_circular_import_is_noted_and_ordered_deterministically():
    sources = {"a.py": "import b\n\n\ndef fa():\n    return 1\n",
               "kernel.py": "import a\n\n\ndef run(x):\n    return a.fa()\n"}
    notes: list[str] = []
    order = nz.order_by_dependency(sources, notes=notes)
    assert order[-1] == "kernel.py" and set(order) == set(sources)
    assert not any("circular" in n for n in notes)   # only one direction here
    cyc = {"a.py": "import b\n\n\ndef fa():\n    return 1\n",
           "b.py": "import a\n\n\ndef fb():\n    return 2\n",
           "kernel.py": "import a\n\n\ndef run(x):\n    return a.fa()\n"}
    notes2: list[str] = []
    order2 = nz.order_by_dependency(cyc, notes=notes2)
    assert order2[-1] == "kernel.py"
    assert any("circular" in n for n in notes2)


def test_merging_notes_a_runtime_loader_that_the_merge_cannot_rewrite():
    sources = {
        "kernel.py": ("import importlib.util, os\n\n\ndef _load():\n"
                      "    spec = importlib.util.spec_from_file_location('m', 'm.py')\n\n\n"
                      "def run(x):\n    return x\n"),
        "m.py": "def helper():\n    return 1\n",
    }
    result = nz.merge_python_sources(sources, entry_path="kernel.py")
    assert result.ok
    assert any("run time" in n or "still reads from disk" in n for n in result.notes)


# --- reference bundles ---------------------------------------------------------------------


def test_a_bundle_parses_keeps_every_byte_and_refuses_loudly():
    files = {"dsa.cu": "// cuda\n__global__ void k() {}", "dsa.cuh": "// header\n",
             "kernel.py": "def run(q):\n    raise SystemExit\n"}
    result = nz.bundle_files(files, language="cuda-inline", reason="loads siblings",
                             header="An artifact")
    assert result.ok
    text = result.text()
    ast.parse(text)
    assert "'dsa.cu':" in text and "__global__ void k()" in text
    assert "'kernel.py':" in text
    run = next(n for n in ast.parse(text).body
               if isinstance(n, ast.FunctionDef) and n.name == "run")
    assert any(isinstance(n, ast.Raise) for n in ast.walk(run))


def test_string_literal_is_safe_for_quotes_backslashes_and_triple_quotes():
    for text in ['a """b""" c', "back\\slash\n", "'''three'''", "ends with backslash\\\\"]:
        literal = nz.string_literal(text)
        assert ast.literal_eval(literal) == (text if text.endswith("\n") else text + "\n")


# --- CUDA → inline -------------------------------------------------------------------------

_CU = '''// a kernel file
// fk-cuda-cflags: -lineinfo -Xptxas=-v

#include "common.cuh"

extern "C" __global__ void scale_kernel(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= 0.5f;
}

torch::Tensor run(torch::Tensor x) {
  scale_kernel<<<1, 32>>>(x.data_ptr<float>(), (int)x.numel());
  return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
'''

_HEADER = "#pragma once\nstruct Helper { int v; };\n"


def test_cuda_to_inline_embeds_builds_once_and_derives_the_arch_at_run_time():
    result = nz.cuda_to_inline({"kernel.cu": _CU, "common.cuh": _HEADER}, n_outputs=1,
                               extra_ldflags=["-lcudart"])
    assert result.ok, result.reason
    text = result.text()
    ast.parse(text)
    assert "load_inline(" in text and "functions=None" in text
    assert "get_device_capability" in text
    assert "inlined from common.cuh" in text and "struct Helper" in text
    assert "-lineinfo" in text and "-Xptxas=-v" in text
    assert "-lcudart" in text
    assert "_N_OUTPUTS = 1" in text and "outs[0].copy_(result)" in text
    # no pinned architecture anywhere in the generated file
    assert "compute_100a" not in text and "sm_100a" not in text
    assert not nz.static_rules_check(nz._wrapper_only(text))


def test_cuda_to_inline_drops_a_pinned_codegen_target():
    pinned = _CU.replace("// fk-cuda-cflags: -lineinfo -Xptxas=-v",
                         "// fk-cuda-cflags: -lineinfo -gencode=arch=compute_100a,code=sm_100a")
    result = nz.cuda_to_inline({"kernel.cu": pinned})
    assert result.ok, result.reason
    text = result.text()
    # the build flags and the wrapper carry no pinned target; the pinned comment survives only
    # inside the embedded original source (data, never a flag)
    flags_line = next(line for line in text.splitlines() if line.startswith("_CUDA_FLAGS"))
    assert "-gencode" not in flags_line and "-arch" not in flags_line
    assert "compute_100a" not in nz._wrapper_only(text)
    assert any("dropped pinned code-generation flags" in n for n in result.notes)
    assert "get_device_capability" in text


def test_cuda_to_inline_requires_a_binding():
    result = nz.cuda_to_inline({"kernel.cu": "__global__ void k() {}\n"})
    assert not result.ok and "PYBIND11_MODULE" in result.reason


def test_cuda_to_inline_merges_translation_units_and_notes_the_collision_risk():
    sources = {
        "a.cu": _CU.replace("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"run\", &run); }",
                            "void a_helper() {}")
                    .replace("torch::Tensor run(torch::Tensor x)",
                             "torch::Tensor a_run(torch::Tensor x)"),
        "main.cu": ('extern "C" void a_helper();\n'
                    "torch::Tensor run(torch::Tensor x) { return x; }\n"
                    'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }\n'),
    }
    result = nz.cuda_to_inline(sources)
    assert result.ok, result.reason
    text = result.text()
    assert "source: a.cu" in text and "source: main.cu" in text
    assert any("2 translation units merged" in n for n in result.notes)


# --- the static target rules ---------------------------------------------------------------


def test_static_rules_flag_a_literal_arch_and_a_literal_sm_count():
    bad = "ARCH = 'sm_100a'\nSM_COUNT = 148\n"
    problems = nz.static_rules_check(bad, require_arch_provenance=False)
    assert any("literal" in p for p in problems)
    assert any("SM count" in p for p in problems)


def test_static_rules_require_the_arch_provenance_only_when_asked():
    text = ("cc = torch.cuda.get_device_capability()\n"
            "n = torch.cuda.get_device_properties(0).multi_processor_count\n")
    assert nz.static_rules_check(text) == []
    assert nz.static_rules_check("x = 1\n") == [
        "the code-generation target is not derived from the running device "
        "(no get_device_capability)"]
    assert nz.static_rules_check("x = 1\n", require_arch_provenance=False) == []


def test_compiles_cuda_is_true_only_for_a_runtime_build():
    assert nz.compiles_cuda("from torch.utils.cpp_extension import load_inline\n")
    assert nz.compiles_cuda("x = 'load_inline'\n")
    assert not nz.compiles_cuda("import torch\n")


# --- small helpers -------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("/* SPDX-License-Identifier: BSD-3-Clause */\n", "BSD-3-Clause"),
    ("// Licensed under the Apache License, Version 2.0 (the \"License\");\n", "Apache-2.0"),
    ("# Permission is hereby granted, free of charge, to any person\n", "MIT"),
    ("/* Redistribution and use in source and binary forms, with or without */\n", "BSD-3-Clause"),
    ("#!/usr/bin/env python\nprint('hi')\n", ""),
])
def test_license_of_text_reads_only_what_the_file_says(text, expected):
    assert nz.license_of_text(text) == expected


def test_trap_line_quotes_the_variant_header():
    header = ("# Variant: x\n# Measured: 1.2x\n# !!! MEASURED DEAD-END — correct but slow.\n"
              "# more text\n")
    line = nz.trap_line(header)
    assert line.startswith("!!! MEASURED DEAD-END")
    assert nz.trap_line("# nothing to see\n") == ""


def test_function_source_extracts_a_helper_for_testing():
    source = ("def a():\n    return 1\n\n\ndef _klib_call_with_flags(torch, fn, args):\n"
              "    return fn(*args)\n")
    segment = nz.function_source(source, "_klib_call_with_flags")
    assert segment is not None and segment.startswith("def _klib_call_with_flags")
    assert nz.function_source(source, "missing") is None
