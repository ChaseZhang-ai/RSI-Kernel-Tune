"""Regression tests for the audit-corps fix wave 3 (the guard bypass surface and the
kernel loading path): tree-root shadow modules, the correctness monitor's hook
precedence on PYTHONPATH, the multi-name and dynamic import scan, the C++ torch API
spellings, and the kernel directory's position on sys.path."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_harness_guard import _backend, _card, _commit, _git, _tree  # noqa: E402

from peregrine_kernel.loop import cli, goal_ref as gr, harness_guard  # noqa: E402


# --- BUG-SEED-07: tree-root shadow modules are harness --------------------------------

@pytest.mark.parametrize("path", ["sitecustomize.py", "usercustomize.py", "json.py",
                                  "os.py", "time.py", "subprocess.py", "random.py",
                                  "torch.py", "numpy.py", "triton.py"])
def test_root_shadow_modules_are_harness(path):
    """A root file named like a module Python or the judge imports is imported INSTEAD
    of that module by every process that reaches the tree root first on sys.path."""
    assert harness_guard.is_harness_path(path, "kernel.py") is True


@pytest.mark.parametrize("path", ["utils.py", "helpers.py", "evaluation.py", "my_ops.py",
                                  "kernel.py", "kern/json.py", "json.py.bak",
                                  "notes/json.py", "window.toml", "README.md"])
def test_the_students_own_root_files_are_not_harness(path):
    """The kernel may import root siblings by name: only the names Python itself or the
    judge imports are harness, and only at the tree root."""
    assert harness_guard.is_harness_path(path, "kernel.py") is False


def test_the_kernel_file_is_never_a_shadow_even_when_its_name_is():
    assert not harness_guard.is_harness_path("json.py", "json.py")


def test_tree_harness_files_lists_root_shadow_modules(tmp_path):
    card, ws = _card(tmp_path)
    (ws / "sitecustomize.py").write_text("STUDENT\n")
    (ws / "json.py").write_text("STUDENT\n")
    head = _commit(ws, "shadow")
    assert harness_guard.tree_harness_files(ws, head, "kernel.py") is not None
    tree = harness_guard.tree_harness_files(ws, head, "kernel.py")
    assert "sitecustomize.py" in tree and "json.py" in tree
    assert "kernel.py" not in tree                     # the kernel file is not harness


def test_a_delivery_adding_root_shadows_is_caught_and_dropped_from_jobs(tmp_path):
    """The delivery guard and the job overlay must see the shadow modules the same way
    they see an edited eval/ -- and leave the student's own root files alone."""
    card, ws = _card(tmp_path)
    be = _backend(ws, card)
    base = be.head()
    (ws / "sitecustomize.py").write_text("import os; os.environ['STUDENT'] = '1'\n")
    (ws / "json.py").write_text("STUDENT SHADOW\n")
    (ws / "utils.py").write_text("# the student's own root sibling\n")
    (ws / "kern").mkdir()
    (ws / "kern" / "inner.py").write_text("# the student's own nested module\n")
    (ws / "kernel.py").write_text("def run(): return 1\n")
    head = _commit(ws, "shadow")
    assert be.harness_changes(base, head) == ["json.py", "sitecustomize.py"]
    job = be._with_job_env(head)
    tree = _tree(ws, job)
    assert "json.py" not in tree and "sitecustomize.py" not in tree
    assert "kernel.py" in tree and "utils.py" in tree and "kern/inner.py" in tree
    assert _git(ws, "rev-parse", "HEAD") == head        # the lineage itself is untouched


def test_untracked_root_shadows_show_as_working_tree_drift(tmp_path):
    card, ws = _card(tmp_path)
    be = _backend(ws, card)
    anchor = be.head()
    (ws / "sitecustomize.py").write_text("STUDENT\n")
    (ws / "json.py").write_text("STUDENT\n")
    assert be.harness_drift(anchor) == ["json.py", "sitecustomize.py"]


def test_preflight_refuses_a_fresh_start_over_root_shadows(tmp_path, capsys):
    card, ws = _card(tmp_path)
    (ws / "sitecustomize.py").write_text("STUDENT\n")
    (ws / "json.py").write_text("STUDENT\n")
    _commit(ws, "shadow")
    be = _backend(ws)
    rc = cli._harness_guard_preflight(card, ws, "kernel.py", be, resuming=False,
                                      dry_run=False)
    assert rc == cli.EXIT_HARNESS_DRIFT
    assert "-sitecustomize.py" in capsys.readouterr().err


def test_preflight_accepts_a_workspace_with_harmless_root_files(tmp_path, capsys):
    card, ws = _card(tmp_path)
    (ws / "utils.py").write_text("# the student's own root sibling\n")
    _commit(ws, "sibling")
    be = _backend(ws)
    assert cli._harness_guard_preflight(card, ws, "kernel.py", be, resuming=False,
                                        dry_run=False) == 0
    assert "identical to the workspace" in capsys.readouterr().out


# --- BUG-T2C-01: the monitor hook stays ahead of the workspace on PYTHONPATH ----------

def _rendered_gpu_entry(tmp_path):
    from peregrine_kernel.cardkit import render_gpu_entry
    script = render_gpu_entry(name="t", suite="unit", arch="90")
    (tmp_path / "gpu_entry.sh").write_text(script)
    export_line = next(l for l in script.splitlines() if l.startswith("export PYTHONPATH="))
    return export_line


def test_gpu_entry_keeps_the_hook_directory_ahead_of_the_workspace(tmp_path):
    export_line = _rendered_gpu_entry(tmp_path)
    ws, hook, other = tmp_path / "ws", tmp_path / "hook", tmp_path / "other"
    out = subprocess.run(
        ["bash", "-c", f'WORKSPACE={ws}; PK_MON_HOOK_DIR={hook}; PYTHONPATH={other}; '
                       f'{export_line}; printf \'%s\\n\' "$PYTHONPATH"'],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    order = out.stdout.strip().split(":")
    assert order.index(str(hook)) < order.index(str(ws))


def test_a_tree_root_sitecustomize_does_not_take_the_hooks_place(tmp_path):
    """The sitecustomize import goes to the first sys.path directory that has one: with
    the entry's export re-ordering PYTHONPATH, that must stay the hook, not a shadow
    the student left at the tree root."""
    export_line = _rendered_gpu_entry(tmp_path)
    ws, hook = tmp_path / "ws", tmp_path / "hook"
    (ws / "eval").mkdir(parents=True)
    hook.mkdir()
    (ws / "sitecustomize.py").write_text(
        "import os\nos.environ['STUDENT_SHADOW_RAN'] = '1'\n")
    (hook / "sitecustomize.py").write_text(
        "import os\nos.environ['PK_HOOK_RAN'] = '1'\n")
    (ws / "eval" / "probe.py").write_text(
        "import os\n"
        "print('HOOK=' + os.environ.get('PK_HOOK_RAN', 'unset'),\n"
        "      'SHADOW=' + os.environ.get('STUDENT_SHADOW_RAN', 'unset'))\n")
    # PYTHONPATH is what .pk_monitor/run.py hands the entry: the hook directory first
    out = subprocess.run(
        ["bash", "-c", f'WORKSPACE={ws}; PK_MON_HOOK_DIR={hook}; PYTHONPATH={hook}; '
                       f'{export_line}; cd {ws} && python3 eval/probe.py'],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "HOOK=1" in out.stdout, out.stdout
    assert "SHADOW=unset" in out.stdout, \
        "the student's tree-root sitecustomize ran instead of the hook"


def test_the_monitor_runner_hands_the_hook_dir_to_the_entry(tmp_path, monkeypatch):
    import importlib.util
    from peregrine_kernel import monitor
    run_path = Path(monitor.__file__).resolve().parent / "runtime" / "run.py"
    spec = importlib.util.spec_from_file_location("_t2c01_pk_mon_run", run_path)
    mon_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mon_run)
    captured = {}

    class FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return FakeCompleted()

    monkeypatch.setattr(mon_run.subprocess, "run", fake_run)
    mon_run._run_task("gpu_entry.sh", "correctness", "base", str(tmp_path / "out"), 120, None)
    assert captured["env"]["PK_MON_HOOK_DIR"] == mon_run.HOOK_DIR
    assert captured["env"]["PYTHONPATH"].split(":")[0] == mon_run.HOOK_DIR


# --- BUG-SEED-08: the import scan sees every name and every spelling ------------------

def test_a_comma_import_list_is_caught():
    src = "import os, flashinfer\ndef run(a, b, out):\n    out.copy_(a)\n"
    assert gr.library_findings(src, "kernel.py") == ["imports the kernel library flashinfer"]


def test_a_parenthesised_import_is_caught():
    src = ("import (\n    os,\n    flashinfer,\n)\n"
           "def run(a, b, out):\n    out.copy_(a)\n")
    assert gr.library_findings(src, "kernel.py") == ["imports the kernel library flashinfer"]


def test_dynamic_imports_are_caught():
    importlib_call = ('import importlib\n'
                      'def run(a, b, out):\n'
                      '    m = importlib.import_module("flashinfer")\n')
    assert "dynamically imports the kernel library flashinfer" in \
        gr.library_findings(importlib_call, "kernel.py")
    dunder = 'def run(a, b, out):\n    m = __import__(\'vllm\')\n'
    assert "dynamically imports the kernel library vllm" in \
        gr.library_findings(dunder, "kernel.py")


def test_ordinary_imports_are_not_flagged():
    src = ("import os, json\n"
           "import torch\n"
           "from torch import nn\n"
           "from torch.nn import functional\n"
           "import flashinfer_utils\n"                  # a different module...
           "from mypkg import flashinfer_thing\n"       # ...and a different name
           "import (os, json)\n"
           "def run(a, b, out):\n    out.copy_(a)\n"
           "# from flashinfer import x\n"               # a comment
           '"""from flashinfer import x"""\n')          # a docstring
    assert gr.library_findings(src, "kernel.py") == []


def test_the_import_scan_still_names_the_direct_form():
    assert gr.library_findings("import flashinfer\n", "kernel.py") == \
        ["imports the kernel library flashinfer"]


# --- BUG-C3-01: the torch C++ API spellings -------------------------------------------

@pytest.mark.parametrize("call, name", [
    ("torch::matmul(x, x)", "matmul"),
    ("at::mm(x, y)", "mm"),
    ("torch::nn::functional::linear(x, y)", "linear"),
    ("at::softmax(w, -1)", "softmax"),
    ("c10::bmm(a, b)", "bmm"),
    ("torch::ops::my_gemm(x, y)", None),        # a registered op, named as itself
])
def test_the_torch_cpp_api_spellings_are_caught(call, name):
    src = f"#include <torch/extension.h>\nvoid run() {{ {call}; }}\n"
    found = gr.library_findings(src, "kernel.cu")
    assert found, call
    if name is not None:
        assert any(f"torch C++ API ({name})" in f for f in found), found


def test_the_cpp_kernel_importing_cublas_is_still_caught():
    src = ("#include <torch/extension.h>\n"
           "void run(at::Tensor x) {\n"
           "    at::Tensor y = torch::matmul(x, x);  // the whole computation\n"
           "}\n")
    assert any("torch C++ API (matmul)" in f for f in gr.library_findings(src, "kernel.cu"))


def test_ordinary_cuda_sources_are_not_flagged():
    src = ("#include <torch/extension.h>\n"
           "__global__ void k(float* x) { x[threadIdx.x] = x[threadIdx.x] * 2.0f; }\n"
           "void run() {\n"
           "    sum_mm(x, n);                                  // the student's own\n"
           "    auto s = at::cuda::getCurrentCUDAStream();     // plumbing\n"
           "    c10::cuda::CUDACachingAllocator::raw_alloc(n); // plumbing\n"
           "    x[torch_i::2];\n"
           "}\n")
    assert gr.library_findings(src, "kernel.cu") == []


# --- BUG-SEED-09: the kernel directory never precedes the real modules on sys.path ----

_BENCHKIT = None


def _benchkit():
    global _BENCHKIT
    if _BENCHKIT is None:
        import importlib.util
        from peregrine_kernel import monitor
        path = (Path(monitor.__file__).resolve().parents[1] / "card_template" / "eval"
                / "benchkit.py")
        spec = importlib.util.spec_from_file_location("benchkit_corps_fix", path)
        _BENCHKIT = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_BENCHKIT)
    return _BENCHKIT


def _a_stdlib_name_not_yet_imported():
    import importlib.util
    for name in ("zoneinfo", "wave", "chunk", "netrc", "statistics", "filecmp"):
        if name in sys.modules:
            continue
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except (ImportError, ValueError):
            continue
    pytest.skip("no unimported stdlib module name left to probe with")


def test_load_entry_appends_the_kernel_dir_so_siblings_cannot_shadow(tmp_path):
    """A sibling named like a module must stay importable for the kernel WITHOUT
    shadowing the standard library for the call-time imports that follow the load."""
    bk = _benchkit()
    kern = tmp_path / "kern"
    kern.mkdir()
    shadowed = _a_stdlib_name_not_yet_imported()
    (kern / f"{shadowed}.py").write_text(
        "raise RuntimeError('the kernel directory shadowed the standard library')\n")
    (kern / "kern_helper.py").write_text("SIBLING_VALUE = 21\n")
    (kern / "kernel.py").write_text(
        "import kern_helper\n"
        "def run():\n"
        f"    import {shadowed}\n"
        "    return kern_helper.SIBLING_VALUE * 2\n")
    saved_path = list(sys.path)
    try:
        fn = bk.load_entry(kern / "kernel.py", "run")
        kdir = str(kern)
        assert sys.path[-1] == kdir      # appended, not prepended
        assert fn() == 42          # the sibling import still works, the shadow file never ran
        module = sys.modules[shadowed]
        assert not module.__file__.startswith(kdir)
    finally:
        sys.path[:] = saved_path
