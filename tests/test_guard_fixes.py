"""Regression tests for the audit-corps R2 guard + benchkit fix cluster.

One section per ledger bug:

* BUG-SEED-07: a tree-root python module a harness process executes without the tree
  naming it (sitecustomize.py, a stdlib shadow like json.py) is harness -- the job
  overlay drops it, the delivery diff names it, the hygiene drift sees it
  (loop/harness_guard.py).
* BUG-SEED-08: the goal-ref library scan reads every name of an import statement and
  the string literals of importlib.import_module / __import__ (loop/goal_ref.py).
* BUG-C3-01: the torch C++ call forms (torch:: / at:: / c10::) are library findings
  on .cu / .cpp sources (loop/goal_ref.py).
* BUG-SEED-09: load_entry APPENDS the kernel's directory to sys.path, so the stdlib
  and the already-loaded reference are not shadowed (card_template/eval/benchkit.py).
* BUG-C5-02: a Python kernel that dies at import is a structured IMPORT_ERROR verdict,
  not a bare crash of the bench process (card_template/eval/benchkit.py).
* BUG-C5-03: an empty input_groups workload is an INPUT_ERROR, never a fake PASS, and
  the NCU window refuses it instead of dividing by zero (card_template/eval/benchkit.py).

CPU only where a GPU would be needed; the harness tests shell out to git.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "src" / "rsi_kernel_tune" / "card_template" / "eval"


def _load_eval(name: str):
    """The card-template eval module under its own name (benchkit.py is copied into
    cards and loaded by path there too, so tests do the same)."""
    spec = importlib.util.spec_from_file_location(f"rkt_guard_fixes_{name}",
                                                  EVAL_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from rsi_kernel_tune.loop import cli, harness_guard  # noqa: E402
from rsi_kernel_tune.loop.goal_ref import library_findings  # noqa: E402

lb = _load_eval("benchkit")
precision = _load_eval("precision")


# --- BUG-SEED-07: a tree-root module the harness imports without naming it is harness --

def test_a_root_import_hijack_is_harness_and_the_kernel_is_not():
    # the demonstrated attacks: the interpreter's startup hook, a stdlib shadow
    assert harness_guard.is_harness_path("sitecustomize.py", "kernel.py")
    assert harness_guard.is_harness_path("usercustomize.py", "kernel.py")
    assert harness_guard.is_harness_path("./sitecustomize.py", "kernel.py")
    assert harness_guard.is_harness_path("json.py", "kernel.py")
    assert harness_guard.is_harness_path("csv.py", "kernel.py")
    assert harness_guard.is_harness_path("os.py", "kernel.py")
    # the kernel file and the old rules stay
    assert not harness_guard.is_harness_path("kernel.py", "kernel.py")
    # the default kernel_file="" keeps the existing convention: the kernel file itself
    # is not harness (nothing imports it implicitly), but a root stdlib shadow is
    assert not harness_guard.is_harness_path("kernel.py")
    assert harness_guard.is_harness_path("time.py")
    assert not harness_guard.is_harness_path("eval/kernel.py", "eval/kernel.py")
    # deeper paths are the student's, whatever the name
    assert not harness_guard.is_harness_path("deep/dir/module.py", "kernel.py")
    assert not harness_guard.is_harness_path("notes/json.py", "kernel.py")
    # a root module nothing imports implicitly (a helper the tree names itself) is not
    assert not harness_guard.is_harness_path("newmod.py", "kernel.py")
    assert not harness_guard.is_harness_path("gpu_entry.sh.bak", "kernel.py")
    assert not harness_guard.is_harness_path("window.toml", "kernel.py")
    # the harness prefixes still win, at depth too
    assert harness_guard.is_harness_path("eval/sitecustomize.py", "kernel.py")


def test_git_pathspecs_carry_the_root_module_glob():
    # git ls-files / ls-tree / diff (the listings, the overlay and the delivery and
    # hygiene checks in cli.py) take the wildcard; is_harness_path narrows the depth
    # and the name afterwards.
    assert harness_guard.ROOT_PY_PATHSPEC == "*.py"
    assert "*.py" in harness_guard.HARNESS_PATHSPECS
    assert tuple(p.rstrip("/") for p in harness_guard.HARNESS_PREFIXES) \
        == harness_guard.HARNESS_PATHSPECS[:-1]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, msg: str) -> str:
    _git(repo, "add", "-A")
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-m", msg], check=True)
    return _git(repo, "rev-parse", "HEAD")


def test_tree_listings_carry_root_hijack_modules(tmp_path):
    """tree_harness_files feeds the overlay drop (cli.py _overlay_wanted) and the
    hygiene drift: a root sitecustomize.py / json.py in a job tree must be IN it."""
    tree = tmp_path / "tree"
    (tree / "eval").mkdir(parents=True)
    (tree / "kernel.py").write_text("def run(): pass\n")
    (tree / "eval/bench.py").write_text("print('judge')\n")
    (tree / "sitecustomize.py").write_text("import os\nos.environ['HACK'] = '1'\n")
    (tree / "json.py").write_text("loads = lambda *a: {'pass': True}\n")
    (tree / "deep").mkdir()
    (tree / "deep/lib.py").write_text("x = 1\n")          # not root: not harness
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    head = _commit(tree, "seed")
    want = {"eval/bench.py", "sitecustomize.py", "json.py"}
    assert set(harness_guard.tree_harness_files(tree, head, "kernel.py")) == want
    assert set(harness_guard.harness_only(["sitecustomize.py", "json.py", "newmod.py",
                                           "deep/lib.py"], "kernel.py")) == want - {"eval/bench.py"}


# --- BUG-SEED-08: the import scan sees every name and the dynamic forms ---------------

def test_every_import_name_is_scanned():
    """One name per statement read the scan before: the second name of
    ``import os, flashinfer`` escaped, as did ``;``-joined statements and the
    string literals of importlib.import_module / __import__ (BUG-SEED-08)."""
    cases = [
        "import os, flashinfer\n",                       # the escape: first name is clean
        "import torch\nimport flashinfer.rope\n",        # a submodule still names the lib
        "import flashinfer as fi, torch\n",              # an as-binding on the way
        "import a; import flashinfer\n",                 # two statements on one line
        "from flashinfer.rope import apply_rope\n",
        "from flashinfer.rope import (\n    apply_rope,\n    Rope,\n)\n",
        'import importlib\nimportlib.import_module("flashinfer")\n',
        'import importlib.util\nx = importlib.util\nimportlib.import_module("vllm.model")\n',
        'm = __import__("sgl_kernel.attention")\n',
        'importlib.import_module(flashinfer_name)\n',    # not a literal: not flagged
    ]
    for i, src in enumerate(cases[:-1]):
        found = library_findings(src, "kernel.py")
        assert any("flashinfer" in f or "vllm" in f or "sgl_kernel" in f for f in found), \
            f"case {i} escaped: {found}"
    # a variable target cannot be read -- the heuristic names what it can see only
    assert library_findings(cases[-1], "kernel.py") == []


def test_the_import_scan_stays_quiet_on_clean_and_foreign_text():
    clean = '"""contract: out = a + b"""\nimport torch\n' \
            "def run(a, b, out):\n    out.copy_(a + b)\n"
    assert library_findings(clean, "kernel.py") == []
    assert library_findings("from . import helper\nfrom ops import thing\n", "kernel.py") == []
    # a CUDA source has no import statement
    cu = '#include <cublas_v2.h>\nvoid run() { cublasGemmEx(h); }\n'
    found = library_findings(cu, "kernel.cu")
    assert len(found) == 2 and not any("import" in f for f in found)


# --- BUG-C3-01: the torch C++ call forms are library findings -------------------------

def test_cpp_torch_call_forms_are_flagged():
    """The Python patterns name dotted spellings only, so a .cu/.cpp kernel computing
    through the torch C++ API read as a kernel of its own (BUG-C3-01)."""
    calls = ["torch::matmul(a, b)", "at::mm(x, y)", "torch::bmm(x, y)",
             "at::addmm(bias, x, y)", "at::baddbmm(b, x, y)",
             "torch::einsum(\"bhm,mh->bh\", q, k)", "at::softmax(x, -1)",
             "at::log_softmax(x, -1)", "torch::layer_norm(x, m, v)",
             "at::rms_norm(x, w)", "torch::group_norm(x, 8, m, v)",
             "at::conv2d(x, w)", "at::conv_transpose2d(x, w)",
             "at::linear(x, w, b)", "at::scaled_dot_product_attention(q, k, v)",
             "at::_scaled_mm(a, b)", "at::cdist(a, b)"]
    for call in calls:
        for suffix in (".cu", ".cpp"):
            src = f"#include <torch/torch.h>\nvoid f() {{ auto o = {call}; }}\n"
            found = library_findings(src, f"kernel{suffix}")
            assert found, f"{call} escaped on {suffix}: {found}"
    assert library_findings("auto o = torch::ops::aten::mm::call(a, b);\n", "kernel.cu")
    assert library_findings("auto o = torch::nn::functional::linear(x, w);\n", "kernel.cpp")


def test_the_python_dotted_forms_still_fire():
    assert library_findings("def f(a, b):\n    return torch.matmul(a, b)\n", "kernel.py")
    assert library_findings("import torch\n"
                            "def f(q, k, v):\n"
                            "    return torch.nn.functional.scaled_dot_product_attention(q, k, v)\n",
                            "kernel.py")


def test_a_handwritten_cuda_kernel_stays_quiet():
    src = ("#include <cuda_runtime.h>\n"
           "__global__ void scale_kernel(const float* x, float* y, int n) {\n"
           "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "    if (i < n) y[i] = 1.5f * x[i];\n"
           "}\n"
           "void launch(const float* x, float* y, int n) {\n"
           "    scale_kernel<<<16, 256>>>(x, y, n);\n"
           "    cudaDeviceSynchronize();\n"
           "}\n")
    assert library_findings(src, "kernel.cu") == []


# --- BUG-SEED-09: load_entry must not shadow already-visible modules ------------------

def test_load_entry_appends_so_the_stdlib_and_the_reference_are_not_shadowed(tmp_path):
    """The kernel's directory used to be PREPENDED to sys.path and stayed there for the
    life of the process: the reference loads first (run_bench 2440) and the kernel second
    (2455), so the reference's lazy imports resolved into the kernel's directory -- a
    kernel-side csv.py was the csv the judge imported (BUG-SEED-09)."""
    sfx = uuid.uuid4().hex[:8]                      # unique names: no sys.modules carry-over
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    k_dir = tmp_path / "kern"
    k_dir.mkdir()
    (k_dir / "csv.py").write_text("GOT = 'kernel-forged'\n")     # the stdlib shadow
    (ref_dir / f"helper{sfx}.py").write_text("W = 'reference'\n")
    # the reference imports its helper LAZILY, at call time -- after the kernel's
    # directory is already on sys.path
    (ref_dir / "refimpl.py").write_text(f"def run():\n    import helper{sfx}\n"
                                        f"    return helper{sfx}.W\n")
    (k_dir / "kernel.py").write_text("def run(): return 'k'\n")
    saved_path = list(sys.path)
    saved_csv = sys.modules.pop("csv", None)
    try:
        ref_fn = lb.load_entry(ref_dir / "refimpl.py", "run")     # reference loads first
        kernel_fn = lb.load_entry(k_dir / "kernel.py", "run")     # kernel loads second
        assert sys.path.index(str(ref_dir)) < sys.path.index(str(k_dir))
        import csv as after
        assert Path(after.__file__).resolve() != (k_dir / "csv.py").resolve()
        assert ref_fn() == "reference"
        # a sibling import of the kernel's own still resolves from the kernel's dir
        (k_dir / f"own{sfx}.py").write_text("V = 42\n")
        (k_dir / "sibling.py").write_text(f"def run():\n    import own{sfx}\n"
                                          f"    return own{sfx}.V\n")
        assert lb.load_entry(k_dir / "sibling.py", "run")() == 42
    finally:
        sys.path[:] = saved_path
        for name in (f"helper{sfx}", f"own{sfx}", "csv"):
            sys.modules.pop(name, None)
        if saved_csv is not None:
            sys.modules["csv"] = saved_csv


# --- BUG-C5-02: a Python kernel that dies at import is a structured verdict -----------

_TASK_JUDGE = """import torch

def run(hidden_states, weight):
    return hidden_states

def make_inputs(axes, group='plain', seed=0, device='cpu'):
    b = axes['batch_size']
    g = torch.Generator().manual_seed(seed)
    return {'hidden_states': torch.randn(b, 64, generator=g),
            'weight': torch.ones(64)}
"""


def _task_card(tmp_path: Path, kernel_src: str, kernel_file: str = "kernel.py",
               input_groups=None) -> Path:
    """A minimal task card: eval/problem/problem.json + the kernel file. Returns the
    problem.json path (what run_bench / ncu_window read)."""
    card = tmp_path / "card"
    (card / "eval" / "problem").mkdir(parents=True)
    (card / kernel_file).write_text(kernel_src)
    workload = {"uuid": "w0", "axes": {"batch_size": 5},
                "input_groups": ([{"name": "plain", "kind": "generated", "seed": 1}]
                                 if input_groups is None else input_groups)}
    defn = {"name": "rmsnorm_h64", "op_type": "rmsnorm",
            "axes": {"batch_size": {"type": "var"}, "hidden_size": {"type": "const", "value": 64}},
            "inputs": {"hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"},
                       "weight": {"shape": ["hidden_size"], "dtype": "float32"}},
            "outputs": {"output": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"}},
            "reference": _TASK_JUDGE, "custom_inputs_entrypoint": "make_inputs"}
    problem = {"definition": defn, "workload": workload, "dps": True,
               "kernel_file": kernel_file}
    path = card / "eval" / "problem" / "problem.json"
    path.write_text(json.dumps(problem))
    return path


def _emit_json(capsys) -> dict:
    text = capsys.readouterr().out
    return json.loads(text.split("=== JSON ===")[1])


def test_a_python_kernel_that_dies_at_import_is_a_structured_verdict(tmp_path, capsys):
    """A module-level exception used to escape run_bench as a bare raise: the job
    burned with no eval JSON at all (BUG-C5-02). Non-CUDA kernels now report their
    load failure like a CUDA build failure reports its own."""
    path = _task_card(tmp_path, "import os\nraise RuntimeError('boom at import')\n")
    assert lb.run_bench(["--mode", "correctness"], problem_path=path) == 0
    out = _emit_json(capsys)
    assert out["correctness"]["pass"] is False
    assert out["correctness"]["status"] == "IMPORT_ERROR"
    assert "boom at import" in out["correctness"]["error"]
    assert out["performance"] is None


def test_a_cuda_kernel_build_failure_still_reports_compile_error(tmp_path, monkeypatch,
                                                                 capsys):
    real_impl = lb._impl

    def boom(problem, which):
        if which == "kernel":
            raise ValueError("nvcc exploded")
        return real_impl(problem, which)      # the reference side loads as usual

    monkeypatch.setattr(lb, "_impl", boom)
    path = _task_card(tmp_path, "// not built in tests\n", kernel_file="kernel.cu")
    assert lb.run_bench(["--mode", "correctness"], problem_path=path) == 0
    out = _emit_json(capsys)
    assert out["correctness"]["status"] == "COMPILE_ERROR"
    assert "nvcc exploded" in out["correctness"]["error"]



# --- BUG-C5-03: an empty input_groups is an INPUT_ERROR, never a pass -----------------

def test_empty_input_groups_is_an_input_error_not_a_pass():
    """The group loop ran zero times and `verdict["pass"] = True` fell out of it: an
    empty input_groups PASSED correctness, then timing blew up on no work and the ncu
    window on a division by zero -- a card defect reading as a win (BUG-C5-03)."""
    problem = json.loads(json.dumps(_TASK_PROBLEM_EMPTY))
    verdict, sets = lb.task_correctness(problem, lambda *a, **k: [], False, "cpu")
    assert verdict["pass"] is False
    assert verdict["status"] == "INPUT_ERROR"
    assert "input_groups" in verdict["error"]
    assert sets == []


_TASK_PROBLEM_EMPTY = {
    "definition": {
        "name": "rmsnorm_h64", "op_type": "rmsnorm",
        "axes": {"batch_size": {"type": "var"}, "hidden_size": {"type": "const", "value": 64}},
        "inputs": {"hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"},
                   "weight": {"shape": ["hidden_size"], "dtype": "float32"}},
        "outputs": {"output": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"}},
        "reference": _TASK_JUDGE, "custom_inputs_entrypoint": "make_inputs",
    },
    "workload": {"uuid": "w0", "axes": {"batch_size": 5}, "input_groups": []},
    "dps": True, "_dir": ".",
}


def test_ncu_window_refuses_a_workload_with_no_input_groups(tmp_path, monkeypatch, capsys):
    """ncu_window spread calls with `per = max(1, calls // len(pairs))`, which divides
    by zero on the empty pairs an empty input_groups produces (BUG-C5-03)."""
    path = _task_card(tmp_path, "def run(hidden_states, weight):\n    return hidden_states\n",
                      input_groups=[])
    monkeypatch.setattr(lb, "kernel_fence", lambda problem: None)
    monkeypatch.setattr(lb, "_impl", lambda problem, which: (lambda *a: None, False))
    with pytest.raises(SystemExit) as ei:
        lb.ncu_window(problem_path=path)
    assert ei.value.code == 3
    assert "INPUT_ERROR" in capsys.readouterr().out


# --- ported from the R2 fix-wave archive (2026-11-05) -------------------------------
#
# Ported and holding in the latest tree: the two import-scan tests above are the
# archive's verbatim (BUG-SEED-08), the load_entry test above merges the archive's
# ordering / lazy-reference / sibling assertions with the stdlib-shadow scenario
# (BUG-SEED-09), and the harness unit tests above carry the archive's is_harness_path
# cases under the narrower root-module rule this tree pins (the shared
# test_harness_guard.py freezes "evaluation.py" -- a root module nothing imports
# implicitly -- as NOT harness, so the blanket root-.py rule of the archive could not
# land here; the hijack names a harness executes without being named do).
#
# Dropped from the archive, with reasons:
# * fib_judge / SUITE_TASK / FIB_DEFAULTS (non-finite reference, task stress rows):
#   the task-bench engine they tested is gone from the latest tree; the benchkit
#   successor documents gen_inputs' custom_inputs_fn as "accepted and ignored", i.e.
#   the stress-rows-draw-make_inputs fix is not part of this tree and outside this
#   cluster's fix list.
# * timing _geomean / aggregate degrade (3 tests): the latest timing.py still geomeans
#   over the subset when a reading is None and round(None) crashes on a 0 reading --
#   the archive fix is not in this tree and timing.py is outside the cluster's
#   editable files.
# * precision length-mismatch (1 test): the latest PrecisionRecorder.measure still
#   truncates silently via zip (verified: a short reference side yields a 'pass' row);
#   precision.py is outside the cluster's editable files.
# * timing_audit host_path_note (2 tests): SEED-23 is another engineer's lane.

def _card(tmp_path: Path) -> Path:
    """A card whose tracked harness includes a root sitecustomize.py (the card's own
    startup hook): the overlay must ride it byte for byte."""
    card = tmp_path / "card"
    (card / "eval").mkdir(parents=True)
    (card / "kernel.py").write_text("def run(): pass\n")
    (card / "eval/bench.py").write_text("print('judge')\n")
    (card / "gpu_entry.sh").write_text("#!/bin/bash\n")
    (card / "sitecustomize.py").write_text("# the card's own root helper\n")
    (card / "deep").mkdir()
    (card / "deep/lib.py").write_text("x = 1\n")          # not root: not harness
    subprocess.run(["git", "init", "-q", str(card)], check=True)
    _commit(card, "card")
    return card


def test_tree_and_card_listings_carry_root_modules(tmp_path):
    card = _card(tmp_path)
    want = {"eval/bench.py", "gpu_entry.sh", "sitecustomize.py"}
    assert set(harness_guard.tree_harness_files(card, "HEAD", "kernel.py")) == want
    assert set(harness_guard.card_harness_files(card, "kernel.py")) == want
    # the default kernel_file="" keeps the pre-existing convention: only the names a
    # harness executes without being named count, so a plain root kernel does not
    assert "kernel.py" not in harness_guard.tree_harness_files(card, "HEAD")


def test_a_student_root_module_drifts_and_is_dropped_from_the_job_tree(tmp_path):
    card = _card(tmp_path)
    ws = tmp_path / "ws"
    subprocess.run(["git", "clone", "-q", str(card), str(ws)], check=True)
    (ws / "sitecustomize.py").write_text("import os\nos.environ['HACK'] = '1'\n")
    (ws / "json.py").write_text("def loads(*a): return {'pass': True}\n")  # the card has no such file
    (ws / "kernel.py").write_text("def run(): return 1\n")
    head = _commit(ws, "student")
    base = _git(ws, "rev-parse", "HEAD~1")

    be = cli.RealBackend("demo", ws, None, "t", "t@t")
    be.kernel_file = "kernel.py"
    be.harness_card = card

    # the delivery diff names them (the driver reverts such a commit before any job)
    assert be.harness_changes(base, head) == ["json.py", "sitecustomize.py"]
    # the overlay restores the card's file, drops the student-only one, keeps the kernel
    job = be._with_job_env(head)
    assert job != head and _git(ws, "rev-parse", "HEAD") == head
    tree = set(_git(ws, "ls-tree", "-r", "--name-only", job).splitlines())
    assert "json.py" not in tree
    assert "kernel.py" in tree and "eval/bench.py" in tree
    shown = subprocess.run(["git", "-C", str(ws), "show", f"{job}:sitecustomize.py"],
                           check=True, capture_output=True).stdout
    assert shown == b"# the card's own root helper\n"


# --- PORT union: the precision stress record measures real draws (adapted to this tree's loaders) ---

# --- the precision stress record measures real draws -------------------------------

TASK_JUDGE = """import torch

def run(x, w):
    y = x.to(torch.float64)
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * w.to(torch.float64)
    return y.to(x.dtype)
"""


def _task_problem() -> dict:
    defn = {"name": "rms", "axes": {"m": {"type": "var"}, "n": {"type": "var"}},
            "inputs": {"x": {"shape": ["m", "n"], "dtype": "float32"},
                       "w": {"shape": ["n"], "dtype": "float32"}},
            "outputs": {"y": {"shape": ["m", "n"], "dtype": "float32"}},
            "reference": TASK_JUDGE}
    return {"definition": defn,
            "workload": {"uuid": "s", "axes": {"m": 4, "n": 16},
                         "input_groups": [{"name": "plain", "kind": "generated", "seed": 1},
                                          {"name": "spiky", "kind": "generated", "seed": 2}]},
            "dps": False, "_dir": "."}


def test_the_precision_stress_record_measures_real_draws():
    problem = _task_problem()

    def kernel(x, w):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * w

    prec = precision.PrecisionRecorder()
    lb._precision_stress(problem, kernel, False, "cpu", prec)
    rows = [r for r in prec.rows if r.get("case") == "stress"]
    assert {r.get("stress") for r in rows} == set(precision.STRESS_PROFILES)
    assert all(r["status"] in ("pass", "fail") for r in rows), prec.rows
    # the regression this pins: not one row may say the stress had nothing to move
    assert not any("no plain random float input" in str(r.get("why")) for r in prec.rows)
