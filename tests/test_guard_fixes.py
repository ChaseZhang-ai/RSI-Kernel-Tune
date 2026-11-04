"""Regression tests for the guard and eval fixes (corps fix batch, audit r2):

* harness: every tree-root python module is harness (a sitecustomize.py / json.py a
  student drops at the root hijacks a harness that runs with the workspace on
  PYTHONPATH), covered by the overlay drop, the delivery-diff check, the hygiene
  drift and the tree/card listings -- loop/harness_guard.py.
* goal-ref: the library scan sees every name of an import statement, the string
  literals of importlib.import_module / __import__ / exec, a load_inline source
  string, and the C++ (torch:: / at:: / c10::) spellings of the torch compute
  APIs -- loop/goal_ref.py.
* benchkit: load_entry appends the kernel's directory to sys.path instead of
  prepending it (the reference's lazy imports must not be shadowed); a python
  kernel or reference that cannot load lands in the verdict JSON instead of a
  bare traceback; a workload with no input groups is an INPUT_ERROR, never a
  0-check PASS -- card_template/eval/benchkit.py.
* gpu_entry: the workspace rides PYTHONPATH behind the inherited front (the
  monitor's hook directory stays first) -- cardkit.py.
* monitor hook: _chain() never chains a student-owned sitecustomize --
  monitor/runtime/hook/sitecustomize.py.
* the precision stress record still measures the task's inputs -- benchkit.py.

Ported from the archived upstream fix batch. Dropped, with reasons in CHANGES:
fib_judge (board lane gone), host_path_note (SEED-23 -- another engineer's
fix), the timing-aggregate degrade trio and the precision length-mismatch test
(fixes not carried to this tree; timing.py frozen for this batch), and the
gen_inputs make_inputs routing test (dropped by design in this tree).

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
EVAL_DIR = REPO_ROOT / "src" / "peregrine_kernel" / "card_template" / "eval"


def _load_eval(name: str):
    """The card-template eval module under its own name (benchkit.py, timing.py and
    precision.py are copied into cards and loaded by path there too)."""
    spec = importlib.util.spec_from_file_location(f"pk_guard_fixes_{name}",
                                                  EVAL_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from peregrine_kernel.loop import cli, harness_guard  # noqa: E402
from peregrine_kernel.loop.goal_ref import library_findings  # noqa: E402
from peregrine_kernel import cardkit  # noqa: E402

lb = _load_eval("benchkit")
precision = _load_eval("precision")


# --- harness: tree-root python modules are harness -----------------------------------

def test_a_root_python_module_is_harness_and_the_kernel_is_not():
    assert harness_guard.is_harness_path("sitecustomize.py", "kernel.py")
    assert harness_guard.is_harness_path("json.py", "kernel.py")
    assert harness_guard.is_harness_path("./usercustomize.py", "kernel.py")
    assert not harness_guard.is_harness_path("kernel.py", "kernel.py")     # the kernel file
    # the default kernel_file="" keeps the existing convention: no kernel exclusion
    assert harness_guard.is_harness_path("kernel.py")
    assert not harness_guard.is_harness_path("deep/dir/module.py", "kernel.py")
    assert not harness_guard.is_harness_path("notes/probe.py", "kernel.py")
    assert not harness_guard.is_harness_path("gpu_entry.sh.bak", "kernel.py")
    assert not harness_guard.is_harness_path("window.toml", "kernel.py")


def test_git_pathspecs_carry_the_root_module_glob():
    # git diff / ls-files (the delivery and hygiene checks in cli.py) take the
    # wildcard; only the depth is narrowed later, by is_harness_path.
    assert harness_guard.ROOT_PY_PATHSPEC == "*.py"
    assert "*.py" in harness_guard.HARNESS_PATHSPECS
    assert tuple(p.rstrip("/") for p in harness_guard.HARNESS_PREFIXES) \
        == harness_guard.HARNESS_PATHSPECS[:-1]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, msg: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _card(tmp_path: Path) -> Path:
    card = tmp_path / "card"
    (card / "eval").mkdir(parents=True)
    (card / "kernel.py").write_text("def run(): pass\n")
    (card / "eval/bench.py").write_text("print('judge')\n")
    (card / "gpu_entry.sh").write_text("#!/bin/bash\n")
    (card / "sitecustomize.py").write_text("# the card's own root helper\n")
    (card / "deep").mkdir()
    (card / "deep/lib.py").write_text("x = 1\n")          # not root: not harness
    _git(tmp_path, "init", "-q", str(card))
    _commit(card, "card")
    return card


def test_tree_and_card_listings_carry_root_modules(tmp_path):
    card = _card(tmp_path)
    want = {"eval/bench.py", "gpu_entry.sh", "sitecustomize.py"}
    assert set(harness_guard.tree_harness_files(card, "HEAD", "kernel.py")) == want
    assert set(harness_guard.card_harness_files(card, "kernel.py")) == want
    # the default kernel_file="" carries the existing convention: no kernel exclusion
    assert "kernel.py" in harness_guard.tree_harness_files(card, "HEAD")


def test_a_student_root_module_drifts_and_is_dropped_from_the_job_tree(tmp_path):
    card = _card(tmp_path)
    ws = tmp_path / "ws"
    subprocess.run(["git", "clone", "-q", str(card), str(ws)], check=True)
    (ws / "sitecustomize.py").write_text("import os\nos.environ['HACK'] = '1'\n")
    (ws / "newmod.py").write_text("class json: pass\n")   # the card has no such file
    (ws / "kernel.py").write_text("def run(): return 1\n")
    head = _commit(ws, "student")
    base = _git(ws, "rev-parse", "HEAD~1")

    be = cli.RealBackend("demo", ws, None, "t", "t@t")
    be.kernel_file = "kernel.py"
    be.harness_card = card

    # the delivery diff names them (the driver reverts such a commit before any job)
    assert be.harness_changes(base, head) == ["newmod.py", "sitecustomize.py"]
    # the overlay restores the card's file, drops the student-only one, keeps the kernel
    job = be._with_job_env(head)
    assert job != head and _git(ws, "rev-parse", "HEAD") == head
    tree = set(_git(ws, "ls-tree", "-r", "--name-only", job).splitlines())
    assert "newmod.py" not in tree
    assert "kernel.py" in tree and "eval/bench.py" in tree
    shown = subprocess.run(["git", "-C", str(ws), "show", f"{job}:sitecustomize.py"],
                           check=True, capture_output=True).stdout
    assert shown == b"# the card's own root helper\n"


def test_gpu_entry_puts_the_workspace_last_on_pythonpath():
    # the monitor runner puts its hook directory first on PYTHONPATH; the generated
    # entry must not put the workspace back in front of it (a root sitecustomize.py
    # would overtake the hook), and eval/ must sit in front of the workspace too
    line = next(l for l in cardkit.GPU_ENTRY_TEMPLATE.splitlines()
                if "export PYTHONPATH" in l)
    # the template doubles its braces (${{PYTHONPATH:-}} renders as ${PYTHONPATH:-})
    assert line.strip() == 'export PYTHONPATH="${{PYTHONPATH:-}}:$WORKSPACE/eval:$WORKSPACE"', \
        line


# --- monitor hook: never chain a student-owned sitecustomize -------------------------

HOOK_DIR = REPO_ROOT / "src" / "peregrine_kernel" / "monitor" / "runtime" / "hook"


def _chain_run(pythonpath: str, cwd: Path, code: str) -> str:
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": pythonpath}
    proc = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True,
                          text=True, cwd=str(cwd), env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_the_hook_never_chains_a_student_sitecustomize(tmp_path):
    """The deployed layout (<ws>/.pk_monitor/runtime/hook): the hook is imported from
    the front of PYTHONPATH, and its _chain must skip the student's tree root -- only
    an image-shipped sitecustomize is handed over."""
    hook = tmp_path / "ws" / ".pk_monitor" / "runtime" / "hook"
    hook.mkdir(parents=True)
    for name in ("sitecustomize.py", "pk_monitor_hook.py"):
        (hook / name).write_text((HOOK_DIR / name).read_text(encoding="utf-8"))
    (tmp_path / "ws" / "sitecustomize.py").write_text(
        "import os\nos.environ['STUDENT_SC_RAN'] = '1'\n")
    out = _chain_run(f"{hook}:{tmp_path / 'ws'}", tmp_path / "ws",
                     "import os, sys\n"
                     "print('hook installed:', 'pk_monitor_hook' in sys.modules)\n"
                     "print('student ran:', os.environ.get('STUDENT_SC_RAN') is not None)\n"
                     "print('sc file:', getattr(sys.modules.get('sitecustomize'), "
                     "'__file__', None))\n")
    assert "hook installed: True" in out
    assert "student ran: False" in out
    sc = out.split("sc file: ")[1]
    assert str(tmp_path / "ws") not in sc


def test_the_hook_chain_still_hands_over_outside_a_workspace(tmp_path):
    """Outside the deployed layout there is no student root to skip: the next
    sitecustomize on the path is chained as before."""
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "sitecustomize.py").write_text(
        "import os\nos.environ['BENIGN_SC'] = '1'\n")
    out = _chain_run(f"{HOOK_DIR}:{plain}", tmp_path,
                     "import os, sys\n"
                     "print('benign ran:', os.environ.get('BENIGN_SC') is not None)\n"
                     "print('sc file:', getattr(sys.modules.get('sitecustomize'), "
                     "'__file__', None))\n")
    assert "benign ran: True" in out
    assert str(plain / "sitecustomize.py") in out


# --- goal-ref: the import scan cannot be escaped by commas or importlib --------------

def test_every_import_name_is_scanned():
    cases = [
        "import os, flashinfer\n",                       # the escape: first name is clean
        "import torch\nimport flashinfer.rope\n",        # a submodule still names the lib
        "import flashinfer as fi, torch\n",              # an as-binding on the way
        "import a; import flashinfer\n",                 # two statements on one line
        "a = 1;import flashinfer\n",                     # ... mid-line, unanchored
        "from flashinfer.rope import apply_rope\n",
        "from flashinfer.rope import (\n    apply_rope,\n    Rope,\n)\n",
        'import importlib\nimportlib.import_module("flashinfer")\n',
        'import importlib.util\nx = importlib.util\nimportlib.import_module("vllm.model")\n',
        'm = __import__("sgl_kernel.attention")\n',
        "exec('import flashinfer')\n",                   # an exec'd literal still runs it
        'importlib.import_module(flashinfer_name)\n',    # not a literal: not flagged
    ]
    for i, src in enumerate(cases[:-1]):
        found = library_findings(src, "kernel.py")
        assert any("flashinfer" in f or "vllm" in f or "sgl_kernel" in f for f in found), \
            f"case {i} escaped: {found}"
    # a variable target cannot be read -- the heuristic names what it can see only
    assert library_findings(cases[-1], "kernel.py") == []


def test_the_import_scan_survives_a_source_that_does_not_parse():
    # a syntax error higher in the file must not switch the scan off
    src = "def broken(:\n    pass\nimport os, flashinfer\n"
    found = library_findings(src, "kernel.py")
    assert any("flashinfer" in f for f in found), found


def test_the_import_scan_stays_quiet_on_clean_and_foreign_text():
    clean = '"""contract: out = a + b"""\nimport torch\n' \
            "def run(a, b, out):\n    out.copy_(a + b)\n"
    assert library_findings(clean, "kernel.py") == []
    assert library_findings("from . import helper\nfrom ops import thing\n", "kernel.py") == []
    # a CUDA source has no import statement
    cu = '#include <cublas_v2.h>\nvoid run() { cublasGemmEx(h); }\n'
    found = library_findings(cu, "kernel.cu")
    assert len(found) == 2 and not any("import" in f for f in found)


def test_the_cpp_compute_forms_are_screened():
    for src, want in (
        ("void k(at::Tensor x, at::Tensor a) { auto y = torch::matmul(x, a); }",
         "computes with torch::matmul"),
        ("void k(at::Tensor x, at::Tensor a) { auto y = at::mm(x, a); }",
         "computes with at::mm"),
        ("void k() { auto t = at::softmax(z, -1); }", "computes with at::softmax"),
        ("void k() { auto t = torch::ops::matmul(x, a); }",
         "calls the registered op torch::ops::matmul"),
        ("void k() { auto t = at::linear(x, a); }", "computes with at::linear"),
    ):
        found = library_findings(src, "kernel.cu")
        assert any(want in f for f in found), f"{src!r} escaped: {found}"
    # no torch API, no finding: a hand-written kernel stays clean
    innocent = ("#include <cuda_runtime.h>\n__global__ void g(float* p) {}\n"
                "void k() { g<<<1, 1>>>(nullptr); }\n")
    assert library_findings(innocent, "kernel.cu") == []


def test_a_load_inline_source_string_is_screened():
    src = ("import torch\n"
           "torch.utils.cpp_extension.load_inline(\n"
           '    name="k", cpp_sources="",\n'
           '    cuda_sources="#include <cublas_v2.h>\\nvoid k() {}")\n')
    found = library_findings(src, "kernel.py")
    assert any("includes cublas" in f for f in found), found
    # prose naming the library is not a call
    assert library_findings('"""the cublas docs say hi"""\nimport torch\n', "kernel.py") == []


# --- benchkit: load_entry must not shadow already-visible modules --------------------

def test_load_entry_appends_so_a_lazy_reference_import_is_not_shadowed(tmp_path):
    sfx = uuid.uuid4().hex[:8]                      # unique module name: no sys.modules
    ref_dir = tmp_path / "ref"                      # carry-over from other tests
    ref_dir.mkdir()
    k_dir = tmp_path / "kern"
    k_dir.mkdir()
    (ref_dir / f"helper{sfx}.py").write_text("W = 'reference'\n")
    # the reference imports its helper LAZILY, at call time -- after the kernel's
    # directory is already on sys.path
    (ref_dir / "refimpl.py").write_text(f"def run():\n"
                                        f"    import helper{sfx}\n"
                                        f"    return helper{sfx}.W\n")
    (k_dir / f"helper{sfx}.py").write_text("W = 'kernel-forged'\n")
    (k_dir / "kernel.py").write_text("def run(): return 'k'\n")
    saved_path = list(sys.path)
    try:
        ref_fn = lb.load_entry(ref_dir / "refimpl.py", "run")     # reference loads first
        kernel_fn = lb.load_entry(k_dir / "kernel.py", "run")     # kernel loads second
        assert sys.path.index(str(ref_dir)) < sys.path.index(str(k_dir))
        assert ref_fn() == "reference"
        # a sibling import of the kernel's own still resolves from the kernel's dir
        (k_dir / f"own{sfx}.py").write_text("V = 42\n")
        (k_dir / "sibling.py").write_text(f"def run():\n    import own{sfx}\n"
                                          f"    return own{sfx}.V\n")
        assert lb.load_entry(k_dir / "sibling.py", "run")() == 42
    finally:
        sys.path[:] = saved_path
        sys.modules.pop(f"helper{sfx}", None)
        sys.modules.pop(f"own{sfx}", None)


# --- benchkit: a load failure lands in the verdict JSON, never a bare traceback ------

JUDGE = """import torch

def run(x, w):
    y = x.to(torch.float64)
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * w.to(torch.float64)
    return y.to(x.dtype)

def make_inputs(axes, group, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(axes["m"], axes["n"], generator=g)
    return {"x": x.to(device), "w": torch.ones(axes["n"]).to(device)}
"""


def _problem_card(tmp_path: Path, *, kernel: str, workloads=None) -> Path:
    card = tmp_path / "card"
    problem_dir = card / "eval" / "problem"
    problem_dir.mkdir(parents=True)
    problem_dir.joinpath("problem.json").write_text(json.dumps(
        {"suite": "task",
         "definition": {"name": "rms", "axes": {"m": {"type": "var"}, "n": {"type": "var"}},
                        "inputs": {"x": {"shape": ["m", "n"], "dtype": "float32"},
                                   "w": {"shape": ["n"], "dtype": "float32"}},
                        "outputs": {"y": {"shape": ["m", "n"], "dtype": "float32"}},
                        "reference": JUDGE, "custom_inputs_entrypoint": "make_inputs"},
         "workload": workloads or {"uuid": "s", "axes": {"m": 4, "n": 16},
                                   "input_groups": [{"name": "plain", "kind": "generated",
                                                     "seed": 1}]},
         "kernel_file": "kernel.py", "dps": False, "entry": "run"}))
    (card / "kernel.py").write_text(kernel)
    return problem_dir / "problem.json"


def _run_bench_stdout(problem_path: Path, argv=("--mode", "correctness")) -> str:
    out = __import__("io").StringIO()
    with __import__("contextlib").redirect_stdout(out):
        rc = lb.run_bench(list(argv), problem_path=str(problem_path))
    return f"rc={rc}\n" + out.getvalue()


def _eval_block(text: str) -> dict:
    return json.loads(text.split("=== JSON ===", 1)[1])


def test_a_broken_python_kernel_load_lands_in_the_verdict_json(tmp_path):
    problem = _problem_card(tmp_path, kernel="def run(a, w)\n    return a * w\n")
    text = _run_bench_stdout(problem)
    assert "=== JSON ===" in text
    block = _eval_block(text)
    corr = block["correctness"]
    assert corr["pass"] is False and corr["status"] == "IMPORT_ERROR"
    assert "SyntaxError" in corr["error"] and block["performance"] is None


def test_a_broken_reference_load_lands_in_the_verdict_json(tmp_path):
    problem_path = _problem_card(tmp_path, kernel="def run(x, w):\n    return x * w\n")
    problem = json.loads(problem_path.read_text(encoding="utf-8"))
    problem["definition"]["reference"] = "def run(x, w):\n    return x * w\n"  # no make_inputs
    problem_path.write_text(json.dumps(problem))
    text = _run_bench_stdout(problem_path)
    assert "=== JSON ===" in text
    block = _eval_block(text)
    corr = block["correctness"]
    assert corr["pass"] is False and corr["status"] == "REFERENCE_ERROR"


# --- benchkit: an empty input_groups workload is an INPUT_ERROR, never a PASS --------

def test_an_empty_input_groups_workload_is_an_input_error(tmp_path):
    problem_path = _problem_card(tmp_path, kernel="def run(x, w):\n    return x * w\n",
                                 workloads={"uuid": "s", "axes": {"m": 4, "n": 16},
                                            "input_groups": []})
    problem = lb.load_problem(str(problem_path))

    def kernel(x, w):
        return x * w

    verdict, sets = lb.task_correctness(problem, kernel, False, "cpu")
    assert verdict["pass"] is False
    assert verdict["status"] == "INPUT_ERROR" and "input_groups" in verdict["error"]
    assert sets == []


def test_the_ncu_window_refuses_an_empty_workload(tmp_path):
    problem_path = _problem_card(tmp_path, kernel="def run(x, w):\n    return x * w\n",
                                 workloads={"uuid": "s", "axes": {"m": 4, "n": 16},
                                            "input_groups": []})
    with pytest.raises(SystemExit) as ei:
        lb.ncu_window(str(problem_path))
    assert ei.value.code == 3


# --- the precision stress record measures the task's inputs (ported; pins current) ---

def _task_problem() -> dict:
    defn = {"name": "rms", "axes": {"m": {"type": "var"}, "n": {"type": "var"}},
            "inputs": {"x": {"shape": ["m", "n"], "dtype": "float32"},
                       "w": {"shape": ["n"], "dtype": "float32"}},
            "outputs": {"y": {"shape": ["m", "n"], "dtype": "float32"}},
            "reference": JUDGE, "custom_inputs_entrypoint": "make_inputs"}
    return {"suite": "task", "definition": defn,
            "workload": {"uuid": "s", "axes": {"m": 4, "n": 16},
                         "input_groups": [{"name": "plain", "kind": "generated", "seed": 1},
                                          {"name": "spiky", "kind": "generated", "seed": 2}]},
            "dps": False, "_dir": "."}


def test_the_precision_stress_record_measures_the_task_distribution():
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


# The archived file's timing-aggregate tests (a geomean over a missing or
# non-positive reading degrades to None, and workloads_performance propagates it)
# are NOT ported: the current timing._geomean skips None entries and
# workloads_performance crashes on a 0.0 reading (round(None)). That is the
# archived aggregate fix not carried over; timing.py is frozen for this batch
# (only the ncu divide point may move), so the fix belongs to its own task.
# Likewise the precision length-mismatch test (measure over outputs/refs of
# different lengths is unavailable, never silently zip-truncated) pins a guard
# precision.py never carried -- outside this batch's fix list. And
# test_task_stress_rows_draw_make_inputs pins gen_inputs routing the task's
# make_inputs, which peregrine dropped BY DESIGN (docstring: custom_inputs_fn is
# accepted and ignored; the task's draw channel is task_inputs).
