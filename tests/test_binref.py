"""The binary reference (forgekernel.binref) on a generated task card: the card's
sealed in-job reference, the recon tree, the guards, install."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from forgekernel import binref

REPO_ROOT = Path(__file__).resolve().parents[1]
CU = """#include <torch/extension.h>
void run(torch::Tensor x, torch::Tensor w, torch::Tensor out) { out.copy_(x * w); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
"""

DESCRIPTION = "y = x / sqrt(mean(x^2) + 1e-6) * w over the last axis\n"
SHAPES = [{"name": "small", "axes": {"m": 4, "n": 16}},
          {"name": "large", "axes": {"m": 64, "n": 16}}]
TASK_TOML = '[task]\nlanguage = "cutedsl"\narch = "sm100"\n'
PROTOCOL_TOML = TASK_TOML + '[protocol]\nsm_clock_mhz = 1500\nmem_clock_mhz = 3996\n'
REF_PY = "import torch\n\ndef run(x, w):\n    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w\n"
DEFINITION = {"inputs": {"x": {"shape": ["m", "n"], "dtype": "float32"},
                         "w": {"shape": ["n"], "dtype": "float32"}},
              "outputs": {"y": {"shape": ["m", "n"], "dtype": "float32"}}}
JUDGE = ("import torch\n\n"
         "def run(x, w):\n"
         "    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w\n")
GROUPS = {s["name"]: [{"name": "plain", "intent": "randn", "seed": 1}] for s in SHAPES}


def solution(**over) -> dict:
    sol = {"name": "rms_cu", "definition": "rmsnorm", "author": "t",
           "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200"],
                    "entry_point": "kernel.cu::run",
                    "dependencies": [], "destination_passing_style": True},
           "sources": [{"path": "kernel.cu", "content": CU}]}
    sol.update(over)
    return sol


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Path:
    """A scratch place for runs; this machine (local), like test_launch's."""
    monkeypatch.setenv("FORGEKERNEL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FORGEKERNEL_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("FORGEKERNEL_OPERATORS_DIR", raising=False)
    monkeypatch.setenv("FK_DELIVERY", "local")
    monkeypatch.setenv("FK_GPU", "0")
    return tmp_path


@pytest.fixture
def base(repo: Path) -> Path:
    shutil.copytree(REPO_ROOT / "meta", repo / "meta")
    shutil.copytree(REPO_ROOT / "src" / "forgekernel" / "card_template",
                    repo / "src" / "forgekernel" / "card_template")
    return repo


def new_card(base: Path, sol: dict | None = None, *, protocol: bool = True,
             definition: str = "rmsnorm") -> tuple[Path, Path, str]:
    """A generated task card (taskcard.write_card, the normal-task path) carrying the
    binary reference: write_ref_tree, the card guard, the problem's reference_impl."""
    from forgekernel import judge as judge_mod, taskcard
    from forgekernel.localenv import load_machine
    from forgekernel.runs import new_run
    from forgekernel.task import load_task
    sol = sol or solution()
    d = base / "tasks" / "rmsnorm"
    (d / "reference").mkdir(parents=True, exist_ok=True)
    (d / "task.toml").write_text(PROTOCOL_TOML if protocol else TASK_TOML)
    (d / "description.md").write_text(DESCRIPTION)
    (d / "shapes.json").write_text(json.dumps(SHAPES))
    (d / "reference" / "run.py").write_text(REF_PY)
    machine = load_machine(base, environ={})
    run = new_run(load_task(d), machine=machine)
    task = load_task(run.task_dir, name=run.task_name)
    check_dir = run.dir / judge_mod.CHECK_DIR
    check_dir.mkdir(parents=True, exist_ok=True)
    (check_dir / judge_mod.DEFINITION).write_text(json.dumps(DEFINITION))
    (check_dir / judge_mod.JUDGE).write_text(JUDGE)
    (check_dir / judge_mod.GROUPS).write_text(json.dumps(GROUPS))
    check = judge_mod.load_check(check_dir)
    op_dir = taskcard.write_card(run, task, check, machine)
    errs = binref.solution_problems(sol, definition)
    assert not errs, errs
    module = binref.module_name(sol)
    impl = binref.reference_impl(sol, module)
    if protocol:
        taskcard.update_problem(op_dir, reference_impl=impl, time_reference=True)
    ref_dir = run.operators_dir / taskcard.ref_tree(task.name)
    binref.write_ref_tree(ref_dir, op_dir, sol, module)
    entry = op_dir / "gpu_entry.sh"
    entry.write_text(binref.guard_card_gpu_entry(entry.read_text(), module))
    (op_dir / "loop.toml.example").write_text("ref_source_paths   = []\nref_source_urls    = []\n",
                                              encoding="utf-8")
    binref.fill_ref_source_paths(
        op_dir, list(dict.fromkeys([str(ref_dir / "src"), "/elsewhere/plain"])))
    (op_dir / "ref_bin").mkdir(exist_ok=True)
    return op_dir, ref_dir, module


def test_card_times_a_sealed_binary_reference(base: Path) -> None:
    card, ref, module = new_card(base)
    problem = json.loads((card / "eval" / "problem" / "problem.json").read_text())
    assert problem["reference_impl"] == {
        "kind": "binary", "name": f"sealed binary reference {module}",
        "path": f"../../ref_bin/{module}.so", "module": module, "entry": "run", "dps": True}
    assert problem["time_reference"] is True           # a real kernel: timed in-job
    assert not [p for p in card.rglob("*") if p.suffix in (".cu", ".cpp", ".h")]
    assert (card / "ref_bin").is_dir()
    assert "loop.toml" in (card / ".gitignore").read_text().split()
    loop = (card / "loop.toml.example").read_text()
    assert json.dumps([str(ref / "src"), "/elsewhere/plain"]) in loop
    # the declared measurement clock rides the card and its recon tree
    assert "LB_SM_CLOCK_MHZ-1500" in (card / "gpu_entry.sh").read_text()
    assert "LB_SM_CLOCK_MHZ-1500" in (ref / "gpu_entry.sh").read_text()


def test_recon_tree_holds_the_sources_and_a_copy_of_the_bench(base: Path) -> None:
    card, ref, module = new_card(base)
    assert (ref / "src" / "kernel.cu").read_text() == CU
    meta = json.loads((ref / "binref.json").read_text())
    assert meta["module"] == module and meta["sources"] == ["kernel.cu"] and meta["dps"] is True
    assert (ref / "binref.py").read_text() == binref.TOOL.read_text()
    assert (ref / "eval" / "benchkit.py").read_text() == (card / "eval" / "benchkit.py").read_text()
    assert json.loads((ref / "eval" / "problem" / "problem.json").read_text())["reference_impl"]["path"] \
        == f"../../ref_bin/{module}.so"
    assert 'ncu_window(which="ref")' in (ref / "eval" / "ncu_run.py").read_text()
    assert "ref_bin/" in (ref / ".gitignore").read_text()
    for tree in (card, ref):
        assert subprocess.run(["bash", "-n", str(tree / "gpu_entry.sh")]).returncode == 0
        assert (tree / "gpu_entry.sh").read_text().count("BLACKBOX_FORBID=") == 1
    from forgekernel.loop.cli import fresh_start_contamination
    assert fresh_start_contamination(card, card / "milestone.md", card / "student" / "status.jsonl",
                                     card / "meta_status.jsonl", ref, evidence_dir=card / "evidence") == []


def test_recon_tree_refuses_every_task_where_the_clock_lock_does_not_hold(base: Path) -> None:
    """The recon tree inherits the card's declared clock lock and its refusal: its build /
    binref / ncu / cmd jobs on a node at SM 1155 MHz print the line ForgeKernel withholds."""
    from forgekernel.submit.jobs import clock_lock_refusal
    card, ref, _ = new_card(base)
    smi = base / "fake-smi"
    smi.write_text('#!/bin/bash\ncase "$*" in *clocks.current.graphics*) echo "1155, 3996";; '
                   '*) echo FAKE;; esac\n')
    smi.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("LB_")}
    env.update(LB_NVIDIA_SMI=str(smi), LB_CLOCK_SETTLE_S="0")
    for task in ("build", "binref", "ncu", "cmd"):
        proc = subprocess.run(["bash", str(ref / "gpu_entry.sh"), task, "true"], cwd=ref,
                              env=env, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 3 and clock_lock_refusal(proc.stdout), (task, proc.stdout)


FAKE_TORCH = """import json, os, sys, types
from pathlib import Path
torch = types.ModuleType("torch"); torch.__version__ = "0"; torch.version = types.SimpleNamespace(cuda="0")
ext = types.ModuleType("torch.utils.cpp_extension")
def load(**kw):
    root = Path(kw["build_directory"])
    kw["_staged"] = {str(f.relative_to(root)): f.read_text() for f in sorted(root.rglob("*")) if f.is_file()}
    Path(os.environ["LOAD_LOG"]).write_text(json.dumps(kw))
    (Path(kw["build_directory"]) / (kw["name"] + ".so")).write_bytes(b"\\x7fELF")
ext.load = load
torch.utils = types.ModuleType("torch.utils"); torch.utils.cpp_extension = ext
sys.modules.update({"torch": torch, "torch.utils": torch.utils, "torch.utils.cpp_extension": ext})
"""


@pytest.mark.parametrize("opts, cuda, ld", [
    (None, [], []),                                                          # no compile_options: no flags
    ({}, ["-O3", "--use_fast_math"], ["-lcuda"]),                            # present: the defaults
    ({"cuda_cflags": ["-O2"]}, ["-O2"], ["-lcuda"]),                         # TMA GEMMs: driver API needs -lcuda
    ({"ld_flags": ["-lcublas"]}, ["-O3", "--use_fast_math"], ["-lcublas"]),  # explicit ld_flags replace it
    ({"ld_flags": []}, ["-O3", "--use_fast_math"], []),
])
def test_build_resolves_flags_from_the_solution(base: Path, tmp_path: Path,
                                                opts, cuda, ld) -> None:
    spec = dict(solution()["spec"])
    if opts is not None:
        spec["compile_options"] = opts
    _, ref, module = new_card(base, solution(spec=spec))
    assert json.loads((ref / "binref.json").read_text())["compile_options"] == opts
    (tmp_path / "fake").mkdir()
    (tmp_path / "fake" / "sitecustomize.py").write_text(FAKE_TORCH)
    log = tmp_path / "load.json"
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "fake"), "LOAD_LOG": str(log),
           "BINREF_BUILD_DIR": str(tmp_path / "build")}
    r = subprocess.run([sys.executable, "binref.py", "build"], cwd=ref, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    kw = json.loads(log.read_text())
    assert kw["extra_cuda_cflags"] == ["-gencode=arch=compute_100a,code=sm_100a"] + cuda
    assert kw["extra_ldflags"] == ld and kw["extra_cflags"] == []
    assert f"SO_LDFLAGS {json.dumps(ld)}" in r.stdout and (ref / "ref_bin" / f"{module}.so").exists()


def _fake_build(ref: Path, tmp_path: Path, **env_over) -> tuple[subprocess.CompletedProcess, dict]:
    """`binref.py build` against FAKE_TORCH: (the run, the kwargs cpp_extension.load got)."""
    (tmp_path / "fake").mkdir(exist_ok=True)
    (tmp_path / "fake" / "sitecustomize.py").write_text(FAKE_TORCH)
    log = tmp_path / "load.json"
    if log.exists():
        log.unlink()
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "fake"), "LOAD_LOG": str(log),
           "BINREF_BUILD_DIR": str(tmp_path / "build")}
    for k, v in env_over.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    r = subprocess.run([sys.executable, "binref.py", "build"], cwd=ref, env=env, capture_output=True, text=True)
    return r, (json.loads(log.read_text()) if log.exists() else {})


# A spliced solution: each component's files under cN/, a top-level cN__x.cu that
# wraps its #include in a namespace, and a dispatcher with the binding. Two components both ship a
# kernel.cu: compiled as sources they were two ninja rules for one kernel.cuda.o.
SPLICE_SOURCES = [
    ("c0/kernel.cu", "__global__ void k0() {}\n"),
    ("c0__kernel.cu", "namespace sl_c0 {\n#include \"c0/kernel.cu\"\n}\n"),
    ("c1/kernel.cu", "__global__ void k1() {}\n"),
    ("c1/util.cuh", "#pragma once\n"),
    ("c1/extra.cpp", "int c1_helper() { return 1; }\n"),
    ("c1__kernel.cu", "namespace sl_c1 {\n#include \"c1/kernel.cu\"\n}\n"),
    ("splice_dispatch.cu", CU),
]


def test_build_compiles_only_the_top_level_sources(base: Path, tmp_path: Path) -> None:
    sol = solution(sources=[{"path": p, "content": c} for p, c in SPLICE_SOURCES])
    sol["spec"]["entry_point"] = "splice_dispatch.cu::run"
    _, ref, module = new_card(base, sol)
    r, kw = _fake_build(ref, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    build = tmp_path / "build" / module
    assert kw["sources"] == [str(build / "c0__kernel.cu"), str(build / "c1__kernel.cu"),
                             str(build / "splice_dispatch.cu")]
    # every file is staged for the #includes, and the build root is on the include path
    for rel, content in SPLICE_SOURCES:
        assert kw["_staged"][rel] == content
    assert kw["extra_include_paths"][0] == str(build)
    # ... and gone once the .so is out: the staging dir sits outside the tree's clone dir on a
    # node other trees' jobs may borrow, and holds the sealed sources in the clear
    assert not build.exists() and (ref / "ref_bin" / f"{module}.so").exists()
    (tmp_path / "binref-meta").mkdir()
    (tmp_path / "binref-meta" / "binref.json").write_text(json.dumps({"module": "m", "sources": []}))
    ns: dict = {"__name__": "binref_tool_t",
                "__file__": str(tmp_path / "binref-meta" / "binref.py")}
    exec(compile(binref.TOOL.read_text(), str(binref.TOOL), "exec"), ns)
    assert ns["compiled_sources"](["a/k.cu", "k.cu", "k.cuh", "b\\x.cpp", "x.c", "d/e/f.cc"]) == ["k.cu", "x.c"]


def test_a_solution_with_no_top_level_source_is_refused() -> None:
    sol = solution(sources=[{"path": "c0/kernel.cu", "content": CU}])
    sol["spec"]["entry_point"] = "c0/kernel.cu::run"
    assert any("top of the solution" in e for e in binref.solution_problems(sol, "rmsnorm"))
    assert not binref.solution_problems(solution(), "rmsnorm")


@pytest.mark.parametrize("deps", [[], ["CUTLASS_4_4"]])
def test_build_puts_cutlass_on_the_include_path_whatever_the_dependencies(base: Path,
                                                                          tmp_path: Path,
                                                                          deps) -> None:
    """The build: $CUTLASS_DIR (else /usr/local/cutlass) include/ and tools/util/include/
    on every CUDA C++ build -- spec.dependencies (CUTLASS_4_4 or none) play no part.
    Headers missing there: a warning that names the directory, then the build."""
    sol = solution()
    sol["spec"]["dependencies"] = deps
    _, ref, _ = new_card(base, sol)
    cut = tmp_path / "cutlass"
    (cut / "include" / "cutlass").mkdir(parents=True, exist_ok=True)
    (cut / "include" / "cutlass" / "cutlass.h").write_text("#pragma once\n")
    r, kw = _fake_build(ref, tmp_path, CUTLASS_DIR=str(cut))
    assert r.returncode == 0, r.stdout + r.stderr
    assert kw["extra_include_paths"][1:] == [f"{cut}/include", f"{cut}/tools/util/include"]
    assert "CUTLASS headers not found" not in r.stdout
    # CUTLASS_DIR unset: the image default; this test host has no headers there
    r, kw = _fake_build(ref, tmp_path, CUTLASS_DIR=None)
    assert r.returncode == 0, r.stdout + r.stderr
    assert kw["extra_include_paths"][1:] == ["/usr/local/cutlass/include",
                                             "/usr/local/cutlass/tools/util/include"]
    if not Path("/usr/local/cutlass/include/cutlass/cutlass.h").is_file():
        assert "CUTLASS headers not found under /usr/local/cutlass/include (CUTLASS_DIR unset" in r.stdout
    r, kw = _fake_build(ref, tmp_path, CUTLASS_DIR=str(tmp_path / "nowhere"))
    assert r.returncode == 0 and f"CUTLASS headers not found under {tmp_path / 'nowhere'}/include" in r.stdout


def _guard(tree: Path, cmd: str) -> str:
    # the guard's block: BLACKBOX_FORBID= through the `}` closing blackbox_guard() (a recon
    # tree's block defines blackbox_soft() first)
    script = ("DEF=$(awk '/^BLACKBOX_FORBID=/{b=\"\";on=1;g=0} on{b=b $0 \"\\n\"} /^blackbox_guard\\(\\)/{g=1} "
              "on&&g&&/^}/{on=0;l=b} END{printf \"%s\", l}' gpu_entry.sh); eval \"$DEF\"; "
              "(blackbox_guard \"$1\"; echo ALLOWED) | tail -1")
    r = subprocess.run(["bash", "-c", script, "guard", cmd], cwd=tree, capture_output=True, text=True)
    return "ALLOWED" if r.stdout.strip() == "ALLOWED" else "REFUSED"


def test_guards(base: Path) -> None:
    card, ref, module = new_card(base)
    so = f"ref_bin/{module}.so"
    for cmd, want in [("cat src/kernel.cu", "REFUSED"), ("grep run binref.json", "REFUSED"),
                      ("ncu --print-source cuda python3 eval/ncu_run.py", "REFUSED"),
                      (f"cuobjdump -ptx {so}", "REFUSED"),
                      (f"cuobjdump -sass {so}", "ALLOWED"),
                      ("ncu --print-source sass python3 eval/ncu_run.py", "ALLOWED"),
                      ("python3 notes/r1.py", "ALLOWED")]:
        assert _guard(ref, cmd) == want, ("recon tree", cmd)
    for cmd, want in [(f"cuobjdump -sass {so}", "REFUSED"), ("nm -D x.so", "REFUSED"),
                      (f"python3 -c 'import {module}'", "REFUSED"),
                      ("python3 eval/bench.py --mode all", "ALLOWED")]:
        assert _guard(card, cmd) == want, ("card", cmd)


PROBE_PY = '''import os, subprocess
SRC = """#include <cute/tensor.hpp>
#include <cuda_runtime.h>
__global__ void k(unsigned *a) { asm volatile("add.cc.u32 %0, %0, 1;" : "+r"(a[0])); }
"""
os.makedirs("/tmp/r3", exist_ok=True)
open("/tmp/r3/probe.cu", "w").write(SRC)
subprocess.run(["nvcc", "-x", "cu", "-cubin", "-o", "/tmp/r3/probe.cubin", "/tmp/r3/probe.cu"])
subprocess.run(["nvcc", "notes/p_expr.cu", "-o", "/tmp/r3/p"])
'''


@pytest.mark.parametrize("cmd, want", [
    ("nvcc -O3 notes/p_expr.cu -o /tmp/p/a.out", "ALLOWED"),
    ("nvcc -x cu -cubin -o /tmp/p.cubin /tmp/r4/probe.cu", "ALLOWED"),
    ("grep -c cta_group /usr/local/cutlass/include/cute/arch/copy_sm100.hpp", "ALLOWED"),
    ("grep -c x /usr/local/cuda-13.1/include/cuda_runtime.h", "ALLOWED"),
    ("cat /workspace/repo/k.cu", "REFUSED"),                                  # the tree
    ("cat /usr/../workspace/k.cu", "REFUSED"),
    ("ls *.cu", "REFUSED"),
    ("cat ../x.cuh", "REFUSED"),
    ("cat src/x", "REFUSED"),
    ("nvcc notes/kernel.cu", "REFUSED"),                                      # the sealed source's name
])
def test_the_recon_tree_lets_a_cpp_probe_compile_and_still_refuses_a_reach(
        base: Path, cmd: str, want: str) -> None:
    _, ref, _ = new_card(base)
    assert _guard(ref, cmd) == want, cmd


@pytest.mark.parametrize("name, text, want", [
    ("r3.py", PROBE_PY, "ALLOWED"),
    ("p_expr.cu", "#include <cutlass/cutlass.h>\n#include \"probe_util.cuh\"\n__global__ void k() {}\n", "ALLOWED"),
    ("r3.py", "import glob\nprint(glob.glob('**/*.cu', recursive=True))\n", "REFUSED"),
    ("r3.py", "import os\nprint([f for f in os.listdir('.') if f.endswith('.cuh')])\n", "REFUSED"),
    ("r3.py", "print(open('../kernel.cu').read())\n", "REFUSED"),
    ("r3.py", "print(open('notes/kernel.cu').read())\n", "REFUSED"),
    ("p_expr.cu", "#include \"../src/kernel.cu\"\n", "REFUSED"),
    ("p_expr.cu", "#include \"kernel.cu\"\n", "REFUSED"),
])
def test_the_recon_tree_scans_probe_files_after_taking_their_own_spellings_out(
        base: Path, name: str, text: str, want: str) -> None:
    _, ref, _ = new_card(base)
    (ref / "notes").mkdir(exist_ok=True)
    (ref / "notes" / name).write_text(text)
    assert _guard(ref, "python3 notes/r1.py") == want, (name, text)


def test_the_loops_own_jobs_pass_both_guards_with_notes_seeded_as_the_loop_seeds_them(
        base: Path) -> None:
    """Production shape: the guards scan notes/, and the loop seeds it — the pricing toolkit
    into the student's tree (a clone of the card) and into the recon tree (also the ref-pricing
    tree), the recon toolkit and the shared blackbox-guard.sh into the recon tree. Every job the
    loop itself sends must pass; a probe that reaches for the reference must not."""
    from types import SimpleNamespace
    from forgekernel.loop import driver
    card, ref, module = new_card(base)
    stub = SimpleNamespace(commit=lambda *a, **k: None)
    driver._seed_pricing_toolkits(card, stub)
    driver._seed_pricing_toolkits(ref, stub)
    driver._seed_sass_tools(SimpleNamespace(recon_dir=ref, oracle_mode="blackbox"), stub)
    assert driver.BLACKBOX_GUARD_NAME == binref.SEEDED_GUARD and (ref / "notes" / binref.SEEDED_GUARD).is_file()
    loop_cmds = [c.removeprefix("cmd:") for c in (
        driver._ORCH_CENSUS_CMD, driver._LEVER_PRICING_CMD, driver._TIMELINE_CMD,
        driver._JOB_ENV_CHECK_CMD, driver._STALL_PRICE_CMD)]
    # a group card's other meta workloads: the same jobs, one workload each
    loop_cmds += [c.removeprefix("cmd:") + " --workload wl1" for c in (
        driver._LEVER_PRICING_CMD, driver._TIMELINE_CMD, driver._STALL_PRICE_CMD)]
    for cmd in loop_cmds:
        assert _guard(card, cmd) == "ALLOWED", ("card", cmd)
    for cmd in (driver._STALL_PRICE_CMD.removeprefix("cmd:") + " --app python3 binref.py ncu",
                driver._STALL_PRICE_CMD.removeprefix("cmd:") + " --workload wl1 --app python3 binref.py ncu",
                "python3 notes/r1.py"):
        assert _guard(ref, cmd) == "ALLOWED", ("recon tree", cmd)
    (card / "notes" / "r1.py").write_text("import glob\nprint(open(glob.glob('ref_bin/*.so')[0], 'rb').read(64))\n")
    assert _guard(card, "python3 notes/r1.py") == "REFUSED"
    (card / "notes" / "r1.py").write_text(f"import {module}\n")
    assert _guard(card, "python3 notes/r1.py") == "REFUSED"
    (ref / "notes" / "r2.sh").write_text("cat src/kernel.cu\n")
    assert _guard(ref, "bash notes/r2.sh") == "REFUSED"


@pytest.mark.parametrize("over, why", [
    ({"spec": {"languages": ["triton"], "entry_point": "k.py::run"}}, "closed_ref"),
    ({"definition": "002_other"}, "the card for"),
    ({"sources": [{"path": "kernel.cu", "content": "PYBIND11_MODULE(my_mod, m) {}"}]}, "TORCH_EXTENSION_NAME"),
])
def test_unusable_solutions_are_refused(base: Path, over, why) -> None:
    errs = binref.solution_problems(solution(**over), "rmsnorm")
    assert any(why in e for e in errs), errs


def test_install_commits_nothing_without_git_and_records_provenance(base: Path) -> None:
    card, _, module = new_card(base)
    so = base / "built.so"
    so.write_bytes(b"\x7fELF fake")
    dst = binref.install(card, so, {"image": "img:1"}, commit=False)
    assert dst == card / "ref_bin" / f"{module}.so" and dst.read_bytes() == so.read_bytes()
    prov = (card / "ref_bin" / "PROVENANCE.txt").read_text()
    assert hashlib.sha256(so.read_bytes()).hexdigest() in prov and "img:1" in prov


def test_install_commits_the_same_binary_into_card_and_recon_tree(base: Path, monkeypatch) -> None:
    card, ref, module = new_card(base)
    for tree in (card, ref):
        subprocess.run(["git", "init", "-q", str(tree)], check=True)
        subprocess.run(["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t", "commit",
                        "-q", "--allow-empty", "-m", "init"], check=True)
    for k, v in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@t"),
                 ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t@t")):
        monkeypatch.setenv(k, v)
    so = base / "built.so"
    so.write_bytes(b"\x7fELF fake")
    # relative paths, as the CLI is usually called: the operators container of the run
    monkeypatch.chdir(card.parent)
    binref.install(Path(card.name), so, commit=True, ref_tree=Path(ref.name))
    for tree in (card, ref):  # ref_bin/ and *.so are gitignored in the recon tree: forced in
        files = subprocess.run(["git", "-C", str(tree), "ls-files", "ref_bin"], capture_output=True,
                               text=True, check=True).stdout.split()
        assert files == ["ref_bin/PROVENANCE.txt", f"ref_bin/{module}.so"]
        assert (tree / "ref_bin" / f"{module}.so").read_bytes() == so.read_bytes()


def test_a_failed_build_leaves_no_staged_source_behind(base: Path, tmp_path: Path) -> None:
    sol = solution(sources=[{"path": p, "content": c} for p, c in SPLICE_SOURCES])
    sol["spec"]["entry_point"] = "splice_dispatch.cu::run"
    _, ref, module = new_card(base, sol)
    (tmp_path / "fake").mkdir(exist_ok=True)
    (tmp_path / "fake" / "sitecustomize.py").write_text(FAKE_TORCH.replace(
        'def load(**kw):', 'def load(**kw):\n    raise RuntimeError("nvcc: error")'))
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "fake"), "LOAD_LOG": str(tmp_path / "l.json"),
           "BINREF_BUILD_DIR": str(tmp_path / "build")}
    r = subprocess.run([sys.executable, "binref.py", "build"], cwd=ref, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "nvcc: error" in r.stderr
    assert not (tmp_path / "build" / module).exists()
