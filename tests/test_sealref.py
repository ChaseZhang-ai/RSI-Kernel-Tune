"""The sealed SOURCE reference (rsi_kernel_tune.sealref) on a generated task card: the card,
the recon tree, the key, the fence, the guards."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from rsi_kernel_tune import sealref

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_EVAL = REPO_ROOT / "src" / "rsi_kernel_tune" / "card_template" / "eval"
KERNEL = ("from launch_helpers import launch_the_scaled_product_kernel\n\n\n"
          "def run(hidden_states, weight, output):\n"
          "    launch_the_scaled_product_kernel(hidden_states, weight, output)\n")
HELPERS = ("def launch_the_scaled_product_kernel(hidden_states, weight, output):\n"
           "    output.append(('secret_tiling_choice_for_the_product', hidden_states, weight))\n")

DESCRIPTION = "y = x / sqrt(mean(x^2) + 1e-6) * w over the last axis\n"
SHAPES = [{"name": "small", "axes": {"m": 4, "n": 16}},
          {"name": "large", "axes": {"m": 64, "n": 16}}]
TASK_TOML = ('[task]\nlanguage = "cutedsl"\narch = "sm100"\n'
             '[reference]\nrun = "reference/run.py:run"\nblackbox = true\n')
REF_PY = "import torch\n\ndef run(x, w):\n    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w\n"
DEFINITION = {"inputs": {"x": {"shape": ["m", "n"], "dtype": "float32"},
                         "w": {"shape": ["n"], "dtype": "float32"}},
              "outputs": {"y": {"shape": ["m", "n"], "dtype": "float32"}}}
JUDGE = ("import torch\n\n"
         "def run(x, w):\n"
         "    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w\n")
GROUPS = {s["name"]: [{"name": "plain", "intent": "randn", "seed": 1}] for s in SHAPES}


def solution(**over) -> dict:
    sol = {"name": "rms_triton", "definition": "rmsnorm", "author": "t",
           "spec": {"languages": ["triton"], "target_hardware": ["B200"],
                    "entry_point": "kernel.py::run",
                    "dependencies": ["triton"], "destination_passing_style": True},
           "sources": [{"path": "kernel.py", "content": KERNEL},
                       {"path": "launch_helpers.py", "content": HELPERS}]}
    sol.update(over)
    return sol


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("RSI-KERNEL-TUNE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("RSI-KERNEL-TUNE_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("RSI-KERNEL-TUNE_OPERATORS_DIR", raising=False)
    monkeypatch.setenv("RKT_DELIVERY", "local")
    monkeypatch.setenv("RKT_GPU", "0")
    return tmp_path


@pytest.fixture
def base(repo: Path) -> Path:
    shutil.copytree(REPO_ROOT / "meta", repo / "meta")
    shutil.copytree(REPO_ROOT / "src" / "rsi_kernel_tune" / "card_template",
                    repo / "src" / "rsi_kernel_tune" / "card_template")
    return repo


def new_card(base: Path, sol: dict | None = None, *, git: bool = False,
             definition: str = "rmsnorm") -> tuple[Path, Path, str]:
    """A generated task card (taskcard.write_card) with the reference sealed into it:
    sealref.write_trees, the loop.toml of a blackbox run, the recon tree next to it."""
    from rsi_kernel_tune import judge as judge_mod, taskcard
    from rsi_kernel_tune.localenv import load_machine
    from rsi_kernel_tune.runs import new_run
    from rsi_kernel_tune.task import load_task
    sol = sol or solution()
    d = base / "tasks" / "rmsnorm"
    (d / "reference").mkdir(parents=True, exist_ok=True)
    (d / "task.toml").write_text(TASK_TOML)
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
    errs = sealref.solution_problems(sol, definition, TEMPLATE_EVAL)
    assert not errs, errs
    module = sealref.module_name(sol)
    impl = sealref.reference_impl(sol, module)
    taskcard.update_problem(op_dir, reference_impl=impl, time_reference=True)
    deny = [str(base / "solution"), "/elsewhere/plain"]
    taskcard.write_loop_toml(run, task, op_dir, deny=deny, ncu=True)
    ref_dir = run.operators_dir / taskcard.ref_tree(task.name)
    sealref.write_trees(op_dir, ref_dir, sol, module, TEMPLATE_EVAL,
                        kernel_file=taskcard.kernel_file(task))
    if git:
        for tree in (op_dir, ref_dir):
            subprocess.run(["git", "init", "-q"], cwd=tree, check=True)
            subprocess.run(["git", "-C", str(tree), "add", "-A"], check=True)
            subprocess.run(["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t",
                            "commit", "-q", "-m", "init"], check=True)
    return op_dir, ref_dir, module


def _closed_ref():
    spec = importlib.util.spec_from_file_location("_test_sealref_closed_ref", TEMPLATE_EVAL / "closed_ref.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _key(tree: Path) -> str:
    import tomllib
    return tomllib.loads((tree / "submit.local.toml").read_text())["env"]["RKT_REF_KEY"]


def test_card_times_a_sealed_source_reference(base: Path) -> None:
    card, ref, module = new_card(base)
    problem = json.loads((card / "eval" / "problem" / "problem.json").read_text())
    assert problem["reference_impl"] == {
        "kind": "sealed", "name": f"sealed source reference {module}",
        "path": "../../ref_bin/ref_payload.bin", "module": module, "entry": "run", "dps": True,
        "key_env": "RKT_REF_KEY", "languages": ["triton"]}
    assert problem["time_reference"] is True            # a real kernel: timed in-job
    assert (card / "eval" / "closed_ref.py").read_text() == (TEMPLATE_EVAL / "closed_ref.py").read_text()
    loop = tomllib_load(card / "loop.toml")["loop"]
    assert loop["oracle_mode"] == "blackbox" and loop["recon_op"] == "_rmsnormref"
    assert loop["ref_mode"] == "sealref" and loop["ref_price_app"] == ["python3", "sealref.py", "ncu"]
    assert str(base / "solution") in loop["ref_source_paths"] \
        and "/elsewhere/plain" in loop["ref_source_paths"]
    assert "loop.toml" in (card / ".gitignore").read_text().split()


def tomllib_load(p: Path) -> dict:
    import tomllib
    with open(p, "rb") as fh:
        return tomllib.load(fh)


def test_the_payload_opens_with_the_key_of_either_tree_and_no_plaintext_is_written(
        base: Path) -> None:
    card, ref, _ = new_card(base)
    cr = _closed_ref()
    payload = (card / "ref_bin" / "ref_payload.bin").read_bytes()
    assert (ref / "ref_bin" / "ref_payload.bin").read_bytes() == payload
    key = _key(card)
    assert _key(ref) == key and len(key) == 64
    assert cr.unseal_bundle(payload, key)["files"] == {"kernel.py": KERNEL, "launch_helpers.py": HELPERS}
    with pytest.raises(PermissionError):
        cr.unseal_bundle(payload)
    for tree in (card, ref):
        assert cr.check_tree(str(tree), payload, key) == []
        text = "\n".join(p.read_text(errors="replace") for p in tree.rglob("*") if p.is_file())
        assert "secret_tiling_choice" not in text and "launch_the_scaled_product_kernel" not in text
    meta = json.loads((ref / "sealref.json").read_text())
    assert meta["module"] == sealref.module_name(solution()) and meta["files"] == 2
    assert meta["payload_sha256"] == cr.payload_digest(payload)
    assert (ref / "sealref.py").read_text() == sealref.TOOL.read_text()
    assert (ref / "eval" / "benchkit.py").read_text() == (card / "eval" / "benchkit.py").read_text()
    assert 'ncu_window(which="ref")' in (ref / "eval" / "ncu_run.py").read_text()


def test_the_key_is_gitignored_in_both_trees_and_the_payload_committed(
        base: Path, monkeypatch) -> None:
    for k, v in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t"),
                 ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t")):
        monkeypatch.setenv(k, v)
    card, ref, _ = new_card(base, git=True)
    for tree in (card, ref):
        files = subprocess.run(["git", "-C", str(tree), "ls-files"], capture_output=True, text=True,
                               check=True).stdout.split()
        assert "ref_bin/ref_payload.bin" in files and "submit.local.toml" not in files
        assert (tree / "submit.local.toml").is_file()
    from rsi_kernel_tune.submit.config import load_config
    for tree in (card, ref):
        assert load_config(tree).env["RKT_REF_KEY"] == _key(card)


def test_the_agent_policy_denies_both_keys(base: Path, monkeypatch) -> None:
    from rsi_kernel_tune.loop import agent_policy
    card, ref, _ = new_card(base)
    monkeypatch.setenv("RSI-KERNEL-TUNE_OPERATORS_DIR", str(card.parent))
    secrets = agent_policy.secret_files_for(base, "rmsnorm", [ref])
    assert card / "submit.local.toml" in secrets and ref / "submit.local.toml" in secrets


def test_recon_tree_gpu_entry_and_fresh_start(base: Path) -> None:
    card, ref, _ = new_card(base)
    for tree in (card, ref):
        assert subprocess.run(["bash", "-n", str(tree / "gpu_entry.sh")]).returncode == 0
        assert (tree / "gpu_entry.sh").read_text().count("BLACKBOX_FORBID=") == 1
    entry = (ref / "gpu_entry.sh").read_text()
    for tree in (card, ref):   # a Triton reference needs the image's fbtriton launch.h fix
        assert "backends/nvidia/launch.h" in (tree / "gpu_entry.sh").read_text()
    for task in ("check)", "crosscheck)", "bench|ref)", "ncu)", "sealref)", "cmd)"):
        assert f"    {task}" in entry
    assert "binref" not in entry.split('case "$TASK" in')[-1]
    from rsi_kernel_tune.loop.cli import fresh_start_contamination
    assert fresh_start_contamination(card, card / "milestone.md", card / "student" / "status.jsonl",
                                     card / "meta_status.jsonl", ref, evidence_dir=card / "evidence") == []


def _guard(tree: Path, cmd: str) -> str:
    script = ("DEF=$(awk '/^BLACKBOX_FORBID=/{b=\"\";on=1} on{b=b $0 \"\\n\"} on&&/^}/{on=0;l=b} "
              "END{printf \"%s\", l}' gpu_entry.sh); eval \"$DEF\"; (blackbox_guard \"$1\"; echo ALLOWED) | tail -1")
    r = subprocess.run(["bash", "-c", script, "guard", cmd], cwd=tree, capture_output=True, text=True)
    return "ALLOWED" if r.stdout.strip() == "ALLOWED" else "REFUSED"


def test_guards(base: Path) -> None:
    card, ref, module = new_card(base)
    refused_both = ["python3 eval/closed_ref.py digest ref_bin/ref_payload.bin", "echo $RKT_REF_KEY",
                    "cat /proc/self/environ", "printenv", "python3 -c 'import os; print(os.environ)'",
                    "cat submit.local.toml | grep _REF_KEY", "ls ~/.triton/cache", "cat /tmp/x.ttgir",
                    "TRITON_KERNEL_DUMP=1 python3 eval/bench.py", "python3 -c 'import linecache'",
                    f"python3 -c 'import {module}'", "env", "bash -c set", "compgen -v"]
    for cmd in refused_both:
        assert _guard(card, cmd) == "REFUSED", ("card", cmd)
        assert _guard(ref, cmd) == "REFUSED", ("recon tree", cmd)
    for cmd, want in [("python3 eval/bench.py --mode all", "ALLOWED"), ("python3 notes/r1.py", "ALLOWED"),
                      ("cuobjdump -sass /tmp/a.cubin", "REFUSED"), ("nm -D x.so", "REFUSED")]:
        assert _guard(card, cmd) == want, ("card", cmd)
    for cmd, want in [("python3 sealref.py ncu", "ALLOWED"), ("python3 sealref.py crosscheck", "ALLOWED"),
                      ("ncu --print-source sass python3 eval/ncu_run.py", "ALLOWED"),
                      ("ncu --print-source python python3 eval/ncu_run.py", "REFUSED"),
                      ("ncu --import-source yes python3 eval/ncu_run.py", "REFUSED"),
                      ("python3 notes/r1.py", "ALLOWED")]:
        assert _guard(ref, cmd) == want, ("recon tree", cmd)
    # the card also scans its kernel.py: a `cmd` that imports it runs it outside the fence
    (card / "kernel.py").write_text("import zlib\nKEY = open('submit.local.toml').read()\n"
                                    "P = open('ref_bin/ref_payload.bin').read()\n")
    assert _guard(card, "python3 -c 'import kernel'") == "REFUSED"


def test_the_loops_own_jobs_pass_both_guards_with_notes_seeded_as_the_loop_seeds_them(
        base: Path) -> None:
    """As binref's: the loop seeds the pricing toolkit into the student's tree and the
    recon tree, the recon toolkit and blackbox-guard.sh into the recon tree; every job
    the loop itself sends must pass, a probe reaching for the reference must not."""
    from types import SimpleNamespace
    from rsi_kernel_tune import binref
    from rsi_kernel_tune.loop import driver
    card, ref, module = new_card(base)
    stub = SimpleNamespace(commit=lambda *a, **k: None)
    driver._seed_pricing_toolkits(card, stub)
    driver._seed_pricing_toolkits(ref, stub)
    driver._seed_sass_tools(SimpleNamespace(recon_dir=ref, oracle_mode="blackbox"), stub)
    assert (ref / "notes" / binref.SEEDED_GUARD).is_file()
    for cmd in (driver._ORCH_CENSUS_CMD, driver._LEVER_PRICING_CMD, driver._TIMELINE_CMD,
                driver._JOB_ENV_CHECK_CMD, driver._STALL_PRICE_CMD,
                # a group card's other meta workloads: the same jobs, one workload each
                driver._LEVER_PRICING_CMD + " --workload wl1",
                driver._TIMELINE_CMD + " --workload wl1", driver._STALL_PRICE_CMD + " --workload wl1"):
        assert _guard(card, cmd.removeprefix("cmd:")) == "ALLOWED", ("card", cmd)
    for cmd in (driver._STALL_PRICE_CMD.removeprefix("cmd:") + " --app python3 sealref.py ncu",
                driver._STALL_PRICE_CMD.removeprefix("cmd:") + " --workload wl1 --app python3 sealref.py ncu",
                "python3 notes/r1.py"):
        assert _guard(ref, cmd) == "ALLOWED", ("recon tree", cmd)
    (card / "notes" / "r1.py").write_text("print(open('ref_bin/ref_payload.bin').read(64))\n")
    assert _guard(card, "python3 notes/r1.py") == "REFUSED"
    (ref / "notes" / "r2.py").write_text("import linecache\nprint(linecache.cache)\n")
    assert _guard(ref, "python3 notes/r2.py") == "REFUSED"


def test_the_generic_routes_to_the_key_are_words_of_the_cmd_guard(base: Path) -> None:
    card, ref, _ = new_card(base)
    for cmd in ("python3 -c \"import os; print(os.getenv('RKT_' + 'REF' + '_KEY'))\"",
                "cat /proc/$PPID/cmdline", "cat /proc/self/e?viron", "cat ref_bin/ref_payload.key"):
        assert _guard(card, cmd) == "REFUSED", ("card", cmd)
        assert _guard(ref, cmd) == "REFUSED", ("recon tree", cmd)


FAKE_TIMEOUT = '#!/bin/bash\n# timeout --signal=KILL <s> CMD...\nshift 2\nexec "$@"\n'
DUMP_ENV = ("import json, os, sys\n"
            "json.dump({'env': dict(os.environ), 'argv': sys.argv}, open(sys.argv[1], 'w'))\n")


@pytest.mark.parametrize("tree_name", ["card", "recon"])
def test_the_key_leaves_the_job_environment_before_any_task_runs(base: Path,
                                                                 tree_name) -> None:
    """The key arrives in the job environment; gpu_entry.sh moves it into the key file beside
    the payload and re-executes itself without it, so a `cmd` job (or anything it starts)
    inherits none of it — while the loader still finds it."""
    card, ref, _ = new_card(base)
    tree = card if tree_name == "card" else ref
    key = _key(tree)
    bin_dir = tree.parent / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    for name, text in (("timeout", FAKE_TIMEOUT), ("nvidia-smi", "#!/bin/bash\necho FAKE\n")):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    (tree / "notes").mkdir(exist_ok=True)
    (tree / "notes" / "dump.py").write_text(DUMP_ENV)
    out = tree.parent / "dump.json"
    env = {k: v for k, v in os.environ.items()
           if k not in ("LB_SM_CLOCK_MHZ", "LB_MEM_CLOCK_MHZ", "LB_REQUIRE_CLOCK_LOCK")}
    env.update({"PATH": f"{bin_dir}:{os.environ['PATH']}", "LB_SM_CLOCK_MHZ": "", "LB_CLOCK_SETTLE_S": "0",
                "TASK_TIMEOUT": "60", "RKT_REF_KEY": key})
    proc = subprocess.run(["bash", str(tree / "gpu_entry.sh"), "cmd", f"python3 notes/dump.py {out}"],
                          cwd=tree, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    seen = json.loads(out.read_text())
    assert "RKT_REF_KEY" not in seen["env"] and not any(key in v for v in seen["env"].values())
    assert key not in " ".join(seen["argv"]) and key not in proc.stdout + proc.stderr
    kf = tree / sealref.KEY_FILE
    assert kf.read_text().strip() == key and (kf.stat().st_mode & 0o777) == 0o600
    cr = _closed_ref()
    assert cr.key_file(str(tree / sealref.PAYLOAD)) == str(kf)
    assert cr.key_from_file(str(tree / sealref.PAYLOAD)) == key
    assert sealref.KEY_FILE in (tree / ".gitignore").read_text().split()


@pytest.mark.parametrize("over, why", [
    ({"spec": {"languages": ["cuda_cpp"], "entry_point": "k.cu::run"}}, "binary"),
    ({"definition": "002_other"}, "the card for"),
    ({"spec": {"languages": ["triton"], "entry_point": "kernel.py"}}, "<file>.py::<function>"),
    ({"sources": [{"path": "../kernel.py", "content": "x"}]}, "inside the bundle"),
])
def test_unusable_solutions_are_refused(base: Path, over, why) -> None:
    errs = sealref.solution_problems(solution(**over), "rmsnorm", TEMPLATE_EVAL)
    assert any(why in e for e in errs), errs


# --- the job side: the card's bench engine loads the reference, the fence -------------


class _Any:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return _Any()
    def __call__(self, *a, **k): return _Any()
    def __iter__(self): return iter(())
    def __mro_entries__(self, bases): return (object,)
    def __hash__(self): return id(self)
    def __or__(self, o): return _Any()


@pytest.fixture
def card_lb(base: Path, monkeypatch):
    """The card's eval/benchkit.py, imported with a stand-in torch (the fence and the
    loader need none)."""
    card, ref, module = new_card(base)
    if importlib.util.find_spec("torch") is None:
        fake = types.ModuleType("torch")
        fake.__getattr__ = lambda n: _Any()
        # a stable version string: the CUDA loader keys its build on it, and a fresh
        # _Any() per read would key two loads of the same source differently
        fake.__version__ = "0.0+fake"
        monkeypatch.setitem(sys.modules, "torch", fake)
    spec = importlib.util.spec_from_file_location("_test_card_bk", card / "eval" / "benchkit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    problem = json.loads((card / "eval" / "problem" / "problem.json").read_text())
    problem["_dir"] = str(card / "eval" / "problem")
    yield mod, problem, card, module
    for name in [m for m in sys.modules if m.startswith(module)]:
        sys.modules.pop(name, None)
    sys.meta_path[:] = [f for f in sys.meta_path if getattr(f, "pkg", None) != module]


def test_the_bench_loads_the_sealed_reference_with_the_job_key_and_drops_it(card_lb, monkeypatch) -> None:
    lbm, problem, card, module = card_lb
    monkeypatch.delenv("RKT_REF_KEY", raising=False)
    with pytest.raises(PermissionError, match="RKT_REF_KEY"):
        lbm._impl(problem, "ref")
    monkeypatch.setenv("RKT_REF_KEY", _key(card))
    fn, dps = lbm._impl(problem, "ref")
    import os
    assert dps is True and "RKT_REF_KEY" not in os.environ   # decoded: the key is gone
    out = []
    fn("x", "w", out)
    assert out == [("secret_tiling_choice_for_the_product", "x", "w")]
    assert f"{module}.launch_helpers" in sys.modules and "launch_helpers" not in sys.modules


@pytest.mark.parametrize("src, fenced", [
    ("import torch\n\ndef run(hidden_states, weight, output):\n    output.copy_(hidden_states * weight)\n", False),
    ("import closed_ref\n", True),
    ("P = open('../ref_bin/ref_payload.bin').read()\n", True),
    ("import os\nK = os.environ.get('RKT_' + 'REF_KEY')\n", True),
    ("import linecache\n", True),
    ("import sys\nsys.settrace(lambda *a: None)\n", True),
    ("def run(*a):\n    return run.__globals__\n", True),
    ("import glob\nglob.glob('/root/.triton/cache/*/*.ttgir')\n", True),
    ("open('/proc/self/environ').read()\n", True),
    ("# closed_ref in a comment is not a reach\ndef run(*a):\n    pass\n", False),
])
def test_the_fence(card_lb, src, fenced) -> None:
    lbm, problem, card, module = card_lb
    (card / "kernel.py").write_text(src)
    assert bool(lbm.kernel_fence(problem)) is fenced
    (card / "kernel.py").write_text(f"import {module}\n")
    assert lbm.kernel_fence(problem)


def test_the_timeline_probes_preamble_passes_the_sealed_fence(card_lb) -> None:
    lbm, problem, card, module = card_lb
    from rsi_kernel_tune.skills import timeline_probe
    pre = timeline_probe.stamped_preamble()
    (card / "kernel.py").write_text(pre + "\n\ndef run(*a):\n    pass\n")
    assert lbm.kernel_fence(problem) is None
