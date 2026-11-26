"""The bench runner: the loop's job path behind a temporary run, with a fake GPU.

No test here starts a real GPU job or a real agent: ``RealBackend`` is replaced by a log
script, and everything else (the card install, the workspace, the git commit/push, the
cleanup) is the real thing -- those are local and fast.
"""

import importlib.util
import json
import subprocess

import pytest

from rsi_kernel_tune import taskcard
from rsi_kernel_tune.benchx import runner
from rsi_kernel_tune.localenv import Machine
from rsi_kernel_tune.loop import cli as loop_cli
from rsi_kernel_tune.leaderboard import sol_score

MACHINE = Machine(delivery="local", gpu="0")


# -- fixtures ---------------------------------------------------------------------------------


def problem(group=2, dps=True) -> dict:
    entries = [{"index": i, "name": f"wl{i}",
                "workload": {"uuid": f"u{i}", "axes": {"m": 128 * (i + 1)}},
                "sol_board": {"kernel_id": "k", "workload_index": i,
                              "t_b_ms": 1.0 * (i + 1), "t_sol_ms": 0.2 * (i + 1),
                              "source": "test"}}
               for i in range(group)]
    p = {
        "schema": 1, "suite": "sol-execbench",
        "definition": {"name": "demo_op", "inputs": {}, "outputs": {},
                       "reference": "def run(x):\n    return x"},
        "workload": entries[0]["workload"], "dps": dps, "kernel_file": "kernel.py",
        "entry": "run", "reference_impl": {"kind": "definition"},
        "time_reference": False, "sol_board": entries[0]["sol_board"],
    }
    if group > 1:
        p["group"] = entries
    return p


def eval_log(p: dict, t_ms: float = 0.5, *, passed=True, error=None) -> str:
    names = [str(g["name"]) for g in p.get("group") or ()] or ["u0"]
    if len(names) > 1:
        blocks = [{"name": n, "device_median_ms": t_ms, "bwd_median_ms": t_ms * 1.2,
                   "leaderboard": {"sol_board": {
                       "t_b_ms": 1.0 * (i + 1), "t_sol_ms": 0.2 * (i + 1),
                       "sol_score": round(sol_score(t_ms, 1.0 * (i + 1),
                                                    0.2 * (i + 1)), 4)}}}
                  for i, n in enumerate(names)]
        scores = {b["name"]: b["leaderboard"]["sol_board"]["sol_score"] for b in blocks}
        perf = {"device_median_ms": t_ms, "n_workloads": len(blocks),
                "device_window_basis": "cupti_kernel_span", "workloads": blocks,
                "leaderboard": {"suite": "sol-execbench", "clock": "locked sm=1500",
                                "protocol": {}, "sol_board": {
                                    "sol_score_mean": round(
                                        sum(scores.values()) / len(scores), 4),
                                    "sol_score": scores}}}
        corr = {"pass": passed, "status": "PASSED" if passed else "FAILED",
                "n_workloads": len(names),
                "workloads": [{"name": n, "status": "PASSED" if passed else "FAILED"}
                              for n in names]}
    else:
        perf = {"device_median_ms": t_ms, "bwd_median_ms": t_ms * 1.2,
                "device_window_basis": "cupti_kernel_span",
                "leaderboard": {"suite": "sol-execbench", "clock": "locked sm=1500",
                                "protocol": {}, "sol_board": {
                                    "t_b_ms": 1.0, "t_sol_ms": 0.2,
                                    "sol_score": round(sol_score(t_ms, 1.0, 0.2), 4)}}}
        corr = {"pass": passed, "status": "PASSED" if passed else "FAILED",
                "n_workloads": 1}
    if error:
        corr["error"] = error
        perf = None
    return ("=== gpu_entry.sh (demo_op) ===\n  task: all\n"
            "\n=== JSON ===\n"
            + json.dumps({"correctness": corr, "performance": perf}) + "\n")


class FakeBackend:
    """One log per mode (the cold probe's line included), and a call record."""

    calls: list = []
    logs: dict = {}
    fail_on: dict = {}
    job_counter = 100

    def __init__(self, op, workspace, agent_cmd, git_name, git_email, **kwargs):
        self.op = op
        self.workspace = workspace
        FakeBackend.last_job_id = ""

    def submit_gpu(self, branch, mode):
        FakeBackend.calls.append((self.op, str(branch), mode))
        if mode in FakeBackend.fail_on:
            raise FakeBackend.fail_on[mode]
        FakeBackend.job_counter += 1
        FakeBackend.last_job_id = str(FakeBackend.job_counter)
        if mode.startswith("cmd:"):
            return ("=== gpu_entry.sh (demo_op) ===\n  task: cmd\n"
                    "[benchx] COLD_START_S 12.500 (card demo_op)\n")
        return FakeBackend.logs.get(mode, "")


@pytest.fixture
def gpu(monkeypatch, tmp_path):
    FakeBackend.calls = []
    FakeBackend.logs = {}
    FakeBackend.fail_on = {}
    FakeBackend.job_counter = 100
    monkeypatch.setattr(loop_cli, "RealBackend", FakeBackend)
    monkeypatch.setenv("RSI_KERNEL_TUNE_RUNS_DIR", str(tmp_path / "artifacts"))
    return FakeBackend


def make_task(tmp_path, p=None) -> "object":
    """A board task under a throwaway repo layout: task.toml + leaderboard/card."""
    p = p or problem()
    d = tmp_path / "tasks" / "demo_op"
    card = d / "leaderboard" / "card"
    (card / "eval" / "leaderboard").mkdir(parents=True)
    (d / "task.toml").write_text(
        '[task]\nlanguage = "cutedsl"\narch = "sm100"\n\n'
        '[leaderboard]\nsuite = "sol-execbench"\ncard = "leaderboard/card"\n',
        encoding="utf-8")
    (card / "eval" / "leaderboard" / "problem.json").write_text(
        json.dumps(p, indent=2), encoding="utf-8")
    (card / "kernel.py").write_text("def run(*args):\n    pass\n", encoding="utf-8")
    return d


def candidate_file(tmp_path, body="def run(*args):\n    return None\n") -> "object":
    p = tmp_path / "cand.py"
    p.write_text(body, encoding="utf-8")
    return p


def bench_runs(tmp_path) -> list:
    root = tmp_path / "artifacts"
    return sorted(root.glob("benchx_*")) if root.is_dir() else []


def card_git(run_dir, *args) -> str:
    op = run_dir / "operators" / "demo_op"
    proc = subprocess.run(["git", "-C", str(op), *args], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


# -- the order of operations ------------------------------------------------------------------


def test_the_bench_is_install_clone_commit_probe_eval_clean(monkeypatch, tmp_path, gpu):
    events: list = []
    real_install = taskcard.install_board_trees
    real_clone = taskcard.clone_workspace
    real_commit = runner.commit_candidate

    def install(run, task, machine):
        events.append(("install", task.name, machine.delivery))
        return real_install(run, task, machine)

    def clone(op_dir, ws):
        events.append(("clone", op_dir.name))
        return real_clone(op_dir, ws)

    def commit(ws, message, branch=taskcard.BRANCH):
        sha = real_commit(ws, message, branch)
        events.append(("commit", message.split(":")[0]))
        return sha

    monkeypatch.setattr(runner.taskcard, "install_board_trees", install)
    monkeypatch.setattr(runner.taskcard, "clone_workspace", clone)
    monkeypatch.setattr(runner, "commit_candidate", commit)

    taskdir = make_task(tmp_path)
    gpu.logs["all"] = eval_log(problem())
    task, p = runner.load_board(taskdir)
    result = runner.CardBench(machine=MACHINE, repeat=2).bench_task(
        task, p, candidate_file(tmp_path), workloads="all")

    assert [e[0] for e in events] == ["install", "clone", "commit"]
    assert gpu.calls == [("demo_op", "main", runner.COLD_PROBE_MODE),
                         ("demo_op", "main", "all"),
                         ("demo_op", "main", "all")]
    assert result.correct and result.job == "102,103"   # 101 was the probe
    assert bench_runs(tmp_path) == []                   # cleaned, no --keep


def test_keep_keeps_the_run_and_the_card_branch_has_the_candidate(tmp_path, gpu):
    p = problem()
    body = "def run(*args):\n    return 'mine'\n"
    taskdir = make_task(tmp_path, p)
    gpu.logs["all"] = eval_log(p)
    task, prob = runner.load_board(taskdir)
    result = runner.CardBench(machine=MACHINE, keep=True).bench_task(
        task, prob, candidate_file(tmp_path, body))
    assert result.correct
    (run_dir,) = bench_runs(tmp_path)
    assert run_dir.name.startswith("benchx_")
    assert card_git(run_dir, "show", "main:kernel.py") == body
    probe = card_git(run_dir, "show", "main:benchx_coldstart.py")
    assert "[benchx] COLD_START_S" in probe
    assert json.loads((run_dir / "run.json").read_text())["kind"] == "benchx"
    assert result.integrity["commit"][:7] in card_git(run_dir, "log", "--oneline", "main")


def test_a_subset_trims_the_temp_card_and_says_so(tmp_path, gpu):
    p = problem(group=3)
    gpu.logs["all"] = eval_log(p)
    task, prob = runner.load_board(make_task(tmp_path, p))
    result = runner.CardBench(machine=MACHINE, keep=True).bench_task(
        task, prob, candidate_file(tmp_path), workloads="1")
    assert result.correct
    assert result.integrity["workload_subset"] == {"requested": ["wl1"],
                                                   "card_workloads": 3}
    assert [w.name for w in result.workloads] == ["wl1"]
    (run_dir,) = bench_runs(tmp_path)
    trimmed = json.loads(card_git(run_dir, "show", "main:eval/leaderboard/problem.json"))
    assert [g["name"] for g in trimmed["group"]] == ["wl1"]
    assert trimmed["workload"]["uuid"] == "u1"
    assert trimmed["sol_board"]["workload_index"] == 1


def test_cleanup_happens_when_the_job_path_fails(tmp_path, gpu):
    p = problem()
    gpu.fail_on["all"] = loop_cli.ClockLockRefusedError(
        "the board's clock lock did not hold on the GPU (3 refusals in a row)")
    task, prob = runner.load_board(make_task(tmp_path, p))
    result = runner.CardBench(machine=MACHINE).bench_task(
        task, prob, candidate_file(tmp_path))
    assert result.correct is False
    assert "clock lock" in result.error
    assert result.integrity["failure_kind"] == "ClockLockRefusedError"
    assert result.cold_start_s == 12.5       # the probe did run
    assert bench_runs(tmp_path) == []


def test_cleanup_happens_on_a_bench_error(monkeypatch, tmp_path, gpu):
    task, prob = runner.load_board(make_task(tmp_path, problem()))

    def broken(*a, **k):
        raise runner.BenchError("no such workload")

    monkeypatch.setattr(runner, "write_candidate", broken)
    with pytest.raises(runner.BenchError):
        runner.CardBench(machine=MACHINE).bench_task(task, prob, candidate_file(tmp_path))
    assert bench_runs(tmp_path) == []


# -- the candidate ----------------------------------------------------------------------------


def test_a_python_candidate_lands_at_the_cards_kernel_file(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    c = runner.write_candidate(ws, candidate_file(tmp_path), kernel_file="kernel.py",
                               problem=problem())
    assert c.dps_adapter == "" and c.files == ["kernel.py"]
    assert (ws / "kernel.py").read_text() == "def run(*args):\n    return None\n"


def test_a_value_returning_solution_under_a_dps_card_gets_an_adapter(
        tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    solution = {"spec": {"entry_point": "solution.py::run",
                         "destination_passing_style": False},
                "sources": [{"path": "solution.py",
                             "content": "def run(x):\n    return x * 2\n"}]}
    p = problem(dps=True)
    p["definition"]["outputs"] = {"y": {"dtype": "float32", "shape": None}}
    c = runner.write_candidate(ws, solution, kernel_file="kernel.py", problem=p)
    assert c.dps_adapter == "value-returning-to-dps"
    assert sorted(c.files) == ["_benchx_entry.py", "kernel.py"]
    adapter = (ws / "kernel.py").read_text()
    compile(adapter, "kernel.py", "exec")
    assert "_OUTPUTS = ['y']" in adapter
    assert (ws / "_benchx_entry.py").read_text() == "def run(x):\n    return x * 2\n"
    # the adapter's copy semantics, on CPU torch: run(x) -> preallocated out. It is
    # imported as a module (what the job does with kernel.py), not executed as text.
    import torch

    monkeypatch.syspath_prepend(str(ws))
    spec = importlib.util.spec_from_file_location("_benchx_adapter_under_test",
                                                  ws / "kernel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    x = torch.arange(4, dtype=torch.float32)
    out = torch.empty(4, dtype=torch.float32)
    mod.run(x, out)
    assert torch.equal(out, x * 2)


def test_a_dps_solution_under_a_value_returning_card_gets_the_other_adapter(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    solution = {"spec": {"entry_point": "solution.py", "destination_passing_style": True},
                "sources": [{"path": "solution.py", "content": "def run(x, out):\n"
                                                                 "    out.copy_(x)\n"}]}
    c = runner.write_candidate(ws, solution, kernel_file="kernel.py",
                               problem=problem(dps=False))
    assert c.dps_adapter == "dps-to-value-returning"
    adapter = (ws / "kernel.py").read_text()
    compile(adapter, "kernel.py", "exec")
    assert "_lb.allocate_outputs(_PROBLEM, \"cuda\")" in adapter


def test_a_solution_json_on_disk_reads_like_a_dict(tmp_path):
    path = tmp_path / "solution.json"
    path.write_text(json.dumps({"spec": {"entry_point": "k.py::run"},
                                "sources": [{"path": "k.py", "content": "x = 1\n"}]}))
    entry, files, dps = runner.candidate_sources(path)
    assert (entry, files["k.py"], dps) == ("k.py", "x = 1\n", None)


def test_a_candidate_that_is_neither_py_nor_json_is_refused(tmp_path):
    p = tmp_path / "k.cu"
    p.write_text("// nothing")
    with pytest.raises(runner.BenchError, match="neither a .py file nor a solution JSON"):
        runner.candidate_sources(p)


# -- workloads --------------------------------------------------------------------------------


def test_select_workloads_parses_indices_ranges_and_names():
    names = ["wl0", "wl1", "wl2", "wl3"]
    assert runner.select_workloads("all", names) is None
    assert runner.select_workloads(None, names) is None
    assert runner.select_workloads("0,3", names) == ["wl0", "wl3"]
    assert runner.select_workloads("1-2", names) == ["wl1", "wl2"]
    assert runner.select_workloads("wl2", names) == ["wl2"]
    assert runner.select_workloads("wl1,1", names) == ["wl1"]


def test_select_workloads_refuses_bad_input():
    names = ["wl0", "wl1"]
    for spec in ("7", "0-9", "wl9", "2-1", ","):
        with pytest.raises(runner.BenchError):
            runner.select_workloads(spec, names)


# -- the target check -------------------------------------------------------------------------


def test_the_builtin_target_check_compiles_and_flags_features():
    check = runner.target_compile_check(
        "import triton\n@triton.jit\ndef k():\n    pass\n",
        kernel_file="kernel.py", language="cutedsl", arch="sm100", problem={})
    assert check["status"] == "ok" and check["engine"] == "builtin-syntax"
    assert "triton" in check["features"] and check["external"] == "unavailable"

    inline = runner.target_compile_check(
        "from torch.utils.cpp_extension import load_inline\n"
        "mod = load_inline(name='k', cuda_sources=['__global__ void f() {}'])\n",
        kernel_file="kernel.py", language="cutedsl", arch="sm100", problem={})
    assert "inline_cuda" in inline["features"]

    bad = runner.target_compile_check("def run(:\n", kernel_file="kernel.py",
                                      language="cutedsl", arch="sm100", problem={})
    assert bad["status"] == "failed" and "kernel.py:" in bad["error"]


def test_a_targethw_module_is_used_when_present_and_never_required(monkeypatch):
    import sys
    import types

    present = types.ModuleType("rsi_kernel_tune.targethw.rules")
    present.target_compile = lambda source, **kw: {"status": "ok", "arch": kw["arch"],
                                                   "compiled": True}
    present.confidence_hints = lambda source, **kw: ["fp4-bound", "sfu-bound"]
    monkeypatch.setitem(sys.modules, "rsi_kernel_tune.targethw.rules", present)
    integrity: dict = {}
    check = runner.target_compile_check("x = 1\n", kernel_file="kernel.py",
                                        language="cutedsl", arch="sm100", problem={})
    assert check["external"] == {"status": "ok", "arch": "sm100", "compiled": True}
    assert runner.confidence_hints("x = 1\n", kernel_file="kernel.py",
                                   language="cutedsl", arch="sm100",
                                   integrity=integrity) == ["dev-only", "fp4-bound",
                                                            "sfu-bound"]

    broken = types.ModuleType("rsi_kernel_tune.targethw.rules")

    def boom(source, **kw):
        raise RuntimeError("no nvcc")

    broken.target_compile = boom
    broken.confidence_hints = boom
    monkeypatch.setitem(sys.modules, "rsi_kernel_tune.targethw.rules", broken)
    check = runner.target_compile_check("x = 1\n", kernel_file="kernel.py",
                                        language="cutedsl", arch="sm100", problem={})
    assert check["external"]["status"] == "error" and "no nvcc" in check["external"]["error"]
    # a broken hint source must not fail the bench either
    assert runner.confidence_hints("x = 1\n", kernel_file="kernel.py", language="cutedsl",
                                   arch="sm100", integrity=integrity) == ["dev-only"]
    assert "no nvcc" in integrity["confidence_hints_note"]


# -- the card / the command -------------------------------------------------------------------


def test_a_plain_task_is_refused_with_a_clear_message():
    from rsi_kernel_tune.paths import repo_root

    with pytest.raises(runner.BenchError, match="not a leaderboard task"):
        runner.load_board(repo_root() / "tasks" / "rmsnorm")


def test_a_bare_card_directory_is_enough(tmp_path):
    p = problem()
    card = tmp_path / "card"
    (card / "eval" / "leaderboard").mkdir(parents=True)
    (card / "eval" / "leaderboard" / "problem.json").write_text(json.dumps(p))
    task, prob = runner.load_board(card)
    assert isinstance(task, runner.BareCardTask)
    assert task.name == "demo_op" and task.leaderboard.card == card
    assert runner.kernel_file_of(task, prob) == "kernel.py"


def test_target_mode_runs_nothing_and_writes_no_run(tmp_path, gpu):
    task, prob = runner.load_board(make_task(tmp_path))
    result = runner.CardBench(machine=MACHINE).bench_task(
        task, prob, candidate_file(tmp_path), mode="target")
    assert gpu.calls == []                                   # nothing ran
    assert not (tmp_path / "artifacts").exists()
    assert result.integrity["mode"] == "target"
    assert "not run" in result.integrity["verdict"]
    assert result.integrity["target_compile"]["status"] == "ok"


def test_the_command_help_and_refusal_routes():
    import pytest as _pytest

    from rsi_kernel_tune import __main__ as forge_main
    from rsi_kernel_tune.paths import repo_root

    assert forge_main.main(["bench", "--help"]) == 0
    for argv in (["bench", "compare", "--help"], ["bench", "best-of", "--help"]):
        with _pytest.raises(SystemExit) as stopped:      # argparse's own help exit
            forge_main.main(argv)
        assert stopped.value.code == 0
    assert forge_main.main(["bench", str(repo_root() / "tasks" / "rmsnorm"),
                            "--candidate", __file__]) == 2
