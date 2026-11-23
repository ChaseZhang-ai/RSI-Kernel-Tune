"""`campaign run`: the per-problem pipeline, stage by stage, with every tool faked.

The fakes are the seam the task file asks for: `tools` carries one callable per outside
world (bench, init, new, run, evolve, lib find/baselines/adapt, shapes compose,
cuda_to_inline, licence check, export check, board submit), so each test asserts the
arguments the campaign passed and the state it left behind -- never a GPU.
"""

from __future__ import annotations

import ast
import json
import threading
import time
from pathlib import Path

import pytest

import campaign_helpers as H
from forgekernel.campaign import compose, plan as P, report as report_mod, runner
from forgekernel.campaign import solscore, sync
from forgekernel.campaign import tools as tools_mod
from forgekernel.ext import campaign as cli
from forgekernel.task import load_task

KERNEL_A = """import torch


def run(input, weight, eps, output):
    output.copy_(input * weight)
"""
KERNEL_B = """import torch


def helper(x):
    return x


def run(input, weight, eps, output):
    output.copy_(input * weight * 2.0)
"""


def _plan(tmp_path, *, selection="069_rms_norm", **kw) -> dict:
    from forgekernel.campaign import targets as T
    snap = T.load_snapshot(H.SNAPSHOT)
    kw.setdefault("dataset", str(H.DATASET))
    plan = P.build_plan(snap, selection=selection, work_dir=tmp_path / "work",
                        root=tmp_path, **kw)
    return plan.as_json()


def _run_loop_with(tmp_path, kernels, readings, *, status_extra=None):
    """A `forge run` fake that leaves `kernels` committed and their readings in
    status.jsonl. `readings[i]` maps a workload name to its device ms."""
    def run_loop(argv, env=None):
        run_dir = Path(argv[-1])
        shas = H.commit_kernels(run_dir, kernels)
        rows = []
        for i, (sha, lat) in enumerate(zip(shas, readings)):
            rows.append({"iter": i, "verdict": "KEEP", "correctness": True, "sha": sha,
                         "workloads": [{"name": n, "device_median_ms": t}
                                       for n, t in lat.items()]})
        rows += list(status_extra or [])
        H.write_status(run_dir, rows)
        return 0
    return run_loop


def _seed_record(work: Path, kernel_id: int, rec: dict) -> None:
    """The board's kernels/<id> payload where `run` looks for it first (no network)."""
    d = Path(work) / runner.RECORDS_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{kernel_id}.json").write_text(json.dumps(rec), encoding="utf-8")


def _tools(tmp_path, *, rec, bench_table=None, cold=2.0, work=None, root=None, **kw):
    """Tools + the work dir seeded with the board record (so no test reaches the API)."""
    work = Path(work) if work else tmp_path / "work"
    if rec:
        _seed_record(work, rec["id"], rec)
    board = H.board_mapping(rec) if rec else None
    bench = H.FakeBench(bench_table or {"reference": {"wl0": 0.05, "wl1": 0.09, "wl2": 0.2}},
                        board=board, cold_start=cold)
    return H.recording_tools(root or tmp_path / f"repo-{work.name}", bench=bench,
                             records={H.L1_PROBLEM: rec} if rec else {}, **kw)


def test_full_pipeline_produces_a_submission(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    tools = _tools(
        tmp_path, rec=rec,
        bench_table={"reference": {"wl0": 0.05, "wl1": 0.09, "wl2": 0.2}},
        run_loop=_run_loop_with(tmp_path, [KERNEL_A, KERNEL_B],
                                [{"wl0": 0.03, "wl1": 0.04, "wl2": 0.06},
                                 {"wl0": 0.02, "wl1": 0.03, "wl2": 0.05}]))
    plan = _plan(tmp_path)
    states = runner.run_plan(plan, tools=tools, work_dir=tmp_path / "work")
    assert len(states) == 1 and states[0].status == "done", states[0].error
    st = states[0]
    kinds = [c[0] for c in tools.calls]
    assert kinds.count("init") == 1 and kinds.count("new") == 1 and kinds.count("run") == 1

    # the card was built with the board's record and every workload of the problem
    init_argv = next(c[1] for c in tools.calls if c[0] == "init")
    assert "--sol-board" in init_argv and "--no-sol-board" not in init_argv
    assert init_argv[init_argv.index("--workload") + 1] == "0-2"

    # the target is the best start's geomean over the step
    task = load_task(Path(st.task_dir))
    assert task.target.ms == pytest.approx(0.05 * 0.09 * 0.2 and
                                           solscore.geomean([0.05, 0.09, 0.2]) / 1.1)
    # the best version of the run is the exported one
    assert Path(st.export["path"]).is_file() and st.export["kind"] == "python"
    text = Path(st.export["path"]).read_text()
    tree = ast.parse(text)
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"run", "_fk_candidate_run", "_fk_reference_run"} <= names
    assert "2.0" in text, "the run's later (better) kernel should be the exported one"
    # the ledger holds the start reading and the compose
    events = [e["event"] for e in sync.Ledger(tmp_path / "work" / sync.LEDGER_FILE).entries()]
    assert "starter" in events and "compose" in events


def test_target_uses_the_fastest_start_not_the_run(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    tools = _tools(tmp_path, rec=rec, bench_table={"reference": {"wl0": 0.1, "wl1": 0.1,
                                                                 "wl2": 0.1}})
    plan = _plan(tmp_path, step=1.25)
    runner.run_plan(plan, tools=tools, work_dir=tmp_path / "work", step=1.25)
    task = load_task(tmp_path / "repo-work" / "tasks" / "069_rms_norm")
    assert task.target.ms == pytest.approx(0.1 / 1.25)


def test_workload_selector_all_expands_from_the_dataset(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    tools = _tools(tmp_path, rec=rec)
    plan = _plan(tmp_path)
    pr = runner.ProblemRunner(tools, plan["problems"][0], work_dir=tmp_path / "work",
                              state=runner.StateStore(tmp_path / "work"),
                              ledger=sync.Ledger(tmp_path / "work" / "ledger.jsonl"),
                              dataset=str(H.DATASET))
    assert pr.workload_selector() == "0-2"
    pr.dataset = str(tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError, match="workload list"):
        pr.workload_selector()


def test_offline_without_a_record_builds_a_no_sol_board_card(tmp_path):
    tools = _tools(tmp_path, rec=None)
    plan = _plan(tmp_path)
    st = runner.run_plan(plan, tools=tools, work_dir=tmp_path / "work", offline=True)[0]
    init_argv = next(c[1] for c in tools.calls if c[0] == "init")
    assert "--no-sol-board" in init_argv and "--sol-board" not in init_argv
    # without T_b / T_SOL a reading is a latency reading: no SOL score anywhere, and the
    # target comes from the latency alone
    assert st.status == "done", st.error
    assert all(c["mean_s"] is None for c in st.candidates)
    task = load_task(tmp_path / "repo-work" / "tasks" / "069_rms_norm")
    assert task.target.ms is not None


def test_a_cached_board_record_is_used_without_network(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    work = tmp_path / "work"
    _seed_record(work, 69, rec)
    tools = _tools(tmp_path, rec=rec)
    plan = _plan(tmp_path)
    runner.run_plan(plan, tools=tools, work_dir=work, offline=True)
    init_argv = next(c[1] for c in tools.calls if c[0] == "init")
    assert init_argv[init_argv.index("--sol-board") + 1].endswith("board_records/69.json")


def test_cold_start_gate_rejects_and_allows(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    plan = _plan(tmp_path)
    fast = {"latencies": {"wl0": 0.03, "wl1": 0.04, "wl2": 0.06}, "cold_start": 5.0}
    slow = {"latencies": {"wl0": 0.02, "wl1": 0.03, "wl2": 0.05}, "cold_start": 150.0}
    for allow in (False, True):
        work = tmp_path / ("work-" + str(allow))
        tools = _tools(tmp_path, rec=rec, work=work, root=tmp_path / f"repo-{allow}",
                       bench_table={"reference": {"wl0": 0.05, "wl1": 0.09, "wl2": 0.2},
                                    "baseline:a": fast, "baseline:b": slow})
        # a is the safe pick (5 s cold start); b scores better but takes 150 s to compile
        tools.lib_baselines = _two_baselines()
        st = runner.run_plan(plan, tools=tools, work_dir=work,
                             allow_slow_cold_start=allow)[0]
        assert st.status == "done", st.error
        rejected = [r["candidate"] for r in st.export.get("rejected") or []]
        if allow:
            assert st.export["cold_start"]["state"] == "too-slow"
            assert st.export["candidate"] == "baseline:b"
            assert any("allowed by --allow-slow-cold-start" in n for n in st.export["notes"])
        else:
            assert any("cold start" in r["reason"] for r in st.export["rejected"])
            assert st.export["candidate"] == "baseline:a"


def _two_baselines():
    def make(problem, out_dir, **kw):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "a.py").write_text(KERNEL_A)
        (out_dir / "b.py").write_text(KERNEL_B)
        return out_dir
    return make


def test_an_unmeasured_cold_start_is_exported_with_a_note(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    tools = _tools(tmp_path, rec=rec, cold=None)
    st = runner.run_plan(_plan(tmp_path), tools=tools, work_dir=tmp_path / "work")[0]
    assert st.status == "done", st.error
    assert st.export["cold_start"] == {"state": "unmeasured",
                                       "why": "the bench did not time a cold start",
                                       "s": None}
    assert any("cold start unmeasured" in n for n in st.export["notes"])


def test_a_loop_that_runs_out_of_budget_still_yields_versions(tmp_path):
    """`forge run` exits 2 when it stops short of the target; the versions it did produce
    are still collected and exported."""
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    inner = _run_loop_with(tmp_path, [KERNEL_A, KERNEL_B],
                           [{"wl0": 0.03, "wl1": 0.04, "wl2": 0.06},
                            {"wl0": 0.02, "wl1": 0.03, "wl2": 0.05}])

    def run_loop(argv, env=None):
        inner(argv, env)
        return 2                                   # stopped on the budget, target unmet
    tools = _tools(tmp_path, rec=rec, run_loop=run_loop)
    st = runner.run_plan(_plan(tmp_path), tools=tools, work_dir=tmp_path / "work")[0]
    assert st.status == "done", st.error
    assert len([c for c in st.candidates if c["origin"] == "run"]) == 2
    assert Path(st.export["path"]).is_file()
    assert "2.0" in Path(st.export["path"]).read_text()


def test_a_cuda_version_needs_klib_to_export(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    plan = _plan(tmp_path)
    cuda_kernel = ("#include <torch/extension.h>\n"
                   "torch::Tensor run(torch::Tensor x) { return x; }\n"
                   "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"run\", &run); }\n")

    def run_loop_cu(argv, env=None):
        run_dir = Path(argv[-1])
        shas = H.commit_kernels(run_dir, [cuda_kernel], kernel_file="kernel.cu")
        H.write_status(run_dir, [{"iter": 0, "verdict": "KEEP", "correctness": True,
                                  "sha": shas[0], "workloads": [
                                      {"name": n, "device_median_ms": 0.01}
                                      for n in ("wl0", "wl1", "wl2")]}])
        return 0

    # 1) with no klib: the .cu cannot enter a submission, and the export says so while
    #    the reference (the only Python version) still ships
    work = tmp_path / "work"
    tools = _tools(tmp_path, rec=rec, work=work, run_loop=run_loop_cu)
    st = runner.run_plan(plan, tools=tools, work_dir=work, route="cuda")[0]
    assert st.status == "done", st.error
    assert st.export["kind"] == "python"
    reasons = " ".join(r["reason"] for r in st.export["rejected"])
    assert "needs klib" in reasons
    assert any(c["language"] == "cuda-cpp" for c in st.candidates)

    # 2) with a klib stand-in the converted file wins (it scores far better)
    inline = "import torch\n\n\ndef run(input, weight, eps, output):\n    output.copy_(input)\n"
    calls = []
    work2, root2 = tmp_path / "work2", tmp_path / "repo2"
    _seed_record(work2, 69, rec)
    tools2 = H.recording_tools(
        root2, bench=H.FakeBench({"reference": {"wl0": 0.05, "wl1": 0.09, "wl2": 0.2}},
                                 board=H.board_mapping(rec)),
        records={H.L1_PROBLEM: rec}, run_loop=run_loop_cu,
        cuda_to_inline=lambda cu, **kw: (calls.append((str(cu), kw)), inline)[1])
    st2 = runner.run_plan(plan, tools=tools2, work_dir=work2, route="cuda")[0]
    assert st2.status == "done", st2.error
    assert st2.export["kind"] == "cuda-inline"
    assert any("inline-cuda" in r for r in st2.export["risk"])


def test_adapt_is_used_when_only_a_family_neighbour_exists(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    adapted = tmp_path / "adapt" / "submission.py"
    adapted.parent.mkdir(parents=True)
    adapted.write_text(KERNEL_B)

    def lib_find(problem, k=5):
        return [{"id": "k9", "problem_keys": ["rms_norm"], "family": "rms_norm",
                 "language": "triton", "license": "MIT"}]

    def lib_adapt(problem, seed, out_dir, **kw):
        assert seed == "k9" and kw.get("dataset") == str(H.DATASET)
        return adapted

    tools = _tools(tmp_path, rec=rec, bench_table={
        "reference": {"wl0": 0.05, "wl1": 0.09, "wl2": 0.2},
        "adapt:k9": {"wl0": 0.02, "wl1": 0.03, "wl2": 0.05}})
    tools.lib_find = lib_find
    tools.lib_adapt = lib_adapt
    st = runner.run_plan(_plan(tmp_path), tools=tools, work_dir=tmp_path / "work")[0]
    origins = {c["origin"] for c in st.candidates}
    assert "adapt" in origins
    assert any(e.get("candidate") == "adapt:k9" for e in
               sync.Ledger(tmp_path / "work" / sync.LEDGER_FILE).entries())


def test_shapes_compose_is_preferred_when_installed(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    dispatched = tmp_path / "shapes" / "submission.py"
    dispatched.parent.mkdir(parents=True)
    dispatched.write_text(KERNEL_B)

    tools = _tools(tmp_path, rec=rec, bench_table={
        "reference": {"wl0": 0.05, "wl1": 0.09, "wl2": 0.2},
        "baseline:a": {"wl0": 0.02, "wl1": 0.03, "wl2": 0.05},
        "baseline:b": {"wl0": 0.03, "wl1": 0.04, "wl2": 0.06}})
    tools.lib_baselines = _two_baselines()
    seen = {}
    tools.shapes_compose = lambda task_dir, also, out, extra=None, **kw: (
        seen.update(task=str(task_dir), out=str(out)), dispatched)[1]
    st = runner.run_plan(_plan(tmp_path), tools=tools, work_dir=tmp_path / "work")[0]
    assert st.status == "done", st.error
    assert st.export["kind"] == "shapes"
    assert seen["task"].endswith("tasks/069_rms_norm")
    assert "forge shapes" in " ".join(st.export["notes"])


def test_licence_check_blocks_and_its_banner_lands_in_the_file(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    plan = _plan(tmp_path)
    seen = {}

    def check(meta):
        seen.update(meta)
        return {"ok": False, "problems": ["GPL-3.0 is not allowed in a submission"]}

    tools = _tools(tmp_path, rec=rec)
    tools.license_check = check
    st = runner.run_plan(plan, tools=tools, work_dir=tmp_path / "work")[0]
    assert st.status == "failed" and "licence check refused" in st.error
    assert seen, "license_check must be called with the candidate's meta"
    assert "Apache" not in json.dumps(seen)          # the reference's meta is not a licence

    work2, root2 = tmp_path / "work2", tmp_path / "repo2"
    _seed_record(work2, 69, rec)
    good = H.recording_tools(root2,
                             bench=H.FakeBench({"reference": {"wl0": .05, "wl1": .09,
                                                             "wl2": .2}},
                                               board=H.board_mapping(rec)),
                             records={H.L1_PROBLEM: rec})
    good.license_check = lambda meta: {"ok": True, "problems": []}
    good.notices_banner = lambda entries: "# NOTICE: contains code from X (Apache-2.0)"
    st2 = runner.run_plan(plan, tools=good, work_dir=work2)[0]
    assert st2.status == "done"
    head = Path(st2.export["path"]).read_text().splitlines()[0]
    assert head.startswith("# NOTICE: contains code from X")


def test_a_failed_stage_stops_the_problem_and_resumes(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    plan = _plan(tmp_path)
    tools = _tools(tmp_path, rec=rec)
    tools.init_task = lambda argv, env=None: 7
    st = runner.run_plan(plan, tools=tools, work_dir=tmp_path / "work")[0]
    assert st.status == "failed" and st.stages["task"] == "failed"
    assert "forge init exited 7" in st.error

    # resume: a working init finishes the problem, and the failed stage runs again
    tools.init_task = lambda argv, env=None: H.fake_init(
        tmp_path / "repo-work", H.DATASET, {H.L1_PROBLEM: rec})(argv)
    st2 = runner.run_plan(plan, tools=tools, work_dir=tmp_path / "work")[0]
    assert st2.status == "done", st2.error
    assert st2.stages["task"] == "done"


def test_a_finished_problem_is_not_rerun(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    tools = _tools(tmp_path, rec=rec)
    plan = _plan(tmp_path)
    runner.run_plan(plan, tools=tools, work_dir=tmp_path / "work")
    before = len(tools.calls)
    runner.run_plan(plan, tools=tools, work_dir=tmp_path / "work")
    assert len(tools.calls) == before


def test_submit_is_off_by_default_and_on_with_the_flag(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    plan = _plan(tmp_path)
    submitted = []
    tools = _tools(tmp_path, rec=rec)
    tools.board_submit = lambda problem, path, **kw: submitted.append((problem, str(path))) or {
        "rc": 0}
    st = runner.run_plan(plan, tools=tools, work_dir=tmp_path / "work")[0]
    assert st.status == "done" and submitted == []
    assert st.stages["submit"] == "skipped"

    tools2 = _tools(tmp_path, rec=rec, work=tmp_path / "work2",
                    root=tmp_path / "repo-other")
    tools2.board_submit = lambda problem, path, **kw: submitted.append((problem, str(path))) or {
        "rc": 0}
    st2 = runner.run_plan(plan, tools=tools2, work_dir=tmp_path / "work2", submit=True)[0]
    assert st2.status == "done" and len(submitted) == 1
    assert submitted[0][0] == H.L1_PROBLEM


def test_context_env_reaches_the_child_processes(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    seen = {}

    def new_run(argv, env=None):
        seen["new"] = dict(env or {})
        return H._mk_run(tmp_path / "repo-work", argv)

    def run_loop(argv, env=None):
        seen["run"] = dict(env or {})
        return 0
    tools = _tools(tmp_path, rec=rec, new_run=new_run, run_loop=run_loop)
    (tmp_path / "repo-work" / "target").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo-work" / "target" / "B200.json").write_text("{}")
    runner.run_plan(_plan(tmp_path), tools=tools, work_dir=tmp_path / "work")
    assert seen["new"].get("FK_TARGET_PROFILE_PATH", "").endswith("B200.json")
    assert seen["run"].get("FK_TARGET_PROFILE_PATH", "").endswith("B200.json")


def test_evolve_is_used_when_installed(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    plan = _plan(tmp_path)
    calls = []

    def evolve(argv, env=None):
        calls.append(list(argv))
        return 0
    tools = _tools(tmp_path, rec=rec, evolve=evolve)
    st = runner.run_plan(plan, tools=tools, work_dir=tmp_path / "work")[0]
    assert st.status == "done"
    assert calls and calls[0][0] == "evolve" and "--lineages" in calls[0]
    assert calls[0][calls[0].index("--lineages") + 1] == "3"
    assert "--langs" in calls[0]
    assert not [c for c in tools.calls if c[0] == "new"]


def test_parallel_bounds_the_problems_in_flight(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    lock = threading.Lock()
    state = {"now": 0, "max": 0}

    class SlowBench(H.FakeBench):
        def __call__(self, task_dir, candidate, **kw):
            with lock:
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
            try:
                time.sleep(0.02)
                return super().__call__(task_dir, candidate, **kw)
            finally:
                with lock:
                    state["now"] -= 1

    plan = _plan(tmp_path, selection="069_rms_norm,001_fp8_attention_output_projection")
    assert len(plan["problems"]) == 2
    tools = _tools(tmp_path, rec=rec)
    tools.bench = SlowBench({"reference": {"wl0": 0.05, "wl1": 0.09, "wl2": 0.2}},
                            board=H.board_mapping(rec))
    states = runner.run_plan(plan, tools=tools, work_dir=tmp_path / "work", parallel=2)
    assert all(s.status == "done" for s in states), [s.error for s in states]
    assert 1 <= state["max"] <= 2


def test_unknown_problem_names_are_refused(tmp_path):
    plan = _plan(tmp_path)
    tools = _tools(tmp_path, rec=None)
    with pytest.raises(ValueError, match="not in the plan"):
        runner.run_plan(plan, tools=tools, work_dir=tmp_path / "work",
                        problem_names=["nope"])


def test_status_and_run_report(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    tools = _tools(tmp_path, rec=rec)
    plan = _plan(tmp_path)
    runner.run_plan(plan, tools=tools, work_dir=tmp_path / "work")
    text = runner.status(tmp_path / "work", plan)
    assert H.L1_PROBLEM in text and "done" in text
    report = runner.write_run_report(tmp_path / "work", plan)
    assert "## 069_rms_norm" in report.read_text()


# --- compose / export, directly ----------------------------------------------------------

def _card_problem(tmp_path, rec=None, kernel_file="kernel.py") -> Path:
    root = tmp_path / "repo"
    return H.make_board_task(root, "069_rms_norm", H.DATASET, H.L1_DEFINITION, "0-2",
                             rec or H.board_record("L1", H.L1_PROBLEM, 69),
                             kernel_file=kernel_file)


def test_the_exported_tail_falls_back_to_the_reference(tmp_path):
    task = _card_problem(tmp_path)
    cand = compose.Candidate(name="v1", source=KERNEL_B, mean_s=0.9, cold_start_s=3.0)
    out = tmp_path / "out" / "submission.py"
    res = compose.compose(task, [cand], out=out, tools=tools_mod.Tools(root=tmp_path))
    assert res.ok and res.kind == "python"
    text = out.read_text()
    tree = ast.parse(text)
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert names.count("run") == 1
    run_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "run")
    handlers = [h for n in ast.walk(run_fn) if isinstance(n, ast.Try)
                for h in n.handlers]
    assert handlers, "run() must wrap the candidate in try/except"
    called = {n.id for h in handlers for n in ast.walk(h)
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert compose.REFERENCE_RUN in called


def test_the_reference_helpers_get_their_own_names():
    """A reference helper keeps its identity: prefixed, so it cannot collide with the
    candidate's helper of the same name."""
    problem = {"dps": True,
               "definition": {"name": "x", "outputs": {"o": {"shape": [], "dtype": "float32"}},
                              "reference": "def _scale(x):\n    return x * 2\n\n"
                                           "def run(x):\n    return _scale(x)\n"}}
    text = compose.reference_block(problem)
    tree = ast.parse(text)
    defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "def _fk_ref__scale(x):" in text
    assert compose.REFERENCE_RUN in defs and "_fk_ref__scale" in defs
    assert not ({"_scale", "run"} & defs)


def test_the_fallback_computes_the_reference_math(tmp_path):
    """Import the exported file: the candidate raises, the reference answers."""
    import importlib.util
    import sys
    task = _card_problem(tmp_path)
    cand = compose.Candidate(
        name="v1", cold_start_s=1.0,
        source="import torch\n\ndef _scale(x):\n    return x\n\n"
               "def run(input, weight, eps, output):\n    raise RuntimeError('boom')\n")
    out = tmp_path / "out" / "submission.py"
    assert compose.compose(task, [cand], out=out,
                           tools=tools_mod.Tools(root=tmp_path)).ok
    saved = {k: sys.modules.get(k) for k in ("torch", "triton")}
    try:
        sys.modules["torch"] = _fake_torch()
        sys.modules["triton"] = _module_with(jit=lambda f: f)
        spec = importlib.util.spec_from_file_location("fk_export_under_test", out)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        T = sys.modules["torch"].T
        inp, w, dst = T([3.0, 4.0]), T([1.0, 2.0]), T([0.0, 0.0])
        assert mod.run(inp, w, 1e-5, dst) is None
        # rms = sqrt((9+16)/2) = 3.5355 -> [3/3.5355*1, 4/3.5355*2]
        assert [round(x, 4) for x in dst.data] == [0.8485, 2.2627]
        assert inp.data == [3.0, 4.0] and w.data == [1.0, 2.0]
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _module_with(**attrs):
    import types
    m = types.ModuleType("fake")
    m.__dict__.update(attrs)
    return m


def _fake_torch():
    """A four-function torch stand-in: enough for the fixture reference and the writer."""
    import types
    m = types.ModuleType("torch")

    class T:
        def __init__(self, data):
            self.data = list(data)
            self.dtype = "f32"

        def pow(self, n):
            return T([x ** n for x in self.data])

        def to(self, dtype):
            return self

        def mean(self, dim=None, keepdim=False):
            return T([sum(self.data) / len(self.data)])

        def copy_(self, other):
            self.data = list(other.data if isinstance(other, T) else other)
            return self

        def _broadcast(self, other):
            if isinstance(other, T):
                return [other.data[0] if len(other.data) == 1 else other.data[i]
                        for i in range(len(self.data))]
            return [other] * len(self.data)

        def __add__(self, other):
            return T([a + b for a, b in zip(self.data, self._broadcast(other))])

        def __mul__(self, other):
            return T([a * b for a, b in zip(self.data, self._broadcast(other))])

        def __len__(self):
            return len(self.data)

    m.T = T
    m.Tensor = T
    m.float32 = "f32"
    m.rsqrt = lambda x: T([1.0 / (v ** 0.5) for v in x.data])
    m.as_tensor = lambda v: v if isinstance(v, T) else T(v)
    return m


def test_every_candidate_rejected_means_no_export(tmp_path):
    task = _card_problem(tmp_path)
    bad = compose.Candidate(name="bad", source=KERNEL_A, correct=False,
                            notes=["correctness FAIL on wl1"])
    res = compose.compose(task, [bad], out=tmp_path / "out" / "submission.py",
                          tools=tools_mod.Tools(root=tmp_path))
    assert not res.ok and res.error == "no exportable candidate"
    assert res.rejected[0]["candidate"] == "bad"


def test_a_reference_only_submission_is_a_valid_dps_runner(tmp_path):
    prob = compose.load_problem(_card_problem(tmp_path))
    text = compose.reference_only_submission(prob)
    ast.parse(text)
    assert "def run(*args):" in text and compose.REFERENCE_RUN in text
    with pytest.raises(ValueError, match="no reference"):
        compose.reference_only_submission({"definition": {"name": "x", "outputs": {}}})


def test_inline_renames_the_entry_consistently(tmp_path):
    """The board entry is renamed; a nested function of the same name is renamed with it,
    so every reference inside the file keeps resolving exactly as before."""
    import importlib.util
    src = "def run(a, b):\n    def run():\n        return 1\n    return run()\n"
    text, found = compose.inline_source(src)
    assert found
    tree = ast.parse(text)
    top = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert top == [compose.CANDIDATE_RUN]
    assert [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
            and n.name == "run"] == []
    path = tmp_path / "k.py"
    path.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("fk_inline_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._fk_candidate_run(0, 0) == 1
    text, found = compose.inline_source("x = 1\n")
    assert not found and text == "x = 1\n"


# --- the campaign's own paper trail, through the CLI -------------------------------------

def _fake_plan_file(tmp_path, tools, *, name="069_rms_norm") -> Path:
    """A plan on disk plus a finished fake run of it."""
    plan = _plan(tmp_path, selection=name)
    work = tmp_path / "work"
    P.write_plan(P.Plan(**{**{k: v for k, v in plan.items()
                             if k in P.Plan.__dataclass_fields__ and k != "problems"},
                          "problems": [P.PlanProblem(**p) for p in plan["problems"]]}),
                 work / "plan.json")
    runner.run_plan(plan, tools=tools, work_dir=work)
    return work


def test_the_report_cli_writes_all_three_files(tmp_path, capsys):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    tools = _tools(tmp_path, rec=rec)
    work = _fake_plan_file(tmp_path, tools)
    rc = cli.main(["report", "--work-dir", str(work), "--snapshot", str(H.SNAPSHOT),
                   "--root", str(tmp_path)])
    assert rc == 0
    md = (work / report_mod.REPORT_MD).read_text()
    assert "instrument panel" in md and H.L1_PROBLEM in md
    assert "No problem has both a local reading and a board reading yet" in md
    doc = json.loads((work / report_mod.REPORT_JSON).read_text())
    entry = next(p for p in doc["problems"] if p["name"] == H.L1_PROBLEM)
    assert entry["planned"] and entry["status"] == "done" and entry["export"]
    assert entry["candidates"] >= 1
    assert (work / report_mod.REPORT_CSV).read_text().splitlines()[0].startswith("kernel_id")
    assert "wrote markdown" in capsys.readouterr().out


def test_the_report_corrects_a_local_score_with_the_family_ratio(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    tools = _tools(tmp_path, rec=rec)
    work = _fake_plan_file(tmp_path, tools)
    ledger = sync.Ledger(work / sync.LEDGER_FILE)
    ledger.append({"event": "board", "problem": H.L1_PROBLEM, "track": "l1",
                   "source": "board", "mean_s": 0.60, "correct": True})
    cli.main(["report", "--work-dir", str(work), "--snapshot", str(H.SNAPSHOT),
              "--root", str(tmp_path)])
    doc = json.loads((work / report_mod.REPORT_JSON).read_text())
    entry = next(p for p in doc["problems"] if p["name"] == H.L1_PROBLEM)
    assert entry["dev_s"] is not None and entry["board_s"] == 0.60
    assert entry["ratio"] == pytest.approx(0.60 / entry["dev_s"])
    assert entry["estimated_s"] == 0.60
    assert entry["estimate_basis"] == "board reading"


def test_the_status_export_and_tools_subcommands(tmp_path, capsys):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    tools = _tools(tmp_path, rec=rec)
    work = _fake_plan_file(tmp_path, tools)
    assert cli.main(["status", "--work-dir", str(work), "--root", str(tmp_path)]) == 0
    assert H.L1_PROBLEM in capsys.readouterr().out

    dest = tmp_path / "here" / "submission.py"
    assert cli.main(["export", H.L1_PROBLEM, "--work-dir", str(work), "--out", str(dest),
                     "--root", str(tmp_path)]) == 0
    assert dest.is_file() and "def run" in dest.read_text()

    assert cli.main(["tools", "--work-dir", str(work), "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "init_task" in out and "notes:" in out


def test_run_refuses_without_a_plan(tmp_path):
    with pytest.raises(SystemExit, match="no plan"):
        cli.main(["run", "--work-dir", str(tmp_path / "empty"), "--root", str(tmp_path)])
    # with a plan on disk but a name that is not in it
    plan = _plan(tmp_path)
    P.write_plan(P.Plan(**{**{k: v for k, v in plan.items()
                             if k in P.Plan.__dataclass_fields__ and k != "problems"},
                          "problems": [P.PlanProblem(**p) for p in plan["problems"]]}),
                 tmp_path / "work" / "plan.json")
    with pytest.raises(SystemExit, match="not in the plan"):
        cli.main(["export", "nope", "--work-dir", str(tmp_path / "work"),
                  "--root", str(tmp_path)])


def test_versions_skip_reverts_and_wrong_answers(tmp_path):
    rec = H.board_record("L1", H.L1_PROBLEM, 69)
    root = tmp_path / "repo"
    H.make_board_task(root, "069_rms_norm", H.DATASET, H.L1_DEFINITION, "0-2", rec)
    state = runner.StateStore(tmp_path / "work")
    st = runner.ProblemState(op="069_rms_norm", name=H.L1_PROBLEM, track="l1", kernel_id=69)
    st.candidates = [{"name": "reference", "source": KERNEL_A, "language": "python",
                      "mean_s": None, "geomean_ms": None, "cold_start_s": 1.0,
                      "correct": True, "sha": "", "run": "", "kernel_file": "",
                      "meta": {}, "origin": "starter", "notes": []}]
    state.save(st)
    run_dir = tmp_path / "repo" / ".artifacts" / "069_rms_norm_1" / "run.json"
    run_dir.parent.mkdir(parents=True)
    run_dir.write_text(json.dumps({"task": "069_rms_norm", "id": "1",
                                   "snapshot_sha256": "x"}))
    run_dir = run_dir.parent
    shas = H.commit_kernels(run_dir, [KERNEL_A, KERNEL_B, KERNEL_A])
    H.write_status(run_dir, [
        {"iter": 0, "verdict": "KEEP", "correctness": True, "sha": shas[0],
         "workloads": [{"name": "wl0", "device_median_ms": 0.05}]},
        {"iter": 1, "verdict": "REVERT", "correctness": False, "sha": shas[1],
         "workloads": [{"name": "wl0", "device_median_ms": 0.01}]},
        {"iter": 2, "verdict": "NEUTRAL", "correctness": None, "sha": shas[2],
         "workloads": []},
        {"iter": 3, "verdict": "KEEP", "correctness": True, "sha": shas[2],
         "reading_of": 0, "workloads": []}])
    env = runner.ProblemRunner(
        tools_mod.Tools(root=tmp_path / "repo"), {"op": "069_rms_norm",
                                                 "name": H.L1_PROBLEM, "track": "l1",
                                                 "kernel_id": 69},
        work_dir=tmp_path / "work", state=state,
        ledger=sync.Ledger(tmp_path / "work" / "ledger.jsonl"), dataset=str(H.DATASET))
    env.stage_versions(st)
    runs = [c for c in st.candidates if c["origin"] == "run"]
    assert len(runs) == 1, runs                      # one KEEP with a commit + its re-read
    assert runs[0]["sha"] == shas[0]
    assert runs[0]["mean_s"] is not None             # scored against the card's boards
