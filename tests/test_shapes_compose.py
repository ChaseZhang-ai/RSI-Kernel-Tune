"""`shapes compose`: the inlined candidates, the dispatcher, the solution JSON and its check.

The dispatcher is generated and then *imported and called* on CPU tensors with a fake card:
routing on the input shapes, the fallback for a shape the table does not name, the outer
try/except, and "no input tensor is modified" are all observable without a GPU. The
readings come from ``FakeBench`` -- compose never times anything itself.
"""

from __future__ import annotations

import importlib.util
import json
import py_compile
from pathlib import Path

import pytest
import torch

from rsi_kernel_tune.benchx.api import BenchResult, FakeBench, WorkloadReading
from rsi_kernel_tune.shapes import compose as CC
from rsi_kernel_tune.shapes import evalx, matrix as M, plan as P

KERNEL = "def run(*args):\n    return None\n"


# --- a small card ------------------------------------------------------------------------

SMALL_DEFINITION = {
    "name": "small_gemm",
    "description": "A 2-D product whose row count is the only variable (test card).",
    "axes": {"M": {"type": "var"}, "N": {"type": "const", "value": 4},
             "K": {"type": "const", "value": 4}},
    "inputs": {"A": {"shape": ["M", "K"], "dtype": "float32"},
               "B": {"shape": ["K", "N"], "dtype": "float32"}},
    "outputs": {"C": {"shape": ["M", "N"], "dtype": "float32"}},
    "reference": "import torch\n\n\ndef run(A, B):\n    return A @ B\n",
}


def _small_task(tmp_path: Path, *, rows=(2, 3, 64, 96), board=True) -> Path:
    """A card with four workloads in two size classes (M = 2, 3 and M = 64, 96)."""
    task = tmp_path / "tasks" / "small"
    card = task / "eval" / "leaderboard"
    card.mkdir(parents=True, exist_ok=True)
    group = []
    for i, m in enumerate(rows):
        group.append({
            "index": i, "name": f"wl{i}",
            "workload": {"uuid": f"small-wl{i}", "axes": {"M": m},
                         "inputs": {"A": {"type": "random"}, "B": {"type": "random"}}},
            **({"sol_board": {"kernel_id": 901, "workload_index": i,
                              "t_b_ms": float(m), "t_sol_ms": float(m) / 8.0,
                              "source": "test"}} if board else {}),
        })
    doc = {"schema": 1, "suite": "sol-execbench", "definition": SMALL_DEFINITION,
           "workload": group[0]["workload"], "dps": True, "kernel_file": "kernel.py",
           "entry": "run", "reference_impl": {"kind": "definition"},
           "source": {"dataset": "https://example.invalid", "dataset_root": "/tmp/dataset",
                      "problem": "L1/small_gemm", "upstream": "test"},
           "group": group}
    if board:
        doc["sol_board"] = group[0]["sol_board"]
    (card / "problem.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return task


def _solution(path: Path, source: str, name: str = "solution") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "name": name, "definition": "small_gemm", "author": "test",
        "spec": {"languages": ["pytorch"], "target_hardware": ["B200"],
                 "entry_point": "kernel.py::run", "destination_passing_style": True,
                 "dependencies": ["torch"]},
        "sources": [{"path": "kernel.py", "content": source}]}), encoding="utf-8")
    return path


def _reading(times: dict, *, correct: dict | None = None) -> BenchResult:
    correct = correct or {}
    return BenchResult(correct=True, workloads=[
        WorkloadReading(name=name, t_ms=t if correct.get(name, True) else None,
                        correct=correct.get(name, True),
                        s=(1 / (1 + t)) if correct.get(name, True) else None,
                        error="" if correct.get(name, True) else "the card said no")
        for name, t in times.items()], mean_s=0.5, cold_start_s=1.5)


def _import(path: Path):
    spec = importlib.util.spec_from_file_location(f"sub_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the AST machinery -------------------------------------------------------------------

def test_prefix_module_isolates_every_binding():
    src = (
        "from __future__ import annotations\n"
        "import torch\n"
        "import triton.language as tl\n"
        "from math import sqrt as root\n"
        "BLOCK = 128\n"
        "CUDA = '''__global__ void k() {}'''\n"
        "class State:\n"
        "    pass\n"
        "def helper(x):\n"
        "    return root(x) + BLOCK\n"
        "@tl.constexpr\n"
        "def kern(x):\n"
        "    return x\n"
        "def run(A, B, C):\n"
        "    C.copy_(helper(A) * kern(B))\n"
        "    return CUDA\n"
    )
    text, entry, features = CC.prefix_module(src, "_fk_0_")
    assert entry == "_fk_0_run"
    assert features == ["annotations"]
    tree_src = text
    for name in ("_fk_0_torch", "_fk_0_tl", "_fk_0_root", "_fk_0_BLOCK", "_fk_0_CUDA",
                 "_fk_0_State", "_fk_0_helper", "_fk_0_kern", "_fk_0_run"):
        assert name in tree_src, name
    # a string literal is not a name: an inlined CUDA source keeps its text
    assert "__global__ void k() {}" in tree_src
    # no bare `torch` reference survived the rename (the import is aliased with it)
    assert "import torch as _fk_0_torch" in tree_src
    # and the module compiles and keeps its behaviour
    namespace: dict = {}
    exec(compile(text, "<candidate>", "exec"), namespace)
    assert namespace["_fk_0_helper"](4.0) == 2.0 + 128


def test_prefix_module_refuses_a_source_without_run():
    with pytest.raises(Exception, match="no top-level run"):
        CC.prefix_module("X = 1\n", "_fk_0_")


def test_isolate_reference_prefixes_everything_and_fixes_the_entry():
    text, features = CC.isolate_reference(
        "import torch\nSCALE = 2\n\ndef run(x):\n    return x * SCALE\n")
    assert features == []
    namespace: dict = {}
    exec(compile(text, "<reference>", "exec"), namespace)
    assert "_fk_ref_SCALE" in namespace
    assert namespace["_fk_reference_run"](3.0) == 6.0


def test_a_shared_helper_name_in_two_candidates_does_not_collide():
    one = "def helper():\n    return 1\n\ndef run():\n    return helper()\n"
    two = "def helper():\n    return 2\n\ndef run():\n    return helper()\n"
    namespace: dict = {}
    for prefix, source in (("_fk_0_", one), ("_fk_1_", two)):
        text, _entry, _features = CC.prefix_module(source, prefix)
        exec(compile(text, "<c>", "exec"), namespace)
    assert namespace["_fk_0_run"]() == 1
    assert namespace["_fk_1_run"]() == 2


# --- the dispatcher, on CPU ---------------------------------------------------------------

C0 = '''
import torch

def run(A, B, C):
    raise RuntimeError("this kernel only handles the shape it was written for")
'''

C1 = '''
import torch

def run(A, B, C):
    C.fill_(2.0)
'''


def _composed(tmp_path: Path, *, allow_slower: bool = False, out=None):
    task = _small_task(tmp_path)
    plan = P.build_plan(task)
    doc = plan.as_json()
    product = Path(doc["product_dir"])
    _solution(product / "clusters" / "small-c0" / "solution.json", C0, "small-c0")
    _solution(product / "clusters" / "small-c1" / "solution.json", C1, "small-c1")

    def script(call):
        stem = Path(call.candidate).stem
        if stem == "small-c0":
            return _reading({"wl0": 0.1, "wl1": 0.1, "wl2": 100.0, "wl3": 100.0},
                            correct={"wl2": False, "wl3": False})
        if stem == "small-c1":
            return _reading({"wl0": 100.0, "wl1": 100.0, "wl2": 0.2, "wl3": 0.2},
                            correct={"wl0": False, "wl1": False})
        return _reading({"wl0": 1.0, "wl1": 1.0, "wl2": 1.0, "wl3": 1.0})

    bench = FakeBench(script=script)
    res = CC.compose_task(task, opts=CC.ComposeOptions(bench=bench, with_lib=False,
                                                       with_versions=False,
                                                       allow_slower=allow_slower, out=out),
                          plan_doc=doc)
    return task, res, bench


def test_the_dispatcher_routes_by_shape_and_falls_back(tmp_path):
    task, res, _bench = _composed(tmp_path)
    assert res.ok, res.error
    assert res.submission.is_file()
    py_compile.compile(str(res.submission), doraise=True)
    v = res.validation
    assert v["ok"], v["findings"]
    assert v["info"]["signature"] == "run(*args)"
    assert v["info"]["routes"] == 4

    mod = _import(res.submission)
    torch.manual_seed(0)
    A = torch.randn(2, 4)
    B = torch.randn(4, 4)
    C = torch.zeros(2, 4)
    a_before, b_before = A.clone(), B.clone()
    # wl0 routes to small-c0, which raises: the outer try/except runs the reference
    mod.run(A, B, C)
    assert torch.allclose(C, A @ B)
    assert torch.equal(A, a_before) and torch.equal(B, b_before)
    # wl3 (M = 96) routes to small-c1, which writes 2.0 everywhere
    A96 = torch.randn(96, 4)
    C96 = torch.zeros(96, 4)
    mod.run(A96, B, C96)
    assert torch.all(C96 == 2.0)
    # a shape the table does not name runs the fallback (here: the eager reference)
    A7 = torch.randn(7, 4)
    C7 = torch.zeros(7, 4)
    mod.run(A7, B, C7)
    assert torch.allclose(C7, A7 @ B)


def test_the_dispatcher_has_no_named_arguments(tmp_path):
    _task, res, _bench = _composed(tmp_path)
    text = res.submission.read_text(encoding="utf-8")
    assert "def run(*args):" in text
    assert "def run(A, B, C)" not in text          # the candidates were renamed
    # the file is one source: no imports of sibling files, no exec of a string body
    assert "exec(" not in text
    assert "from kernel" not in text


def test_compose_writes_every_product_and_the_report(tmp_path):
    task, res, bench = _composed(tmp_path)
    assert [Path(c.candidate).name for c in bench.calls] == [
        "small-c0.py", "small-c1.py", "eager.py"]
    product = Path(res.submission).parent
    assert res.solution.is_file() and res.solution.name == "solution.json"
    assert res.matrix_json.is_file() and res.routes_json.is_file()
    assert res.report.is_file()
    assert product == task / "shapes"
    routes = json.loads(res.routes_json.read_text(encoding="utf-8"))
    assert [r["winner"] for r in routes["routes"]] == ["small-c0", "small-c0",
                                                       "small-c1", "small-c1"]
    assert routes["mean_s"] is not None
    assert routes["candidates"] == ["small-c0", "small-c1", "eager"]
    assert routes["validation"]["ok"] is True
    assert "Per workload" in res.report.read_text(encoding="utf-8")
    assert res.solution_validation["ok"], res.solution_validation["findings"]
    solution = json.loads(res.solution.read_text(encoding="utf-8"))
    assert solution["sources"][0]["content"] == res.submission.read_text(encoding="utf-8")
    assert solution["spec"]["entry_point"] == "submission.py::run"


def test_out_may_name_a_directory_or_a_file(tmp_path):
    _task, res, _bench = _composed(tmp_path / "a", out=tmp_path / "export")
    assert res.submission == tmp_path / "export" / "submission.py"
    assert res.solution == tmp_path / "export" / "solution.json"
    _task, res2, _bench = _composed(tmp_path / "b", out=tmp_path / "mine.py")
    assert res2.submission == tmp_path / "mine.py"


def test_allow_slower_keeps_the_faster_candidate_above_the_floor(tmp_path):
    # both candidates are slower than eager on every workload; by default the reference
    # takes them, with --allow-slower the specialists do.
    task = _small_task(tmp_path)
    doc = P.build_plan(task).as_json()
    product = Path(doc["product_dir"])
    _solution(product / "clusters" / "small-c0" / "solution.json", C1, "small-c0")
    _solution(product / "clusters" / "small-c1" / "solution.json", C0, "small-c1")

    def script(call):
        stem = Path(call.candidate).stem
        if stem.startswith("small"):
            return _reading({"wl0": 3.0, "wl1": 3.0, "wl2": 3.0, "wl3": 3.0})
        return _reading({"wl0": 1.0, "wl1": 1.0, "wl2": 1.0, "wl3": 1.0})

    default = CC.compose_task(task, plan_doc=doc, opts=CC.ComposeOptions(
        bench=FakeBench(script=script), with_lib=False, with_versions=False))
    assert [r.winner for r in default.routes.routes] == ["eager"] * 4
    allowed = CC.compose_task(task, plan_doc=doc, opts=CC.ComposeOptions(
        bench=FakeBench(script=script), with_lib=False, with_versions=False,
        allow_slower=True, out=tmp_path / "allowed"))
    # 3.0 vs 3.0 is a tie, so the first candidate in the preference order takes all four
    assert [r.winner for r in allowed.routes.routes] == ["small-c0"] * 4
    assert allowed.routes.guarantee["violations"] == ["wl0", "wl1", "wl2", "wl3"]


def test_a_candidate_that_cannot_inline_is_reported_not_used(tmp_path):
    task = _small_task(tmp_path)
    doc = P.build_plan(task).as_json()
    product = Path(doc["product_dir"])
    _solution(product / "clusters" / "small-c0" / "solution.json",
              "def run(*args):\n    return None\n", "small-c0")
    _solution(product / "clusters" / "small-c1" / "solution.json",
              "def run(*args):\n    return None\n", "small-c1")

    def script(call):
        stem = Path(call.candidate).stem
        if stem == "small-c0":
            return _reading({"wl0": 0.1, "wl1": 0.1, "wl2": 0.1, "wl3": 0.1})
        return _reading({"wl0": 1.0, "wl1": 1.0, "wl2": 1.0, "wl3": 1.0})

    real = CC.prefix_module

    def flaky(source, prefix, **kw):
        if "small_c0" in prefix:
            raise CC.ShapesError("no source to inline")
        return real(source, prefix, **kw)

    CC.prefix_module = flaky
    try:
        res = CC.compose_task(task, plan_doc=doc, opts=CC.ComposeOptions(
            bench=FakeBench(script=script), with_lib=False, with_versions=False))
    finally:
        CC.prefix_module = real
    assert res.ok
    assert any("cannot be inlined" in n for n in res.notes)
    # the un-inlinable candidate is not routed to; the other one is (a tie with the
    # reference inside the noise floor goes to the specialist)
    assert [r.winner for r in res.routes.routes] == ["small-c1"] * 4
    assert "small_c0" not in res.submission.read_text(encoding="utf-8")


def test_a_cuda_candidate_is_left_out_with_a_note(tmp_path):
    task = _small_task(tmp_path)
    doc = P.build_plan(task).as_json()
    product = Path(doc["product_dir"])
    _solution(product / "clusters" / "small-c0" / "solution.json", C1, "small-c0")
    _solution(product / "clusters" / "small-c1" / "solution.json", C1, "small-c1")
    cu = tmp_path / "hand_written.cu"
    cu.write_text("__global__ void k() {}\nvoid run() {}\n", encoding="utf-8")

    def script(call):
        return _reading({"wl0": 1.0, "wl1": 1.0, "wl2": 1.0, "wl3": 1.0})

    res = CC.compose_task(task, plan_doc=doc, opts=CC.ComposeOptions(
        bench=FakeBench(script=script), with_lib=False, with_versions=False,
        also=[str(cu)]))
    assert res.ok
    assert any("CUDA C++ is not composable" in n for n in res.notes)
    assert "hand_written" not in res.submission.read_text(encoding="utf-8")


def test_compose_says_it_needs_benchx(tmp_path, monkeypatch):
    task = _small_task(tmp_path)
    doc = P.build_plan(task).as_json()
    product = Path(doc["product_dir"])
    _solution(product / "clusters" / "small-c0" / "solution.json", C1, "small-c0")

    def no_benchx(**_kw):
        raise evalx.ShapesError("`shapes compose` needs benchx")

    monkeypatch.setattr(evalx, "make_bench", no_benchx)
    res = CC.compose_task(task, plan_doc=doc, opts=CC.ComposeOptions(
        with_lib=False, with_versions=False))
    assert res.submission is None
    assert "needs benchx" in res.error


def test_compose_without_any_candidate_says_what_to_do(tmp_path):
    task = _small_task(tmp_path)
    doc = P.build_plan(task).as_json()
    # no cluster exports, no --also, no lib: only the eager reference remains
    res = CC.compose_task(task, plan_doc=doc, opts=CC.ComposeOptions(
        bench=FakeBench(t_ms={"wl0": 1.0, "wl1": 1.0, "wl2": 1.0, "wl3": 1.0}),
        with_lib=False, with_versions=False))
    assert res.ok                                # the reference alone is a legal solution
    assert res.routes.routes[0].winner == "eager"


def test_a_value_returning_card_gets_a_value_returning_dispatcher(tmp_path):
    task = _small_task(tmp_path)
    path = task / "eval/leaderboard/problem.json"
    card = json.loads(path.read_text(encoding="utf-8"))
    card["dps"] = False
    path.write_text(json.dumps(card, indent=1), encoding="utf-8")
    doc = P.build_plan(task).as_json()
    product = Path(doc["product_dir"])
    value_kernel = "def run(A, B):\n    return A * 0.0 + 2.0\n"
    _solution(product / "clusters" / "small-c0" / "solution.json", value_kernel,
              "small-c0")
    _solution(product / "clusters" / "small-c1" / "solution.json", value_kernel,
              "small-c1")

    def script(call):
        stem = Path(call.candidate).stem
        if stem == "small-c0":
            return _reading({"wl0": 0.1, "wl1": 0.1, "wl2": 0.1, "wl3": 0.1})
        return _reading({"wl0": 1.0, "wl1": 1.0, "wl2": 1.0, "wl3": 1.0})

    res = CC.compose_task(task, plan_doc=doc, opts=CC.ComposeOptions(
        bench=FakeBench(script=script), with_lib=False, with_versions=False))
    assert res.ok, res.error
    mod = _import(res.submission)
    A = torch.randn(2, 4)
    B = torch.randn(4, 4)
    out = mod.run(A, B)                          # no outputs in the arguments: returned
    assert torch.all(out == 2.0)                 # routed to small-c0
    A7 = torch.randn(7, 4)
    assert torch.allclose(mod.run(A7, B), A7 @ B)   # unknown shape: the reference
    assert "_FK_OUTPUTS = []" in res.submission.read_text(encoding="utf-8")


# --- the validators ----------------------------------------------------------------------

def test_validate_submission_findings(tmp_path):
    task = _small_task(tmp_path)
    problem, _card = P.load_problem(task)
    good = tmp_path / "good.py"
    good.write_text("def run(*args):\n    return None\n", encoding="utf-8")
    assert CC.validate_submission(good, problem)["ok"]
    named = tmp_path / "named.py"
    named.write_text("def run(A, B, C):\n    return None\n", encoding="utf-8")
    result = CC.validate_submission(named, problem)
    assert result["ok"] and result["info"]["signature_kind"] == "named"
    wrong = tmp_path / "wrong.py"
    wrong.write_text("def run(x, y):\n    return None\n", encoding="utf-8")
    result = CC.validate_submission(wrong, problem)
    assert not result["ok"] and "does not take the definition's names" in result["findings"][0]
    none = tmp_path / "none.py"
    none.write_text("X = 1\n", encoding="utf-8")
    assert not CC.validate_submission(none, problem)["ok"]
    broken = tmp_path / "broken.py"
    broken.write_text("def run(:\n", encoding="utf-8")
    result = CC.validate_submission(broken, problem)
    assert not result["ok"] and "does not compile" in result["findings"][0]
    assert CC.validate_submission(tmp_path / "missing.py", problem)["ok"] is False


def test_validate_solution_single_and_multi_source(tmp_path):
    task = _small_task(tmp_path)
    problem, _card = P.load_problem(task)
    single = tmp_path / "submission.py"
    single.write_text("def run(*args):\n    return None\n", encoding="utf-8")
    doc = CC.solution_json(problem, single.read_text(encoding="utf-8"), name="x")
    result = CC.validate_solution(doc, problem=problem, submission=single)
    assert result["ok"], result["findings"]
    assert result["sources_count"] == 1 and result["entry"] == "submission.py::run"

    # the official schema does allow several sources (single-problem submissions)
    multi = {"name": "m", "definition": "small_gemm", "author": "t",
             "spec": {"languages": ["pytorch"], "target_hardware": ["B200"],
                      "entry_point": "kernel.py::run",
                      "destination_passing_style": True, "dependencies": ["torch"]},
             "sources": [{"path": "kernel.py", "content": single.read_text()},
                         {"path": "src/helper.py", "content": "X = 1\n"}]}
    assert CC.validate_solution(multi, problem=problem)["ok"]


def test_validate_solution_findings(tmp_path):
    task = _small_task(tmp_path)
    problem, _card = P.load_problem(task)

    def doc(**over):
        base = {"name": "m", "definition": "small_gemm", "author": "t",
                "spec": {"languages": ["pytorch"], "target_hardware": ["B200"],
                         "entry_point": "kernel.py::run",
                         "destination_passing_style": True, "dependencies": ["torch"]},
                "sources": [{"path": "kernel.py", "content": "def run(*a):\n    pass\n"}]}
        base.update(over)
        return base

    assert CC.validate_solution(doc(), problem=problem)["ok"]
    missing = doc()
    missing.pop("author")
    assert not CC.validate_solution(missing)["ok"]
    dup = doc(sources=[{"path": "kernel.py", "content": "x"},
                       {"path": "kernel.py", "content": "y"}])
    assert any("appears twice" in f for f in CC.validate_solution(dup)["findings"])
    escape = doc(sources=[{"path": "kernel.py", "content": "x"},
                          {"path": "../evil.py", "content": "y"}])
    assert any("not relative" in f for f in CC.validate_solution(escape)["findings"])
    absent = doc(spec={**doc()["spec"], "entry_point": "other.py::run"})
    assert any("not in sources" in f for f in CC.validate_solution(absent)["findings"])
    mixed = doc(spec={**doc()["spec"], "languages": ["cuda_cpp", "pytorch"]})
    assert any("mixes C++ and Python" in f for f in CC.validate_solution(mixed)["findings"])
    wrong_dps = doc(spec={**doc()["spec"], "destination_passing_style": False})
    assert any("dps" in f for f in CC.validate_solution(wrong_dps,
                                                        problem=problem)["findings"])
    bad_entry = doc(spec={**doc()["spec"], "entry_point": "run"})
    assert any("file.py::function" in f for f in CC.validate_solution(bad_entry)["findings"])
    odd_dep = doc(spec={**doc()["spec"], "dependencies": ["torch", "flashinfer"]})
    result = CC.validate_solution(odd_dep, problem=problem)
    assert result["ok"] and any("allowed libraries" in w for w in result["warnings"])


def test_solution_and_submission_must_agree(tmp_path):
    task = _small_task(tmp_path)
    problem, _card = P.load_problem(task)
    sub = tmp_path / "submission.py"
    sub.write_text("def run(*args):\n    return None\n", encoding="utf-8")
    doc = CC.solution_json(problem, "def run(*args):\n    return 1\n", name="x")
    result = CC.validate_solution(doc, problem=problem, submission=sub)
    assert not result["ok"] and "differ" in result["findings"][0]


# --- candidates --------------------------------------------------------------------------

def test_load_candidate_accepts_py_and_solution_json(tmp_path):
    py = tmp_path / "mine.py"
    py.write_text("def run(*args):\n    return None\n", encoding="utf-8")
    cand = evalx.load_candidate(py, evalx.KIND_EXTRA)
    assert cand.language == "python" and cand.path == py and cand.sha
    sol = _solution(tmp_path / "sol" / "solution.json", "def run(*args):\n    pass\n", "s")
    cand = evalx.load_candidate(sol, evalx.KIND_OVERALL, label="overall")
    assert "def run" in cand.source and cand.label == "overall"
    bad = tmp_path / "bad.py"
    bad.write_text("X = 1\n", encoding="utf-8")
    with pytest.raises(Exception, match="no top-level run"):
        evalx.load_candidate(bad, evalx.KIND_EXTRA)
    with pytest.raises(Exception, match="not a solution JSON"):
        broken = tmp_path / "broken.json"
        broken.write_text("{", encoding="utf-8")
        evalx.load_candidate(broken, evalx.KIND_EXTRA)


def test_library_candidates_take_submittable_entries_for_this_problem(tmp_path):
    root = tmp_path / "repo"
    (root / "kernel_library" / "kernels" / "mine").mkdir(parents=True)
    (root / "kernel_library" / "kernels" / "mine" / "kernel.py").write_text(
        "def run(*args):\n    return None\n", encoding="utf-8")
    (root / "kernel_library" / "kernels" / "theirs").mkdir(parents=True)
    (root / "kernel_library" / "kernels" / "theirs" / "kernel.py").write_text(
        "def run(*args):\n    return None\n", encoding="utf-8")
    (root / "kernel_library" / "kernels" / "other").mkdir(parents=True)
    (root / "kernel_library" / "kernels" / "other" / "kernel.py").write_text(
        "def run(*args):\n    return None\n", encoding="utf-8")

    def entry(e_id, keys, license, usable=True):
        return {"id": e_id, "problem_keys": list(keys), "family": "gemm",
                "path": f"kernels/{e_id}/kernel.py", "language": "triton", "entry": "run",
                "source": {"repo": "r", "commit": "c", "path": "p"}, "license": license,
                "reported": None, "verified_local": None,
                "usable_in_submission": usable}

    index = {"version": 1, "entries": [
        entry("mine", ["small_gemm", "gemm"], "MIT"),
        entry("theirs", ["small_gemm", "gemm"], "unknown-proprietary", usable=False),
        entry("other", ["something_else", "gemm"], "MIT")]}
    (root / "kernel_library" / "index.json").write_text(json.dumps(index), encoding="utf-8")
    problem, _card = P.load_problem(_small_task(tmp_path))
    cands, notes = evalx.library_candidates(problem, root=root, out_dir=tmp_path / "c")
    assert [c.label for c in cands] == ["lib:mine"]
    assert cands[0].kind == evalx.KIND_LIBRARY and cands[0].path.is_file()
    assert "1 submittable entry" in notes[0]
    assert "theirs" in notes[0]                       # the licence refusal is reported
    # a root without the library says so instead of failing
    empty, notes = evalx.library_candidates(problem, root=tmp_path / "nothing")
    assert empty == [] and "no index at" in notes[0]


def test_versions_of_run_reads_status_jsonl_and_the_workspace_git(tmp_path):
    import subprocess

    run = tmp_path / "run1"
    ws = run / "ws"
    op = run / "operators" / "small"
    (op / "student").mkdir(parents=True)
    ws.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"task": "small"}), encoding="utf-8")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}

    def git(*args):
        subprocess.run(["git", "-C", str(ws), *args], check=True, capture_output=True,
                       env={**__import__("os").environ, **env})

    git("init", "-q")
    (ws / "kernel.py").write_text("def run(*args):\n    return 1\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "one")
    sha = subprocess.run(["git", "-C", str(ws), "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    (op / "student" / "status.jsonl").write_text("\n".join([
        json.dumps({"iter": 1, "verdict": "KEEP", "correctness": True, "sha": sha}),
        json.dumps({"iter": 2, "verdict": "REVERT", "correctness": True, "sha": sha}),
        json.dumps({"iter": 3, "verdict": "KEEP", "correctness": False, "sha": sha}),
    ]) + "\n", encoding="utf-8")
    versions, notes = evalx.versions_of_run(run, "kernel.py")
    assert [c.label for c in versions] == ["v:run1:it1"]
    assert versions[0].kind == evalx.KIND_VERSION and versions[0].sha == sha
    assert "return 1" in versions[0].source
    empty, notes = evalx.versions_of_run(tmp_path / "nothing")
    assert empty == [] and notes
