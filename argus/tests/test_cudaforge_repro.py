import pytest

from argus import cudaforge as cf


def test_metric_set_is_the_upstream_24():
    assert len(cf.CUDAFORGE_METRICS) == 24
    for key in ("dram__bytes_read.sum", "sm__inst_executed.sum", "launch__registers_per_thread",
                "smsp__warp_issue_stalled_barrier_per_warp_active.pct"):
        assert key in cf.CUDAFORGE_METRICS
    assert cf.CUDAFORGE_PIN == "eaba693fa01bdafc9848334cada2d789e240ce3f"
    assert cf.CUDAFORGE_DEFAULT_TOL == 1e-3


def test_extract_code_block_and_json():
    raw = "blah\n```cuda\n__global__ void k(){}\n```\ntail"
    assert cf.extract_code_block(raw) == "__global__ void k(){}"
    assert cf.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert cf.extract_json('prefix {"b": 2} suffix') == {"b": 2}
    assert cf.extract_json("no json here") == {}


def test_extract_cuda_kernel_names():
    src = "__global__ void a() {}\n__global__ __launch_bounds__(128) void b() {}\n"
    assert cf.extract_cuda_kernel_names(src) == ["a", "b"]


def test_prompt_contracts_present():
    seed = cf.build_seed_prompt("t", "B300", "class M: pass", "rmsnorm")
    assert "Task: t" in seed and "Reference PyTorch module" in seed
    judge = cf.build_judge_optimization_prompt("t", "B300", "## metrics", "code")
    assert "Profiling evidence" in judge and "strict JSON" in judge
    opt = cf.build_optimization_prompt("code", "B300", {"mechanism": "pipeline"})
    assert "Judge strategy" in opt and "pipeline" in opt


def test_greedy_chain_is_single_chained():
    from argus.search import greedy_chain, MECHANISMS
    calls = {"n": 0}

    def seed_fn(l):
        return "seed"

    def mut(parent, m, l):
        calls["n"] += 1
        return f"{parent}+{m}"

    def ev(code, f):
        from argus.search import EvalResult
        return EvalResult(1.0, float(len(code)), {}, True, "")

    sr = greedy_chain(seed_fn, mut, ev, mechanisms=list(MECHANISMS), max_evals=5)
    # the chain is a path: each candidate's parent is the previous candidate
    for i in range(1, len(sr.candidates)):
        assert sr.candidates[i].parent == sr.candidates[i - 1].cid
    assert sr.candidates[0].mechanism == "seed"
