from pathlib import Path

from argus.memory import MechanismMemory, MechanismRecord
from argus.schedule import TaskPriority, plan_eval_budget, priority_weight


def test_memory_round_trip(tmp_path: Path):
    p = tmp_path / "mem.jsonl"
    m = MechanismMemory(p)
    m.add(MechanismRecord("rmsnorm", "memory", "l2_tiling", 1.4, task="t1"))
    m.add(MechanismRecord("rmsnorm", "memory", "l2_tiling", 1.6, task="t2"))
    m.add(MechanismRecord("rmsnorm", "memory", "tensor_core", 0.95, task="t3"))
    m.add(MechanismRecord("rmsnorm", "memory", "tensor_core", 0.93, task="t4"))
    m2 = MechanismMemory(p)
    assert len(m2.records) == 4
    mg, n = m2.mean_gain("rmsnorm", "memory", "l2_tiling")
    assert n == 2 and mg > 1.4
    assert "tensor_core" in m2.refuted("rmsnorm", "memory")     # refuted: consistently <1
    prior = m2.prior_for("rmsnorm", "memory", ["l2_tiling", "tensor_core"])
    assert prior[("rmsnorm", "l2_tiling")] > 0
    assert prior[("rmsnorm", "tensor_core")] == 0.0


def test_schedule_prefers_headroom_and_feasibility():
    tasks = [
        TaskPriority("low_headroom", score=0.98, max_feasible_gain=1.02),
        TaskPriority("high_headroom", score=0.55, max_feasible_gain=1.6),
    ]
    assert priority_weight(tasks[1]) > priority_weight(tasks[0])
    alloc = plan_eval_budget(tasks, total_evals=20, min_evals=2)
    assert sum(alloc.values()) >= 20 - len(tasks)
    assert alloc["high_headroom"] > alloc["low_headroom"]


def test_schedule_respects_tiny_budget():
    tasks = [TaskPriority("a", 0.6), TaskPriority("b", 0.7)]
    alloc = plan_eval_budget(tasks, total_evals=1, min_evals=1)
    assert sum(alloc.values()) == 1
