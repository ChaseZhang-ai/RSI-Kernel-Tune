import math
import unittest
from pathlib import Path

from vulcan import dataset
from vulcan.backends.mock import MockBackend
from vulcan.supervisor import Supervisor, reference_latency

DS = Path(__file__).resolve().parents[1] / "data" / "sol_execbench_b200_flashinfer.json"


def _anchors_bounds(specs):
    anchors, bounds = {}, {}
    for s in specs:
        tb = math.exp(sum(math.log(w.t_b_ms) for w in s.state.workloads)
                      / len(s.state.workloads))
        ts = math.exp(sum(math.log(max(w.t_sol_ms, 1e-12)) for w in s.state.workloads)
                      / len(s.state.workloads))
        anchors[s.name] = reference_latency(s.state)
        bounds[s.name] = (tb, ts)
    return anchors, bounds


def _run(ds, strategy, hours, seed):
    specs = dataset.build_specs(ds, current="forgekernel")
    anchors, bounds = _anchors_bounds(specs)
    backend = MockBackend(anchors, bounds, seed=seed)
    sup = Supervisor(backend, strategy=strategy, quantum=0.5,
                     portfolio_budgets=(1.0, 3.0, 7.0), eta=2)
    return sup.run(specs, hours), backend


class TestSupervisorMock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = dataset.load(str(DS))

    def test_budget_respected(self):
        rep, _ = _run(self.ds, "vulcan", 100.0, seed=0)
        self.assertLessEqual(sum(rep.allocation.values()), 100.0 + 1e-6)

    def test_vulcan_beats_uniform_on_average(self):
        seeds = [0, 1, 2, 3]
        vul, uni = [], []
        for sd in seeds:
            v, _ = _run(self.ds, "vulcan", 150.0, sd)
            u, _ = _run(self.ds, "uniform", 150.0, sd)
            vul.append(v.score_after)
            uni.append(u.score_after)
            # never catastrophically worse on any single seed
            self.assertGreaterEqual(v.score_after, u.score_after - 0.03)
        self.assertGreater(sum(vul) / len(vul), sum(uni) / len(uni))
        # and by a clear margin on average
        self.assertGreater((sum(vul) - sum(uni)) / len(vul), 0.01)

    def test_determinism(self):
        a, _ = _run(self.ds, "vulcan", 80.0, seed=5)
        b, _ = _run(self.ds, "vulcan", 80.0, seed=5)
        self.assertEqual(round(a.score_after, 9), round(b.score_after, 9))

    def test_portfolio_increases_backend_calls(self):
        _, b_uni = _run(self.ds, "uniform", 60.0, seed=1)
        _, b_vul = _run(self.ds, "vulcan", 60.0, seed=1)
        self.assertGreater(b_vul.calls, b_uni.calls)

    def test_invalid_kernel_reverts(self):
        # all architectures invalid for one fake task -> state must not move
        from vulcan.curriculum import TaskSpec
        from vulcan.allocator import KernelState, WorkloadState

        state = KernelState("dead", [WorkloadState(2.0, 3.0, 0.1)])
        spec = TaskSpec("dead", state, ["dead/a", "dead/b"])

        class DeadBackend:
            calls = 0
            def evaluate(self, task, ref, arch, hours, variant=""):
                from vulcan.backends.base import EvalResult
                self.calls += 1
                return EvalResult(float("inf"), correct=False)
            def export(self, task):
                return None

        before = spec.state.value_at(0.0)
        sup = Supervisor(DeadBackend(), strategy="vulcan", quantum=1.0,
                         portfolio_budgets=(1.0, 2.0), eta=2)
        rep = sup.run([spec], 10.0)
        self.assertAlmostEqual(spec.state.value_at(0.0), before)
        self.assertFalse(rep.outcomes[0].promoted)


if __name__ == "__main__":
    unittest.main()
