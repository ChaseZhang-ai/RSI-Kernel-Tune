import random
import unittest

from vulcan import allocator as al
from vulcan.allocator import KernelState, WorkloadState


def _random_kernels(rng, n=6):
    ks = []
    for i in range(n):
        t_sol = 0.1 + rng.random() * 0.5
        t_b = t_sol + 0.5 + rng.random() * 3.0
        t0 = t_b * (0.6 + rng.random() * 1.5)
        ws = [WorkloadState(t_ms=t0, t_b_ms=t_b, t_sol_ms=t_sol,
                            rate=0.02 + rng.random() * 0.1,
                            floor_frac=0.2 + rng.random() * 0.5)
              for _ in range(rng.randint(1, 4))]
        ks.append(KernelState(f"k{i}", ws))
    return ks


class TestAllocator(unittest.TestCase):
    def test_dp_at_least_greedy_at_least_uniform(self):
        rng = random.Random(0)
        for _ in range(25):
            ks = _random_kernels(rng)
            dp = al.allocate(ks, 40.0, 1.0, method="dp")
            gr = al.allocate(ks, 40.0, 1.0, method="greedy")
            un = al.allocate_uniform(ks, 40.0)
            self.assertGreaterEqual(dp.score_after + 1e-9, gr.score_after)
            self.assertGreaterEqual(gr.score_after + 1e-9, un.score_after - 1e-9)

    def test_dp_beats_uniform_on_lopsided(self):
        # one kernel far from the floor should soak the budget
        good = KernelState("far", [WorkloadState(2.0, 3.0, 0.1, rate=0.1)])
        near = KernelState("near", [WorkloadState(0.12, 3.0, 0.1, rate=0.1)])
        ks = [good, near]
        dp = al.allocate(ks, 30.0, 1.0, method="dp")
        un = al.allocate_uniform(ks, 30.0)
        self.assertGreater(dp.score_after, un.score_after)
        self.assertGreater(dp.hours["far"], dp.hours["near"])

    def test_budget_respected(self):
        rng = random.Random(3)
        ks = _random_kernels(rng)
        dp = al.allocate(ks, 37.0, 0.5, method="dp")
        self.assertLessEqual(sum(dp.hours.values()), 37.0 + 1e-9)

    def test_model_floors_and_is_monotone(self):
        w = WorkloadState(10.0, 12.0, 1.0, rate=0.5, floor_frac=0.3)
        self.assertAlmostEqual(w.latency_at(0.0), 10.0)
        self.assertLess(w.latency_at(5.0), w.latency_at(1.0))
        # never below the campaign floor
        self.assertGreaterEqual(w.latency_at(1e6), 1.0 + 0.3 * (10.0 - 1.0) - 1e-6)

    def test_fast1_weight_changes_allocation(self):
        ks = [KernelState("a", [WorkloadState(1.9, 2.0, 1.0, rate=0.3)]),
              KernelState("b", [WorkloadState(5.0, 200.0, 1.0, rate=0.01)])]
        plain = al.allocate(ks, 10.0, 1.0, fast1_weight=0.0, method="dp")
        weighted = al.allocate(ks, 10.0, 1.0, fast1_weight=1.0, method="dp")
        # the Fast1 bonus should pull hours toward the straggler 'a'
        self.assertGreaterEqual(weighted.hours["a"], plain.hours["a"])

    def test_projection_uses_mean_over_kernels(self):
        ks = [KernelState("a", [WorkloadState(1.0, 2.0, 0.1)]),
              KernelState("b", [WorkloadState(1.0, 2.0, 0.1)])]
        sc, _ = al.current_collection(ks)
        self.assertAlmostEqual(sc, ks[0].value_at(0.0))


if __name__ == "__main__":
    unittest.main()
