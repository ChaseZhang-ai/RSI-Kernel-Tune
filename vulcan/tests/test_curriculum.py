import unittest

from vulcan import curriculum as cur
from vulcan.allocator import KernelState, WorkloadState


def _spec(name, t, tb, ts, n=1):
    ws = [WorkloadState(t, tb, ts, rate=0.05) for _ in range(n)]
    return cur.TaskSpec(name=name, state=KernelState(name, ws))


class TestCurriculum(unittest.TestCase):
    def test_family_of(self):
        self.assertEqual(cur.family_of("004_gemm_n128_k2048"), "gemm")
        self.assertEqual(cur.family_of("021_rmsnorm_h128"), "rmsnorm")
        self.assertEqual(cur.family_of("013_gqa_paged_decode_h32_kv8_d128_ps1"),
                         "gqa_paged_decode")
        self.assertEqual(cur.family_of("001_fused_add_rmsnorm_h2048"),
                         "fused_add_rmsnorm")

    def test_value_order_prefers_headroom(self):
        low = _spec("004_gemm_n128_k2048", 0.11, 1.0, 0.1)   # essentially at floor
        high = _spec("009_gemm_n5120_k2048", 3.0, 4.0, 0.1)  # lots of headroom
        order = cur.value_order([low, high])
        self.assertEqual(order[0], "009_gemm_n5120_k2048")

    def test_curriculum_starts_families_easy(self):
        easy = _spec("024_rmsnorm_h2048", 0.2, 1.0, 0.1)
        hard = _spec("025_rmsnorm_h4096", 5.0, 6.0, 0.1)
        other = _spec("004_gemm_n128_k2048", 0.5, 1.0, 0.1)
        order = cur.curriculum_order([hard, easy, other])
        self.assertLess(order.index("024_rmsnorm_h2048"),
                        order.index("025_rmsnorm_h4096"))

    def test_order_strategies_length(self):
        specs = [_spec(f"00{i}_gemm_n128_k2048", 1.0 + i, 3.0, 0.1) for i in range(5)]
        for strat in ("value", "curriculum", "hybrid"):
            self.assertEqual(len(cur.order(specs, strat)), 5)

    def test_seed_plan_groups(self):
        specs = [_spec("004_gemm_n128_k2048", 1, 2, .1),
                 _spec("005_gemm_n256_k7168", 2, 4, .1),
                 _spec("021_rmsnorm_h128", 1, 2, .1)]
        plan = dict(cur.family_seed_plan(specs))
        self.assertEqual(set(plan), {"gemm", "rmsnorm"})
        self.assertEqual(len(plan["gemm"]), 2)

    def test_fast1_gap_counts(self):
        s = _spec("004_gemm_n128_k2048", t=2.0, tb=1.0, ts=0.1, n=3)
        self.assertEqual(s.fast1_gap(), 3)


if __name__ == "__main__":
    unittest.main()
