import math
import unittest

from vulcan import objective as ob


class TestScoreAlgebra(unittest.TestCase):
    def test_score_at_baseline_is_half(self):
        self.assertAlmostEqual(ob.sol_score(1.0, 1.0, 0.2), 0.5)

    def test_score_at_floor_is_one(self):
        self.assertEqual(ob.sol_score(0.2, 1.0, 0.2), 1.0)

    def test_inverse_roundtrip(self):
        for s in (0.2, 0.5, 0.63, 0.8, 0.95, 0.999):
            t = ob.latency_for_score(s, 2.0, 0.1)
            self.assertAlmostEqual(ob.sol_score(t, 2.0, 0.1), s, places=9)

    def test_monotone_decreasing_in_latency(self):
        prev = 1.0
        for t in [0.2, 0.5, 1.0, 2.0, 10.0]:
            s = ob.sol_score(t, 1.0, 0.2)
            self.assertLessEqual(s, prev + 1e-12)
            prev = s

    def test_bad_bounds_rejected(self):
        with self.assertRaises(ob.ScoreError):
            ob.sol_score(1.0, 0.2, 0.5)
        with self.assertRaises(ob.ScoreError):
            ob.sol_score(-1.0, 1.0, 0.2)

    def test_sensitivity_positive(self):
        self.assertGreater(ob.score_sensitivity(0.8, 1.0, 0.2), 0.0)
        self.assertEqual(ob.score_sensitivity(0.1, 1.0, 0.2), 0.0)


class TestAggregation(unittest.TestCase):
    def _wl(self, b, s):
        return ob.Workload("w", b, s)

    def test_kernel_is_geomean(self):
        wls = [self._wl(1.0, 0.1), self._wl(2.0, 0.2)]
        lats = [0.5, 1.0]
        res = ob.KernelResult("k", lats, wls)
        s0 = ob.sol_score(0.5, 1.0, 0.1)
        s1 = ob.sol_score(1.0, 2.0, 0.2)
        self.assertAlmostEqual(ob.kernel_score(res), math.sqrt(s0 * s1))

    def test_collection_is_unweighted_mean(self):
        self.assertAlmostEqual(ob.collection_score([0.2, 0.8, 1.0]), 2.0 / 3.0)

    def test_required_kernel_score(self):
        others = [0.8, 0.8, 0.8]
        need = ob.required_kernel_score_for_collection(0.85, others)
        self.assertAlmostEqual((0.8 * 3 + need) / 4.0, 0.85)

    def test_fast1_strict(self):
        wls = [self._wl(1.0, 0.1), self._wl(1.0, 0.1)]
        res = ob.KernelResult("k", [0.5, 1.0], wls)
        self.assertEqual(ob.kernel_fast1(res), (1, 2))


if __name__ == "__main__":
    unittest.main()
