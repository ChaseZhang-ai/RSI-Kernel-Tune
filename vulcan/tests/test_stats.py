import statistics
import unittest

from vulcan import stats


class TestStats(unittest.TestCase):
    def test_bootstrap_ci_contains_median(self):
        xs = [1.0, 1.1, 0.9, 1.05, 0.95, 1.2, 0.8]
        lo, hi = stats.bootstrap_ci(xs, statistics.median)
        self.assertLessEqual(lo, statistics.median(xs))
        self.assertGreaterEqual(hi, statistics.median(xs))

    def test_paired_ratio(self):
        ref = [1.0, 1.0, 1.0]
        cand = [0.5, 0.5, 0.5]
        lo, hi = stats.paired_ratio_ci(ref, cand)
        self.assertAlmostEqual((lo + hi) / 2, 2.0, places=6)

    def test_accept_clear_gain(self):
        ref = [1.00, 1.01, 0.99, 1.00, 1.02, 0.98]
        cand = [0.80, 0.81, 0.79, 0.80, 0.82, 0.78]
        v = stats.accept_gain(ref, cand, min_margin=0.05)
        self.assertTrue(v.accepted)

    def test_reject_identical(self):
        xs = [1.0, 1.0, 1.0, 1.0]
        v = stats.accept_gain(xs, xs, min_margin=0.01)
        self.assertFalse(v.accepted)

    def test_reject_gain_below_margin(self):
        ref = [1.0] * 10
        cand = [0.99] * 10  # 1% faster, margin demands 5%
        v = stats.accept_gain(ref, cand, min_margin=0.05)
        self.assertFalse(v.accepted)

    def test_noise_floor(self):
        self.assertEqual(stats.noise_floor_relative([1.0, 1.0]), float("inf"))
        self.assertLess(stats.noise_floor_relative([1.0, 1.01, 0.99, 1.0, 1.0]), 0.05)

    def test_auto_margin_clamped(self):
        self.assertGreaterEqual(stats.auto_margin([1, 1, 1, 1, 1]), 0.005)
        self.assertLessEqual(stats.auto_margin([1.0, 1.5, 0.5, 1.2, 0.7]), 0.05)


if __name__ == "__main__":
    unittest.main()
