import unittest

from vulcan import calibration as cal
from vulcan.objective import Workload


class TestCalibration(unittest.TestCase):
    def test_affine_fit_exact(self):
        samples = [(1.0, 2.05), (2.0, 4.05), (3.0, 6.05), (4.0, 8.05)]
        c = cal.AffineCalibration.fit(samples)
        self.assertAlmostEqual(c.a, 2.0, places=9)
        self.assertAlmostEqual(c.b, 0.05, places=9)
        self.assertAlmostEqual(c.residual_std, 0.0, places=9)
        self.assertAlmostEqual(c.r2, 1.0, places=9)

    def test_ratio_fit(self):
        c = cal.RatioCalibration.fit([(1.0, 3.0), (2.0, 6.0), (3.0, 9.0)])
        self.assertAlmostEqual(c.c, 3.0, places=9)
        self.assertAlmostEqual(c.predict(2.0), 6.0)

    def test_score_roundtrip_through_calibration(self):
        w = Workload("w", 1.0, 0.1)
        c = cal.AffineCalibration.fit([(1.0, 2.05), (2.0, 4.05), (3.0, 6.05)])
        dt = cal.device_target(0.8, w, c)
        s, lo, hi = cal.predicted_score(dt, w, c)
        self.assertAlmostEqual(s, 0.8, places=6)
        self.assertLessEqual(lo, s + 1e-9)
        self.assertGreaterEqual(hi, s - 1e-9)

    def test_assumed_calibration_has_interval(self):
        w = Workload("w", 1.0, 0.1)
        c = cal.AffineCalibration.assumed(ratio=0.5)
        self.assertTrue(c.interval_is_assumption())
        s, lo, hi = cal.predicted_score(0.5, w, c)
        self.assertLess(lo, s)
        self.assertGreater(hi, s)

    def test_fit_needs_two_points(self):
        with self.assertRaises(ValueError):
            cal.AffineCalibration.fit([(1.0, 2.0)])

    def test_fit_from_baselines(self):
        board = [1.0, 2.0, 4.0]
        device = [3.0, 6.0, 12.0]
        c = cal.fit_from_baselines(board, device)
        self.assertAlmostEqual(c.c, 1.0 / 3.0, places=9)

    def test_expected_collection_and_submit(self):
        mean, lo, hi = cal.expected_collection([(0.83, 0.81, 0.85), (0.80, 0.78, 0.82)])
        self.assertAlmostEqual(mean, 0.815)
        self.assertAlmostEqual(lo, 0.795)
        self.assertAlmostEqual(hi, 0.835)
        self.assertTrue(cal.should_submit((0.83, 0.82, 0.85), 0.815321, require_confidence=True))
        self.assertFalse(cal.should_submit((0.83, 0.81, 0.85), 0.815321, require_confidence=True))
        self.assertTrue(cal.should_submit((0.83, 0.81, 0.85), 0.815321, require_confidence=False))


if __name__ == "__main__":
    unittest.main()
