import unittest

from vulcan import roofline as rl


class TestRoofline(unittest.TestCase):
    def setUp(self):
        self.peak = rl.HardwarePeak(1.0e12, 100.0e12)  # ridge = 100 flop/byte

    def test_memory_bound(self):
        c = rl.Counters(elapsed_s=1.0, dram_bytes=8.0e12, flops=8.0e11)  # AI = 0.1
        d = rl.diagnose(c, self.peak)
        self.assertEqual(d.bound, "memory")
        self.assertGreater(d.pct_of_bw_ceiling, 90)
        self.assertTrue(d.levers)

    def test_compute_bound(self):
        c = rl.Counters(elapsed_s=1.0, dram_bytes=1.0e10, flops=8.0e13)  # AI = 8000
        d = rl.diagnose(c, self.peak)
        self.assertEqual(d.bound, "compute")

    def test_latency_bound(self):
        c = rl.Counters(elapsed_s=1.0, dram_bytes=1.0e8, flops=1.0e10)
        d = rl.diagnose(c, self.peak)
        self.assertEqual(d.bound, "latency")

    def test_features_present(self):
        c = rl.Counters(1.0, 1e11, 1e12)
        d = rl.diagnose(c, self.peak)
        for k in ("dram_pct", "tensor_pct", "bytes_per_flop", "arithmetic_intensity"):
            self.assertIn(k, d.features)

    def test_bad_elapsed(self):
        with self.assertRaises(ValueError):
            rl.Counters(0.0, 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
