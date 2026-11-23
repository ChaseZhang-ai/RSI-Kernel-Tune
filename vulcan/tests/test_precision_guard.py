import unittest

from vulcan import precision_guard as pg


class TestPrecisionGuard(unittest.TestCase):
    def test_accum_downgrade_fails(self):
        req = pg.default_contract("rmsnorm")          # accum fp32
        bad = pg.PrecisionContract("bf16", "bf16", "bf16")
        codes = [f.code for f in pg.check_contract(bad, req)]
        self.assertIn("accum-precision", codes)

    def test_clean_contract_passes(self):
        req = pg.default_contract("gemm")
        ok = pg.PrecisionContract("fp16", "fp32", "fp16")
        self.assertEqual(pg.check_contract(ok, req), [])

    def test_exact_requires_identical(self):
        req = pg.default_contract("pack")
        near = pg.PrecisionContract("int8", "int16", "int8", exact=True)
        findings = pg.check_contract(near, req)
        self.assertTrue(any(f.code == "exactness" for f in findings))

    def test_output_narrower_fails(self):
        req = pg.PrecisionContract("bf16", "fp32", "fp32")
        cand = pg.PrecisionContract("bf16", "fp32", "fp16")
        self.assertTrue(any(f.code == "output-precision"
                            for f in pg.check_contract(cand, req)))

    def test_probes_cover_key_hacks(self):
        names = {p.name for p in pg.adversarial_probes()}
        self.assertIn("correlated_keys", names)
        self.assertIn("long_memory", names)
        self.assertIn("near_cancellation", names)
        self.assertIn("zero_output_guard", names)

    def test_hack_scan(self):
        src = (
            "import torch.nn.functional as F\n"
            "def run(x):\n"
            "    return F.rms_norm(x)\n"
        )
        codes = {f.code for f in pg.reward_hack_scan(src)}
        self.assertIn("calls-torch-op", codes)

        src2 = "import flashinfer\ndef run(x):\n    return x\n"
        codes2 = {f.code for f in pg.reward_hack_scan(src2)}
        self.assertIn("calls-flashinfer", codes2)
        self.assertIn("returns-input", codes2)

    def test_clean_source_has_no_fail(self):
        src = "def run(x):\n    return x * 2.0\n"
        repairs = [f for f in pg.reward_hack_scan(src) if f.severity == "fail"]
        self.assertEqual(repairs, [])

    def test_gate(self):
        req = pg.default_contract("attention")
        ok = pg.PrecisionContract("bf16", "fp32", "bf16")
        self.assertEqual(pg.evaluate_gate(ok, req).verdict, "PASS")
        bad = pg.PrecisionContract("bf16", "bf16", "bf16")
        self.assertEqual(pg.evaluate_gate(bad, req).verdict, "FAIL")
        self.assertEqual(
            pg.evaluate_gate(ok, req, probe_failures=["long_memory"]).verdict, "FAIL")


if __name__ == "__main__":
    unittest.main()
