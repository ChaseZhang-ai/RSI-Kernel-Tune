import math
import unittest
from pathlib import Path

from vulcan import dataset, objective

DS = Path(__file__).resolve().parents[1] / "data" / "sol_execbench_b200_flashinfer.json"


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = dataset.load(str(DS))

    def test_shape(self):
        self.assertEqual(len(self.ds["kernels"]), 26)
        total = sum(e["n_workloads"] for e in self.ds["kernels"].values())
        self.assertEqual(total, 660)

    def test_published_scores_reproduced(self):
        self.assertAlmostEqual(dataset.collection_score_of(self.ds, "forgekernel"),
                               0.815204, places=6)
        self.assertAlmostEqual(dataset.collection_score_of(self.ds, "golden_bridge"),
                               0.815321, places=6)
        self.assertAlmostEqual(dataset.collection_score_of(self.ds, "best"),
                               0.819011, places=6)

    def test_workload_bounds_valid(self):
        for name, e in self.ds["kernels"].items():
            for w in e["workloads"]:
                self.assertGreater(w["t_b_ms"], w["t_sol_ms"], name)
                self.assertGreater(w["t_sol_ms"], 0.0, name)

    def test_build_specs_matches_scores(self):
        specs = dataset.build_specs(self.ds, current="forgekernel")
        self.assertEqual(len(specs), 26)
        for s in specs:
            published = self.ds["kernels"][s.name]["scores"]["forgekernel"]
            self.assertAlmostEqual(s.current_score(), published, places=6)

    def test_build_specs_honours_fast1_misses(self):
        specs = dataset.build_specs(self.ds, current="forgekernel", honor_fast1=True)
        fast = total = 0
        for s in specs:
            fast += sum(1 for w in s.state.workloads if w.is_fast_at(0.0))
            total += len(s.state.workloads)
        published = self.ds["meta"]["published"]["forgekernel"]
        self.assertEqual(total, 660)
        self.assertEqual(fast, 660 - 13)          # ForgeKernel's published Fast_1 count

    def test_best_current_has_no_miss_data_so_is_clean(self):
        specs = dataset.build_specs(self.ds, current="best", honor_fast1=True)
        fast = sum(1 for s in specs for w in s.state.workloads if w.is_fast_at(0.0))
        self.assertEqual(fast, 660)

    def test_headroom_sorted(self):
        rows = dataset.headroom_table(self.ds, "forgekernel")
        heads = [r["headroom"] for r in rows]
        self.assertEqual(heads, sorted(heads, reverse=True))

    def test_best_beats_rank_one_competitor(self):
        # the central strategic claim of the plan
        self.assertGreater(dataset.collection_score_of(self.ds, "best"),
                           dataset.collection_score_of(self.ds, "golden_bridge"))

    def test_latency_target_hits_score(self):
        e = self.ds["kernels"]["009_gemm_n5120_k2048"]
        w = e["workloads"][0]
        t = objective.latency_for_score(0.75, w["t_b_ms"], w["t_sol_ms"])
        self.assertAlmostEqual(objective.sol_score(t, w["t_b_ms"], w["t_sol_ms"]), 0.75,
                               places=9)


if __name__ == "__main__":
    unittest.main()
