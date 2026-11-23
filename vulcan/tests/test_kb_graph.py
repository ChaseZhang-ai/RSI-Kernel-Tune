import unittest

from vulcan import kb_graph as kb


class TestKnowledgeGraph(unittest.TestCase):
    def _graph(self):
        g = kb.KnowledgeGraph()
        g.add(kb.KnowledgeEntry("tma", "sm100", "TMA bulk prefetch",
                                {"dram_pct": (60, 100)}, 0.12, kb.STATUS_CERTIFIED))
        g.add(kb.KnowledgeEntry("occ", "sm100", "raise occupancy",
                                {"occupancy": (0.0, 0.4)}, 0.05, kb.STATUS_CERTIFIED))
        g.add(kb.KnowledgeEntry("ws", "sm100", "warp specialization",
                                {"tensor_pct": (50, 100)}, 0.20, kb.STATUS_CERTIFIED,
                                tags=["pipeline"]))
        return g

    def test_predicate_match(self):
        g = self._graph()
        self.assertEqual([e.id for e in g.retrieve({"dram_pct": 80}, "sm100")], ["tma"])
        self.assertEqual(g.retrieve({"dram_pct": 10}, "sm100"), [])

    def test_unknown_feature_does_not_match(self):
        g = self._graph()
        # 'tma' needs dram_pct; a query without it must not retrieve tma
        self.assertNotIn("tma", [e.id for e in g.retrieve({"tensor_pct": 90}, "sm100")])
        self.assertEqual([e.id for e in g.retrieve({"occupancy": 0.1}, "sm100")], ["occ"])

    def test_ranking_by_effect(self):
        g = self._graph()
        hits = g.retrieve({"dram_pct": 80, "tensor_pct": 60, "occupancy": 0.2}, "sm100")
        self.assertEqual(hits[0].id, "ws")   # largest effect

    def test_negatives_are_separate(self):
        g = self._graph()
        g.add_negative("ws", kb.KnowledgeEntry(
            "ws-over", "sm100", "warp specialization with 8 stages",
            {"tensor_pct": (50, 100), "dram_pct": (60, 100)}, -0.1, kb.STATUS_REFUTED,
            evidence="job 42 regressed"))
        feats = {"tensor_pct": 60, "dram_pct": 70}
        cert = [e.id for e in g.retrieve(feats, "sm100")]
        neg = [e.id for e in g.negative_matches(feats, "sm100")]
        self.assertIn("ws", cert)
        self.assertNotIn("ws-over", cert)
        self.assertEqual(neg, ["ws-over"])

    def test_arch_filter(self):
        g = self._graph()
        self.assertEqual(g.retrieve({"dram_pct": 80}, "sm90"), [])

    def test_analog_lift_is_candidate(self):
        g = self._graph()
        lifted = g.lift_analog("ws", "sm90", "ws@sm90")
        self.assertEqual(lifted.status, kb.STATUS_CANDIDATE)
        self.assertEqual(lifted.analog_of, "ws")
        # a candidate is never returned as certified, on either arch
        self.assertEqual(g.retrieve({"tensor_pct": 70}, "sm90"), [])
        # but it surfaces as an analog hypothesis on the source arch
        analogs = g.retrieve_analogs({"tensor_pct": 70}, "sm100")
        self.assertEqual([e.id for e in analogs], ["ws@sm90"])

    def test_jsonl_roundtrip(self):
        g = self._graph()
        text = g.to_jsonl()
        g2 = kb.KnowledgeGraph.from_jsonl(text)
        self.assertEqual(len(g), len(g2))
        self.assertEqual([e.id for e in g2.retrieve({"dram_pct": 80}, "sm100")], ["tma"])

    def test_bad_status(self):
        with self.assertRaises(ValueError):
            kb.KnowledgeEntry("x", "sm100", "m", {}, 0.0, "bogus")


if __name__ == "__main__":
    unittest.main()
