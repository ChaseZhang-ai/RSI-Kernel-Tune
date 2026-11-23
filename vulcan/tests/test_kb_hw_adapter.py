import tempfile
import unittest
from pathlib import Path

from vulcan import kb_graph as kb
from vulcan import kb_hw_adapter as ad


def _graph():
    g = kb.KnowledgeGraph()
    g.add(kb.KnowledgeEntry(
        "tma_bulk_prefetch", "sm100", "TMA bulk prefetch of the next tile",
        {"dram_pct": (60, 100)}, 0.12, kb.STATUS_CERTIFIED,
        evidence="PROBE PASS tma_bulk_prefetch/issue_order",
        cost="one extra smem stage", limitations="needs 16B alignment",
        recipe="cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier ...",
        tags=["pipeline"]))
    g.add(kb.KnowledgeEntry("candidate_only", "sm100", "unproven idea",
                            {"dram_pct": (0, 100)}, 0.5, kb.STATUS_CANDIDATE))
    g.add(kb.KnowledgeEntry("refuted_8stage", "sm100", "8-stage TMA ring",
                            {"dram_pct": (60, 100)}, -0.1, kb.STATUS_REFUTED,
                            evidence="job 42 regressed"))
    return g


class TestHwKbAdapter(unittest.TestCase):
    def test_markdown_has_required_sections(self):
        e = next(iter(_graph()))
        md = ad.entry_to_markdown(e, "cutedsl")
        for header in ("## Purpose", "## When needed (trigger)",
                       "## Minimal runnable example", "## Gotchas", "## Sources"):
            self.assertIn(header, md)

    def test_predicates_rendered(self):
        md = ad.entry_to_markdown(next(iter(_graph())))
        self.assertIn("`dram_pct` in [60, 100]", md)

    def test_no_banned_tokens(self):
        md = ad.entry_to_markdown(next(iter(_graph())))
        self.assertEqual(ad.lint_text(md), [])
        self.assertEqual(ad.lint_text("see the oracle kernel"), ["oracle"])

    def test_write_only_certified(self):
        with tempfile.TemporaryDirectory() as d:
            res = ad.write_hw_kb(_graph(), d, arch="sm100", lang="cutedsl",
                                 provenance_dir=str(Path(d).parent / "prov"))
            names = sorted(Path(p).name for p in res.written)
            self.assertEqual(names, ["tma_bulk_prefetch.md"])
            self.assertTrue(res.ok)
            self.assertTrue(res.provenance)

    def test_negatives_are_separate(self):
        with tempfile.TemporaryDirectory() as d:
            res = ad.write_hw_kb(_graph(), d, arch="sm100",
                                 provenance_dir=str(Path(d).parent / "prov"),
                                 include_negatives=True)
            self.assertEqual([Path(p).name for p in res.negatives], ["negatives.md"])
            text = Path(res.negatives[0]).read_text()
            self.assertIn("NEGATIVE KNOWLEDGE", text)

    def test_arch_filter(self):
        with tempfile.TemporaryDirectory() as d:
            res = ad.write_hw_kb(_graph(), d, arch="sm90", lang="cuda")
            self.assertEqual(res.written, [])


if __name__ == "__main__":
    unittest.main()
