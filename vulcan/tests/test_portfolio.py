import math
import unittest

from vulcan import portfolio


class TestPortfolio(unittest.TestCase):
    def test_halving_finds_best_arm(self):
        quality = {"a": 0.2, "b": 0.9, "c": 0.4, "d": 0.6, "e": 0.5, "f": 0.3}

        def ev(arm, hrs):
            return quality[arm] * (1 - math.exp(-hrs))

        res = portfolio.successive_halving(list(quality), ev, [1, 3, 9], eta=2)
        self.assertEqual(res.best.name, "b")
        self.assertTrue(res.history)

    def test_halving_single_arm(self):
        res = portfolio.successive_halving(["only"], lambda a, h: 1.0, [1, 2])
        self.assertEqual(res.best.name, "only")

    def test_bad_eta(self):
        with self.assertRaises(ValueError):
            portfolio.successive_halving(["a"], lambda a, h: 1.0, [1], eta=1)

    def test_ucb_identifies_best(self):
        quality = {"a": 0.1, "b": 0.2, "c": 0.95}
        best, pulls = portfolio.ucb_identify_best(
            list(quality), lambda a: quality[a], total_pulls=60)
        self.assertEqual(best, "c")
        self.assertGreater(pulls["c"], pulls["a"])

    def test_ucb1_averages(self):
        b = portfolio.UCB1(["x"])
        b.update("x", 1.0)
        b.update("x", 3.0)
        self.assertEqual(b.means["x"], 2.0)

    def test_hypotheses_are_distinct(self):
        hs = portfolio.architecture_hypotheses("009_gemm")
        names = [h[0] for h in hs]
        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(len(names), 4)


if __name__ == "__main__":
    unittest.main()
