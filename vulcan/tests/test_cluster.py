import unittest

from vulcan import cluster


class TestCluster(unittest.TestCase):
    def test_exact_kmeans_recovers_groups(self):
        vals = [1.0, 1.1, 1.2, 10.0, 10.2, 9.9]
        assign, centers = cluster.kmeans_1d_exact(vals, 2)
        self.assertEqual(assign[0], assign[1])
        self.assertEqual(assign[0], assign[2])
        self.assertEqual(assign[3], assign[4])
        self.assertNotEqual(assign[0], assign[3])
        self.assertEqual(len(centers), 2)

    def test_choose_k(self):
        vals = [1.0, 1.05, 1.1, 10.0, 10.05, 10.1]
        self.assertEqual(cluster.choose_k_exact(vals, max_k=5, variance_explained=0.95), 2)

    def test_single_value(self):
        self.assertEqual(cluster.choose_k_exact([3.0]), 1)
        assign, centers = cluster.kmeans_1d_exact([3.0], 1)
        self.assertEqual(assign, [0])

    def test_cluster_values_weights(self):
        clusters = cluster.cluster_values([1, 1.1, 10, 10.1], k=2, weights=[2, 2, 1, 1])
        weights = sorted(c.weight for c in clusters)
        self.assertEqual(weights, [2.0, 4.0])

    def test_cluster_workloads_indices(self):
        items = [{"n": 128}, {"n": 130}, {"n": 8192}, {"n": 8000}]
        groups = cluster.cluster_workloads(items, lambda it: it["n"], k=2)
        self.assertEqual(sorted(len(g) for g in groups), [2, 2])
        small = [g for g in groups if max(items[i]["n"] for i in g) < 1000][0]
        self.assertEqual(sorted(small), [0, 1])

    def test_representative_is_real_item(self):
        items = [{"n": 100}, {"n": 200}, {"n": 9000}]
        idx = cluster.representative_index(items, [0, 1], lambda it: it["n"])
        self.assertIn(idx, (0, 1))


if __name__ == "__main__":
    unittest.main()
