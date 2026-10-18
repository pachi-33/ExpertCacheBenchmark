from types import SimpleNamespace
import unittest
import numpy as np
from expertcache.summary import HitRateSummary


class SummaryTests(unittest.TestCase):
    def summarize(self, counts):
        class Detail:
            def access_chunks(self, phase):
                assert phase == "evaluation"
                yield phase, np.array([(x,) for x in counts], dtype=[("hit_count", "u4")])
        return HitRateSummary().process(Detail(), SimpleNamespace(dataset=SimpleNamespace(top_k=4)))

    def test_descending_percentile_ranks(self):
        result = self.summarize([0, 1, 2, 3, 4])
        for name, expected in {"p10": .9, "p50": .5, "p90": .1, "p99": .01}.items():
            self.assertAlmostEqual(result[name], expected)
        self.assertEqual(result["percentile_order"], "descending")
        self.assertEqual(result["mean"], .5)
        self.assertEqual(result["median"], .5)

    def test_empty_decisions(self):
        result = self.summarize([])
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["percentile_order"], "descending")
        for name in ("p10", "p50", "p90", "p99", "mean", "median"):
            self.assertIsNone(result[name])

    def test_single_decision(self):
        result = self.summarize([3])
        for name in ("p10", "p50", "p90", "p99", "mean", "median"):
            self.assertEqual(result[name], .75)
