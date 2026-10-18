from types import SimpleNamespace
import unittest
import numpy as np
from expertcache.summary import histogram_hit_summary, TaskHitRateSummary


class TaskSummaryTests(unittest.TestCase):
    def test_histogram_matches_expanded_linear_percentiles(self):
        rng = np.random.default_rng(20261018)
        for size in (1, 2, 7, 117, 16385):
            counts = rng.integers(0, 7, size=size)
            result = histogram_hit_summary(np.bincount(counts, minlength=7), 6)
            expected = np.percentile(counts / 6, [90, 50, 10, 1], method='linear')
            for field, value in zip(('p10', 'p50', 'p90', 'p99'), expected):
                self.assertAlmostEqual(result[field], value, places=14)
            self.assertAlmostEqual(result['mean'], (counts / 6).mean(), places=14)

    def test_task_groups_use_annotation_weights_across_shards(self):
        class Detail:
            def access_chunks(self, phase):
                assert phase == 'evaluation'
                for rows in ([(1, 1), (0, 0)], [(0, 4), (0, 4)]):
                    yield phase, np.array(rows, dtype=[('request_index', '<u4'), ('hit_count', '<u4')])
        manifest = SimpleNamespace(dataset=SimpleNamespace(top_k=4), requests={'evaluation': (
            SimpleNamespace(metadata={'task_type': 'math'}), SimpleNamespace(metadata={'task_type': 'chat'}))})
        groups = TaskHitRateSummary().process(Detail(), manifest)['groups']
        self.assertEqual(groups['math']['count'], 3)
        self.assertEqual(groups['math']['requests'], 1)
        self.assertAlmostEqual(groups['math']['mean'], 2/3)
        self.assertEqual(groups['math']['hit_count_histogram'], [1, 0, 0, 0, 2])
        self.assertEqual(groups['math']['expert_hits'], 8)
        self.assertAlmostEqual(groups['math']['p90'], .2)
        self.assertEqual(groups['chat']['mean'], .25)
