import contextlib
import io
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from expertcache.cli import main
from expertcache.contracts import TokenPositionSummary, TokenPositionPoint
from expertcache.data import digest_file
from expertcache.detail import DetailReader
from expertcache.summary import TokenPositionHitRateSummary, postprocess
from test_benchmark import Fixture

HOOK = 'expertcache.summary:TokenPositionHitRateSummary'
DTYPE = [('request_index', '<u4'), ('token_index', '<u4'), ('layer_id', '<u4'), ('hit_count', '<u4')]


class TokenPositionTests(unittest.TestCase):
    def test_rolling_mean_weights_actual_accesses(self):
        plot = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/plot_token_position.py'))
        points = [SimpleNamespace(expert_hits=h, expert_accesses=a) for h, a in ((1, 2), (1, 10), (4, 4))]
        np.testing.assert_allclose(plot['rolling_mean'](points, 1), [.5, .1, 1.])
        np.testing.assert_allclose(plot['rolling_mean'](points, 2), [.5, 2/12, 5/14])

    def fixture(self, max_tokens=None):
        values = [[[0, 2], [2, 2], [1, 1]], [[1, 1]], [[0, 0], [0, 2]]]
        requests = tuple(SimpleNamespace(token_count=len(v), metadata={'task_type': task})
                         for v, task in zip(values, ('math', 'math', 'chat')))
        manifest = SimpleNamespace(requests={'evaluation': requests},
            dataset=SimpleNamespace(valid_layers=(1, 3), top_k=2),
            config=SimpleNamespace(scheduler=SimpleNamespace(max_tokens=max_tokens)))
        rows = [(r, t, l, hit) for r, tokens in enumerate(values)
                for t, hits in enumerate(tokens[:max_tokens]) for l, hit in zip((1, 3), hits)]
        return manifest, np.array(rows, dtype=DTYPE)

    def process(self, manifest, rows, shard_size=3):
        class Detail:
            def access_chunks(self, phase):
                assert phase == 'evaluation'
                for start in range(0, len(rows), shard_size):
                    yield phase, rows[start:start + shard_size]
        return TokenPositionHitRateSummary().process(Detail(), manifest)

    def test_aggregate_layers_before_request_percentiles_and_ragged_support(self):
        manifest, rows = self.fixture()
        np.random.default_rng(7).shuffle(rows)
        result = self.process(manifest, rows)
        self.assertEqual([p['request_count'] for p in result['overall']], [3, 2, 1])
        self.assertEqual(result['metadata']['request_token_count'], 6)
        self.assertEqual(result['metadata']['annotation_count'], 12)
        self.assertEqual(result['metadata']['sample_unit'], 'request_token')
        p = result['overall'][0]
        self.assertEqual(p['expert_hits'], 4)
        self.assertEqual(p['expert_accesses'], 12)
        self.assertAlmostEqual(p['mean'], 1/3)
        for field, expected in [('p10', .5), ('p50', .5), ('p90', .1), ('p99', .01)]:
            self.assertAlmostEqual(p[field], expected)
        self.assertEqual([p['mean'] for p in result['by_task']['chat']], [0., .5])
        self.assertEqual([p['mean'] for p in result['by_task']['math']], [.5, 1., .5])

    def test_scheduler_token_limit_and_missing_task_metadata(self):
        manifest, rows = self.fixture(max_tokens=2)
        manifest.requests['evaluation'][0].metadata = {}
        result = self.process(manifest, rows, 1)
        self.assertEqual(len(result['overall']), 2)
        self.assertEqual(result['metadata']['request_token_count'], 5)
        self.assertIn('unknown', result['by_task'])

    def test_duplicate_rows_rejected_within_and_across_shards(self):
        manifest, rows = self.fixture()
        bad = np.concatenate((rows, rows[:1]))
        for size in (1, 100):
            with self.subTest(shard_size=size), self.assertRaisesRegex(ValueError, 'duplicate'):
                self.process(manifest, bad, size)

    def test_missing_layer_and_invalid_indices_rejected(self):
        manifest, rows = self.fixture()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.process(manifest, rows[:-1])
        for field, value in [('request_index', 3), ('token_index', 3), ('layer_id', 0), ('layer_id', 9), ('hit_count', 3)]:
            bad = rows.copy()
            bad[field][0] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.process(manifest, bad)

    def test_more_than_64_layers_and_empty_evaluation(self):
        manifest, _ = self.fixture(max_tokens=1)
        manifest.dataset.valid_layers = tuple(range(70))
        rows = np.array([(r, 0, l, r) for r in range(3) for l in range(70)], dtype=DTYPE)
        result = self.process(manifest, rows, 31)
        self.assertEqual(result['metadata']['expert_accesses_per_request_token'], 140)
        self.assertEqual(result['overall'][0]['mean'], .5)
        manifest.requests['evaluation'] = ()
        result = self.process(manifest, rows[:0])
        self.assertEqual(result['overall'], [])
        self.assertEqual(result['by_task'], {})

    def test_contract_rejects_bad_means_and_ascending_percentiles(self):
        manifest, rows = self.fixture()
        result = self.process(manifest, rows)
        for change in ({'mean': .9}, {'p10': .01, 'p99': .5}):
            with self.assertRaises(ValueError):
                TokenPositionPoint.model_validate(dict(result['overall'][0], **change))
        result['by_task'].pop('math')
        with self.assertRaises(ValueError):
            TokenPositionSummary.model_validate(result)


class TokenPositionIntegrationTests(Fixture):
    def test_real_bs2_ragged_detail_matches_brute_force(self):
        rng = np.random.default_rng(9)
        traces = [[[[int(e) for e in rng.choice(8, 3, replace=False)] for _ in range(4)]
                   for _ in range(length)] for length in (3, 5, 2)]
        path = self.dataset(traces, valid_layers=(1, 3))
        result, reader, _ = self.run_bench(path, capacity=5, batch_size=2, detail_chunk_rows=1,
                                         summary_hooks=(HOOK,))
        curve = result['hit_rate_by_token_position']
        grouped = {}
        for row in reader.records():
            grouped.setdefault((row.request_id, row.token_index), []).append(row.hit_count)
        for point in curve['overall']:
            values = [sum(hits)/6 for (_, token), hits in grouped.items() if token == point['token_index']]
            self.assertEqual(point['request_count'], len(values))
            self.assertAlmostEqual(point['mean'], np.mean(values))
            for name, expected in zip(('p10', 'p50', 'p90', 'p99'), np.percentile(values, [90, 50, 10, 1])):
                self.assertAlmostEqual(point[name], expected)
        self.assertAlmostEqual(sum(p['expert_hits'] for p in curve['overall']) /
                               sum(p['expert_accesses'] for p in curve['overall']), result['hit_rate']['mean'])

    def test_selective_cli_needs_no_events_routes_or_replay_and_preserves_source(self):
        path = self.dataset([[[[0, 1]], [[0, 2]]], [[[1, 2]]]])
        _, _, run = self.run_bench(path, batch_size=2, detail_chunk_rows=1)
        path.unlink()
        (run / 'detail/cache_events.jsonl.gz').unlink()
        # External analysis of a sealed run must not mark it complete as a side effect.
        p = run / 'manifest.json'
        value = json.loads(p.read_text()); value['status'] = 'detail_complete'
        p.write_text(json.dumps(value))
        before = {str(p.relative_to(run)): digest_file(p) for p in run.rglob('*') if p.is_file()}
        destination = self.root / 'analysis/summary.json'
        with (patch('expertcache.summary.TransferSummary.process', side_effect=AssertionError('event scan')),
             patch('expertcache.summary.HitRateSummary.process', side_effect=AssertionError('unselected hook')),
             patch('expertcache.runner.Benchmark.run', side_effect=AssertionError('replay')),
             patch.object(DetailReader, 'events', side_effect=AssertionError('events')),
             contextlib.redirect_stdout(io.StringIO())):
            main(['postprocess', str(run), '--only-hook', HOOK, '--output', str(destination)])
        after = {str(p.relative_to(run)): digest_file(p) for p in run.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        output = json.loads(destination.read_text())
        self.assertNotIn('transfer', output)
        self.assertNotIn('hit_rate', output)
        self.assertEqual(output['summary_hooks'], [HOOK])
        self.assertEqual(output['source']['manifest_sha256'], before['manifest.json'])
        self.assertEqual(output['postprocess']['verified_detail_files'], 3)
        self.assertEqual(output['hit_rate_by_token_position']['metadata']['request_token_count'], 3)

    def test_selective_output_guards_and_default_backward_compatibility(self):
        path = self.dataset([[[[0]], [[0]]]])
        original, _, run = self.run_bench(path)
        self.assertEqual(original, postprocess(run))
        destination = self.root / 'selected.json'
        for kwargs in (dict(only_hooks=[HOOK]), dict(only_hooks=[] ,summary_output=destination),
                       dict(only_hooks=[HOOK], extra_hooks=[], summary_output=destination),
                       dict(only_hooks=[HOOK], summary_output=run / 'summary.json')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                postprocess(run, **kwargs)
        postprocess(run, only_hooks=[HOOK], summary_output=destination)
        before = destination.read_bytes()
        with self.assertRaises(FileExistsError):
            postprocess(run, only_hooks=[HOOK], summary_output=destination)
        self.assertEqual(before, destination.read_bytes())
