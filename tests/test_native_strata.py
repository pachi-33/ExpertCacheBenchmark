import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np

from expertcache.contracts import CacheConfig, InitConfig, PolicyConfig, RunConfig, RunManifest, SchedulerConfig
from expertcache.data import Dataset, build_profile, write_json
from expertcache.detail import DetailReader
from expertcache.native_strata import NativeStrataBenchmark, _native_count_threshold
from expertcache.runner import Benchmark
from expertcache.scheduler import select_requests
from test_benchmark import Fixture


@unittest.skipUnless(shutil.which('clang++') or shutil.which('g++'), 'C++ compiler not installed')
class NativeStrataTests(Fixture):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def test_complete_trace_and_state_match_python_reference(self):
        cases = [({}, 6), ({'adapt_every': 1, 'min_count': 0, 'min_gain': 0, 'decay': .1, 'max_swaps': 3}, 8),
                 ({'adapt_every': 0}, 5), ({'max_swaps': 0}, 5),
                 ({'decay': 1., 'min_gain': .1, 'max_swaps': 1}, 16),
                 ({'adapt_every': 1, 'min_count': 2.1, 'min_gain': 0, 'decay': .7}, 6)]
        for case, (params, capacity) in enumerate(cases):
            with self.subTest(params=params):
                rng = np.random.default_rng(100 + case)
                traces = [[[[int(e) for e in rng.choice(8, 3, replace=False)] for _ in range(2)]
                           for _ in range(length)] for length in (64, 71, 52, 89)]
                path = self.dataset(traces, name=f'data-{case}')
                dataset = Dataset(path)
                scheduler = SchedulerConfig(batch_size=1, order='shuffled', seed=20261018,
                                             request_offset=1)
                ordering = select_requests(dataset, scheduler.model_copy(update={'request_offset': 0}))
                profile = build_profile(dataset, ordering[:1])
                profile_path = self.root / f'profile-{case}.json'
                write_json(profile_path, profile.model_dump(mode='json'))
                config = RunConfig(dataset=str(path), cache=CacheConfig(capacity_units=capacity),
                    initialization=InitConfig(mode='profile', profile_path=str(profile_path)),
                    scheduler=scheduler, policy=PolicyConfig(name='strata', params=params), detail_chunk_rows=17,
                    summary_hooks=('expertcache.summary:TaskHitRateSummary',))
                paths = [self.root/f'reference-{case}', self.root/f'native-{case}']
                reference = Benchmark(config).run(paths[0])
                native = NativeStrataBenchmark(config, self.build.name).run(paths[1])
                for key in ('hit_rate', 'transfer', 'hit_rate_by_task'):
                    self.assertEqual(reference[key], native[key], key)
                readers = [DetailReader(p, RunManifest.model_validate_json((p/'manifest.json').read_text())) for p in paths]
                arrays = [np.concatenate([a for _, a in reader.access_chunks()]) for reader in readers]
                np.testing.assert_array_equal(arrays[0], arrays[1])
                self.assertEqual(list(readers[0].events()), list(readers[1].events()))
                for filename in ('initial_state.json', 'final_state.json'):
                    self.assertEqual(json.loads((paths[0]/filename).read_text()), json.loads((paths[1]/filename).read_text()))
                audit = json.loads((paths[1]/'native-event-audit.json').read_text())
                self.assertEqual(audit['status'], 'verified')

    def test_rejects_unsupported_batch_size(self):
        config = RunConfig(dataset='unused', cache=CacheConfig(capacity_units=4),
                           policy=PolicyConfig(name='strata'), scheduler=SchedulerConfig(batch_size=2))
        with self.assertRaises(ValueError):
            NativeStrataBenchmark(config, self.build.name)


class NativeThresholdTests(unittest.TestCase):
    def test_float32_boundary_matches_numpy_scalar_comparison(self):
        for threshold in (0., 2., 2.1, 1/3, 16777216.5, 1e40):
            with np.errstate(over="ignore"):
                boundary = np.float32(threshold)
            scores = [np.float32(0), np.float32(1), boundary,
                      np.nextafter(boundary, np.float32(-np.inf)),
                      np.nextafter(boundary, np.float32(np.inf))]
            for score in scores:
                with self.subTest(threshold=threshold, score=score):
                    self.assertEqual(bool(score >= threshold), float(score) >= _native_count_threshold(threshold))
