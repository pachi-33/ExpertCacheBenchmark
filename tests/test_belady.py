from itertools import combinations
import json
import random
from pathlib import Path
import importlib.util
import numpy as np
from expertcache.api import CachePolicy, CachePlan
from expertcache.contracts import CacheConfig, InitConfig, PolicyConfig, RunConfig, SchedulerConfig
from expertcache.runner import Benchmark
from expertcache.scheduler import DecodeScheduler
from test_benchmark import Fixture


def exhaustive_loads(groups, capacity, initial):
    """Independent dynamic program over every legal post-demand cache subset."""
    states = {frozenset(initial): 0}
    for group in groups:
        following = {}
        for resident, cost in states.items():
            cost += len(group - resident)
            available = sorted(resident | group)
            for size in range(min(capacity, len(available)) + 1):
                for kept in combinations(available, size):
                    kept = frozenset(kept)
                    following[kept] = min(following.get(kept, float("inf")), cost)
        states = following
    return min(states.values())


class BeladyTests(Fixture):
    def belady(self, path, capacity=1, initial=(), **kwargs):
        init = InitConfig(mode="preload", experts=tuple(dict(layer_id=l, expert_id=e) for l, e in initial)) if initial else InitConfig()
        return self._run(path, capacity, init, **kwargs)

    def _run(self, path, capacity, init, name="run", scheduler=None):
        config = RunConfig(dataset=str(path), cache=CacheConfig(capacity_units=capacity),
                           initialization=init, visibility="oracle", policy=PolicyConfig(name="belady"),
                           scheduler=scheduler or SchedulerConfig())
        output = self.root / name
        result = Benchmark(config).run(output)
        from expertcache.contracts import RunManifest
        from expertcache.detail import DetailReader
        manifest = RunManifest.model_validate_json((output / "manifest.json").read_text())
        return result, DetailReader(output, manifest), output

    def test_bypass_avoids_evicting_nearer_reuse(self):
        path = self.dataset([[[[0]], [[1]], [[0]]]])
        result, reader, _ = self.belady(path, initial=((0, 0),))
        self.assertEqual([r.hit_count for r in reader.records()], [1, 0, 1])
        self.assertEqual(result["transfer"]["phases"]["evaluation"]["load_count"], 1)

    def test_batch_snapshot_deduplicates_demand_without_creating_hits(self):
        path = self.dataset([[[[0]], [[0]]]] * 4)
        result, reader, _ = self.belady(path, scheduler=SchedulerConfig(batch_size=4))
        self.assertEqual([r.hit_count for r in reader.records()], [0] * 4 + [1] * 4)
        self.assertEqual(result["transfer"]["phases"]["evaluation"]["load_count"], 1)

    def test_matches_exhaustive_batched_optimum(self):
        rng = random.Random(719)
        for capacity in range(4):
            for trial in range(4):
                traces = [[[[rng.randrange(3)] for _ in range(2)] for _ in range(rng.randrange(1, 5))] for _ in range(3)]
                name = f"case-{capacity}-{trial}"
                path = self.dataset(traces, name=name)
                initial = tuple((l, e) for l, e in rng.sample([(l, e) for l in range(2) for e in range(3)], capacity))
                result, reader, _ = self.belady(path, capacity=capacity, initial=initial,
                                               name=f"run-{name}", scheduler=SchedulerConfig(batch_size=2))
                groups = {}
                for row in reader.records():
                    groups.setdefault((row.step_id, row.layer_id), set()).update((row.layer_id, e) for e in row.expert_ids)
                expected = exhaustive_loads(list(groups.values()), capacity, initial)
                self.assertEqual(result["transfer"]["phases"]["evaluation"]["load_count"], expected, name)

    def test_requires_explicit_oracle_and_uniform_global_pool(self):
        path = self.dataset([[[[0]]]])
        config = RunConfig(dataset=str(path), cache=CacheConfig(capacity_units=1), policy=PolicyConfig(name="belady"))
        with self.assertRaisesRegex(ValueError, "visibility=oracle"):
            Benchmark(config).run(self.root / "causal")
        for cache in (CacheConfig(capacity_units=1, expert_sizes=({"layer_id": 0, "expert_id": 0, "size_units": 2},), workspace_units=2),
                      CacheConfig(capacity_units=1, allocation="fixed_per_layer", layer_capacity_units={0: 1})):
            with self.assertRaises(ValueError):
                Benchmark(config.model_copy(update={"cache": cache, "visibility": "oracle"})).run(self.root / f"bad-{cache.allocation}")

    def test_restore_rebuilds_future_index_and_warmup_is_phase_scoped(self):
        warmup = self.dataset([[[[0]], [[1]]]], name="warmup")
        evaluation = self.dataset([[[[1]], [[0]], [[1]]]], name="eval")
        config = RunConfig(dataset=str(evaluation), cache=CacheConfig(capacity_units=1),
                           initialization=InitConfig(mode="warmup", warmup_dataset=str(warmup)),
                           policy=PolicyConfig(name="belady"), visibility="oracle")
        output = self.root / "warm-run"
        first = Benchmark(config).run(output)
        restore = config.model_copy(update={"initialization": InitConfig(mode="restore", snapshot_path=str(output / "final_state.json"))})
        restored = Benchmark(restore).run(self.root / "restore")
        self.assertEqual(first["hit_rate"], restored["hit_rate"])

    def test_oracle_schedule_matches_shuffled_limited_continuous_batching(self):
        path = self.dataset([[[[i], [i]]] * (i + 1) for i in range(5)])
        captured, actual = [], []
        class Policy(CachePolicy):
            def on_initialize(self, ctx, profile):
                captured.extend(ctx.oracle.batches())
                return CachePlan()
            def on_run_start(self, ctx):
                self.saved = ctx.oracle
                return CachePlan()
            def on_routes_ready(self, ctx, routes, probe):
                actual.append(routes)
                return CachePlan()
        class Scheduler(DecodeScheduler):
            calls = 0
            def steps(self, *args):
                self.calls += 1
                yield from super().steps(*args)
        scheduler, policy = Scheduler(), Policy()
        config = RunConfig(dataset=str(path), cache=CacheConfig(capacity_units=2), visibility="oracle",
                           scheduler=SchedulerConfig(batch_size=2, order="shuffled", seed=5, request_offset=1, max_requests=3, max_tokens=2))
        Benchmark(config, policy=policy, scheduler=scheduler).run(self.root / "oracle")
        self.assertEqual(captured, actual)
        self.assertEqual(list(policy.saved.batches()), actual)
        self.assertEqual(scheduler.calls, 1)

    def test_weighted_hit_rate_is_not_claimed_optimal(self):
        path = self.dataset([[[[0]], [[0]], [[1]]], [[[1]], [[2]], [[1]]], [[[1]], [[3]], [[1]]]])
        result, reader, output = self.belady(path, initial=((0, 0),), scheduler=SchedulerConfig(batch_size=3))
        self.assertEqual(sum(r.hit_count for r in reader.records()), 2)
        metadata = json.loads((output / "manifest.json").read_text())["policy_metadata"]
        self.assertFalse(metadata["occurrence_weighted_hit_rate_optimal"])
        self.assertEqual(result["transfer"]["phases"]["evaluation"]["load_count"], 4)

    def test_independent_event_and_greedy_boundary_audit(self):
        spec = importlib.util.spec_from_file_location("sweep_audit",Path(__file__).resolve().parents[1]/"scripts/verify_capacity_sweep.py")
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        traces=[[[[i%3],[(i+1)%3]], [[(i+1)%3],[i%3]], [[i%3],[i%3]]] for i in range(4)]
        path=self.dataset(traces)
        source=np.asarray(traces,dtype=np.uint8)
        for batch in (1,2,4):
            with self.subTest(batch_size=batch):
                _,reader,output=self.belady(path,capacity=2,initial=((0,0),(1,1)),
                                           name=f'audit-bs{batch}',scheduler=SchedulerConfig(batch_size=batch))
                initial=json.loads((output/"initial_state.json").read_text())
                masks=np.concatenate([a['hit_mask'] for _,a in reader.access_chunks()])
                result=audit.check_belady(output,reader.manifest,source,initial,masks)
                self.assertEqual(result['belady_greedy_boundaries_verified'],24//batch)
                masks[0,0]=~masks[0,0]
                with self.assertRaisesRegex(AssertionError,'snapshot'):
                    audit.check_belady(output,reader.manifest,source,initial,masks)
