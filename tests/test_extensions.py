import json
import random
from pathlib import Path
from unittest.mock import patch
import numpy as np
from expertcache.api import CachePlan, CachePolicy, Action, RouteBatch, Step, StepRow
from expertcache.contracts import RunConfig, CacheConfig, SchedulerConfig, InitConfig, PolicyConfig, RunManifest
from expertcache.data import Dataset
from expertcache.detail import DetailReader
from expertcache.engine import CacheEngine
from expertcache.policies.lfu import LFU
from expertcache.runner import Benchmark
from expertcache.scheduler import DecodeScheduler
from expertcache.summary import postprocess
from test_benchmark import Fixture


class CustomSummary:
    namespace = "custom"
    def process(self, detail, manifest):
        return {"all_hit_decisions": sum(int(np.count_nonzero(a["hit_count"] == manifest.dataset.top_k))
                                          for _, a in detail.access_chunks())}


class BrokenSummary:
    namespace = "broken"
    def process(self, detail, manifest): raise RuntimeError("intentional summary failure")


class ExtensionTests(Fixture):
    def test_custom_summary_can_rerun_without_trace_or_policy(self):
        path = self.dataset([[[[0]], [[0]]]])
        result, _, output = self.run_bench(path, summary_hooks=("test_extensions:CustomSummary",))
        self.assertEqual(result["custom"], {"all_hit_decisions": 1})
        path.unlink()
        with patch.object(Benchmark, "run", side_effect=AssertionError("must not replay")):
            self.assertEqual(result, postprocess(output))

    def test_failed_summary_preserves_reprocessable_detail(self):
        path = self.dataset([[[[0]]]])
        with self.assertRaisesRegex(RuntimeError, "summary failure"):
            self.run_bench(path, summary_hooks=("test_extensions:BrokenSummary",))
        output = self.root / "run"
        self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "detail_complete")
        result = postprocess(output, extra_hooks=[])
        self.assertEqual(result["hit_rate"]["count"], 1)

    def test_continuous_batching_and_layer_order(self):
        path = self.dataset([[[[0], [1]]] * 1, [[[2], [3]]] * 3, [[[4], [5]]] * 2])
        _, reader, _ = self.run_bench(path, batch_size=2)
        actual = [(r.step_id, r.request_id, r.token_index, r.layer_id) for r in reader.records()]
        expected = []
        for step, rows in enumerate([[("data-0", 0), ("data-1", 0)], [("data-1", 1), ("data-2", 0)], [("data-1", 2), ("data-2", 1)]]):
            for layer in (0, 1):
                expected.extend((step, req, token, layer) for req, token in rows)
        self.assertEqual(actual, expected)

    def test_oracle_is_explicit_and_context_readonly(self):
        path = self.dataset([[[[0]], [[1]]]])
        outer = self
        class OraclePolicy(CachePolicy):
            def on_request_start(self, ctx, request):
                outer.assertEqual(ctx.oracle.routes(request.request_id, 1, 0), (1,))
                with outer.assertRaises(PermissionError): ctx.history.routes(request.request_id, 1, 0)
                with outer.assertRaises(TypeError): ctx.catalog[(0, 0)] = 999
                with outer.assertRaises(AttributeError): ctx.cache.resident.add((0, 0))
                return CachePlan()
        self.run_bench(path, policy=OraclePolicy(), visibility="oracle")

    def test_lfu_heap_matches_independent_reference(self):
        rng = random.Random(31415)
        path = self.dataset([[[[0, 1], [0, 1]]]])
        spec = Dataset(path).spec
        for allocation in ("global", "fixed_per_layer"):
            for capacity in (0, 1, 2, 5, 10):
                quotas = {0: capacity // 2, 1: capacity - capacity // 2} if allocation != "global" else {}
                config = RunConfig(dataset=str(path), cache=CacheConfig(capacity_units=capacity, allocation=allocation,
                                                                        layer_capacity_units=quotas))
                policy = LFU()
                bench = Benchmark(config, policy=policy)
                resident, freq, last = set(), {}, {}
                for tick in range(200):
                    layer = tick % 2
                    ids = tuple(tuple(rng.sample(range(8), 2)) for _ in range(3))
                    routes = RouteBatch(Step(tick, tuple(StepRow(f"r{i}", tick, i) for i in range(3))), layer, ids)
                    probe = bench.engine.probe(routes)
                    expected_hits = tuple(tuple((layer, e) in resident for e in row) for row in ids)
                    self.assertEqual(probe.hit_mask, expected_hits)
                    from collections import Counter
                    for key, n in Counter(routes.keys).items():
                        freq[key] = freq.get(key, 0) + n
                        last[key] = tick + 1
                    snapshot_missing = tuple(k for k in dict.fromkeys(routes.keys) if k not in resident)
                    for key in snapshot_missing:
                        limit = quotas.get(layer, capacity)
                        if not capacity or not limit: continue
                        while len(resident) >= capacity or sum(k[0] == layer for k in resident) >= limit:
                            layer_full = sum(k[0] == layer for k in resident) >= limit
                            eligible = [k for k in resident if not layer_full or k[0] == layer]
                            victim = min(eligible, key=lambda k: (freq.get(k, 0), last.get(k, -1), k))
                            resident.remove(victim)
                            freq.pop(victim, None)
                            last.pop(victim, None)
                        resident.add(key)
                    freq = {k: freq[k] for k in resident}
                    last = {k: last[k] for k in resident}
                    plan = policy.on_routes_ready(bench.context("on_routes_ready"), routes, probe)
                    events = bench.engine.execute(plan, "evaluation", tick, "on_routes_ready", probe)
                    policy.on_actions_applied(bench.context("on_routes_ready"), events)
                    self.assertEqual(bench.engine.resident, resident)
                    self.assertEqual(policy.frequency, freq)
                    self.assertEqual(policy.last_seen, last)

    def test_strata_snapshot_restores_adaptation_clock(self):
        path = self.dataset([[[[1]]]])
        config = RunConfig(dataset=str(path), cache=CacheConfig(capacity_units=1), policy=PolicyConfig(name="strata", params={"adapt_every": 2, "decay": 1}),
                           initialization=InitConfig(mode="preload", experts=({"layer_id": 0, "expert_id": 0},)))
        first = self.root / "first"
        Benchmark(config).run(first)
        restore = config.model_copy(update={"initialization": InitConfig(mode="restore", snapshot_path=str(first / "final_state.json"))})
        second = self.root / "second"
        result = Benchmark(restore).run(second)
        state = json.loads((second / "final_state.json").read_text())
        self.assertEqual(state["policy"]["steps"], 2)
        self.assertEqual(state["cache"]["resident"], [[0, 1]])
        self.assertEqual(result["transfer"]["phases"]["evaluation"]["adaptation_units"], 1)

    def test_benchmark_instance_cannot_leak_state_into_second_run(self):
        path = self.dataset([[[[0]]]])
        bench = Benchmark(RunConfig(dataset=str(path), cache=CacheConfig(capacity_units=1)))
        bench.run(self.root / "first")
        with self.assertRaisesRegex(RuntimeError, "single-use"):
            bench.run(self.root / "second")

    def test_probe_tick_reconstructs_cross_layer_prefetch(self):
        class Prefetch(CachePolicy):
            def on_layer_end(self, ctx, result):
                if ctx.layer_id == 0:
                    return CachePlan((Action("load", (1, 1), reason="prefetch"),))
                return CachePlan()
        path = self.dataset([[[[0], [1]]]])
        _, reader, _ = self.run_bench(path, policy=Prefetch())
        records = list(reader.records())
        events = list(reader.events())
        prefetch = next(e for e in events if e['kind'] == 'transfer_complete' and e['reason'] == 'prefetch')
        self.assertEqual((prefetch['context_layer_id'], prefetch['layer_id']), (0, 1))
        self.assertEqual(records[1].probe_tick, prefetch['logical_tick'])
        self.assertEqual(records[1].hit_count, 1)
        resident, cursor = set(), 0
        for record in records:
            while cursor < len(events) and events[cursor]['logical_tick'] <= record.probe_tick:
                event = events[cursor]
                if event['kind'] == 'transfer_complete' and event['destination'] == 'cache':
                    resident.add((event['layer_id'], event['expert_id']))
                if event['kind'] == 'evict': resident.remove((event['layer_id'], event['expert_id']))
                cursor += 1
            self.assertEqual(record.hit_mask, tuple((record.layer_id, e) in resident for e in record.expert_ids))
