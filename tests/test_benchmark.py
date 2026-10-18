import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from expertcache.api import CachePolicy, CachePlan, Action
from expertcache.contracts import (DatasetManifest, TraceRequest, RunConfig, CacheConfig, SchedulerConfig,
                                   InitConfig, ExpertRef, ExpertSize, RunManifest, CacheEvent, Profile, ProfileEntry)
from expertcache.data import write_json, Dataset
from expertcache.detail import DetailReader
from expertcache.runner import Benchmark
from expertcache.summary import postprocess
from expertcache.engine import CacheEngine


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self): self.temp.cleanup()

    def dataset(self, traces, name="data", valid_layers=None):
        root = self.root / name
        root.mkdir()
        requests = []
        for i, trace in enumerate(traces):
            a = np.asarray(trace, dtype=np.uint8)
            np.save(root / f"{i}.npy", a)
            requests.append(TraceRequest(request_id=f"{name}-{i}", route_file=f"{i}.npy", token_count=len(a)))
        layers = a.shape[1]
        valid_layers = tuple(range(layers)) if valid_layers is None else valid_layers
        spec = DatasetManifest(model_id="test", num_layers=layers, num_experts=8, top_k=a.shape[2],
                               valid_layers=valid_layers, requests=tuple(requests),
                               excluded_layer_reasons={l: "unobserved" for l in range(layers) if l not in valid_layers})
        write_json(root / "benchmark.json", spec.model_dump(mode="json"))
        return root / "benchmark.json"

    def run_bench(self, path, capacity=2, batch_size=1, name="run", policy=None, **kwargs):
        config = RunConfig(dataset=str(path), cache=CacheConfig(capacity_units=capacity),
                           scheduler=SchedulerConfig(batch_size=batch_size), **kwargs)
        output = self.root / name
        result = Benchmark(config, policy=policy).run(output)
        manifest = RunManifest.model_validate_json((output / "manifest.json").read_text())
        return result, DetailReader(output, manifest), output


class CoreTests(Fixture):
    def test_lfu_and_summary_per_decision(self):
        path = self.dataset([[[[0, 1]], [[0, 2]], [[1, 2]]]])
        result, reader, output = self.run_bench(path)
        self.assertEqual([r.hit_count for r in reader.records()], [0, 1, 1])
        hit = result["hit_rate"]
        self.assertEqual(hit["count"], 3)
        self.assertAlmostEqual(hit["mean"], 1 / 3)
        self.assertEqual(hit["percentile_order"], "descending")
        self.assertEqual(hit["p10"], .5)
        self.assertAlmostEqual(hit["p90"], .1)
        self.assertAlmostEqual(hit["p99"], .01)
        self.assertEqual([hit[k] for k in ("p50", "median")], [.5] * 2)
        self.assertEqual(result, postprocess(output))
        self.assertEqual(result["transfer"]["total_units"], 4)
        for event in reader.events(): CacheEvent.model_validate(event)

    def test_batch_snapshot_and_coalescing(self):
        path = self.dataset([[[[0, 1]]], [[[0, 1]]]])
        result, reader, _ = self.run_bench(path, capacity=1, batch_size=2)
        self.assertEqual([r.hit_count for r in reader.records()], [0, 0])
        self.assertEqual(result["transfer"]["total_units"], 2)
        self.assertEqual(result["transfer"]["evaluation_coalesced_miss_occurrences"], 2)

    def test_zero_capacity_and_full_capacity(self):
        path = self.dataset([[[[0, 1]], [[0, 1]]]])
        result, reader, _ = self.run_bench(path, capacity=0)
        self.assertEqual([r.hit_count for r in reader.records()], [0, 0])
        self.assertEqual(result["transfer"]["total_units"], 4)
        result, reader, _ = self.run_bench(path, capacity=8, name="full")
        self.assertEqual([r.hit_count for r in reader.records()], [0, 2])
        self.assertEqual(result["transfer"]["total_units"], 2)

    def test_lifecycle_visibility_and_prefetch(self):
        calls = []
        outer = self
        class Policy(CachePolicy):
            def on_initialize(self, ctx, profile):
                calls.append("initialize")
                return CachePlan()
            def on_run_start(self, ctx):
                calls.append("run_start")
                outer.assertIsNone(ctx.oracle)
                return CachePlan()
            def on_request_start(self, ctx, request):
                calls.append("request_start")
                with outer.assertRaises(PermissionError): ctx.history.routes(request.request_id, 0, 0)
                return CachePlan()
            def on_step_start(self, ctx, step):
                calls.append("step_start")
                return CachePlan()
            def on_layer_start(self, ctx, layer):
                calls.append(f"layer_start{layer}")
                return CachePlan((Action("load", (0, 0), reason="prefetch"),)) if layer == 0 else CachePlan()
            def on_routes_ready(self, ctx, routes, probe):
                calls.append(f"routes{routes.layer_id}")
                req = routes.step.rows[0].request_id
                outer.assertEqual(ctx.history.routes(req, 0, routes.layer_id), routes.expert_ids[0])
                if routes.layer_id == 0:
                    with outer.assertRaises(PermissionError): ctx.history.routes(req, 0, 1)
                return CachePlan()
            def on_layer_end(self, ctx, result):
                calls.append(f"layer_end{result.routes.layer_id}")
                return CachePlan()
            def on_step_end(self, ctx, step):
                calls.append("step_end")
                return CachePlan()
            def on_request_end(self, ctx, request):
                calls.append("request_end")
                return CachePlan()
            def on_run_end(self, ctx):
                calls.append("run_end")
                return CachePlan()
        path = self.dataset([[[[0, 1], [0, 1]]]])
        result, reader, _ = self.run_bench(path, policy=Policy())
        self.assertEqual([r.hit_count for r in reader.records()], [1, 0])
        self.assertEqual(result["transfer"]["phases"]["evaluation"]["prefetch_units"], 1)
        self.assertEqual(calls, ["initialize", "run_start", "request_start", "step_start", "layer_start0", "routes0", "layer_end0",
                                 "layer_start1", "routes1", "layer_end1", "step_end", "request_end", "run_end"])

    def test_current_route_load_is_miss(self):
        class Policy(CachePolicy):
            def on_routes_ready(self, ctx, routes, probe):
                return CachePlan(tuple(Action("load", k) for k in probe.missing))
        path = self.dataset([[[[0, 1]]]])
        result, reader, _ = self.run_bench(path, policy=Policy())
        self.assertEqual(next(reader.records()).hit_count, 0)
        self.assertEqual(result["transfer"]["phases"]["evaluation"]["demand_units"], 2)

    def test_preload_restore_and_warmup(self):
        path = self.dataset([[[[0, 1]]]])
        init = InitConfig(mode="preload", experts=(ExpertRef(layer_id=0, expert_id=0),))
        result, reader, output = self.run_bench(path, initialization=init)
        self.assertEqual(next(reader.records()).hit_count, 1)
        self.assertEqual(result["transfer"]["phases"]["initialization"]["units"], 1)
        init = InitConfig(mode="restore", snapshot_path=str(output / "final_state.json"))
        result, reader, _ = self.run_bench(path, initialization=init, name="restore")
        self.assertEqual(next(reader.records()).hit_count, 2)
        self.assertEqual(result["transfer"]["total_units"], 0)
        warmup = self.dataset([[[[0, 1]]]], name="train")
        result, reader, _ = self.run_bench(path, initialization=InitConfig(mode="warmup", warmup_dataset=str(warmup)), name="warm")
        self.assertEqual(next(reader.records()).hit_count, 2)
        self.assertEqual(result["hit_rate"]["count"], 1)
        self.assertEqual(result["transfer"]["phases"]["warmup"]["units"], 2)

    def test_profile_overlap_rejected(self):
        path = self.dataset([[[[0, 1]]]])
        profile = Profile(model_id="test", source_fingerprint="abc", source_request_ids=("data-0",), provenance="test", entries=())
        profile_path = self.root / "profile.json"
        write_json(profile_path, profile.model_dump(mode="json"))
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.run_bench(path, initialization=InitConfig(mode="profile", profile_path=str(profile_path)))

    def test_variable_sizes_fixed_quota_and_pinning(self):
        path = self.dataset([[[[0, 1], [0, 1]]]])
        spec = Dataset(path).spec
        config = CacheConfig(capacity_units=4, workspace_units=2, allocation="fixed_per_layer", layer_capacity_units={0: 2, 1: 2},
                             expert_sizes=(ExpertSize(layer_id=0, expert_id=0, size_units=2),))
        engine = CacheEngine(config, spec)
        engine.execute(CachePlan((Action("load", (0, 0)), Action("pin", (0, 0)))), "initialization", -1, "test")
        with self.assertRaisesRegex(ValueError, "layer capacity"):
            engine.execute(CachePlan((Action("load", (0, 1)),)), "evaluation", 0, "test")
        with self.assertRaisesRegex(ValueError, "pinned"):
            engine.execute(CachePlan((Action("evict", (0, 0)),)), "evaluation", 0, "test")
        self.assertEqual(engine.view().used_units, 2)

    def test_dynamic_quotas(self):
        path = self.dataset([[[[0, 1], [0, 1]]]])
        config = CacheConfig(capacity_units=4, allocation="dynamic_per_layer", layer_capacity_units={0: 2, 1: 2})
        engine = CacheEngine(config, Dataset(path).spec)
        actions = (Action("quota", (0, -1), units=1), Action("quota", (1, -1), units=3))
        engine.execute(CachePlan(actions), "evaluation", 0, "test")
        self.assertEqual(dict(engine.view().layer_capacity), {0: 1, 1: 3})

    def test_excluded_layers_and_shards(self):
        path = self.dataset([[[[0, 0], [0, 1]], [[0, 0], [0, 1]]]], valid_layers=(1,))
        result, reader, _ = self.run_bench(path, detail_chunk_rows=1)
        self.assertEqual(result["hit_rate"]["count"], 2)
        self.assertEqual([r.layer_id for r in reader.records()], [1, 1])
        self.assertEqual(len(reader.index.shards), 2)

    def test_failed_run_cannot_postprocess(self):
        class BadPolicy(CachePolicy):
            def on_initialize(self, ctx, profile):
                return CachePlan((Action("load", (0, 0)), Action("load", (0, 1))))
        path = self.dataset([[[[0, 1]]]])
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.run_bench(path, capacity=1, policy=BadPolicy())
        manifest = json.loads((self.root / "run" / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        with self.assertRaises(ValueError): postprocess(self.root / "run")

    def test_detail_tampering_detected(self):
        path = self.dataset([[[[0, 1]]]])
        _, _, output = self.run_bench(path)
        shard = next((output / "detail").glob("access-*.npy"))
        with shard.open("ab") as f: f.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "checksum"):
            postprocess(output)


if __name__ == "__main__": unittest.main()
