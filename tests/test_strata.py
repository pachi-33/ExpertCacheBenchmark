from dataclasses import replace
from expertcache.api import Step
from expertcache.contracts import InitConfig, ExpertRef, Profile, ProfileEntry
from expertcache.data import write_json
from expertcache.policies.strata import Strata
from expertcache.runner import Benchmark
from test_benchmark import Fixture


class StrataTests(Fixture):
    def preload(self, *keys):
        return InitConfig(mode="preload", experts=tuple(ExpertRef(layer_id=l, expert_id=e) for l, e in keys))

    def test_interval_layer_local_global_cap_and_decay(self):
        path = self.dataset([[[[1], [1]]] * 5])
        policy = Strata(adapt_every=2, max_swaps=1, decay=.5)
        result, reader, _ = self.run_bench(path, policy=policy, initialization=self.preload((0, 0), (1, 0)))
        records = list(reader.records())
        self.assertEqual([r.hit_count for r in records if r.layer_id == 0], [0, 0, 1, 1, 1])
        self.assertEqual([r.hit_count for r in records if r.layer_id == 1], [0, 0, 0, 0, 1])
        self.assertEqual(policy.adaptations, 2)
        self.assertEqual(policy.swaps, 2)
        self.assertEqual(policy.usage[(0, 1)], 2.5)
        self.assertEqual(result["transfer"]["phases"]["evaluation"]["adaptation_units"], 2)
        self.assertEqual(result["transfer"]["total_units"], 10)

    def test_batch_counts_occurrences_not_unique_experts(self):
        path = self.dataset([[[[1]], [[1]]], [[[1]], [[1]]]])
        policy = Strata(adapt_every=1, decay=1)
        _, reader, _ = self.run_bench(path, capacity=1, batch_size=2, policy=policy,
                                     initialization=self.preload((0, 0)))
        self.assertEqual([r.hit_count for r in reader.records()], [0, 0, 1, 1])
        self.assertEqual(policy.swaps, 1)

    def test_minimum_count(self):
        path = self.dataset([[[[1]]]])
        policy = Strata(adapt_every=1, min_gain=0, decay=.7)
        _, _, _ = self.run_bench(path, capacity=1, policy=policy, initialization=self.preload((0, 0)))
        self.assertEqual(policy.swaps, 0)
        self.assertAlmostEqual(float(policy.usage[(0, 1)]), .7, places=6)

    def test_gain_is_additive_and_boundary_inclusive(self):
        path = self.dataset([[[[1]]]])
        from expertcache.contracts import RunConfig, CacheConfig
        policy = Strata(adapt_every=1)
        bench = Benchmark(RunConfig(dataset=str(path), cache=CacheConfig(capacity_units=1)), policy=policy)
        from expertcache.api import CachePlan, Action
        bench.engine.execute(CachePlan((Action("load", (0, 0)),)), "initialization", -1, "test")
        policy.usage = {(0, 0): 2.0, (0, 1): 3.5}
        plan = policy.on_step_end(bench.context("on_step_end"), Step(0, ()))
        self.assertEqual([(a.kind, a.key) for a in plan.actions], [("evict", (0, 0)), ("load", (0, 1))])

    def test_profile_initialization_does_not_seed_online_counts(self):
        path = self.dataset([[[[1]]]])
        profile = Profile(model_id="test", source_fingerprint="training", source_request_ids=("training-0",),
                          provenance="independent training routes",
                          entries=(ProfileEntry(layer_id=0, expert_id=0, count=1000), ProfileEntry(layer_id=0, expert_id=1, count=10)))
        file = self.root / "profile.json"
        write_json(file, profile.model_dump(mode="json"))
        policy = Strata(adapt_every=0)
        _, reader, _ = self.run_bench(path, capacity=1, policy=policy,
                                     initialization=InitConfig(mode="profile", profile_path=str(file)))
        self.assertEqual(next(reader.records()).hit_count, 0)
        self.assertEqual(dict(policy.usage), {(0, 1): 1})

    def test_layer_without_initial_slots_cannot_gain_slots(self):
        path = self.dataset([[[[0], [1]]] * 4])
        policy = Strata(adapt_every=1, decay=1)
        _, reader, _ = self.run_bench(path, capacity=1, policy=policy, initialization=self.preload((0, 0)))
        self.assertEqual([r.hit_count for r in reader.records() if r.layer_id == 1], [0] * 4)
        self.assertEqual(policy.swaps, 0)

    def test_empty_positive_pool_requires_initialization(self):
        path = self.dataset([[[[0]]]])
        with self.assertRaisesRegex(ValueError, "requires profile"):
            self.run_bench(path, policy=Strata())
