"""Offline MIN with batch snapshots, global equal-size slots, and bypass.

The objective is deduplicated demand loads, not occurrence-weighted hits.
Only demanded experts may be admitted; there is no oracle prefetch.
"""
from array import array
from collections import defaultdict
import heapq
from ..api import Action, CachePlan, CachePolicy
from .lfu import profile_plan


class Belady(CachePolicy):
    def __init__(self):
        self._reset()

    def _reset(self):
        self._phase = None
        self._positions = self._cursors = self._next = None
        self._steps = self._layers = None
        self._resident = frozenset()
        self._working, self._heap = set(), []
        self._time = self._horizon = 0

    def _validate(self, ctx):
        if ctx.oracle is None:
            raise ValueError("Belady requires visibility=oracle")
        if ctx.cache.allocation != "global" or ctx.cache.pinned:
            raise ValueError("Belady requires a global pool without pinned experts")
        sizes = set(ctx.catalog.values())
        if len(sizes) != 1:
            raise ValueError("Belady requires equal-sized experts")
        return ctx.cache.capacity_units // next(iter(sizes))

    def on_initialize(self, ctx, profile):
        self._validate(ctx)
        return profile_plan(ctx, profile)

    def _build_index(self, ctx):
        positions = defaultdict(lambda: array("Q"))
        self._steps, self._layers = array("Q"), array("I")
        for time, batch in enumerate(ctx.oracle.batches()):
            self._steps.append(batch.step.step_id)
            self._layers.append(batch.layer_id)
            for key in dict.fromkeys(batch.keys):
                positions[key].append(time)
        self._positions = dict(positions)
        self._cursors = dict.fromkeys(positions, 0)
        self._horizon = len(self._steps)
        self._next = {key: times[0] for key, times in positions.items()}
        self._phase, self._time = ctx.phase, 0
        self._resident = ctx.cache.resident
        self._working = set(self._resident)
        self._rebuild()

    def _rank(self, key):
        return (-self._next.get(key, self._horizon), key)

    def _rebuild(self):
        self._heap = [self._rank(key) for key in self._working]
        heapq.heapify(self._heap)

    def _worst(self):
        while self._heap:
            rank = self._heap[0]
            if rank[1] in self._working and rank == self._rank(rank[1]):
                return rank[1]
            heapq.heappop(self._heap)
        raise RuntimeError("Belady heap lost resident state")

    def on_routes_ready(self, ctx, routes, probe):
        if self._phase != ctx.phase:
            slots = self._validate(ctx)
            self._build_index(ctx)
            self._slots = slots
        if self._time >= self._horizon or (routes.step.step_id, routes.layer_id) != (self._steps[self._time], self._layers[self._time]):
            raise ValueError("Belady oracle and actual schedule differ")
        if self._resident != ctx.cache.resident:
            self._resident = ctx.cache.resident
            self._working = set(self._resident)
            self._rebuild()
        for key in probe.unique_keys:
            cursor = self._cursors[key]
            times = self._positions[key]
            if times[cursor] != self._time:
                raise ValueError("Belady oracle and actual routes differ")
            cursor += 1
            self._cursors[key] = cursor
            self._next[key] = times[cursor] if cursor < len(times) else self._horizon
            if key in self._working:
                heapq.heappush(self._heap, self._rank(key))
        admitted, evicted = set(), set()
        for key in probe.missing:
            if not self._slots:
                continue
            if len(self._working) == self._slots:
                victim = self._worst()
                if self._next[key] >= self._next.get(victim, self._horizon):
                    continue  # Bypass the new item, including ties.
                heapq.heappop(self._heap)
                self._working.remove(victim)
                if victim in admitted:
                    admitted.remove(victim)
                else:
                    evicted.add(victim)
            self._working.add(key)
            admitted.add(key)
            heapq.heappush(self._heap, self._rank(key))
        actions = [Action("evict", key, reason="belady") for key in sorted(evicted)]
        actions.extend(Action("load", key, reason="demand") for key in probe.missing if key in admitted)
        self._time += 1
        if len(self._heap) > max(32, 4 * len(self._working)):
            self._rebuild()
        return CachePlan(tuple(actions))

    def on_actions_applied(self, ctx, events):
        # Keep the immutable Engine view by identity between unchanged layers.
        self._resident = ctx.cache.resident

    def state_dict(self):
        # Future indices belong to the new phase's schedule and are rebuilt on restore.
        return {"version": 1}

    def load_state_dict(self, state):
        if state != {"version": 1}:
            raise ValueError("unsupported Belady snapshot")
        self._reset()

    def metadata(self):
        return dict(variant="batched_belady_min_with_bypass", requires_visibility="oracle",
                    objective="deduplicated_demand_loads", admission="demand_only",
                    allocation="global", expert_sizes="uniform", batch_hits="pre_demand_snapshot",
                    occurrence_weighted_hit_rate_optimal=False,
                    indexed_batch_layers=self._horizon)
