from collections import Counter, defaultdict
import heapq
from ..api import Action, CachePlan, CachePolicy


def profile_plan(ctx, profile):
    if profile is None: return CachePlan()
    actions, used, per_layer = [], ctx.cache.used_units, Counter(ctx.cache.layer_used)
    for entry in sorted(profile.entries, key=lambda x: (-x.count, x.layer_id, x.expert_id)):
        key = entry.key
        if key not in ctx.catalog: raise ValueError("profile references unknown expert")
        if key in ctx.cache.resident: continue
        size = ctx.catalog[key]
        if used + size > ctx.cache.capacity_units: continue
        if ctx.cache.layer_capacity and per_layer[key[0]] + size > ctx.cache.layer_capacity[key[0]]: continue
        actions.append(Action("load", key, reason="initialization"))
        used += size
        per_layer[key[0]] += size
    return CachePlan(tuple(actions))


class LFU(CachePolicy):
    """Resident LFU, LRU tie-break; counts reset on eviction.

    Every routed occurrence counts, including repeated experts across batch rows.
    Hits are observed together; unique misses are serviced in first-use order.
    """
    def __init__(self):
        self.frequency = {}
        self.last_seen = {}
        self.clock = 0
        self._resident = frozenset()
        self._heap = []
        self._layer_heaps = defaultdict(list)

    def _rank(self, key):
        return (self.frequency.get(key, 0), self.last_seen.get(key, -1), key)

    def _push(self, key):
        rank = self._rank(key)
        heapq.heappush(self._heap, rank)
        heapq.heappush(self._layer_heaps[key[0]], rank)

    def _rebuild(self, resident):
        self._heap = [self._rank(k) for k in resident]
        heapq.heapify(self._heap)
        self._layer_heaps = defaultdict(list)
        for rank in self._heap: self._layer_heaps[rank[2][0]].append(rank)
        for heap in self._layer_heaps.values(): heapq.heapify(heap)

    def _sync(self, ctx):
        if self._resident != ctx.cache.resident:
            self.frequency = {k: self.frequency.get(k, 0) for k in ctx.cache.resident}
            self.last_seen = {k: self.last_seen.get(k, -1) for k in ctx.cache.resident}
            self._resident = ctx.cache.resident
            self._rebuild(self._resident)

    def _victim(self, heap, resident, pinned):
        deferred, victim = [], None
        while heap:
            rank = heapq.heappop(heap)
            key = rank[2]
            if key not in resident or rank != self._rank(key): continue
            if key in pinned:
                deferred.append(rank)
                continue
            victim = key
            break
        for rank in deferred: heapq.heappush(heap, rank)
        return victim

    def on_initialize(self, ctx, profile):
        return profile_plan(ctx, profile)

    def on_routes_ready(self, ctx, routes, probe):
        self._sync(ctx)
        counts = Counter(routes.keys)
        self.clock += 1
        for key, count in counts.items():
            self.frequency[key] = self.frequency.get(key, 0) + count
            self.last_seen[key] = self.clock
            if key in ctx.cache.resident: self._push(key)
        resident = set(ctx.cache.resident)
        per_layer, used, actions = Counter(ctx.cache.layer_used), ctx.cache.used_units, []
        for key in probe.missing:
            size = ctx.catalog[key]
            limit = ctx.cache.layer_capacity.get(key[0], ctx.cache.capacity_units)
            if size > min(ctx.cache.capacity_units, limit): continue
            victims = []
            while used + size > ctx.cache.capacity_units or per_layer[key[0]] + size > limit:
                heap = self._layer_heaps[key[0]] if per_layer[key[0]] + size > limit else self._heap
                victim = self._victim(heap, resident, ctx.cache.pinned)
                if victim is None:
                    # A pinned pool may make the whole admission impossible.
                    for old in victims:
                        resident.add(old)
                        used += ctx.catalog[old]
                        per_layer[old[0]] += ctx.catalog[old]
                        self._push(old)
                    break
                victims.append(victim)
                resident.remove(victim)
                used -= ctx.catalog[victim]
                per_layer[victim[0]] -= ctx.catalog[victim]
            else:
                for victim in victims:
                    actions.append(Action("evict", victim, reason="lfu"))
                    self.frequency.pop(victim, None)
                    self.last_seen.pop(victim, None)
                actions.append(Action("load", key, reason="demand"))
                resident.add(key)
                used += size
                per_layer[key[0]] += size
                self._push(key)
        # Bypassed observations do not survive in resident LFU state.
        for key in counts.keys() - resident:
            self.frequency.pop(key, None)
            self.last_seen.pop(key, None)
        self._resident = frozenset(resident)
        # Bound lazy invalidation memory; heap ordering is only an optimization.
        if len(self._heap) + sum(map(len, self._layer_heaps.values())) > max(32, 8 * len(resident)):
            self._rebuild(resident)
        return CachePlan(tuple(actions))

    def on_actions_applied(self, ctx, events):
        self._sync(ctx)

    def state_dict(self):
        return {"clock": self.clock, "entries": [[*k, v, self.last_seen[k]] for k, v in sorted(self.frequency.items())]}

    def load_state_dict(self, state):
        self._resident = frozenset()
        self._heap = []
        self._layer_heaps = defaultdict(list)
        self.clock = state["clock"]
        self.frequency = {(l, e): f for l, e, f, t in state["entries"]}
        self.last_seen = {(l, e): t for l, e, f, t in state["entries"]}

    def metadata(self):
        return {"variant": "resident_lfu_lru_tiebreak", "batch_counts": "all_occurrences", "miss_order": "first_use"}
