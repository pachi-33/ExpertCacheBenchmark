from collections import Counter
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

    def on_initialize(self, ctx, profile):
        return profile_plan(ctx, profile)

    def on_routes_ready(self, ctx, routes, probe):
        counts = Counter(routes.keys)
        self.clock += 1
        for key, count in counts.items():
            self.frequency[key] = self.frequency.get(key, 0) + count
            self.last_seen[key] = self.clock
        resident = set(ctx.cache.resident)
        per_layer, used, actions = Counter(ctx.cache.layer_used), ctx.cache.used_units, []
        for key in probe.missing:
            size = ctx.catalog[key]
            limit = ctx.cache.layer_capacity.get(key[0], ctx.cache.capacity_units)
            if size > min(ctx.cache.capacity_units, limit): continue  # engine streams it
            # Calculate victims first so an impossible admission doesn't evict needlessly.
            victims, projected, layer_projected = [], used, per_layer[key[0]]
            candidates = sorted(resident - ctx.cache.pinned,
                                key=lambda k: (self.frequency.get(k, 0), self.last_seen.get(k, -1), k))
            while projected + size > ctx.cache.capacity_units or layer_projected + size > limit:
                layer_full = layer_projected + size > limit
                victim = next((k for k in candidates if not layer_full or k[0] == key[0]), None)
                if victim is None: break
                candidates.remove(victim)
                victims.append(victim)
                projected -= ctx.catalog[victim]
                if victim[0] == key[0]: layer_projected -= ctx.catalog[victim]
            else:
                for victim in victims:
                    actions.append(Action("evict", victim, reason="lfu"))
                    resident.remove(victim)
                    used -= ctx.catalog[victim]
                    per_layer[victim[0]] -= ctx.catalog[victim]
                    self.frequency.pop(victim, None)
                    self.last_seen.pop(victim, None)
                actions.append(Action("load", key, reason="demand"))
                resident.add(key)
                used += size
                per_layer[key[0]] += size
        return CachePlan(tuple(actions))

    def on_actions_applied(self, ctx, events):
        # Includes preload/restore and actions from other lifecycle hooks.
        self.frequency = {k: self.frequency.get(k, 0) for k in ctx.cache.resident}
        self.last_seen = {k: self.last_seen.get(k, -1) for k in ctx.cache.resident}

    def state_dict(self):
        return {"clock": self.clock, "entries": [[*k, v, self.last_seen[k]] for k, v in sorted(self.frequency.items())]}

    def load_state_dict(self, state):
        self.clock = state["clock"]
        self.frequency = {(l, e): f for l, e, f, t in state["entries"]}
        self.last_seen = {(l, e): t for l, e, f, t in state["entries"]}

    def metadata(self):
        return {"variant": "resident_lfu_lru_tiebreak", "batch_counts": "all_occurrences", "miss_order": "first_use"}
