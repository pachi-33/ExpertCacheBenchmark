"""Strata's synchronous adaptive residency policy, replayed on decode steps.

Source commit: 296b59d2f73c8f04a805df29c2fa97655adace78.
This models cache decisions, not Strata's CPU/GPU execution or MTP timing.
"""
from collections import Counter, defaultdict
import numpy as np
from pydantic import Field
from ..api import Action, CachePlan, CachePolicy
from ..contracts import Contract
from .lfu import profile_plan


class StrataParams(Contract):
    adapt_every: int = Field(default=4, ge=0)
    decay: float = Field(default=0.7, ge=0, le=1)
    min_count: float = Field(default=2.0, ge=0)
    min_gain: float = Field(default=1.5, ge=0)
    max_swaps: int = Field(default=96, ge=0)


class Strata(CachePolicy):
    def __init__(self, **params):
        self.params = StrataParams(**params)
        self.usage = {}
        self.steps = self.adaptations = self.swaps = 0

    def on_initialize(self, ctx, profile):
        # Global profile rank determines initial per-layer slot counts. Online
        # usage starts at zero, as in the source's drive.d.usage.assign(..., 0).
        return profile_plan(ctx, profile)

    def on_run_start(self, ctx):
        if not ctx.cache.resident and ctx.cache.capacity_units:
            raise ValueError("Strata requires profile, preload or restore initialization: same-layer swaps cannot create slots in an empty pool")
        sizes = defaultdict(set)
        for (layer, _), units in ctx.catalog.items(): sizes[layer].add(units)
        if any(len(values) != 1 for values in sizes.values()):
            raise ValueError("Strata swaps require uniform expert size within each layer")
        return CachePlan()

    def on_routes_ready(self, ctx, routes, probe):
        for key, count in Counter(routes.keys).items():
            self.usage[key] = np.float32(self.usage.get(key, 0) + count)
        # Demand misses are streamed by the engine, without changing residency.
        return CachePlan()

    def on_step_end(self, ctx, step):
        self.steps += 1
        p = self.params
        if not p.adapt_every or not p.max_swaps or self.steps % p.adapt_every:
            return CachePlan()
        candidates, victims = defaultdict(list), defaultdict(list)
        for key, score in self.usage.items():
            if key not in ctx.cache.resident and score >= p.min_count:
                candidates[key[0]].append((float(score), key))
        for key in ctx.cache.resident - ctx.cache.pinned:
            victims[key[0]].append((float(self.usage.get(key, 0)), key))
        swaps = []
        for layer in ctx.valid_layers:
            incoming = sorted(candidates[layer], key=lambda x: (-x[0], x[1]))
            outgoing = sorted(victims[layer], key=lambda x: (x[0], x[1]))
            for (hot, key_in), (cold, key_out) in zip(incoming, outgoing):
                if hot < cold + p.min_gain: break
                swaps.append((hot - cold, key_in, key_out))
        swaps.sort(key=lambda x: (-x[0], x[1], x[2]))
        chosen = swaps[:p.max_swaps]
        actions = []
        for _, key_in, key_out in chosen:
            actions.extend((Action("evict", key_out, reason="adaptation"),
                            Action("load", key_in, reason="adaptation")))
        self.usage = {key: np.float32(score * np.float32(p.decay)) for key, score in self.usage.items()}
        self.adaptations += 1
        self.swaps += len(chosen)
        return CachePlan(tuple(actions))

    def state_dict(self):
        return {"steps": self.steps, "adaptations": self.adaptations, "swaps": self.swaps,
                "usage": [[*key, float(score)] for key, score in sorted(self.usage.items())]}

    def load_state_dict(self, state):
        self.steps, self.adaptations, self.swaps = state["steps"], state["adaptations"], state["swaps"]
        self.usage = {(l, e): np.float32(score) for l, e, score in state["usage"]}

    def metadata(self):
        return dict(source_commit="296b59d2f73c8f04a805df29c2fa97655adace78",
                    source_url="https://github.com/Niko1221/Strata/blob/296b59d2f73c8f04a805df29c2fa97655adace78/src/program/generate.cpp",
                    variant="synchronous_residency_policy", clock_unit="decode_step",
                    tie_break="layer_id_then_expert_id", score_dtype="float32",
                    params=self.params.model_dump(), adaptations=self.adaptations, swaps=self.swaps,
                    limitations=["No MTP verification-window trace", "All misses use streamed weight loading; no CPU expert execution",
                                 "No asynchronous copy overlap, multi-device ownership, or eviction writeback"])
