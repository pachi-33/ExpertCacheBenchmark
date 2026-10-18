from collections import Counter
from types import MappingProxyType
from .api import CacheView, Probe, CachePlan, Action, InstantTransfer


class CacheEngine:
    """Authoritative resource accounting; all demand hits use one batch snapshot.

    Within a layer, snapshot hits are consumed first. Missing experts are then
    serviced serially. This permits working sets larger than cache + workspace
    without turning row iteration order into artificial cache hits.
    """
    def __init__(self, config, spec, backend=None):
        self.config = config
        catalog = {(l, e): config.default_expert_units for l in spec.valid_layers for e in range(spec.num_experts)}
        seen = set()
        for entry in config.expert_sizes:
            if entry.key not in catalog or entry.key in seen:
                raise ValueError("invalid or duplicated expert size override")
            seen.add(entry.key)
            catalog[entry.key] = entry.size_units
        self.catalog = MappingProxyType(catalog)
        if config.workspace_units < max(catalog.values()):
            raise ValueError("workspace must fit the largest expert for serial bypass")
        quotas = dict(config.layer_capacity_units)
        if config.allocation == "global":
            if quotas: raise ValueError("global allocation cannot have layer quotas")
        elif set(quotas) != set(spec.valid_layers) or sum(quotas.values()) > config.capacity_units:
            raise ValueError("layer quotas must cover all valid layers and fit total capacity")
        self.quotas = quotas
        self.resident = set()
        self.pinned = set()
        self.used = 0
        self.layer_used = Counter()
        self.backend = backend or InstantTransfer()
        if self.backend.mode != "instant":
            raise ValueError("v1 only supports instant transfers")
        self.event_id = self.transfer_id = self.tick = 0
        self._view = None

    def view(self):
        if self._view is None:
            self._view = CacheView(frozenset(self.resident), frozenset(self.pinned), self.config.capacity_units,
                                   self.used, self.config.workspace_units, self.config.allocation,
                                   MappingProxyType(dict(self.quotas)), MappingProxyType(dict(self.layer_used)))
        return self._view

    def probe(self, routes):
        keys = tuple(dict.fromkeys(routes.keys))
        return Probe(tuple(tuple((routes.layer_id, e) in self.resident for e in row) for row in routes.expert_ids),
                     keys, tuple(k for k in keys if k not in self.resident))

    def _event(self, kind, phase, step_id, hook, key=(-1, -1), units=0, reason="", transfer_id=None, destination=None):
        self.event_id += 1
        self.tick += 1
        return dict(event_id=self.event_id, phase=phase, step_id=step_id, hook=hook, kind=kind,
                    layer_id=key[0], expert_id=key[1], units=units, reason=reason,
                    transfer_id=transfer_id, destination=destination, logical_tick=self.tick, simulated_time=None)

    def execute(self, plan, phase, step_id, hook, probe=None):
        if not isinstance(plan, CachePlan):
            raise TypeError("policy hooks must return CachePlan")
        events = []
        missing = set(probe.missing) if probe else set()
        served = set(probe.unique_keys) - missing if probe else set()
        actions = list(plan.actions)
        # Any unmet demand falls back to a single streamed load, never free service.
        requested_loads = {a.key for a in actions if a.kind == "load"}
        if probe:
            actions.extend(Action("load", k, retain=False, reason="demand") for k in probe.missing if k not in requested_loads)
        for action in actions:
            key = action.key
            if action.kind == "quota":
                if self.config.allocation != "dynamic_per_layer":
                    raise ValueError("quota changes require dynamic_per_layer allocation")
                layer, units = key[0], action.units
                if layer not in self.quotas or not isinstance(units, int) or units < self.layer_used[layer]:
                    raise ValueError("quota smaller than residency or invalid layer")
                if sum(self.quotas.values()) - self.quotas[layer] + units > self.config.capacity_units:
                    raise ValueError("quota change exceeds global capacity; release quota first")
                self.quotas[layer] = units
                events.append(self._event("quota", phase, step_id, hook, (layer, -1), units))
            else:
                if key not in self.catalog: raise ValueError(f"unknown expert {key}")
                units = self.catalog[key]
                if action.kind == "evict":
                    if key not in self.resident or key in self.pinned:
                        raise ValueError(f"cannot evict nonresident/pinned expert {key}")
                    self.resident.remove(key)
                    self.used -= units
                    self.layer_used[key[0]] -= units
                    events.append(self._event("evict", phase, step_id, hook, key, units, action.reason))
                elif action.kind == "load":
                    if key in self.resident:
                        served.add(key)
                        continue
                    if action.retain:
                        if self.used + units > self.config.capacity_units:
                            raise ValueError("cache capacity exceeded")
                        if self.quotas and self.layer_used[key[0]] + units > self.quotas[key[0]]:
                            raise ValueError("layer capacity exceeded")
                    elif units > self.config.workspace_units:
                        raise ValueError("workspace capacity exceeded")
                    reason = "demand" if key in missing and key not in served else action.reason
                    destination = "cache" if action.retain else "workspace"
                    self.transfer_id += 1
                    events.append(self._event("transfer_start", phase, step_id, hook, key, units, reason, self.transfer_id, destination))
                    self.backend.transfer(key, units, destination)
                    events.append(self._event("transfer_complete", phase, step_id, hook, key, units, reason, self.transfer_id, destination))
                    if action.retain:
                        self.resident.add(key)
                        self.used += units
                        self.layer_used[key[0]] += units
                    served.add(key)
                elif action.kind in ("pin", "unpin"):
                    if key not in self.resident: raise ValueError("only resident experts can be pinned/unpinned")
                    if action.kind == "pin": self.pinned.add(key)
                    else: self.pinned.discard(key)
                    events.append(self._event(action.kind, phase, step_id, hook, key, units))
                else:
                    raise ValueError(f"unknown action {action.kind}")
            self._view = None
        if missing - served: raise ValueError("plan left unresolved demand")
        return tuple(events)

    def state_dict(self):
        return {"resident": sorted(self.resident), "pinned": sorted(self.pinned), "quotas": self.quotas}

    def restore(self, state):
        if self.resident: raise ValueError("restore requires empty cache")
        if self.config.allocation == "dynamic_per_layer":
            quotas = {int(k): v for k, v in state["quotas"].items()}
            if set(quotas) != set(self.quotas) or any(v < 0 for v in quotas.values()) or sum(quotas.values()) > self.config.capacity_units:
                raise ValueError("invalid restored quotas")
            self.quotas = quotas
        elif {int(k): v for k, v in state["quotas"].items()} != self.quotas:
            raise ValueError("snapshot quota mismatch")
        resident = [tuple(k) for k in state["resident"]]
        pinned = set(tuple(k) for k in state["pinned"])
        if len(set(resident)) != len(resident) or not pinned <= set(resident):
            raise ValueError("invalid restored residency")
        for key in resident:
            if key not in self.catalog: raise ValueError("unknown snapshot expert")
            self.used += self.catalog[key]
            self.layer_used[key[0]] += self.catalog[key]
        if self.used > self.config.capacity_units or any(self.layer_used[l] > q for l, q in self.quotas.items()):
            raise ValueError("snapshot exceeds capacity")
        self.resident = set(resident)
        self.pinned = pinned
        self._view = None
        return (self._event("restore", "initialization", -1, "restore", units=self.used, reason="preexisting_residency_no_transfer"),)
