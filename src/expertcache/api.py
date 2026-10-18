"""Stable plugin API. Contexts contain immutable views; no raw trace is public."""
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Literal

ExpertKey = tuple[int, int]


@dataclass(frozen=True)
class Action:
    kind: Literal["load", "evict", "pin", "unpin", "quota"]
    key: ExpertKey = (-1, -1)
    retain: bool = True
    reason: str = "prefetch"
    units: int = 0  # quota action only


@dataclass(frozen=True)
class CachePlan:
    actions: tuple[Action, ...] = ()


@dataclass(frozen=True)
class CacheView:
    resident: frozenset[ExpertKey]
    pinned: frozenset[ExpertKey]
    capacity_units: int
    used_units: int
    workspace_units: int
    allocation: str
    layer_capacity: Mapping[int, int]
    layer_used: Mapping[int, int]


@dataclass(frozen=True)
class RequestInfo:
    request_id: str
    metadata: Mapping


@dataclass(frozen=True)
class StepRow:
    request_id: str
    token_index: int
    request_index: int


@dataclass(frozen=True)
class Step:
    step_id: int
    rows: tuple[StepRow, ...]


@dataclass(frozen=True)
class RouteBatch:
    step: Step
    layer_id: int
    expert_ids: tuple[tuple[int, ...], ...]

    @property
    def keys(self):
        return tuple((self.layer_id, e) for row in self.expert_ids for e in row)


@dataclass(frozen=True)
class Probe:
    hit_mask: tuple[tuple[bool, ...], ...]
    unique_keys: tuple[ExpertKey, ...]
    missing: tuple[ExpertKey, ...]
    cache_tick: int = 0


@dataclass(frozen=True)
class Resolution:
    routes: RouteBatch
    probe: Probe
    events: tuple[dict, ...]


class RouteHistory(Protocol):
    def routes(self, request_id: str, token_index: int, layer_id: int) -> tuple[int, ...]: ...


class OracleRoutes(RouteHistory, Protocol):
    def batches(self) -> Iterable[RouteBatch]:
        """All batch-layer routes in this phase's actual scheduled order."""
        ...


@dataclass(frozen=True)
class Context:
    phase: str
    hook: str
    step_id: int
    layer_id: int | None
    model_id: str
    valid_layers: tuple[int, ...]
    catalog: Mapping[ExpertKey, int]
    cache: CacheView
    history: RouteHistory
    oracle: OracleRoutes | None = None


class CachePolicy:
    """Return plans; hooks are called in the runner's documented fixed order."""
    def on_initialize(self, ctx, profile): return CachePlan()
    def on_run_start(self, ctx): return CachePlan()
    def on_request_start(self, ctx, request): return CachePlan()
    def on_step_start(self, ctx, step): return CachePlan()
    def on_layer_start(self, ctx, layer_id): return CachePlan()
    def on_routes_ready(self, ctx, routes, probe): return CachePlan()
    def on_layer_end(self, ctx, resolution): return CachePlan()
    def on_step_end(self, ctx, step): return CachePlan()
    def on_request_end(self, ctx, request): return CachePlan()
    def on_run_end(self, ctx): return CachePlan()
    def on_actions_applied(self, ctx, events): pass  # observation only
    def state_dict(self): return {}
    def load_state_dict(self, state):
        if state: raise ValueError("policy does not implement state restoration")
    def metadata(self): return {}


class SummaryHook(Protocol):
    namespace: str
    def process(self, detail, manifest) -> dict: ...


class TransferBackend(Protocol):
    """Future timed backends must also supply scheduler/completion semantics."""
    mode: str
    def transfer(self, key, units, destination): ...


class InstantTransfer:
    mode = "instant"
    def transfer(self, key, units, destination):
        return None
