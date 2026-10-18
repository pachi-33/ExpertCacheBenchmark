"""Versioned JSON contracts. Dense access data uses the same semantics in NumPy."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Nonnegative = Annotated[int, Field(ge=0)]
Positive = Annotated[int, Field(gt=0)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ExpertRef(Contract):
    layer_id: Nonnegative
    expert_id: Nonnegative

    @property
    def key(self):
        return (self.layer_id, self.expert_id)


class ExpertSize(ExpertRef):
    size_units: Positive


class TraceRequest(Contract):
    request_id: str
    route_file: str
    token_count: Positive
    metadata: dict = Field(default_factory=dict)


class DatasetManifest(Contract):
    schema_version: Literal["1.0"] = "1.0"
    model_id: str
    num_layers: Positive
    num_experts: Positive
    top_k: Positive
    route_dtype: Literal["uint8", "uint16", "uint32"] = "uint8"
    axes: tuple[str, str, str] = ("token", "layer", "topk")
    valid_layers: tuple[Nonnegative, ...]
    excluded_layer_reasons: dict[int, str] = Field(default_factory=dict)
    requests: tuple[TraceRequest, ...]

    @model_validator(mode="after")
    def coherent(self):
        if self.axes != ("token", "layer", "topk"):
            raise ValueError("axes must be token, layer, topk")
        if not self.valid_layers or tuple(sorted(set(self.valid_layers))) != self.valid_layers:
            raise ValueError("valid_layers must be nonempty, unique, ascending")
        if self.valid_layers[-1] >= self.num_layers or self.top_k > self.num_experts:
            raise ValueError("invalid layer or top-k dimensions")
        if len({r.request_id for r in self.requests}) != len(self.requests) or not self.requests:
            raise ValueError("request IDs must be unique and nonempty")
        excluded = set(range(self.num_layers)) - set(self.valid_layers)
        if set(self.excluded_layer_reasons) != excluded:
            raise ValueError("every excluded layer needs a reason")
        return self


class CacheConfig(Contract):
    capacity_units: Nonnegative
    workspace_units: Positive = 1
    default_expert_units: Positive = 1
    expert_sizes: tuple[ExpertSize, ...] = ()
    unit_name: str = "virtual_expert"
    allocation: Literal["global", "fixed_per_layer", "dynamic_per_layer"] = "global"
    layer_capacity_units: dict[int, Nonnegative] = Field(default_factory=dict)


class InitConfig(Contract):
    mode: Literal["cold", "preload", "profile", "warmup", "restore"] = "cold"
    experts: tuple[ExpertRef, ...] = ()
    profile_path: str | None = None
    warmup_dataset: str | None = None
    warmup_max_requests: Positive | None = None
    warmup_max_tokens: Positive | None = None
    snapshot_path: str | None = None

    @model_validator(mode="after")
    def valid_input(self):
        fields = {"preload": "experts", "profile": "profile_path", "warmup": "warmup_dataset", "restore": "snapshot_path"}
        for mode, field in fields.items():
            if self.mode == mode and not getattr(self, field):
                raise ValueError(f"{mode} initialization requires {field}")
            if self.mode != mode and getattr(self, field):
                raise ValueError(f"{field} is only valid for {mode}")
        if self.mode != "warmup" and (self.warmup_max_requests or self.warmup_max_tokens):
            raise ValueError("warmup limits require warmup mode")
        return self


class PolicyConfig(Contract):
    name: str = "lfu"  # Built-in name or trusted Python module:Class.
    params: dict = Field(default_factory=dict)


class SchedulerConfig(Contract):
    batch_size: Positive = 1
    order: Literal["dataset", "shuffled"] = "dataset"
    seed: int = 0
    request_offset: Nonnegative = 0
    max_requests: Positive | None = None
    max_tokens: Positive | None = None


class RunConfig(Contract):
    schema_version: Literal["1.0"] = "1.0"
    dataset: str
    cache: CacheConfig
    initialization: InitConfig = Field(default_factory=InitConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    visibility: Literal["causal", "oracle"] = "causal"
    timing_mode: Literal["instant"] = "instant"
    summary_hooks: tuple[str, ...] = ()  # trusted module:Class, additions to built-ins
    detail_chunk_rows: Positive = 16384


class ProfileEntry(ExpertRef):
    count: Annotated[float, Field(ge=0)]


class Profile(Contract):
    schema_version: Literal["1.0"] = "1.0"
    model_id: str
    source_fingerprint: str
    source_request_ids: tuple[str, ...]
    provenance: str
    entries: tuple[ProfileEntry, ...]


class AccessRecord(Contract):
    probe_tick: Nonnegative
    phase: Literal["warmup", "evaluation"]
    step_id: Nonnegative
    request_id: str
    token_index: Nonnegative
    layer_id: Nonnegative
    expert_ids: tuple[Nonnegative, ...]
    hit_mask: tuple[bool, ...]
    hit_count: Nonnegative

    @model_validator(mode="after")
    def aligned(self):
        if not self.expert_ids or len(self.expert_ids) != len(self.hit_mask) or sum(self.hit_mask) != self.hit_count:
            raise ValueError("top-k annotation mismatch")
        return self


class CacheEvent(Contract):
    event_id: Nonnegative
    phase: Literal["initialization", "warmup", "evaluation"]
    step_id: int
    hook: str
    kind: Literal["transfer_start", "transfer_complete", "evict", "pin", "unpin", "quota", "restore"]
    layer_id: int = -1
    expert_id: int = -1
    units: Nonnegative = 0
    reason: str = ""
    transfer_id: int | None = None
    destination: Literal["cache", "workspace"] | None = None
    context_layer_id: int | None = None
    logical_tick: Nonnegative
    simulated_time: float | None = None


class RunManifest(Contract):
    schema_version: Literal["1.0"] = "1.0"
    framework_version: str
    framework_revision: str | None = None
    source_fingerprint: str
    status: Literal["running", "detail_complete", "complete", "failed"]
    config: RunConfig
    dataset_fingerprint: str
    inputs: dict = Field(default_factory=dict)
    dataset: DatasetManifest
    requests: dict[str, tuple[TraceRequest, ...]]
    access_dtype: list
    policy_metadata: dict = Field(default_factory=dict)
    error: str | None = None


class AccessShard(Contract):
    path: str
    phase: Literal["warmup", "evaluation"]
    rows: Nonnegative
    sha256: str


class DetailIndex(Contract):
    schema_version: Literal["1.0"] = "1.0"
    sealed: bool
    shards: tuple[AccessShard, ...]
    events_path: str
    events_sha256: str
    event_count: Nonnegative


Rate = Annotated[float, Field(ge=0, le=1)]


class TokenPositionPoint(Contract):
    token_index: Nonnegative
    request_count: Positive
    expert_hits: Nonnegative
    expert_accesses: Positive
    mean: Rate
    p10: Rate
    p50: Rate
    p90: Rate
    p99: Rate

    @model_validator(mode="after")
    def coherent(self):
        if self.expert_hits > self.expert_accesses:
            raise ValueError("hits exceed accesses")
        if abs(self.mean - self.expert_hits / self.expert_accesses) > 1e-12:
            raise ValueError("mean disagrees with expert counts")
        if not self.p10 >= self.p50 >= self.p90 >= self.p99:
            raise ValueError("percentiles must be descending")
        return self


class TokenPositionMetadata(Contract):
    schema_version: Literal["1.0"] = "1.0"
    phase: Literal["evaluation"] = "evaluation"
    axis: Literal["token_index"] = "token_index"
    index_base: Literal[0] = 0
    sample_unit: Literal["request_token"] = "request_token"
    scale: Literal["0_to_1"] = "0_to_1"
    percentile_order: Literal["descending"] = "descending"
    percentile_method: Literal["linear"] = "linear"
    mean_weighting: Literal["expert_accesses"] = "expert_accesses"
    group_by: Literal["request.metadata.task_type"] = "request.metadata.task_type"
    valid_layers: tuple[Nonnegative, ...]
    top_k: Positive
    expert_accesses_per_request_token: Positive
    request_count: Nonnegative
    request_token_count: Nonnegative
    annotation_count: Nonnegative


class TokenPositionSummary(Contract):
    metadata: TokenPositionMetadata
    overall: tuple[TokenPositionPoint, ...]
    by_task: dict[str, tuple[TokenPositionPoint, ...]]

    @model_validator(mode="after")
    def coherent(self):
        meta = self.metadata
        denominator = len(meta.valid_layers) * meta.top_k
        if not meta.valid_layers or denominator != meta.expert_accesses_per_request_token:
            raise ValueError("invalid request/token denominator")
        if meta.annotation_count != meta.request_token_count * len(meta.valid_layers):
            raise ValueError("annotation count does not cover all layers")
        for curve in (self.overall, *self.by_task.values()):
            if [p.token_index for p in curve] != list(range(len(curve))):
                raise ValueError("token positions must be contiguous from zero")
            if any(a.request_count < b.request_count for a, b in zip(curve, curve[1:])):
                raise ValueError("request support cannot increase with token position")
            if any(p.expert_accesses != p.request_count * denominator for p in curve):
                raise ValueError("access counts disagree with request support")
        if sum(p.request_count for p in self.overall) != meta.request_token_count:
            raise ValueError("request/token total mismatch")
        if (self.overall[0].request_count if self.overall else 0) != meta.request_count:
            raise ValueError("request total mismatch")
        if any(len(curve) > len(self.overall) for curve in self.by_task.values()):
            raise ValueError("task curve exceeds overall horizon")
        for t, point in enumerate(self.overall):
            groups = [curve[t] for curve in self.by_task.values() if t < len(curve)]
            for field in ("request_count", "expert_hits", "expert_accesses"):
                if sum(getattr(p, field) for p in groups) != getattr(point, field):
                    raise ValueError("task counts do not sum to overall")
        return self
