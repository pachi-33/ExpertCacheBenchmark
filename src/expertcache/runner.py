import hashlib
import json
import subprocess
from pathlib import Path
from types import MappingProxyType
from . import __version__
from .api import Action, CachePlan, Context, RequestInfo, Resolution, RouteBatch
from .contracts import RunConfig, RunManifest, Profile
from .data import Dataset, digest_file, write_json
from .detail import DetailWriter, access_dtype
from .engine import CacheEngine
from .policies import make_policy, implementation_identity
from .scheduler import DecodeScheduler, select_requests
from .summary import postprocess


def freeze(value):
    if isinstance(value, dict): return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list): return tuple(freeze(v) for v in value)
    return value


class VisibleHistory:
    def __init__(self, dataset, frontier, allowed, oracle=False):
        self.__dataset, self.__frontier, self.__allowed = dataset, dict(frontier), frozenset(allowed)
        self.__oracle = oracle

    def routes(self, request_id, token_index, layer_id):
        if request_id not in self.__allowed or token_index < 0 or layer_id not in self.__dataset.spec.valid_layers:
            raise ValueError("unknown request/token/layer")
        if not self.__oracle and (request_id not in self.__frontier or (token_index, layer_id) > self.__frontier[request_id]):
            raise PermissionError("routing information is not visible yet")
        return self.__dataset.routes(request_id, token_index, layer_id)


class OracleHistory(VisibleHistory):
    def __init__(self, dataset, requests, steps):
        super().__init__(dataset, {}, (r.request_id for r in requests), True)
        self.__dataset, self.__steps = dataset, steps

    def batches(self):
        for scheduled in self.__steps():
            step = scheduled.step
            for layer in self.__dataset.spec.valid_layers:
                yield RouteBatch(step, layer, tuple(self.__dataset.routes(row.request_id, row.token_index, layer)
                                                   for row in step.rows))


class PlannedSteps:
    """Memoize a phase plan only if an oracle or replay actually needs it."""
    def __init__(self, scheduler, requests, config, start_step):
        self.scheduler, self.requests, self.config, self.start_step = scheduler, requests, config, start_step
        self._steps = None

    def get(self):
        if self._steps is None:
            self._steps = tuple(self.scheduler.steps(self.requests, self.config, self.start_step))
        return self._steps


def load_config(path):
    path = Path(path).resolve()
    config = RunConfig.model_validate_json(path.read_text())
    def absolute(p): return str((path.parent / p).resolve()) if p else None
    init = config.initialization.model_copy(update={k: absolute(getattr(config.initialization, k))
                                                   for k in ("profile_path", "warmup_dataset", "snapshot_path")})
    return config.model_copy(update={"dataset": absolute(config.dataset), "initialization": init})


def code_identity():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError): revision = None
    return revision, h.hexdigest()


class Benchmark:
    def __init__(self, config, policy=None, scheduler=None, backend=None):
        self.config = config
        self.dataset = Dataset(config.dataset)
        self.requests = select_requests(self.dataset, config.scheduler)
        if not self.requests: raise ValueError("request selection is empty")
        self.engine = CacheEngine(config.cache, self.dataset.spec, backend)
        self.policy = policy or make_policy(config.policy)
        self.scheduler = scheduler or DecodeScheduler()
        self.phase, self.step_id, self.layer_id, self.frontier = "initialization", -1, None, {}
        self.active_dataset, self.active_requests = self.dataset, self.requests
        self.active_scheduler_config = config.scheduler
        self.phase_start_step = 0
        self._planned_steps = PlannedSteps(self.scheduler, self.requests, config.scheduler, 0)
        self.writer = None
        self._ran = False

    def context(self, hook):
        ids = tuple(r.request_id for r in self.active_requests)
        history = VisibleHistory(self.active_dataset, self.frontier, ids)
        oracle = OracleHistory(self.active_dataset, self.active_requests, self._planned_steps.get) if self.config.visibility == "oracle" else None
        spec = self.dataset.spec
        return Context(self.phase, hook, self.step_id, self.layer_id, spec.model_id, spec.valid_layers,
                       self.engine.catalog, self.engine.view(), history, oracle)

    def _oracle_steps(self):
        return self._planned_steps.get()

    def _enter_phase(self, dataset, requests, scheduler_config, phase):
        start = self.step_id + 1
        if (dataset is not self.active_dataset or requests != self.active_requests or
                scheduler_config != self.active_scheduler_config or start != self.phase_start_step):
            self._planned_steps = PlannedSteps(self.scheduler, requests, scheduler_config, start)
        self.active_dataset, self.active_requests, self.frontier = dataset, requests, {}
        self.active_scheduler_config = scheduler_config
        self.phase, self.layer_id = phase, None
        self.phase_start_step = start

    def invoke(self, hook, *args, probe=None):
        plan = getattr(self.policy, hook)(self.context(hook), *args)
        events = self.engine.execute(plan, self.phase, self.step_id, hook, probe, self.layer_id)
        self.writer.append_events(events)
        if events: self.policy.on_actions_applied(self.context(hook), tuple(freeze(e) for e in events))
        return events

    def _direct_plan(self, plan, hook):
        events = self.engine.execute(plan, self.phase, self.step_id, hook, context_layer_id=self.layer_id)
        self.writer.append_events(events)
        if events: self.policy.on_actions_applied(self.context(hook), tuple(freeze(e) for e in events))

    def _replay(self, dataset, requests, scheduler_config, phase, progress=None):
        lookup = {r.request_id: r for r in requests}
        steps = self._oracle_steps() if self.config.visibility == "oracle" else self.scheduler.steps(requests, scheduler_config, self.step_id + 1)
        for scheduled in steps:
            step = scheduled.step
            self.step_id = step.step_id
            for r in scheduled.started:
                self.invoke("on_request_start", RequestInfo(r.request_id, freeze(r.metadata)))
            self.invoke("on_step_start", step)
            for layer in dataset.spec.valid_layers:
                self.layer_id = layer
                self.invoke("on_layer_start", layer)
                routes = RouteBatch(step, layer, tuple(dataset.routes(row.request_id, row.token_index, layer) for row in step.rows))
                for row in step.rows: self.frontier[row.request_id] = (row.token_index, layer)
                probe = self.engine.probe(routes)
                events = self.invoke("on_routes_ready", routes, probe, probe=probe)
                resolution = Resolution(routes, probe, tuple(freeze(e) for e in events))
                self.invoke("on_layer_end", resolution)
                self.writer.append_access(phase, routes, probe)
            self.layer_id = None
            self.invoke("on_step_end", step)
            for r in scheduled.finished:
                self.invoke("on_request_end", RequestInfo(r.request_id, freeze(lookup[r.request_id].metadata)))
            if progress and (step.step_id + 1) % 128 == 0: progress(phase, step.step_id + 1)

    def _snapshot(self):
        spec = self.dataset.spec
        return dict(schema_version="1.0", model={k: getattr(spec, k) for k in ("model_id", "num_layers", "num_experts", "top_k", "valid_layers")},
                    cache_config=self.config.cache.model_dump(mode="json"), policy_config=self.config.policy.model_dump(mode="json"),
                    policy_implementation=implementation_identity(self.policy)["class"],
                    cache=self.engine.state_dict(), policy=self.policy.state_dict())

    def run(self, output, progress=None):
        if self._ran:
            raise RuntimeError("Benchmark instances are single-use; create a new instance for each run")
        self._ran = True
        output = Path(output)
        if output.exists(): raise FileExistsError(f"run output already exists: {output}")
        init = self.config.initialization
        inputs, profile, warmup, warmup_requests = {"policy_implementation": implementation_identity(self.policy)}, None, None, ()
        if init.profile_path:
            profile = Profile.model_validate_json(Path(init.profile_path).read_text())
            if profile.model_id != self.dataset.spec.model_id: raise ValueError("profile model mismatch")
            if set(profile.source_request_ids) & {r.request_id for r in self.requests}:
                raise ValueError("profile and evaluation request IDs overlap")
            if len({e.key for e in profile.entries}) != len(profile.entries): raise ValueError("duplicate profile experts")
            inputs["profile_sha256"] = digest_file(init.profile_path)
            inputs["profile_provenance"] = profile.model_dump(mode="json", exclude={"entries"})
        if init.warmup_dataset:
            warmup = Dataset(init.warmup_dataset)
            for field in ("model_id", "num_layers", "num_experts", "top_k", "valid_layers"):
                if getattr(warmup.spec, field) != getattr(self.dataset.spec, field): raise ValueError("warmup model mismatch")
            warmup_requests = warmup.spec.requests[:init.warmup_max_requests]
            if set(r.request_id for r in warmup_requests) & {r.request_id for r in self.requests}:
                raise ValueError("warmup and evaluation request IDs overlap")
            inputs["warmup_fingerprint"] = warmup.fingerprint
        revision, source_hash = code_identity()
        manifest = RunManifest(framework_version=__version__, framework_revision=revision, source_fingerprint=source_hash,
                               status="running", config=self.config, dataset_fingerprint=self.dataset.fingerprint,
                               inputs=inputs, dataset=self.dataset.spec,
                               requests={"evaluation": self.requests, "warmup": warmup_requests},
                               access_dtype=access_dtype(self.dataset.spec).descr, policy_metadata=self.policy.metadata())
        output.mkdir(parents=True)
        write_json(output / "manifest.json", manifest.model_dump(mode="json"))
        self.writer = DetailWriter(output, self.dataset.spec, self.config.detail_chunk_rows)
        closed = False
        try:
            if init.mode == "restore":
                snapshot = json.loads(Path(init.snapshot_path).read_text())
                expected = self._snapshot()
                for field in ("schema_version", "model", "cache_config", "policy_config", "policy_implementation"):
                    if json.dumps(snapshot[field], sort_keys=True) != json.dumps(expected[field], sort_keys=True):
                        raise ValueError(f"snapshot {field} mismatch")
                self.writer.append_events(self.engine.restore(snapshot["cache"]))
                self.policy.load_state_dict(snapshot["policy"])
                inputs["snapshot_sha256"] = digest_file(init.snapshot_path)
            else:
                self.invoke("on_initialize", profile)
                if init.mode == "preload":
                    if len({e.key for e in init.experts}) != len(init.experts): raise ValueError("duplicate preload experts")
                    self._direct_plan(CachePlan(tuple(Action("load", e.key, reason="initialization") for e in init.experts)), "preload")
            if warmup:
                warmup_config = self.config.scheduler.model_copy(update={"max_tokens": init.warmup_max_tokens})
                self._enter_phase(warmup, warmup_requests, warmup_config, "warmup")
                self._replay(warmup, warmup_requests, warmup_config, "warmup", progress)
            write_json(output / "initial_state.json", self._snapshot())
            self._enter_phase(self.dataset, self.requests, self.config.scheduler, "evaluation")
            self.invoke("on_run_start")
            self._replay(self.dataset, self.requests, self.config.scheduler, "evaluation", progress)
            self.invoke("on_run_end")
            write_json(output / "final_state.json", self._snapshot())
            self.writer.close()
            closed = True
            manifest = manifest.model_copy(update={"status": "detail_complete", "inputs": inputs, "policy_metadata": self.policy.metadata()})
            write_json(output / "manifest.json", manifest.model_dump(mode="json"))
        except Exception as exc:
            if not closed: self.writer.close(sealed=False)
            write_json(output / "manifest.json", manifest.model_copy(update={"status": "failed", "error": str(exc)}).model_dump(mode="json"))
            raise
        # Postprocess failures preserve sealed detail, so fixing a hook needs no replay.
        result = postprocess(output)
        write_json(output / "manifest.json", manifest.model_copy(update={"status": "complete"}).model_dump(mode="json"))
        return result
