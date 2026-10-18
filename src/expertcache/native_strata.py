"""Optional compiled Strata BS=1 execution with the standard auditable detail format.

The general runner remains the reference. This backend deliberately rejects other
policies, warmup, non-unit experts, pinning, custom callbacks and non-global pools.
"""
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np

from .contracts import AccessShard, RunManifest
from .data import digest_file, write_json
from .detail import DetailReader
from .policies import import_symbol, implementation_identity
from .policies.strata import Strata
from .runner import Benchmark
from .summary import HitRateSummary


def compile_native(build_dir):
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent / "native" / "strata_bs1.cpp"
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("Native Strata requires clang++ or g++ and zlib headers")
    executable = build_dir / ("strata-bs1-" + digest_file(source)[:16])
    flags = ["-std=c++17", "-O3", "-ffp-contract=off"]
    if not executable.exists():
        subprocess.run([compiler, *flags, str(source), "-lz", "-o", str(executable)], check=True)
    provenance = dict(name="compiled_strata_bs1", cpp_sha256=digest_file(source),
                      executable_sha256=digest_file(executable), compiler=compiler,
                      compiler_version=subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
                      compiler_flags=flags, reference_policy=implementation_identity(Strata()))
    return executable, provenance


def _native_count_threshold(value):
    """Preserve the installed NumPy's float32/Python-scalar comparison boundary."""
    with np.errstate(over="ignore"):
        boundary = np.float32(value)
    if not np.isfinite(boundary):
        return value
    if not bool(boundary >= value):
        boundary = np.nextafter(boundary, np.float32(np.inf))
    return float(boundary) if np.isfinite(boundary) else value


class NativeTransferSummary:
    namespace = "transfer"

    def __init__(self, executable, output):
        self.executable, self.output = executable, Path(output)

    def process(self, detail, manifest):
        events = detail._path(detail.index.events_path)
        detail._verify(events, detail.index.events_sha256)
        audit = json.loads(subprocess.check_output([str(self.executable), "--audit", str(events)], text=True))
        if audit["event_count"] != detail.index.event_count:
            raise ValueError("Native event audit count mismatch")
        misses = sum(int((manifest.dataset.top_k - chunk["hit_count"].astype(np.int64)).sum())
                     for _, chunk in detail.access_chunks("evaluation"))
        if misses != audit["demand_units"]:
            raise ValueError("BS1 demand transfers disagree with detail misses")
        native = json.loads((self.output / "native-result.json").read_text())
        if native["swaps"] != audit["adaptation_units"] or native["demand_loads"] != misses:
            raise ValueError("Native replay totals disagree with the independent event scan")
        phases = {}
        for i, phase in enumerate(("initialization", "warmup", "evaluation")):
            totals = audit["phases"][i]
            phases[phase] = dict(**totals, demand_units=misses if phase == "evaluation" else 0,
                                 prefetch_units=0,
                                 initialization_units=audit["initialization_units"] if phase == "initialization" else 0,
                                 adaptation_units=audit["adaptation_units"] if phase == "evaluation" else 0,
                                 other_units=0)
            if totals["units"] != sum(phases[phase][key] for key in ("demand_units", "initialization_units", "adaptation_units")):
                raise ValueError("Unexpected transfer phase/reason combination")
        audit.update(events_sha256=detail.index.events_sha256, detail_misses=misses, status="verified")
        write_json(self.output / "native-event-audit.json", audit)
        return dict(unit_name=manifest.config.cache.unit_name, phases=phases,
                    total_units=sum(p["units"] for p in phases.values()),
                    evaluation_miss_occurrences=misses, evaluation_unique_miss_demands=misses,
                    evaluation_coalesced_miss_occurrences=0)


class NativeStrataBenchmark(Benchmark):
    def __init__(self, config, build_dir):
        if (config.policy.name != "strata" or config.scheduler.batch_size != 1 or
                config.visibility != "causal" or config.initialization.mode != "profile" or
                config.cache.allocation != "global" or config.cache.default_expert_units != 1 or
                config.cache.expert_sizes or config.timing_mode != "instant"):
            raise ValueError("Native backend requires Strata, BS1, causal/profile, unit experts, global pool, instant timing")
        super().__init__(config)
        if type(self.policy) is not Strata or self.dataset.spec.route_dtype != "uint8":
            raise ValueError("Native backend requires the built-in Strata and uint8 routes")
        self.executable, self.native_provenance = compile_native(build_dir)

    def _replay(self, dataset, requests, scheduler_config, phase, progress=None):
        if phase != "evaluation" or self.step_id != -1 or self.engine.pinned:
            raise ValueError("Native backend does not support warmup, restart, or pinned experts")
        root = self.writer.root.parent
        source = root / "native-ordered-routes.bin"
        spec, params = dataset.spec, self.policy.params
        layers = {layer: i for i, layer in enumerate(spec.valid_layers)}
        lengths = [min(r.token_count, scheduler_config.max_tokens or r.token_count) for r in requests]
        with source.open("xb") as stream:
            for request, length in zip(requests, lengths):
                routes = dataset.arrays[request.request_id][:length, spec.valid_layers, :]
                np.ascontiguousarray(routes).tofile(stream)
        self.native_provenance["ordered_routes_sha256"] = digest_file(source)
        keys = sorted(layers[layer] * spec.num_experts + expert for layer, expert in self.engine.resident)
        settings = root / "native-settings.txt"
        settings.write_text("\n".join([
            " ".join(map(str, (len(requests), len(layers), spec.num_experts, spec.top_k,
                               params.adapt_every, params.max_swaps, _native_count_threshold(params.min_count), params.min_gain,
                               float(np.float32(params.decay)), self.config.detail_chunk_rows))),
            " ".join(map(str, spec.valid_layers)), " ".join(map(str, lengths)),
            str(len(keys)), " ".join(map(str, keys)),
            f"{self.engine.event_id} {self.engine.tick} {self.engine.transfer_id}", ""]))
        self.writer.events.close()
        with subprocess.Popen([str(self.executable), str(settings), str(source), str(self.writer.root)],
                              stdout=subprocess.PIPE, text=True) as process:
            for line in process.stdout:
                print(line.rstrip(), flush=True)
                match = re.search(r"steps=(\d+)", line)
                if progress and match:
                    progress(phase, int(match.group(1)))
            if process.wait():
                raise RuntimeError("Native replay failed; partial detail is not sealed")
        result = json.loads((root / "native-result.json").read_text())
        expected_steps = sum(lengths)
        if result["steps"] != expected_steps or sum(s["rows"] for s in result["shards"]) != expected_steps * len(layers):
            raise ValueError("Native replay omitted annotations")
        self.step_id, self.layer_id = expected_steps - 1, None
        self.frontier = {r.request_id: (n - 1, spec.valid_layers[-1]) for r, n in zip(requests, lengths)}
        self.engine.resident = {tuple(key) for key in result["resident"]}
        if len(self.engine.resident) != len(keys):
            raise ValueError("Native replay changed the number of resident slots")
        self.engine.layer_used = Counter(layer for layer, _ in self.engine.resident)
        self.engine.event_id, self.engine.tick, self.engine.transfer_id = result["event_id"], result["tick"], result["transfer_id"]
        self.engine.context_layer_id, self.engine._view = None, None
        self.policy.load_state_dict({k: result[k] for k in ("steps", "adaptations", "swaps", "usage")})
        for shard in result["shards"]:
            self.writer.shards.append(AccessShard(**shard, phase=phase, sha256=digest_file(self.writer.root / shard["path"])))
        self.writer.event_count = self.engine.event_id
        self.writer.events = gzip.open(self.writer.root / "cache_events.jsonl.gz", "at", encoding="utf-8", compresslevel=1)
        source.unlink()  # reproducible derivative, not the dataset or recorded detail

    def _postprocess(self, output):
        print("POSTPROCESS: histogram summaries and independent compressed-event audit", flush=True)
        output = Path(output)
        manifest = RunManifest.model_validate_json((output / "manifest.json").read_text())
        if manifest.status != "detail_complete":
            raise ValueError("Native summary requires sealed detail")
        detail = DetailReader(output, manifest)
        hooks = [HitRateSummary(), NativeTransferSummary(self.executable, output)]
        hooks.extend(import_symbol(path)() for path in manifest.config.summary_hooks)
        summary = dict(schema_version="1.0", dataset_fingerprint=manifest.dataset_fingerprint,
                       summary_hooks=list(manifest.config.summary_hooks),
                       processor_implementations=[implementation_identity(hook) for hook in hooks])
        for hook in hooks:
            if hook.namespace in summary:
                raise ValueError("duplicate summary namespace")
            summary[hook.namespace] = hook.process(detail, manifest)
        write_json(output / "summary.json", summary)
        return summary

    def run(self, output, progress=None):
        try:
            return super().run(output, progress, postprocessor=self._postprocess)
        finally:
            path = Path(output) / "manifest.json"
            if path.exists():
                manifest = RunManifest.model_validate_json(path.read_text())
                inputs = dict(manifest.inputs, execution_backend=self.native_provenance)
                write_json(path, manifest.model_copy(update={"inputs": inputs}).model_dump(mode="json"))
