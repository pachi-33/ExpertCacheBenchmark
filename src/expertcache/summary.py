from collections import Counter
from pathlib import Path
import numpy as np
from .contracts import RunManifest
from .data import write_json
from .detail import DetailReader
from .policies import import_symbol


class HitRateSummary:
    namespace = "hit_rate"

    def process(self, detail, manifest):
        # One sample per request/token/layer decision. Never per-request/layer means.
        rates = [a["hit_count"].astype(np.float64) / manifest.dataset.top_k
                 for _, a in detail.access_chunks("evaluation")]
        values = np.concatenate(rates) if rates else np.empty(0)
        result = dict(sample_unit="request_token_layer", scale="0_to_1", count=int(values.size),
                      percentile_method="linear", p10=None, p50=None, p90=None, p99=None, mean=None, median=None)
        if values.size:
            for name, v in zip(("p10", "p50", "p90", "p99"), np.percentile(values, [10, 50, 90, 99], method="linear")):
                result[name] = float(v)
            result["mean"] = float(values.mean())
            result["median"] = float(np.median(values))
        return result


class TransferSummary:
    namespace = "transfer"

    def process(self, detail, manifest):
        phases = {p: dict(load_count=0, units=0, demand_units=0, prefetch_units=0,
                          initialization_units=0, adaptation_units=0, other_units=0, evictions=0)
                  for p in ("initialization", "warmup", "evaluation")}
        misses, unique_misses = 0, set()
        # Count unique misses for each step/layer; a group may straddle shards.
        group, deduplicated = None, 0
        for _, chunk in detail.access_chunks("evaluation"):
            for row in chunk:
                next_group = (int(row["step_id"]), int(row["layer_id"]))
                if next_group != group:
                    deduplicated += len(unique_misses)
                    unique_misses.clear()
                    group = next_group
                mask = ~row["hit_mask"]
                misses += int(mask.sum())
                unique_misses.update(int(e) for e in row["expert_ids"][mask])
        deduplicated += len(unique_misses)
        for event in detail.events():
            p = phases[event["phase"]]
            if event["kind"] == "transfer_complete":
                p["load_count"] += 1
                p["units"] += event["units"]
                field = event["reason"] + "_units"
                p[field if field in ("demand_units", "prefetch_units", "initialization_units", "adaptation_units") else "other_units"] += event["units"]
            elif event["kind"] == "evict": p["evictions"] += 1
        return dict(unit_name=manifest.config.cache.unit_name, phases=phases,
                    total_units=sum(p["units"] for p in phases.values()),
                    evaluation_miss_occurrences=misses, evaluation_unique_miss_demands=deduplicated,
                    evaluation_coalesced_miss_occurrences=misses - deduplicated)


def postprocess(output, extra_hooks=None):
    output = Path(output)
    manifest = RunManifest.model_validate_json((output / "manifest.json").read_text())
    if manifest.status not in ("detail_complete", "complete"):
        raise ValueError("run has no complete detail to postprocess")
    detail = DetailReader(output, manifest)
    hooks = [HitRateSummary(), TransferSummary()]
    paths = manifest.config.summary_hooks if extra_hooks is None else tuple(extra_hooks)
    hooks.extend(import_symbol(p)() for p in paths)
    summary = {"schema_version": "1.0", "dataset_fingerprint": manifest.dataset_fingerprint,
               "summary_hooks": list(paths)}
    for hook in hooks:
        if hook.namespace in summary: raise ValueError(f"duplicate summary namespace {hook.namespace}")
        summary[hook.namespace] = hook.process(detail, manifest)
    write_json(output / "summary.json", summary)
    return summary
