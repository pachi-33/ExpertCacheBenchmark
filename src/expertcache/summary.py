from collections import Counter
from pathlib import Path
import numpy as np
from .contracts import RunManifest
from .data import digest_file, write_json
from .detail import DetailReader
from .policies import import_symbol, implementation_identity


class HitRateSummary:
    namespace = "hit_rate"

    def process(self, detail, manifest):
        histogram = np.zeros(manifest.dataset.top_k + 1, dtype=np.int64)
        for _, chunk in detail.access_chunks("evaluation"):
            histogram += np.bincount(chunk["hit_count"].astype(np.int64), minlength=len(histogram))
        return histogram_hit_summary(histogram, manifest.dataset.top_k)


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


def postprocess(output, extra_hooks=None, *, only_hooks=None, summary_output=None):
    """Rebuild summaries, optionally writing a separate, read-only-source analysis.

    The historical default still runs both built-ins and overwrites summary.json.
    Selective processing requires a new output outside the sealed run directory.
    """
    from time import monotonic
    started = monotonic()
    output = Path(output).resolve()
    if only_hooks is not None and extra_hooks is not None:
        raise ValueError("only_hooks and extra_hooks are mutually exclusive")
    if only_hooks is not None and (not only_hooks or summary_output is None):
        raise ValueError("only_hooks requires at least one hook and a separate summary_output")
    destination = Path(summary_output).resolve() if summary_output is not None else output / "summary.json"
    if summary_output is not None:
        if destination.is_relative_to(output):
            raise ValueError("separate summary output must be outside the source run directory")
        if destination.exists():
            raise FileExistsError(destination)
    manifest = RunManifest.model_validate_json((output / "manifest.json").read_text())
    if manifest.status not in ("detail_complete", "complete"):
        raise ValueError("run has no complete detail to postprocess")
    detail = DetailReader(output, manifest)
    if only_hooks is not None:
        paths, hooks = tuple(only_hooks), []
    else:
        paths = manifest.config.summary_hooks if extra_hooks is None else tuple(extra_hooks)
        hooks = [HitRateSummary(), TransferSummary()]
    hooks.extend(import_symbol(p)() for p in paths)
    summary = {"schema_version": "1.0", "dataset_fingerprint": manifest.dataset_fingerprint,
               "summary_hooks": list(paths),
               "processor_implementations": [implementation_identity(h) for h in hooks]}
    if summary_output is not None:
        summary["source"] = dict(run_dir=str(output), manifest_sha256=digest_file(output / "manifest.json"),
            detail_index_sha256=digest_file(output / "detail/index.json"),
            source_summary_sha256=digest_file(output / "summary.json") if (output / "summary.json").exists() else None,
            run_source_fingerprint=manifest.source_fingerprint)
        summary["postprocess"] = dict(mode="selected_hooks" if only_hooks is not None else "full")
    for hook in hooks:
        if hook.namespace in summary:
            raise ValueError(f"duplicate summary namespace {hook.namespace}")
        summary[hook.namespace] = hook.process(detail, manifest)
    if summary_output is not None:
        summary["postprocess"].update(elapsed_seconds=round(monotonic() - started, 6),
                                     verified_detail_files=len(detail._verified))
        destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, summary)
    return summary

def histogram_hit_summary(histogram, top_k):
    """Exact linear percentiles of discrete hit_count/top_k without expanding rows."""
    counts = np.asarray(histogram, dtype=np.int64)
    total = int(counts.sum())
    result = dict(sample_unit="request_token_layer", scale="0_to_1", count=total,
                  percentile_method="linear", percentile_order="descending",
                  p10=None, p50=None, p90=None, p99=None, mean=None, median=None)
    if total:
        cumulative = np.cumsum(counts)
        for name, percentile in (("p10", 90), ("p50", 50), ("p90", 10), ("p99", 1)):
            position = (total - 1) * (percentile / 100)
            lower, upper = int(np.floor(position)), int(np.ceil(position))
            low_value = int(np.searchsorted(cumulative, lower, side="right")) / top_k
            high_value = int(np.searchsorted(cumulative, upper, side="right")) / top_k
            result[name] = low_value + (position - lower) * (high_value - low_value)
        result["mean"] = sum(i * int(n) for i, n in enumerate(counts)) / (total * top_k)
        result["median"] = result["p50"]
    return result


class TaskHitRateSummary:
    """Aggregate evaluation annotations by request metadata.task_type."""
    namespace = "hit_rate_by_task"

    def process(self, detail, manifest):
        requests = manifest.requests["evaluation"]
        labels = [r.metadata.get("task_type") or "unknown" for r in requests]
        names = sorted(set(labels))
        positions = {name: i for i, name in enumerate(names)}
        request_tasks = np.asarray([positions[name] for name in labels], dtype=np.int64)
        bins = manifest.dataset.top_k + 1
        histogram = np.zeros((len(names), bins), dtype=np.int64)
        for _, chunk in detail.access_chunks("evaluation"):
            group = request_tasks[chunk["request_index"]]
            histogram += np.bincount(group * bins + chunk["hit_count"],
                                     minlength=len(names) * bins).reshape(len(names), bins)
        groups = {}
        for i, name in enumerate(names):
            values = histogram[i]
            stats = histogram_hit_summary(values, manifest.dataset.top_k)
            stats.update(requests=labels.count(name),
                         expert_hits=sum(j * int(n) for j, n in enumerate(values)),
                         expert_accesses=stats["count"] * manifest.dataset.top_k,
                         hit_count_histogram=[int(n) for n in values])
            groups[name] = stats
        return dict(group_by="request.metadata.task_type", groups=groups)


# Public hook path, alongside the existing summary hooks.
from .token_position import TokenPositionHitRateSummary
