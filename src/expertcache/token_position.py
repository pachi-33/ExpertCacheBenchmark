"""Request-local token curves derived exclusively from sealed access detail."""
import numpy as np

from .contracts import TokenPositionMetadata, TokenPositionPoint, TokenPositionSummary


class TokenPositionHitRateSummary:
    namespace = "hit_rate_by_token_position"

    def process(self, detail, manifest):
        requests = manifest.requests["evaluation"]
        spec = manifest.dataset
        layers = np.asarray(spec.valid_layers, dtype=np.int64)
        top_k, layer_count = spec.top_k, len(layers)
        denominator = top_k * layer_count
        limit = manifest.config.scheduler.max_tokens
        lengths = np.asarray([min(r.token_count, limit or r.token_count) for r in requests], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        total = int(offsets[-1])
        hits = np.zeros(total, dtype=np.uint64)
        # Ragged request/token storage: absent suffixes are never treated as zeros.
        # Coverage bitsets also reject duplicates spanning access shards. Multiple
        # words support models with more than 64 observed layers.
        words = (layer_count + 63) // 64
        coverage = np.zeros(total * words, dtype=np.uint64)
        annotation_count = 0
        for _, chunk in detail.access_chunks("evaluation"):
            r = chunk["request_index"].astype(np.int64)
            t = chunk["token_index"].astype(np.int64)
            if np.any((r < 0) | (r >= len(requests))):
                raise ValueError("request index outside evaluation manifest")
            if np.any((t < 0) | (t >= lengths[r])):
                raise ValueError("token index outside evaluated request length")
            layer = np.searchsorted(layers, chunk["layer_id"])
            if np.any(layer >= layer_count) or np.any(layers[layer] != chunk["layer_id"]):
                raise ValueError("access contains an excluded or unknown layer")
            if np.any(chunk["hit_count"] > top_k):
                raise ValueError("hit count exceeds top-k")
            slot = offsets[r] + t
            keys = slot * layer_count + layer
            if np.unique(keys).size != len(keys):
                raise ValueError("duplicate request/token/layer annotation within shard")
            cell = slot * words + layer // 64
            bit = np.left_shift(np.uint64(1), (layer % 64).astype(np.uint64))
            if np.any(coverage[cell] & bit):
                raise ValueError("duplicate request/token/layer annotation across shards")
            np.bitwise_or.at(coverage, cell, bit)
            np.add.at(hits, slot, chunk["hit_count"].astype(np.uint64))
            annotation_count += len(chunk)
        expected = np.asarray([(1 << min(64, layer_count - 64 * w)) - 1 for w in range(words)], dtype=np.uint64)
        if annotation_count != total * layer_count or np.any(coverage.reshape(total, words) != expected):
            raise ValueError("incomplete request/token layer coverage")

        labels = [str(r.metadata.get("task_type") or "unknown") for r in requests]
        names = sorted(set(labels))
        task = np.asarray([names.index(label) for label in labels], dtype=np.int64)
        overall, by_task = [], {name: [] for name in names}
        for t in range(int(lengths.max()) if len(lengths) else 0):
            active = np.flatnonzero(lengths > t)
            values = hits[offsets[active] + t]
            overall.append(self._point(t, values, denominator))
            for i, name in enumerate(names):
                selected = values[task[active] == i]
                if len(selected):
                    by_task[name].append(self._point(t, selected, denominator))
        result = TokenPositionSummary(
            metadata=TokenPositionMetadata(valid_layers=spec.valid_layers, top_k=top_k,
                expert_accesses_per_request_token=denominator, request_count=len(requests),
                request_token_count=total, annotation_count=annotation_count),
            overall=overall, by_task=by_task)
        return result.model_dump(mode="json")

    @staticmethod
    def _point(token, hits, denominator):
        percentiles = np.percentile(hits / denominator, [90, 50, 10, 1], method="linear")
        total_hits, accesses = int(hits.sum()), len(hits) * denominator
        return TokenPositionPoint(token_index=token, request_count=len(hits),
            expert_hits=total_hits, expert_accesses=accesses, mean=total_hits / accesses,
            **dict(zip(("p10", "p50", "p90", "p99"), map(float, percentiles))))
