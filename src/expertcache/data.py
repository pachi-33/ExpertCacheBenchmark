from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from .contracts import DatasetManifest, TraceRequest, Profile, ProfileEntry


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Dataset:
    """Full trace storage is private to the runner; policies get causal views."""
    def __init__(self, manifest_path):
        self.path = Path(manifest_path).resolve()
        self.spec = DatasetManifest.model_validate_json(self.path.read_text())
        self.arrays = {}
        h = hashlib.sha256(self.spec.model_dump_json().encode())
        for request in self.spec.requests:
            path = (self.path.parent / request.route_file).resolve()
            if not path.is_relative_to(self.path.parent):
                raise ValueError("route_file must remain inside the dataset directory")
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            if a.dtype.name != self.spec.route_dtype or a.shape != (request.token_count, self.spec.num_layers, self.spec.top_k):
                raise ValueError(f"route shape/dtype mismatch: {path}")
            valid = a[:, self.spec.valid_layers, :]
            if np.any(valid >= self.spec.num_experts):
                raise ValueError(f"expert ID out of range: {path}")
            if self.spec.top_k > 1 and np.any(np.diff(np.sort(valid, axis=-1).astype(np.int64), axis=-1) == 0):
                raise ValueError(f"duplicate top-k IDs in a valid layer: {path}")
            h.update(digest_file(path).encode())
            self.arrays[request.request_id] = a
        self.fingerprint = h.hexdigest()

    def routes(self, request_id, token_index, layer_id):
        return tuple(int(x) for x in self.arrays[request_id][token_index, layer_id])


def import_router_dump(directory, output, valid_layers, excluded_reason):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.parent != directory:
        raise ValueError("manifest must be written beside mapping.jsonl")
    summary = json.loads((directory / "summary.json").read_text())
    requests = []
    for line in (directory / "mapping.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["status"] != "ok" or not row["route_alignment_matches"]:
            raise ValueError("source contains failed or unaligned requests")
        requests.append(TraceRequest(request_id=row["prompt_id"], route_file=row["route_file"],
                                     token_count=row["route_shape"][0],
                                     metadata={k: row[k] for k in ("task_type", "prompt_tokens", "sample_id")}))
    spec = DatasetManifest(model_id=summary["model"].split("/")[-1], num_layers=summary["num_layers"],
                           num_experts=summary["num_routed_experts"], top_k=summary["top_k"],
                           route_dtype=summary["route_dtype"], valid_layers=tuple(valid_layers),
                           excluded_layer_reasons={l: excluded_reason for l in range(summary["num_layers"]) if l not in valid_layers},
                           requests=tuple(requests))
    if output.exists():
        raise FileExistsError(output)
    write_json(output, spec.model_dump(mode="json"))
    try:
        Dataset(output)
    except Exception:
        output.unlink()
        raise
    return spec


def build_profile(dataset, requests, max_tokens=None):
    counts = Counter()
    for request in requests:
        for token in range(min(request.token_count, max_tokens or request.token_count)):
            for layer in dataset.spec.valid_layers:
                counts.update((layer, e) for e in dataset.routes(request.request_id, token, layer))
    return Profile(model_id=dataset.spec.model_id, source_fingerprint=dataset.fingerprint,
                   source_request_ids=tuple(r.request_id for r in requests),
                   provenance=f"observed routing counts; max_tokens={max_tokens}",
                   entries=tuple(ProfileEntry(layer_id=l, expert_id=e, count=c) for (l, e), c in sorted(counts.items())))
