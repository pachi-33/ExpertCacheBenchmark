import gzip
import json
from pathlib import Path
import numpy as np
from .contracts import AccessRecord, AccessShard, DetailIndex
from .data import digest_file, write_json


def access_dtype(spec):
    return np.dtype([("probe_tick", "<u8"), ("step_id", "<u8"), ("request_index", "<u4"), ("token_index", "<u4"),
                     ("layer_id", "<u4"), ("expert_ids", spec.route_dtype, (spec.top_k,)),
                     ("hit_mask", "?", (spec.top_k,)), ("hit_count", "<u4")])


class DetailWriter:
    def __init__(self, output, spec, chunk_rows):
        self.root = Path(output) / "detail"
        self.root.mkdir()
        self.dtype = access_dtype(spec)
        self.buffer = np.empty(chunk_rows, dtype=self.dtype)
        self.position, self.phase, self.event_count = 0, None, 0
        self.shards = []
        self.events = gzip.open(self.root / "cache_events.jsonl.gz", "wt", encoding="utf-8", compresslevel=1)

    def append_access(self, phase, routes, probe):
        if self.phase != phase:
            self.flush()
            self.phase = phase
        for row, experts, hits in zip(routes.step.rows, routes.expert_ids, probe.hit_mask, strict=True):
            self.buffer[self.position] = (probe.cache_tick, routes.step.step_id, row.request_index, row.token_index,
                                          routes.layer_id, experts, hits, sum(hits))
            self.position += 1
            if self.position == len(self.buffer): self.flush()

    def append_events(self, events):
        if events:
            self.events.write("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events))
            self.event_count += len(events)

    def flush(self):
        if not self.position: return
        name = f"access-{len(self.shards):06d}.npy"
        path = self.root / name
        np.save(path, self.buffer[:self.position], allow_pickle=False)
        self.shards.append(AccessShard(path=name, phase=self.phase, rows=self.position, sha256=digest_file(path)))
        self.position = 0

    def close(self, sealed=True):
        self.flush()
        self.events.close()
        index = DetailIndex(sealed=sealed, shards=tuple(self.shards), events_path="cache_events.jsonl.gz",
                            events_sha256=digest_file(self.root / "cache_events.jsonl.gz"), event_count=self.event_count)
        write_json(self.root / "index.json", index.model_dump(mode="json"))


class DetailReader:
    def __init__(self, output, manifest):
        self.root = Path(output) / "detail"
        self.manifest = manifest
        self.index = DetailIndex.model_validate_json((self.root / "index.json").read_text())
        if not self.index.sealed: raise ValueError("detail is incomplete")
        self.dtype = access_dtype(manifest.dataset)
        self._verified = set()

    def _path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()): raise ValueError("detail path escapes run")
        return path

    def _verify(self, path, digest):
        if path not in self._verified:
            if digest_file(path) != digest: raise ValueError(f"detail checksum mismatch: {path}")
            self._verified.add(path)

    def access_chunks(self, phase="evaluation"):
        for shard in self.index.shards:
            if phase is not None and shard.phase != phase: continue
            path = self._path(shard.path)
            self._verify(path, shard.sha256)
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            if a.dtype != self.dtype or a.shape != (shard.rows,): raise ValueError("access shard shape/dtype mismatch")
            if not np.array_equal(a["hit_count"], a["hit_mask"].sum(axis=1)): raise ValueError("invalid hit counts")
            yield shard.phase, a

    def events(self):
        path = self._path(self.index.events_path)
        self._verify(path, self.index.events_sha256)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f: yield json.loads(line)

    def records(self, phase="evaluation"):
        for current_phase, chunk in self.access_chunks(phase):
            requests = self.manifest.requests[current_phase]
            for row in chunk:
                yield AccessRecord(probe_tick=int(row["probe_tick"]), phase=current_phase, step_id=int(row["step_id"]),
                                   request_id=requests[int(row["request_index"])].request_id,
                                   token_index=int(row["token_index"]), layer_id=int(row["layer_id"]),
                                   expert_ids=tuple(map(int, row["expert_ids"])),
                                   hit_mask=tuple(map(bool, row["hit_mask"])), hit_count=int(row["hit_count"]))
