# DeepSeek V4 routed-expert dataset (lossless uint8)

This orphan branch contains the complete int8-processed routed-expert dataset
for the `cascade_balanced_4k` prompt set.

## Contents

- `routes/*.npy`: 4000 lossless `int32` to `uint8` converted matrices.
- Route shape: `(512, 43, 6)`, dtype: `uint8`.
- Route tokens: 2,048,000.
- Route data: 528,384,000 bytes.
- `mapping.jsonl.partNNN.jsonl`: complete uint8 mapping metadata, split on
  JSONL row boundaries.
- `results.jsonl.partNNN.jsonl`: complete generation results and uint8 route
  references, split on JSONL row boundaries.
- `summary.json`, `conversion_manifest.json`, and `errors.jsonl`: original
  conversion metadata copied from the source dataset.
- `SHA256SUMS`: checksums of every original source file.

## Restore the split JSONL files

```bash
cd dataset/cascade_balanced_4k/router_results_uint8
cat mapping.part*.jsonl > mapping.jsonl
cat results.part*.jsonl > results.jsonl
sha256sum -c SHA256SUMS
```

## Batching

Each pushed commit is at most 1 MiB by raw added file size:

- route commits contain 7 files (`7 * 132224 = 925568` bytes);
- split JSONL parts are at most 950,000 bytes, split only at line boundaries.
