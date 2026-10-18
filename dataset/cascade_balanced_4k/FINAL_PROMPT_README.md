# Original final prompt dataset

This directory contains the 4000-row rendered prompt dataset corresponding to
the uint8 routed-expert results in `router_results_uint8/`.

The source file is split only at JSONL row boundaries so that every uploaded
part is below 1 MiB. Restore the original file with:

```bash
cat final_prompt.part*.jsonl > final_prompt.jsonl
sha256sum -c final_prompt.sha256
```

- Original rows: 4000
- Original bytes: 195070898
- SHA-256: `33ae453a0350d175b82d72eddbfb6dea34f974b857a877ac1edfccb0190fd94b`
- Manifest SHA-256: `ae794918f87a7c218224afc73495731a06087e739603c833cd300a0b83a3d436`
- Split parts: 228
