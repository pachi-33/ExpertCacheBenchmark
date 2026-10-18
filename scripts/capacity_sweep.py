"""Replay the held-out evaluation set at each requested expert-cache capacity."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from expertcache.contracts import DatasetManifest, RunManifest
from expertcache.data import digest_file, write_json
from expertcache.runner import Benchmark, code_identity, load_config

PERCENTAGES = (10, 15, 25, 30, 35, 50)
POLICIES = ("lfu", "strata")


def run_one(repo, output, policy, percent, expert_count, dataset_path, profile_path):
    repo, output = Path(repo), Path(output)
    name = f"{policy}-{percent:02d}pct"
    run_dir = output / name
    progress_path = output / f"{name}.progress.json"
    base = load_config(repo / "configs" / f"{policy}-eval.json")
    base = base.model_copy(update={"dataset": dataset_path, "initialization": base.initialization.model_copy(update={"profile_path": profile_path})})
    units = expert_count * percent // 100
    config = base.model_copy(update={"cache": base.cache.model_copy(update={"capacity_units": units})})
    started = time.monotonic()
    if run_dir.exists():
        manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_text())
        if (manifest.status != "complete" or manifest.config != config or
                manifest.source_fingerprint != code_identity()[1]):
            raise ValueError(f"Cannot reuse incomplete or different run: {run_dir}")
        result = json.loads((run_dir / "summary.json").read_text())
        print(f"REUSE {name}", flush=True)
    else:
        write_json(output / f"{name}.config.json", config.model_dump(mode="json"))
        write_json(progress_path, dict(status="running", completed_steps=0, elapsed_seconds=0))
        print(f"START {name}: {units} units", flush=True)
        def progress(phase, steps):
            elapsed = time.monotonic() - started
            write_json(progress_path, dict(status="running", phase=phase, completed_steps=steps, elapsed_seconds=round(elapsed, 2)))
            if steps % 1024 == 0:
                print(f"PROGRESS {name}: {steps}/10240 steps, {elapsed:.1f}s", flush=True)
        try:
            result = Benchmark(config).run(run_dir, progress)
        except Exception as exc:
            write_json(progress_path, dict(status="failed", error=str(exc), elapsed_seconds=round(time.monotonic()-started, 2)))
            raise
        write_json(progress_path, dict(status="complete", completed_steps=10240, elapsed_seconds=round(time.monotonic()-started, 2)))
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_text())
    expected = sum(min(r.token_count, config.scheduler.max_tokens or r.token_count)
                   for r in manifest.requests["evaluation"]) * len(manifest.dataset.valid_layers)
    if result["hit_rate"]["count"] != expected or result["hit_rate"]["percentile_order"] != "descending":
        raise ValueError(f"Unexpected annotation count/order: {name}")
    print(f"DONE {name}: mean={result['hit_rate']['mean']:.8f}, decisions={expected}", flush=True)
    return dict(policy=policy, cache_percent=percent, capacity_units=units, run_dir=str(run_dir),
                dataset_fingerprint=manifest.dataset_fingerprint, source_fingerprint=manifest.source_fingerprint,
                profile_sha256=manifest.inputs["profile_sha256"], manifest_sha256=digest_file(run_dir / "manifest.json"),
                summary_sha256=digest_file(run_dir / "summary.json"), requests=len(manifest.requests["evaluation"]),
                hit_rate=result["hit_rate"], transfer=result["transfer"])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", default="/private/tmp/ExpertCacheBenchmark-capacity-sweep")
    parser.add_argument("--dataset", help="optional local dataset manifest copy")
    parser.add_argument("--profile", help="optional local profile copy")
    parser.add_argument("--workers", type=int, default=4)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    output=Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base=load_config(repo/"configs/lfu-eval.json")
    dataset_path=str(Path(args.dataset or base.dataset).resolve())
    profile_path=str(Path(args.profile or base.initialization.profile_path).resolve())
    spec=DatasetManifest.model_validate_json(Path(dataset_path).read_text())
    expert_count=len(spec.valid_layers)*spec.num_experts
    results=[]
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(run_one,str(repo),str(output),policy,pct,expert_count,dataset_path,profile_path)
               for pct in PERCENTAGES for policy in POLICIES]
        for task in as_completed(tasks):
            results.append(task.result())
            write_json(output/"results.partial.json",dict(completed=len(results),total=len(tasks),results=results))
    results.sort(key=lambda r:(r["cache_percent"],r["policy"]))
    if len({r["dataset_fingerprint"] for r in results})!=1 or len({r["profile_sha256"] for r in results})!=1:
        raise ValueError("Sweep input fingerprints differ")
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=round(time.monotonic()-start,2),
                experiment=dict(expert_count=expert_count, valid_layers=list(spec.valid_layers),
                                num_experts_per_layer=spec.num_experts, batch_size=base.scheduler.batch_size,
                                profile_requests=20,evaluation_requests=80,max_tokens_per_request=512,
                                expected_decisions_per_run=80*512*len(spec.valid_layers),
                                percentile_order="descending",expert_size_units=1,workspace_units=1),
                results=results)
    write_json(output/"results.json",report)
    print(json.dumps({"report":str(output/"results.json"),"elapsed_seconds":report["elapsed_seconds"]}),flush=True)


if __name__=="__main__":
    main()
