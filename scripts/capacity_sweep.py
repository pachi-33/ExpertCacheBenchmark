"""Replay the held-out evaluation set at each requested expert-cache capacity."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from expertcache.contracts import DatasetManifest, RunManifest
from expertcache.data import Dataset, digest_file, write_json
from expertcache.runner import Benchmark, code_identity, load_config

PERCENTAGES = (10, 15, 25, 30, 35, 50)
POLICIES = ("lfu", "strata", "belady")


def run_one(repo, output, policy, percent, expert_count, dataset_path, profile_path, batch_size=None):
    repo, output = Path(repo), Path(output)
    name = f"{policy}-{percent:02d}pct"
    run_dir = output / name
    progress_path = output / f"{name}.progress.json"
    base = load_config(repo / "configs" / f"{policy}-eval.json")
    if batch_size is not None:
        base = base.model_copy(update={"scheduler": base.scheduler.model_copy(update={"batch_size": batch_size})})
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
        print(f"START {name}: {units} units, batch={config.scheduler.batch_size}", flush=True)
        def progress(phase, steps):
            elapsed = time.monotonic() - started
            write_json(progress_path, dict(status="running", phase=phase, completed_steps=steps, total_steps=total_steps, elapsed_seconds=round(elapsed, 2)))
            if steps % 1024 == 0:
                print(f"PROGRESS {name}: {steps}/{total_steps} steps, {elapsed:.1f}s", flush=True)
        try:
            benchmark = Benchmark(config)
            total_steps = sum(1 for _ in benchmark.scheduler.steps(benchmark.requests, config.scheduler))
            result = benchmark.run(run_dir, progress)
        except Exception as exc:
            write_json(progress_path, dict(status="failed", error=str(exc), elapsed_seconds=round(time.monotonic()-started, 2)))
            raise
        write_json(progress_path, dict(status="complete", completed_steps=total_steps, total_steps=total_steps, elapsed_seconds=round(time.monotonic()-started, 2)))
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_text())
    expected = sum(min(r.token_count, config.scheduler.max_tokens or r.token_count)
                   for r in manifest.requests["evaluation"]) * len(manifest.dataset.valid_layers)
    if result["hit_rate"]["count"] != expected or result["hit_rate"]["percentile_order"] != "descending":
        raise ValueError(f"Unexpected annotation count/order: {name}")
    print(f"DONE {name}: mean={result['hit_rate']['mean']:.8f}, decisions={expected}", flush=True)
    return dict(policy=policy, batch_size=config.scheduler.batch_size, cache_percent=percent, capacity_units=units, run_dir=str(run_dir),
                dataset_fingerprint=manifest.dataset_fingerprint, source_fingerprint=manifest.source_fingerprint,
                profile_sha256=manifest.inputs["profile_sha256"], manifest_sha256=digest_file(run_dir / "manifest.json"),
                summary_sha256=digest_file(run_dir / "summary.json"), requests=len(manifest.requests["evaluation"]),
                visibility=manifest.config.visibility, policy_metadata=manifest.policy_metadata,
                hit_rate=result["hit_rate"], transfer=result["transfer"])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", default="/private/tmp/ExpertCacheBenchmark-capacity-sweep")
    parser.add_argument("--dataset", help="optional local dataset manifest copy")
    parser.add_argument("--profile", help="optional local profile copy")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, help="override evaluation batch size for all selected policies")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--extend-report", help="preserve existing results and append disjoint policies")
    args=parser.parse_args()
    if args.workers < 1 or len(set(args.policies)) != len(args.policies):
        parser.error("workers must be positive and policies must be unique")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("batch-size must be positive")
    repo=Path(__file__).resolve().parents[1]
    output=Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base=load_config(repo/"configs/lfu-eval.json")
    if args.batch_size is not None:
        base=base.model_copy(update={"scheduler":base.scheduler.model_copy(update={"batch_size":args.batch_size})})
    dataset_path=str(Path(args.dataset or base.dataset).resolve())
    profile_path=str(Path(args.profile or base.initialization.profile_path).resolve())
    spec=DatasetManifest.model_validate_json(Path(dataset_path).read_text())
    expert_count=len(spec.valid_layers)*spec.num_experts
    previous=json.loads(Path(args.extend_report).read_text()) if args.extend_report else {}
    results=list(previous.get("results", []))
    if results and previous['experiment']['batch_size']!=base.scheduler.batch_size:
        raise ValueError("Use separate reports for different batch sizes")
    if any(r["policy"] in args.policies for r in results):
        raise ValueError("extended policies overlap the existing report")
    if results:
        fingerprint=Dataset(dataset_path).fingerprint
        profile_sha=digest_file(profile_path)
        if any(r['dataset_fingerprint']!=fingerprint or r['profile_sha256']!=profile_sha for r in results):
            raise ValueError("extension inputs differ from the existing report")
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(run_one,str(repo),str(output),policy,pct,expert_count,dataset_path,profile_path,args.batch_size)
               for pct in PERCENTAGES for policy in args.policies]
        for task in as_completed(tasks):
            results.append(task.result())
            write_json(output/"results.partial.json",dict(completed=len(results),total=len(tasks)+len(previous.get("results",[])),results=results))
    results.sort(key=lambda r:(r["cache_percent"],r["policy"]))
    if len({r["dataset_fingerprint"] for r in results})!=1 or len({r["profile_sha256"] for r in results})!=1:
        raise ValueError("Sweep input fingerprints differ")
    report=dict(previous, created_at_utc=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=round(time.monotonic()-start,2),
                experiment=dict(expert_count=expert_count, valid_layers=list(spec.valid_layers),
                                num_experts_per_layer=spec.num_experts, batch_size=base.scheduler.batch_size,
                                profile_requests=20,evaluation_requests=80,max_tokens_per_request=512,
                                expected_decisions_per_run=80*512*len(spec.valid_layers),
                                percentile_order="descending",expert_size_units=1,workspace_units=1),
                results=results)
    manifests=[RunManifest.model_validate_json((Path(r['run_dir'])/'manifest.json').read_text()) for r in results]
    report['framework_revisions']=sorted({m.framework_revision for m in manifests if m.framework_revision})
    report['policy_source_sha256']={r['policy']:m.inputs['policy_implementation']['source_sha256'] for r,m in zip(results,manifests)}
    report['source_fingerprint_variants']=sorted({r['source_fingerprint'] for r in results})
    if 'belady' in {r['policy'] for r in results}:
        report['belady_scope']=dict(visibility='oracle', objective='deduplicated_demand_loads',
                                   assumptions=['uniform_expert_size','global_pool','demand_only_admission','bypass_allowed','fixed_initial_residency','batch_snapshot'],
                                   occurrence_weighted_hit_rate_is_strict_upper_bound=False)
    write_json(output/"results.json",report)
    print(json.dumps({"report":str(output/"results.json"),"elapsed_seconds":report["elapsed_seconds"]}),flush=True)


if __name__=="__main__":
    main()
