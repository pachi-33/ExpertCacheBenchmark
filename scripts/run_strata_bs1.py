"""Reproducible shuffled BS1 Strata experiment with a disjoint profile split."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from expertcache.contracts import CacheConfig, InitConfig, PolicyConfig, RunConfig, RunManifest, SchedulerConfig
from expertcache.data import Dataset, build_profile, digest_file, write_json
from expertcache.native_strata import NativeStrataBenchmark
from expertcache.runner import Benchmark
from expertcache.scheduler import select_requests


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--cache-percent',type=int,default=35)
    parser.add_argument('--seed',type=int,default=20261018)
    parser.add_argument('--profile-requests',type=int,default=20)
    parser.add_argument('--backend',choices=('native','python'),default='native')
    args=parser.parse_args()
    if not 0 < args.cache_percent <= 100 or args.profile_requests < 1:
        parser.error('cache-percent must be in (0,100]; profile-requests must be positive')
    output=Path(args.output).resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    started=time.monotonic()
    created=datetime.now(timezone.utc).isoformat()
    dataset=Dataset(Path(args.dataset).resolve())
    scheduler=SchedulerConfig(batch_size=1,order='shuffled',seed=args.seed,request_offset=args.profile_requests)
    ordered=select_requests(dataset,scheduler.model_copy(update={'request_offset':0}))
    if args.profile_requests >= len(ordered): raise ValueError('Profile must leave evaluation requests')
    training,evaluation=ordered[:args.profile_requests],ordered[args.profile_requests:]
    profile=build_profile(dataset,training)
    profile_path=output/'profile-train20.json' if args.profile_requests==20 else output/'profile.json'
    write_json(profile_path,profile.model_dump(mode='json'))
    expert_count=len(dataset.spec.valid_layers)*dataset.spec.num_experts
    capacity=expert_count*args.cache_percent//100
    config=RunConfig(dataset=str(dataset.path),cache=CacheConfig(capacity_units=capacity,workspace_units=1),
        initialization=InitConfig(mode='profile',profile_path=str(profile_path)),
        policy=PolicyConfig(name='strata'),scheduler=scheduler,
        summary_hooks=('expertcache.summary:TaskHitRateSummary',))
    write_json(output/'experiment.config.json',config.model_dump(mode='json'))
    order=[r.request_id for r in ordered]
    write_json(output/'request-order.json',dict(seed=args.seed,profile=order[:args.profile_requests],evaluation=order[args.profile_requests:]))
    total_steps=sum(r.token_count for r in evaluation)
    expected=total_steps*len(dataset.spec.valid_layers)
    preparation_seconds=time.monotonic()-started
    write_json(output/'progress.json',dict(status='running',stage='replay',completed_steps=0,total_steps=total_steps))
    print(f'START: {len(evaluation)} evaluation requests, {total_steps} steps, {expected} annotations, {capacity} cache units',flush=True)
    def progress(phase,steps):
        elapsed=time.monotonic()-started
        write_json(output/'progress.json',dict(status='running',stage='replay',phase=phase,
            completed_steps=steps,total_steps=total_steps,elapsed_seconds=round(elapsed,2),
            estimated_remaining_replay_seconds=round((elapsed-preparation_seconds)*(total_steps-steps)/steps,2) if steps else None))
    run_dir=output/'strata-35pct' if args.cache_percent==35 else output/f'strata-{args.cache_percent:02d}pct'
    try:
        bench=NativeStrataBenchmark(config,output/'native-build') if args.backend=='native' else Benchmark(config)
        result=bench.run(run_dir,progress)
    except Exception as exc:
        write_json(output/'progress.json',dict(status='failed',error=str(exc),elapsed_seconds=round(time.monotonic()-started,2)))
        raise
    elapsed=time.monotonic()-started
    manifest=RunManifest.model_validate_json((run_dir/'manifest.json').read_text())
    groups=result['hit_rate_by_task']['groups']
    assert result['hit_rate']['count']==expected
    assert sum(group['count'] for group in groups.values())==expected
    assert sum(group['requests'] for group in groups.values())==len(evaluation)
    hits=sum(group['expert_hits'] for group in groups.values())
    assert abs(hits/(expected*dataset.spec.top_k)-result['hit_rate']['mean'])<1e-14
    for label,group in groups.items():
        assert group['count']==sum(r.token_count for r in evaluation if r.metadata.get('task_type')==label)*len(dataset.spec.valid_layers)
        assert group['p10']>=group['p50']>=group['p90']>=group['p99']
    report=dict(created_at_utc=created,elapsed_seconds=round(elapsed,3),
        preparation_seconds=round(preparation_seconds,3),benchmark_seconds=round(elapsed-preparation_seconds,3),
        experiment=dict(policy='strata',backend=args.backend,batch_size=1,cache_percent=args.cache_percent,
            capacity_units=capacity,workspace_units=1,expert_units=1,expert_count=expert_count,
            request_order='shuffled',seed=args.seed,initialization='profile',profile_requests=len(training),
            evaluation_requests=len(evaluation),evaluation_tokens=total_steps,expected_annotations=expected,
            percentile_order='descending',sample_unit='request_token_layer',
            profile_tasks=dict(Counter(r.metadata.get('task_type','unknown') for r in training)),
            evaluation_tasks=dict(Counter(r.metadata.get('task_type','unknown') for r in evaluation))),
        run_dir=str(run_dir),config=config.model_dump(mode='json'),dataset_fingerprint=dataset.fingerprint,
        framework_revision=manifest.framework_revision,source_fingerprint=manifest.source_fingerprint,
        profile_sha256=digest_file(profile_path),request_order_sha256=digest_file(output/'request-order.json'),
        manifest_sha256=digest_file(run_dir/'manifest.json'),summary_sha256=digest_file(run_dir/'summary.json'),
        detail_index_sha256=digest_file(run_dir/'detail/index.json'),execution_backend=manifest.inputs.get('execution_backend'),
        policy_metadata=manifest.policy_metadata,hit_rate=result['hit_rate'],hit_rate_by_task=groups,transfer=result['transfer'])
    write_json(output/'results.json',report)
    lines=['# Full router dataset · Strata · BS=1 · Cache 35%', '',
        f'- Dataset fingerprint: `{dataset.fingerprint}`',
        f'- Request order: shuffled with seed `{args.seed}`; first {len(training)} requests build a profile, remaining {len(evaluation)} are evaluated.',
        f'- Cache: {capacity}/{expert_count} units; workspace: 1 unit; cache persists across requests.',
        '- Sample unit: request/token/layer top-k hit rate. Percentiles rank from high to low.',
        f'- Execution backend: `{args.backend}`. Full elapsed time: {elapsed:.3f} seconds (including profile, replay, sealing and summary).', '',
        '| Task | Requests | Layer annotations | P10 | P50 | P90 | P99 | Mean |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,requests,stats in [('Overall',len(evaluation),result['hit_rate'])]+[(k,v['requests'],v) for k,v in sorted(groups.items())]:
        cells=[name,f'{requests:,}',f'{stats["count"]:,}']+[f'{stats[k]*100:.2f}%' for k in ('p10','p50','p90','p99','mean')]
        lines.append('| '+' | '.join(cells)+' |')
    lines.extend(['','Profile tasks: `'+json.dumps(report['experiment']['profile_tasks'],sort_keys=True)+'`.',
        '',f'Evaluation weight transfers: {result["transfer"]["phases"]["evaluation"]["units"]:,} virtual units.',
        '', 'Task groups share one continuous cache; they are grouped during postprocessing, not replayed independently.',
        '', 'This profile split and shuffled dataset differ from the earlier sample100 experiments; compare configurations before interpreting differences.'])
    (output/'results.md').write_text('\n'.join(lines)+'\n')
    write_json(output/'progress.json',dict(status='complete',completed_steps=total_steps,total_steps=total_steps,
                                         elapsed_seconds=round(elapsed,3),annotations=expected))
    print(json.dumps({'report':str(output/'results.json'),'elapsed_seconds':elapsed,'hit_rate':result['hit_rate'],'by_task':groups},indent=2),flush=True)


if __name__=='__main__':main()
