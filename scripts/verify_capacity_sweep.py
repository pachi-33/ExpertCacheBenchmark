import argparse
import json
import math
from pathlib import Path
import numpy as np
from expertcache.contracts import RunManifest, Profile
from expertcache.data import digest_file, write_json
from expertcache.detail import DetailReader


def check_belady(run, manifest, source, initial, masks):
    """Check every snapshot and greedy boundary without the policy's heap/index.

    Reverse NumPy indexing builds next-use times independently. Recorded events
    reconstruct residency; checking the retained/excluded boundary proves each
    post-demand state retains the nearest next uses among available candidates.
    """
    spec=manifest.dataset; capacity=manifest.config.cache.capacity_units
    assert manifest.config.visibility=='oracle'
    assert manifest.policy_metadata['objective']=='deduplicated_demand_loads'
    assert not manifest.policy_metadata['occurrence_weighted_hit_rate_optimal']
    requests,tokens,layers,top_k=source.shape; batch=manifest.config.scheduler.batch_size
    assert requests%batch==0
    groups=source.reshape(requests//batch,batch,tokens,layers,top_k).transpose(0,2,3,1,4).reshape(-1,batch*top_k).astype(np.int32)
    groups+=np.tile(np.array(spec.valid_layers,dtype=np.int32)*spec.num_experts,(requests//batch)*tokens)[:,None]
    horizon=len(groups); universe=spec.num_layers*spec.num_experts
    next_use=np.full(universe,horizon,dtype=np.int32)
    following=np.empty_like(groups)
    for time in range(horizon-1,-1,-1):
        keys=groups[time]
        following[time]=next_use[keys]
        next_use[keys]=time
    resident=np.zeros(universe,dtype=np.bool_)
    slots=np.empty(capacity,dtype=np.int32)
    positions=np.full(universe,-1,dtype=np.int32)
    initial_keys=[l*spec.num_experts+e for l,e in initial['cache']['resident']]
    used=len(initial_keys)
    slots[:used]=initial_keys; resident[initial_keys]=True
    positions[initial_keys]=np.arange(used)
    events=iter(e for e in DetailReader(run,manifest).events() if e['phase']=='evaluation')
    event=next(events,None); masks=masks.reshape(-1,batch*top_k); loads=0
    for time,keys in enumerate(groups):
        expected=resident[keys]
        assert np.array_equal(expected,masks[time]), f'Belady snapshot {time}'
        missing=set(map(int,keys[~expected])); loaded=set(); started={}; evicted=[]
        next_use[keys]=following[time]
        group=(time//layers,spec.valid_layers[time%layers])
        while event is not None and (event['step_id'],event['context_layer_id'])==group:
            assert event['hook']=='on_routes_ready'
            key=event['layer_id']*spec.num_experts+event['expert_id']
            kind=event['kind']
            if kind=='evict':
                assert resident[key]
                index=int(positions[key]); used-=1
                replacement=int(slots[used]); slots[index]=replacement; positions[replacement]=index
                positions[key]=-1; resident[key]=False; evicted.append(key)
            elif kind=='transfer_start':
                assert key in missing and key not in loaded and key not in started.values()
                started[event['transfer_id']]=key
            elif kind=='transfer_complete':
                assert started.pop(event['transfer_id'])==key and key not in loaded
                assert event['reason']=='demand' and event['units']==1
                loaded.add(key); loads+=1
                if event['destination']=='cache':
                    assert not resident[key] and used<capacity
                    slots[used]=key; positions[key]=used; used+=1; resident[key]=True
                else:
                    assert event['destination']=='workspace'
            else:
                raise AssertionError(f'unexpected Belady event {kind}')
            event=next(events,None)
        assert loaded==missing and not started
        assert used==capacity
        excluded=evicted+[key for key in missing if not resident[key]]
        if excluded and used:
            assert int(next_use[excluded].min())>=int(next_use[slots[:used]].max()), f'Belady boundary {time}'
    assert event is None
    return dict(belady_snapshots_verified=horizon,belady_greedy_boundaries_verified=horizon,
                belady_deduplicated_loads_verified=loads)


def check_run(entry):
    run=Path(entry['run_dir'])
    manifest=RunManifest.model_validate_json((run/'manifest.json').read_text())
    summary=json.loads((run/'summary.json').read_text())
    assert manifest.status=='complete'
    assert digest_file(run/'manifest.json')==entry['manifest_sha256']
    assert digest_file(run/'summary.json')==entry['summary_sha256']
    assert manifest.config.cache.capacity_units==entry['capacity_units']
    assert manifest.config.scheduler.batch_size==4
    assert manifest.config.scheduler.request_offset==20
    assert manifest.config.scheduler.max_requests is None and manifest.config.scheduler.max_tokens is None
    assert manifest.config.cache.default_expert_units==1 and not manifest.config.cache.expert_sizes
    requests=manifest.requests['evaluation']; spec=manifest.dataset
    assert len(requests)==80 and all(r.token_count==512 for r in requests)
    assert len(spec.valid_layers)==40 and spec.num_experts==256 and spec.top_k==6
    source=np.stack([np.load(Path(manifest.config.dataset).parent/r.route_file,allow_pickle=False)[:,spec.valid_layers,:]
                     for r in requests])
    layers=np.full(spec.num_layers,-1,dtype=np.int64)
    layers[list(spec.valid_layers)]=np.arange(len(spec.valid_layers))
    seen=np.zeros((80,512,40),dtype=np.bool_)
    histogram=np.zeros(7,dtype=np.int64)
    masks=[]; offset=0
    for _,chunk in DetailReader(run,manifest).access_chunks():
        r=chunk['request_index'].astype(np.int64); t=chunk['token_index'].astype(np.int64)
        l=layers[chunk['layer_id']]
        assert np.all(l>=0)
        assert np.array_equal(chunk['expert_ids'],source[r,t,l])
        assert np.array_equal(chunk['step_id'],r//4*512+t)
        ordered=((r//4*512+t)*40+l)*4+r%4
        assert np.array_equal(ordered,np.arange(offset,offset+len(chunk)))
        offset+=len(chunk)
        keys=(r*512+t)*40+l
        assert len(np.unique(keys))==len(keys)
        assert not np.any(seen[r,t,l])
        seen[r,t,l]=True
        histogram+=np.bincount(chunk['hit_count'].astype(np.int64),minlength=7)
        if entry['policy']=='belady': masks.append(chunk['hit_mask'].copy())
    assert np.all(seen)
    total=int(histogram.sum()); assert total==1638400
    hit=summary['hit_rate']; assert total==hit['count']
    mean=float(sum(i*int(c) for i,c in enumerate(histogram))/(6*total))
    assert abs(hit['mean']-mean)<1e-12
    def descending_rank(rank):
        accumulated=0
        for i in range(6,-1,-1):
            accumulated+=int(histogram[i])
            if rank<accumulated: return i/6
        raise AssertionError('rank out of range')
    for name,q in [('p10',.1),('p50',.5),('p90',.9),('p99',.99),('median',.5)]:
        pos=(total-1)*q; lo,hi=math.floor(pos),math.ceil(pos)
        expected=descending_rank(lo)+(descending_rank(hi)-descending_rank(lo))*(pos-lo)
        assert abs(hit[name]-expected)<1e-12
    assert hit['percentile_order']=='descending'
    profile=Profile.model_validate_json(Path(manifest.config.initialization.profile_path).read_text())
    assert not set(profile.source_request_ids)&{r.request_id for r in requests}
    expected={e.key for e in sorted(profile.entries,key=lambda e:(-e.count,e.layer_id,e.expert_id))[:entry['capacity_units']]}
    initial=json.loads((run/'initial_state.json').read_text())
    final=json.loads((run/'final_state.json').read_text())
    assert {tuple(k) for k in initial['cache']['resident']}==expected
    assert len(final['cache']['resident'])==entry['capacity_units']
    if entry['policy']=='strata':
        for l in spec.valid_layers:
            assert sum(k[0]==l for k in initial['cache']['resident'])==sum(k[0]==l for k in final['cache']['resident'])
    extra={}
    if entry['policy']=='belady':
        extra=check_belady(run,manifest,source,initial,np.concatenate(masks))
        transfer=summary['transfer']; phase=transfer['phases']['evaluation']
        assert phase['prefetch_units']==phase['adaptation_units']==phase['other_units']==0
        assert phase['load_count']==phase['demand_units']==transfer['evaluation_unique_miss_demands']==extra['belady_deduplicated_loads_verified']
    print(f"VERIFIED {entry['policy']} {entry['cache_percent']}%: {mean:.8f}, all {total} annotations",flush=True)
    return dict(policy=entry['policy'],cache_percent=entry['cache_percent'],annotated_decisions=total,
                hit_count_histogram=histogram.tolist(),mean_independently_recomputed=mean,
                all_routes_match=True,coverage_complete=True,descending_percentiles_verified=True,
                initialization_verified=True,**extra)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('report'); parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=json.loads(Path(args.report).read_text())
    entries=report['results']; policies={e['policy'] for e in entries}
    assert policies and policies <= {'lfu','strata','belady'}
    assert len(entries)==len(policies)*6
    assert {(e['policy'],e['cache_percent']) for e in entries}=={(p,c) for p in policies for c in (10,15,25,30,35,50)}
    for field in ('dataset_fingerprint','profile_sha256'):
        assert len({e[field] for e in entries})==1
    revisions={}; policy_hashes={}
    for e in entries:
        m=RunManifest.model_validate_json((Path(e['run_dir'])/'manifest.json').read_text())
        revisions.setdefault(e['policy'],set()).add(m.framework_revision)
        policy_hashes.setdefault(e['policy'],set()).add(m.inputs['policy_implementation']['source_sha256'])
    assert all(len(v)==1 for v in revisions.values()) and all(len(v)==1 for v in policy_hashes.values())
    checks=[check_run(e) for e in entries]
    write_json(args.output,dict(verified_runs=len(checks),verified_decisions=sum(c['annotated_decisions'] for c in checks),checks=checks))


if __name__=='__main__': main()
