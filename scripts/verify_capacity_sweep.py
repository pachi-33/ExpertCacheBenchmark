import argparse
import json
import math
from pathlib import Path
import numpy as np
from expertcache.contracts import RunManifest, Profile
from expertcache.data import digest_file, write_json
from expertcache.detail import DetailReader


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
    for _,chunk in DetailReader(run,manifest).access_chunks():
        r=chunk['request_index'].astype(np.int64); t=chunk['token_index'].astype(np.int64)
        l=layers[chunk['layer_id']]
        assert np.all(l>=0)
        assert np.array_equal(chunk['expert_ids'],source[r,t,l])
        assert np.array_equal(chunk['step_id'],r//4*512+t)
        keys=(r*512+t)*40+l
        assert len(np.unique(keys))==len(keys)
        assert not np.any(seen[r,t,l])
        seen[r,t,l]=True
        histogram+=np.bincount(chunk['hit_count'].astype(np.int64),minlength=7)
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
    print(f"VERIFIED {entry['policy']} {entry['cache_percent']}%: {mean:.8f}, all {total} annotations",flush=True)
    return dict(policy=entry['policy'],cache_percent=entry['cache_percent'],annotated_decisions=total,
                hit_count_histogram=histogram.tolist(),mean_independently_recomputed=mean,
                all_routes_match=True,coverage_complete=True,descending_percentiles_verified=True,
                initialization_verified=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('report'); parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=json.loads(Path(args.report).read_text())
    entries=report['results']; assert len(entries)==12
    assert {(e['policy'],e['cache_percent']) for e in entries}=={(p,c) for p in ('lfu','strata') for c in (10,15,25,30,35,50)}
    for field in ('dataset_fingerprint','profile_sha256'):
        assert len({e[field] for e in entries})==1
    revisions=set(); policy_hashes={}
    for e in entries:
        m=RunManifest.model_validate_json((Path(e['run_dir'])/'manifest.json').read_text())
        revisions.add(m.framework_revision)
        policy_hashes.setdefault(e['policy'],set()).add(m.inputs['policy_implementation']['source_sha256'])
    assert len(revisions)==1 and all(len(v)==1 for v in policy_hashes.values())
    checks=[check_run(e) for e in entries]
    write_json(args.output,dict(verified_runs=len(checks),verified_decisions=sum(c['annotated_decisions'] for c in checks),checks=checks))


if __name__=='__main__': main()
