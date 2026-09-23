"""Thor pilot: Python owns provenance/Arrow decoding/oracles, Rust generates graphs."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
from oracle import load_source, sha256, verify_subset

HERE=Path(__file__).resolve().parent

def canonical(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    evidence=args.output/'evidence'; evidence.mkdir()
    start=time.monotonic()
    protocol=json.loads((HERE/'protocol.json').read_text())
    dump(evidence/'protocol.json',protocol)
    copied=evidence/'source'; copied.mkdir()
    for file in HERE.iterdir():
        if file.is_file() and file.suffix in ('.rs','.py','.json'): shutil.copy2(file,copied/file.name)
    try:
        expected=protocol['source']
        graph=args.data_root/expected['graph_relative_path']
        qualification=args.data_root/expected['qualification_relative_path']
        meta_path=args.data_root/'banc_888_meta.feather'
        for path,digest in ((graph,expected['graph_sha256']),(qualification,expected['qualification_sha256']),(meta_path,expected['metadata_sha256'])):
            if sha256(path)!=digest: raise ValueError('source hash mismatch: '+path.name)
        source_before={str(p):(p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino) for p in (graph,qualification,meta_path)}
        prior=json.loads(qualification.read_text())
        if not prior['pairwise']['pairwise_equality'] or not prior['graph']['independent_readback_passed']: raise ValueError('missing successful graph qualification')
        if canonical(prior['graph_contract'])!=expected['graph_identity_sha256']: raise ValueError('graph identity mismatch')
        source=load_source(graph)
        ids,offsets,pre,post,weights=source
        if len(ids)!=expected['source_nodes'] or len(post)!=expected['source_pairs'] or int(weights.sum())!=expected['source_contacts']: raise ValueError('source cardinality mismatch')
        meta=feather.read_table(meta_path,columns=['banc_888_id','region'])
        if pa.types.is_floating(meta['banc_888_id'].type): raise ValueError('floating-point IDs are forbidden')
        root_ids=pc.cast(meta['banc_888_id'],pa.uint64(),safe=True)
        if root_ids.null_count: raise ValueError('null root ID')
        roots=root_ids.to_numpy(); order=np.argsort(roots)
        if not np.array_equal(roots[order],ids): raise ValueError('metadata/graph node identity mismatch')
        labels=meta['region'].to_pylist()
        normalized=[]
        for label in labels:
            if label is not None and not isinstance(label,str): raise ValueError('non-string anatomical block')
            normalized.append(label.strip() if label is not None and label.strip() else None)
        keys=[json.dumps({'region':label},sort_keys=True,separators=(',',':')) for label in normalized]
        registry=sorted(set(keys)); lookup={key:i for i,key in enumerate(registry)}
        groups=np.array([lookup[keys[int(i)]] for i in order],dtype=np.uint32)
        blocks=args.output/'blocks.tsv'
        with blocks.open('x') as f:
            for root,group in zip(ids,groups): f.write(f'{int(root)}\t{int(group)}\n')
        dump(evidence/'block_policy.json',{'annotation':'region','labels':registry,'counts':dict(Counter(keys)),'node_block_table_sha256':sha256(blocks),'not_claimed':'functional communities or matched modularity'})
        observations=[]
        for policy in ('weak_bfs','ranked'):
            for count in protocol['grid'][policy+'_sizes']:
                for seed in protocol['grid']['seeds']:
                    name=f'{policy}-n{count}-s{seed}'
                    first=args.output/name; replay=args.output/(name+'-replay')
                    logs=[]
                    t=time.monotonic()
                    for destination in (first,replay):
                        result=subprocess.run([str(args.worker),str(graph),str(blocks),str(destination),str(count),str(seed),policy],capture_output=True,text=True,timeout=protocol['execution']['per_process_timeout_seconds'])
                        logs.append({'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
                        if result.returncode: raise RuntimeError('Rust generation failed: '+result.stderr)
                    measured=verify_subset(source,groups,first)
                    replay_hashes={p.name:sha256(p) for p in replay.iterdir() if p.is_file()}
                    if replay_hashes!={name:item['sha256'] for name,item in measured['files'].items()}: raise ValueError('non-deterministic replay: '+name)
                    measured.update(case_id=name,replay_identical=True,elapsed_seconds_including_replay_and_oracle=time.monotonic()-t)
                    observations.append(measured)
                    dump(evidence/(name+'.json'),measured)
                    dump(evidence/(name+'.process.json'),logs)
                    print(json.dumps({'case':name,'nodes':count,'pairs':measured['pairs'],'replay_identical':True,'blocks':measured['block_labels'],'changed_controls':sum(a['control_changed'] for a in measured['arms'][1:])}),flush=True)
        for path,digest in ((graph,expected['graph_sha256']),(qualification,expected['qualification_sha256']),(meta_path,expected['metadata_sha256'])):
            now=path.stat()
            if (now.st_size,now.st_mtime_ns,now.st_ino)!=source_before[str(path)] or sha256(path)!=digest: raise ValueError('source changed during pilot')
        controls=[a for x in observations for a in x['arms'][1:]]
        report={
            'milestone':'V888-BOOL-0.2a','parent_milestone':'V888-BOOL-0.2',
            'classification':'exploratory_source_software_control_invariants_not_task_or_uniform_null_qualification',
            'source_commit':os.environ.get('GITHUB_SHA','unavailable'),
            'protocol_sha256':canonical(protocol),'worker_sha256':sha256(args.worker),
            'source':expected,'source_rehashed_after_run':True,
            'runtime':{'host':platform.node(),'architecture':platform.machine(),'runner':os.environ.get('RUNNER_NAME','unavailable'),'run_id':os.environ.get('GITHUB_RUN_ID','unavailable'),'elapsed_seconds':time.monotonic()-start,'python_max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'largest_child_max_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss},
            'cases_completed':len(observations),'graphs_per_case':5,'all_replays_identical':all(x['replay_identical'] for x in observations),
            'all_declared_invariants_passed_independent_oracle':True,
            'control_graphs':len(controls),'changed_control_graphs':sum(a['control_changed'] for a in controls),
            'unchanged_controls':[{'case':x['case_id'],'arm':a['arm'],'pairs':x['pairs'],'attempts':a['attempts'],'accepted_swaps':a['accepted_swaps']} for x in observations for a in x['arms'][1:] if not a['control_changed']],
            'cases':[{k:v for k,v in x.items() if k not in ('files',)} for x in observations],
            'limitations':[
                'No neural dynamics, delayed XOR, task training, confirmatory holdout or model promotion ran.',
                'Exact invariants are not a proof of uniform null sampling, chain mixing or independence.',
                'Anatomical region blocks are not inferred functional modules and no modularity value is preserved.',
                'All comparison arms, including reference, use unit adjacency; measured source multiplicities stay separate.',
                'Weak BFS is topology-biased; rank-only sparse subsets and disconnected frozen ports remain visible.',
                'Every crossing edge is recorded relative to the qualified induced graph; unknown raw endpoints remain outside this source scope.',
                'Process RSS peaks are separate maxima, not simultaneous machine memory or hardware speedup.',
            ],
            'next':'V888-BOOL-0.2b null-control diversity/mixing and port/partition sensitivity; parent 0.2 remains active',
        }
        dump(evidence/'subset_control_qualification.json',report)
        print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2),flush=True)
    except BaseException as exc:
        dump(evidence/'failure.json',{'type':type(exc).__name__,'message':str(exc),'scope':'failed pilot, not a successful qualification'})
        raise

if __name__=='__main__': main()
