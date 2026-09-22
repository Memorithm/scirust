#!/usr/bin/env python3
"""Reconcile v3 raw synapses and v3 edgelist on an explicit induced node set.

This is a source-scope audit, not a developmental model. Arrow membership is
checked independently against Python sets; weighted endpoint reduction is Rust.
No metadata v2 synapse counts are used.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from audit import EXPECTED, consume, dump, identity, uint_values


def classify(batch, ids):
    pre = uint_values(batch.column(batch.schema.get_field_index('pre_root_id')))
    post = uint_values(batch.column(batch.schema.get_field_index('post_root_id')))
    known_pre = pc.is_in(pa.array(pre), value_set=ids)
    known_post = pc.is_in(pa.array(post), value_set=ids)
    both = pc.and_(known_pre, known_post)
    count = lambda mask: int(pc.sum(pc.cast(mask, pa.uint64())).as_py() or 0)
    categories = {
        'both_known': count(both),
        'pre_only_known': count(pc.and_(known_pre, pc.invert(known_post))),
        'post_only_known': count(pc.and_(pc.invert(known_pre), known_post)),
        'neither_known': count(pc.and_(pc.invert(known_pre), pc.invert(known_post))),
    }
    if sum(categories.values()) != batch.num_rows: raise AssertionError('membership partition is not exhaustive')
    return batch.filter(both), categories


def self_test():
    big = (1 << 54) + 1
    ids = [10, 20, big]
    pairs = [(10,20),(big,10),(99,10),(20,99),(99,98)]
    batch = pa.record_batch({'pre_root_id':[str(a) for a,b in pairs], 'post_root_id':[str(b) for a,b in pairs]})
    filtered, categories = classify(batch, pa.array(ids, type=pa.uint64()))
    actual = list(zip(filtered['pre_root_id'].to_pylist(), filtered['post_root_id'].to_pylist()))
    expected = [(str(a),str(b)) for a,b in pairs if a in set(ids) and b in set(ids)]
    assert actual == expected
    assert categories == {'both_known':2,'pre_only_known':1,'post_only_known':1,'neither_known':1}
    empty, categories = classify(batch.slice(0,0), pa.array(ids, type=pa.uint64()))
    assert empty.num_rows == 0 and sum(categories.values()) == 0
    print('Arrow membership independently checked, including exact IDs above 2^53', flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', type=Path)
    args=parser.parse_args()
    self_test()
    if args.self_test: return
    if not all((args.data_root,args.output,args.worker)): parser.error('data-root, output and worker required')
    start=time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    evidence=args.output/'evidence'; evidence.mkdir()
    frozen_stats={name:((args.data_root/name).stat().st_size,(args.data_root/name).stat().st_mtime_ns) for name in EXPECTED}
    identities={name:identity(args.data_root/name) for name in EXPECTED}
    dump(evidence/'source_identity.json',identities)
    meta=feather.read_table(args.data_root/'banc_888_meta.feather',columns=['banc_888_id'])
    root_ids=pc.cast(meta['banc_888_id'],pa.uint64(),safe=True)
    if root_ids.null_count: raise ValueError('null metadata root ID')
    ids=np.sort(root_ids.to_numpy())
    if len(np.unique(ids)) != len(ids): raise ValueError('duplicate metadata roots')
    node_file=args.output/'nodes.tsv'
    with node_file.open('w') as f:
        for root in ids: f.write(f'{int(root)}\t0\n')
    value_set=pa.array(ids,type=pa.uint64())
    partition={'both_known':0,'pre_only_known':0,'post_only_known':0,'neither_known':0}
    raw_rows=0
    synapse_source=pq.ParquetFile(args.data_root/'banc_888_synapses_v3_enriched.parquet')
    def batches():
        nonlocal raw_rows
        for batch in synapse_source.iter_batches(batch_size=131072,columns=['pre_root_id','post_root_id']):
            filtered, counts=classify(batch,value_set)
            for key,value in counts.items(): partition[key]+=value
            raw_rows+=batch.num_rows
            if raw_rows % (131072*100) == 0: print(f'raw rows scanned: {raw_rows}',flush=True)
            if filtered.num_rows: yield filtered
    syn=consume(batches(),('pre_root_id','post_root_id',None),args.worker,node_file,args.output/'induced_synapses')
    if raw_rows != synapse_source.metadata.num_rows: raise AssertionError('incomplete raw scan')
    with pa.memory_map(str(args.data_root/'banc_888_edgelist_simple_v3.feather'),'r') as mapped:
        source=ipc.open_file(mapped)
        edge=consume((source.get_batch(i) for i in range(source.num_record_batches)),('pre','post','count'),args.worker,node_file,args.output/'edges')
    a=pd.read_csv(args.output/'induced_synapses.nodes.csv',dtype='uint64')
    b=pd.read_csv(args.output/'edges.nodes.csv',dtype='uint64')
    if not a['root_id'].equals(b['root_id']): raise AssertionError('node order mismatch')
    mismatches=int((a[['incoming_v3','outgoing_v3']] != b[['incoming_v3','outgoing_v3']]).any(axis=1).sum())
    consistent=(mismatches==0 and syn['weight']==edge['weight']==partition['both_known'] and edge['unknown_pre_weight']==edge['unknown_post_weight']==0)
    for name,before in frozen_stats.items():
        after=(args.data_root/name).stat()
        if (after.st_size,after.st_mtime_ns) != before: raise ValueError('source changed during audit')
    report={
        'dataset':'BANC','materialization':'v888','detector':'v3',
        'source_sha256':{name:item['sha256'] for name,item in identities.items()},
        'raw_records_scanned':raw_rows,'metadata_node_count':len(ids),
        'partition_by_metadata_membership':partition,
        'induced_synapses':syn,'edgelist':edge,
        'node_total_mismatches':mismatches,'node_total_reconciliation_passed':consistent,
        'node_set_policy':'unique banc_888_id values from frozen metadata; keep a synapse iff BOTH endpoints belong to this set',
        'host':platform.node(),'elapsed_seconds':time.monotonic()-start,
        'limitations':[
            'not a pair-by-pair or synapse-ID uniqueness certificate; exact weighted node totals only',
            'absent-from-metadata does not establish biological identity or reconstruction quality of an endpoint',
            'no developmental time-course, growth mechanism, causal claim or model-quality promotion',
            'full raw synapse file is retained; excluded rows are excluded only from this induced-graph comparison',
        ],
    }
    dump(evidence/'induced_graph_reconciliation.json',report)
    print(json.dumps(report,indent=2),flush=True)
    if not consistent: raise AssertionError('induced graph node totals do not reconcile; diagnostic evidence retained')

if __name__=='__main__': main()
