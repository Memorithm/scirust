#!/usr/bin/env python3
"""Read every v3 synapse endpoint, aggregate in Rust, reconcile with v3 edges.

Python owns Arrow decoding/provenance and report assembly, not endpoint reduction.
No developmental, causal, or raw-record uniqueness claim is made.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import urllib.request

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

BASE = 'https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/'
EXPECTED = {
    'banc_888_meta.feather': '86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098',
    'banc_888_metrics.feather': '3ee647d402510b6b1c1bc3b0249db83f799ba94a1094c8ec530fcb09fac1ce60',
    'banc_888_edgelist_simple_v3.feather': '8c296e946f3c69a8c7222f30ad75fa8a98eeb189124fec6df829c9125f4be64b',
    'banc_888_synapses_v3_enriched.parquet': '0dfb5cf89ba156d076beab2da38d87eaa63dcbe45d76f86b108570fb5b961dd0',
}

def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')

def identity(path: Path) -> dict:
    before = path.stat()
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open('rb') as source:
        while block := source.read(8 * 1024 * 1024):
            sha.update(block); md5.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('source changed while hashing: ' + path.name)
    if sha.hexdigest() != EXPECTED[path.name]:
        raise ValueError('source differs from frozen phase-0 SHA256: ' + path.name)
    info = {'bytes': after.st_size, 'sha256': sha.hexdigest(), 'md5_base64': base64.b64encode(md5.digest()).decode(), 'upstream_url': BASE + path.name}
    with urllib.request.urlopen(urllib.request.Request(info['upstream_url'], method='HEAD'), timeout=60) as response:
        if int(response.headers['Content-Length']) != after.st_size:
            raise ValueError('upstream size changed: ' + path.name)
        hashes = ','.join(response.headers.get_all('x-goog-hash') or [])
        remote = dict(part.strip().split('=', 1) for part in hashes.split(',') if '=' in part)
        if 'md5' in remote and remote['md5'] != info['md5_base64']:
            raise ValueError('upstream MD5 mismatch: ' + path.name)
        info.update(upstream_generation=response.headers.get('x-goog-generation'), upstream_hashes=remote, upstream_md5_verified='md5' in remote)
    return info

def uint_values(column: pa.Array) -> np.ndarray:
    if column.null_count:
        raise ValueError('null endpoint or weight in source')
    return pc.cast(column, pa.uint64(), safe=True).to_numpy(zero_copy_only=False)

def consume(batches, names, worker, nodes, prefix, extras=False):
    regions = Counter(); rows = 0; below_ten = 0; size_min = None
    with prefix.with_suffix('.stderr.log').open('wb') as err:
        proc = subprocess.Popen([str(worker), str(nodes), str(prefix)], stdin=subprocess.PIPE, stderr=err)
        try:
            assert proc.stdin is not None
            proc.stdin.write(b'V888CNT1')
            for batch in batches:
                pre = uint_values(batch.column(batch.schema.get_field_index(names[0])))
                post = uint_values(batch.column(batch.schema.get_field_index(names[1])))
                weights = np.ones(len(pre), dtype=np.uint64) if names[2] is None else uint_values(batch.column(batch.schema.get_field_index(names[2])))
                records = np.empty((len(pre), 3), dtype='<u8')
                records[:,0] = pre; records[:,1] = post; records[:,2] = weights
                proc.stdin.write(records.tobytes())
                rows += len(pre)
                if extras:
                    size = batch.column(batch.schema.get_field_index('size'))
                    if size.null_count: raise ValueError('null synapse size')
                    minimum = pc.min(size).as_py()
                    if minimum is not None: size_min = minimum if size_min is None else min(size_min, minimum)
                    below_ten += int(pc.sum(pc.cast(pc.less(size, 10), pa.int64())).as_py() or 0)
                    for item in pc.value_counts(batch.column(batch.schema.get_field_index('region'))).to_pylist():
                        regions[str(item['values']) if item['values'] is not None else '__missing__'] += item['counts']
                if rows % (131072 * 100) == 0: print(f'{prefix.name}: parsed {rows} rows', flush=True)
            proc.stdin.close()
            if proc.wait(timeout=120) != 0: raise RuntimeError('Rust reducer failed; see stderr log')
        except BaseException:
            proc.kill(); proc.wait(); raise
    summary = json.loads(Path(str(prefix) + '.summary.json').read_text())
    if summary['records'] != rows: raise AssertionError('stream row count mismatch')
    summary.update(decoded_rows=rows, region_counts=dict(sorted(regions.items())), size_min=size_min, size_below_10=below_ten)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = args.output / 'evidence'; evidence.mkdir()
    start = time.monotonic()
    identities = {name: identity(args.data_root / name) for name in EXPECTED}
    dump(evidence / 'source_identity.json', identities)
    meta = feather.read_table(args.data_root / 'banc_888_meta.feather').to_pandas()
    meta['_root'] = pd.to_numeric(meta['banc_888_id'], errors='raise').astype('uint64')
    if meta['_root'].duplicated().any(): raise ValueError('duplicate metadata neuron IDs')
    meta = meta.sort_values('_root').reset_index(drop=True)
    columns = ['region', 'side', 'hemilineage'] + (['neuromere'] if 'neuromere' in meta.columns else [])
    for column in columns: meta[column] = meta[column].fillna('').astype(str).str.strip()
    keys = list(zip(*(meta[column] for column in columns)))
    labeled = [bool(label) and label.lower() != 'nan' for label in meta['hemilineage']]
    unique = sorted({key for key, keep in zip(keys, labeled, strict=True) if keep})
    lookup = {key: i + 1 for i, key in enumerate(unique)}
    meta['_group'] = [lookup[key] if keep else 0 for key, keep in zip(keys, labeled, strict=True)]
    group_names = pd.DataFrame(unique, columns=columns)
    group_names.insert(0, 'group_id', range(1, len(unique) + 1))
    group_names.to_csv(evidence / 'annotation_groups.csv', index=False)
    nodes = args.output / 'node_index.tsv'
    meta[['_root','_group']].to_csv(nodes, sep='\t', index=False, header=False)
    syn_path = args.data_root / 'banc_888_synapses_v3_enriched.parquet'
    source = pq.ParquetFile(syn_path)
    syn = consume(source.iter_batches(batch_size=131072, columns=['pre_root_id','post_root_id','size','region']), ('pre_root_id','post_root_id',None), args.worker, nodes, args.output / 'synapses', True)
    if syn['records'] != source.metadata.num_rows: raise AssertionError('incomplete parquet scan')
    with pa.memory_map(str(args.data_root / 'banc_888_edgelist_simple_v3.feather'), 'r') as mapped:
        edges = ipc.open_file(mapped)
        edge = consume((edges.get_batch(i) for i in range(edges.num_record_batches)), ('pre','post','count'), args.worker, nodes, args.output / 'edges')
    syn_nodes = pd.read_csv(args.output / 'synapses.nodes.csv', dtype='uint64')
    edge_nodes = pd.read_csv(args.output / 'edges.nodes.csv', dtype='uint64')
    if not syn_nodes['root_id'].equals(edge_nodes['root_id']): raise AssertionError('node order mismatch')
    mismatch = int((syn_nodes[['incoming_v3','outgoing_v3']] != edge_nodes[['incoming_v3','outgoing_v3']]).any(axis=1).sum())
    joined = meta.merge(syn_nodes, left_on='_root', right_on='root_id', validate='one_to_one')
    metrics = feather.read_table(args.data_root / 'banc_888_metrics.feather').to_pandas()
    metrics['_root'] = pd.to_numeric(metrics['banc_888_id'], errors='raise').astype('uint64')
    measurements = ['l2_cable_length_um','volume_nm3','mitochondria','mitochondria_volume']
    joined = joined.drop(columns=[c for c in measurements if c in joined]).merge(metrics[['_root', *measurements]], on='_root', how='left', validate='one_to_one')
    grouped = joined[joined['_group'] > 0].groupby('_group', sort=True)
    table = grouped.size().rename('n_neurons').to_frame()
    for col in ['incoming_v3','outgoing_v3',*measurements]:
        table[col + '__valid_count'] = grouped[col].count()
        table[col + '__sum'] = grouped[col].sum(min_count=1)
        table[col + '__mean'] = grouped[col].mean()
    table.index.name='group_id'
    group_names.merge(table.reset_index(), on='group_id', validate='one_to_one').to_csv(evidence / 'annotation_group_observations.csv', index=False)
    for kind in ('synapses','edges'):
        (evidence / (kind + '_group_mixing.csv')).write_bytes((args.output / (kind + '.mixing.csv')).read_bytes())
    report = {
        'dataset':'BANC', 'materialization':'v888', 'detector':'v3',
        'source_sha256': {k:v['sha256'] for k,v in identities.items()},
        'neuron_metadata_rows':len(meta), 'neurons_with_nonempty_hemilineage_label':sum(labeled),
        'annotation_groups':len(unique), 'annotation_group_key':columns,
        'group_caveat':'literal annotation tuples; not certified developmental lineages; ambiguous labels retained',
        'synapses':syn, 'edges':edge,
        'reconciliation':{'same_weight':syn['weight']==edge['weight'], 'nodes_with_different_in_or_out_total':mismatch},
        'elapsed_seconds':time.monotonic()-start, 'host':platform.node(),
        'limitations':[
            'single adult specimen; no time course, growth rate or causal developmental inference',
            'metadata input_connections/output_connections (v2) are not used for v3 totals',
            'read every endpoint and size/region row; no full 31-column value audit or synapse-ID uniqueness check',
            'compares node totals, not pairwise edge identity; autapses and unknown endpoints are explicitly counted',
            'missing morphological metrics stay missing; zero-valued metrics are not automatically interpreted as absence',
            'no CPU/GPU performance or model-quality promotion',
        ],
    }
    for name, before in identities.items():
        if (args.data_root / name).stat().st_size != before['bytes']: raise ValueError('source size changed')
    dump(evidence / 'full_synapse_audit.json', report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)

if __name__ == '__main__': main()
