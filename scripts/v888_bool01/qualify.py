#!/usr/bin/env python3
"""V888-BOOL-0.1: provenance/Arrow staging around an exact Rust pair reducer.

No new dataset download, no in-place raw filtering, no model-quality claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import struct
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v888_phase1'))
from audit import EXPECTED, dump, identity

SOURCE_NAMES = ('banc_888_meta.feather', 'banc_888_edgelist_simple_v3.feather', 'banc_888_synapses_v3_enriched.parquet')


def exact_uint(column):
    """Never accept float-derived IDs, booleans, nulls or negative integers."""
    if column.null_count or not (
        pa.types.is_integer(column.type) or pa.types.is_string(column.type)
        or pa.types.is_large_string(column.type)
    ):
        raise ValueError('IDs and counts require non-null integer/string storage')
    return pc.cast(column, pa.uint64(), safe=True).to_numpy(zero_copy_only=False)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(8 * 1024 * 1024): value.update(block)
    return value.hexdigest()


def signature(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def readback(path, expected_ids, expected_pairs, expected_weight):
    """Independent Python parser for Rust's little-endian CSR artifact."""
    with path.open('rb') as source:
        h = source.read(24)
    if len(h) != 24 or h[:8] != b'V8CSR001': raise ValueError('invalid CSR header')
    n, m = struct.unpack('<QQ', h[8:])
    if n != len(expected_ids) or m != expected_pairs: raise ValueError('CSR dimensions differ')
    if path.stat().st_size != 32 + 16*n + 12*m: raise ValueError('CSR byte length differs')
    ids = np.memmap(path, mode='r', dtype='<u8', offset=24, shape=(n,))
    offsets = np.memmap(path, mode='r', dtype='<u8', offset=24+8*n, shape=(n+1,))
    if not np.array_equal(ids, expected_ids): raise ValueError('ID mapping differs')
    if offsets[0] != 0 or offsets[-1] != m or np.any(offsets[:-1] > offsets[1:]):
        raise ValueError('invalid row offsets')
    target_offset = 32 + 16*n
    if m:
        targets = np.memmap(path, mode='r', dtype='<u4', offset=target_offset, shape=(m,))
        weights = np.memmap(path, mode='r', dtype='<u8', offset=target_offset+4*m, shape=(m,))
        if np.any(targets >= n) or np.any(weights == 0): raise ValueError('invalid target or weight')
        for row in range(n):
            lo, hi = int(offsets[row]), int(offsets[row+1])
            items = targets[lo:hi]
            if np.any(items == row) or np.any(items[:-1] >= items[1:]):
                raise ValueError('CSR self-loop, duplicate or unsorted row')
        total = sum(int(weights[i:i+65536].sum(dtype=np.uint64)) for i in range(0,m,65536))
    else: total = 0
    if total != expected_weight: raise ValueError('CSR total weight differs')
    return {'nodes': n, 'pairs': m, 'contacts': total, 'bytes': path.stat().st_size,
            'sha256': digest(path), 'independent_readback_passed': True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', type=Path, required=True)
    args = p.parse_args()
    root, out = args.data_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = out/'evidence'; evidence.mkdir()
    start = time.monotonic()
    before = {name: signature(root/name) for name in SOURCE_NAMES}
    sources = {name: identity(root/name) for name in SOURCE_NAMES}
    dump(evidence/'source_identity.json', sources)
    ids = np.sort(exact_uint(feather.read_table(root/SOURCE_NAMES[0], columns=['banc_888_id'])['banc_888_id']))
    if len(ids) != 188508 or len(np.unique(ids)) != len(ids): raise ValueError('unexpected or duplicate nodes')
    node_path = out/'nodes.bin'
    with node_path.open('xb') as f:
        f.write(b'V8NODE01'+struct.pack('<Q',len(ids)))
        f.write(ids.astype('<u8',copy=False).tobytes())
    edge_path = out/'edges.bin'
    with pa.memory_map(str(root/SOURCE_NAMES[1]), 'r') as source:
        reader = ipc.open_file(source)
        edge_rows = sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
        if edge_rows != 13620865: raise ValueError('unexpected edge row count')
        with edge_path.open('xb') as f:
            f.write(b'V8EDGE01'+struct.pack('<Q',edge_rows))
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                block = np.empty((batch.num_rows,3), dtype='<u8')
                for j, name in enumerate(('pre','post','count')): block[:,j] = exact_uint(batch[name])
                f.write(block.tobytes())
    raw = pq.ParquetFile(root/SOURCE_NAMES[2])
    if raw.metadata.num_rows != 198816365: raise ValueError('unexpected raw row count')
    rows = 0
    with (evidence/'rust_stderr.log').open('wb') as err, (evidence/'rust_stdout.log').open('wb') as stdout:
        proc = subprocess.Popen([str(args.worker),str(node_path),str(edge_path),str(out)],
                                stdin=subprocess.PIPE, stdout=stdout, stderr=err)
        try:
            proc.stdin.write(b'V8RAW001'+struct.pack('<Q',raw.metadata.num_rows))
            for batch in raw.iter_batches(batch_size=131072, columns=['pre_root_id','post_root_id']):
                block = np.empty((batch.num_rows,2), dtype='<u8')
                block[:,0] = exact_uint(batch['pre_root_id']); block[:,1] = exact_uint(batch['post_root_id'])
                proc.stdin.write(block.tobytes()); rows += batch.num_rows
                if rows % (131072*100) == 0: print(f'raw rows supplied: {rows}',flush=True)
            proc.stdin.close()
            returncode = proc.wait(timeout=180)
        except BaseException:
            proc.kill(); proc.wait(); raise
    if (out/'pairwise.json').exists():
        (evidence/'pairwise.json').write_bytes((out/'pairwise.json').read_bytes())
    if returncode != 0: raise RuntimeError('Rust pair audit failed; diagnostic retained')
    result = json.loads((out/'pairwise.json').read_text())
    if rows != raw.metadata.num_rows or result['raw_rows'] != rows or not result['pairwise_equality']:
        raise AssertionError('incomplete raw scan or pair disagreement')
    if result['both_known'] != result['reference_weight'] or result['reference_weight'] != 42309621:
        raise AssertionError('induced contact totals differ')
    if sum(result[k] for k in ('both_known','pre_only','post_only','neither_known')) != rows:
        raise AssertionError('non-exhaustive source boundary')
    graph = readback(out/'graph.csr',ids,edge_rows,result['reference_weight'])
    for name in SOURCE_NAMES:
        if signature(root/name) != before[name]: raise ValueError('source modified during qualification')
    protocol = json.loads((Path(__file__).parent/'protocol.json').read_text())
    bound = {'format':'V8CSR001', 'source_sha256':{n:sources[n]['sha256'] for n in SOURCE_NAMES},
             'node_order':'ascending unsigned 64-bit root IDs', 'rows':'presynaptic; targets postsynaptic',
             'filter':'both endpoints in frozen metadata; no further contact-count threshold',
             'edge_weights':'exact v3 contact multiplicity; not conductance',
             'graph_sha256':graph['sha256'], 'node_map_sha256':digest(node_path)}
    graph_identity = hashlib.sha256(json.dumps(bound,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    report = {'milestone':'V888-BOOL-0.1','classification':'source_graph_qualification',
              'source_commit':os.environ.get('GITHUB_SHA'), 'worker_sha256':digest(args.worker),
              'protocol_sha256':hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
              'graph_identity_sha256':graph_identity,'graph_contract':bound,'graph':graph,'pairwise':result,
              'runtime':{'host':platform.node(),'architecture':platform.machine(),
                         'elapsed_seconds':time.monotonic()-start,
                         'rust_max_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                         'python_max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
              'scope_limits':['no synapse-ID uniqueness claim','no physiological signs/delays/thresholds inferred',
                              'no Boolean task, topology advantage, biological growth or hardware speedup claim',
                              'raw sources and derived full graph remain external to Git'],
              'graph_external_path':str(out/'graph.csr')}
    dump(evidence/'graph_qualification.json',report)
    dump(evidence/'protocol.json',protocol)
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__': main()
