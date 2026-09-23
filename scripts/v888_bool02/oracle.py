"""Independent NumPy/Python reader and invariant oracle, not a graph generator."""
from __future__ import annotations
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

ARMS = ('reference', 'edge_count', 'degree', 'degree_reciprocal', 'degree_reciprocal_block')
MASK = (1 << 64) - 1

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def mix(x):
    x = (x + 0x9e3779b97f4a7c15) & MASK
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & MASK
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & MASK
    return x ^ (x >> 31)

def load_source(path):
    with Path(path).open('rb') as f: magic, n, m = struct.unpack('<8sQQ', f.read(24))
    if magic != b'V8CSR001' or not 0 < n <= 1_000_000 or m > 20_000_000: raise ValueError('invalid source dimensions/magic')
    if Path(path).stat().st_size != 32 + 16*n + 12*m: raise ValueError('invalid source byte length')
    ids = np.memmap(path, mode='r', dtype='<u8', offset=24, shape=(n,))
    offsets = np.memmap(path, mode='r', dtype='<u8', offset=24+8*n, shape=(n+1,))
    target_offset = 32+16*n
    targets = np.memmap(path, mode='r', dtype='<u4', offset=target_offset, shape=(m,)) if m else np.array([],dtype=np.uint32)
    weights = np.memmap(path, mode='r', dtype='<u8', offset=target_offset+4*m, shape=(m,)) if m else np.array([],dtype=np.uint64)
    if np.any(ids[:-1] >= ids[1:]) or offsets[0] != 0 or offsets[-1] != m or np.any(offsets[:-1] > offsets[1:]): raise ValueError('invalid source order')
    if np.any(targets >= n) or np.any(weights == 0): raise ValueError('invalid source edge')
    sources = np.repeat(np.arange(n, dtype=np.uint32), np.diff(offsets).astype(np.int64))
    if np.any(sources == targets): raise ValueError('source self edge')
    if m > 1 and np.any((sources[:-1] == sources[1:]) & (targets[:-1] >= targets[1:])): raise ValueError('source duplicate/unsorted pair')
    return ids, offsets, sources, targets, weights

def matrix(path, columns):
    if Path(path).stat().st_size == 0: return np.empty((0, columns), dtype=np.uint64)
    a = np.loadtxt(path, delimiter='\t', dtype=np.uint64, ndmin=2)
    if a.shape[1] != columns: raise ValueError('wrong table column count')
    return a

def edges(path, n):
    a = matrix(path, 2)
    result = [(int(u), int(v)) for u,v in a]
    if any(u >= n or v >= n or u == v for u,v in result): raise ValueError('invalid control edge')
    if result != sorted(set(result)): raise ValueError('duplicate/noncanonical control edge')
    return set(result)

def statistics(n, graph, blocks):
    incoming = [0]*n; outgoing = [0]*n; reciprocal = [0]*n; mixing = Counter()
    for u,v in graph:
        outgoing[u] += 1; incoming[v] += 1
        reciprocal[u] += int((v,u) in graph)
        mixing[int(blocks[u]),int(blocks[v])] += 1
    return incoming,outgoing,reciprocal,mixing

def verify_control(n, reference, candidate, blocks, arm):
    if arm not in ARMS: raise ValueError('undeclared control arm')
    if len(reference) != len(candidate): raise ValueError('edge count mismatch')
    a,b = statistics(n,reference,blocks),statistics(n,candidate,blocks)
    mismatch = {
        'degree_mismatch_nodes':sum(a[0][i]!=b[0][i] or a[1][i]!=b[1][i] for i in range(n)),
        'reciprocal_mismatch_nodes':sum(a[2][i]!=b[2][i] for i in range(n)),
        'block_mismatch_cells':sum(a[3][k]!=b[3][k] for k in a[3].keys() | b[3].keys()),
        'replaced_edges':len(reference-candidate),
    }
    if arm in ('degree','degree_reciprocal','degree_reciprocal_block') and mismatch['degree_mismatch_nodes']: raise ValueError('degree mismatch')
    if arm in ('degree_reciprocal','degree_reciprocal_block') and mismatch['reciprocal_mismatch_nodes']: raise ValueError('reciprocal mismatch')
    if arm == 'degree_reciprocal_block' and mismatch['block_mismatch_cells']: raise ValueError('block mismatch')
    return mismatch

def reachable(n, graph, start):
    adj = [[] for _ in range(n)]
    for u,v in graph: adj[u].append(v)
    seen={start}; todo=[start]
    for u in todo:
        for v in adj[u]:
            if v not in seen: seen.add(v); todo.append(v)
    return seen

def expected_selection_small(source, count, seed, policy):
    ids,_,pre,post,_ = source
    ranked=sorted(range(len(ids)),key=lambda i:(mix(int(ids[i])^seed),int(ids[i])))
    if policy == 'ranked': return sorted(ranked[:count])
    adj=[set() for _ in ids]
    for a,b in zip(pre,post): adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
    seen=set(); queue=deque(); picked=[]
    while len(picked)<count:
        if not queue:
            root=next(i for i in ranked if i not in seen); seen.add(root); queue.append(root)
        u=queue.popleft(); picked.append(u)
        for v in sorted(adj[u],key=lambda i:(mix(int(ids[i])^seed),int(ids[i]))):
            if v not in seen: seen.add(v); queue.append(v)
    return sorted(picked)

def verify_subset(source, global_blocks, directory, selection_oracle=False):
    directory=Path(directory); ids,offsets,pre,post,weights=source
    summary=json.loads((directory/'summary.json').read_text())
    nodes=matrix(directory/'nodes.tsv',3); n=summary['nodes']
    if len(nodes)!=n or not np.array_equal(nodes[:,0],np.arange(n,dtype=np.uint64)): raise ValueError('local node order mismatch')
    positions=np.searchsorted(ids,nodes[:,1])
    if np.any(positions>=len(ids)) or not np.array_equal(ids[positions],nodes[:,1]) or np.any(positions[:-1]>=positions[1:]): raise ValueError('source node identity mismatch')
    if not np.array_equal(nodes[:,2],np.asarray(global_blocks)[positions]): raise ValueError('block identity mismatch')
    if selection_oracle and list(positions)!=expected_selection_small(source,n,summary['seed'],summary['selection']): raise ValueError('selection policy mismatch')
    member=np.zeros(len(ids),dtype=np.uint8); member[positions]=1
    category=2*member[pre]+member[post]
    expected_parts=[{'pairs':int(np.count_nonzero(category==i)), 'contacts':int(weights[category==i].sum(dtype=np.uint64))} for i in range(4)]
    if expected_parts!=summary['partitions_outside_incoming_outgoing_internal']: raise ValueError('boundary partition mismatch')
    internal=category==3; cut=(category==1)|(category==2)
    mapping=np.zeros(len(ids),dtype=np.uint64); mapping[positions]=np.arange(n,dtype=np.uint64)
    expected_pairs=np.column_stack((mapping[pre[internal]],mapping[post[internal]],weights[internal]))
    if not np.array_equal(expected_pairs,matrix(directory/'source_pairs.tsv',3)): raise ValueError('measured source pair mismatch')
    expected_cut=np.column_stack((ids[pre[cut]],ids[post[cut]],weights[cut],category[cut].astype(np.uint64)))
    if not np.array_equal(expected_cut,matrix(directory/'boundary.tsv',4)): raise ValueError('cut edge record mismatch')
    reference=edges(directory/'reference.edges.tsv',n)
    if reference!={(int(u),int(v)) for u,v,_ in expected_pairs}: raise ValueError('reference topology mismatch')
    if len(reference)!=summary['pairs']: raise ValueError('reference count mismatch')
    port_lines=[line.split('\t') for line in (directory/'ports.tsv').read_text().splitlines()]
    expected_ports=sorted(range(n),key=lambda i:(mix(int(nodes[i,1])^summary['seed']^0x1234abcd),i))[:3]
    if port_lines!=[[name,str(i)] for name,i in zip(('input_a','input_b','readout'),expected_ports)]: raise ValueError('port identity mismatch')
    observations=[]
    for i,arm in enumerate(ARMS):
        candidate=edges(directory/(arm+'.edges.tsv'),n)
        measured=verify_control(n,reference,candidate,nodes[:,2],arm)
        if i:
            declared=summary['arms'][i-1]
            if declared['arm']!=arm or any(declared[k]!=v for k,v in measured.items()): raise ValueError('worker/oracle diagnostic mismatch')
        else: declared={'attempts':0,'accepted_swaps':0}
        a,b,recip,_=statistics(n,candidate,nodes[:,2])
        reach=[reachable(n,candidate,p) for p in expected_ports[:2]]
        observations.append({
            'arm':arm,**measured,'attempts':declared['attempts'],'accepted_swaps':declared['accepted_swaps'],
            'reciprocal_directed_edges':sum(recip),'isolated_nodes':sum(x+y==0 for x,y in zip(a,b)),
            'input_a_reaches_readout':expected_ports[2] in reach[0],
            'input_b_reaches_readout':expected_ports[2] in reach[1],
            'control_changed':bool(measured['replaced_edges']),
            'uniform_null_ensemble_qualified':False,
        })
    return {
        'selection':summary['selection'],'seed':summary['seed'],'nodes':n,'pairs':len(reference),
        'bfs_component_starts':summary['bfs_component_starts'],'block_labels':summary['block_labels'],
        'partitions':expected_parts,'arms':observations,
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(directory.iterdir()) if p.is_file()},
        'boundary_scope':'cut edges relative to the qualified metadata-induced graph; raw unknown-endpoint boundary remains separate',
    }
