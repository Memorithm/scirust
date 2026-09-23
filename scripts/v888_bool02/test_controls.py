"""Synthetic source/process tests; never opens a research holdout."""
from __future__ import annotations
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from oracle import ARMS, load_source, sha256, verify_control, verify_subset

WORKER=Path(sys.argv.pop(1)).resolve()

def write_source(path, n, pairs):
    ids=[(1<<54)+1+i*17 for i in range(n)]
    triples=sorted(pairs)
    offsets=[0]
    for i in range(n): offsets.append(offsets[-1]+sum(u==i for u,v,w in triples))
    with path.open('wb') as f:
        f.write(struct.pack('<8sQQ',b'V8CSR001',n,len(triples)))
        for x in ids+offsets: f.write(struct.pack('<Q',x))
        for u,v,w in triples: f.write(struct.pack('<I',v))
        for u,v,w in triples: f.write(struct.pack('<Q',w))
    blocks=[i//4 for i in range(n)]
    block_file=path.with_suffix('.blocks.tsv')
    block_file.write_text(''.join(f'{i}\t{g}\n' for i,g in zip(ids,blocks)))
    return block_file,blocks

def invoke(source, blocks, out, count, seed=7, policy='weak_bfs', ok=True):
    result=subprocess.run([str(WORKER),str(source),str(blocks),str(out),str(count),str(seed),policy],capture_output=True,text=True,timeout=60)
    if ok and result.returncode: raise AssertionError(result.stderr)
    if not ok and result.returncode==0: raise AssertionError('invalid input accepted')
    return result

class Controls(unittest.TestCase):
    def test_seeded_multiset_and_selection_oracles(self):
        for seed in range(12):
            rng=random.Random(seed)
            pairs=[(u,v,rng.randint(1,9)) for u,v in rng.sample([(u,v) for u in range(16) for v in range(16) if u!=v],55)]
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); path=root/'input.csr'; blocks,groups=write_source(path,16,pairs)
                source=load_source(path)
                for policy in ('ranked','weak_bfs'):
                    first=root/policy; second=root/(policy+'-replay')
                    invoke(path,blocks,first,9,seed,policy)
                    report=verify_subset(source,groups,first,selection_oracle=True)
                    self.assertEqual(len(report['arms']),5)
                    invoke(path,blocks,second,9,seed,policy)
                    self.assertEqual({f.name:sha256(f) for f in first.iterdir()},{f.name:sha256(f) for f in second.iterdir()})
    def test_complete_graph_is_explicitly_nondiverse(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp); p=r/'x.csr'; b,g=write_source(p,8,[(u,v,2) for u in range(8) for v in range(8) if u!=v])
            invoke(p,b,r/'out',8)
            result=verify_subset(load_source(p),g,r/'out',True)
            self.assertTrue(all(not a['control_changed'] for a in result['arms']))
            self.assertTrue(all(a['accepted_swaps']==0 for a in result['arms']))
    def test_empty_graph_and_isolated_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp); p=r/'x.csr'; b,g=write_source(p,8,[])
            invoke(p,b,r/'out',8)
            result=verify_subset(load_source(p),g,r/'out',True)
            self.assertEqual(result['pairs'],0)
            self.assertTrue(all(a['isolated_nodes']==8 for a in result['arms']))
    def test_malformed_csr_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp); p=r/'x.csr'; b,_=write_source(p,5,[(0,1,1),(2,3,2)])
            original=p.read_bytes()
            invalid=[original[:-1],b'BROKEN!!'+original[8:],original+b'\x00']
            def mutate(offset, fmt, value):
                data=bytearray(original); struct.pack_into(fmt,data,offset,value); return data
            invalid += [mutate(8,'<Q',2**64-1),mutate(32,'<Q',(1<<54)+1),mutate(24+5*8,'<Q',1),mutate(32+16*5,'<I',99),mutate(32+16*5,'<I',0),mutate(len(original)-8,'<Q',0)]
            for i,data in enumerate(invalid):
                p.write_bytes(data); invoke(p,b,r/str(i),3,ok=False)
    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp); p=r/'x.csr'; b,_=write_source(p,5,[(0,1,1)])
            invoke(p,b,r/'out',5); before={f.name:sha256(f) for f in (r/'out').iterdir()}
            invoke(p,b,r/'out',5,ok=False)
            self.assertEqual(before,{f.name:sha256(f) for f in (r/'out').iterdir()})
    def test_cut_record_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp); p=r/'x.csr'; b,g=write_source(p,8,[(u,v,1) for u in range(8) for v in range(8) if u!=v])
            invoke(p,b,r/'out',4)
            cut=r/'out'/'boundary.tsv'; lines=cut.read_text().splitlines(); fields=lines[0].split('\t'); fields[2]='99'; lines[0]='\t'.join(fields); cut.write_text('\n'.join(lines)+'\n')
            with self.assertRaisesRegex(ValueError,'cut edge record'): verify_subset(load_source(p),g,r/'out')
    def test_degree_only_corruption_does_not_pass_reciprocity(self):
        a={(0,1),(1,0),(2,3)}; b={(0,3),(1,0),(2,1)}
        verify_control(4,a,b,[0]*4,'degree')
        with self.assertRaisesRegex(ValueError,'reciprocal'): verify_control(4,a,b,[0]*4,'degree_reciprocal')
    def test_reciprocity_only_corruption_does_not_pass_blocks(self):
        a={(0,1),(2,3)}; b={(0,3),(2,1)}
        verify_control(4,a,b,[0,0,1,1],'degree_reciprocal')
        with self.assertRaisesRegex(ValueError,'block'): verify_control(4,a,b,[0,0,1,1],'degree_reciprocal_block')
    def test_port_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp); p=r/'x.csr'; b,g=write_source(p,6,[(0,1,1),(2,3,1)])
            invoke(p,b,r/'out',6)
            (r/'out'/'ports.tsv').write_text('input_a\t0\ninput_b\t0\nreadout\t0\n')
            with self.assertRaisesRegex(ValueError,'port identity'): verify_subset(load_source(p),g,r/'out')
    def test_bad_block_identity_or_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp); p=r/'x.csr'; b,_=write_source(p,6,[(0,1,1)])
            invoke(p,b,r/'policy',4,policy='other',ok=False)
            invoke(p,b,r/'size',7,ok=False)
            b.write_text('1\t0\n')
            invoke(p,b,r/'blocks',4,ok=False)

if __name__=='__main__': unittest.main()
