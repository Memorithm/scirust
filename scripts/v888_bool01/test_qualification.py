#!/usr/bin/env python3
"""Independent process tests for V888-BOOL-0.1; only tiny synthetic inputs."""
from collections import Counter
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
from qualify import exact_uint, readback

WORKER = Path(sys.argv.pop(1)).resolve()


def pack_nodes(ids):
    return b'V8NODE01'+struct.pack('<Q',len(ids))+b''.join(struct.pack('<Q',x) for x in ids)


def pack_edges(edges):
    return b'V8EDGE01'+struct.pack('<Q',len(edges))+b''.join(struct.pack('<QQQ',*e) for e in edges)


def pack_raw(pairs):
    return b'V8RAW001'+struct.pack('<Q',len(pairs))+b''.join(struct.pack('<QQ',*p) for p in pairs)


class QualificationTests(unittest.TestCase):
    def execute(self, ids, edges, pairs, *, raw_override=None):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'nodes').write_bytes(pack_nodes(ids)); (root/'edges').write_bytes(pack_edges(edges))
            p=subprocess.run([str(WORKER),str(root/'nodes'),str(root/'edges'),str(root)],
                input=pack_raw(pairs) if raw_override is None else raw_override,
                capture_output=True, timeout=30)
            summary=json.loads((root/'pairwise.json').read_text()) if (root/'pairwise.json').exists() else None
            graph=(root/'graph.csr').read_bytes() if (root/'graph.csr').exists() else None
            if p.returncode == 0:
                parsed=readback(root/'graph.csr',np.array(ids,dtype=np.uint64),len(edges),sum(e[2] for e in edges))
                self.assertTrue(parsed['independent_readback_passed'])
            return p,summary,graph

    def test_random_multisets_match_counter_and_replay(self):
        for seed in range(12):
            rng=random.Random(seed)
            ids=[(1<<54)+1+2*i for i in range(7)]
            pairs=[tuple(rng.sample(ids,2)) for _ in range(100)]
            expected=Counter(pairs)
            edges=[(p,q,w) for (p,q),w in sorted(expected.items())]
            pairs.extend([(ids[0],99),(99,ids[1]),(98,99)])
            p,s,g=self.execute(ids,edges,pairs)
            self.assertEqual(p.returncode,0,p.stderr.decode()); self.assertTrue(s['pairwise_equality'])
            self.assertEqual([s[k] for k in ('both_known','pre_only','post_only','neither_known')],[100,1,1,1])
            rng.shuffle(edges); rng.shuffle(pairs)
            p2,s2,g2=self.execute(ids,edges,pairs)
            self.assertEqual(p2.returncode,0,p2.stderr.decode()); self.assertEqual(s,s2); self.assertEqual(g,g2)

    def test_pair_swap_with_identical_node_totals_rejected(self):
        p,s,g=self.execute([1,2,3,4],[(1,3,1),(2,4,1)],[(1,4),(2,3)])
        self.assertNotEqual(p.returncode,0); self.assertIsNone(g)
        self.assertEqual(s['mismatching_reference_pairs'],2)
        self.assertEqual(s['unexpected_known_pair_records'],2)

    def test_multiplicity_mismatch_rejected(self):
        for pairs in ([],[(1,2)],[(1,2)]*3):
            with self.subTest(pairs=pairs):
                p,s,g=self.execute([1,2],[(1,2,2)],pairs)
                self.assertNotEqual(p.returncode,0); self.assertIsNone(g)
                self.assertEqual(s['mismatching_reference_pairs'],1)

    def test_duplicate_unknown_or_zero_reference_rejected(self):
        for edges in ([(1,2,1),(1,2,1)],[(1,3,1)],[(1,2,0)],[(1,1,1)]):
            with self.subTest(edges=edges):
                p,_,g=self.execute([1,2],edges,[(1,2)])
                self.assertNotEqual(p.returncode,0); self.assertIsNone(g)

    def test_truncated_or_trailing_input_rejected(self):
        raw=pack_raw([(1,2)])
        for bad in (raw[:-1],raw+b'x',b'notmagic'+raw[8:]):
            with self.subTest(raw=bad):
                p,_,g=self.execute([1,2],[(1,2,1)],[],raw_override=bad)
                self.assertNotEqual(p.returncode,0); self.assertIsNone(g)

    def test_integer_decoding_never_accepts_float_ids(self):
        big=(1<<54)+1
        self.assertEqual(exact_uint(pa.array([str(big),str(big+1)])).tolist(),[big,big+1])
        for values in ([1.0,2.0],[True,False],[1,None],[-1,2],['1.5','2']):
            with self.subTest(values=values):
                with self.assertRaises((ValueError,pa.ArrowException)): exact_uint(pa.array(values))

    def test_empty_graph_preserves_isolated_nodes(self):
        p,s,g=self.execute([1,2,3],[],[])
        self.assertEqual(p.returncode,0,p.stderr.decode())
        self.assertEqual(s['nodes'],3); self.assertEqual(len(g),80)

    def test_csr_independent_reader_rejects_corruption(self):
        p,_,g=self.execute([1,2],[(1,2,1)],[(1,2)])
        self.assertEqual(p.returncode,0,p.stderr.decode())
        for bad in (g[:-1],g+b'x',b'badmagic'+g[8:]):
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'graph'; path.write_bytes(bad)
                with self.assertRaises(ValueError): readback(path,np.array([1,2],dtype=np.uint64),1,1)

if __name__=='__main__': unittest.main()
