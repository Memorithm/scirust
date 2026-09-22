"""Independent small Python oracle for the actual compiled Rust process."""
from collections import Counter
import csv
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

WORKER = Path(sys.argv.pop()).resolve()

class EndpointProcessTests(unittest.TestCase):
    def test_weighted_stream_against_independent_counter(self):
        ids = [10,20,(1<<54)+1]
        records = [(10,20,2),(20,10,3),(10,10,1),(99,20,4),((1<<54)+1,10,7)]
        incoming=Counter(); outgoing=Counter()
        for a,b,w in records: incoming[b]+=w; outgoing[a]+=w
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); nodes=root/'nodes.tsv'; prefix=root/'result'
            nodes.write_text(''.join(f'{x}\t{1 if x!=20 else 2}\n' for x in ids))
            payload=b'V888CNT1'+b''.join(struct.pack('<QQQ',*row) for row in records)
            subprocess.run([str(WORKER),str(nodes),str(prefix)],input=payload,check=True,capture_output=True)
            rows=list(csv.DictReader((root/'result.nodes.csv').open()))
            self.assertEqual([int(r['incoming_v3']) for r in rows],[incoming[x] for x in ids])
            self.assertEqual([int(r['outgoing_v3']) for r in rows],[outgoing[x] for x in ids])
            summary=json.loads((root/'result.summary.json').read_text())
            self.assertEqual(summary['weight'],17)
            self.assertEqual(summary['unknown_pre_weight'],4)
            self.assertEqual(summary['same_annotation_group_weight'],8)
    def test_truncated_stream_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); nodes=root/'nodes.tsv'; nodes.write_text('1\t0\n2\t0\n')
            result=subprocess.run([str(WORKER),str(nodes),str(root/'result')],input=b'V888CNT1x',capture_output=True)
            self.assertNotEqual(result.returncode,0)
    def test_actual_arrow_decode_and_counts(self):
        import importlib.util
        import pyarrow as pa
        spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit.py'))
        audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); nodes=root/'nodes.tsv'; nodes.write_text('10\t1\n20\t2\n')
            batch=pa.record_batch({'pre_root_id':['10','20'],'post_root_id':['20','10'],'size':[10,12],'region':['brain',None]})
            out=audit.consume([batch],('pre_root_id','post_root_id',None),WORKER,nodes,root/'out',True)
            self.assertEqual(out['records'],2)
            self.assertEqual(out['size_min'],10)
            self.assertEqual(out['region_counts'],{'__missing__':1,'brain':1})
            with self.assertRaises(ValueError): audit.uint_values(pa.array(['10',None]))

if __name__=='__main__': unittest.main()
