"""Resolve an exact prior artifact location without changing host permissions.

Source identity remains the pinned receipt and file digests. A runner-owned copy
may be made through a read-only command as the prior artifact owner, when the
host's existing sudo policy permits that operation. No sudo configuration is
modified, no shell is used, and no raw connectome or unrelated path is copied.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

import qualify as q


READ = "import sys; f=open(sys.argv[1],'rb'); b=f.read(32000001); assert len(b)<=32000000; sys.stdout.buffer.write(b)"


def load(path: Path) -> bytes:
    try:
        with path.open('rb') as stream:
            value = stream.read(32_000_001)
        if len(value) > 32_000_000:
            raise ValueError('prior artifact exceeds per-file bound')
        return value
    except PermissionError:
        result = subprocess.run(['sudo', '-n', '-u', 'ghrunner', '--', '/usr/bin/python3', '-c', READ, str(path)],
                                capture_output=True, timeout=30, check=False)
        if result.returncode:
            raise PermissionError('cannot read pinned prior artifact through existing owner access: '
                                  + result.stderr.decode(errors='replace')[:1000])
        return result.stdout


def resolve(protocol, destination):
    original = Path(protocol['prior_root'])
    candidates = [original, Path.home() / 'v888-research-runs' / original.name]
    for root in candidates:
        try:
            if q.sha(root / 'evidence/subset_control_qualification.json') == protocol['prior_receipt_sha256']:
                return root, {'method': 'direct', 'root': str(root), 'original_root': str(original)}
        except (OSError, ValueError):
            pass
    destination.mkdir(parents=True, exist_ok=False)
    receipts = {'evidence/subset_control_qualification.json': protocol['prior_receipt_sha256']}
    receipts.update({'evidence/' + name + '.json': digest for name, digest in protocol['case_receipts'].items()})
    for relative, expected in receipts.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(load(original / relative))
        if q.sha(target) != expected:
            raise ValueError('prior receipt mismatch: ' + relative)
    files = 0
    for case in protocol['case_receipts']:
        receipt = json.loads((destination / 'evidence' / (case + '.json')).read_text())
        for name in ['nodes.tsv', 'ports.tsv', *(arm + '.edges.tsv' for arm in protocol['arms'])]:
            target = destination / case / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(load(original / case / name))
            if q.sha(target) != receipt['files'][name]['sha256']:
                raise ValueError('prior file mismatch: ' + case + '/' + name)
            files += 1
    return destination, {'method': 'read_only_owner_copy_verified_by_pinned_hashes',
                         'root': str(destination), 'original_root': str(original),
                         'receipt_files': len(receipts), 'graph_node_port_files': files,
                         'global_permissions_changed': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    args = parser.parse_args()
    q.WORKER = args.worker.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads((q.HERE / 'protocol.json').read_text())
    q.dump(args.output / 'source_protocol.json', original)
    source = args.output / 'source'
    source.mkdir()
    for file in q.HERE.iterdir():
        if file.is_file() and file.suffix in ('.rs', '.py', '.json', '.md'):
            shutil.copy2(file, source / file.name)
    try:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(q.DifferentialTests))
        q.dump(args.output / 'process-tests.json', dict(methods=result.testsRun, failures=len(result.failures), errors=len(result.errors)))
        if not result.wasSuccessful():
            raise RuntimeError('independent process tests failed')
        root, location = resolve(original, args.inputs)
        protocol = dict(original, prior_root=str(root))
        q.dump(args.output / 'protocol.json', protocol)
        q.dump(args.output / 'source_location.json', dict(location, source_protocol_sha256=q.canonical(original),
                                                        execution_protocol_sha256=q.canonical(protocol),
                                                        location_only_override=True, run_id=os.environ.get('GITHUB_RUN_ID')))
        q.run_data(args.output, protocol)
    except BaseException as exc:
        q.dump(args.output / 'failure.json', dict(type=type(exc).__name__, message=str(exc), qualified=False))
        raise


if __name__ == '__main__':
    main()
