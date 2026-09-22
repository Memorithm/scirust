"""Locate exact artifacts across Thor runner accounts, without sudo or chmod of sources.

An account already able to read the qualified subsets may publish a new,
read-only, hash-verified cache on the same host. No source directory permissions
are changed. Unavailable preparation slots are not qualifications; the consumer
must independently verify every pinned receipt and consumed file before work.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import qualify as q


def cache_path(protocol):
    digest = protocol['prior_receipt_sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('invalid cache identity')
    return Path('/tmp') / ('memorithm-v888-bool02-' + digest)


def file_identities(protocol, root):
    identities = {'evidence/subset_control_qualification.json': protocol['prior_receipt_sha256']}
    identities.update({'evidence/' + name + '.json': digest for name, digest in protocol['case_receipts'].items()})
    for relative, expected in identities.items():
        if q.sha(root / relative) != expected:
            raise ValueError('prior receipt mismatch: ' + relative)
    for case in protocol['case_receipts']:
        receipt = json.loads((root / 'evidence' / (case + '.json')).read_text())
        for name in ['nodes.tsv', 'ports.tsv', *(arm + '.edges.tsv' for arm in protocol['arms'])]:
            identities[case + '/' + name] = receipt['files'][name]['sha256']
    return identities


def verify_location(protocol, root):
    identities = file_identities(protocol, root)
    for relative, expected in identities.items():
        if q.sha(root / relative) != expected:
            raise ValueError('prior source mismatch: ' + relative)
    return identities


def prepare(protocol):
    cache = cache_path(protocol)
    if cache.exists():
        identities = verify_location(protocol, cache)
        return dict(ready=True, method='existing_verified_cache', root=str(cache), files=len(identities))
    original = Path(protocol['prior_root'])
    for root in (original, Path.home() / 'v888-research-runs' / original.name):
        try:
            identities = verify_location(protocol, root)
        except (PermissionError, FileNotFoundError):
            continue
        # New cache only: source permissions, existing caches and datasets untouched.
        stage = Path(tempfile.mkdtemp(prefix=cache.name + '-stage-', dir='/tmp'))
        for relative, expected in identities.items():
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
            if q.sha(target) != expected:
                raise ValueError('source changed while caching: ' + relative)
            target.chmod(0o444)
        verify_location(protocol, root)
        for path in sorted(stage.rglob('*'), reverse=True):
            if path.is_dir():
                path.chmod(0o555)
        stage.chmod(0o555)
        try:
            stage.rename(cache)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            # Another authorized producer won the atomic publication. Its bytes
            # are verified independently; our complete duplicate is retained.
        verify_location(protocol, cache)
        return dict(ready=True, method='new_read_only_same_host_cache', root=str(cache),
                    source_root=str(root), files=len(identities), source_permissions_changed=False)
    return dict(ready=False, reason='this_runner_account_cannot_read_prior_artifacts',
                runner=os.environ.get('RUNNER_NAME'), uid=os.getuid())


def resolve(protocol):
    original = Path(protocol['prior_root'])
    for root in (cache_path(protocol), original, Path.home() / 'v888-research-runs' / original.name):
        try:
            identities = verify_location(protocol, root)
            return root, dict(method='verified_location', root=str(root), original_root=str(original),
                              files=len(identities), global_permissions_changed=False)
        except (PermissionError, FileNotFoundError):
            continue
    raise PermissionError('no readable exact artifact location; source ownership must remain unchanged')


class LocationTests(unittest.TestCase):
    def test_invalid_digest_rejected(self):
        for value in ('', '../escape', 'A' * 64):
            with self.assertRaises(ValueError):
                cache_path({'prior_receipt_sha256': value})

    def test_valid_cache_is_content_named(self):
        self.assertEqual(cache_path({'prior_receipt_sha256': 'a' * 64}).name,
                         'memorithm-v888-bool02-' + 'a' * 64)

    def test_corrupt_receipt_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'evidence').mkdir()
            (root / 'evidence/subset_control_qualification.json').write_text('{}')
            with self.assertRaises(ValueError):
                verify_location({'prior_receipt_sha256': 'a' * 64, 'case_receipts': {}}, root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare-cache', action='store_true')
    parser.add_argument('--preparation-output', type=Path)
    args = parser.parse_args()
    original = json.loads((q.HERE / 'protocol.json').read_text())
    if args.prepare_cache:
        result = prepare(original)
        if args.preparation_output:
            q.dump(args.preparation_output, result)
        print(json.dumps(result), flush=True)
        return
    if args.worker is None or args.output is None:
        parser.error('--worker and --output required for qualification')
    q.WORKER = args.worker.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    q.dump(args.output / 'source_protocol.json', original)
    source = args.output / 'source'
    source.mkdir()
    for file in q.HERE.iterdir():
        if file.is_file() and file.suffix in ('.rs', '.py', '.json', '.md'):
            shutil.copy2(file, source / file.name)
    try:
        suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(q.DifferentialTests),
                                    unittest.defaultTestLoader.loadTestsFromTestCase(LocationTests)])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        q.dump(args.output / 'process-tests.json', dict(methods=result.testsRun, failures=len(result.failures), errors=len(result.errors)))
        if not result.wasSuccessful():
            raise RuntimeError('independent process/location tests failed')
        root, location = resolve(original)
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
