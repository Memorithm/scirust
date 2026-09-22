"""Independent PORTS-DYN1 checks. Rust executes dynamics; Python verifies evidence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import tempfile
import time
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
MASK64 = (1 << 64) - 1
WORKER: Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def mix(x):
    x = (x + 0x9e3779b97f4a7c15) & MASK64
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & MASK64
    return x ^ (x >> 31)


def matrix(path, fields):
    result = []
    for line in Path(path).read_text().splitlines():
        parts = line.split('\t')
        if len(parts) != fields or any(not x or not x.isascii() or not x.isdecimal() for x in parts):
            raise ValueError('invalid integer input table')
        result.append(tuple(int(x) for x in parts))
    return result


def distance_layers(adjacency, root):
    # Set-frontier breadth layers, independently of the Rust FIFO implementation.
    depth = [-1] * len(adjacency)
    frontier = {root}
    level = 0
    while frontier:
        for node in frontier:
            depth[node] = level
        candidates = set().union(*(adjacency[node] for node in frontier))
        frontier = {node for node in candidates if depth[node] < 0}
        level += 1
    return depth


def expected_ports(n, pairs, panel):
    adjacency = [set() for _ in range(n)]
    reverse = [set() for _ in range(n)]
    for u, v in pairs:
        adjacency[u].add(v)
        reverse[v].add(u)
    result = []
    for index, (a, b, readout) in enumerate(panel):
        da = distance_layers(adjacency, a)
        db = distance_layers(adjacency, b)
        dr = distance_layers(reverse, readout)
        result.append(dict(kind='ports', index=index, a=a, b=b, readout=readout,
                           a_distance=da[readout], b_distance=db[readout],
                           a_reachable=sum(x >= 0 for x in da), b_reachable=sum(x >= 0 for x in db),
                           common_reachable=sum(x >= 0 and y >= 0 for x, y in zip(da, db)),
                           joint_route_nodes=sum(x >= 0 and y >= 0 and z >= 0 for x, y, z in zip(da, db, dr))))
    return result


def pack_hex(bits):
    padded = np.pad(bits, (0, (-len(bits)) % 4))
    values = padded.reshape(-1, 4).astype(np.uint8) @ np.array([1, 2, 4, 8], dtype=np.uint8)
    return ''.join(format(int(x), 'x') for x in values)


def verify(raw, n, pairs, panel, ticks):
    records = iter(json.loads(line) for line in raw.splitlines())
    if next(records) != dict(kind='graph', nodes=n, edges=len(pairs), ticks_per_trial=ticks):
        raise ValueError('graph declaration mismatch')
    port_results = expected_ports(n, pairs, panel)
    for expected in port_results:
        if next(records) != expected:
            raise ValueError('directed port diagnostic mismatch')
    pre = np.array([u for u, _ in pairs], dtype=np.int64)
    post = np.array([v for _, v in pairs], dtype=np.int64)
    incoming = [[] for _ in range(n)]
    for u, v in sorted(pairs):
        incoming[v].append(u)
    first = np.array([row[0] if len(row) >= 2 else 0 for row in incoming], dtype=np.int64)
    second = np.array([row[1] if len(row) >= 2 else 0 for row in incoming], dtype=np.int64)
    has_pair = np.array([len(row) >= 2 for row in incoming], dtype=np.uint8)
    a, b, readout = panel[0]
    summaries = []
    paths = {}
    all_states = {}
    for nonlinear in (False, True):
        for mask in range(4):
            previous = np.zeros(n, dtype=np.uint8)
            support = previous.copy()
            active = 0
            spike_count = 0
            states = []
            supported = []
            cancellations = []
            readout_ones = []
            last_active = None
            for t in range(ticks):
                # A 0/1 sum over <=4095 predecessors is exact in binary64.
                # This gather/reduction is independent of Rust's scatter/XOR loop.
                sums = np.bincount(post, weights=previous[pre], minlength=n).astype(np.int64)
                active += int(previous[pre].sum())
                state = (sums & 1).astype(np.uint8)
                if nonlinear:
                    state ^= previous[first] & previous[second] & has_pair
                next_support = (np.bincount(post, weights=support[pre], minlength=n) > 0).astype(np.uint8)
                if t == 0:
                    state[a] ^= mask & 1
                    state[b] ^= (mask >> 1) & 1
                    next_support[a] |= mask & 1
                    next_support[b] |= (mask >> 1) & 1
                expected = dict(kind='state', nonlinear=nonlinear, mask=mask, t=t, hex=pack_hex(state))
                if next(records) != expected:
                    raise ValueError(f'state mismatch mode={nonlinear} mask={mask} tick={t}')
                if mask == 0 and state.any():
                    raise ValueError('spontaneous zero-input activity')
                if state.any():
                    last_active = t
                if state[readout]:
                    readout_ones.append(t)
                if next_support[readout]:
                    supported.append(t)
                if not nonlinear and next_support[readout] and not state[readout]:
                    cancellations.append(t)
                if np.any(state & (1 - next_support)):
                    raise ValueError('activity outside causal support')
                spike_count += int(state.sum())
                states.append(state)
                previous, support = state, next_support
            observed = next(records)
            expected = dict(kind='trial', nonlinear=nonlinear, mask=mask, ticks=ticks,
                            edge_visits=len(pairs) * ticks, active_edge_events=active, spikes=spike_count,
                            reset_zero=True, reset_probe_ticks=1, learning_performed=False,
                            direct_vector_payload_bytes=observed.get('direct_vector_payload_bytes'))
            if observed != expected or not isinstance(observed['direct_vector_payload_bytes'], int) or observed['direct_vector_payload_bytes'] <= 0:
                raise ValueError('counter/reset declaration mismatch')
            all_states[nonlinear, mask] = np.stack(states)
            if not nonlinear:
                paths[mask] = supported
            summaries.append(dict(nonlinear=nonlinear, mask=mask, readout_one_ticks=readout_ones,
                                  structurally_supported_ticks=supported, affine_cancellation_ticks=cancellations,
                                  last_active_tick=last_active, **{k: observed[k] for k in ('ticks', 'edge_visits', 'active_edge_events', 'spikes', 'direct_vector_payload_bytes')}))
    if next(records, None) is not None:
        raise ValueError('unexpected trailing output record')
    nonlinear_ticks = {}
    for mode in (False, True):
        delta = all_states[mode, 3] ^ all_states[mode, 1] ^ all_states[mode, 2]
        changed = np.flatnonzero(np.any(delta, axis=1)).tolist()
        if not mode and changed:
            raise ValueError('affine model violates superposition')
        nonlinear_ticks[str(mode).lower()] = dict(any_node=changed, readout=np.flatnonzero(delta[:, readout]).tolist())
    return dict(ports=port_results, trials=summaries, superposition_violation_ticks=nonlinear_ticks,
                oracle_state_comparisons=8 * ticks, simulation_edge_visits=8 * ticks * len(pairs),
                reset_probe_edge_visits=8 * len(pairs), mismatches=0, learning_performed=False)


def invoke(nodes_path, edges_path, panel_path, ticks):
    return subprocess.run([str(WORKER), str(nodes_path), str(edges_path), str(panel_path), str(ticks)],
                          capture_output=True, timeout=180, check=False)


def fixture(directory, n, pairs, panel, ticks):
    node_path = directory / 'nodes.tsv'
    edge_path = directory / 'edges.tsv'
    port_path = directory / 'ports.tsv'
    node_path.write_text(''.join(f'{i}\t{2**54 + i * 7}\t0\n' for i in range(n)))
    edge_path.write_text(''.join(f'{u}\t{v}\n' for u, v in sorted(pairs)))
    port_path.write_text(''.join('\t'.join(map(str, row)) + '\n' for row in panel))
    result = invoke(node_path, edge_path, port_path, ticks)
    if result.returncode:
        raise AssertionError(result.stderr.decode())
    return verify(result.stdout, n, sorted(pairs), panel, ticks), result.stdout


class DifferentialTests(unittest.TestCase):
    def check_graph(self, n, pairs, panel, ticks=17):
        with tempfile.TemporaryDirectory() as folder:
            return fixture(Path(folder), n, pairs, panel, ticks)

    def test_empty_graph(self):
        result, _ = self.check_graph(5, [], [(0, 1, 2)])
        self.assertEqual(result['ports'][0]['a_distance'], -1)
        self.assertTrue(all(not t['readout_one_ticks'] for t in result['trials']))

    def test_direction_and_shortest_path(self):
        result, _ = self.check_graph(4, [(0, 1), (1, 2)], [(0, 3, 2), (2, 3, 0)])
        self.assertEqual(result['ports'][0]['a_distance'], 2)
        self.assertEqual(result['ports'][1]['a_distance'], -1)

    def test_diamond_cancellation(self):
        result, _ = self.check_graph(5, [(0, 1), (0, 2), (1, 3), (2, 3)], [(0, 4, 3)])
        affine = next(t for t in result['trials'] if not t['nonlinear'] and t['mask'] == 1)
        self.assertEqual(affine['structurally_supported_ticks'], [2])
        self.assertEqual(affine['affine_cancellation_ticks'], [2])
        self.assertEqual(affine['readout_one_ticks'], [])

    def test_nonlinear_interaction(self):
        result, _ = self.check_graph(3, [(0, 2), (1, 2)], [(0, 1, 2)])
        self.assertEqual(result['superposition_violation_ticks']['false']['any_node'], [])
        self.assertEqual(result['superposition_violation_ticks']['true']['readout'], [1])

    def test_cycle_retains_impulse(self):
        result, _ = self.check_graph(4, [(0, 1), (1, 0)], [(0, 2, 1)], 129)
        affine = next(t for t in result['trials'] if not t['nonlinear'] and t['mask'] == 1)
        self.assertEqual(affine['last_active_tick'], 128)
        self.assertEqual(affine['readout_one_ticks'], list(range(1, 129, 2)))

    def test_random_graphs_and_replay(self):
        rng = random.Random(17029)
        for n in (3, 7, 16, 25):
            for density in (0.05, 0.3, 0.8):
                pairs = [(u, v) for u in range(n) for v in range(n) if u != v and rng.random() < density]
                panel = [(0, 1, n - 1), (n - 1, 0, 1)]
                with tempfile.TemporaryDirectory() as folder:
                    _, first = fixture(Path(folder), n, pairs, panel, 17)
                    _, second = fixture(Path(folder), n, pairs, panel, 17)
                    self.assertEqual(first, second)

    def test_malformed_input_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root, 3, [(0, 2), (1, 2)], [(0, 1, 2)], 3)
            paths = [root / x for x in ('nodes.tsv', 'edges.tsv', 'ports.tsv')]
            original = [p.read_bytes() for p in paths]
            invalid = [(0, '0\t1.0\t0\n'), (0, '0\t1\t0\n1\t1\t0\n2\t3\t0\n'),
                       (1, '0\t2\n0\t2\n'), (1, '0\t3\n'), (1, '0\t0\n'),
                       (1, '1\t2\n0\t2\n'), (2, '0\t0\t2\n'), (2, ''), (2, '0\t1\t3\n')]
            for index, content in invalid:
                paths[index].write_text(content)
                p = invoke(*paths, 3)
                self.assertNotEqual(p.returncode, 0)
                self.assertEqual(p.stdout, b'')
                paths[index].write_bytes(original[index])
            for horizon in (0, 258):
                self.assertNotEqual(invoke(*paths, horizon).returncode, 0)

    def test_oracle_rejects_corrupted_and_truncated_trace(self):
        result, raw = self.check_graph(3, [(0, 2), (1, 2)], [(0, 1, 2)])
        del result
        with self.assertRaises((ValueError, StopIteration)):
            verify(raw.replace(b'"mask":1', b'"mask":8', 1), 3, [(0, 2), (1, 2)], [(0, 1, 2)], 17)
        with self.assertRaises((ValueError, StopIteration)):
            verify(b'\n'.join(raw.splitlines()[:-1]), 3, [(0, 2), (1, 2)], [(0, 1, 2)], 17)


def run_data(output, protocol):
    prior = Path(protocol['prior_root'])
    receipt_path = prior / 'evidence/subset_control_qualification.json'
    checked = {}

    def check(path, expected):
        observed = sha(path)
        if observed != expected:
            raise ValueError('source identity mismatch: ' + str(path))
        checked[path] = observed

    check(receipt_path, protocol['prior_receipt_sha256'])
    prior_receipt = json.loads(receipt_path.read_text())
    if not prior_receipt['all_declared_invariants_passed_independent_oracle'] or not prior_receipt['all_replays_identical']:
        raise ValueError('prior qualification not accepted')
    cases = []
    started = time.monotonic()
    for case, expected_digest in sorted(protocol['case_receipts'].items()):
        receipt = prior / 'evidence' / (case + '.json')
        check(receipt, expected_digest)
        evidence = json.loads(receipt.read_text())
        directory = prior / case
        for name in ('nodes.tsv', 'ports.tsv', *(arm + '.edges.tsv' for arm in protocol['arms'])):
            check(directory / name, evidence['files'][name]['sha256'])
        node_rows = matrix(directory / 'nodes.tsv', 3)
        n = len(node_rows)
        if not 3 <= n <= 4096 or [x[0] for x in node_rows] != list(range(n)):
            raise ValueError('node identity order mismatch')
        ids = [x[1] for x in node_rows]
        if ids != sorted(set(ids)) or any(x > MASK64 for x in ids):
            raise ValueError('invalid exact source IDs')
        port_rows = [line.split('\t') for line in (directory / 'ports.tsv').read_text().splitlines()]
        if [x[0] for x in port_rows] != ['input_a', 'input_b', 'readout']:
            raise ValueError('original port names mismatch')
        panel = [tuple(int(x[1]) for x in port_rows)]
        seed = evidence['seed']
        for index in range(1, protocol['additional_port_panels'] + 1):
            rank = sorted(range(n), key=lambda i: (mix(ids[i] ^ seed ^ ((index * 0x9e3779b97f4a7c15) & MASK64)), ids[i]))
            panel.append(tuple(rank[:3]))
        panel_path = output / (case + '.panel.tsv')
        panel_path.write_text(''.join('\t'.join(map(str, row)) + '\n' for row in panel))
        for arm in protocol['arms']:
            path = directory / (arm + '.edges.tsv')
            pairs = matrix(path, 2)
            if pairs != sorted(set(pairs)) or any(u >= n or v >= n or u == v for u, v in pairs):
                raise ValueError('invalid or noncanonical directed pairs')
            first = invoke(directory / 'nodes.tsv', path, panel_path, protocol['ticks'])
            if first.returncode:
                raise RuntimeError(first.stderr.decode())
            measured = verify(first.stdout, n, pairs, panel, protocol['ticks'])
            second = invoke(directory / 'nodes.tsv', path, panel_path, protocol['ticks'])
            if second.returncode or second.stdout != first.stdout:
                raise ValueError('non-identical replay')
            measured.update(case=case, arm=arm, nodes=n, edges=len(pairs), replay_identical=True,
                            graph_sha256=sha(path), nodes_sha256=sha(directory / 'nodes.tsv'),
                            original_ports_sha256=sha(directory / 'ports.tsv'), panel_sha256=sha(panel_path),
                            trace_sha256=hashlib.sha256(first.stdout).hexdigest())
            trace = output / (case + '.' + arm + '.trace.jsonl.gz')
            trace.write_bytes(gzip.compress(first.stdout, mtime=0))
            dump(output / (case + '.' + arm + '.json'), measured)
            cases.append(measured)
            print(json.dumps(dict(case=case, arm=arm, nodes=n, edges=len(pairs),
                                  both_original_inputs_reach=all(measured['ports'][0][x] >= 0 for x in ('a_distance', 'b_distance')),
                                  differential_passed=True, replay_identical=True)), flush=True)
    for path, digest in checked.items():
        if sha(path) != digest:
            raise ValueError('input changed during execution: ' + str(path))
    summary = dict(schema='v888-ports-dyn1-result', source_commit=os.environ.get('GITHUB_SHA'),
                   protocol_sha256=canonical(protocol), core_sha256=protocol['core_sha256'],
                   worker_sha256=sha(WORKER), graphs=len(cases), trials=len(cases) * 8,
                   port_configurations=sum(len(c['ports']) for c in cases),
                   state_comparisons=sum(c['oracle_state_comparisons'] for c in cases),
                   simulation_edge_visits=sum(c['simulation_edge_visits'] for c in cases),
                   reset_probe_edge_visits=sum(c['reset_probe_edge_visits'] for c in cases),
                   all_replays_identical=True, mismatches=0, learning_performed=False,
                   input_files_rehashed=len(checked), source_rehashed_after=True,
                   host=platform.node(), architecture=platform.machine(), runner=os.environ.get('RUNNER_NAME'),
                   run_id=os.environ.get('GITHUB_RUN_ID'), elapsed_seconds=time.monotonic() - started,
                   counter_scope='one pass; replay doubles Rust work; reset probes counted separately',
                   source_identity={str(k.relative_to(prior)): v for k, v in checked.items()},
                   cases=cases, limitations=protocol['scope_boundaries'])
    dump(output / 'qualification.json', summary)
    return summary


def main():
    global WORKER
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selftest-only', action='store_true')
    args = parser.parse_args()
    WORKER = args.worker.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    dump(args.output / 'protocol.json', protocol)
    source = args.output / 'source'
    source.mkdir()
    for file in HERE.iterdir():
        if file.is_file() and file.suffix in ('.rs', '.py', '.json', '.md'):
            shutil.copy2(file, source / file.name)
    try:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(DifferentialTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        dump(args.output / 'process-tests.json', dict(methods=result.testsRun, failures=len(result.failures), errors=len(result.errors)))
        if not result.wasSuccessful():
            raise RuntimeError('independent process tests failed')
        if not args.selftest_only:
            run_data(args.output, protocol)
    except BaseException as exc:
        dump(args.output / 'failure.json', dict(type=type(exc).__name__, message=str(exc), qualified=False))
        raise


if __name__ == '__main__':
    main()
