"""Independent full-history oracle for Rust DYN-REF1; synthetic inputs only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
import unittest
from pathlib import Path

MODES = ("lif", "integer", "parity", "nonlinear")
PARAMS = {"lif": (0.75, 4.0, 0, 1), "integer": (2, 3, 8, 1), "parity": (0, 0, 0, 0), "nonlinear": (1, 0, 0, 0)}
HERE = Path(__file__).resolve().parent


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj: object) -> str:
    return digest(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def fixture(mode, n, edges, drives, params=None, name="synthetic"):
    return {"name": name, "mode": mode, "n": n, "edges": edges, "drives": drives,
            "params": params if params is not None else PARAMS[mode],
            "max_ticks": 512, "max_visits": 1_000_000, "max_payload": 1_000_000}


def encode(c):
    mode = c["mode"] if c["mode"] in ("lif", "integer") else "boolean"
    values = ["DYN1", mode, c["n"], len(c["edges"]), len(c["drives"]), *c["params"], c["max_ticks"], c["max_visits"], c["max_payload"]]
    values += [v for edge in c["edges"] for v in edge]
    values += [v for drive in c["drives"] for v in drive]
    return " ".join(map(str, values)) + "\n"


def run(binary, cases):
    raw = "".join(encode(c) for c in cases).encode()
    p = subprocess.run([str(binary), "--batch"], input=raw, capture_output=True, timeout=30)
    if p.returncode:
        raise AssertionError(p.stderr.decode(errors="replace"))
    result = [json.loads(line) for line in p.stdout.splitlines()]
    if len(result) != len(cases):
        raise AssertionError("wrong trace count")
    return result, p.stdout


def oracle(c):
    """Target-by-target evaluation using complete prior history, not a ring buffer."""
    n, mode = c["n"], c["mode"]
    p, q, r, refractory = c["params"]
    incoming = [[] for _ in range(n)]
    for src, dst, weight, delay in sorted(c["edges"]):
        incoming[dst].append((src, weight, delay))
    history = []
    integers, voltage, ref = [0] * n, [0.0] * n, [0] * n
    counters = dict(ticks=0, edge_visits=0, active_edge_events=0, spikes=0, clipped_nodes=0, refractory_nodes=0)
    traces = []
    for t, drive in enumerate(c["drives"]):
        spikes, next_i, next_v, next_ref = [0] * n, [0] * n, [0.0] * n, [0] * n
        for dst in range(n):
            delayed = [history[t - delay][src] if t >= delay else 0 for src, _, delay in incoming[dst]]
            counters["active_edge_events"] += sum(delayed)
            if mode in ("parity", "nonlinear"):
                output = drive[dst]
                for bit in delayed:
                    output ^= bit
                if mode == "nonlinear" and len(delayed) >= 2:
                    output ^= delayed[0] & delayed[1]
                spikes[dst] = output
            elif ref[dst]:
                next_ref[dst] = ref[dst] - 1
                counters["refractory_nodes"] += 1
            else:
                # Python arbitrary-precision sum in reverse incoming order.
                impulse = sum(bit * edge[1] for bit, edge in reversed(list(zip(delayed, incoming[dst]))))
                if mode == "integer":
                    old = integers[dst]
                    leaked = (abs(old) // int(p)) * (-1 if old < 0 else 1)
                    raw = leaked + impulse + drive[dst]
                    potential = min(r, max(-r, raw))
                    counters["clipped_nodes"] += int(potential != raw)
                    if potential >= q:
                        spikes[dst], next_ref[dst] = 1, refractory
                    else:
                        next_i[dst] = potential
                else:
                    potential = (p * voltage[dst] + float(impulse)) + float(drive[dst])
                    if potential >= q:
                        spikes[dst], next_ref[dst] = 1, refractory
                    else:
                        next_v[dst] = potential
        counters["ticks"] += 1
        counters["edge_visits"] += len(c["edges"])
        counters["spikes"] += sum(spikes)
        history.append(spikes)
        integers, voltage, ref = next_i, next_v, next_ref
        traces.append(dict(spikes=spikes, integer=integers, voltage=voltage, refractory=ref, counters=dict(counters)))
    return traces


def compare(c, actual):
    expected = oracle(c)
    if len(actual["ticks"]) != len(expected):
        raise AssertionError("tick count mismatch")
    for t, (want, got) in enumerate(zip(expected, actual["ticks"])):
        for key in ("spikes", "integer", "refractory", "counters"):
            if got[key] != want[key]:
                raise AssertionError(f"{c['name']} tick {t} {key}: {got[key]} != {want[key]}")
        if len(got["voltage"]) != c["n"]:
            raise AssertionError("voltage dimension mismatch")
        for x, y in zip(got["voltage"], want["voltage"]):
            if not math.isfinite(x) or not math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError(f"{c['name']} numeric oracle mismatch: {x} != {y}")
    if not actual["reset_zero"] or not actual["parameters_preserved"] or actual["learning_performed"]:
        raise AssertionError("reset/parameter/learning claim mismatch")
    if actual["edge_count"] != len(c["edges"]):
        raise AssertionError("edge count mismatch")


def panel():
    cases = []
    for seed in range(32):
        rng = random.Random(20260922 + seed)
        n = 2 + seed % 11
        pairs = [(a, b) for a in range(n) for b in range(n) if rng.random() < 0.28]
        delays = [rng.randint(1, 7) for _ in pairs]
        weights = [rng.choice((-3, -1, 1, 2, 5)) for _ in pairs]
        binary_drives = [[rng.randrange(2) if rng.random() < 0.4 else 0 for _ in range(n)] for _ in range(64)]
        for mode in MODES:
            numeric = mode in ("lif", "integer")
            edges = [(a, b, weights[i] if numeric else 1, delays[i]) for i, (a, b) in enumerate(pairs)]
            drives = [[3 * v if numeric else v for v in row] for row in binary_drives]
            cases.append(fixture(mode, n, edges, drives, name=f"seed-{seed}-{mode}"))
    for mode in MODES:
        params = (0.5, 1.0, 0, 0) if mode == "lif" else ((2, 1, 20, 0) if mode == "integer" else PARAMS[mode])
        for delay in (1, 2, 7, 64):
            drives = [[1, 0]] + [[0, 0] for _ in range(delay + 3)]
            cases.append(fixture(mode, 2, [(0, 1, 1, delay)], drives, params, f"delay-{delay}-{mode}"))
        cycle = [(0, 1, 1, 1), (1, 0, 1, 1)]
        cases.append(fixture(mode, 2, cycle, [[1, 0]] + [[0, 0] for _ in range(127)], params, f"programmed-cycle-{mode}"))
        cases.append(fixture(mode, 2, cycle, [[0, 0] for _ in range(128)], params, f"quiescent-{mode}"))
    return cases


class ProcessTests(unittest.TestCase):
    binary: Path

    def test_full_history_differential_and_complete_byte_replay(self):
        cases = panel()
        results, raw = run(self.binary, cases)
        for case, actual in zip(cases, results):
            compare(case, actual)
        self.assertEqual(raw, run(self.binary, cases)[1])

    def test_canonical_edge_reordering(self):
        case = panel()[20]
        reordered = dict(case, edges=list(reversed(case["edges"])))
        self.assertEqual(run(self.binary, [case])[1], run(self.binary, [reordered])[1])

    def test_independent_episode_isolation(self):
        a, b = panel()[0], panel()[-1]
        together, _ = run(self.binary, [a, b, a])
        separate, _ = run(self.binary, [a])
        self.assertEqual(together[0], separate[0])
        self.assertEqual(together[2], separate[0])

    def test_programmed_pulse_cycle_and_quiescence(self):
        cases = [c for c in panel() if c["name"].startswith(("programmed-cycle", "quiescent"))]
        for c, result in zip(cases, run(self.binary, cases)[0]):
            for t, state in enumerate(result["ticks"]):
                self.assertEqual(state["spikes"], ([int(t % 2 == 0), int(t % 2 == 1)] if c["name"].startswith("programmed") else [0, 0]))

    def test_analytical_scalar_lif(self):
        c = fixture("lif", 1, [], [[1] for _ in range(32)], (0.5, 10.0, 0, 0))
        result = run(self.binary, [c])[0][0]
        for t, state in enumerate(result["ticks"], 1):
            self.assertEqual(state["voltage"], [2.0 * (1.0 - 2.0 ** -t)])

    def test_integer_negative_leak_and_clipping(self):
        c = fixture("integer", 1, [], [[-3], [0], [0], [-100], [100]], (2, 8, 10, 0))
        result = run(self.binary, [c])[0][0]
        compare(c, result)
        self.assertEqual([t["integer"][0] for t in result["ticks"]], [-3, -1, 0, -10, 0])

    def test_invalid_topology_config_and_budgets(self):
        base = fixture("integer", 2, [(0, 1, 1, 1)], [[1, 0], [0, 0]])
        changes = [dict(edges=[(0, 1, 1, 0)]), dict(edges=[(0, 1, 1, 65)]), dict(edges=[(2, 1, 1, 1)]),
                   dict(edges=[(0, 1, 1, 1), (0, 1, 1, 2)]), dict(edges=[(0, 1, 0, 1)]),
                   dict(params=(0, 1, 8, 0)), dict(params=(1, 8, 7, 0)), dict(max_ticks=1), dict(max_visits=1), dict(max_payload=1)]
        changes += [dict(mode="lif", params=("NaN", 1, 0, 0)), dict(mode="lif", params=(1.5, 1, 0, 0)),
                    dict(mode="nonlinear", params=PARAMS["nonlinear"], drives=[[2, 0], [0, 0]])]
        for change in changes:
            with self.subTest(change=change):
                p = subprocess.run([str(self.binary), "--batch"], input=encode(dict(base, **change)).encode(), capture_output=True, timeout=5)
                self.assertEqual(p.returncode, 2)
                self.assertFalse(p.stdout)
                self.assertIn(b"protocol_error:", p.stderr)

    def test_parser_truncation_extra_fields_and_size_bound(self):
        raw = encode(panel()[0]).encode()
        invalid = [b"wrong\n", raw[:40], raw.rstrip() + b" extra\n", b"\xff\n", b"x" * 131073]
        for data in invalid:
            with self.subTest(length=len(data)):
                p = subprocess.run([str(self.binary), "--batch"], input=data, capture_output=True, timeout=5)
                self.assertEqual(p.returncode, 2)
                self.assertFalse(p.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ProcessTests.binary = args.binary.resolve()
    start = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProcessTests)
    with (args.output / "process-tests.log").open("w") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    print((args.output / "process-tests.log").read_text(), flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)
    cases = panel()
    rows, raw = run(ProcessTests.binary, cases)
    for c, actual in zip(cases, rows):
        compare(c, actual)
    if run(ProcessTests.binary, cases)[1] != raw:
        raise AssertionError("non-deterministic complete replay")
    (args.output / "traces.jsonl").write_bytes(raw)
    (args.output / "fixtures.json").write_text(json.dumps(cases, sort_keys=True) + "\n")
    protocol = json.loads((HERE / "protocol.json").read_text())
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    report = {
        "slice": "DYN-REF1", "scope": "synthetic CPU recurrent equations, not V888 graph/task/learning qualification",
        "source_commit": os.environ.get("GITHUB_SHA", "unavailable"), "worker_sha256": digest(args.binary.read_bytes()),
        "host": platform.node(), "architecture": platform.machine(), "runner": os.environ.get("RUNNER_NAME", "unavailable"),
        "run_id": os.environ.get("GITHUB_RUN_ID", "unavailable"), "protocol_sha256": canonical(protocol),
        "fixtures_sha256": canonical(cases), "trace_sha256": digest(raw),
        "process_test_methods": result.testsRun, "cases": len(cases), "neuron_updates": sum(c["n"] * len(c["drives"]) for c in cases),
        "simulated_ticks": sum(len(c["drives"]) for c in cases), "oracle_mismatches": 0, "byte_replay_identical": True,
        "programmed_cycle_updates": 128, "models": ["discrete_LIF", "bounded_integer", "affine_Boolean_parity", "nonlinear_Boolean_parity_AND"],
        "model_parameters_trained": False, "V888_source_graph_read": False, "game_learning_performed": False,
        "reset_and_immutable_parameter_checks": "passed for every case; no learned parameters exist yet",
        "pilot_seconds_including_tests_and_oracles": time.monotonic() - start,
        "limitations": ["No uniform-null, physiological, topology-advantage, learned-memory or GPU claim.",
                        "Fixed-step scan of every edge; binary firing is not an event-driven performance result.",
                        "Boolean state stored as bytes; numeric reference scratch is retained and accounted.",
                        "Payload counts exclude allocator, caller inputs, traces and OS/process memory.",
                        "0.2b control/partition/port diagnostics and V888 integration remain open.",
                        "No protected final holdout, raw-source mutation, trading or model promotion."],
    }
    (args.output / "qualification.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
