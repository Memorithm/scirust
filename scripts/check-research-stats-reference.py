#!/usr/bin/env python3
"""Differential qualification against SciPy 1.18.1 and SALib 1.5.2.

Requires a built scirust-stats research_stats example and the explicitly pinned
optional Python environment. Only synthetic, data-free functions are evaluated.
This is numerical interoperability evidence, not a universal coverage proof.
"""
import argparse
import importlib.metadata
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import bootstrap
from SALib.analyze import morris as morris_analyze, sobol as sobol_analyze
from SALib.sample import morris as morris_sample, sobol as sobol_sample
from SALib.test_functions import Ishigami


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    versions = {name: importlib.metadata.version(name) for name in ("scipy", "SALib", "numpy")}
    if versions["scipy"] != "1.18.1" or versions["SALib"] != "1.5.2":
        raise ValueError("reference versions differ from the qualified environment")
    worker = args.worker.resolve(strict=True)
    def call(operation, **kwargs):
        raw = json.dumps(dict(schema=1, operation=operation, **kwargs), allow_nan=False).encode()
        result = subprocess.run([str(worker)], input=raw, capture_output=True, timeout=20, check=True)
        value = json.loads(result.stdout)
        assert value["schema"] == 1 and value["status"] == "computed"
        return value["result"]

    # Finite empirical population: independently enumerate every resampled mean.
    contrasts = np.array([0.0, 1.0, 3.0, 7.0])
    enumeration = np.array([np.mean(row) for row in itertools.product(contrasts, repeat=4)])
    exact = np.quantile(enumeration, [0.025, 0.975], method="linear")
    reference = bootstrap((contrasts,), np.mean, method="percentile", n_resamples=100_000,
                          confidence_level=0.95, rng=np.random.default_rng(731))
    ci = call("paired_mean_percentile", contrasts=contrasts.tolist(), resamples=100_000, confidence=0.95, seed="731")
    # Discrete empirical endpoints are 0.25 apart. This fixture's 2.5% tails
    # lie strictly inside atoms, so both independent MC implementations must
    # recover the enumerated endpoint; no tolerance is fitted to observations.
    np.testing.assert_allclose([ci["lower"], ci["upper"]], exact, rtol=0, atol=0)
    np.testing.assert_allclose(reference.confidence_interval, exact, rtol=0, atol=0)
    assert ci["estimate"] == np.mean(contrasts)
    assert call("paired_mean_percentile", contrasts=contrasts.tolist(), resamples=100_000, confidence=0.95, seed="731") == ci
    np.testing.assert_array_equal(call("holm", p_values=[0.01, 0.04, 0.03])["adjusted"], [0.03, 0.06, 0.06])

    problem = {"num_vars": 3, "names": ["x", "y", "z"], "bounds": [[0, 1]] * 3}
    x = morris_sample.sample(problem, N=16, num_levels=4, seed=731)
    y = x[:, 0] + 2 * x[:, 1] ** 2 + x[:, 0] * x[:, 2]
    ref = morris_analyze.analyze(problem, x, y, num_levels=4, num_resamples=100, seed=731)
    actual = call("morris", inputs=x.tolist(), outputs=y.tolist())
    for key in ("mu", "mu_star", "sigma"):
        np.testing.assert_allclose(actual[key], ref[key], rtol=5e-12, atol=5e-12)

    for problem, function, analytical in (
        (problem, lambda x: x @ np.array([1., 2., 3.]), [1/14, 4/14, 9/14]),
        ({"num_vars": 3, "names": ["x", "y", "z"], "bounds": [[-np.pi, np.pi]] * 3}, Ishigami.evaluate, None),
    ):
        x = sobol_sample.sample(problem, N=1024, calc_second_order=False, seed=731)
        y = function(x)
        ref = sobol_analyze.analyze(problem, y, calc_second_order=False, num_resamples=100, seed=731)
        layout = y.reshape(-1, 5)
        actual = call("sobol", a=layout[:, 0].tolist(), b=layout[:, -1].tolist(), ab=layout[:, 1:-1].T.tolist())
        np.testing.assert_allclose(actual["first"], ref["S1"], rtol=5e-12, atol=5e-12)
        np.testing.assert_allclose(actual["total"], ref["ST"], rtol=5e-12, atol=5e-12)
        if analytical is not None:
            np.testing.assert_allclose(actual["first"], analytical, rtol=0, atol=0.015)
            np.testing.assert_allclose(actual["total"], analytical, rtol=0, atol=0.015)

    for raw in (b'{"schema":1,"operation":"holm","p_values":[0.1],"p_values":[0.9]}',
                b'{"schema":1,"operation":"holm","p_values":[1e999]}',
                b'{"schema":2,"operation":"holm","p_values":[0.1]}',
                b' ' * (1024 * 1024 + 1)):
        result = subprocess.run([str(worker)], input=raw, capture_output=True, timeout=20)
        assert result.returncode == 21
        assert json.loads(result.stdout)["status"] == "rejected"
    print(json.dumps({"schema": 1, "status": "passed", "references": versions,
                      "scope": "synthetic numerical differential fixtures; no universal statistical coverage or hardware claim"}))


if __name__ == "__main__":
    main()
