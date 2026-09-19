#!/usr/bin/env python3
"""Process-level TPE overhead benchmark for released Optuna and Rustuna.

Each invocation runs one engine/configuration and emits one CSV record. Launch
each repeat in a fresh process to keep Linux VmHWM interpretable.
"""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

LOW = -5.0
HIGH = 5.0


def proc_status_kib(field: str) -> int:
    try:
        text = Path("/proc/self/status").read_text()
    except OSError:
        return 0
    for line in text.splitlines():
        name, sep, rest = line.partition(":")
        if sep and name == field:
            fields = rest.split()
            return int(fields[0]) if fields else 0
    return 0


def target(index: int) -> float:
    return ((index % 5) - 2.0) * 0.5


def objective(values: list[float]) -> float:
    return sum((value - target(i)) ** 2 for i, value in enumerate(values))


def version_of(module: object) -> str:
    version = getattr(module, "__version__", None)
    if version is not None:
        return str(version)
    metadata = importlib.import_module("importlib.metadata")
    return str(metadata.version(module.__name__))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("optuna", "rustuna"), required=True)
    parser.add_argument("--dims", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--header", action="store_true")
    args = parser.parse_args()

    if args.header:
        print(
            "engine,version,dims,trials,seed,proposal_ns,tell_ns,total_ns,"
            "proposal_ns_per_trial,tell_ns_per_trial,total_ns_per_trial,"
            "rss_start_kib,rss_end_kib,peak_rss_kib,best"
        )
        return
    if args.dims <= 0 or args.trials <= 0:
        raise SystemExit("dims and trials must be positive")

    module = importlib.import_module(args.engine)
    if args.engine == "optuna":
        module.logging.set_verbosity(module.logging.WARNING)

    sampler = module.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=10,
        multivariate=False,
    )
    study = module.create_study(direction="minimize", sampler=sampler)

    rss_start_kib = proc_status_kib("VmRSS")
    proposal_ns = 0
    tell_ns = 0
    best = float("inf")
    total_start = time.perf_counter_ns()

    for _ in range(args.trials):
        proposal_start = time.perf_counter_ns()
        trial = study.ask()
        values = [
            trial.suggest_float(f"x{i}", LOW, HIGH)
            for i in range(args.dims)
        ]
        proposal_ns += time.perf_counter_ns() - proposal_start

        value = objective(values)
        best = min(best, value)

        tell_start = time.perf_counter_ns()
        # Rustuna 0.1.0 requires a trial number. Optuna accepts the same form.
        study.tell(trial.number, value)
        tell_ns += time.perf_counter_ns() - tell_start

    total_ns = time.perf_counter_ns() - total_start
    rss_end_kib = proc_status_kib("VmRSS")
    peak_rss_kib = proc_status_kib("VmHWM")

    print(
        f"{args.engine},{version_of(module)},{args.dims},{args.trials},{args.seed},"
        f"{proposal_ns},{tell_ns},{total_ns},"
        f"{proposal_ns / args.trials:.3f},{tell_ns / args.trials:.3f},"
        f"{total_ns / args.trials:.3f},{rss_start_kib},{rss_end_kib},"
        f"{peak_rss_kib},{best:.17e}"
    )


if __name__ == "__main__":
    main()
