#!/usr/bin/env python3
"""Fail closed when SciRust's documented FLAT consumer graph drifts."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "docs/integration/flat-consumer-matrix-v1.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FLAT consumer matrix mismatch: {message}")


def load_toml(path: str) -> dict:
    with (ROOT / path).open("rb") as handle:
        return tomllib.load(handle)


def verify_dependency(actual: dict | str, expected: dict, integration_id: str) -> None:
    key = expected["key"]
    if "version" in expected:
        version = actual.get("version") if isinstance(actual, dict) else actual
        require(version == expected["version"], f"{integration_id}.{key} version")
        return

    require(isinstance(actual, dict), f"{integration_id}.{key} dependency shape")
    require(
        actual.get("git") == expected["repository"],
        f"{integration_id}.{key} repository",
    )
    require(
        actual.get("rev") == expected["revision"],
        f"{integration_id}.{key} revision",
    )
    require(
        actual.get("package", key) == expected["package"],
        f"{integration_id}.{key} package",
    )
    require(
        actual.get("default-features", True) == expected["default_features"],
        f"{integration_id}.{key} default-features",
    )
    require(
        sorted(actual.get("features", [])) == sorted(expected["features"]),
        f"{integration_id}.{key} features",
    )


def verify_manifests(matrix: dict) -> None:
    require(matrix.get("schema") == "scirust-flat-consumer-matrix/v1", "schema")
    require(matrix.get("msrv") == "1.89.0", "declared matrix MSRV")

    root_manifest = load_toml("Cargo.toml")
    require(
        root_manifest["package"]["rust-version"] == matrix["msrv"].removesuffix(".0"),
        "root MSRV",
    )

    lock_text = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
    for integration in matrix["integrations"]:
        manifest = load_toml(integration["manifest"])
        dependencies = manifest["dependencies"]
        for expected in integration["dependencies"]:
            key = expected["key"]
            require(key in dependencies, f"{integration['id']}.{key} missing")
            verify_dependency(dependencies[key], expected, integration["id"])
            revision = expected.get("revision")
            if revision:
                require(
                    revision in lock_text,
                    f"{integration['id']}.{key} absent from Cargo.lock",
                )


def verify_graph(path: Path, integration: dict) -> None:
    graph = path.read_text(encoding="utf-8")
    for required in integration["active_graph"]["required"]:
        require(required in graph, f"{integration['id']} graph missing {required!r}")
    for forbidden in integration["active_graph"]["forbidden"]:
        require(forbidden not in graph, f"{integration['id']} graph contains {forbidden!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-tree", type=Path)
    parser.add_argument("--contextual-tree", type=Path)
    args = parser.parse_args()

    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    verify_manifests(matrix)

    integrations = {item["id"]: item for item in matrix["integrations"]}
    if args.runtime_tree is not None:
        verify_graph(args.runtime_tree, integrations["flat-runtime-wgpu-v1"])
    if args.contextual_tree is not None:
        verify_graph(
            args.contextual_tree,
            integrations["flat-contextual-host-advisory-v1"],
        )

    print("FLAT consumer matrix matches manifests, lockfile and supplied graphs")


if __name__ == "__main__":
    main()
