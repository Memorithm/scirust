#!/usr/bin/env python3
"""Fail-closed FLAT runtime pin attestation for SciRust M33 release candidates.

Usage: python3 scripts/check-flat-m33-pin.py <40-hex-release-candidate-sha>
Requires Python 3.11+ (tomllib). Does not fetch remote refs or run a GPU.
"""
import argparse
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHA = re.compile(r"[0-9a-f]{40}\Z")
REPO = "git+https://github.com/Memorithm/FLAT-ATTENTION"


def verify(root: pathlib.Path, candidate: str) -> list[str]:
    errors = []
    if not SHA.fullmatch(candidate):
        return ["release candidate must be an explicit full lowercase 40-hex SHA"]
    try:
        manifest = tomllib.loads((root / "scirust-gpu/Cargo.toml").read_text())
        lock = tomllib.loads((root / "Cargo.lock").read_text())
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        return [f"manifest or lockfile unreadable: {exc}"]
    dependencies = manifest.get("dependencies", {})
    if not isinstance(dependencies, dict):
        return ["runtime manifest dependencies must be a TOML table"]
    dep = dependencies.get("flat-attention")
    if not isinstance(dep, dict) or dep.get("git") != REPO.removeprefix("git+") or dep.get("optional") is not True:
        errors.append("runtime FLAT dependency must be an explicit optional Git dependency")
        return errors
    declared = dep.get("rev")
    if not isinstance(declared, str) or not SHA.fullmatch(declared):
        errors.append("runtime manifest FLAT rev is not a full immutable SHA")
    elif declared != candidate:
        errors.append(f"runtime manifest rev {declared} != candidate {candidate}")
    packages = lock.get("package", [])
    if not isinstance(packages, list) or not all(isinstance(p, dict) for p in packages):
        return errors + ["Cargo.lock package entries must be TOML tables"]
    matches = [p for p in packages if p.get("name") == "flat-attention"
               and isinstance(p.get("source"), str) and p["source"].startswith(REPO)]
    expected = f"{REPO}?rev={candidate}#{candidate}"
    if not any(p["source"] == expected for p in matches):
        errors.append("Cargo.lock has no runtime FLAT package with matching rev AND resolved SHA")
    # Reject ambiguous lockfile identities even when one entry matches exactly.
    # A second planner revision is permitted only if its declared rev differs.
    candidate_rev = f"{REPO}?rev={candidate}#"
    if any(p["source"].startswith(candidate_rev) and p["source"] != expected for p in matches):
        errors.append("Cargo.lock has conflicting resolved SHA for runtime FLAT rev")
    if not matches:
        errors.append("Cargo.lock contains no FLAT Git package")
    # A second planner revision is permitted, but must never be mistaken for the runtime.
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", help="immutable 40-hex FLAT release candidate")
    args = parser.parse_args()
    failures = verify(ROOT, args.candidate)
    if failures:
        for failure in failures:
            print("BLOCKED:", failure, file=sys.stderr)
        return 1
    print("PASS: manifest and Cargo.lock runtime FLAT pin match candidate")
    print("SCOPE: static pin only; no remote provenance, GPU or numerical qualification")
    return 0


if __name__ == "__main__":
    sys.exit(main())
