#!/usr/bin/env python3
"""Audit adjacent fenced Rust examples from the existing API lexicon index.

This is a conservative source inventory, not a Rust parser or doctest runner.
Only closed, nonempty, ordinary Rust fences count as runnable candidates.
Rustdoc/cargo test remain authoritative for compilation and execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

# Keep the declaration surface aligned with scripts/api-lexicon.py.
PUBLIC_FN_RE = re.compile(
    r'^\s*pub\s+(?:(?:const|async|unsafe)\s+)*'
    r'(?:extern(?:\s+"[^"]*")?\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\b'
)
DOC_RE = re.compile(r"^\s*///(?!/) ?(.*)$")
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
KINDS = (
    "runnable_candidates", "compile_only", "compile_fail", "ignored",
    "should_panic", "other_fences", "empty_fences", "unclosed_fences",
)
RUST_TAGS = {
    "rust", "no_run", "compile_fail", "ignore", "should_panic",
    "edition2015", "edition2018", "edition2021", "edition2024",
    "test_harness", "standalone_crate",
}


def read_json(path: Path) -> dict:
    """Read a JSON object; reject arrays and scalar roots with a useful error."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def source_path(root: Path, relative: str) -> Path:
    """Resolve a source without permitting absolute paths or directory escapes."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("source must be a nonempty repository-relative POSIX path")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".rs":
        raise ValueError(f"invalid Rust source path: {relative}")
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"source escapes the repository or is not a file: {relative}")
    return resolved


def adjacent_docs(lines: list[str], line: int) -> list[str]:
    """Read adjacent /// lines, skipping blanks and single-line attributes only.

    Multi-line attributes, block docs and #[doc = ...] are deliberately not
    inferred. Missing observations on those constructs are not proof of absent
    documentation. Source-level declaration limitations are inherited upstream.
    """
    docs: list[str] = []
    for previous in reversed(lines[:line - 1]):
        match = DOC_RE.match(previous)
        if match:
            docs.append(match.group(1))
        elif not previous.strip():
            continue
        elif re.fullmatch(r"\s*#\[.*\]\s*", previous):
            continue
        else:
            break
    return list(reversed(docs))


def fence_kind(info: str) -> str:
    """Classify a fence conservatively; unknown language/tags never pass a gate."""
    tags = {tag for tag in re.split(r"[\s,]+", info.strip()) if tag}
    if any(tag not in RUST_TAGS and not tag.startswith("ignore-") for tag in tags):
        return "other_fences"
    if "ignore" in tags or any(tag.startswith("ignore-") for tag in tags):
        return "ignored"
    if "compile_fail" in tags:
        return "compile_fail"
    if "no_run" in tags:
        return "compile_only"
    if "should_panic" in tags:
        return "should_panic"
    return "runnable_candidates"


def count_examples(docs: list[str]) -> dict[str, int]:
    """Count closed Markdown fences, not arbitrary backtick occurrences."""
    counts = dict.fromkeys(KINDS, 0)
    opening: tuple[str, int, str] | None = None
    body: list[str] = []
    for line in docs:
        match = FENCE_RE.match(line)
        if opening is None:
            if match:
                marker, info = match.groups()
                # Markdown forbids backticks in a backtick fence's info string.
                if marker[0] == "`" and "`" in info:
                    continue
                opening = (marker[0], len(marker), fence_kind(info))
                body = []
            continue
        if match:
            marker, info = match.groups()
            if marker[0] == opening[0] and len(marker) >= opening[1] and not info.strip():
                nonempty = any(part.strip() for part in body)
                counts[opening[2] if nonempty else "empty_fences"] += 1
                opening = None
                continue
        body.append(line)
    if opening is not None:
        counts["unclosed_fences"] += 1
    return counts


def audit_index(root: Path, payload: dict) -> list[dict]:
    """Validate index records and bind observations to current source contents."""
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("example audit requires API index schema_version 1")
    if payload.get("filters") != "none":
        raise ValueError("example audit requires a full, unfiltered API index")
    records = payload.get("entries")
    if not isinstance(records, list) or not records:
        raise ValueError("API index entries must be a nonempty list")
    cache: dict[str, tuple[list[str], str]] = {}
    seen: set[tuple[str, str, int]] = set()
    result = []
    for entry in records:
        if not isinstance(entry, dict):
            raise ValueError("each API index entry must be an object")
        package, symbol = entry.get("package"), entry.get("symbol")
        source, line = entry.get("source"), entry.get("line")
        if not isinstance(package, str) or not package:
            raise ValueError("index entry requires a nonempty package")
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol):
            raise ValueError("index entry requires an indexed Rust function name")
        if type(line) is not int or line < 1:
            raise ValueError("index entry line must be a positive integer")
        if not isinstance(source, str):
            raise ValueError("index entry source must be a string")
        if source not in cache:
            raw = source_path(root, source).read_bytes()
            cache[source] = (raw.decode("utf-8").splitlines(), hashlib.sha256(raw).hexdigest())
        lines, digest = cache[source]
        match = PUBLIC_FN_RE.match(lines[line - 1]) if line <= len(lines) else None
        if not match or match.group(1) != symbol:
            raise ValueError(f"stale/mismatched API index: {source}:{line} ({symbol})")
        identity = (package, source, line)
        if identity in seen:
            raise ValueError(f"duplicate API index entry: {identity}")
        seen.add(identity)
        result.append({
            "package": package, "symbol": symbol, "source": source, "line": line,
            "source_sha256": digest,
            "examples": count_examples(adjacent_docs(lines, line)),
        })
    return sorted(result, key=lambda item: (item["package"], item["source"], item["line"]))


def check_policy(root: Path, policy: dict, entries: list[dict]) -> list[str]:
    """Require examples on reviewed source files; reject vacuous/stale scopes."""
    if type(policy.get("schema_version")) is not int or policy["schema_version"] != 1:
        raise ValueError("example policy schema_version must be 1")
    rules = policy.get("sources")
    if not isinstance(rules, list) or not rules:
        raise ValueError("example policy needs a nonempty sources list")
    problems: list[str] = []
    seen: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("each example policy rule must be an object")
        source = rule.get("source")
        path = source_path(root, source)
        if source in seen:
            raise ValueError(f"duplicate policy source: {source}")
        seen.add(source)
        minimum = rule.get("min_runnable_examples")
        if type(minimum) is not int or minimum < 1:
            raise ValueError("min_runnable_examples must be a positive integer")
        required = rule.get("required_symbols")
        if (not isinstance(required, list) or not required
                or any(not isinstance(s, str) or not s for s in required)
                or len(set(required)) != len(required)):
            raise ValueError("required_symbols must be a nonempty list of unique names")
        required_counts = rule.get("required_symbol_counts", {})
        if (not isinstance(required_counts, dict)
                or any(not isinstance(symbol, str) or not symbol for symbol in required_counts)
                or any(type(count) is not int or count < 1 for count in required_counts.values())
                or any(symbol not in required for symbol in required_counts)):
            raise ValueError(
                "required_symbol_counts must map required symbol names to positive integers"
            )
        selected = [entry for entry in entries if entry["source"] == source]
        actual_lines = {
            n for n, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if PUBLIC_FN_RE.match(text)
        }
        indexed_lines = {entry["line"] for entry in selected}
        if not actual_lines or actual_lines != indexed_lines:
            problems.append(f"{source}: policy source is empty or incompletely indexed")
        symbol_counts = Counter(entry["symbol"] for entry in selected)
        missing = set(required) - set(symbol_counts)
        if missing:
            problems.append(f"{source}: required symbols absent: {', '.join(sorted(missing))}")
        for symbol, expected in sorted(required_counts.items()):
            actual = symbol_counts[symbol]
            if actual != expected:
                problems.append(
                    f"{source}: required symbol {symbol} occurs {actual} times; require exactly {expected}"
                )
        for entry in selected:
            count = entry["examples"]["runnable_candidates"]
            if count < minimum:
                problems.append(
                    f"{source}:{entry['line']} {entry['symbol']}: "
                    f"{count} runnable candidates; require {minimum}"
                )
    return problems


def report_payload(entries: list[dict], violations: list[str]) -> dict:
    """Build a deterministic report; source hashes bind it to inspected contents."""
    totals = Counter({kind: 0 for kind in KINDS})
    for entry in entries:
        totals.update(entry["examples"])
    return {
        "schema_version": 1,
        "scope": "Adjacent closed fenced code in the source-level API index; not execution evidence",
        "public_callables": len(entries),
        "with_runnable_candidates": sum(e["examples"]["runnable_candidates"] > 0 for e in entries),
        "with_two_runnable_candidates": sum(e["examples"]["runnable_candidates"] >= 2 for e in entries),
        "fence_totals": dict(totals), "policy_violations": violations, "entries": entries,
    }


def render_markdown(report: dict) -> str:
    """Render coverage and a per-symbol backlog without calling it test coverage."""
    lines = [
        "# SciRust API example inventory", "",
        "> Fenced-code observations, not proof of compilation or execution.",
        "> Run cargo test --doc for actual doctest evidence.", "",
        f"Indexed callables: **{report['public_callables']}**.",
        f"With at least one runnable candidate: **{report['with_runnable_candidates']}**.",
        f"With at least two runnable candidates: **{report['with_two_runnable_candidates']}**.", "",
        "## Fence classifications", "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in report["fence_totals"].items())
    lines.extend(["", "## Policy violations", ""])
    lines.extend(report["policy_violations"] or ["None."])
    lines.extend(["", "## Fewer than two runnable candidates", "",
                  "| Package | Symbol | Candidates | Source |", "|---|---|---:|---|"])
    for entry in report["entries"]:
        count = entry["examples"]["runnable_candidates"]
        if count < 2:
            cells = [str(entry[k]).replace("|", "\\|").replace("\n", " ")
                     for k in ("package", "symbol", "source")]
            lines.append(f"| `{cells[0]}` | `{cells[1]}` | {count} | `{cells[2]}:{entry['line']}` |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write observations even when a reviewed source fails its example policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        entries = audit_index(root, read_json(args.index))
        violations = check_policy(root, read_json(args.policy), entries) if args.policy else []
        report = report_payload(entries, violations)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.markdown:
            args.markdown.write_text(render_markdown(report), encoding="utf-8")
        for problem in violations:
            print(problem, file=sys.stderr)
        return 1 if violations else 0
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"API example audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
