#!/usr/bin/env python3
"""Generate a deterministic source-level lexicon of SciRust public callables.

This tool intentionally complements rustdoc instead of replacing it. It scans
workspace library source trees for directly declared public functions and
methods (`pub fn`, including async/const/unsafe/extern variants), records the
first rustdoc sentence when available, and emits a compact Markdown inventory.

Rustdoc remains authoritative for effective visibility, re-exports,
macro-generated items, trait-provided methods, signatures, cfg expansion, and
intra-doc links.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable


PUBLIC_FN_RE = re.compile(
    r"^\s*pub\s+"
    r"(?:(?:const|async|unsafe)\s+)*"
    r"(?:extern(?:\s+\"[^\"]*\")?\s+)?"
    r"fn\s+([A-Za-z_][A-Za-z0-9_]*)\b"
)
DOC_LINE_RE = re.compile(r"^\s*///\s?(.*)$")
ATTRIBUTE_RE = re.compile(r"^\s*#\[")
LIB_KINDS = {"lib", "rlib", "dylib", "staticlib", "cdylib", "proc-macro"}


@dataclass(frozen=True, order=True)
class Entry:
    package: str
    symbol: str
    source: str
    line: int
    summary: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the SciRust public-function lexicon from workspace libraries."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository/workspace root (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write Markdown to this path instead of stdout",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        help="write machine-readable coverage statistics to this path",
    )
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        help="limit generation to a workspace library package; repeatable",
    )
    return parser.parse_args()


def cargo_metadata(root: Path) -> dict:
    command = [
        "cargo",
        "metadata",
        "--format-version",
        "1",
        "--no-deps",
        "--manifest-path",
        str(root / "Cargo.toml"),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("cargo was not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"cargo metadata failed: {detail}") from exc
    return json.loads(result.stdout)


def workspace_packages(metadata: dict, selected: set[str]) -> list[tuple[str, Path]]:
    member_ids = set(metadata.get("workspace_members", []))
    packages: list[tuple[str, Path]] = []
    selected_found: set[str] = set()

    for package in metadata.get("packages", []):
        if package.get("id") not in member_ids:
            continue
        name = str(package["name"])
        if selected and name not in selected:
            continue

        library_targets = []
        for target in package.get("targets", []):
            kinds = set(target.get("kind", []))
            if kinds & LIB_KINDS:
                library_targets.append(Path(target["src_path"]).parent)

        if not library_targets:
            continue

        selected_found.add(name)
        for source_root in sorted(set(library_targets)):
            packages.append((name, source_root))

    packages.sort(key=lambda item: (item[0], item[1].as_posix()))
    if selected:
        missing = sorted(selected - selected_found)
        if missing:
            raise RuntimeError(
                "unknown workspace library package(s): " + ", ".join(missing)
            )
    return packages


def first_sentence(lines: Iterable[str]) -> str:
    text = " ".join(part.strip() for part in lines if part.strip()).strip()
    if not text:
        return ""
    match = re.search(r"(?<=[.!?])\s+", text)
    if match:
        text = text[: match.start()].strip()
    return text.replace("|", "\\|")


def is_library_source(source_root: Path, path: Path) -> bool:
    relative = path.relative_to(source_root)
    if path.name == "main.rs":
        return False
    # Conventional Cargo binary modules are not part of a library's public API.
    if relative.parts and relative.parts[0] == "bin":
        return False
    return True


def scan_file(package: str, path: Path, repo_root: Path) -> list[Entry]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"source is not valid UTF-8: {path}") from exc

    entries: list[Entry] = []
    docs: list[str] = []

    for number, line in enumerate(lines, start=1):
        doc_match = DOC_LINE_RE.match(line)
        if doc_match:
            docs.append(doc_match.group(1))
            continue

        match = PUBLIC_FN_RE.match(line)
        if match:
            relative = path.relative_to(repo_root).as_posix()
            entries.append(
                Entry(
                    package=package,
                    symbol=match.group(1),
                    source=relative,
                    line=number,
                    summary=first_sentence(docs),
                )
            )
            docs = []
            continue

        # Attributes such as #[cfg(...)] and #[inline] may legitimately sit
        # between rustdoc comments and the declaration they describe.
        if ATTRIBUTE_RE.match(line) or not line.strip():
            continue

        docs = []

    return entries


def collect_entries(root: Path, selected: set[str]) -> list[Entry]:
    metadata = cargo_metadata(root)
    entries: set[Entry] = set()
    for package, source_root in workspace_packages(metadata, selected):
        if not source_root.is_dir():
            continue
        for path in sorted(source_root.rglob("*.rs")):
            if not is_library_source(source_root, path):
                continue
            entries.update(scan_file(package, path, root))
    return sorted(entries)


def render_markdown(entries: list[Entry]) -> str:
    documented = sum(bool(entry.summary) for entry in entries)
    packages = sorted({entry.package for entry in entries})
    lines = [
        "# SciRust public function lexicon",
        "",
        "> Generated from the current workspace library sources by `scripts/api-lexicon.py`.",
        "> This is a source-level discovery index, not a replacement for rustdoc.",
        "",
        f"- Directly declared public callables indexed: **{len(entries)}**",
        f"- Workspace library packages represented: **{len(packages)}**",
        f"- Entries with an adjacent `///` summary: **{documented}**",
        "",
        "The scanner indexes `pub fn` declarations, including async, const, unsafe,",
        "and extern variants. It deliberately does not claim effective public",
        "reachability: re-exports, macro-generated items, trait-provided methods, cfg",
        "expansion, and exact signatures remain authoritative in rustdoc.",
        "",
    ]

    current = None
    for entry in entries:
        if entry.package != current:
            if current is not None:
                lines.append("")
            current = entry.package
            lines.extend(
                [
                    f"## `{current}`",
                    "",
                    "| Symbol | Summary | Source |",
                    "|---|---|---|",
                ]
            )
        summary = entry.summary or "_No adjacent `///` summary detected._"
        source = f"`{entry.source}:{entry.line}`"
        lines.append(f"| `{entry.symbol}` | {summary} | {source} |")

    lines.append("")
    return "\n".join(lines)


def stats(entries: list[Entry]) -> dict:
    by_package: dict[str, dict[str, int]] = {}
    for entry in entries:
        package = by_package.setdefault(
            entry.package, {"public_callables": 0, "documented": 0}
        )
        package["public_callables"] += 1
        if entry.summary:
            package["documented"] += 1

    documented = sum(bool(entry.summary) for entry in entries)
    return {
        "schema_version": 1,
        "public_callables": len(entries),
        "documented": documented,
        "undocumented": len(entries) - documented,
        "packages": by_package,
    }


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    try:
        entries = collect_entries(root, set(args.package))
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"api-lexicon: {exc}", file=sys.stderr)
        return 2

    if not entries:
        print("api-lexicon: no public callables found", file=sys.stderr)
        return 1

    markdown = render_markdown(entries)
    if args.output:
        write_text(args.output, markdown)
    else:
        sys.stdout.write(markdown)

    if args.stats_json:
        write_text(args.stats_json, json.dumps(stats(entries), indent=2, sort_keys=True) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
