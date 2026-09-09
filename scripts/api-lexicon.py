#!/usr/bin/env python3
"""Generate a deterministic source-level lexicon of SciRust public callables.

This tool complements rustdoc instead of replacing it. It scans workspace
library source trees for directly declared public functions and methods
(`pub fn`, including async/const/unsafe/extern variants), records the first
adjacent rustdoc sentence when available, and emits searchable Markdown/JSON
indexes.

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
DOMAIN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
LIB_KINDS = {"lib", "rlib", "dylib", "staticlib", "cdylib", "proc-macro"}


@dataclass(frozen=True, order=True)
class Entry:
    package: str
    symbol: str
    source: str
    line: int
    summary: str


@dataclass(frozen=True)
class Domain:
    id: str
    title: str
    description: str
    packages: frozenset[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate/search the SciRust public-function lexicon."
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
        help="write the (optionally filtered) Markdown lexicon to this path",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        help="write aggregate documentation statistics to this path",
    )
    parser.add_argument(
        "--index-json",
        type=Path,
        help="write every selected entry as machine-readable JSON",
    )
    parser.add_argument(
        "--domain-index",
        type=Path,
        help="write the concise capability/domain index to this path",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("docs/api-domains.json"),
        help="domain taxonomy JSON, relative to --root by default",
    )
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        help="limit results to a workspace library package; repeatable",
    )
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        help="limit results to a capability domain id; repeatable",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help=(
            "case-insensitive capability search across symbol, rustdoc summary, "
            "package and source; repeatable, and all whitespace-separated terms match"
        ),
    )
    parser.add_argument(
        "--missing-docs",
        action="store_true",
        help="show only entries without an adjacent /// summary",
    )
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="print available capability domain ids and exit",
    )
    parser.add_argument(
        "--require-domain-coverage",
        action="store_true",
        help="fail if a represented library package is absent from the taxonomy",
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


def workspace_member_names(metadata: dict) -> set[str]:
    member_ids = set(metadata.get("workspace_members", []))
    return {
        str(package["name"])
        for package in metadata.get("packages", [])
        if package.get("id") in member_ids
    }


def workspace_library_targets(
    metadata: dict, selected: set[str]
) -> list[tuple[str, Path]]:
    member_ids = set(metadata.get("workspace_members", []))
    targets: list[tuple[str, Path]] = []
    selected_found: set[str] = set()

    for package in metadata.get("packages", []):
        if package.get("id") not in member_ids:
            continue
        name = str(package["name"])
        if selected and name not in selected:
            continue

        package_targets = []
        for target in package.get("targets", []):
            kinds = set(target.get("kind", []))
            if kinds & LIB_KINDS:
                package_targets.append(Path(target["src_path"]).parent)

        if not package_targets:
            continue

        selected_found.add(name)
        for source_root in sorted(set(package_targets)):
            targets.append((name, source_root))

    targets.sort(key=lambda item: (item[0], item[1].as_posix()))
    if selected:
        missing = sorted(selected - selected_found)
        if missing:
            raise RuntimeError(
                "unknown workspace library package(s): " + ", ".join(missing)
            )
    return targets


def taxonomy_path(root: Path, configured: Path) -> Path:
    return configured if configured.is_absolute() else root / configured


def load_domains(path: Path, workspace_names: set[str]) -> list[Domain]:
    if not path.is_file():
        raise RuntimeError(f"domain taxonomy does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid domain taxonomy JSON: {exc}") from exc

    if payload.get("schema_version") != 1:
        raise RuntimeError("domain taxonomy schema_version must be 1")

    raw_domains = payload.get("domains")
    if not isinstance(raw_domains, list) or not raw_domains:
        raise RuntimeError("domain taxonomy must contain a non-empty `domains` list")

    domains: list[Domain] = []
    seen_ids: set[str] = set()
    for raw in raw_domains:
        if not isinstance(raw, dict):
            raise RuntimeError("each domain must be a JSON object")
        domain_id = str(raw.get("id", "")).strip()
        title = str(raw.get("title", "")).strip()
        description = str(raw.get("description", "")).strip()
        packages = raw.get("packages")

        if not DOMAIN_ID_RE.fullmatch(domain_id):
            raise RuntimeError(
                f"invalid domain id `{domain_id}`; use lowercase letters, digits and hyphens"
            )
        if domain_id in seen_ids:
            raise RuntimeError(f"duplicate domain id: {domain_id}")
        if not title or not description:
            raise RuntimeError(f"domain `{domain_id}` needs title and description")
        if not isinstance(packages, list) or not packages:
            raise RuntimeError(f"domain `{domain_id}` needs a non-empty package list")

        normalized = [str(package).strip() for package in packages]
        if any(not package for package in normalized):
            raise RuntimeError(f"domain `{domain_id}` contains an empty package name")
        if len(set(normalized)) != len(normalized):
            raise RuntimeError(f"domain `{domain_id}` contains duplicate packages")

        unknown = sorted(set(normalized) - workspace_names)
        if unknown:
            raise RuntimeError(
                f"domain `{domain_id}` references unknown workspace package(s): "
                + ", ".join(unknown)
            )

        seen_ids.add(domain_id)
        domains.append(
            Domain(
                id=domain_id,
                title=title,
                description=description,
                packages=frozenset(normalized),
            )
        )

    return domains


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

        if ATTRIBUTE_RE.match(line) or not line.strip():
            continue

        docs = []

    return entries


def collect_entries(root: Path, targets: list[tuple[str, Path]]) -> list[Entry]:
    entries: set[Entry] = set()
    for package, source_root in targets:
        if not source_root.is_dir():
            continue
        for path in sorted(source_root.rglob("*.rs")):
            if not is_library_source(source_root, path):
                continue
            entries.update(scan_file(package, path, root))
    return sorted(entries)


def membership_by_package(domains: list[Domain]) -> dict[str, list[str]]:
    membership: dict[str, list[str]] = {}
    for domain in domains:
        for package in domain.packages:
            membership.setdefault(package, []).append(domain.id)
    for package_domains in membership.values():
        package_domains.sort()
    return membership


def selected_domain_packages(domains: list[Domain], selected: set[str]) -> set[str]:
    if not selected:
        return set()
    by_id = {domain.id: domain for domain in domains}
    unknown = sorted(selected - set(by_id))
    if unknown:
        raise RuntimeError("unknown domain(s): " + ", ".join(unknown))
    packages: set[str] = set()
    for domain_id in selected:
        packages.update(by_id[domain_id].packages)
    return packages


def query_terms(queries: list[str]) -> list[str]:
    terms: list[str] = []
    for query in queries:
        terms.extend(term.casefold() for term in query.split() if term.strip())
    return terms


def filter_entries(
    entries: list[Entry],
    domains: list[Domain],
    selected_domains: set[str],
    queries: list[str],
    missing_docs: bool,
) -> list[Entry]:
    allowed_packages = selected_domain_packages(domains, selected_domains)
    terms = query_terms(queries)
    filtered: list[Entry] = []

    for entry in entries:
        if allowed_packages and entry.package not in allowed_packages:
            continue
        if missing_docs and entry.summary:
            continue
        if terms:
            haystack = " ".join(
                [entry.symbol, entry.summary, entry.package, entry.source]
            ).casefold()
            if not all(term in haystack for term in terms):
                continue
        filtered.append(entry)

    return filtered


def filter_description(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.package:
        parts.append("packages=" + ",".join(sorted(set(args.package))))
    if args.domain:
        parts.append("domains=" + ",".join(sorted(set(args.domain))))
    if args.query:
        parts.append("query=" + " ".join(args.query))
    if args.missing_docs:
        parts.append("missing-docs-only")
    return "; ".join(parts) if parts else "none"


def render_markdown(entries: list[Entry], filters: str) -> str:
    documented = sum(bool(entry.summary) for entry in entries)
    packages = sorted({entry.package for entry in entries})
    lines = [
        "# SciRust public function lexicon",
        "",
        "> Generated from the current workspace library sources by `scripts/api-lexicon.py`.",
        "> This is a source-level discovery index, not a replacement for rustdoc.",
        "",
        f"- Filters: **{filters}**",
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


def package_counts(entries: list[Entry]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for entry in entries:
        package = result.setdefault(
            entry.package, {"public_callables": 0, "documented": 0, "undocumented": 0}
        )
        package["public_callables"] += 1
        if entry.summary:
            package["documented"] += 1
        else:
            package["undocumented"] += 1
    return result


def domain_counts(
    entries: list[Entry], domains: list[Domain]
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for domain in domains:
        domain_entries = [entry for entry in entries if entry.package in domain.packages]
        documented = sum(bool(entry.summary) for entry in domain_entries)
        result[domain.id] = {
            "title": domain.title,
            "public_callables": len(domain_entries),
            "documented": documented,
            "undocumented": len(domain_entries) - documented,
            "packages_with_entries": sorted({entry.package for entry in domain_entries}),
        }
    return result


def stats_payload(
    entries: list[Entry],
    domains: list[Domain],
    filters: str,
) -> dict:
    documented = sum(bool(entry.summary) for entry in entries)
    memberships = membership_by_package(domains)
    represented = {entry.package for entry in entries}
    unclassified = sorted(represented - set(memberships))
    return {
        "schema_version": 2,
        "filters": filters,
        "public_callables": len(entries),
        "documented": documented,
        "undocumented": len(entries) - documented,
        "packages": package_counts(entries),
        "domains": domain_counts(entries, domains),
        "unclassified_packages": unclassified,
    }


def index_payload(entries: list[Entry], domains: list[Domain], filters: str) -> dict:
    memberships = membership_by_package(domains)
    return {
        "schema_version": 1,
        "filters": filters,
        "entries": [
            {
                "package": entry.package,
                "symbol": entry.symbol,
                "summary": entry.summary,
                "source": entry.source,
                "line": entry.line,
                "domains": memberships.get(entry.package, []),
            }
            for entry in entries
        ],
    }


def percentage(documented: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{(documented * 100.0 / total):.1f}%"


def render_domain_index(entries: list[Entry], domains: list[Domain]) -> str:
    counts = package_counts(entries)
    represented = set(counts)
    membership = membership_by_package(domains)
    classified = represented & set(membership)
    unclassified = sorted(represented - set(membership))

    lines = [
        "# SciRust capability index",
        "",
        "> Generated from `docs/api-domains.json` and the current source-level function lexicon.",
        "> Domains are a navigation taxonomy; rustdoc remains authoritative for API semantics.",
        "",
        f"- Library packages represented by public callables: **{len(represented)}**",
        f"- Represented packages mapped to at least one domain: **{len(classified)}**",
        f"- Represented packages not yet classified: **{len(unclassified)}**",
        "",
    ]

    for domain in domains:
        domain_packages = sorted(domain.packages & represented)
        domain_entries = [entry for entry in entries if entry.package in domain.packages]
        documented = sum(bool(entry.summary) for entry in domain_entries)
        lines.extend(
            [
                f"## {domain.title} (`{domain.id}`)",
                "",
                domain.description,
                "",
                f"Indexed callables: **{len(domain_entries)}**; "
                f"with adjacent `///` summary: **{documented}** "
                f"({percentage(documented, len(domain_entries))}).",
                "",
                "| Package | Callables | With `///` summary | Summary coverage |",
                "|---|---:|---:|---:|",
            ]
        )
        for package in domain_packages:
            package_stats = counts[package]
            lines.append(
                f"| `{package}` | {package_stats['public_callables']} | "
                f"{package_stats['documented']} | "
                f"{percentage(package_stats['documented'], package_stats['public_callables'])} |"
            )
        if not domain_packages:
            lines.append("| _No represented package in this revision_ | 0 | 0 | n/a |")
        lines.append("")

    lines.extend(["## Unclassified represented packages", ""])
    if unclassified:
        lines.append(
            "These packages expose directly declared public callables but are not yet "
            "assigned to a capability domain:"
        )
        lines.append("")
        for package in unclassified:
            lines.append(f"- `{package}`")
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    try:
        metadata = cargo_metadata(root)
        workspace_names = workspace_member_names(metadata)
        domains = load_domains(taxonomy_path(root, args.taxonomy), workspace_names)

        if args.list_domains:
            for domain in domains:
                print(f"{domain.id}\t{domain.title}")
            return 0

        selected_packages = set(args.package)
        targets = workspace_library_targets(metadata, selected_packages)
        all_entries = collect_entries(root, targets)

        memberships = membership_by_package(domains)
        represented = {entry.package for entry in all_entries}
        unclassified = sorted(represented - set(memberships))
        if args.require_domain_coverage and unclassified:
            raise RuntimeError(
                "represented package(s) missing from domain taxonomy: "
                + ", ".join(unclassified)
            )

        entries = filter_entries(
            all_entries,
            domains,
            set(args.domain),
            args.query,
            args.missing_docs,
        )
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"api-lexicon: {exc}", file=sys.stderr)
        return 2

    if not entries:
        print("api-lexicon: no public callables matched the requested filters", file=sys.stderr)
        return 1

    filters = filter_description(args)
    markdown = render_markdown(entries, filters)
    if args.output:
        write_text(args.output, markdown)
    else:
        sys.stdout.write(markdown)

    if args.stats_json:
        write_text(
            args.stats_json,
            json.dumps(stats_payload(entries, domains, filters), indent=2, sort_keys=True)
            + "\n",
        )

    if args.index_json:
        write_text(
            args.index_json,
            json.dumps(index_payload(entries, domains, filters), indent=2, sort_keys=True)
            + "\n",
        )

    if args.domain_index:
        write_text(args.domain_index, render_domain_index(entries, domains))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
