# SciRust documentation

This directory contains user guidance, architecture records, generated API
navigation and revision-bound research/audit material.

## Start here

- [Quickstart](QUICKSTART.md) — first use of the workspace and its primary APIs.
- [Architecture](ARCHITECTURE.md) — major components and architectural boundaries.
- [Command and API reference](REFERENCE.md) — command-oriented reference material.
- [Public API lexicon](API_LEXICON.md) — generated/searchable source-level callable index.
- [API example inventory](API_EXAMPLES.md) — executable-example coverage and audit guidance.
- [API documentation standard](API_DOCUMENTATION_STANDARD.md) — required semantic contract and example policy for public APIs.
- [API documentation progress](API_DOCUMENTATION_PROGRESS.md) — reviewed slices and remaining program.
- [GPU guide](GPU.md) — GPU status, capabilities and qualifications.
- [Test protocol](TEST_PROTOCOL.md) — validation and evidence conventions.
- [Release process](RELEASING.md) — release procedure.

## Lexicon versus rustdoc

The lexicon is optimized for repository-wide discovery and audit. It does not
replace rustdoc. Rustdoc remains authoritative for effective visibility,
re-exports, macro expansion, trait-provided methods, exact signatures and
intra-document links.

## Historical and research documents

The directory also contains experiment reports, roadmaps, audits and integration
notes. Such documents are evidence for the revision or experiment they name;
they must not be read as a statement that the current `master` still has the
same implementation or qualification status.
