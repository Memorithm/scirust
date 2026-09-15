# API documentation tooling plan

Tool changes for #1431 should be implemented in reviewable stages.

## Stage A

Preserve `api-lexicon.py` output and tests; consume `api-example-audit.py`
observations in reporting without changing reachability semantics.

## Stage B

Add robust signature/doc-section observations. Parser fixtures must cover
attributes, multiline declarations, impl/trait methods and function modifiers.
Do not infer `# Errors`/`# Panics` requiredness from names alone.

## Stage C

Add compiler/rustdoc public-surface extraction and reconciliation. Record
canonical public paths and explicit unknown/unmatched cases.

## Stage D

Generate Markdown/JSON v2, migrate baselines with a reconciliation report and
make new reachable APIs fail when they introduce uncovered documentation/example
debt.

Every stage must retain deterministic output and source provenance.
