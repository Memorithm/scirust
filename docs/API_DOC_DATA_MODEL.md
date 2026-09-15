# API documentation observation data model

Treat lexicon data as observations with provenance:

**Callable identity** — package + compiler/public path when available, otherwise
source path/line/symbol as provisional identity.

**Source observation** — declaration/signature/adjacent docs/source hash.

**Semantic observation** — summary and presence of standard contract sections;
presence does not prove correctness.

**Example observation** — classified code fences associated with the callable.

**Execution evidence** — exact-revision doctest/check results.

**Reachability evidence** — compiler/rustdoc public-surface reconciliation.

Keeping these records separate prevents a single boolean `documented=true` from
collapsing several materially different guarantees.
