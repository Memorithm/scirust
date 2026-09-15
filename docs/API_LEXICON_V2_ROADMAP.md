# API lexicon v2 migration roadmap

Tracking issue: #1431.

## Phase 1 — semantic documentation slices

Document public APIs by coherent module. Each slice adds or improves Rustdoc,
ordinary examples and focused doctest execution without changing implementation
unless the documentation review exposes a separately tested defect.

Current first `scirust-core` slice: `compute_backend`.

## Phase 2 — enrich source observations

Extend the existing source lexicon rather than replacing it. Add observations
for signatures, documentation sections, example counts and source fingerprints.
Parser changes require fixture tests for attributes, multiline signatures,
trait methods, unsafe/extern/async/const combinations and malformed fences.

## Phase 3 — compiler-derived reachability

Reconcile source observations with the generated rustdoc public surface. Keep
`unknown` reachability explicit until an item has compiler-derived evidence.
This phase prevents syntactically public functions hidden behind private modules
from distorting the final public-API completion metric.

## Phase 4 — monotonic repository-wide gate

Once v2 observations are validated against the existing baseline, introduce a
non-regression gate for new reachable callables and expand reviewed two-example
policy scopes crate by crate.

## Definition of done

The documentation program is complete only when every externally reachable
public callable has a meaningful semantic contract and its required executable
examples, all documented exceptions are justified, the generated lexicon is
revision-bound and searchable, and the normal workspace/rustdoc/doctest gates
are green. Raw source-regex debt reaching zero is not by itself sufficient.
