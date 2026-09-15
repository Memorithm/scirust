# API documentation quality model

Issue #1431 measures several independent dimensions. No single percentage is a
sufficient quality score.

## Coverage

Does every externally reachable callable have documentation? Source-level
adjacent-summary debt is an interim indicator until compiler-derived
reachability is reconciled.

## Semantic completeness

Does the documentation state the operation, input/output conventions and
caller-visible failure behavior? This requires review against implementation and
tests; it cannot be proven by counting `///` lines.

## Example coverage

Does each policy-covered non-trivial callable have two complementary ordinary
examples? Example observation is measured separately from execution.

## Executability

Do ordinary examples actually pass `cargo test --doc` on the same revision?
Hardware/network exceptions require reviewed alternative evidence.

## Claim discipline

Are performance, determinism, numerical-accuracy and hardware claims bounded by
actual evidence? Revision-bound reports must not silently become evergreen API
guarantees.

## Navigability

Can a user move from capability domain to package/module to callable and from a
callable to examples, concepts and source? The API lexicon, glossary and rustdoc
serve different parts of this requirement.

## Freshness

Are generated inventories and evidence tied to the source revision they
represent? Stale generated documentation must be detectable rather than
presented as current.
