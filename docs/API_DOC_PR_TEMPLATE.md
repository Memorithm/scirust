# API documentation PR template

Use the following structure in #1431 implementation PRs.

## Scope

List exact package/module/source paths and whether runtime behavior changes.

## Contracts reviewed

Summarize shapes/domains/units/errors/panics/safety/backend/numerical conventions
that were verified against implementation/tests.

## Examples

State the number of policy-covered callables, ordinary examples observed and
ordinary examples actually executed as doctests. Keep presence and execution
counts separate.

## Defects found

List verified implementation/test defects discovered during documentation
review and their regression tests. If none, state that no runtime change is
claimed.

## Evidence

Record exact-head rustdoc/doctest/package/workspace checks. Do not copy evidence
from an earlier head after changing the branch.

## Debt delta

Report documentation/example baseline changes. Improvements are reviewed before
baseline refresh; regressions are not accepted.

## Non-claims

State hardware/performance/security/effective-reachability claims not established
by the slice.
