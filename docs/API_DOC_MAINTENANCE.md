# API documentation maintenance

The #1431 effort is useful only if completion remains true as SciRust evolves.

## New public APIs

A new externally reachable callable should arrive with its semantic Rustdoc and
required ordinary examples in the same change. A feature is not complete when
its implementation lands first and its public contract is deferred indefinitely.

## Changed behavior

When a public contract changes, update implementation tests, Rustdoc examples,
error/panic documentation and any task-oriented guide that relies on the old
behavior. Generated lexicon artifacts are refreshed only after the source change
is final.

## Removed APIs

Remove stale guides and generated entries. Historical audit/evidence documents
may retain the old symbol when they clearly identify the historical revision.

## Review

Review documentation changes with the same skepticism as code: verify formulas,
units, shapes, boundary cases and claims against implementation/tests. Avoid
accepting prose solely because doctests compile; a compiling example can still
assert the wrong scientific convention.

## Automation

CI should prevent regression in covered scopes and expand monotonically. It
should not auto-write semantic descriptions. Automation is appropriate for
inventory, stale-source detection, example observation, compiler execution and
navigation generation; semantic truth remains a reviewed property.
