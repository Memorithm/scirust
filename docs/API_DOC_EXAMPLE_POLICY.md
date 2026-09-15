# Public API example policy

The target requested by issue #1431 is one or two usage examples for every
externally reachable public callable.

## Default

A non-trivial callable requires **two ordinary runnable examples**:

1. minimal nominal use with a semantic assertion;
2. a complementary boundary, error, alternative configuration, shape, state or
   determinism example.

## One-example cases

A single ordinary example can be sufficient for a genuinely trivial accessor or
constructor only after the callable is included in a reviewed policy scope that
records this classification. The classification is not inferred merely because
the implementation is short.

## What counts

An example counts as executable evidence only when it is an ordinary Rust
fence and the relevant `cargo test --doc` succeeds on the same source revision.
`no_run`, `ignore`, `compile_fail`, `should_panic`, unknown-language and malformed
fences are tracked separately.

## Quality

Two copies of the same example count as one semantic example. Examples should
assert observable behavior rather than only instantiate a type or print output.
Keep fixtures deterministic, small and free of external credentials/services.

## Exceptions

Hardware/network/OS-only constraints are recorded in `API_DOC_EXCEPTIONS.md`
with alternative evidence and a removal condition. Exceptions are not a general
mechanism for bypassing broken doctests.
