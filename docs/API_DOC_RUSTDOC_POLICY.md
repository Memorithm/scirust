# Rustdoc policy

Item-level Rustdoc is the canonical human contract for public Rust APIs.

Use module docs for shared concepts/invariants and item docs for callable-specific
semantics. Prefer intra-doc links to related types/errors. Keep examples beside
the callable so signature/behavior changes break close to the source through
doctests.

Task guides may explain multi-step workflows and the generated lexicon may index
thousands of callables, but neither should duplicate complete item contracts.
This keeps the authoritative semantics close to implementation while preserving
repository-wide navigation.
