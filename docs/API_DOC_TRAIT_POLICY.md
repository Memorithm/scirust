# Public trait documentation policy

Public traits define contracts across implementations and require documentation
at both trait and method level.

Trait documentation should state the abstraction, invariants and implementation
expectations. Each public method documents its own inputs/output/failure
semantics. Default methods state whether implementors may override behavior and
which invariants overrides must preserve.

Examples should use a concrete public implementation when available. For traits
intended for downstream implementation, include a minimal implementor example
when it adds information beyond ordinary method use.

The lexicon must eventually account for trait-provided methods even when source
regex discovery of `pub fn` does not see them as ordinary public declarations.
