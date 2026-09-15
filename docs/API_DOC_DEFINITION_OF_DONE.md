# Definition of done — SciRust public API documentation

Issue #1431 is complete when all of the following are true on one exact
`master` revision:

1. The externally reachable public callable surface is derived from compiler /
   rustdoc evidence, with source-level discoveries reconciled rather than
   silently assumed reachable.
2. Every reachable callable has a meaningful Rustdoc contract satisfying the
   documentation standard and conventions.
3. Every non-trivial reachable callable has two complementary ordinary runnable
   examples, except explicit reviewed exceptions; permitted trivial one-example
   cases are recorded by policy rather than inferred ad hoc.
4. All ordinary examples counted as executable pass the relevant doctest jobs on
   that exact revision.
5. Every public unsafe callable states its safety invariants; caller-observable
   errors and panics are documented where applicable.
6. The generated Markdown and JSON lexicons expose capability navigation,
   documentation/example observations, source provenance and reachability
   status without overstating evidence.
7. README/documentation navigation links the lexicon, glossary, standards,
   examples and major task guides.
8. Documentation debt and example-policy gates reject new regressions.
9. Normal workspace CI, rustdoc and documentation-specific gates are green.
10. No unresolved documentation review finding is disguised by a lowered
    tolerance, ignored doctest, empty prose or blanket exception.

Completing only the current source-regex summary counter is insufficient. The
objective is a usable and verifiable public contract for SciRust.
