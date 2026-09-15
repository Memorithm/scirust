# Documentation infrastructure implementation order

1. Establish standards, glossary/navigation and focused package doctest gate.
2. Complete/review the first `scirust-core` module slice.
3. Extend example-policy scope to that reviewed module using the existing
   observer.
4. Add lexicon v2 source observations with parser tests.
5. Add compiler/rustdoc reachability reconciliation.
6. Migrate global metrics/baselines with an explicit reconciliation report.
7. Continue module/package slices until reachable public debt is zero.

Do not block semantic documentation work on the final reachability tool, but do
not claim repository completion until that tool reconciles the public surface.
