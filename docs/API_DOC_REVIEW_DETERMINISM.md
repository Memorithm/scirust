# Determinism review prompts

For public APIs claiming/relying on determinism, inspect:

- iteration order of maps/sets;
- RNG seed/state;
- thread scheduling/reduction order;
- backend-specific floating-point behavior;
- clocks/random/global identifiers;
- filesystem/network ordering;
- canonical serialization/fingerprint ordering.

Tests should compare repeated/alternative constructions that would expose order
or schedule dependence. A single nonzero fingerprint or successful run is not a
determinism test.
