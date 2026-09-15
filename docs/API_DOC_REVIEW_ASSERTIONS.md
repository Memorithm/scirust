# Semantic assertions in examples

Prefer assertions that teach the contract:

- exact output for hand-verifiable integer/simple floating cases;
- output shape/dtype/length invariants;
- typed error variants;
- monotonicity/bounds when those are documented properties;
- deterministic equality under a fixed seed/configuration;
- round-trip equality for serialization;
- analytic derivative/residual for autodiff/solvers.

Avoid examples whose only assertion is `is_ok()` when a stronger result can be
checked, or that merely print/debug a value. Use approximate comparisons only
with a justified tolerance and scale.
