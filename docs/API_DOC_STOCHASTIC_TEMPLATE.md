# Stochastic API documentation template

For random/stochastic public APIs, document:

- distribution/process parameterization;
- support/domain and units;
- seed/RNG ownership and reproducibility semantics;
- whether equal seeds are promised to produce equal sequences across versions or
  only within the current implementation;
- treatment of invalid/non-finite parameters;
- numerical behavior in extreme tails;
- sampling cost or state mutation when relevant.

One example should use a fixed seed and assert a deterministic structural
property. A second should cover validation, support bounds or a known analytic
moment with an appropriately designed deterministic test. Do not use a flaky
single-run statistical assertion as a doctest.
