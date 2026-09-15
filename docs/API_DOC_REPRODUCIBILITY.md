# Reproducibility documentation

Use precise levels:

- **deterministic in one configuration** — repeated runs under stated conditions
  produce the same result;
- **seed-reproducible** — RNG behavior is reproducible under a stated seed and
  compatibility scope;
- **numerically equivalent** — results agree under a stated tolerance;
- **bit-identical** — floating-point/result bit patterns are identical;
- **cross-backend reproducible** — a stated property has been tested across named
  backends/targets.

Do not use these terms interchangeably. Parallel reductions, BLAS/GPU kernels,
compiler targets and algorithm changes can preserve numerical equivalence while
changing bit patterns.
