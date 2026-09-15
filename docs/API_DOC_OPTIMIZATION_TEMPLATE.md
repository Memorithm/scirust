# Optimization API documentation template

For optimization/training public APIs, document:

- objective and parameter representation;
- minimization/maximization convention;
- gradient/subgradient requirements;
- learning-rate/step/tolerance semantics;
- initialization and mutable optimizer state;
- stopping/convergence criterion and the fact that termination is not
  necessarily proof of a global optimum;
- determinism, batching and parallel reduction behavior;
- errors for invalid hyperparameters, non-finite objectives or shape mismatch.

Examples should include a small objective with a known optimum or one-step
update and a second validation/state/convergence example. Performance or
convergence-rate comparisons require separate benchmark/reference evidence.
