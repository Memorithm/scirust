# Numerical API documentation template

Use this as a review aid, not as boilerplate to copy verbatim.

```text
/// Computes <quantity> using <named convention/algorithm>.
///
/// Explain input domain, dimensions/units and mathematical convention. State
/// behavior for empty/non-finite/extreme inputs when relevant. State the output
/// meaning and any numerical limitation that affects interpretation.
///
/// # Errors
/// Describe each caller-observable error condition when the API returns Result.
///
/// # Panics
/// Describe caller-triggerable panics, if any. Prefer removing accidental ones.
///
/// # Examples
/// Nominal example with a meaningful assertion.
///
/// ```
/// ...
/// ```
///
/// Boundary/error/alternative example with a different semantic assertion.
///
/// ```
/// ...
/// ```
```

For stochastic APIs, include seed/reproducibility semantics. For tensor APIs,
include shape/layout/dtype conventions. For iterative solvers, document stopping
criteria and non-convergence. For backend-specific APIs, distinguish compile
support from actual device execution. For floating-point comparisons, justify
tolerances from the algorithm/reference instead of copying an arbitrary epsilon.
