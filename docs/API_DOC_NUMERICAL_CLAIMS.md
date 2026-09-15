# Numerical accuracy documentation policy

Numerical documentation must distinguish mathematical identity, implementation
formula and observed error.

When accuracy matters, state:

- reference/exact quantity;
- input domain and scale;
- absolute/relative/ULP tolerance convention;
- exceptional values (`NaN`, infinities, subnormals) policy;
- algorithmic regimes/crossovers if different formulas are used;
- reproducibility/backend qualifications.

A tolerance should be justified by the algorithm/reference and test domain. Do
not widen a tolerance solely because a CI environment fails. First inspect the
computed residual and determine whether implementation, reference, execution
model or tolerance is responsible.
