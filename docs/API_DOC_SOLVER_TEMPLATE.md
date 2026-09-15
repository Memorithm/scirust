# Solver API documentation template

For iterative/direct solver APIs, document:

- mathematical problem and matrix/operator assumptions;
- dimensions and storage/layout requirements;
- tolerance definition (absolute, relative, residual norm, etc.);
- maximum-iteration and stopping semantics;
- initial guess/preconditioner behavior;
- singularity/breakdown/non-convergence handling;
- determinism and parallel reduction qualifications;
- output residual/status information.

Examples should include a small hand-verifiable problem and a distinct
validation/non-convergence/boundary case. Tolerances in examples should follow
from the problem scale and algorithm rather than an unexplained epsilon.
