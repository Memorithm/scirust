# Matrix/linear algebra API review prompts

For public matrix/linear algebra APIs, document/review:

- dimensions and row/column-major semantics;
- transpose/conjugate-transpose convention;
- broadcasting (usually none unless explicit);
- singular/rank-deficient behavior;
- pivoting/normalization convention;
- tolerance/conditioning behavior;
- BLAS/backend reproducibility differences;
- output layout/ownership.

Examples should use a small hand-verifiable matrix identity/solve and a distinct
dimension/singularity/boundary case.
