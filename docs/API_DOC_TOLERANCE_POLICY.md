# Numerical tolerance review policy

A tolerance in a public example/test is part of the numerical contract evidence.

Before changing it:

1. compute/report the actual residual/error;
2. identify the mathematical/reference value;
3. inspect scaling, conditioning and execution-model differences;
4. determine whether implementation, reference or tolerance is wrong;
5. document the chosen absolute/relative/ULP convention and domain.

Never widen a tolerance solely because a CI runner fails. The Miri
hypergeometric discrepancy recorded in `MIRI_STATUS.md` remains open under this
policy until its residual and reference comparison are established.
