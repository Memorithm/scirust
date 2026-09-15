# Probability/statistics API review prompts

Document/review:

- distribution/statistic definition and parameterization;
- support/domain;
- sample versus population convention;
- normalization/interpolation convention;
- log-domain versus direct probability behavior;
- invalid parameter/non-finite input handling;
- extreme-tail/overflow/underflow behavior;
- RNG reproducibility for sampling APIs;
- exact/reference validation where practical.

Examples should use known analytic values or exact small combinatorial cases and
a distinct invalid/boundary case. Probability sums/normalization tolerances need
a justified scale/reference.
