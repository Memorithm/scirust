# Numerical implementation review prompts

While documenting numerical code, inspect:

- intermediate overflow even when the final mathematical result is representable;
- cancellation and subtracting nearly/equally large values;
- division by zero and zero-length denominators;
- integer narrowing/size products;
- `NaN` ordering/comparison behavior;
- infinite endpoints and `0 * infinity` forms;
- accumulation order and parallel/backend differences;
- logarithmic-domain formulas near probability boundaries;
- tolerance scaling.

The quantile/attention audit demonstrated why this matters: several real defects
had no TODO marker and appeared only at contract boundaries.
