# Non-finite floating-point review prompts

For public floating-point APIs, determine/document applicable behavior for:

- `NaN` input and ordering;
- positive/negative infinity;
- signed zero;
- extreme finite values;
- overflow/underflow/subnormals;
- non-finite intermediate values;
- whether invalid values are rejected, propagated, ignored or canonicalized.

Do not assume library `min`/`max`/sort/interpolation behavior matches the intended
API contract; test the actual edge cases when they matter.
