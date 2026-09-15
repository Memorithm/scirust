# Boundary cases for public API review

Select applicable cases rather than mechanically testing every item:

- empty input / scalar / singleton;
- zero-sized dimensions;
- minimum/maximum integer dimensions and checked narrowing;
- duplicate/equal values;
- `NaN`, infinities, signed zero and extreme finite floats;
- overflow/cancellation/subnormal regimes;
- mismatched shapes/dtypes/representations;
- invalid enum/configuration combinations;
- missing/unknown identifiers;
- maximum iteration/resource limits;
- unavailable backend/feature;
- malformed/truncated/untrusted serialized input.

Boundary examples are especially useful as the second Rustdoc example when they
express a stable caller contract. Expensive cases should be tested with metadata
or small fixtures where possible.
