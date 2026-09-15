# Example of a high-quality callable review

Consider a public numerical function returning a percentile.

A weak comment says: "Returns a quantile."

A useful review establishes:

- which interpolation convention is used;
- probability clamping/rejection policy;
- empty/NaN/infinity behavior;
- whether input is mutated;
- complexity/allocation behavior when relevant;
- numerical behavior at extreme finite endpoints;
- one ordinary example with a known percentile;
- a second example for a boundary/non-finite contract;
- doctest and regression evidence.

The same method applies outside statistics: identify the conventions a caller
cannot safely infer from the Rust signature and turn them into tested public
contracts.
