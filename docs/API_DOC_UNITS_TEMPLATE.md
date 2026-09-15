# Units and metrology API documentation template

For public units/metrology APIs, document:

- physical quantity and canonical unit;
- accepted input/output units and conversion direction;
- dimensional consistency rules;
- uncertainty/tolerance representation and propagation convention;
- rounding and significant-digit behavior when relevant;
- domain/range and non-finite handling;
- errors for incompatible dimensions or invalid calibration data.

Examples should include a known conversion/quantity identity and a second
uncertainty, incompatibility or boundary case. Never leave unit conventions to
be inferred from variable names alone.
