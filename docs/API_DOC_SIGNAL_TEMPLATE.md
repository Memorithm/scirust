# Signal-processing API documentation template

For public signal-processing APIs, document:

- sampling-rate and frequency units;
- real/complex representation and channel layout;
- transform/window/filter normalization convention;
- boundary/padding behavior;
- phase/sign/order convention;
- required lengths and zero-length behavior;
- numerical precision and backend differences;
- errors for invalid rates, cutoffs, shapes or filter parameters.

Examples should use a short deterministic signal with a hand-verifiable or
analytic property, plus a second example covering an alternative configuration
or invalid/boundary request. Spectral claims should state normalization and bin
mapping explicitly.
