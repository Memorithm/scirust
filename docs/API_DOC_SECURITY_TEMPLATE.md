# Security-sensitive API documentation template

For cryptographic, privacy, robustness-certification, authentication or other
security-sensitive public APIs, documentation must separate implemented
mechanism from security guarantee.

Document:

- threat/security model and explicit non-goals;
- parameter/key/randomness requirements;
- side-channel or constant-time status when relevant;
- misuse resistance and validation behavior;
- provenance of algorithms/standards;
- whether the implementation is experimental, reference-only or independently
  audited;
- error behavior without leaking sensitive material;
- evidence supporting any security/certification claim.

Examples must use synthetic keys/data and must not encourage embedding secrets.
Do not describe research/reference cryptography as production hardened unless
there is evidence supporting that qualification.
