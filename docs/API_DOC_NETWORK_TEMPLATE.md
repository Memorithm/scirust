# Network API documentation template

For public network-facing APIs, document:

- protocol and endpoint semantics;
- authentication/credential source without embedding secrets in examples;
- timeout/retry/idempotency behavior;
- request/response validation and size limits;
- TLS/security assumptions;
- offline/failure behavior;
- determinism and external-service dependence;
- error separation between local validation, transport and remote protocol.

Ordinary doctests should exercise request construction, parsing or a local
fixture rather than depend on a mutable Internet service. A live integration
example can be documented separately but is not counted as executed evidence
unless a controlled integration job actually runs it.
