# Cryptographic API documentation template

For public cryptographic APIs, document:

- primitive/scheme and standards/reference version;
- key/signature/hash sizes and encoding;
- randomness requirements;
- verification/failure semantics;
- domain separation/context handling;
- constant-time/side-channel status when known;
- experimental versus production-hardening status;
- malformed/untrusted input validation and allocation bounds;
- `# Safety` for unsafe implementation boundaries.

Examples should use fixed synthetic test vectors or round trips, plus a distinct
invalid/tampered-input case. Do not embed real secrets or imply a security audit
that has not occurred.
