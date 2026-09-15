# Serialization/checkpoint API documentation template

For public persistence APIs, document:

- format/version identifier and compatibility policy;
- byte order, dtype/layout assumptions and integrity checks where applicable;
- atomicity/partial-write behavior;
- maximum/validated lengths before allocation;
- malformed/untrusted input handling;
- whether unknown fields/versions are rejected or preserved;
- deterministic encoding guarantees, if any;
- errors for I/O, validation, checksum and version mismatch.

Examples should include a small round trip and a distinct malformed/version or
validation case. Avoid examples that write to a fixed global path; use in-memory
buffers or temporary paths when the public API supports them.
