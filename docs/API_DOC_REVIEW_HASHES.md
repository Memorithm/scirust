# Hash/fingerprint API documentation prompts

Document whether a hash is cryptographic or merely a deterministic fingerprint.
State input framing/domain separation, byte ordering/serialization and versioning
that affect identity. Do not imply collision resistance/authentication for FNV or
other non-cryptographic hashes.

Examples should compare stable known/equivalent inputs and a distinct input, not
only assert that the output is nonzero.
