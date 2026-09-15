# Provenance and evidence API review prompts

For APIs that emit evidence, attestations, manifests or provenance records,
document/review:

- canonical fields/order and schema version;
- source/workload/environment identity;
- hash/signature algorithm and security non-claims;
- timestamp/clock semantics;
- deterministic serialization;
- verification failure behavior;
- compatibility/versioning;
- distinction between self-reported metadata and independently verified evidence.

Examples should create/verify a small deterministic record and a distinct
mismatch/tamper/version case when the public API supports it.
