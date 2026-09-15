# Fingerprint and cache identity review prompts

For APIs producing workload/cache/provenance fingerprints, document/review:

- canonical fields and ordering;
- version/domain-separation tag;
- which semantic/representation changes alter identity;
- which irrelevant insertion/order changes must not alter identity;
- collision/security non-claims when using non-cryptographic hashes;
- persistence/version compatibility;
- mutability hazards when a public record can change after fingerprint creation.

Tests should compare independently constructed equivalent/different cases rather
than merely checking that a fingerprint is nonzero.
