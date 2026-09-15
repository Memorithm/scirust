# Representation-aware API review prompts

When logical values can have multiple physical representations, document/review:

- logical shape/dtype independent of storage encoding;
- exact representation identity/layout metadata;
- storage accounting and overflow;
- reconstruction parameters/geometry;
- which representations are only describable versus executable;
- kernel/backend mapping and unsupported errors;
- transitions/rebinding validation;
- fingerprint/cache identity implications.

Never silently treat an unsupported sparse/quantized representation as dense.
Examples should make unsupported execution explicit when that is the current
contract.
