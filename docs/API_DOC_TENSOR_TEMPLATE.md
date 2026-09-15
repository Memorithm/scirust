# Tensor API documentation template

For public tensor operations, document at least:

- rank and shape requirements for every operand;
- whether broadcasting exists and its exact rules;
- logical dtype and physical-representation requirements;
- layout/stride/contiguity assumptions;
- output shape/dtype and ownership;
- zero-sized dimension behavior;
- dimension/storage overflow behavior;
- backend support and explicit unsupported cases;
- determinism/reproducibility qualifications;
- errors/panics for malformed shapes or incompatible representations.

Examples should normally include one nominal shape transformation/computation
and one boundary or incompatibility case. Avoid allocating huge tensors merely
to demonstrate overflow: use metadata-only constructors when the API permits it.

Representation-aware APIs must distinguish "describable representation" from
"executable representation". Documentation must not imply a kernel mapping
exists merely because a representation can be stored or accounted for.
