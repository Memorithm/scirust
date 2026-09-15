# Factorization and low-rank API review prompts

Document/review:

- factor shapes/order and reconstruction formula;
- rank constraints;
- normalization/sign/ordering ambiguity;
- approximation/error metric;
- deterministic initialization/decomposition behavior;
- storage accounting;
- executable versus representation-only support;
- errors for incompatible dimensions/rank.

Examples should reconstruct a tiny exact low-rank object and a distinct invalid
rank/shape or approximation case.
