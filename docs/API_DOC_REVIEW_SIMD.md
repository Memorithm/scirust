# SIMD API review prompts

For public SIMD APIs, document/review:

- required target features and fallback;
- lane width/type;
- alignment/load/store requirements;
- tail handling;
- scalar equivalence/numerical differences;
- portable-simd/nightly feature requirements;
- unsafe invariants;
- architecture-specific qualification.

Examples should demonstrate a tiny scalar-equivalent operation and a distinct
tail/capability case. Performance claims require benchmarks, not merely vectorized
source.
