# Cache API review prompts

For public cache/memoization/KV-cache APIs, document/review:

- key identity/canonicalization;
- value ownership/lifetime;
- capacity/eviction policy;
- consistency/invalidation semantics;
- thread/concurrency behavior;
- persistence/device tiering if applicable;
- hit/miss/error behavior;
- determinism/reproducibility impact;
- memory/storage accounting.

Examples should demonstrate hit/miss or insert/retrieve semantics and a distinct
invalidation/eviction/boundary case. Performance claims require benchmark
evidence separate from functional examples.
