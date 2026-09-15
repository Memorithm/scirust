# WASM/WebGPU API review prompts

For public WASM/browser/WebGPU-facing APIs, document/review:

- target/features/browser/runtime requirements;
- async initialization semantics;
- JS/WASM ownership/copy boundaries;
- WebGPU capability/limit detection;
- error/promise mapping;
- memory growth/resource cleanup;
- deterministic differences from native paths;
- compile versus browser/device execution evidence.

Examples should use compileable target-neutral contracts or controlled WASM test
harnesses; do not imply browser/GPU execution from native hosted compilation.
