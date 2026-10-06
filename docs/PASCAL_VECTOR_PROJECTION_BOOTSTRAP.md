# Pascal Vector Projection (PVP) — SciRust reusable-primitives bootstrap

Status: shared-primitives research programme.

## Role

SciRust owns only the reusable, product-neutral substrate required to express
and qualify PVP efficiently:

- versioned bitplane layout;
- checked pack/unpack/transpose;
- scalar reference implementation;
- CPU SIMD candidates and runtime capability dispatch;
- backend-neutral storage/accounting contracts;
- independent correctness and benchmark utilities.

SML-GENIUS owns Pascal/ANF model semantics. FLAT-ATTENTION owns specialized
portable GPU kernel realization. NNIS owns optional portable runtime/session
qualification.

## Canonical candidate layout

For K Boolean addresses and G independent ANF gates:

- CPU: `[K, ceil(G/64)] u64`;
- portable GPU correspondence: `[K, ceil(G/128)] vec4<u32>`;
- address is the outer logical coordinate;
- gate words are the contiguous inner coordinate.

This is an SoA/AoSoA-style bit-sliced representation. It is designed so a
single word/vector operation advances many independent gates simultaneously.

The Pascal/subset-zeta hot path is a regular staged XOR butterfly. SciRust must
not represent the subset lattice as graph nodes or adjacency lists for this
shared primitive.

## Existing SciRust assets

Reuse and extend rather than duplicate:

- `scirust-simd` scalar/SSE2/AVX2/AVX-512/NEON/SVE dispatch;
- stable `core::arch` kernels;
- optional `portable_simd` and nightly SVE paths;
- `scirust-compute` ISA/capability probing;
- deterministic candidate planning/qualification patterns from SIMD GEMM;
- CPU/reference and WGPU adapter conventions where generic.

## Hardware sovereignty

The PVP programme is ecosystem-first. New PVP work must not require CUDA,
NVRTC, cuDNN, TensorRT/TensorRT-LLM, NVML, CUTLASS or other NVIDIA runtime/SDK
software. The installed GPU driver required by an open backend is the only
vendor-specific layer admitted by the target programme.

Existing optional CUDA facilities in SciRust remain separate from PVP.

## Programme

1. SR-PVP0: define schema/versioned layout, checked dimensions and exact storage
   accounting.
2. SR-PVP1: deterministic scalar `u64` transform plus pack/unpack/transpose
   differential tests.
3. SR-PVP2: NEON/AVX2/AVX-512/SVE candidates operating over contiguous gate
   words; scalar fallback always preserved.
4. SR-PVP3: candidate descriptors, capability filtering and oracle
   qualification before timing.
5. SR-PVP4: same-host physical benchmark matrix over K, G, alignment and batch;
   retain losing candidates.
6. SR-PVP5: publish the minimal backend-neutral layout/primitive contract for
   FLAT and NNIS consumers.
7. SR-PVP6: promote only primitives that are genuinely shared by multiple
   consumers; product/model policy remains outside SciRust.

## SML boundary

SciRust is not part of the required final SML runtime. SML may internalize a
qualified minimal primitive after versioned equivalence and destination
qualification. The future SML-HARNESS is also outside SciRust ownership.
