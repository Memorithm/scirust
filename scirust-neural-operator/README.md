# scirust-neural-operator

Native Rust operator learning for SciRust.

This crate is Memorithm/SciRust's independent implementation of the core ideas
needed for neural operators. It does **not** embed or wrap the Python
`neuraloperator` package.

Current foundation:

- trainable 1-D Fourier Neural Operator through SciRust N-D autodiff;
- trainable 2-D FNO with canonical conjugate Fourier-mode representatives and exact N-D autograd;
- exact `F_2` affine operators and bit-packed Boolean vectors;
- exact ANF/Zhegalkin operators with truth-table Möbius synthesis;
- complete bounded sparse-ANF synthesis over unique Development assignments, refusing candidate-budget truncation, with a distinct post-selection Validation type;
- ordered Boolean routing for surrogate/exact/verify/abstain execution;
- hybrid executor that invokes the learned and authoritative paths according to the route and falls back to the exact output when verification exceeds tolerance;
- Boolean early-dispatch of Fourier modes before spectral channel mixing, with explicit arithmetic-work accounting;
- existing SciRust FNO / DeepONet / PINN primitives exposed from one package;
- deterministic portable FFT spectral derivatives and 1-D/2-D Laplacians;
- operator datasets with strict shape/finite-value validation;
- per-channel normalization;
- absolute and relative Lp losses;
- exact-vs-surrogate accuracy metrics;
- training-amortisation / break-even metrics.

The reference-solver rule is deliberate: a learned operator is a surrogate. It
must be qualified against an exact or otherwise authoritative solver, including
out-of-distribution and conservation/invariant checks, before it is used as an
accelerator.

The architecture now has continuous, Boolean and hybrid rails from the foundation.
Planned layers include N-D trainable spectral convolution, tensor-factorized
spectral weights, geometry-aware operators, physics/dynamics-informed losses,
uncertainty/OOD gating, multi-fidelity active learning, checkpointing and
backend-specific CPU/WGPU/CUDA inference.
