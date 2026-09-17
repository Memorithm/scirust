# scirust-neural-operator

Native Rust operator learning for SciRust.

This crate is Memorithm/SciRust's independent implementation of the core ideas
needed for neural operators. It does **not** embed or wrap the Python
`neuraloperator` package.

Current foundation:

- trainable 1-D Fourier Neural Operator through SciRust N-D autodiff;
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

Planned layers include N-D trainable spectral convolution, tensor-factorized
spectral weights, geometry-aware operators, physics/dynamics-informed losses,
uncertainty/OOD gating, multi-fidelity active learning, checkpointing and
backend-specific CPU/WGPU/CUDA inference.
