# Technical claim policy for API documentation

Public documentation must not overstate what code or CI establishes.

- Do not say "GPU supported" when only a feature compiles; name WGPU/CUDA and
  whether execution actually ran.
- Do not say "safe" merely because Rust code compiles; state the checked
  invariant/evidence.
- Do not say "numerically stable" without naming the relevant algorithm or
  tested property.
- Do not say "deterministic" across backends unless cross-backend evidence
  supports it.
- Do not say "faster", "more accurate" or "state of the art" without a
  reproducible comparison.
- Do not turn a research hypothesis or roadmap item into an implemented API
  claim.
- Do not treat a representable tensor encoding as an executable kernel path.

Use `API_DOC_EVIDENCE.md` to state the available evidence level. Conservative,
precise documentation is preferable to broad claims that cannot be reproduced.
