# SciRust Rustdoc conventions

These conventions make the public documentation searchable and comparable
without forcing every scientific domain into identical prose.

## First paragraph

Start with a declarative statement of the operation. Prefer "Computes…",
"Returns…", "Applies…", "Constructs…" or another precise verb. Do not start
with implementation history or marketing language.

## Mathematics and conventions

State the convention when multiple definitions are common: sample versus
population variance, interpolation type, FFT normalization, coordinate order,
matrix layout, inclusive/exclusive bounds, units, angle convention, or sign
convention. Use equations when they remove ambiguity; do not restate obvious
Rust syntax as prose.

## Floating-point behavior

For numerical public APIs, document relevant handling of empty inputs, `NaN`,
infinities, overflow/underflow and tolerances. Do not claim "numerically stable"
without naming the algorithm/property that supports the claim.

## Errors and panics

A `Result`-returning API should document caller-observable error conditions in
`# Errors`. A public operation that can panic because of caller-controlled input
needs `# Panics`; accidental panics found during review should normally be fixed
and regression-tested instead of normalized as API behavior.

## Safety

Every public unsafe callable requires `# Safety` with the invariants the caller
must uphold. Restating "caller must be safe" is not sufficient.

## Examples

Examples should use the public path a downstream crate would use. Assert a
meaningful result, invariant or error. Prefer small deterministic fixtures. Two
examples should normally cover different aspects of the contract.

## Claims

Distinguish implementation, validation and qualification:

- "implemented" means code exists;
- "tested" names the test/evidence that ran;
- "qualified on CUDA/WGPU/aarch64/etc." requires evidence on that path;
- "faster" or "more accurate" requires a stated benchmark/reference.

Historical evidence must remain tied to its revision and environment.
