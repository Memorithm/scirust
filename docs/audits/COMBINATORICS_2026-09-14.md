# Combinatorics follow-up: 2026-09-14

## Observed blocker

PR #1430 head `e5fee8165ccc6ad9f4b1e6f2a50845ab99c34c18` had 24 successful
jobs and one failed job in general CI run `34896662563`. Miri job
`104152684876` reported a runtime failure in
`discrete::tests::hypergeometric_matches_exact_6_of_49` at the unchanged
`close(total, 1.0, 1e-13)` assertion. The new `describe` regressions passed
in that same Miri run. This was not a compiler error or a missing runner.

Sources: [failed job](https://github.com/Memorithm/scirust/actions/runs/34896662563/job/104152684876),
[original combinatorics](https://github.com/Memorithm/scirust/blob/e5fee8165ccc6ad9f4b1e6f2a50845ab99c34c18/scirust-stats/src/comb.rs),
[original test](https://github.com/Memorithm/scirust/blob/e5fee8165ccc6ad9f4b1e6f2a50845ab99c34c18/scirust-stats/src/discrete.rs#L1766-L1784).

## Implementation

`ln_binomial` previously subtracted three approximate log-factorials for all
valid inputs. It now first uses the existing checked `u128` coefficient.
Representable counts require one rounded conversion and one logarithm,
without subtracting large log-factorials. Count-one endpoints return exactly
zero. Coefficients exceeding `u128` retain the existing log-gamma fallback;
this change does **not** establish uniform error bounds for that fallback.

The same source review exposed two independent boundary defects:

- `permutations(u64::MAX, 0)` formed `n + 1` before constructing an empty
  product. It now returns the empty-product identity before range arithmetic.
- `multichoose` used unchecked `n + k - 1`. It now handles zero selections
  first, then checks `n + (k - 1)`, preserving `multichoose(u64::MAX, 1)`.
  An unrepresentable binomial parameter returns `None`, even when a wider
  parameter representation might allow the resulting count to fit `u128`.
  That API limitation is explicit rather than hidden by integer wrapping.

Five new unit tests cover exact-count logarithms, large-n/small-k cancellation,
the log-gamma fallback and the two integer boundaries. Each of the six public
functions in `comb` has two ordinary Rustdoc examples. Existing statistical
assertions and tolerances are unchanged; no tests are ignored or removed.

## Validation boundary

The editor has Python but no Rust toolchain. Local Python independently
computed `ln(C(1000,500))` with 80 decimal digits and evaluated the revised
6/49 formula with binary64 arithmetic (mass `1.0000000000000007`). These
calculations are diagnostic oracles, **not Rust or Miri execution evidence**.

The dedicated workflow runs actual debug/release tests, doctests, and Miri
with normal floating-point perturbations (seeds 0, 1, 7, 42 for the previously
failing test). A pinned original-source negative control must compile and fail
one named runtime assertion for `ln_binomial(u64::MAX, 1)`. Compilation
failures cannot satisfy that control. Hosted exact-head results remain the
merge authority. A green focused workflow does not waive general CI.

Relevant language semantics: [Rust f64 logarithm precision](https://doc.rust-lang.org/std/primitive.f64.html#method.ln)
and [Miri floating-point behavior](https://github.com/rust-lang/miri).
No downstream adoption, speedup, universal numerical accuracy or GPU execution
is claimed.
