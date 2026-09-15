# Miri qualification status

## 2026-09-14 / PR #1430 evidence

The general CI run for the pre-merge audit head completed every listed job
successfully except the targeted Miri job. The Miri failure was not a memory
safety diagnostic in the newly changed quantile code: `scirust-stats` reached
`discrete::tests::hypergeometric_matches_exact_6_of_49` and failed its numerical
normalization assertion `close(total, 1.0, 1e-13)`.

The same run's ordinary stable/nightly workspace tests passed. This means the
Miri result is a real qualification discrepancy to investigate, but it does not
by itself establish undefined behavior or a production regression. The Miri
execution model can change floating-point behavior and is substantially slower;
the correct follow-up is to reproduce the exact hypergeometric sum, compare the
computed residual against a high-precision/exact oracle, and decide whether the
implementation or the test tolerance is wrong. The tolerance must not simply be
relaxed to make CI green.

The run also emitted a Miri provenance warning for the empty `AlignedVec` path in
`scirust-arena`: `alignment as *mut T` constructs a non-null aligned sentinel by
integer-to-pointer cast. The tests passed, but Miri explicitly warns that this
can hide pointer bugs unless Strict Provenance APIs are used. This should be
migrated and then exercised with strict-provenance Miri before claiming that
path fully qualified.

These two findings are tracked as audit follow-ups; neither is silently waived
by the documentation program.
