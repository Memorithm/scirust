# Findings discovered during API documentation review

Documentation review doubles as contract audit. Record findings here until they
are fixed or moved to a dedicated issue/PR.

## Open

### Miri hypergeometric normalization discrepancy

The targeted Miri CI for the audit integration reached
`scirust-stats::discrete::tests::hypergeometric_matches_exact_6_of_49` and failed
the assertion that the probability sum is within `1e-13` of one. Ordinary
stable/nightly workspace tests passed. Reproduce and measure the residual before
changing implementation or tolerance.

### `AlignedVec` empty sentinel provenance warning

Miri warns about `alignment as *mut T` in `scirust-arena/src/aligned.rs` for an
empty aligned vector. Tests pass, but Miri states that integer-to-pointer casts
can hide pointer bugs. Replace with an appropriate Strict Provenance-compatible
construction and run strict-provenance Miri before considering the path fully
qualified.

## Resolved

The earlier attention-intent and descriptive-quantile findings were corrected
and merged through #1430/#1432. Their detailed evidence remains in the audit
archive and PR discussions.
