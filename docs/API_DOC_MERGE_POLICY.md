# API documentation merge policy

Documentation slices follow the same exact-head discipline as code changes.

- Inspect all applicable checks after the final branch update.
- Fix formatting/rustdoc/doctest/normal CI failures before merge.
- Do not merge because a docs-specific job is green while normal CI is red.
- Do not reuse green evidence from an earlier head after changing examples or
  source.
- After merge, start the next slice from current `master` and regenerate global
  observations as needed.

This keeps progress incremental without accumulating stale stacked branches.
