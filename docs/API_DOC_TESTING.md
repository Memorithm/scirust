# Testing documentation changes

A documentation-only diff can still break builds, examples or user contracts.
Use layered checks.

## Syntax/navigation

Build rustdoc with warnings denied to catch invalid links and documentation
syntax problems.

## Examples

Run `cargo test -p <package> --doc --locked` for the changed package. A code
fence counted as ordinary must not be claimed executed until this passes on the
exact head.

## Package behavior

Run package unit/integration tests and Clippy when the review changes code or
exposes behavior-dependent examples.

## Repository behavior

Use the normal workspace CI before merge. Feature/backends affected by the
changed contract require their applicable checks.

## Negative controls

When documentation review exposes a real bug, add a regression that fails on the
pinned old implementation and passes after the fix when practical. Do not
manufacture negative controls for prose-only improvements.
