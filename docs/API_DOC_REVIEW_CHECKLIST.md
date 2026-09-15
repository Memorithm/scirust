# Public API documentation review checklist

Use this checklist for each module slice in issue #1431.

- [ ] Identify the externally reachable API surface; do not equate textual
      `pub` with proven reachability.
- [ ] Read implementation and existing tests before writing semantics.
- [ ] State input domains, shapes, units and ordering where relevant.
- [ ] State output meaning and invariants.
- [ ] Document non-finite floating-point behavior when applicable.
- [ ] Add `# Errors` for caller-observable `Result` failures.
- [ ] Add `# Panics` for caller-triggerable panics; prefer fixing accidental
      panics rather than merely documenting them.
- [ ] Add `# Safety` to every public unsafe callable.
- [ ] State backend, determinism and reproducibility qualifications when they
      affect correct use.
- [ ] Provide two complementary ordinary runnable examples for each non-trivial
      callable covered by policy.
- [ ] Avoid examples that merely construct and discard a value; assert a
      semantic property or error contract.
- [ ] Execute the relevant doctests on the exact PR head.
- [ ] Build rustdoc with warnings denied.
- [ ] Run the source lexicon/example audit and confirm documentation debt does
      not regress.
- [ ] Keep implementation changes separate unless documentation review exposes
      a verified bug; bug fixes require regression tests.
- [ ] Record hardware/network-only exceptions explicitly in
      `API_DOC_EXCEPTIONS.md`.
- [ ] Update progress and generated lexicon artifacts only from the final source
      revision.
