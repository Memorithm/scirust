# SciRust API documentation master plan

Tracking: #1431.

## Objective

Every externally reachable public callable has a precise Rustdoc contract and
one/two executable usage examples (two by default for non-trivial APIs), backed
by a qualitative generated lexicon and glossary/navigation.

## Workstreams

1. **Semantic slices** — review/fix/document APIs module by module.
2. **Example qualification** — observe fences, enforce reviewed scopes, execute
   doctests.
3. **Lexicon v2** — add signatures/sections/examples/provenance and compiler
   reachability without losing the existing source inventory.
4. **Navigation** — README/docs hub, glossary and task guides.
5. **Audit findings** — fix real bugs exposed by documentation review with
   regression evidence.

## Current state

Stats `describe`/`comb` example policy is merged via #1432. Attention/statistics
correctness audit is merged via #1430. Current core slice is `compute_backend`.
Miri hypergeometric normalization and aligned-empty-pointer provenance remain
explicit follow-ups.

## Completion

Use `API_DOC_DEFINITION_OF_DONE.md`; source-regex summary coverage alone is not
the completion criterion.
