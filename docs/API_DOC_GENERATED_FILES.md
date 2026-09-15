# Generated documentation artifact rules

Generated API inventories are useful only when their provenance is clear.

## Generated

The source-level API lexicon, capability index, JSON callable index, example
observations and aggregate metrics are generated from code/taxonomy by scripts.
Do not hand-edit generated rows to conceal a parser problem; fix the generator or
source documentation and regenerate.

## Hand-reviewed

The glossary, documentation standards, domain review aids, task guides and audit
interpretations are semantic documents and remain hand-reviewed.

## Provenance

Generated JSON should include schema version and source revision/fingerprints
where practical. A generated artifact copied from another revision must not be
presented as current.

## CI

CI may compare generated output with checked-in artifacts, enforce monotonic
baselines and execute examples. It should not synthesize semantic Rustdoc prose
as an acceptance mechanism.
