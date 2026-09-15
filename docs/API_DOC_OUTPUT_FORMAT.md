# Generated API lexicon output requirements

Human Markdown output should be concise/searchable and link to authoritative
source/rustdoc. Machine JSON output should preserve complete observations and
provenance.

Both outputs should identify:

- schema/generator version where applicable;
- source revision or source fingerprints;
- filters/scope used to generate the view;
- distinction between observed source declaration and confirmed reachability;
- documentation/example observation status without presenting unexecuted fences
  as tested examples.

Filtered views are useful for work queues, but baseline operations must use the
full intended scope to avoid hiding debt outside the filter.
