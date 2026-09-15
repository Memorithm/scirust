# Minimum lexicon fields

Until compiler-derived reconciliation is complete, every source-level callable
entry should at least preserve:

- package;
- source path and line;
- symbol;
- first adjacent Rustdoc summary or explicit absence;
- capability-domain membership;
- source revision/fingerprint in generated evidence artifacts.

As v2 lands, add signature, example observations, contract-section observations
and reachability state. Unknown values should remain explicit (`unknown`/null)
rather than guessed from naming conventions.

The Markdown view may omit low-level fields for readability, but the JSON view
should retain enough provenance to reproduce the metric and locate the source.
