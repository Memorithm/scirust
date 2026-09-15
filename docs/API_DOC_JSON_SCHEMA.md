# Target machine-readable API lexicon semantics

The v2 JSON index should use explicit states rather than ambiguous booleans when
evidence can be incomplete.

Suggested per-entry fields:

```json
{
  "package": "...",
  "domains": [],
  "source": "...",
  "line": 1,
  "symbol": "...",
  "signature": null,
  "summary": "...",
  "reachability": "unknown",
  "ordinary_examples_observed": 0,
  "examples_execution": "not_checked",
  "sections": {
    "errors": "unknown",
    "panics": "unknown",
    "safety": "unknown"
  },
  "source_sha256": "..."
}
```

Use schema versions and validate enum values. `unknown` is materially different
from `false`: the former means evidence has not established the property. Avoid
encoding semantic requiredness until the analyzer can derive it reliably.
