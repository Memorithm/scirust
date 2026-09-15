# Human-facing API lexicon layout

The Markdown lexicon should optimize discovery rather than dump every metadata
field into an unreadable table.

Recommended navigation hierarchy:

```text
capability domain
  -> package
     -> module/source
        -> callable
```

For each callable show symbol/signature, concise summary, documentation/example
status and source/rustdoc link. Put detailed provenance, section flags and hashes
in the JSON index.

Support filters/search by package, domain, symbol/summary text, missing docs and
missing reviewed examples. Generated pages should state source-level discovery
limits and the revision represented.
