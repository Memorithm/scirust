# Complete documentation architecture map

```text
README.md
  -> docs/README.md
       -> Quickstart / task guides
       -> Architecture / reference
       -> API_DOC_INDEX.md
            -> standards + conventions + review aids
            -> progress + evidence + findings
            -> lexicon v2 specification/roadmap
       -> API_LEXICON.md (generated discovery)
       -> API_EXAMPLES.md (generated example observations)
       -> LEXICON.md
            -> GLOSSARY.md (concepts)
            -> API_LEXICON.md (callables)

Rust source
  -> item/module rustdoc (canonical callable contracts)
  -> doctests (executable usage evidence)
  -> scripts/api-lexicon.py (source discovery)
  -> scripts/api-example-audit.py (example observation)
  -> future rustdoc/compiler reconciliation (effective reachability)
```

This structure separates semantics, navigation and evidence so that improving
one does not require hand-maintaining thousands of duplicated API entries.
