# Stable documentation entry points

Keep these paths stable as the #1431 program evolves:

- `README.md` — repository overview and primary documentation links;
- `docs/README.md` — documentation hub;
- `docs/API_LEXICON.md` — generated callable discovery;
- `docs/API_EXAMPLES.md` — example inventory;
- `docs/LEXICON.md` / `docs/GLOSSARY.md` — terminology navigation;
- `docs/API_DOC_INDEX.md` — documentation-program navigation;
- package/module rustdoc — authoritative callable contracts.

New tooling versions should update content behind these entry points where
possible rather than forcing users to discover a new documentation hierarchy on
every iteration.
