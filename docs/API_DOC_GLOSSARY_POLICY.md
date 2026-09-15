# Glossary maintenance policy

`GLOSSARY.md` defines recurring concepts whose meaning crosses crate/module
boundaries. It should not become a duplicate of every function name in the API
lexicon.

Add a glossary entry when a term:

- appears across multiple modules or guides;
- has a SciRust-specific convention that users must understand;
- distinguishes concepts commonly conflated (for example logical tensor versus
  physical representation, representable versus executable, or source
  observation versus execution evidence);
- is necessary to interpret qualification/reproducibility claims.

Function-specific formulas, parameter definitions and errors belong in Rustdoc.
Capability navigation belongs in the generated API lexicon. The glossary should
link to those surfaces when a concept has a canonical implementation.

Definitions must be neutral and implementation-grounded. Research conjectures
should be labelled as such rather than promoted to established terminology.
