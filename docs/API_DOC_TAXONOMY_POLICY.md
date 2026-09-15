# Capability taxonomy maintenance policy

`docs/api-domains.json` groups packages for navigation. Domains are not maturity
levels and do not imply that every package exposes equivalent capabilities.

When adding/moving a package:

- choose domains based on implemented public responsibility;
- allow multiple domains when the package genuinely spans them;
- avoid a generic catch-all merely to satisfy coverage;
- keep domain descriptions neutral and capability-oriented;
- validate package names against Cargo metadata;
- regenerate the capability index after taxonomy changes.

The final lexicon may add module/item-level tags, but package-domain taxonomy
remains a coarse navigation layer rather than a substitute for callable semantics.
