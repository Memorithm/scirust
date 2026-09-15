# Error type review prompts

For public error enums/structs supporting reviewed callables, document/review:

- semantic meaning of each variant;
- caller action/recoverability when relevant;
- associated indices/values/units;
- source chaining/context;
- stability if downstream code is expected to pattern-match;
- whether sensitive data can appear in Display/Debug.

Examples should pattern-match variants when the variant is the stable contract.
Avoid depending on exact human-readable strings unless those strings are
explicitly stable API.
