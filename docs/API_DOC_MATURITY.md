# Capability maturity labels

When maturity matters, use descriptive states rather than implying that every
public API has equal readiness:

- **research skeleton** — contract/data structure exists but core execution or
  validation remains intentionally incomplete;
- **reference implementation** — executable correctness-oriented path exists;
- **tested implementation** — named tests execute the stated path;
- **backend-qualified** — named backend/target execution evidence exists;
- **production-hardened** — use only when operational/security/compatibility
  evidence actually supports the term.

These labels are documentation vocabulary, not automatic promotion stages. A
module can be mature in one backend/configuration and experimental in another.
