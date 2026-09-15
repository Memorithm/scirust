# Integer and size arithmetic review prompts

For public APIs converting/combining sizes/indices, inspect/document:

- checked versus wrapping/saturating arithmetic;
- `usize`/fixed-width narrowing;
- signed/unsigned conversion;
- product/sum overflow for element/storage counts;
- zero divisors/modulo;
- platform pointer-width differences;
- byte versus element units.

Prefer checked conversion/errors at public boundaries. Use metadata-only tests to
exercise huge dimensions when allocating the represented payload would be
impractical.
