# Formula documentation conventions

Use equations when they disambiguate an API's mathematical contract.

- Define every symbol that is not obvious from the signature.
- State index ranges and shape correspondence for tensor/matrix formulas.
- State sample/population normalization and denominator conventions.
- State log/base, angle, sign and transform normalization conventions.
- State units for physical quantities.
- Distinguish exact mathematical formula from numerically rearranged
  implementation used to avoid overflow/cancellation.
- Link/reference standard definitions when a named convention is used.

An equation does not replace boundary/error documentation. IEEE-754 behavior,
invalid domains and implementation limitations still need prose/tests.
