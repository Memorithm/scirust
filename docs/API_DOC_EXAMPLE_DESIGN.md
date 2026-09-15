# Designing high-information API examples

A documentation example should answer a caller question, not merely increase a
counter.

Good pairs include:

- normal result + typed error;
- scalar/small input + multidimensional/alternative shape;
- default configuration + explicit configuration;
- deterministic seed + invalid parameter;
- training mode + inference mode;
- dense executable representation + explicitly unsupported representation;
- serialization round trip + malformed/version rejection.

Keep each example minimal enough that the assertion is understandable without
reading unrelated setup. If two examples assert the same property with different
numbers, they are duplicates for policy purposes.
