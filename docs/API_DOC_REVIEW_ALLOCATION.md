# Allocation and resource review prompts

For public APIs that allocate or scale with input, document/review:

- output/scratch allocation size;
- checked size products and conversion overflow;
- input-controlled allocation from parsers/deserializers;
- reuse/in-place alternatives;
- device/host memory distinction;
- behavior for empty/huge metadata-only inputs;
- cleanup/ownership after errors.

Avoid examples that prove a size contract by actually allocating impractical
memory when metadata-only validation can exercise the same boundary.
