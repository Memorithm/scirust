# Parser/deserializer API review prompts

For public parsers/deserializers, document/review:

- accepted grammar/format/version;
- encoding/endian conventions;
- size/depth/count limits before allocation/recursion;
- duplicate/unknown field handling;
- trailing data behavior;
- malformed/truncated/non-finite value handling;
- typed error location/context;
- deterministic canonicalization if any.

Examples should include a minimal valid input and a malformed/unsupported case.
Fuzzing complements these examples but does not replace the documented format
contract.
