# File-format API review prompts

For public file-format APIs, document/review:

- magic/version/schema;
- encoding/endian/alignment;
- required/optional fields;
- size/count bounds;
- checksums/integrity;
- unknown-version/field behavior;
- deterministic writing;
- malformed/truncated input errors;
- compatibility guarantees.

Examples should use a tiny in-memory/temp-file round trip and a distinct invalid
header/version/truncation case.
