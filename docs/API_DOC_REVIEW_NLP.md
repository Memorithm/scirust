# NLP/tokenization API review prompts

For public NLP/tokenization APIs, document/review:

- input encoding/Unicode normalization;
- token/vocabulary identity and unknown-token behavior;
- byte/character/token offset convention;
- truncation/padding/special-token behavior;
- deterministic ordering;
- model/vocabulary format/version;
- malformed input/errors;
- allocation/length limits for untrusted text.

Examples should use short fixed text with exact tokens/offsets and a distinct
unknown/Unicode/boundary case.
