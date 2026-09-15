# I/O API review prompts

For public I/O APIs, inspect/document:

- accepted path/stream/buffer forms;
- encoding/format/version;
- maximum lengths/allocation checks for untrusted input;
- atomicity and partial write/read behavior;
- overwrite behavior;
- error propagation and context;
- deterministic output ordering;
- cleanup/flush semantics;
- security implications of path traversal or untrusted metadata where relevant.

Use temporary/in-memory fixtures in examples when possible.
