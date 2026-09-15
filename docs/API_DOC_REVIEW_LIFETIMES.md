# Ownership and lifetime API review prompts

For public APIs where ownership matters, document/review:

- borrowed versus owned output;
- validity duration of views/handles;
- mutation invalidation;
- cloning/copy cost/semantics;
- ownership transfer to runtimes/devices/FFI;
- cleanup/drop behavior;
- aliasing/exclusivity for unsafe APIs.

Examples should make ownership behavior clear through ordinary public use rather
than relying on implementation fields that callers cannot access.
