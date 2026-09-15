# FFI API documentation template

For public FFI/native-boundary APIs, document:

- ABI and linked library/runtime requirements;
- ownership and lifetime transfer across the boundary;
- pointer validity/alignment and buffer length units;
- thread-safety and synchronization;
- error-code/exception translation;
- string encoding and struct representation;
- version compatibility;
- `# Safety` invariants for unsafe calls.

Examples should prefer safe wrappers. A second example can demonstrate error
translation or capability detection without invoking undefined behavior. A
successful compile/link is not evidence that every native/device runtime path
executed.
