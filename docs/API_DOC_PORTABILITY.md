# Portability documentation policy

When a public API's behavior/availability depends on OS, architecture, pointer
width, endianness, SIMD feature or backend, state the dependency.

Portable examples should be preferred for general Rustdoc. Architecture-specific
examples belong in qualified sections/jobs. Metadata-only tests are useful for
large dimension/pointer-width contracts because they avoid impractical payload
allocation.

A successful x86_64 hosted doctest does not establish identical behavior on
Windows/macOS/aarch64; stronger portability claims require the applicable normal
CI evidence.
