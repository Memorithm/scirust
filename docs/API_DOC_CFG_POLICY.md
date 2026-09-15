# Feature/cfg API documentation policy

Public APIs gated by Cargo features or target `cfg` conditions must state the
condition under which they exist and the capability it enables.

The final compiler-derived lexicon should record feature/target reachability
rather than flattening all configurations into one unconditional surface.
Examples that require a feature should be executed by a doctest/check job with
that feature when practical; otherwise their qualification must be explicit.

A feature compiling on hosted CI is not evidence that hardware-specific runtime
behavior executed. Keep compile, software-adapter and physical-device evidence
separate.
