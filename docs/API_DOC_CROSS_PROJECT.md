# Cross-project reuse of documented SciRust primitives

When a SciRust API is considered for another Memorithm project, treat its
Rustdoc/tests as the source contract and inspect the consumer's current
conventions before integration.

Do not copy an API merely because it is now documented. Verify dependency
compatibility, numerical conventions, data shapes, feature/backend assumptions
and evidence requirements in the consumer. Prefer a shared dependency or a
small explicit adapter over duplicated implementations when architecture allows.

Cross-project adoption is a separate tested change; documentation completion in
SciRust is not evidence that TDI, KVLab, Forge or another repository already uses
the primitive.
