# Dependency-related API documentation policy

Public documentation should expose dependency behavior only when callers need it
for correct use: feature requirements, external runtime/library prerequisites,
format/protocol compatibility or important semantic conventions.

Do not couple ordinary examples to incidental dependency internals. Prefer
SciRust public paths/types. When a dependency upgrade changes public behavior,
update the contract/examples and qualification evidence in the same integration
slice.
