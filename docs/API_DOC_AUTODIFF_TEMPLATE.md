# Autodiff API documentation template

SciRust contains more than one autodiff/tensor stack. Public autodiff APIs must
therefore state which stack they belong to and must not imply interoperability
that does not exist.

Document:

- scalar, 2-D or N-D value model;
- tape/graph ownership and lifetime semantics;
- supported operations and differentiability restrictions;
- gradient accumulation/reset behavior;
- shape requirements;
- higher-order derivative support, if any;
- determinism/backend qualifications;
- errors or panics for foreign handles, missing gradients or invalid shapes.

Examples should include a small derivative with an analytic answer and a second
example covering composition, shape/error behavior or gradient reset. When
comparing derivatives numerically, state the expected analytic derivative rather
than only checking that a gradient exists.
