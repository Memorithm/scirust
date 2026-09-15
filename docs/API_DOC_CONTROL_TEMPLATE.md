# Control and estimation API documentation template

For public control/estimation APIs, document:

- state, input and measurement dimensions;
- discrete/continuous-time convention and sample period units;
- matrix ordering and model equations;
- covariance/noise assumptions;
- initialization and mutable state;
- stability/observability/controllability assumptions actually required;
- saturation/constraint behavior;
- numerical breakdown and invalid-dimension errors.

Examples should include a small model with a hand-verifiable update and a second
validation, steady-state or constraint case. Do not claim closed-loop stability
unless the API/test evidence establishes it under stated assumptions.
