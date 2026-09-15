# Simulation API documentation template

For public simulation APIs, document:

- state vector and units;
- time units, interval and step-size convention;
- integrator/model assumptions;
- initial-condition requirements;
- observer/event/stopping semantics;
- conservation/stability properties that are actually guaranteed or tested;
- deterministic versus stochastic behavior and seed handling;
- errors for malformed state, non-finite evolution or invalid parameters.

Examples should include a small model with an analytic or conserved property and
a second example covering validation, stopping/event behavior or a different
integrator configuration. Avoid claiming physical accuracy from numerical
stability alone.
