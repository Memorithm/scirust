# Glossary expansion seeds

As module slices are reviewed, extend `GLOSSARY.md` only for recurring terms.
Candidate domains include:

- autodiff: tape, primal, adjoint, reverse mode, dual number;
- tensors: shape, stride, layout, logical dtype, storage dtype, representation;
- optimization: objective, gradient, step, convergence criterion;
- probability: PMF/PDF/CDF, quantile, support, sample/population statistic;
- solvers: residual, tolerance, preconditioner, breakdown;
- signals: sampling rate, window, normalization, phase;
- simulation/control: state, timestep, integrator, observation, covariance;
- execution: backend, feature gate, fallback, qualification, reproducibility.

Do not add definitions from memory alone when SciRust uses a specific convention;
confirm the implementation/docs of the reviewed module first.
