# Reinforcement-learning API review prompts

For public RL APIs, document/review:

- observation/action spaces and shapes;
- reward definition and scaling;
- episode/reset/termination/truncation semantics;
- stochasticity and seed behavior;
- discount/return convention;
- mutable environment/agent state;
- invalid action/state handling;
- determinism/reproducibility scope.

Examples should use a tiny deterministic environment/transition and a distinct
reset/terminal/invalid-action case. Benchmark reward claims require a separate
training/evaluation protocol.
