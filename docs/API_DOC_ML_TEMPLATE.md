# Machine-learning API documentation template

For public ML/model/layer APIs, document:

- expected input/output shapes and batch convention;
- parameter shapes, initialization and dtype;
- training versus inference behavior;
- mutable state such as running statistics, caches or optimizer state;
- randomness/seed behavior;
- supported backend/precision paths;
- loss/reduction convention where applicable;
- errors for shape, dtype, invalid hyperparameters or unsupported execution.

Examples should use tiny deterministic tensors/models and assert a shape/value or
state transition. A second example should cover training/inference distinction,
validation or deterministic seeding. Accuracy/performance claims require
separate dataset/benchmark evidence.
