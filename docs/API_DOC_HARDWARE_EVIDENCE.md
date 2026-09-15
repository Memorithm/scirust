# Hardware qualification vocabulary

Use explicit terms for backend evidence:

- **feature configured** — Cargo feature selected;
- **compiled** — code compiled for the target/feature;
- **software-adapter executed** — e.g. WGPU through a software Vulkan adapter;
- **physical-device executed** — operation ran on named compatible hardware;
- **cross-device qualified** — stated property tested across named devices.

A CUDA compile/lint job without a CUDA runtime is compile evidence only. A WGPU
lavapipe job is execution evidence for the software adapter, not a discrete GPU.
API documentation should link to revision-bound evidence for stronger claims.
