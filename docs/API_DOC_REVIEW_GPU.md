# GPU API review prompts

For public GPU APIs, document/review:

- backend (WGPU/CUDA/etc.) and feature requirements;
- device selection/availability;
- supported dtype/shape/layout;
- host-device transfer/ownership;
- synchronization/stream semantics;
- allocation/launch errors;
- fallback behavior;
- numerical/reference differences;
- compile/software-adapter/physical-device evidence level.

Examples should use portable capability/contract paths when possible. Do not
mark a GPU example ordinary/executed if hosted CI cannot run its required device;
use an explicit qualification/exception with separate hardware evidence.
