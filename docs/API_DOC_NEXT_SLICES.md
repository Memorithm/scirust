# Next `scirust-core` documentation slices

After `compute_backend` is merged and re-based on current `master`, review
coherent modules in roughly this order, adjusting when actual source/test audit
shows a higher-value boundary:

1. checkpoint/persistence contracts;
2. compute capability/backend selection contracts;
3. tensor shape/view/tiling boundaries;
4. reverse and N-D autodiff public surfaces;
5. quantization/mixed-precision public surfaces;
6. neural-network layer/model APIs;
7. remaining numerical/convenience modules.

Each slice reads implementation and tests first, adds meaningful item Rustdoc and
examples, fixes verified defects with regressions when discovered, and expands
policy coverage only after exact-head doctest/rustdoc evidence is green.

After `scirust-core`, continue the large debt packages recorded in #1431 rather
than declaring the repository documentation complete from the core slice alone.
