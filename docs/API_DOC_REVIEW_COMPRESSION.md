# Compression and quantization API review prompts

Document/review:

- logical versus storage dtype;
- scale/zero-point/group/block parameterization;
- rounding/clamping convention;
- packed bit ordering/layout;
- reconstruction formula;
- exact storage accounting and padding;
- supported execution kernels versus representation-only support;
- error metrics/accuracy claims with evidence;
- invalid shape/parameter handling.

Examples should include a tiny encode/decode or storage-accounting case and a
boundary/unsupported representation case.
