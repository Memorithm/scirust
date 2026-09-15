# Attention API review prompts

For attention-related public APIs, document/review:

- tensor axis order and Q/K/V dimensions;
- Q-head versus KV-head grouping rules;
- Q length versus KV length/rectangular support;
- head/value dimension relation;
- causal masking convention;
- logical/storage dtype and representation support;
- executable versus describable representation paths;
- shape narrowing/storage overflow validation;
- backend/kernel mapping and explicit unsupported cases;
- workload/fingerprint identity fields.

Examples should include a valid small dense/GQA intent and a distinct invalid or
non-executable representation/shape case. Avoid lossy adapters that erase Q/KV
length or batch/head distinctions without an explicit rejection contract.
