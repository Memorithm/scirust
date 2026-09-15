# Data/frame API documentation template

For public data/frame APIs, document:

- row/column orientation and ordering;
- column dtype/null/non-finite policy;
- schema requirements and name collision behavior;
- copy versus view/borrow semantics;
- sort/group/join stability and key semantics;
- CSV/text quoting, encoding and malformed-input behavior where applicable;
- errors for missing columns, incompatible lengths or unsupported dtypes.

Examples should include a small table with an exact expected result and a second
projection/filter/join/error case. Avoid examples whose correctness is only
"the call did not panic"; assert schema and values.
