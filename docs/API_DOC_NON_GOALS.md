# API documentation program non-goals

Issue #1431 does not aim to:

- rewrite SciRust's architecture merely to simplify documentation;
- claim all research APIs are production-ready;
- make every private helper part of the public lexicon;
- turn doctests into performance benchmarks or hardware qualification;
- duplicate rustdoc manually in large Markdown tables;
- hide deprecated/experimental APIs to improve coverage percentages;
- lower numerical/test/documentation standards to obtain green CI;
- replace implementation tests with prose.

Documentation review may trigger focused API/implementation improvements when a
verified contract defect is found. Those changes remain separately tested and
identified.
