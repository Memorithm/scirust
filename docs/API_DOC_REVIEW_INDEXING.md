# Indexing API review prompts

For public indexing/slicing APIs, document/review:

- zero- versus one-based indexing;
- inclusive/exclusive bounds;
- negative index support if any;
- row/column/axis order;
- empty ranges;
- out-of-bounds behavior (error/None/panic);
- checked offset/stride arithmetic;
- view versus copy semantics.

Examples should include a nominal slice/index and a boundary/out-of-range case
using the documented failure mode.
