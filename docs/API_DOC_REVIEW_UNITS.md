# Unit annotation review prompts

For every public numeric parameter/result representing a physical or scaled
quantity, verify that units are explicit in Rustdoc unless encoded unambiguously
in a strong type.

Common ambiguity sources: seconds versus milliseconds, Hz versus rad/s, degrees
versus radians, bytes versus bits, element counts versus byte counts, normalized
versus raw coordinates, percentages versus probabilities.

Examples should reinforce the unit convention through values/assertions rather
than relying solely on parameter names.
