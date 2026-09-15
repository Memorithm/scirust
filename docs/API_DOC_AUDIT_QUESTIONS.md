# Questions for each public callable review

- What mathematical/system operation does this actually implement?
- What inputs are valid, including shape/unit/dtype/range?
- What happens for empty, extreme or non-finite input?
- What does the return value mean and what invariants hold?
- Which errors can the caller cause/observe?
- Can caller input trigger a panic? Should that be fixed?
- Does unsafe code impose caller invariants?
- Does behavior differ by feature/backend/target?
- What determinism/reproducibility is actually established?
- What is the smallest meaningful nominal example?
- What distinct second example teaches a boundary/error/alternative?
- Which tests/reference support these statements?
- Is any existing documentation claim stronger than the evidence?
