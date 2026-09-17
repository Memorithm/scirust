//! Boolean and finite-field operator primitives for hybrid operator learning.
//!
//! The module provides exact, deterministic operators over `F_2` and algebraic
//! normal form (ANF, also called Zhegalkin polynomials). These are not relaxed
//! floating approximations: evaluation uses XOR/AND semantics exactly.

use crate::error::{NeuralOperatorError, Result};

/// Structural complexity of a Boolean polynomial/operator surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BooleanComplexity {
    /// Number of Boolean inputs read by the representation.
    pub input_arity: usize,
    /// Number of ANF monomials retained.
    pub monomials: usize,
    /// Sum of monomial degrees (literal occurrences).
    pub literal_occurrences: usize,
    /// Largest monomial degree.
    pub max_degree: usize,
}

/// Compact bit-packed Boolean vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedBits {
    len: usize,
    words: Vec<u64>,
}

impl PackedBits {
    /// Pack Boolean values into little-indexed 64-bit words.
    pub fn from_bools(values: &[bool]) -> Self {
        let mut words = vec![0_u64; values.len().div_ceil(64)];
        for (index, value) in values.iter().copied().enumerate()
        {
            if value
            {
                words[index / 64] |= 1_u64 << (index % 64);
            }
        }
        Self {
            len: values.len(),
            words,
        }
    }

    /// Number of represented Boolean coordinates.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the vector has no coordinates.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read one bit, failing closed on an invalid coordinate.
    pub fn get(&self, index: usize) -> Result<bool> {
        if index >= self.len
        {
            return Err(NeuralOperatorError::BooleanIndexOutOfBounds {
                index,
                len: self.len,
            });
        }
        Ok(((self.words[index / 64] >> (index % 64)) & 1) != 0)
    }

    /// Expand to one `bool` per coordinate.
    pub fn to_bools(&self) -> Vec<bool> {
        (0..self.len)
            .map(|index| ((self.words[index / 64] >> (index % 64)) & 1) != 0)
            .collect()
    }

    fn words(&self) -> &[u64] {
        &self.words
    }
}

/// Exact affine operator `y = A x XOR b` over `F_2`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct F2LinearOperator {
    input_dim: usize,
    rows: Vec<PackedBits>,
    bias: PackedBits,
}

impl F2LinearOperator {
    /// Construct from a dense Boolean matrix in output-major row order.
    pub fn from_dense(weights: &[Vec<bool>], bias: &[bool]) -> Result<Self> {
        if weights.is_empty()
        {
            return Err(NeuralOperatorError::Empty {
                what: "F2 weight rows",
            });
        }
        let input_dim = weights[0].len();
        if input_dim == 0
        {
            return Err(NeuralOperatorError::Empty {
                what: "F2 input dimension",
            });
        }
        if bias.len() != weights.len()
        {
            return Err(NeuralOperatorError::BooleanShapeMismatch {
                what: "F2 bias",
                expected: weights.len(),
                got: bias.len(),
            });
        }
        let mut rows = Vec::with_capacity(weights.len());
        for row in weights
        {
            if row.len() != input_dim
            {
                return Err(NeuralOperatorError::BooleanShapeMismatch {
                    what: "F2 weight row",
                    expected: input_dim,
                    got: row.len(),
                });
            }
            rows.push(PackedBits::from_bools(row));
        }
        Ok(Self {
            input_dim,
            rows,
            bias: PackedBits::from_bools(bias),
        })
    }

    /// Number of Boolean inputs.
    pub const fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Number of Boolean outputs.
    pub fn output_dim(&self) -> usize {
        self.rows.len()
    }

    /// Apply the affine map exactly over `F_2`.
    pub fn apply(&self, input: &[bool]) -> Result<Vec<bool>> {
        if input.len() != self.input_dim
        {
            return Err(NeuralOperatorError::BooleanShapeMismatch {
                what: "F2 input",
                expected: self.input_dim,
                got: input.len(),
            });
        }
        let packed = PackedBits::from_bools(input);
        self.apply_packed(&packed).map(|value| value.to_bools())
    }

    /// Apply the affine map to an already packed input.
    pub fn apply_packed(&self, input: &PackedBits) -> Result<PackedBits> {
        if input.len() != self.input_dim
        {
            return Err(NeuralOperatorError::BooleanShapeMismatch {
                what: "packed F2 input",
                expected: self.input_dim,
                got: input.len(),
            });
        }
        let mut output = Vec::with_capacity(self.rows.len());
        for (row_index, row) in self.rows.iter().enumerate()
        {
            let parity = row
                .words()
                .iter()
                .zip(input.words())
                .fold(0_u32, |acc, (left, right)| {
                    acc ^ ((left & right).count_ones() & 1)
                });
            output.push((parity != 0) ^ self.bias.get(row_index)?);
        }
        Ok(PackedBits::from_bools(&output))
    }
}

/// One exact Boolean function represented in algebraic normal form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnfPolynomial {
    variables: usize,
    monomials: Vec<u64>,
}

impl AnfPolynomial {
    /// Build an ANF polynomial from explicit monomial bitmasks.
    ///
    /// Bit `i` in a monomial mask means variable `x_i` participates in that
    /// product; mask zero is the constant-one monomial. Duplicate monomials
    /// cancel in `F_2`.
    pub fn from_monomials(variables: usize, monomials: &[u64]) -> Result<Self> {
        if variables > 63
        {
            return Err(NeuralOperatorError::TooManyBooleanVariables {
                variables,
                maximum: 63,
            });
        }
        let valid_mask = if variables == 0
        {
            0
        }
        else
        {
            (1_u64 << variables) - 1
        };
        if let Some(&mask) = monomials.iter().find(|&&mask| mask & !valid_mask != 0)
        {
            return Err(NeuralOperatorError::InvalidMonomialMask { mask, variables });
        }
        let mut canonical = monomials.to_vec();
        canonical.sort_unstable();
        let mut reduced = Vec::with_capacity(canonical.len());
        let mut cursor = 0;
        while cursor < canonical.len()
        {
            let value = canonical[cursor];
            let mut count = 1_usize;
            cursor += 1;
            while cursor < canonical.len() && canonical[cursor] == value
            {
                count += 1;
                cursor += 1;
            }
            if count % 2 == 1
            {
                reduced.push(value);
            }
        }
        Ok(Self {
            variables,
            monomials: reduced,
        })
    }

    /// Recover the exact ANF coefficients from a complete truth table using the
    /// Boolean Möbius transform.
    pub fn from_truth_table(variables: usize, truth_table: &[bool]) -> Result<Self> {
        if variables > 20
        {
            return Err(NeuralOperatorError::TooManyBooleanVariables {
                variables,
                maximum: 20,
            });
        }
        let expected = 1_usize << variables;
        if truth_table.len() != expected
        {
            return Err(NeuralOperatorError::TruthTableLength {
                variables,
                expected,
                got: truth_table.len(),
            });
        }
        let mut coefficients = truth_table.to_vec();
        for bit_index in 0..variables
        {
            let bit = 1_usize << bit_index;
            for mask in 0..expected
            {
                if mask & bit != 0
                {
                    coefficients[mask] ^= coefficients[mask ^ bit];
                }
            }
        }
        let monomials = coefficients
            .iter()
            .enumerate()
            .filter_map(|(mask, coefficient)| coefficient.then_some(mask as u64))
            .collect::<Vec<_>>();
        Self::from_monomials(variables, &monomials)
    }

    /// Evaluate the polynomial exactly on a Boolean input vector.
    pub fn evaluate(&self, input: &[bool]) -> Result<bool> {
        if input.len() != self.variables
        {
            return Err(NeuralOperatorError::BooleanShapeMismatch {
                what: "ANF input",
                expected: self.variables,
                got: input.len(),
            });
        }
        let mut packed = 0_u64;
        for (index, value) in input.iter().copied().enumerate()
        {
            if value
            {
                packed |= 1_u64 << index;
            }
        }
        Ok(self.monomials.iter().fold(false, |value, monomial| {
            value ^ ((packed & monomial) == *monomial)
        }))
    }

    /// Number of input variables.
    pub const fn variables(&self) -> usize {
        self.variables
    }

    /// Canonical ANF monomial masks after `F_2` cancellation.
    pub fn monomials(&self) -> &[u64] {
        &self.monomials
    }

    /// Structural complexity of this ANF representation.
    pub fn complexity(&self) -> BooleanComplexity {
        let literal_occurrences = self
            .monomials
            .iter()
            .map(|mask| mask.count_ones() as usize)
            .sum();
        let max_degree = self
            .monomials
            .iter()
            .map(|mask| mask.count_ones() as usize)
            .max()
            .unwrap_or(0);
        BooleanComplexity {
            input_arity: self.variables,
            monomials: self.monomials.len(),
            literal_occurrences,
            max_degree,
        }
    }
}

/// Multi-output Boolean operator composed of one ANF polynomial per output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnfOperator {
    variables: usize,
    outputs: Vec<AnfPolynomial>,
}

impl AnfOperator {
    /// Create a multi-output ANF operator with a common input arity.
    pub fn new(variables: usize, outputs: Vec<AnfPolynomial>) -> Result<Self> {
        if outputs.is_empty()
        {
            return Err(NeuralOperatorError::Empty {
                what: "ANF outputs",
            });
        }
        if let Some(output) = outputs
            .iter()
            .find(|output| output.variables() != variables)
        {
            return Err(NeuralOperatorError::BooleanShapeMismatch {
                what: "ANF output arity",
                expected: variables,
                got: output.variables(),
            });
        }
        Ok(Self { variables, outputs })
    }

    /// Evaluate every output coordinate exactly.
    pub fn apply(&self, input: &[bool]) -> Result<Vec<bool>> {
        self.outputs
            .iter()
            .map(|output| output.evaluate(input))
            .collect()
    }

    /// Number of input variables.
    pub const fn input_dim(&self) -> usize {
        self.variables
    }

    /// Number of Boolean outputs.
    pub fn output_dim(&self) -> usize {
        self.outputs.len()
    }

    /// Aggregate ANF representation complexity across all outputs.
    pub fn complexity(&self) -> BooleanComplexity {
        let per_output = self
            .outputs
            .iter()
            .map(AnfPolynomial::complexity)
            .collect::<Vec<_>>();
        BooleanComplexity {
            input_arity: self.variables,
            monomials: per_output.iter().map(|item| item.monomials).sum(),
            literal_occurrences: per_output.iter().map(|item| item.literal_occurrences).sum(),
            max_degree: per_output
                .iter()
                .map(|item| item.max_degree)
                .max()
                .unwrap_or(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f2_linear_operator_matches_hand_parity() {
        let operator = F2LinearOperator::from_dense(
            &[vec![true, false, true], vec![true, true, false]],
            &[false, true],
        )
        .unwrap();
        assert_eq!(
            operator.apply(&[true, true, false]).unwrap(),
            vec![true, true]
        );
    }

    #[test]
    fn mobius_transform_recovers_every_three_variable_function() {
        for function in 0_u16..256
        {
            let truth = (0..8)
                .map(|index| ((function >> index) & 1) != 0)
                .collect::<Vec<_>>();
            let polynomial = AnfPolynomial::from_truth_table(3, &truth).unwrap();
            for (index, expected) in truth.iter().copied().enumerate()
            {
                let input = (0..3)
                    .map(|bit| ((index >> bit) & 1) != 0)
                    .collect::<Vec<_>>();
                assert_eq!(polynomial.evaluate(&input).unwrap(), expected);
            }
        }
    }

    #[test]
    fn zhegalkin_x0_xor_x0x1_is_exact() {
        let polynomial = AnfPolynomial::from_monomials(2, &[0b01, 0b11]).unwrap();
        assert!(!polynomial.evaluate(&[false, false]).unwrap());
        assert!(polynomial.evaluate(&[true, false]).unwrap());
        assert!(!polynomial.evaluate(&[true, true]).unwrap());
        assert_eq!(
            polynomial.complexity(),
            BooleanComplexity {
                input_arity: 2,
                monomials: 2,
                literal_occurrences: 3,
                max_degree: 2,
            }
        );
    }

    #[test]
    fn duplicate_anf_terms_cancel() {
        let polynomial = AnfPolynomial::from_monomials(2, &[1, 1, 2]).unwrap();
        assert_eq!(polynomial.monomials(), &[2]);
    }
}
