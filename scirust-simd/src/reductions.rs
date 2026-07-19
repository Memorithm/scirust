//! SIMD reductions with explicit numerical semantics.
//!
//! The module separates three accumulation strategies:
//!
//! - [`sum_f32_fast`]: SIMD accumulation optimized for throughput;
//! - [`sum_f32_deterministic`]: fixed left-to-right scalar order;
//! - [`sum_f32_kahan`]: compensated scalar accumulation for improved accuracy.
//!
//! The deterministic and compensated variants deliberately avoid SIMD
//! reassociation because floating-point addition is not associative.

/// Result of an indexed reduction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndexedValue<T> {
    /// Index of the selected element.
    pub index: usize,
    /// Selected value.
    pub value: T,
}

/// Fast SIMD sum.
///
/// The result is deterministic for a fixed build, target and SIMD width, but it
/// is not guaranteed to be bit-identical across architectures because the
/// horizontal reduction order may differ.
#[cfg(feature = "portable-simd")]
#[inline]
#[must_use]
pub fn sum_f32_fast(values: &[f32]) -> f32 {
    use std::simd::{f32x8, num::SimdFloat};

    let (chunks, remainder) = values.as_chunks::<8>();
    let mut acc = f32x8::splat(0.0);

    for chunk in chunks
    {
        acc += f32x8::from_array(*chunk);
    }

    let mut total = acc.reduce_sum();
    for &value in remainder
    {
        total += value;
    }
    total
}

/// Scalar fallback for [`sum_f32_fast`].
#[cfg(not(feature = "portable-simd"))]
#[inline]
#[must_use]
pub fn sum_f32_fast(values: &[f32]) -> f32 {
    values.iter().copied().sum()
}

/// Fixed-order left-to-right sum.
///
/// This function preserves a stable operation order independent of SIMD width.
#[inline]
#[must_use]
pub fn sum_f32_deterministic(values: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for &value in values
    {
        total += value;
    }
    total
}

/// Kahan compensated sum.
///
/// This is usually more accurate than a naive sum when magnitudes differ
/// significantly, at the cost of a sequential dependency chain.
#[inline]
#[must_use]
pub fn sum_f32_kahan(values: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut compensation = 0.0f32;

    for &value in values
    {
        let corrected = value - compensation;
        let next = sum + corrected;
        compensation = (next - sum) - corrected;
        sum = next;
    }

    sum
}

/// Maximum value using Rust's `f32::max` semantics.
///
/// Returns `None` for an empty slice. Ties preserve the first occurrence.
#[inline]
#[must_use]
pub fn max_f32(values: &[f32]) -> Option<f32> {
    let (&first, rest) = values.split_first()?;
    let mut maximum = first;

    for &value in rest
    {
        maximum = maximum.max(value);
    }

    Some(maximum)
}

/// Index and value of the first maximum element.
///
/// Returns `None` for an empty slice. Ties preserve the lowest index.
#[inline]
#[must_use]
pub fn argmax_f32(values: &[f32]) -> Option<IndexedValue<f32>> {
    let (&first, rest) = values.split_first()?;
    let mut best = IndexedValue {
        index: 0,
        value: first,
    };

    for (offset, &value) in rest.iter().enumerate()
    {
        if value > best.value
        {
            best = IndexedValue {
                index: offset + 1,
                value,
            };
        }
    }

    Some(best)
}

/// Fast SIMD dot product.
#[cfg(feature = "portable-simd")]
#[inline]
#[must_use]
pub fn dot_f32_fast(lhs: &[f32], rhs: &[f32]) -> f32 {
    use std::simd::{StdFloat, f32x8, num::SimdFloat};

    assert_eq!(lhs.len(), rhs.len(), "dot_f32_fast: length mismatch");

    let (lhs_chunks, lhs_remainder) = lhs.as_chunks::<8>();
    let (rhs_chunks, rhs_remainder) = rhs.as_chunks::<8>();
    let mut acc = f32x8::splat(0.0);

    for (left, right) in lhs_chunks.iter().zip(rhs_chunks)
    {
        acc = f32x8::from_array(*left).mul_add(f32x8::from_array(*right), acc);
    }

    let mut total = acc.reduce_sum();
    for (&left, &right) in lhs_remainder.iter().zip(rhs_remainder)
    {
        total += left * right;
    }
    total
}

/// Scalar fallback for [`dot_f32_fast`].
#[cfg(not(feature = "portable-simd"))]
#[inline]
#[must_use]
pub fn dot_f32_fast(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len(), "dot_f32_fast: length mismatch");
    lhs.iter()
        .zip(rhs)
        .map(|(&left, &right)| left * right)
        .sum()
}

/// Fixed-order left-to-right dot product.
#[inline]
#[must_use]
pub fn dot_f32_deterministic(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "dot_f32_deterministic: length mismatch"
    );

    let mut total = 0.0f32;
    for (&left, &right) in lhs.iter().zip(rhs)
    {
        total += left * right;
    }
    total
}

/// L1 norm, `sum(abs(x[i]))`, using the fast reduction path.
#[cfg(feature = "portable-simd")]
#[inline]
#[must_use]
pub fn l1_norm_f32(values: &[f32]) -> f32 {
    use std::simd::{f32x8, num::SimdFloat};

    let (chunks, remainder) = values.as_chunks::<8>();
    let mut acc = f32x8::splat(0.0);

    for chunk in chunks
    {
        acc += f32x8::from_array(*chunk).abs();
    }

    let mut total = acc.reduce_sum();
    for &value in remainder
    {
        total += value.abs();
    }
    total
}

/// Scalar fallback for [`l1_norm_f32`].
#[cfg(not(feature = "portable-simd"))]
#[inline]
#[must_use]
pub fn l1_norm_f32(values: &[f32]) -> f32 {
    values.iter().map(|value| value.abs()).sum()
}

/// Euclidean norm using the fast dot-product path.
#[inline]
#[must_use]
pub fn l2_norm_f32(values: &[f32]) -> f32 {
    dot_f32_fast(values, values).sqrt()
}

/// Cosine similarity.
///
/// Returns `None` when either input has zero norm.
#[inline]
#[must_use]
pub fn cosine_similarity_f32(lhs: &[f32], rhs: &[f32]) -> Option<f32> {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "cosine_similarity_f32: length mismatch"
    );

    let lhs_norm = l2_norm_f32(lhs);
    let rhs_norm = l2_norm_f32(rhs);
    let denominator = lhs_norm * rhs_norm;

    if denominator == 0.0
    {
        return None;
    }

    Some(dot_f32_fast(lhs, rhs) / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_reductions_have_defined_results() {
        assert_eq!(sum_f32_fast(&[]), 0.0);
        assert_eq!(sum_f32_deterministic(&[]), 0.0);
        assert_eq!(sum_f32_kahan(&[]), 0.0);
        assert_eq!(max_f32(&[]), None);
        assert_eq!(argmax_f32(&[]), None);
        assert_eq!(l1_norm_f32(&[]), 0.0);
        assert_eq!(l2_norm_f32(&[]), 0.0);
    }

    #[test]
    fn sum_variants_match_on_exact_integer_inputs() {
        let values = [1.0f32, -2.0, 3.0, 4.0, -5.0, 6.0, 7.0, -8.0, 9.0];
        assert_eq!(sum_f32_fast(&values), 15.0);
        assert_eq!(sum_f32_deterministic(&values), 15.0);
        assert_eq!(sum_f32_kahan(&values), 15.0);
    }

    #[test]
    fn kahan_is_closer_to_f64_reference_than_naive_sum() {
        let mut values = vec![1.0f32];
        values.extend(std::iter::repeat_n(1.0e-7f32, 100_000));

        let reference = values.iter().map(|&value| f64::from(value)).sum::<f64>();
        let naive = f64::from(sum_f32_deterministic(&values));
        let compensated = f64::from(sum_f32_kahan(&values));

        let naive_error = (naive - reference).abs();
        let compensated_error = (compensated - reference).abs();

        assert!(
            compensated_error < naive_error,
            "Kahan error {compensated_error} must be lower than naive error {naive_error}"
        );
    }

    #[test]
    fn max_and_argmax_preserve_first_tie() {
        let values = [-3.0f32, 7.0, 2.0, 7.0, 1.0];
        assert_eq!(max_f32(&values), Some(7.0));
        assert_eq!(
            argmax_f32(&values),
            Some(IndexedValue {
                index: 1,
                value: 7.0
            })
        );
    }

    #[test]
    fn dot_products_match_on_exact_inputs() {
        let lhs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let rhs = [9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(dot_f32_fast(&lhs, &rhs), 165.0);
        assert_eq!(dot_f32_deterministic(&lhs, &rhs), 165.0);
    }

    #[test]
    fn norms_and_cosine_are_correct() {
        let x = [3.0f32, -4.0];
        let y = [6.0f32, -8.0];

        assert_eq!(l1_norm_f32(&x), 7.0);
        assert_eq!(l2_norm_f32(&x), 5.0);
        assert_eq!(cosine_similarity_f32(&x, &y), Some(1.0));
        assert_eq!(cosine_similarity_f32(&x, &[0.0, 0.0]), None);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn dot_rejects_length_mismatch() {
        let _ = dot_f32_fast(&[1.0], &[1.0, 2.0]);
    }

    #[test]
    fn deterministic_sum_is_bit_reproducible() {
        let values = [0.25f32, -12.0, 3.5, 1.0e-4, -7.75, 2048.0, -1024.0, 0.125];
        let expected = sum_f32_deterministic(&values).to_bits();

        for _ in 0..100
        {
            assert_eq!(sum_f32_deterministic(&values).to_bits(), expected);
        }
    }
}
