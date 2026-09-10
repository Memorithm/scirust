use scirust_core::autodiff::nd::{NdTape, NdVar};
use scirust_core::tensor::tensor_nd::TensorND;

use crate::error::{Result, VariationalError};

pub fn nd_tanh<'t>(tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
    let two = tape.input(TensorND::new(vec![2.0], vec![1, 1]));
    let one = tape.input(TensorND::new(vec![1.0], vec![1, 1]));
    x.mul(two).sigmoid().mul(two).sub(one)
}

pub fn finite_difference_gradient<F>(f: &mut F, x: &[f32], eps: f32) -> Vec<f32>
where
    F: FnMut(&[f32]) -> f32,
{
    try_finite_difference_gradient(f, x, eps)
        .unwrap_or_else(|err| panic!("finite_difference_gradient: {err}"))
}

/// Compute a centered finite-difference gradient after validating inputs and evaluations.
pub fn try_finite_difference_gradient<F>(f: &mut F, x: &[f32], eps: f32) -> Result<Vec<f32>>
where
    F: FnMut(&[f32]) -> f32,
{
    validate_epsilon(eps)?;
    validate_finite_slice("finite-difference coordinate", x)?;

    let n = x.len();
    let mut grad = vec![0.0; n];
    for i in 0..n
    {
        let mut xp = x.to_vec();
        xp[i] += eps;
        validate_finite_value("perturbed finite-difference coordinate", xp[i])?;
        let fp = f(&xp);
        validate_finite_value("finite-difference function value", fp)?;

        let mut xm = x.to_vec();
        xm[i] -= eps;
        validate_finite_value("perturbed finite-difference coordinate", xm[i])?;
        let fm = f(&xm);
        validate_finite_value("finite-difference function value", fm)?;

        let value = (fp - fm) / (2.0 * eps);
        validate_finite_value("finite-difference gradient", value)?;
        grad[i] = value;
    }
    Ok(grad)
}

pub fn finite_difference_hessian<F>(f: &mut F, x: &[f32], eps: f32, n: usize) -> Vec<Vec<f32>>
where
    F: FnMut(&[f32]) -> Vec<f32>,
{
    try_finite_difference_hessian(f, x, eps, n)
        .unwrap_or_else(|err| panic!("finite_difference_hessian: {err}"))
}

/// Compute a centered finite-difference Jacobian/Hessian block with checked dimensions.
pub fn try_finite_difference_hessian<F>(
    f: &mut F,
    x: &[f32],
    eps: f32,
    n: usize,
) -> Result<Vec<Vec<f32>>>
where
    F: FnMut(&[f32]) -> Vec<f32>,
{
    validate_epsilon(eps)?;
    validate_finite_slice("finite-difference coordinate", x)?;
    if n > x.len()
    {
        return Err(VariationalError::DimensionMismatch {
            expected: x.len(),
            got: n,
            context: "try_finite_difference_hessian requested dimension".into(),
        });
    }

    let mut h = vec![vec![0.0; n]; n];
    for j in 0..n
    {
        let mut xp = x.to_vec();
        xp[j] += eps;
        validate_finite_value("perturbed finite-difference coordinate", xp[j])?;
        let fp = f(&xp);
        validate_output_dimension(&fp, n, "try_finite_difference_hessian positive output")?;
        validate_finite_slice("finite-difference function output", &fp[..n])?;

        let mut xm = x.to_vec();
        xm[j] -= eps;
        validate_finite_value("perturbed finite-difference coordinate", xm[j])?;
        let fm = f(&xm);
        validate_output_dimension(&fm, n, "try_finite_difference_hessian negative output")?;
        validate_finite_slice("finite-difference function output", &fm[..n])?;

        for i in 0..n
        {
            let value = (fp[i] - fm[i]) / (2.0 * eps);
            validate_finite_value("finite-difference Hessian", value)?;
            h[i][j] = value;
        }
    }
    Ok(h)
}

pub fn solve_linear_system(a: &[Vec<f32>], b: &[f32], n: usize) -> Result<Vec<f32>> {
    validate_linear_system_dimensions(a, b, n)?;
    for row in a
    {
        validate_finite_slice("linear-system matrix", row)?;
    }
    validate_finite_slice("linear-system rhs", b)?;

    let mut augmented = vec![vec![0.0; n + 1]; n];
    for i in 0..n
    {
        for j in 0..n
        {
            augmented[i][j] = a[i][j];
        }
        augmented[i][n] = b[i];
    }

    for col in 0..n
    {
        let mut max_row = col;
        let mut max_val = augmented[col][col].abs();
        for row in col + 1..n
        {
            let val = augmented[row][col].abs();
            if val > max_val
            {
                max_val = val;
                max_row = row;
            }
        }
        validate_finite_value("linear-system pivot", max_val)?;
        if max_val < 1e-16
        {
            return Err(VariationalError::SingularVelocityHessian {
                condition_number: f32::INFINITY,
                tolerance: 1e-8,
            });
        }
        if max_row != col
        {
            augmented.swap(col, max_row);
        }
        for row in col + 1..n
        {
            let factor = augmented[row][col] / augmented[col][col];
            validate_finite_value("linear-system elimination factor", factor)?;
            for k in col..=n
            {
                augmented[row][k] -= factor * augmented[col][k];
                validate_finite_value("linear-system elimination value", augmented[row][k])?;
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev()
    {
        let mut sum = augmented[i][n];
        for j in i + 1..n
        {
            sum -= augmented[i][j] * x[j];
        }
        validate_finite_value("linear-system back substitution", sum)?;
        if augmented[i][i].abs() < 1e-16
        {
            return Err(VariationalError::LinearSolveFailure {
                details: format!("zero pivot at row {i}"),
            });
        }
        x[i] = sum / augmented[i][i];
        validate_finite_value("linear_solve", x[i])?;
    }

    Ok(x)
}

fn validate_epsilon(eps: f32) -> Result<()> {
    if !eps.is_finite()
    {
        return Err(VariationalError::NonFiniteValue {
            component: "finite-difference epsilon",
            value: eps,
        });
    }
    if eps <= 0.0
    {
        return Err(VariationalError::UnsupportedOperation {
            details: "finite-difference epsilon must be strictly positive".into(),
        });
    }
    Ok(())
}

fn validate_finite_slice(component: &'static str, values: &[f32]) -> Result<()> {
    for &value in values
    {
        validate_finite_value(component, value)?;
    }
    Ok(())
}

fn validate_finite_value(component: &'static str, value: f32) -> Result<()> {
    if !value.is_finite()
    {
        return Err(VariationalError::NonFiniteValue { component, value });
    }
    Ok(())
}

fn validate_output_dimension(values: &[f32], n: usize, context: &str) -> Result<()> {
    if values.len() < n
    {
        return Err(VariationalError::DimensionMismatch {
            expected: n,
            got: values.len(),
            context: context.into(),
        });
    }
    Ok(())
}

fn validate_linear_system_dimensions(a: &[Vec<f32>], b: &[f32], n: usize) -> Result<()> {
    if a.len() != n
    {
        return Err(VariationalError::DimensionMismatch {
            expected: n,
            got: a.len(),
            context: "solve_linear_system matrix rows".into(),
        });
    }
    for (row_index, row) in a.iter().enumerate()
    {
        if row.len() != n
        {
            return Err(VariationalError::DimensionMismatch {
                expected: n,
                got: row.len(),
                context: format!("solve_linear_system matrix row {row_index}"),
            });
        }
    }
    if b.len() != n
    {
        return Err(VariationalError::DimensionMismatch {
            expected: n,
            got: b.len(),
            context: "solve_linear_system rhs".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_gradient_rejects_zero_epsilon() {
        let mut f = |x: &[f32]| x[0] * x[0];
        let err = try_finite_difference_gradient(&mut f, &[1.0], 0.0).unwrap_err();
        assert!(matches!(err, VariationalError::UnsupportedOperation { .. }));
    }

    #[test]
    fn checked_gradient_does_not_evaluate_unused_center_point() {
        let mut calls = 0usize;
        let mut f = |x: &[f32]| {
            calls += 1;
            x.iter().sum()
        };
        let grad = try_finite_difference_gradient(&mut f, &[1.0, 2.0], 1e-3).unwrap();

        assert_eq!(calls, 4);
        assert_eq!(grad.len(), 2);
    }

    #[test]
    fn checked_hessian_rejects_requested_dimension_larger_than_input() {
        let mut f = |x: &[f32]| x.to_vec();
        let err = try_finite_difference_hessian(&mut f, &[1.0], 1e-3, 2).unwrap_err();
        assert!(matches!(
            err,
            VariationalError::DimensionMismatch {
                expected: 1,
                got: 2,
                ..
            }
        ));
    }

    #[test]
    fn checked_hessian_rejects_short_function_output() {
        let mut f = |_x: &[f32]| Vec::<f32>::new();
        let err = try_finite_difference_hessian(&mut f, &[1.0], 1e-3, 1).unwrap_err();
        assert!(matches!(
            err,
            VariationalError::DimensionMismatch {
                expected: 1,
                got: 0,
                ..
            }
        ));
    }

    #[test]
    fn linear_solver_rejects_matrix_dimension_mismatch() {
        let err = solve_linear_system(&[vec![1.0]], &[1.0, 2.0], 2).unwrap_err();
        assert!(matches!(
            err,
            VariationalError::DimensionMismatch {
                expected: 2,
                got: 1,
                ..
            }
        ));
    }

    #[test]
    fn linear_solver_solves_well_conditioned_system() {
        let solution =
            solve_linear_system(&[vec![3.0, 2.0], vec![1.0, 2.0]], &[5.0, 5.0], 2).unwrap();

        assert!((solution[0] - 0.0).abs() < 1e-5);
        assert!((solution[1] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn linear_solver_rejects_non_finite_coefficients() {
        let err = solve_linear_system(&[vec![f32::NAN]], &[1.0], 1).unwrap_err();
        assert!(matches!(
            err,
            VariationalError::NonFiniteValue {
                component: "linear-system matrix",
                ..
            }
        ));
    }
}
