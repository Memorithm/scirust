//! Décomposition de Cholesky pour matrices symétriques définies positives.
//!
//! A = L · L^T où L est triangulaire inférieure.

use super::Matrix;
use crate::{SolverError, SolverResult};
use tracing::warn;

/// Pivot-rejection threshold `n · eps · scale` (Golub & Van Loan, *Matrix
/// Computations*, §3.4.6), where `scale` is the sum of the magnitudes of the
/// terms that produced the pivot (`a_jj` and the `l_jk²` subtracted from it).
/// A squared pivot at or below this cannot be told apart from the rounding
/// error of its own computation.
///
/// The threshold is local to each pivot rather than a fixed absolute constant
/// (a regular matrix at a small physical scale was declared non-SPD) or a
/// fraction of the matrix's global `max|a_ij|` (a regular matrix whose
/// variables are in very different units, e.g. `diag(1e34, 1)`, was declared
/// non-SPD or singular, since its small-scale pivots fall below it).
fn pivot_tol(n: usize, scale: f64) -> f64 {
    (n as f64) * f64::EPSILON * scale
}

fn check_finite(value: f64) -> Result<(), SolverError> {
    if !value.is_finite()
    {
        return Err(SolverError::NanDetected { iter: 0, value });
    }
    Ok(())
}

/// Cholesky en place : remplit la partie triangulaire inférieure de A avec L.
/// L'entrée doit être symétrique définie positive ; sinon `NotSpd`.
/// Renvoie L (matrice triangulaire inf avec zéros au-dessus).
pub fn cholesky_decompose(a: Matrix) -> SolverResult<Matrix> {
    let n = a.ensure_square()?;
    let mut l = Matrix::zeros(n, n);
    // negligible[j]: the pivot l_jj is indistinguishable from the rounding
    // error of its own computation (see `pivot_tol`), so it must not be used
    // as a divisor.
    let mut negligible = vec![false; n];

    for i in 0..n
    {
        for j in 0..=i
        {
            let mut s = 0.0;
            for k in 0..j
            {
                let lik = l[(i, k)];
                let ljk = l[(j, k)];
                check_finite(lik)?;
                check_finite(ljk)?;
                s += lik * ljk;
            }
            let aij = a[(i, j)];
            check_finite(aij)?;

            if i == j
            {
                let val = aij - s;
                if val <= 0.0
                {
                    warn!(
                        target: "solver",
                        "Cholesky failed: a[{i},{i}] - s = {:.3e} <= 0 (not SPD)",
                        val
                    );
                    return Err(SolverError::NotSpd);
                }
                negligible[i] = val <= pivot_tol(n, aij.abs() + s);
                let root = val.sqrt();
                check_finite(root)?;
                l[(i, j)] = root;
            }
            else
            {
                let ljj = l[(j, j)];
                if negligible[j]
                {
                    warn!(
                        target: "solver",
                        "Cholesky: l[{j},{j}] = {:.3e} near-zero at row {} — not SPD",
                        ljj, j
                    );
                    return Err(SolverError::NotSpd);
                }
                let entry = (aij - s) / ljj;
                check_finite(entry)?;
                l[(i, j)] = entry;
            }
        }
    }
    Ok(l)
}

fn row_inf_norm(m: &Matrix) -> f64 {
    (0..m.rows())
        .map(|i| (0..m.cols()).map(|j| m[(i, j)].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// Nombre de conditionnement en norme infinie, `cond_∞(A) = ‖A‖_∞ · ‖A⁻¹‖_∞`.
///
/// `A` n'est pas reconstructible à partir de `L` seule (`cholesky_decompose`
/// consomme son entrée), d'où le paramètre `a` explicite — la même matrice
/// que celle passée à `cholesky_decompose` pour produire `l`.
///
/// Forme `A⁻¹` explicitement via `n` résolutions triangulaires (une par
/// colonne de l'identité), `O(n³)` — même ordre que la factorisation
/// elle-même. Renvoie `f64::INFINITY` (jamais `NaN`) si `A` est singulière ;
/// `rcond_cholesky` renvoie `0.0` dans ce cas.
pub fn cond_cholesky(l: &Matrix, a: &Matrix) -> SolverResult<f64> {
    let n = l.rows();
    if a.rows() != n || a.cols() != n
    {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: a.rows(),
        });
    }
    let a_norm = row_inf_norm(a);
    if a_norm == 0.0
    {
        return Ok(f64::INFINITY);
    }
    let mut inv = Matrix::zeros(n, n);
    let mut e = vec![0.0; n];
    for j in 0..n
    {
        e.iter_mut().for_each(|x| *x = 0.0);
        e[j] = 1.0;
        let col = solve_cholesky(l, &e)?;
        for i in 0..n
        {
            inv[(i, j)] = col[i];
        }
    }
    Ok(a_norm * row_inf_norm(&inv))
}

/// Inverse du nombre de conditionnement, `1 / cond_∞(A)` — `0.0` si `A` est
/// singulière (au lieu de `NaN`).
pub fn rcond_cholesky(l: &Matrix, a: &Matrix) -> SolverResult<f64> {
    let c = cond_cholesky(l, a)?;
    Ok(if c.is_infinite() { 0.0 } else { 1.0 / c })
}

/// Résout A · x = b sachant A = L·L^T, en deux passes triangulaires.
pub fn solve_cholesky(l: &Matrix, b: &[f64]) -> SolverResult<Vec<f64>> {
    let n = l.rows();
    if b.len() != n
    {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    // Vérifier que b est fini
    for &bi in b
    {
        check_finite(bi)?;
    }

    // L n'est pas reçue avec la matrice A d'origine ; l'échelle de référence
    // de chaque pivot est celle de sa propre ligne, Σ_k l_ik² (= a_ii pour un
    // L issu de cholesky_decompose), comme dans cholesky_decompose : un pivot
    // n'est pas comparé au plus grand l_ii de toute la matrice.
    let diag_ok: Vec<bool> = (0..n)
        .map(|i| {
            // Normalised by the row's largest entry so the squares cannot
            // overflow; a zero row (or a NaN) leaves the pivot rejected.
            let m = (0..=i).fold(0.0f64, |acc, k| acc.max(l[(i, k)].abs()));
            let row_sq: f64 = (0..=i).map(|k| (l[(i, k)] / m).powi(2)).sum();
            let d = l[(i, i)] / m;
            m > 0.0 && d * d > pivot_tol(n, row_sq)
        })
        .collect();

    // L · y = b (substitution avant)
    let mut y = vec![0.0; n];
    for i in 0..n
    {
        let mut s = b[i];
        for j in 0..i
        {
            s -= l[(i, j)] * y[j];
        }
        let diag = l[(i, i)];
        if !diag_ok[i]
        {
            return Err(SolverError::Singular {
                row: i,
                pivot: diag,
            });
        }
        y[i] = s / diag;
        check_finite(y[i])?;
    }
    // L^T · x = y (substitution arrière)
    let mut x = vec![0.0; n];
    for i in (0..n).rev()
    {
        let mut s = y[i];
        for j in (i + 1)..n
        {
            s -= l[(j, i)] * x[j];
        }
        let diag = l[(i, i)];
        if !diag_ok[i]
        {
            return Err(SolverError::Singular {
                row: i,
                pivot: diag,
            });
        }
        x[i] = s / diag;
        check_finite(x[i])?;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cholesky_3x3() -> SolverResult<()> {
        let a = Matrix::from_row_major(
            3,
            3,
            vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
        );
        let l = cholesky_decompose(a.clone())?;
        assert_relative_eq!(l[(0, 0)], 2.0, epsilon = 1e-10);
        assert_relative_eq!(l[(1, 1)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(l[(2, 2)], 3.0, epsilon = 1e-10);
        let lt = l.transpose();
        let prod = l.matmul(&lt)?;
        for i in 0..3
        {
            for j in 0..3
            {
                assert_relative_eq!(prod[(i, j)], a[(i, j)], epsilon = 1e-10);
            }
        }
        Ok(())
    }

    #[test]
    fn cholesky_solve() -> SolverResult<()> {
        let a = Matrix::from_row_major(
            3,
            3,
            vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
        );
        let b = vec![1.0, 2.0, 3.0];
        let l = cholesky_decompose(a.clone())?;
        let x = solve_cholesky(&l, &b)?;
        let ax = a.matvec(&x)?;
        for (axi, bi) in ax.iter().zip(&b)
        {
            assert_relative_eq!(*axi, *bi, epsilon = 1e-9);
        }
        Ok(())
    }

    #[test]
    fn cholesky_solve_at_a_tiny_physical_scale() -> SolverResult<()> {
        // Regression test for a P1 audit finding: PIVOT_EPS was a fixed
        // absolute 1e-15 compared directly against L's diagonal — the same
        // regular SPD system as `cholesky_solve` above, scaled down so that L's
        // diagonal (which scales as √scale) falls well below the old cutoff,
        // was declared not-SPD even though it is perfectly well-conditioned.
        let scale = 1e-34;
        let a = Matrix::from_row_major(
            3,
            3,
            vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0]
                .into_iter()
                .map(|v| v * scale)
                .collect(),
        );
        let b = vec![1.0 * scale, 2.0 * scale, 3.0 * scale];
        let l = cholesky_decompose(a.clone())?;
        let x = solve_cholesky(&l, &b)?;
        let ax = a.matvec(&x)?;
        for (axi, bi) in ax.iter().zip(&b)
        {
            assert_relative_eq!(*axi, *bi, epsilon = 1e-9, max_relative = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn cholesky_with_variables_of_very_different_scales() -> SolverResult<()> {
        // Regression test: the pivot threshold was `n · eps · max|·|`, global
        // to the whole matrix. `A = D·B·D` with `B` the SPD matrix of
        // `cholesky_solve` and `D = diag(1e17, 1, 1)` (one variable expressed
        // in units 1e17 times larger) is as well-conditioned as `B` after
        // diagonal scaling, but its second pivot (1.0) fell below the global
        // threshold (~2.7e19) and the factorization returned `NotSpd`.
        let b_mat = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
        let d = [1e17, 1.0, 1.0];
        let a = Matrix::from_row_major(
            3,
            3,
            (0..9).map(|k| d[k / 3] * b_mat[k] * d[k % 3]).collect(),
        );
        let l = cholesky_decompose(a.clone())?;
        assert_relative_eq!(l[(0, 0)], 2e17, max_relative = 1e-12);
        assert_relative_eq!(l[(1, 1)], 1.0, max_relative = 1e-9);
        assert_relative_eq!(l[(2, 2)], 3.0, max_relative = 1e-9);
        // Solution of `A·x = D·[1, 2, 3]` is `x = D⁻¹·B⁻¹·[1, 2, 3]`.
        let rhs = [1.0, 2.0, 3.0];
        let x = solve_cholesky(&l, &[d[0] * rhs[0], rhs[1], rhs[2]])?;
        let b_dense = Matrix::from_row_major(3, 3, b_mat.to_vec());
        let y = solve_cholesky(&cholesky_decompose(b_dense)?, &rhs)?;
        for i in 0..3
        {
            assert_relative_eq!(x[i] * d[i], y[i], max_relative = 1e-9);
        }
        Ok(())
    }

    #[test]
    fn solve_cholesky_with_a_badly_scaled_diagonal() -> SolverResult<()> {
        // Regression test for `A = diag(1e34, 1)`: `cholesky_decompose`
        // compared `l_00 = 1e17` to `n · eps · max|a_ij|` (~4.4e18, in the
        // units of `A`, not of `L`) and returned `NotSpd`; `solve_cholesky`
        // compared each `l_ii` to `n · eps · max_i |l_ii|` (~44) and reported
        // the second diagonal entry (1.0) singular.
        let a = Matrix::from_row_major(2, 2, vec![1e34, 0.0, 0.0, 1.0]);
        let l = cholesky_decompose(a)?;
        assert_relative_eq!(l[(0, 0)], 1e17, max_relative = 1e-12);
        let x = solve_cholesky(&l, &[2e34, 3.0])?;
        assert_relative_eq!(x[0], 2.0, max_relative = 1e-12);
        assert_relative_eq!(x[1], 3.0, max_relative = 1e-12);

        let l = Matrix::from_row_major(2, 2, vec![1e17, 0.0, 0.0, 1.0]);
        let x = solve_cholesky(&l, &[2e34, 3.0])?;
        assert_relative_eq!(x[0], 2.0, max_relative = 1e-12);
        assert_relative_eq!(x[1], 3.0, max_relative = 1e-12);
        Ok(())
    }

    #[test]
    fn solve_cholesky_still_rejects_a_zero_diagonal() {
        let l = Matrix::from_row_major(2, 2, vec![1.0, 0.0, 1.0, 0.0]);
        assert!(matches!(
            solve_cholesky(&l, &[1.0, 1.0]),
            Err(SolverError::Singular { row: 1, .. })
        ));
    }

    #[test]
    fn rejects_non_spd() {
        // Pas SPD (négative)
        let a = Matrix::from_row_major(2, 2, vec![-1.0, 0.0, 0.0, -1.0]);
        assert!(matches!(cholesky_decompose(a), Err(SolverError::NotSpd)));
    }

    #[test]
    fn cond_of_identity_is_one() -> SolverResult<()> {
        let a = Matrix::from_row_major(
            3,
            3,
            vec![
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0,
            ],
        );
        let l = cholesky_decompose(a.clone())?;
        assert_relative_eq!(cond_cholesky(&l, &a)?, 1.0, epsilon = 1e-10);
        assert_relative_eq!(rcond_cholesky(&l, &a)?, 1.0, epsilon = 1e-10);
        Ok(())
    }

    #[test]
    fn cond_of_diagonal_matrix_matches_ratio_of_extremes() -> SolverResult<()> {
        // A diagonale SPD : cond_∞ = max(diag) / min(diag).
        let a = Matrix::from_row_major(
            3,
            3,
            vec![
                100.0, 0.0, 0.0, //
                0.0, 10.0, 0.0, //
                0.0, 0.0, 1.0,
            ],
        );
        let l = cholesky_decompose(a.clone())?;
        assert_relative_eq!(cond_cholesky(&l, &a)?, 100.0, epsilon = 1e-8);
        assert_relative_eq!(rcond_cholesky(&l, &a)?, 0.01, epsilon = 1e-8);
        Ok(())
    }

    #[test]
    fn cond_rejects_shape_mismatch() -> SolverResult<()> {
        let a3 = Matrix::from_row_major(
            3,
            3,
            vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
        );
        let l3 = cholesky_decompose(a3)?;
        let a2 = Matrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        assert!(matches!(
            cond_cholesky(&l3, &a2),
            Err(SolverError::DimensionMismatch { .. })
        ));
        Ok(())
    }
}

/// LAPACK-style property tests: reconstruction and residual checks over
/// randomly generated SPD matrices, rather than fixed point values.
#[cfg(test)]
mod proptests {
    use super::*;
    use crate::linalg::{Matrix, norm2};
    use proptest::prelude::*;

    /// A = MᵀM + n·I is SPD for any M: MᵀM is PSD, and adding n·I shifts
    /// every eigenvalue up by exactly n, making it strictly positive
    /// definite — so `cholesky_decompose` must always succeed on it.
    fn spd_from(n: usize, raw: &[f64]) -> Matrix {
        let m = Matrix::from_row_major(n, n, raw.to_vec());
        let mut a = m.transpose().matmul(&m).unwrap();
        for i in 0..n
        {
            a[(i, i)] += n as f64;
        }
        a
    }

    proptest! {
        /// L·Lᵀ must reconstruct the original SPD matrix.
        #[test]
        fn reconstructs_spd_matrix_as_l_l_t(raw in prop::collection::vec(-10.0f64..10.0, 16)) {
            let n = 4;
            let a = spd_from(n, &raw);
            let l = cholesky_decompose(a.clone()).expect("A = MᵀM + n·I is always SPD");
            let prod = l.matmul(&l.transpose()).unwrap();
            for i in 0..n
            {
                for j in 0..n
                {
                    let tol = 1e-8 * (1.0 + a[(i, j)].abs());
                    prop_assert!(
                        (prod[(i, j)] - a[(i, j)]).abs() < tol,
                        "L·Lᵀ != A at ({i},{j}): {} vs {}", prod[(i, j)], a[(i, j)]
                    );
                }
            }
        }

        /// Residual check for `solve_cholesky`.
        #[test]
        fn solve_residual_is_small(
            raw in prop::collection::vec(-10.0f64..10.0, 16),
            b in prop::collection::vec(-10.0f64..10.0, 4),
        ) {
            let n = 4;
            let a = spd_from(n, &raw);
            let l = cholesky_decompose(a.clone()).unwrap();
            let x = solve_cholesky(&l, &b).expect("solve must succeed on an SPD system");
            let ax = a.matvec(&x).unwrap();
            let b_norm = norm2(&b).max(1e-300);
            let res = ax.iter().zip(&b).map(|(axi, bi)| (axi - bi).powi(2)).sum::<f64>().sqrt();
            prop_assert!(res / b_norm < 1e-7, "relative residual {} too large", res / b_norm);
        }
    }
}
