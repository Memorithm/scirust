//! Estimation d'état de réseau électrique par moindres carrés pondérés
//! (WLS) et détection de données aberrantes — la couche qui, en amont de
//! [`crate::grid_frequency`]/[`crate::rocof`], reconstruit l'état complet
//! du réseau (tensions/angles à chaque nœud) à partir d'un jeu de mesures
//! redondant et bruité (flux de puissance, injections, tensions).
//!
//! Référence : Abur & Expósito, *Power System State Estimation: Theory and
//! Implementation* (CRC Press, 2004), chapitres 2 (WLS) et 5 (détection de
//! données aberrantes). Formulation linéaire (DC ou déjà linéarisée autour
//! d'un point de fonctionnement) : `z = H·x + e`, bruit `e` de covariance
//! diagonale `R = diag(1/w_i)`.
//!
//! ## Estimateur WLS
//! `x̂ = argmin (z-Hx)ᵀW(z-Hx)` a pour solution fermée
//! `x̂ = (HᵀWH)⁻¹HᵀWz` — la matrice de gain `G = HᵀWH` doit être inversible
//! (réseau **observable** avec ce jeu de mesures) ; sinon l'estimation est
//! indéterminée (variable(s) d'état non observables).
//!
//! ## Détection de données aberrantes
//! Deux tests complémentaires (Abur & Expósito §5.3–5.4) :
//! - **Test du χ²** global sur l'objectif `J(x̂) = r̂ᵀWr̂` : au-delà d'un
//!   seuil tabulé pour `dof = m-n` degrés de liberté (table du χ², niveau
//!   de confiance choisi par l'appelant — comme `scirust_multivariate::
//!   mahalanobis_outliers`, ce module ne réimplémente pas l'inverse de la
//!   fonction gamma incomplète, il consomme un seuil déjà lu dans une
//!   table), signale la présence d'au moins une mesure aberrante sans dire
//!   laquelle.
//! - **Test du plus grand résidu normalisé** : `r_N,i = |r̂_i| / √Ω_ii`
//!   avec `Ω = W⁻¹ - HG⁻¹Hᵀ` la covariance des résidus ; la mesure au
//!   résidu normalisé le plus élevé au-delà du seuil (usuellement 3.0, ~3σ)
//!   est la suspecte la plus probable.
//!
//! **Limite honnête** : avec une redondance faible (`m-n` petit, mesures
//! dites "critiques" ou en paire critique), plusieurs résidus normalisés
//! peuvent être quasi identiques — le test détecte qu'une mesure est
//! mauvaise sans pouvoir l'identifier de façon unique. C'est une limite
//! physique de la méthode (Abur & Expósito §5.6), pas un défaut
//! d'implémentation.

use scirust_solvers::SolverError;
use scirust_solvers::linalg::{Matrix, solve};
use thiserror::Error;

/// Erreurs de l'estimation d'état / détection de données aberrantes.
#[derive(Debug, Clone, Error, PartialEq)]
pub enum GridError {
    #[error(
        "dimension mismatch: H is {h_rows}x{h_cols}, z has {z_len} measurements, weights has {w_len}"
    )]
    DimensionMismatch {
        h_rows: usize,
        h_cols: usize,
        z_len: usize,
        w_len: usize,
    },

    #[error("non-positive or non-finite weight at measurement {index}: {weight}")]
    InvalidWeight { index: usize, weight: f64 },

    #[error(
        "numerical failure in state estimation (commonly: gain matrix singular \u{2014} the \
         network is unobservable with this measurement set): {0}"
    )]
    Numerical(String),
}

impl From<SolverError> for GridError {
    fn from(e: SolverError) -> Self {
        GridError::Numerical(e.to_string())
    }
}

/// Résultat d'une estimation d'état WLS.
#[derive(Debug, Clone, PartialEq)]
pub struct StateEstimate {
    /// État estimé `x̂` (par ex. angles de tension, un par nœud non-bilan).
    pub x: Vec<f64>,
    /// Résidus de mesure `r̂ = z - Hx̂`.
    pub residuals: Vec<f64>,
    /// Fonction objectif pondérée `J(x̂) = r̂ᵀWr̂`, utilisée par le test du χ².
    pub objective: f64,
}

fn validate_inputs(h: &Matrix, z: &[f64], weights: &[f64]) -> Result<(usize, usize), GridError> {
    let (m, n) = h.shape();
    if z.len() != m || weights.len() != m
    {
        return Err(GridError::DimensionMismatch {
            h_rows: m,
            h_cols: n,
            z_len: z.len(),
            w_len: weights.len(),
        });
    }
    for (i, &w) in weights.iter().enumerate()
    {
        if !(w.is_finite() && w > 0.0)
        {
            return Err(GridError::InvalidWeight {
                index: i,
                weight: w,
            });
        }
    }
    Ok((m, n))
}

/// `H` pondérée ligne par ligne par `weights` (i.e. `W·H`, `W` diagonale).
fn weighted_h(h: &Matrix, weights: &[f64]) -> Matrix {
    let (m, n) = h.shape();
    Matrix::from_fn(m, n, |i, j| h[(i, j)] * weights[i])
}

/// Estimation d'état linéaire par moindres carrés pondérés.
///
/// `h` est la matrice de mesure (Jacobienne, `m` mesures × `n` variables
/// d'état), `z` le vecteur de mesures, `weights` les poids `w_i = 1/σ_i²`
/// (précision de chaque capteur). Renvoie [`GridError::Numerical`] si le
/// réseau n'est pas observable avec ce jeu de mesures (`HᵀWH` singulière).
pub fn wls_state_estimate(
    h: &Matrix,
    z: &[f64],
    weights: &[f64],
) -> Result<StateEstimate, GridError> {
    validate_inputs(h, z, weights)?;
    let wh = weighted_h(h, weights);
    let g = h.transpose().matmul(&wh)?;
    let wz: Vec<f64> = z.iter().zip(weights).map(|(zi, wi)| zi * wi).collect();
    let rhs = h.transpose().matvec(&wz)?;
    let x = solve(g, &rhs)?;
    let hx = h.matvec(&x)?;
    let residuals: Vec<f64> = z.iter().zip(&hx).map(|(zi, hxi)| zi - hxi).collect();
    let objective = residuals.iter().zip(weights).map(|(r, w)| w * r * r).sum();
    Ok(StateEstimate {
        x,
        residuals,
        objective,
    })
}

/// Test global du χ² sur l'objectif WLS : `objective > threshold` signale
/// la présence d'au moins une mesure aberrante (sans l'identifier). Le
/// degré de liberté est `m - n` (mesures moins variables d'état) ;
/// `threshold` se lit dans une table du χ² standard pour ce degré de
/// liberté et le niveau de confiance voulu (typiquement 95% ou 99%).
pub fn chi_squared_test(objective: f64, threshold: f64) -> bool {
    objective > threshold
}

/// Rapport du test du plus grand résidu normalisé.
#[derive(Debug, Clone, PartialEq)]
pub struct BadDataReport {
    /// Résidu normalisé `r_N,i = |r̂_i| / √Ω_ii` pour chaque mesure.
    pub normalized_residuals: Vec<f64>,
    /// Index de la mesure la plus suspecte (résidu normalisé maximal et
    /// au-delà du seuil), `None` si aucune ne dépasse le seuil.
    pub suspect_index: Option<usize>,
}

/// Test du plus grand résidu normalisé (Abur & Expósito §5.4).
///
/// `h`/`weights` doivent être les mêmes que ceux passés à
/// [`wls_state_estimate`] ; `residuals` son résultat `residuals`.
/// `threshold` est usuellement ~3.0 (≈3σ sous H0 : résidu normal centré
/// réduit). Une mesure dont `Ω_ii` est négligeable devant sa variance
/// `1/w_i` (mesure critique, `Ω_ii/(1/w_i) ≤ 1e-10`) reçoit un résidu
/// normalisé de 0 : son erreur n'est pas détectable par ce test. Le seuil
/// est relatif, donc le résultat ne dépend pas des unités des mesures.
pub fn largest_normalized_residual_test(
    h: &Matrix,
    weights: &[f64],
    residuals: &[f64],
    threshold: f64,
) -> Result<BadDataReport, GridError> {
    let (m, _n) = validate_inputs(h, residuals, weights)?;
    let wh = weighted_h(h, weights);
    let g = h.transpose().matmul(&wh)?;
    let g_inv = g.inverse()?;
    let hg = h.matmul(&g_inv)?; // m x n
    let mut normalized_residuals = vec![0.0; m];
    for i in 0..m
    {
        let mut hgh_ii = 0.0;
        for j in 0..h.cols()
        {
            hgh_ii += hg[(i, j)] * h[(i, j)];
        }
        // Ω_ii = R_ii·(1 - K_ii) with 0 ≤ K_ii ≤ 1, so the zero test is made
        // relative to the measurement variance R_ii = 1/w_i: an absolute
        // cut-off would depend on the units of z and zero out every
        // normalized residual once the variances are small enough. A
        // critical measurement (K_ii = 1) still lands below the cut-off.
        let r_ii = 1.0 / weights[i];
        let omega_ii = r_ii - hgh_ii;
        normalized_residuals[i] = if omega_ii > 1e-10 * r_ii
        {
            residuals[i].abs() / omega_ii.sqrt()
        }
        else
        {
            0.0
        };
    }
    let mut suspect_index = None;
    let mut best = threshold;
    for (i, &rn) in normalized_residuals.iter().enumerate()
    {
        if rn > best
        {
            best = rn;
            suspect_index = Some(i);
        }
    }
    Ok(BadDataReport {
        normalized_residuals,
        suspect_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Exemple 3-nœuds en flux de puissance DC (Abur & Expósito, chapitre
    /// 2, forme du réseau), vérifié indépendamment par calcul numérique
    /// direct (numpy) : x̂ = [-0.019833, -0.034983], résidus
    /// [0.003667, -0.001833, 0.001833].
    fn three_bus_example() -> (Matrix, Vec<f64>) {
        let h = Matrix::from_row_major(3, 2, vec![-10.0, 0.0, 0.0, -10.0, 20.0, -10.0]);
        let z = vec![0.202, 0.348, -0.045];
        (h, z)
    }

    #[test]
    fn matches_the_three_bus_worked_example() {
        let (h, z) = three_bus_example();
        let weights = vec![1.0, 1.0, 1.0];
        let est = wls_state_estimate(&h, &z, &weights).unwrap();
        assert_relative_eq!(est.x[0], -0.019_833_333_333, epsilon = 1e-9);
        assert_relative_eq!(est.x[1], -0.034_983_333_333, epsilon = 1e-9);
        assert_relative_eq!(est.residuals[0], 0.003_666_666_667, epsilon = 1e-9);
        assert_relative_eq!(est.residuals[1], -0.001_833_333_333, epsilon = 1e-9);
        assert_relative_eq!(est.residuals[2], 0.001_833_333_333, epsilon = 1e-9);
    }

    #[test]
    fn rejects_mismatched_dimensions() {
        let (h, _z) = three_bus_example();
        let z_short = vec![0.202, 0.348];
        let weights = vec![1.0, 1.0, 1.0];
        assert_eq!(
            wls_state_estimate(&h, &z_short, &weights).unwrap_err(),
            GridError::DimensionMismatch {
                h_rows: 3,
                h_cols: 2,
                z_len: 2,
                w_len: 3,
            }
        );
    }

    #[test]
    fn rejects_non_positive_weight() {
        let (h, z) = three_bus_example();
        let weights = vec![1.0, 0.0, 1.0];
        assert_eq!(
            wls_state_estimate(&h, &z, &weights).unwrap_err(),
            GridError::InvalidWeight {
                index: 1,
                weight: 0.0
            }
        );
    }

    #[test]
    fn reports_unobservable_network_as_numerical_error() {
        // Deux mesures colinéaires ne contraignent qu'une seule direction
        // de x (n=2) : HᵀWH est singulière.
        let h = Matrix::from_row_major(2, 2, vec![1.0, 1.0, 2.0, 2.0]);
        let z = vec![1.0, 2.0];
        let weights = vec![1.0, 1.0];
        assert!(matches!(
            wls_state_estimate(&h, &z, &weights),
            Err(GridError::Numerical(_))
        ));
    }

    /// Cas synthétique à redondance suffisante (m=4, n=2) pour que
    /// l'identification de la mesure aberrante soit possible — vérifié
    /// indépendamment par calcul numpy. Sans bruit, x_true=[3,2] donne
    /// z_true=[3,2,5,1] exactement (résidus nuls). En injectant +5.0 sur
    /// la mesure d'indice 2, le résidu normalisé de cette mesure (≈2.887)
    /// dépasse celui des autres (≈2.041, ≈0) et un seuil de 2.5.
    fn redundant_example() -> Matrix {
        Matrix::from_row_major(4, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0])
    }

    #[test]
    fn chi_squared_test_is_silent_on_clean_data() {
        let h = redundant_example();
        let z = vec![3.0, 2.0, 5.0, 1.0]; // exactly H * [3, 2]
        let weights = vec![1.0; 4];
        let est = wls_state_estimate(&h, &z, &weights).unwrap();
        assert_relative_eq!(est.objective, 0.0, epsilon = 1e-20);
        assert!(!chi_squared_test(est.objective, 2.0));
    }

    #[test]
    fn identifies_the_injected_bad_measurement() {
        let h = redundant_example();
        let mut z = vec![3.0, 2.0, 5.0, 1.0];
        z[2] += 5.0; // inject a large error on measurement index 2
        let weights = vec![1.0; 4];
        let est = wls_state_estimate(&h, &z, &weights).unwrap();

        assert!(chi_squared_test(est.objective, 2.0));

        let report = largest_normalized_residual_test(&h, &weights, &est.residuals, 2.5).unwrap();
        assert_eq!(report.suspect_index, Some(2));
        assert_relative_eq!(
            report.normalized_residuals[2],
            2.886_751_346,
            epsilon = 1e-6
        );
        assert_relative_eq!(report.normalized_residuals[3], 0.0, epsilon = 1e-6);
    }

    /// Regression: the residual covariance `Ω_ii` was compared against an
    /// absolute `1e-12`, so once the measurement variances `1/w_i` were that
    /// small in the caller's units every normalized residual was reported as
    /// 0 and the gross error went unflagged. Re-expressing the same problem
    /// in units scaled by `s` (z and H scaled by `s`, σ scaled by `s`, so
    /// `w` scales by `1/s²`) must leave the normalized residuals unchanged.
    #[test]
    fn normalized_residuals_do_not_depend_on_measurement_units() {
        let h = redundant_example();
        let mut z = vec![3.0, 2.0, 5.0, 1.0];
        z[2] += 5.0;
        let weights = vec![1.0; 4];
        let est = wls_state_estimate(&h, &z, &weights).unwrap();
        let reference =
            largest_normalized_residual_test(&h, &weights, &est.residuals, 2.5).unwrap();

        for s in [1e-3_f64, 1e-7, 1e-9, 1e3]
        {
            let (m, n) = h.shape();
            let hs = Matrix::from_fn(m, n, |i, j| h[(i, j)] * s);
            let zs: Vec<f64> = z.iter().map(|v| v * s).collect();
            let ws = vec![1.0 / (s * s); 4];
            let est_s = wls_state_estimate(&hs, &zs, &ws).unwrap();
            let report = largest_normalized_residual_test(&hs, &ws, &est_s.residuals, 2.5).unwrap();
            assert_eq!(report.suspect_index, Some(2), "scale {s}: {report:?}");
            for (a, b) in report
                .normalized_residuals
                .iter()
                .zip(&reference.normalized_residuals)
            {
                assert_relative_eq!(*a, *b, epsilon = 1e-6, max_relative = 1e-6);
            }
        }
    }

    /// A critical measurement (no redundancy, `Ω_ii = 0`) keeps a normalized
    /// residual of 0 at any measurement scale.
    #[test]
    fn critical_measurement_has_zero_normalized_residual_at_any_scale() {
        for s in [1.0_f64, 1e-7, 1e5]
        {
            let h = Matrix::from_row_major(2, 2, vec![s, 0.0, 0.0, s]);
            let z = vec![3.0 * s, 2.0 * s];
            let ws = vec![1.0 / (s * s); 2];
            let est = wls_state_estimate(&h, &z, &ws).unwrap();
            let report = largest_normalized_residual_test(&h, &ws, &est.residuals, 2.5).unwrap();
            assert_eq!(report.normalized_residuals, vec![0.0, 0.0], "scale {s}");
            assert_eq!(report.suspect_index, None);
        }
    }
}
