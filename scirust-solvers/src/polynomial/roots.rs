//! Calcul des racines d'un polynôme par l'algorithme de Durand-Kerner
//! (a.k.a. Weierstrass).
//!
//! Itère sur toutes les racines simultanément. Chaque itération :
//!   z_i ← z_i - p(z_i) / Π_{j≠i} (z_i - z_j)
//!
//! Convergence quadratique près de racines simples ; converge globalement
//! depuis presque tous les points de départ, sur des polynômes modérément
//! mal conditionnés (racines entières bien séparées de degré ~5-10, voir
//! `tests::degree_5`).
//!
//! Les racines nulles exactes sont extraites d'abord, puis le polynôme est
//! mis à l'échelle (`x = s·y`, `s` puissance de deux bornant le module des
//! racines) : la tolérance est donc relative à l'échelle des racines et le
//! résultat ne dépend pas de l'unité de `x`.
//!
//! **Limite connue** : cette implémentation travaille sur les coefficients
//! sous forme développée (base monomiale), qui est numériquement instable
//! pour des degrés élevés — le polynôme de Wilkinson classique (degré 20,
//! racines 1..20) a des coefficients allant jusqu'à ~2.4×10¹⁸, et
//! l'évaluation de Horner en `f64` y produit des annulations catastrophiques.
//! L'itération n'y converge pas : [`durand_kerner_strict`] renvoie
//! [`SolverError::NoConvergence`] (ou [`SolverError::NanDetected`] si un pas
//! devient non fini) plutôt que des racines silencieusement fausses, tandis
//! que [`durand_kerner`] reste best-effort. Un tel polynôme a besoin d'une
//! méthode dédiée (déflation, arithmétique étendue, ou passage par la
//! matrice compagnon avec un solveur d'eigenvalues, comme le fait LAPACK).

use super::Polynomial;
use crate::{SolverError, SolverResult};

/// Trouve toutes les racines (complexes) via Durand-Kerner. Best-effort : si
/// `max_iter` est épuisé sans que le pas maximal descende sous `tol`, renvoie
/// quand même la meilleure estimation courante (voir [`durand_kerner_strict`]
/// pour une variante qui signale l'échec au lieu de le masquer).
/// Renvoie un Vec<(re, im)> de longueur `degree`.
///
/// **Mise à l'échelle** : les racines nulles exactes (coefficients de plus bas
/// degré nuls) sont extraites telles quelles, puis le polynôme restant est
/// réécrit en `x = s·y`, où `s` est une puissance de deux proche de la borne
/// `max_k |c_k / c_n|^(1/(n-k))` du module des racines. L'itération travaille
/// sur `y` (racines de module ≲ 2), donc `tol` est un critère **relatif à
/// l'échelle des racines** : le résultat ne dépend pas de l'unité dans laquelle
/// le polynôme est exprimé (racines ~1e-20 ou ~1e10 traitées comme ~1).
pub fn durand_kerner(p: &Polynomial, max_iter: usize, tol: f64) -> SolverResult<Vec<(f64, f64)>> {
    durand_kerner_table(p, max_iter, tol).map(|run| run.roots)
}

/// Variante stricte de [`durand_kerner`] : renvoie
/// [`SolverError::NoConvergence`] si le pas maximal n'est pas descendu sous
/// `tol` après `max_iter` itérations, plutôt que de renvoyer silencieusement
/// la meilleure estimation disponible. Comme pour [`durand_kerner`], `tol` et
/// le `residual` rapporté sont exprimés dans la variable mise à l'échelle
/// `y = x / s`, c'est-à-dire relativement à l'échelle des racines.
pub fn durand_kerner_strict(
    p: &Polynomial,
    max_iter: usize,
    tol: f64,
) -> SolverResult<Vec<(f64, f64)>> {
    let run = durand_kerner_table(p, max_iter, tol)?;
    if run.last_step < tol
    {
        Ok(run.roots)
    }
    else
    {
        Err(SolverError::NoConvergence {
            iterations: max_iter,
            residual: run.last_step,
        })
    }
}

/// Résultat interne d'une exécution de Durand-Kerner.
struct DkRun {
    /// Racines dans la variable d'origine `x`.
    roots: Vec<(f64, f64)>,
    /// Dernier pas maximal, dans la variable mise à l'échelle `y = x / s`.
    last_step: f64,
    /// Échelle `s` des racines (puissance de deux ; `1.0` si toutes nulles).
    scale: f64,
}

/// Multiplie `x` par `2^k` sans débordement intermédiaire de `2^k` lui-même
/// (exact tant que le résultat reste normal).
fn mul_pow2(mut x: f64, mut k: i64) -> f64 {
    const STEP: i64 = 1000;
    while k > STEP
    {
        x *= 2f64.powi(STEP as i32);
        k -= STEP;
    }
    while k < -STEP
    {
        x *= 2f64.powi(-STEP as i32);
        k += STEP;
    }
    x * 2f64.powi(k as i32)
}

/// Runs Durand-Kerner and returns the roots, the last max step (in the
/// scaled variable) and the root scale. A `last_step` of `0.0` for a
/// polynomial with no non-zero root, or a first-iteration
/// all-coincident-roots skip, is treated as converged by both callers above.
fn durand_kerner_table(p: &Polynomial, max_iter: usize, tol: f64) -> SolverResult<DkRun> {
    let n = p.degree();
    if n == 0
    {
        return Ok(DkRun {
            roots: Vec::new(),
            last_step: 0.0,
            scale: 1.0,
        });
    }
    let lead = *p.coeffs.last().unwrap();
    if lead == 0.0
    {
        return Err(SolverError::InvalidInput(
            "leading coefficient is zero".into(),
        ));
    }

    // Racines nulles exactes : c_0 = … = c_{m-1} = 0 ⇒ x^m divise p.
    let zeros = p.coeffs.iter().take_while(|&&c| c == 0.0).count();
    let q = &p.coeffs[zeros..];
    let d = q.len() - 1;
    let mut out: Vec<(f64, f64)> = vec![(0.0, 0.0); zeros];
    if d == 0
    {
        return Ok(DkRun {
            roots: out,
            last_step: 0.0,
            scale: 1.0,
        });
    }

    // Échelle s = 2^e ≥ max_k |q_k / q_d|^(1/(d-k)) (calculée en log2 pour
    // éviter tout débordement du rapport des coefficients).
    let log_lead = q[d].abs().log2();
    let bound = q[..d]
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != 0.0)
        .map(|(k, c)| (c.abs().log2() - log_lead) / (d - k) as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let e: i64 = if bound.is_finite()
    {
        bound.ceil() as i64
    }
    else
    {
        0
    };
    let scale = mul_pow2(1.0, e);

    // Polynôme unitaire en y = x / s : a_k = (q_k / q_d) · s^(k-d), |a_k| ≲ 1.
    let monic: Vec<f64> = q
        .iter()
        .enumerate()
        .map(|(k, &c)| {
            let shift = -e * (d - k) as i64;
            let r = c / q[d];
            if c == 0.0 || r.is_normal()
            {
                mul_pow2(r, shift)
            }
            else
            {
                mul_pow2(c, shift) / q[d]
            }
        })
        .collect();

    // Initialisation : racines de l'unité multipliées par 0.4 + 0.9i
    // (classique, évite les coïncidences avec des racines réelles)
    let mut z: Vec<(f64, f64)> = Vec::with_capacity(d);
    let base = (0.4_f64, 0.9_f64);
    let theta_step = 2.0 * std::f64::consts::PI / d as f64;
    for i in 0..d
    {
        let angle = theta_step * i as f64;
        let r = base.0.hypot(base.1);
        let phi = base.1.atan2(base.0);
        let total = phi + angle;
        z.push((r * total.cos(), r * total.sin()));
    }

    // Évaluation de p(z) (complexe) par Horner
    let eval_complex = |z: (f64, f64)| -> (f64, f64) {
        let mut acc = (0.0_f64, 0.0_f64);
        for &c in monic.iter().rev()
        {
            // acc = acc * z + c
            let (ar, ai) = acc;
            let (zr, zi) = z;
            let nr = ar * zr - ai * zi + c;
            let ni = ar * zi + ai * zr;
            acc = (nr, ni);
        }
        acc
    };

    let mut last_max_step = f64::INFINITY;
    for _ in 0..max_iter
    {
        let mut max_step = 0.0_f64;
        for i in 0..d
        {
            // Calcule p(z_i)
            let pz = eval_complex(z[i]);
            // Calcule le produit Π_{j != i} (z_i - z_j)
            let mut denom = (1.0_f64, 0.0_f64);
            for j in 0..d
            {
                if i == j
                {
                    continue;
                }
                let dr = z[i].0 - z[j].0;
                let di = z[i].1 - z[j].1;
                // denom *= (dr, di)
                let (or_, oi) = denom;
                denom = (or_ * dr - oi * di, or_ * di + oi * dr);
            }
            let dmag = denom.0.hypot(denom.1);
            if dmag < 1e-30
            {
                continue; // racines confondues — saute cette itération
            }
            // step = p(z) / denom  (division complexe)
            let nr = (pz.0 * denom.0 + pz.1 * denom.1) / (dmag * dmag);
            let ni = (pz.1 * denom.0 - pz.0 * denom.1) / (dmag * dmag);
            let step_mag = nr.hypot(ni);
            // `f64::max` treats NaN as the *smaller* operand (IEEE 754
            // minNum/maxNum semantics), so `max_step.max(step_mag)` would
            // silently drop a NaN step instead of propagating it — a
            // catastrophic-cancellation breakdown (huge coefficients, e.g.
            // the classic degree-20 Wilkinson polynomial expanded to
            // monomial form) would then report `last_max_step` near 0 and
            // look "converged" while every root is NaN. Reject explicitly.
            if !step_mag.is_finite()
            {
                return Err(SolverError::NanDetected {
                    iter: i,
                    value: step_mag,
                });
            }
            z[i].0 -= nr;
            z[i].1 -= ni;
            max_step = max_step.max(step_mag);
        }
        last_max_step = max_step;
        if max_step < tol
        {
            break;
        }
    }
    // Hors-tolérance, `durand_kerner` renvoie quand même la meilleure
    // estimation ; `durand_kerner_strict` la rejette via `last_step`.
    out.extend(
        z.into_iter()
            .map(|(re, im)| (mul_pow2(re, e), mul_pow2(im, e))),
    );
    Ok(DkRun {
        roots: out,
        last_step: last_max_step,
        scale,
    })
}

/// Alias par défaut pour `durand_kerner` avec tolérances raisonnables.
pub fn roots(p: &Polynomial) -> SolverResult<Vec<(f64, f64)>> {
    durand_kerner(p, 200, 1e-12)
}

/// Filtre les racines réelles : celles dont la partie imaginaire est < eps
/// (seuil absolu, dans l'unité de `x`). Trie par ordre croissant.
///
/// Deux racines réelles distantes de moins de `1e-6 · min(1, s)` (où `s` est
/// l'échelle des racines décrite dans [`durand_kerner`]) sont fusionnées : le
/// seuil de fusion n'est jamais plus grossier que l'ancien seuil absolu `1e-6`,
/// et suit l'échelle du polynôme lorsque ses racines sont petites.
pub fn real_roots(p: &Polynomial, eps: f64) -> SolverResult<Vec<f64>> {
    let run = durand_kerner_table(p, 200, 1e-12)?;
    let reals: Vec<f64> = run
        .roots
        .into_iter()
        .filter(|&(_, im)| im.abs() < eps)
        .map(|(re, _)| re)
        .collect();
    Ok(sort_and_dedup_reals(reals, 1e-6 * run.scale.min(1.0)))
}

/// Trie par ordre croissant puis dédupe les racines réelles distantes de
/// moins de `merge_tol`.
///
/// Utilise `total_cmp` plutôt que `partial_cmp().unwrap()` afin de garder un
/// ordre total déterministe même si une racine dégénère en `NaN` (ce qui peut
/// arriver sur des polynômes mal conditionnés) : `unwrap` paniquerait alors.
fn sort_and_dedup_reals(mut reals: Vec<f64>, merge_tol: f64) -> Vec<f64> {
    reals.sort_by(|a, b| a.total_cmp(b));
    // Dédupe (deux racines complexes conjuguées peuvent donner deux versions
    // de la même racine réelle si la partie imaginaire est sous epsilon)
    reals.dedup_by(|a, b| (*a - *b).abs() < merge_tol);
    reals
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn linear_root() {
        // 2x + 4 = 0  →  x = -2
        let p = Polynomial::new(vec![4.0, 2.0]);
        let r = real_roots(&p, 1e-8).unwrap();
        assert_eq!(r.len(), 1);
        assert_relative_eq!(r[0], -2.0, epsilon = 1e-10);
    }

    #[test]
    fn quadratic_two_real_roots() {
        // x² - 3x + 2 = 0  →  x = 1 et x = 2
        let p = Polynomial::new(vec![2.0, -3.0, 1.0]);
        let r = real_roots(&p, 1e-8).unwrap();
        assert_eq!(r.len(), 2);
        assert_relative_eq!(r[0], 1.0, epsilon = 1e-9);
        assert_relative_eq!(r[1], 2.0, epsilon = 1e-9);
    }

    #[test]
    fn quadratic_complex_roots() {
        // x² + 1 = 0  →  pas de racine réelle, racines ±i
        let p = Polynomial::new(vec![1.0, 0.0, 1.0]);
        let r = real_roots(&p, 1e-8).unwrap();
        assert!(r.is_empty());
        // Mais en complexe
        let rc = roots(&p).unwrap();
        assert_eq!(rc.len(), 2);
        // Une racine doit avoir une partie imaginaire proche de +1, l'autre -1
        let imags: Vec<f64> = rc.iter().map(|(_, im)| *im).collect();
        let has_pos_i = imags.iter().any(|&im| (im - 1.0).abs() < 1e-6);
        let has_neg_i = imags.iter().any(|&im| (im + 1.0).abs() < 1e-6);
        assert!(has_pos_i && has_neg_i);
    }

    #[test]
    fn cubic_three_real() {
        // (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6
        let p = Polynomial::from_descending(vec![1.0, -6.0, 11.0, -6.0]);
        let r = real_roots(&p, 1e-6).unwrap();
        assert_eq!(r.len(), 3);
        assert_relative_eq!(r[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(r[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(r[2], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn cubic_irrational_root() {
        // x³ - 2x - 5 = 0, une seule racine réelle ≈ 2.0945514815
        let p = Polynomial::from_descending(vec![1.0, 0.0, -2.0, -5.0]);
        let r = real_roots(&p, 1e-6).unwrap();
        assert_eq!(r.len(), 1);
        assert_relative_eq!(r[0], 2.094_551_481_542_326_6, epsilon = 1e-6);
    }

    #[test]
    fn sort_with_nan_does_not_panic() {
        // Régression : avec `partial_cmp().unwrap()`, un NaN dans la liste des
        // racines réelles (possible sur un polynôme mal conditionné, où la
        // partie imaginaire ~0 mais la partie réelle dégénère) faisait paniquer
        // le tri. `total_cmp` garantit un ordre total sans panique.
        let sorted = sort_and_dedup_reals(vec![3.0, f64::NAN, 1.0, 2.0], 1e-6);
        // Les valeurs finies restent triées ; NaN est ordonné de façon
        // déterministe (en fin de liste) sans panique.
        assert!(sorted.len() >= 3);
        let finite: Vec<f64> = sorted.iter().copied().filter(|x| x.is_finite()).collect();
        assert_eq!(finite, vec![1.0, 2.0, 3.0]);
        assert!(sorted.iter().any(|x| x.is_nan()));
    }

    #[test]
    fn degree_5() {
        // (x-1)(x-2)(x-3)(x-4)(x-5)
        let mut p = Polynomial::from_descending(vec![1.0, -1.0]);
        for k in 2..=5
        {
            // Multiplie p par (x - k)
            let q = Polynomial::from_descending(vec![1.0, -(k as f64)]);
            let mut new_coeffs = vec![0.0; p.coeffs.len() + q.coeffs.len() - 1];
            for (i, &a) in p.coeffs.iter().enumerate()
            {
                for (j, &b) in q.coeffs.iter().enumerate()
                {
                    new_coeffs[i + j] += a * b;
                }
            }
            p = Polynomial::new(new_coeffs);
        }
        let r = real_roots(&p, 1e-5).unwrap();
        assert_eq!(r.len(), 5);
        for (i, expected) in (1..=5).enumerate()
        {
            assert_relative_eq!(r[i], expected as f64, epsilon = 1e-4);
        }
    }

    /// Builds `Π_{k=1}^{degree} (x - k)` in monomial form (the Wilkinson
    /// polynomial family) — huge, ill-conditioned coefficients at higher
    /// degree, which is exactly what breaks a monomial-basis root finder.
    fn wilkinson_style(degree: i64) -> Polynomial {
        let mut p = Polynomial::from_descending(vec![1.0, -1.0]);
        for k in 2..=degree
        {
            let q = Polynomial::from_descending(vec![1.0, -(k as f64)]);
            let mut new_coeffs = vec![0.0; p.coeffs.len() + q.coeffs.len() - 1];
            for (i, &a) in p.coeffs.iter().enumerate()
            {
                for (j, &b) in q.coeffs.iter().enumerate()
                {
                    new_coeffs[i + j] += a * b;
                }
            }
            p = Polynomial::new(new_coeffs);
        }
        p
    }

    /// Regression test for a P2 audit finding: `f64::max` treats NaN as the
    /// *smaller* operand, so a NaN step used to get silently absorbed into
    /// `max_step` and reported as "converged" with every root equal to NaN.
    /// A NaN coefficient reproduces such a step; it must surface as
    /// `NanDetected` instead of fake convergence.
    #[test]
    fn nan_step_reports_nan_instead_of_fake_convergence() {
        let p = Polynomial::new(vec![1.0, f64::NAN, 1.0]);
        let result = durand_kerner(&p, 500, 1e-6);
        assert!(
            matches!(result, Err(SolverError::NanDetected { .. })),
            "expected NanDetected on a NaN step, got {result:?}"
        );
    }

    /// The classic degree-20 Wilkinson polynomial (monomial coefficients up
    /// to ~2.4e18) is out of reach of Durand-Kerner in `f64`. Before root
    /// scaling the iteration overflowed (`NanDetected`); with scaling it stays
    /// finite but does not converge — the strict variant must still refuse to
    /// report it as converged.
    #[test]
    fn wilkinson_degree_20_is_not_reported_as_converged() {
        let p = wilkinson_style(20);
        let result = durand_kerner_strict(&p, 500, 1e-12);
        assert!(
            matches!(
                result,
                Err(SolverError::NoConvergence { .. } | SolverError::NanDetected { .. })
            ),
            "expected a convergence failure on Wilkinson-20, got {result:?}"
        );
    }

    /// Monic polynomial with the given real roots, in monomial form.
    fn from_roots(rs: &[f64]) -> Polynomial {
        let mut c = vec![1.0];
        for &r in rs
        {
            let mut next = vec![0.0; c.len() + 1];
            for (i, &a) in c.iter().enumerate()
            {
                next[i + 1] += a;
                next[i] -= a * r;
            }
            c = next;
        }
        Polynomial::new(c)
    }

    fn assert_roots_rel(got: &[f64], want: &[f64], rel: f64) {
        assert_eq!(got.len(), want.len(), "got {got:?}, want {want:?}");
        for (g, w) in got.iter().zip(want)
        {
            assert!(
                ((g - w) / w).abs() < rel,
                "root {g:e} vs {w:e} (got {got:?})"
            );
        }
    }

    /// Regression: the fixed `1e-6` merge threshold of `real_roots` fused
    /// distinct roots of small magnitude — (x − 1e-7)(x − 2e-7) returned only
    /// `[1e-7]`, and four roots 1e-8…4e-8 collapsed to `[1e-8]`.
    #[test]
    fn real_roots_keeps_distinct_small_roots() {
        let r = real_roots(&from_roots(&[1e-7, 2e-7]), 1e-8).unwrap();
        assert_roots_rel(&r, &[1e-7, 2e-7], 1e-9);

        let want = [1e-8, 2e-8, 3e-8, 4e-8];
        let r = real_roots(&from_roots(&want), 1e-12).unwrap();
        assert_roots_rel(&r, &want, 1e-9);
    }

    /// Regression: the absolute step tolerance stopped the iteration long
    /// before tiny roots were resolved — (x − 1e-20)(x − 2e-20) returned a
    /// single "root" ≈ −9.9e-14.
    #[test]
    fn tiny_scale_roots_are_relatively_accurate() {
        let r = real_roots(&from_roots(&[1e-20, 2e-20]), 1e-30).unwrap();
        assert_roots_rel(&r, &[1e-20, 2e-20], 1e-9);
    }

    /// Regression: with an absolute step tolerance of 1e-12, roots of order
    /// 1e10 can never satisfy the strict criterion, so `durand_kerner_strict`
    /// reported `NoConvergence` although the roots were accurate.
    #[test]
    fn strict_converges_for_large_scale_roots() {
        let want = [1e10, 2e10, 3e10];
        let mut z = durand_kerner_strict(&from_roots(&want), 200, 1e-12).unwrap();
        z.sort_by(|a, b| a.0.total_cmp(&b.0));
        let re: Vec<f64> = z.iter().map(|&(re, _)| re).collect();
        assert_roots_rel(&re, &want, 1e-9);
        assert!(z.iter().all(|&(_, im)| im.abs() < 1e-3));
    }

    /// Regression: a leading coefficient below the absolute cut-off `1e-30`
    /// was rejected as zero, although `1e-40·x − 1e-40` has the root `1`.
    #[test]
    fn tiny_leading_coefficient_is_accepted() {
        let r = real_roots(&Polynomial::new(vec![-1e-40, 1e-40]), 1e-8).unwrap();
        assert_roots_rel(&r, &[1.0], 1e-12);
    }

    /// Zero low-order coefficients give exact zero roots (x² + x → {−1, 0}).
    #[test]
    fn zero_roots_are_exact() {
        let r = real_roots(&Polynomial::new(vec![0.0, 1.0, 1.0]), 1e-8).unwrap();
        assert_eq!(r.len(), 2);
        assert!((r[0] + 1.0).abs() < 1e-12);
        assert_eq!(r[1], 0.0);
        let z = roots(&Polynomial::new(vec![0.0, 0.0, 1.0])).unwrap();
        assert_eq!(z, vec![(0.0, 0.0), (0.0, 0.0)]);
    }

    /// `durand_kerner` stays best-effort on ordinary (non-NaN) non-convergence;
    /// `durand_kerner_strict` reports the same case as `NoConvergence`.
    #[test]
    fn durand_kerner_strict_reports_non_convergence() {
        let p = wilkinson_style(6);
        // A single iteration is nowhere near enough to converge, but stays
        // numerically tame (unlike degree 20): no NaN breakdown, just an
        // ordinary "ran out of iterations" case.
        let lenient = durand_kerner(&p, 1, 1e-12);
        assert!(
            lenient.is_ok(),
            "best-effort variant must not error: {lenient:?}"
        );

        let strict = durand_kerner_strict(&p, 1, 1e-12);
        assert!(
            matches!(strict, Err(SolverError::NoConvergence { .. })),
            "expected NoConvergence, got {strict:?}"
        );
    }
}
