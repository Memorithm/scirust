//! Fiabilité des **systèmes** — associations de composants : série, parallèle
//! (redondance active) et redondance `k`-sur-`n`.
//!
//! ```text
//! série       R = Π Ri
//! parallèle   R = 1 − Π(1 − Ri)
//! k-sur-n     R = Σ_{i=k}^{n} C(n,i)·r^i·(1 − r)^{n−i}   (composants identiques r)
//! ```
//!
//! `Ri` fiabilités des composants (dans `[0, 1]`), `r` fiabilité d'un composant
//! identique, `k` nombre minimal de composants sains, `n` nombre total. Un
//! système **série** est moins fiable que son maillon le plus faible ; un système
//! **parallèle** est plus fiable que son meilleur composant.
//!
//! **Convention** : fiabilités sans dimension dans `[0, 1]`. **Limite honnête** :
//! composants **indépendants** ; `k`-sur-`n` suppose des composants **identiques**
//! (mêmes `r`). Ne modélise ni les défaillances de cause commune, ni la
//! redondance passive (stand-by) avec commutation.

/// Fiabilité d'un système **série** `R = Π Ri`.
///
/// Panique si un `Ri` sort de `[0, 1]`.
pub fn series_reliability(component_reliabilities: &[f64]) -> f64 {
    component_reliabilities
        .iter()
        .map(|&r| {
            assert!(
                (0.0..=1.0).contains(&r),
                "chaque fiabilité doit être dans [0, 1]"
            );
            r
        })
        .product()
}

/// Fiabilité d'un système **parallèle** `R = 1 − Π(1 − Ri)`.
///
/// Panique si un `Ri` sort de `[0, 1]`.
pub fn parallel_reliability(component_reliabilities: &[f64]) -> f64 {
    let product_unreliability: f64 = component_reliabilities
        .iter()
        .map(|&r| {
            assert!(
                (0.0..=1.0).contains(&r),
                "chaque fiabilité doit être dans [0, 1]"
            );
            1.0 - r
        })
        .product();
    1.0 - product_unreliability
}

/// Fiabilité d'une redondance `k`-sur-`n` à composants identiques
/// `R = Σ_{i=k}^{n} C(n,i)·r^i·(1 − r)^{n−i}`.
///
/// Chaque terme est évalué en logarithmes (`ln C(n,i) + i·ln r +
/// (n−i)·ln(1−r)`) : aucun coefficient entier `C(n,i)` ne déborde et aucun
/// produit `∞·0` n'apparaît, le résultat reste fini pour tout `n`.
///
/// Panique si `r` sort de `[0, 1]`, `n == 0` ou `k > n`.
pub fn k_out_of_n_reliability(component_reliability: f64, k: u32, n: u32) -> f64 {
    assert!(
        (0.0..=1.0).contains(&component_reliability),
        "r doit être dans [0, 1]"
    );
    assert!(n > 0 && k <= n, "0 < k ≤ n requis");
    let r = component_reliability;
    // Cas dégénérés : ln 0 = −∞ rendrait 0·(−∞) indéterminé ci-dessous.
    if k == 0 || r == 1.0
    {
        return 1.0;
    }
    if r == 0.0
    {
        return 0.0;
    }
    let ln_r = r.ln();
    let ln_q = (-r).ln_1p();
    let nf = f64::from(n);
    // ln C(n, k), puis récurrence ln C(n, i+1) = ln C(n, i) + ln((n−i)/(i+1)).
    let mut ln_c = ln_binomial(n, k);
    let mut sum = 0.0;
    for i in k..=n
    {
        let fi = f64::from(i);
        sum += (ln_c + fi * ln_r + (nf - fi) * ln_q).exp();
        if i < n
        {
            ln_c += ((nf - fi) / (fi + 1.0)).ln();
        }
    }
    sum.min(1.0)
}

/// `ln C(n, k)` par somme de logarithmes (aucun entier intermédiaire, donc
/// aucun débordement).
fn ln_binomial(n: u32, k: u32) -> f64 {
    let k = k.min(n - k);
    (0..k)
        .map(|i| (f64::from(n - i) / f64::from(i + 1)).ln())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn series_is_less_than_weakest() {
        // Série de 0,9 et 0,8 → 0,72 < 0,8.
        let r = series_reliability(&[0.9, 0.8]);
        assert_relative_eq!(r, 0.72, epsilon = 1e-12);
        assert!(r < 0.8);
    }

    #[test]
    fn parallel_beats_best_component() {
        // Parallèle de 0,9 et 0,8 → 1 − 0,1·0,2 = 0,98 > 0,9.
        let r = parallel_reliability(&[0.9, 0.8]);
        assert_relative_eq!(r, 0.98, epsilon = 1e-12);
        assert!(r > 0.9);
    }

    #[test]
    fn k_equals_n_is_series_product() {
        // n-sur-n (tous requis) = série de composants identiques = rⁿ.
        assert_relative_eq!(
            k_out_of_n_reliability(0.9, 3, 3),
            0.9f64.powi(3),
            epsilon = 1e-12
        );
    }

    #[test]
    fn one_out_of_n_is_parallel() {
        // 1-sur-n = parallèle de n composants identiques.
        let koon = k_out_of_n_reliability(0.9, 1, 3);
        let par = parallel_reliability(&[0.9, 0.9, 0.9]);
        assert_relative_eq!(koon, par, epsilon = 1e-9);
    }

    #[test]
    fn two_out_of_three_voting() {
        // 2-sur-3 (redondance majoritaire) : R = 3r²−2r³. Pour r=0,9 → 0,972.
        let r = 0.9;
        assert_relative_eq!(
            k_out_of_n_reliability(r, 2, 3),
            3.0 * r * r - 2.0 * r * r * r,
            epsilon = 1e-9
        );
    }

    #[test]
    fn large_n_matches_binomial_tail_reference() {
        // `C(n, i)` for i near n/2 no longer fits in u64 from n = 63 on; the
        // sum must still match the exact binomial tail P(X ≥ k), X ~ B(n, r)
        // (references: mpmath at 40 digits rounded to f64, equal to scipy.stats.binom.sf(k−1, n, r)).
        let cases = [
            (0.5, 32, 64, 0.549_673_376_873_983_4),
            (0.3, 20, 64, 0.460_207_865_514_165),
            (0.95, 60, 64, 0.784_357_043_608_697_1),
            (0.6, 51, 101, 0.979_103_308_995_299_5),
            (0.9, 1800, 2000, 0.518_820_400_591_036_2),
        ];
        for (r, k, n, want) in cases
        {
            let got = k_out_of_n_reliability(r, k, n);
            assert_relative_eq!(got, want, max_relative = 1e-12);
        }
        // 1-sur-70 = parallèle : 1 − (1 − r)⁷⁰, sans débordement.
        let par = k_out_of_n_reliability(0.2, 1, 70);
        assert_relative_eq!(par, 1.0 - 0.8f64.powi(70), max_relative = 1e-12);
    }

    #[test]
    fn degenerate_component_reliability() {
        assert_eq!(k_out_of_n_reliability(1.0, 64, 64), 1.0);
        assert_eq!(k_out_of_n_reliability(0.0, 1, 64), 0.0);
        assert_eq!(k_out_of_n_reliability(0.0, 0, 64), 1.0);
        assert_eq!(k_out_of_n_reliability(1.0, 0, 64), 1.0);
    }

    #[test]
    #[should_panic(expected = "0 < k ≤ n")]
    fn k_greater_than_n_panics() {
        k_out_of_n_reliability(0.9, 4, 3);
    }
}
