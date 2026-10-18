//! Contrôle de réception par **échantillonnage** (plan simple, loi **binomiale**) :
//! probabilité d'acceptation d'un lot et **qualité moyenne après contrôle** (AOQ).
//!
//! ```text
//! coefficient binomial   C(n,k) = n! / (k!·(n−k)!)
//! prob. d'acceptation    Pa = Σ_{k=0}^{c} C(n,k)·p^k·(1−p)^(n−k)
//! qualité moyenne sortie  AOQ ≈ Pa·p           (lot supposé infini)
//! ```
//!
//! `n` (`sample_size`) taille de l'échantillon, `c` (`acceptance_number`) nombre
//! d'acceptation (nombre maximal de défectueux toléré), `p` (`defect_fraction`)
//! proportion de défectueux du lot (`0 ≤ p ≤ 1`), `Pa` probabilité d'accepter le
//! lot (sans dimension), `AOQ` proportion moyenne de défectueux acceptés
//! (sans dimension). Toutes ces grandeurs sont des fractions ou des comptes,
//! sans unité physique.
//!
//! **Limite honnête** : plan d'échantillonnage **simple** basé sur la loi
//! binomiale (lot supposé infini, tirages indépendants) ; ni double ni multiple
//! échantillonnage, ni correction hypergéométrique pour lot fini. Les paramètres
//! du plan `n` et `c` sont **fournis** par l'appelant (issus d'une table ISO 2859
//! ou d'un cahier des charges) ; ce module ne les dimensionne pas et n'invente
//! aucune valeur « par défaut ».

/// Coefficient binomial `C(n,k) = n!/(k!·(n−k)!)`, calculé de façon exacte et
/// stable par produit itératif (pas de factorielle intermédiaire).
///
/// La valeur est renvoyée en `f64` : elle dépasse `f64::MAX` (résultat `+∞`)
/// dès que `C(n,k) > 1,8·10³⁰⁸`, par exemple pour `k ≈ n/2` à partir de
/// `n ≈ 1030`. [`sampling_binomial_pmf`] ne passe donc pas par ce coefficient.
///
/// Panique si `k > n`.
pub fn sampling_binomial_coefficient(n: u32, k: u32) -> f64 {
    assert!(
        k <= n,
        "le rang k doit vérifier k <= n pour le coefficient binomial"
    );
    // On exploite la symétrie C(n,k) = C(n,n−k) pour minimiser le nombre d'itérations.
    let k = k.min(n - k);
    let mut result = 1.0_f64;
    for i in 0..k
    {
        result *= f64::from(n - i);
        result /= f64::from(i + 1);
    }
    result
}

/// `ln C(n,k)` par somme de logarithmes de rapports (aucun coefficient
/// intermédiaire, donc aucun débordement). Suppose `k <= n`.
fn ln_binomial_coefficient(n: u32, k: u32) -> f64 {
    let k = k.min(n - k);
    (0..k)
        .map(|i| (f64::from(n - i) / f64::from(i + 1)).ln())
        .sum()
}

/// Probabilité qu'un tirage binomial donne exactement `k` défectueux :
/// `C(n,k)·p^k·(1−p)^(n−k)`.
///
/// Le terme est évalué en logarithmes (`ln C(n,k) + k·ln p + (n−k)·ln(1−p)`) :
/// le résultat reste fini et dans `[0, 1]` pour tout `n`, alors que `C(n,k)`
/// seul dépasse `f64::MAX` pour `k ≈ n/2` dès `n ≈ 1030` (tailles d'échantillon
/// ISO 2859-1 de 1250 et 2000 comprises).
///
/// Panique si `p` sort de `[0, 1]` ou si `k > n`.
pub fn sampling_binomial_pmf(sample_size: u32, k: u32, defect_fraction: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&defect_fraction),
        "la proportion de défectueux doit être dans [0, 1]"
    );
    assert!(
        k <= sample_size,
        "le nombre de défectueux k doit vérifier k <= n"
    );
    let n = sample_size;
    let p = defect_fraction;
    // Cas dégénérés : ln 0 = −∞ rendrait 0·(−∞) indéterminé ci-dessous.
    if p == 0.0
    {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    if p == 1.0
    {
        return if k == n { 1.0 } else { 0.0 };
    }
    let kf = f64::from(k);
    let ln_term = ln_binomial_coefficient(n, k) + kf * p.ln() + (f64::from(n) - kf) * (-p).ln_1p();
    ln_term.exp().min(1.0)
}

/// Probabilité d'acceptation d'un lot pour un plan simple :
/// `Pa = Σ_{k=0}^{c} C(n,k)·p^k·(1−p)^(n−k)`.
///
/// Chaque terme est évalué en logarithmes comme dans
/// [`sampling_binomial_pmf`], avec la récurrence
/// `ln C(n,k+1) = ln C(n,k) + ln((n−k)/(k+1))` ; le résultat reste fini et
/// dans `[0, 1]` pour tout `n`.
///
/// Panique si `p` sort de `[0, 1]` ou si `c > n`.
pub fn probability_of_acceptance_binomial(
    sample_size: u32,
    acceptance_number: u32,
    defect_fraction: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&defect_fraction),
        "la proportion de défectueux doit être dans [0, 1]"
    );
    assert!(
        acceptance_number <= sample_size,
        "le nombre d'acceptation c doit vérifier c <= n"
    );
    let n = sample_size;
    let p = defect_fraction;
    // Cas dégénérés : lot parfait toujours accepté ; lot entièrement défectueux
    // accepté seulement si c = n.
    if p == 0.0 || acceptance_number == n
    {
        return 1.0;
    }
    if p == 1.0
    {
        return 0.0;
    }
    let ln_p = p.ln();
    let ln_q = (-p).ln_1p();
    let nf = f64::from(n);
    let mut ln_c = 0.0; // ln C(n, 0)
    let mut sum = 0.0;
    for k in 0..=acceptance_number
    {
        let kf = f64::from(k);
        sum += (ln_c + kf * ln_p + (nf - kf) * ln_q).exp();
        ln_c += ((nf - kf) / (kf + 1.0)).ln();
    }
    sum.min(1.0)
}

/// Qualité moyenne après contrôle (Average Outgoing Quality) `AOQ ≈ Pa·p`,
/// approximation valable pour un lot supposé infini.
///
/// Panique si `paccept` ou `defect_fraction` sort de `[0, 1]`.
pub fn average_outgoing_quality(paccept: f64, defect_fraction: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&paccept),
        "la probabilité d'acceptation doit être dans [0, 1]"
    );
    assert!(
        (0.0..=1.0).contains(&defect_fraction),
        "la proportion de défectueux doit être dans [0, 1]"
    );
    paccept * defect_fraction
}

/// Probabilité de **rejet** (risque de refus) du lot `Pr = 1 − Pa`.
///
/// Panique si `p` sort de `[0, 1]` ou si `c > n`.
pub fn probability_of_rejection_binomial(
    sample_size: u32,
    acceptance_number: u32,
    defect_fraction: f64,
) -> f64 {
    1.0 - probability_of_acceptance_binomial(sample_size, acceptance_number, defect_fraction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn binomial_coefficient_symmetry_and_known_values() {
        // Symétrie C(n,k) = C(n,n−k) et valeurs connues C(5,2)=10, C(6,3)=20.
        assert_relative_eq!(sampling_binomial_coefficient(5, 2), 10.0, epsilon = 1e-9);
        assert_relative_eq!(
            sampling_binomial_coefficient(5, 2),
            sampling_binomial_coefficient(5, 3),
            epsilon = 1e-9
        );
        assert_relative_eq!(sampling_binomial_coefficient(6, 3), 20.0, epsilon = 1e-9);
        assert_relative_eq!(sampling_binomial_coefficient(8, 0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn pmf_sums_to_one_over_all_k() {
        // La somme de toutes les probabilités binomiales vaut 1 (loi de probabilité).
        let n = 12;
        let p = 0.17;
        let total: f64 = (0..=n).map(|k| sampling_binomial_pmf(n, k, p)).sum();
        assert_relative_eq!(total, 1.0, max_relative = 1e-12);
    }

    #[test]
    fn acceptance_bounds_at_extreme_defect_fractions() {
        // Lot parfait (p=0) → toujours accepté ; c=n → toujours accepté quel que soit p.
        assert_relative_eq!(
            probability_of_acceptance_binomial(20, 2, 0.0),
            1.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            probability_of_acceptance_binomial(20, 20, 0.35),
            1.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn acceptance_realistic_case_matches_direct_binomial() {
        // Plan n=10, c=1, p=0.10 : Pa = (0.9)^10 + 10·0.1·(0.9)^9.
        let pa = probability_of_acceptance_binomial(10, 1, 0.10);
        let expected = 0.9_f64.powi(10) + 10.0 * 0.10 * 0.9_f64.powi(9);
        assert_relative_eq!(pa, expected, max_relative = 1e-12);
    }

    #[test]
    fn acceptance_and_rejection_are_complementary() {
        // Pa + Pr = 1 (réciprocité) pour un plan quelconque.
        let pa = probability_of_acceptance_binomial(25, 3, 0.08);
        let pr = probability_of_rejection_binomial(25, 3, 0.08);
        assert_relative_eq!(pa + pr, 1.0, max_relative = 1e-12);
    }

    #[test]
    fn aoq_is_proportional_to_defect_fraction() {
        // AOQ = Pa·p : à Pa fixé, doubler p double l'AOQ.
        let pa = 0.6;
        assert_relative_eq!(
            average_outgoing_quality(pa, 0.04),
            2.0 * average_outgoing_quality(pa, 0.02),
            max_relative = 1e-12
        );
        assert_relative_eq!(average_outgoing_quality(0.5, 0.03), 0.015, epsilon = 1e-12);
    }

    #[test]
    fn pmf_large_sample_matches_binomial_reference() {
        // Pour n ≳ 1030, C(n,k) seul dépasse f64::MAX autour de k ≈ n/2 : la pmf
        // doit rester finie et égale à la loi binomiale exacte (références :
        // mpmath à 50 chiffres, arrondies en f64). n = 1250 et 2000 sont des
        // tailles d'échantillon ISO 2859-1 (lettres-codes Q et R).
        let cases = [
            (2000, 1000, 0.5, 0.017_839_011_145_854_32),
            (1250, 500, 0.4, 0.023_028_081_301_696_677),
            (1500, 300, 0.2, 0.025_744_103_678_851_077),
            (3150, 1000, 0.3, 0.001_590_013_994_250_316_6),
            (2000, 40, 0.02, 0.063_586_057_975_971_2),
            // p^k seul sous-déborde (0,15⁴⁰⁰ < 5·10⁻³²⁴) alors que le terme est
            // représentable.
            (1000, 400, 0.15, 6.078_579_890_408_046e-82),
        ];
        for (n, k, p, want) in cases
        {
            let got = sampling_binomial_pmf(n, k, p);
            assert_relative_eq!(got, want, max_relative = 1e-10);
        }
    }

    #[test]
    fn acceptance_large_sample_matches_binomial_cdf_reference() {
        // Pa = P(X ≤ c), X ~ B(n, p) (références : mpmath à 50 chiffres).
        let cases = [
            (2000, 1000, 0.5, 0.508_919_505_572_927_2),
            (1250, 520, 0.4, 0.881_563_484_721_862_1),
            (1250, 600, 0.45, 0.984_516_674_761_360_8),
            (2000, 21, 0.008, 0.911_628_410_315_821_8),
        ];
        for (n, c, p, want) in cases
        {
            let pa = probability_of_acceptance_binomial(n, c, p);
            assert_relative_eq!(pa, want, max_relative = 1e-10);
            let pr = probability_of_rejection_binomial(n, c, p);
            assert_relative_eq!(pr, 1.0 - want, max_relative = 1e-8);
        }
        // c = n : lot toujours accepté, y compris quand C(n, n/2) déborde.
        assert_eq!(probability_of_acceptance_binomial(2000, 2000, 0.001), 1.0);
        assert_eq!(probability_of_acceptance_binomial(2000, 2000, 0.5), 1.0);
    }

    #[test]
    fn pmf_degenerate_defect_fractions() {
        assert_eq!(sampling_binomial_pmf(2000, 0, 0.0), 1.0);
        assert_eq!(sampling_binomial_pmf(2000, 1, 0.0), 0.0);
        assert_eq!(sampling_binomial_pmf(2000, 2000, 1.0), 1.0);
        assert_eq!(sampling_binomial_pmf(2000, 1999, 1.0), 0.0);
        assert_eq!(probability_of_acceptance_binomial(2000, 1999, 1.0), 0.0);
        assert_eq!(probability_of_acceptance_binomial(2000, 0, 0.0), 1.0);
    }

    #[test]
    #[should_panic(expected = "proportion de défectueux")]
    fn defect_fraction_above_one_panics() {
        probability_of_acceptance_binomial(10, 1, 1.5);
    }
}
