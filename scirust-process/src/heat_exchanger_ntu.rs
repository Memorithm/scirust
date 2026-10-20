//! Échangeur de chaleur — **méthode ε-NTU** (efficacité / nombre d'unités de
//! transfert), pour prédire le flux thermique échangé quand les **températures de
//! sortie sont inconnues**.
//!
//! ```text
//! nombre d'unités  NTU = U·A / Cmin                              `sans dimension`
//! rapport capacité Cr  = Cmin / Cmax                             `sans dimension`
//!
//! efficacité contre-courant (Cr ≠ 1)
//!   ε = (1 − exp`−NTU·(1 − Cr)`) / (1 − Cr·exp`−NTU·(1 − Cr)`)   `sans dimension`
//! efficacité contre-courant (Cr = 1)
//!   ε = NTU / (1 + NTU)                                          `sans dimension`
//! efficacité co-courant
//!   ε = (1 − exp`−NTU·(1 + Cr)`) / (1 + Cr)                      `sans dimension`
//!
//! flux thermique   Q = ε·Cmin·(Th,in − Tc,in)                   `W`
//! ```
//!
//! `U` coefficient global de transfert `W/(m²·K)`, `A` aire d'échange `m²`,
//! `Cmin`/`Cmax` débits calorifiques (`ṁ·cp`) minimal et maximal des deux
//! courants `W/K`, `NTU` nombre d'unités de transfert `sans dimension`, `Cr`
//! rapport des capacités calorifiques `sans dimension, dans ``0, 1```, `ε`
//! efficacité `sans dimension, dans ``0, 1```, `Th,in`/`Tc,in` températures
//! d'entrée du fluide chaud et froid `K`, `Q` puissance thermique échangée `W`.
//! `ΔTmax = Th,in − Tc,in` est l'écart de température maximal théorique `K`.
//!
//! **Limite honnête** : le coefficient global `U`, l'aire `A` et les débits
//! calorifiques `Cmin`/`Cmax` (donc les débits massiques et les capacités
//! thermiques `cp`) sont **fournis par l'appelant** — mesurés, tirés d'une
//! corrélation de transfert (Nu, Colburn, Dittus-Boelter…) ou d'une table :
//! aucune propriété physique ni coefficient de transfert n'est inventé. La
//! **corrélation d'efficacité dépend de la configuration** (contre-courant,
//! co-courant, courants croisés, faisceau-calandre…) et c'est **l'appelant** qui
//! choisit la fonction adaptée à son échangeur. Les relations supposent un
//! **régime permanent**, un `U` **constant** sur toute la surface, des débits
//! calorifiques constants (pas de changement de phase, `cp` invariant) et des
//! pertes vers l'extérieur négligeables.

/// Nombre d'unités de transfert `NTU = U·A / Cmin` `sans dimension`.
///
/// `overall_coefficient` `U` coefficient global de transfert `W/(m²·K)`,
/// `area` `A` aire d'échange `m²`, `min_heat_capacity_rate` `Cmin` plus petit
/// des deux débits calorifiques `ṁ·cp` `W/K`. Le `NTU` mesure la « taille
/// thermique » de l'échangeur relative au courant limitant.
///
/// Panique si `overall_coefficient` ou `area` est négatif ou non fini, ou si
/// `min_heat_capacity_rate` n'est pas strictement positif (division).
pub fn ntu_number(overall_coefficient: f64, area: f64, min_heat_capacity_rate: f64) -> f64 {
    assert!(
        overall_coefficient.is_finite() && overall_coefficient >= 0.0,
        "le coefficient global U doit être fini et positif ou nul (W/(m²·K))"
    );
    assert!(
        area.is_finite() && area >= 0.0,
        "l'aire d'échange doit être finie et positive ou nulle (m²)"
    );
    assert!(
        min_heat_capacity_rate > 0.0,
        "le débit calorifique minimal Cmin doit être strictement positif (W/K)"
    );
    overall_coefficient * area / min_heat_capacity_rate
}

/// Rapport des capacités calorifiques `Cr = Cmin / Cmax` `sans dimension`.
///
/// `min_heat_capacity_rate` `Cmin` débit calorifique minimal `W/K`,
/// `max_heat_capacity_rate` `Cmax` débit calorifique maximal `W/K`. Par
/// construction `Cr` appartient à ``0, 1`` : `Cr = 0` correspond à un courant
/// changeant de phase (`Cmax → ∞`), `Cr = 1` à deux courants équilibrés.
///
/// Panique si l'un des débits n'est pas fini, si `min_heat_capacity_rate` est
/// négatif, si `max_heat_capacity_rate` n'est pas strictement positif, ou si
/// `min_heat_capacity_rate > max_heat_capacity_rate` (définition violée).
pub fn ntu_capacity_ratio(min_heat_capacity_rate: f64, max_heat_capacity_rate: f64) -> f64 {
    assert!(
        min_heat_capacity_rate.is_finite() && min_heat_capacity_rate >= 0.0,
        "le débit calorifique minimal Cmin doit être fini et positif ou nul (W/K)"
    );
    assert!(
        max_heat_capacity_rate > 0.0,
        "le débit calorifique maximal Cmax doit être strictement positif (W/K)"
    );
    assert!(
        min_heat_capacity_rate <= max_heat_capacity_rate,
        "Cmin doit être inférieur ou égal à Cmax (par définition du rapport Cr)"
    );
    min_heat_capacity_rate / max_heat_capacity_rate
}

/// Efficacité d'un échangeur à **contre-courant**
/// `ε = (1 − exp`−NTU·(1 − Cr)`) / (1 − Cr·exp`−NTU·(1 − Cr)`)`, avec le cas
/// limite `ε = NTU/(1 + NTU)` lorsque `Cr = 1` `sans dimension`.
///
/// `ntu` `NTU` nombre d'unités de transfert `sans dimension`, `capacity_ratio`
/// `Cr` rapport des capacités calorifiques `sans dimension, dans ``0, 1```. Le
/// contre-courant est la configuration la plus efficace : `ε → 1` quand
/// `NTU → ∞`.
///
/// Panique si `ntu` est négatif ou non fini, ou si `capacity_ratio` sort de
/// ``0, 1``.
pub fn ntu_effectiveness_counterflow(ntu: f64, capacity_ratio: f64) -> f64 {
    assert!(
        ntu.is_finite() && ntu >= 0.0,
        "le NTU doit être fini et positif ou nul (sans dimension)"
    );
    assert!(
        capacity_ratio.is_finite() && (0.0..=1.0).contains(&capacity_ratio),
        "le rapport de capacités Cr doit être compris dans [0, 1] (sans dimension)"
    );
    if capacity_ratio >= 1.0
    {
        ntu / (1.0 + ntu)
    }
    else
    {
        // Forme équivalente sans annulation : avec m = expm1(−NTU·(1 − Cr)) < 0,
        // 1 − exp(…) = −m et 1 − Cr·exp(…) = (1 − Cr) − Cr·m (deux termes ≥ 0).
        // Reste précise quand NTU·(1 − Cr) → 0 et rejoint NTU/(1 + NTU) en Cr → 1.
        let one_minus_cr = 1.0 - capacity_ratio;
        let m = (-ntu * one_minus_cr).exp_m1();
        -m / (one_minus_cr - capacity_ratio * m)
    }
}

/// Efficacité d'un échangeur à **co-courant** (courants parallèles)
/// `ε = (1 − exp`−NTU·(1 + Cr)`) / (1 + Cr)` `sans dimension`.
///
/// `ntu` `NTU` nombre d'unités de transfert `sans dimension`, `capacity_ratio`
/// `Cr` rapport des capacités calorifiques `sans dimension, dans ``0, 1```. Le
/// co-courant est borné : `ε → 1/(1 + Cr)` quand `NTU → ∞`, toujours inférieur
/// ou égal au contre-courant.
///
/// Panique si `ntu` est négatif ou non fini, ou si `capacity_ratio` sort de
/// ``0, 1``.
pub fn ntu_effectiveness_parallel(ntu: f64, capacity_ratio: f64) -> f64 {
    assert!(
        ntu.is_finite() && ntu >= 0.0,
        "le NTU doit être fini et positif ou nul (sans dimension)"
    );
    assert!(
        capacity_ratio.is_finite() && (0.0..=1.0).contains(&capacity_ratio),
        "le rapport de capacités Cr doit être compris dans [0, 1] (sans dimension)"
    );
    // −expm1(x) = 1 − exp(x) sans annulation quand NTU·(1 + Cr) → 0.
    -(-ntu * (1.0 + capacity_ratio)).exp_m1() / (1.0 + capacity_ratio)
}

/// Flux thermique échangé `Q = ε·Cmin·(Th,in − Tc,in)` `W` — la puissance réelle
/// vaut l'efficacité fois le maximum thermodynamique `Cmin·ΔTmax`.
///
/// `effectiveness` `ε` efficacité de l'échangeur `sans dimension, dans ``0, 1```,
/// `min_heat_capacity_rate` `Cmin` débit calorifique minimal `W/K`, `hot_inlet`
/// `Th,in` température d'entrée du fluide chaud `K`, `cold_inlet` `Tc,in`
/// température d'entrée du fluide froid `K`. Un résultat positif suppose
/// `Th,in ≥ Tc,in` (sens de transfert du chaud vers le froid).
///
/// Panique si `effectiveness` sort de ``0, 1``, si `min_heat_capacity_rate` est
/// négatif ou non fini, ou si l'une des températures n'est pas finie.
pub fn ntu_duty(
    effectiveness: f64,
    min_heat_capacity_rate: f64,
    hot_inlet: f64,
    cold_inlet: f64,
) -> f64 {
    assert!(
        effectiveness.is_finite() && (0.0..=1.0).contains(&effectiveness),
        "l'efficacité ε doit être comprise dans [0, 1] (sans dimension)"
    );
    assert!(
        min_heat_capacity_rate.is_finite() && min_heat_capacity_rate >= 0.0,
        "le débit calorifique minimal Cmin doit être fini et positif ou nul (W/K)"
    );
    assert!(
        hot_inlet.is_finite() && cold_inlet.is_finite(),
        "les températures d'entrée doivent être finies (K)"
    );
    effectiveness * min_heat_capacity_rate * (hot_inlet - cold_inlet)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::E;

    #[test]
    fn ntu_number_known_case() {
        // U = 500 W/(m²·K), A = 4 m², Cmin = 1000 W/K
        // → NTU = 500·4/1000 = 2000/1000 = 2.
        assert_relative_eq!(ntu_number(500.0, 4.0, 1000.0), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn ntu_number_scales_inversely_with_cmin() {
        // À U·A fixé, NTU ∝ 1/Cmin : doubler Cmin divise NTU par deux.
        let n1 = ntu_number(300.0, 5.0, 750.0);
        let n2 = ntu_number(300.0, 5.0, 1500.0);
        assert_relative_eq!(n2, n1 / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn capacity_ratio_bounds() {
        // Cmin = 800, Cmax = 2000 → Cr = 0,4 ; courants équilibrés → Cr = 1.
        assert_relative_eq!(ntu_capacity_ratio(800.0, 2000.0), 0.4, epsilon = 1e-12);
        assert_relative_eq!(ntu_capacity_ratio(1200.0, 1200.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn counterflow_beats_parallel() {
        // À NTU et Cr identiques, le contre-courant est toujours >= au co-courant.
        let ntu = 1.5_f64;
        let cr = 0.6_f64;
        let counter = ntu_effectiveness_counterflow(ntu, cr);
        let parallel = ntu_effectiveness_parallel(ntu, cr);
        assert!(counter >= parallel);
        // Les deux efficacités restent dans [0, 1].
        assert!((0.0..=1.0).contains(&counter));
        assert!((0.0..=1.0).contains(&parallel));
    }

    #[test]
    fn counterflow_cr_zero_matches_phase_change_limit() {
        // Cr = 0 (condenseur/évaporateur) : ε = 1 − exp(−NTU), indépendant de la
        // configuration. Avec NTU = 1 → ε = 1 − 1/e.
        let expected = 1.0 - 1.0 / E;
        assert_relative_eq!(
            ntu_effectiveness_counterflow(1.0, 0.0),
            expected,
            epsilon = 1e-12
        );
        // Co-courant à Cr = 0 donne la même limite.
        assert_relative_eq!(
            ntu_effectiveness_parallel(1.0, 0.0),
            expected,
            epsilon = 1e-12
        );
    }

    #[test]
    fn counterflow_balanced_case() {
        // Cr = 1 exactement : branche limite ε = NTU/(1+NTU). NTU = 3 → ε = 3/4.
        // Recalcul : 3/(1+3) = 3/4 = 0,75.
        assert_relative_eq!(
            ntu_effectiveness_counterflow(3.0, 1.0),
            0.75,
            epsilon = 1e-12
        );
    }

    #[test]
    fn duty_known_case() {
        // ε = 0,5 ; Cmin = 2000 W/K ; Th,in = 400 K ; Tc,in = 300 K.
        // ΔTmax = 400 − 300 = 100 K → Q = 0,5·2000·100 = 100 000 W.
        // Recalcul : 0,5 × 2000 = 1000 ; 1000 × 100 = 100 000 W.
        assert_relative_eq!(
            ntu_duty(0.5, 2000.0, 400.0, 300.0),
            100_000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn duty_is_linear_in_effectiveness() {
        // À Cmin et températures fixés, Q ∝ ε : tripler ε triple Q.
        let q1 = ntu_duty(0.2, 1500.0, 380.0, 300.0);
        let q3 = ntu_duty(0.6, 1500.0, 380.0, 300.0);
        assert_relative_eq!(q3, 3.0 * q1, epsilon = 1e-9);
    }

    #[test]
    fn counterflow_keeps_precision_for_small_ntu_times_one_minus_cr() {
        // Régression : 1 − exp(−NTU·(1 − Cr)) perdait sa précision quand
        // NTU·(1 − Cr) → 0 (soustraction de deux nombres voisins de 1).
        // Références calculées en précision étendue (mpmath, 50 chiffres) :
        // NTU = 1e−6, Cr = 1 − 1,01e−9 → ε = 9,99999000001000e−7
        // (l'ancienne forme donnait 9,893e−7, écart relatif ≈ 1,1 %).
        let cr = 1.0 - 1.01e-9;
        assert_relative_eq!(
            ntu_effectiveness_counterflow(1.0e-6, cr),
            9.999_990_000_010_005e-7,
            epsilon = 0.0,
            max_relative = 1e-12
        );
        // NTU = 1e−6, Cr = 1 − 2e−9 → ε = 9,99999000001001e−7 (ancien : 9,992e−7).
        assert_relative_eq!(
            ntu_effectiveness_counterflow(1.0e-6, 1.0 - 2.0e-9),
            9.999_990_000_010_01e-7,
            epsilon = 0.0,
            max_relative = 1e-12
        );
        // Cr = 0, NTU = 1e−12 → ε = −expm1(−1e−12) = 9,999999999995e−13
        // (ancien : 9,99978e−13, écart relatif ≈ 2,2e−5).
        assert_relative_eq!(
            ntu_effectiveness_counterflow(1.0e-12, 0.0),
            9.999_999_999_995e-13,
            epsilon = 0.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn counterflow_is_continuous_across_balanced_limit() {
        // Juste sous Cr = 1, la forme générale doit rejoindre la limite
        // NTU/(1 + NTU) sans saut.
        for &ntu in &[1.0e-6, 0.5, 3.0, 1.0e3]
        {
            let balanced = ntu_effectiveness_counterflow(ntu, 1.0);
            let near = ntu_effectiveness_counterflow(ntu, 1.0 - f64::EPSILON);
            assert_relative_eq!(near, balanced, epsilon = 0.0, max_relative = 1e-12);
        }
    }

    #[test]
    fn parallel_keeps_precision_for_small_ntu() {
        // Régression : même perte de précision en co-courant pour NTU·(1 + Cr) → 0.
        // NTU = 1e−12, Cr = 0 → ε = 9,999999999995e−13 (ancien : 9,99978e−13).
        assert_relative_eq!(
            ntu_effectiveness_parallel(1.0e-12, 0.0),
            9.999_999_999_995e-13,
            epsilon = 0.0,
            max_relative = 1e-12
        );
        // NTU = 1e−10, Cr = 1 → ε = −expm1(−2e−10)/2 = 9,999999999e−11
        // (ancien : 1,0000000827e−10, écart relatif ≈ 8,3e−8).
        assert_relative_eq!(
            ntu_effectiveness_parallel(1.0e-10, 1.0),
            9.999_999_999e-11,
            epsilon = 0.0,
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "Cmin doit être inférieur ou égal à Cmax")]
    fn capacity_ratio_rejects_cmin_above_cmax() {
        // Cmin > Cmax viole la définition de Cr : entrée rejetée.
        ntu_capacity_ratio(3000.0, 2000.0);
    }
}
