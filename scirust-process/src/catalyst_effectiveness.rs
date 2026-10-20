//! Catalyse hétérogène — module de Thiele et facteur d'efficacité d'un grain de
//! catalyseur, reliant la vitesse observée à la vitesse intrinsèque limitée par
//! la diffusion interne.
//!
//! ```text
//! module de Thiele (réaction d'ordre 1)
//!   φ = L · sqrt(k / D_e)                                      `sans dimension`
//! facteur d'efficacité d'une plaque (slab, ordre 1)
//!   η = tanh(φ) / φ                                            `sans dimension`
//! facteur d'efficacité d'une sphère (ordre 1)
//!   η = (3/φ) · (1/tanh(φ) − 1/φ) = (3/φ) · (coth(φ) − 1/φ)    `sans dimension`
//! vitesse observée
//!   r_obs = η · r_int                                          `mol·m⁻³·s⁻¹`
//! ```
//!
//! `L` longueur caractéristique du grain (rapport volume/surface V/S) `m`, `k`
//! constante de vitesse intrinsèque d'ordre 1 `s⁻¹`, `D_e` diffusivité effective
//! dans le grain `m²·s⁻¹` ; `φ` module de Thiele `sans dimension` ; `η` facteur
//! d'efficacité `sans dimension` ; `r_int` vitesse intrinsèque (sur grain sans
//! limitation) `mol·m⁻³·s⁻¹` ; `r_obs` vitesse observée `mol·m⁻³·s⁻¹`.
//!
//! **Limite honnête** : grain de catalyseur **isotherme**, réaction d'**ordre 1**.
//! La **diffusivité effective** `D_e`, la **constante de vitesse intrinsèque** `k`
//! et la **longueur caractéristique** `L = V/S` sont **FOURNIES** par l'appelant :
//! aucune constante cinétique, diffusivité, enthalpie, volatilité ou isotherme
//! d'adsorption n'est calculée ni inventée ici. `η < 1` traduit la limitation par
//! la **diffusion interne** (φ élevé) ; ce module ne traite ni la diffusion
//! **externe** (film) ni les effets **non isothermes** (grain à température non
//! uniforme).

/// Module de Thiele pour une réaction d'ordre 1
/// `φ = L · sqrt(k / D_e)` (sans dimension), rapport de la vitesse de réaction à
/// la vitesse de diffusion dans le grain.
///
/// `characteristic_length` (L) longueur caractéristique V/S `m`, `rate_constant`
/// (k) constante de vitesse intrinsèque d'ordre 1 `s⁻¹`, `effective_diffusivity`
/// (D_e) diffusivité effective `m²·s⁻¹`.
///
/// Panique si `L < 0`, si `k < 0`, ou si `D_e ≤ 0`.
pub fn cateff_thiele_modulus_first_order(
    characteristic_length: f64,
    rate_constant: f64,
    effective_diffusivity: f64,
) -> f64 {
    assert!(
        characteristic_length >= 0.0,
        "L ≥ 0 requis (longueur caractéristique V/S)"
    );
    assert!(
        rate_constant >= 0.0,
        "k ≥ 0 requis (constante de vitesse intrinsèque)"
    );
    assert!(
        effective_diffusivity > 0.0,
        "D_e > 0 requis (diffusivité effective)"
    );
    characteristic_length * (rate_constant / effective_diffusivity).sqrt()
}

/// Facteur d'efficacité d'une plaque (slab) pour une réaction d'ordre 1
/// `η = tanh(φ) / φ` (sans dimension) ; tend vers 1 quand `φ → 0` (pas de
/// limitation diffusionnelle) et vers `1/φ` quand `φ` est grand.
///
/// `thiele_modulus` (φ) module de Thiele `sans dimension`.
///
/// Panique si `φ ≤ 0`.
pub fn cateff_effectiveness_slab(thiele_modulus: f64) -> f64 {
    assert!(
        thiele_modulus > 0.0,
        "φ > 0 requis (module de Thiele) pour le facteur d'efficacité"
    );
    thiele_modulus.tanh() / thiele_modulus
}

/// Facteur d'efficacité d'une sphère pour une réaction d'ordre 1
/// `η = (3/φ) · (1/tanh(φ) − 1/φ)` (sans dimension), avec `1/tanh(φ) = coth(φ)` ;
/// tend vers 1 quand `φ → 0` et vers `3/φ` quand `φ` est grand.
///
/// `thiele_modulus` (φ) module de Thiele `sans dimension`.
///
/// Pour `φ < 0.5`, la différence `coth(φ) − 1/φ ≈ φ/3` se perd par annulation
/// entre deux termes voisins de `1/φ` (erreur relative 4e-8 à `φ = 1e-4`, η > 1
/// vers `φ ≈ 1e-5`, nul pour `φ ≤ 1e-8`) ; η y est donc évalué par son
/// développement en série `η = 3 Σₙ≥₁ 2²ⁿ B₂ₙ φ²ⁿ⁻² / (2n)!
/// = 1 − φ²/15 + 2φ⁴/315 − …` (nombres de Bernoulli `B₂ₙ`), tronqué à 12
/// termes (reste < 1e-19 pour `φ < 0.5`).
///
/// Panique si `φ ≤ 0`.
pub fn cateff_effectiveness_sphere(thiele_modulus: f64) -> f64 {
    assert!(
        thiele_modulus > 0.0,
        "φ > 0 requis (module de Thiele) pour le facteur d'efficacité"
    );
    if thiele_modulus < SPHERE_SERIES_SWITCH
    {
        // Horner en φ² sur les coefficients 3·2²ⁿ·B₂ₙ/(2n)!.
        let x2 = thiele_modulus * thiele_modulus;
        return SPHERE_SERIES
            .iter()
            .rev()
            .fold(0.0, |acc, &c| acc.mul_add(x2, c));
    }
    (3.0 / thiele_modulus) * (1.0 / thiele_modulus.tanh() - 1.0 / thiele_modulus)
}

/// Seuil sous lequel [`cateff_effectiveness_sphere`] passe à la série : au-delà,
/// la forme fermée ne perd qu'un facteur ≈ 3/φ² ≤ 12 ulp par annulation.
const SPHERE_SERIES_SWITCH: f64 = 0.5;

/// Coefficients `3·2²ⁿ·B₂ₙ/(2n)!`, `n = 1..=12`, du développement
/// `η_sphère(φ) = Σ cₙ φ²⁽ⁿ⁻¹⁾` (1, −1/15, 2/315, −1/1575, …).
const SPHERE_SERIES: [f64; 12] = [
    1.0,
    -0.066_666_666_666_666_67,
    0.006_349_206_349_206_349,
    -0.000_634_920_634_920_634_9,
    6.413_339_746_673_08e-5,
    -6.493_212_842_419_191e-6,
    6.577_784_355_562_133e-7,
    -6.664_382_636_993_903e-8,
    6.752_353_955_042_697_5e-9,
    -6.841_545_361_377_655e-10,
    6.931_929_779_700_787e-11,
    -7.023_512_045_947_465_5e-12,
];

/// Vitesse observée sur le grain `r_obs = η · r_int` (mol·m⁻³·s⁻¹), vitesse
/// intrinsèque pondérée par le facteur d'efficacité.
///
/// `effectiveness_factor` (η) facteur d'efficacité `sans dimension`, `intrinsic_rate`
/// (r_int) vitesse intrinsèque `mol·m⁻³·s⁻¹`.
///
/// Panique si `η < 0` ou si `η > 1` (le facteur d'efficacité isotherme d'ordre 1
/// est borné à ``0, 1``).
pub fn cateff_observed_rate(effectiveness_factor: f64, intrinsic_rate: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&effectiveness_factor),
        "0 ≤ η ≤ 1 requis (facteur d'efficacité isotherme d'ordre 1)"
    );
    effectiveness_factor * intrinsic_rate
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn thiele_realistic_case() {
        // L = 5 mm, k = 0.16 s⁻¹, D_e = 1e-6 m²/s :
        //   φ = 5e-3 · sqrt(0.16 / 1e-6) = 5e-3 · sqrt(160000)
        //     = 5e-3 · 400 = 2.0.
        let phi = cateff_thiele_modulus_first_order(5.0e-3_f64, 0.16_f64, 1.0e-6_f64);
        assert_relative_eq!(phi, 2.0, max_relative = 1e-3);
    }

    #[test]
    fn thiele_scales_linearly_with_characteristic_length() {
        // φ ∝ L : doubler la longueur caractéristique double le module de Thiele.
        let single = cateff_thiele_modulus_first_order(1.0e-3_f64, 5.0_f64, 2.0e-7_f64);
        let double = cateff_thiele_modulus_first_order(2.0e-3_f64, 5.0_f64, 2.0e-7_f64);
        assert_relative_eq!(double, 2.0 * single, max_relative = 1e-12);
    }

    #[test]
    fn slab_effectiveness_realistic_case() {
        // φ = 2 : η = tanh(2)/2 = 0.96402758.../2 = 0.48201379...
        let eta = cateff_effectiveness_slab(2.0_f64);
        assert_relative_eq!(eta, 0.482_013_79, max_relative = 1e-3);
    }

    #[test]
    fn sphere_effectiveness_realistic_case() {
        // φ = 2 : η = (3/2)·(1/tanh(2) − 1/2)
        //          = 1.5·(1.0373147... − 0.5) = 1.5·0.5373147... = 0.8059721...
        let eta = cateff_effectiveness_sphere(2.0_f64);
        assert_relative_eq!(eta, 0.805_972_1, max_relative = 1e-3);
    }

    #[test]
    fn effectiveness_tends_to_one_for_small_thiele() {
        // φ → 0 : plaque et sphère tendent toutes deux vers η = 1 (régime
        // chimique, aucune limitation diffusionnelle).
        let eta_slab = cateff_effectiveness_slab(1.0e-3_f64);
        let eta_sphere = cateff_effectiveness_sphere(1.0e-3_f64);
        assert_relative_eq!(eta_slab, 1.0, max_relative = 1e-3);
        assert_relative_eq!(eta_sphere, 1.0, max_relative = 1e-3);
    }

    #[test]
    fn sphere_effectiveness_keeps_full_precision_for_small_thiele() {
        // Références mpmath (50 chiffres, arrondies en f64) de η = (3/φ)·(coth φ − 1/φ).
        // Avant correction, la forme fermée donnait η(1e-4) = 0.99999995655,
        // η(1e-5) = 1.00000034, η(3e-8) = 1.49 et η(1e-8) = 0.
        let cases = [
            (1.0e-9, 1.0),
            (1.0e-8, 1.0),
            (3.0e-8, 0.999_999_999_999_999_9),
            (1.0e-7, 0.999_999_999_999_999_3),
            (1.0e-6, 0.999_999_999_999_933_4),
            (1.0e-5, 0.999_999_999_993_333_3),
            (1.0e-4, 0.999_999_999_333_333_4),
            (1.0e-2, 0.999_993_333_396_824_7),
            (0.1, 0.999_333_967_619_688_3),
            (0.3, 0.994_050_969_884_082_6),
            (0.5, 0.983_720_482_431_917_1),
            (0.7, 0.968_786_602_419_432_2),
            (1.0, 0.939_105_856_497_993_9),
            (2.0, 0.805_972_081_091_322_1),
            (10.0, 0.270_000_001_236_692_16),
        ];
        for (phi, want) in cases
        {
            let eta = cateff_effectiveness_sphere(phi);
            assert_relative_eq!(eta, want, max_relative = 1e-14);
            assert!(eta <= 1.0, "η({phi}) = {eta} > 1");
        }
        // Continuité au changement de formule (φ = 0.5).
        let below = cateff_effectiveness_sphere(0.5 - 1e-12);
        let above = cateff_effectiveness_sphere(0.5);
        assert_relative_eq!(below, above, max_relative = 1e-13);
    }

    #[test]
    fn observed_rate_accepts_sphere_effectiveness_for_small_thiele() {
        // η(φ = 1e-5) valait 1.00000034 > 1 et faisait paniquer
        // cateff_observed_rate ; η = 1 − φ²/15 + … (référence mpmath).
        let eta = cateff_effectiveness_sphere(1.0e-5_f64);
        assert_relative_eq!(
            cateff_observed_rate(eta, 7.5_f64),
            7.5 * 0.999_999_999_993_333_3,
            max_relative = 1e-14
        );
    }

    #[test]
    fn observed_rate_is_effectiveness_times_intrinsic() {
        // r_obs = η · r_int : η = 0.48201379 sur r_int = 10 mol·m⁻³·s⁻¹
        //   ⇒ r_obs = 4.8201379..., et η = 1 restitue la vitesse intrinsèque.
        let eta = cateff_effectiveness_slab(2.0_f64);
        assert_relative_eq!(
            cateff_observed_rate(eta, 10.0_f64),
            4.820_137_9,
            max_relative = 1e-3
        );
        assert_relative_eq!(
            cateff_observed_rate(1.0_f64, 7.5_f64),
            7.5,
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "φ > 0 requis")]
    fn slab_panics_on_nonpositive_thiele() {
        // φ = 0 ⇒ division par φ = 0 ⇒ entrée rejetée.
        let _ = cateff_effectiveness_slab(0.0_f64);
    }
}
