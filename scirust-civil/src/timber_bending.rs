//! **Structure bois — flexion simple** d'une poutre (Eurocode 5, EN 1995-1-1) :
//! résistance de calcul en flexion, contrainte de flexion, coefficient d'effet
//! d'échelle (hauteur) `kh` et taux de travail.
//!
//! ```text
//! résistance de calcul   fm,d = kmod · kh · fm,k / γ_M
//! contrainte de flexion  σm,d = M_d / W
//! coefficient de hauteur kh   = min( (h_ref / h)^s , kh,max )   si h < h_ref
//!                        kh   = 1                               si h ≥ h_ref
//! taux de travail        η    = σm,d / fm,d
//! ```
//!
//! `kmod` coefficient de modification (classe de service × durée de la charge,
//! sans dimension), `kh` coefficient d'effet d'échelle sur la hauteur (sans
//! dimension), `fm,k` résistance caractéristique en flexion du bois (MPa),
//! `γ_M` coefficient partiel de sécurité sur le matériau (sans dimension),
//! `fm,d` résistance de calcul en flexion (MPa) ; `M_d` moment fléchissant de
//! calcul (N·mm), `W` module de flexion (élastique) de la section (mm³), `σm,d`
//! contrainte de flexion de calcul (MPa) ; `h_ref` hauteur de référence (mm),
//! `h` hauteur réelle de la section fléchie (mm), `s` exposant de l'effet
//! d'échelle (sans dimension), `kh,max` valeur plafond de `kh` (sans dimension) ;
//! `η` taux de travail (sans dimension, `η ≤ 1` = section vérifiée).
//!
//! **Convention** : unités **N, mm, MPa** (1 MPa = 1 N/mm²), cohérentes entre
//! elles (Eurocode) ; les moments s'expriment donc en **N·mm** (1 kN·m = 10⁶
//! N·mm), les modules de flexion en **mm³** et les contraintes/résistances en
//! **MPa**. Types `f64`.
//!
//! **Limite honnête** : **flexion simple** d'une poutre bois autour d'un seul
//! axe — l'**instabilité** (déversement `kcrit`, flambement des poteaux) n'est
//! **pas** traitée, de même que le **cisaillement**, la **compression
//! transversale** et la flexion déviée. Le coefficient `kmod` (couple classe de
//! service / durée de charge), la **résistance caractéristique** `fm,k` **et le
//! coefficient partiel de sécurité** `γ_M` sont **fournis par l'appelant**
//! d'après l'**Eurocode 5 (EN 1995-1-1)** et son **Annexe Nationale** — aucune
//! valeur « par défaut » n'est inventée. Les paramètres de l'effet d'échelle
//! (`h_ref`, `s`, `kh,max`, p. ex. 150 mm, 0,2 et 1,3 pour le bois massif) sont
//! eux aussi **normatifs** et fournis par l'appelant. Conformément à
//! l'EN 1995-1-1 (éq. 3.1, 3.2 et 3.3), `kh` ne s'applique qu'aux hauteurs
//! **inférieures** à `h_ref` : au-delà, `kh = 1` (pas de minoration).

/// Résistance de calcul en flexion `fm,d = kmod · kh · fm,k / γ_M` (MPa)
/// (Eurocode 5, EN 1995-1-1 §2.4.1 et §6.1.6).
///
/// `kmod` coefficient de modification (sans dimension), `kh` coefficient d'effet
/// d'échelle (sans dimension), `characteristic_bending_strength` = `fm,k` (MPa),
/// `gamma_m` = `γ_M` (sans dimension) fourni par l'Eurocode 5 et son Annexe
/// Nationale ; renvoie la résistance de calcul (MPa).
///
/// Panique si `kmod < 0`, si `kh < 0`, si `characteristic_bending_strength < 0`
/// ou si `gamma_m <= 0` (division par zéro).
pub fn timber_design_bending_strength(
    kmod: f64,
    kh: f64,
    characteristic_bending_strength: f64,
    gamma_m: f64,
) -> f64 {
    assert!(
        kmod >= 0.0,
        "le coefficient de modification kmod doit être ≥ 0"
    );
    assert!(
        kh >= 0.0,
        "le coefficient d'effet d'échelle kh doit être ≥ 0"
    );
    assert!(
        characteristic_bending_strength >= 0.0,
        "la résistance caractéristique fm,k doit être ≥ 0"
    );
    assert!(
        gamma_m > 0.0,
        "le coefficient partiel γ_M doit être strictement positif"
    );
    kmod * kh * characteristic_bending_strength / gamma_m
}

/// Contrainte de flexion de calcul `σm,d = M_d / W` (MPa) (Eurocode 5,
/// EN 1995-1-1 §6.1.6).
///
/// `design_moment` = `M_d` moment fléchissant de calcul (N·mm), `section_modulus`
/// = `W` module de flexion élastique de la section (mm³) ; renvoie la contrainte
/// de flexion (MPa, car N·mm / mm³ = N/mm² = MPa).
///
/// Panique si `section_modulus <= 0` (division par zéro).
pub fn timber_bending_stress(design_moment: f64, section_modulus: f64) -> f64 {
    assert!(
        section_modulus > 0.0,
        "le module de flexion W doit être strictement positif"
    );
    design_moment / section_modulus
}

/// Coefficient d'effet d'échelle sur la hauteur
/// `kh = min( (h_ref / h)^s , kh,max )` si `h < h_ref`, et `kh = 1` si
/// `h ≥ h_ref` (sans dimension) (Eurocode 5, EN 1995-1-1 §3.2 éq. 3.1, §3.3
/// éq. 3.2, §3.4 éq. 3.3).
///
/// L'Eurocode 5 autorise une **majoration** de `fm,k` pour les hauteurs
/// inférieures à la hauteur de référence ; il ne prévoit **aucune minoration**
/// pour les hauteurs supérieures. Par exemple, pour du bois massif
/// (`h_ref = 150 mm`, `s = 0,2`, `kh,max = 1,3`) et `h = 200 mm`, `kh = 1`
/// (et non `(150/200)^0,2 ≈ 0,9441`).
///
/// `reference_depth` = `h_ref` hauteur de référence (mm, p. ex. 150 mm pour le
/// bois massif), `actual_depth` = `h` hauteur réelle de la section fléchie (mm),
/// `exponent` = `s` exposant de l'effet d'échelle (sans dimension, p. ex. 0,2),
/// `maximum` = `kh,max` valeur plafond (sans dimension, p. ex. 1,3) ; tous
/// **fournis par l'Eurocode 5** et son Annexe Nationale selon le matériau.
/// Renvoie le coefficient `kh`.
///
/// Panique si `reference_depth <= 0`, si `actual_depth <= 0` (base de la
/// puissance non définie) ou si `maximum <= 0`.
pub fn timber_size_factor_depth(
    reference_depth: f64,
    actual_depth: f64,
    exponent: f64,
    maximum: f64,
) -> f64 {
    assert!(
        reference_depth > 0.0,
        "la hauteur de référence h_ref doit être strictement positive"
    );
    assert!(
        actual_depth > 0.0,
        "la hauteur réelle h doit être strictement positive"
    );
    assert!(
        maximum > 0.0,
        "le plafond kh,max doit être strictement positif"
    );
    if actual_depth >= reference_depth
    {
        // EN 1995-1-1 éq. 3.1-3.3 : pas d'effet d'échelle au-delà de h_ref.
        return 1.0;
    }
    (reference_depth / actual_depth).powf(exponent).min(maximum)
}

/// Taux de travail en flexion `η = σm,d / fm,d` (sans dimension), rapport de la
/// contrainte de calcul à la résistance de calcul (`η ≤ 1` = section vérifiée)
/// (Eurocode 5, EN 1995-1-1 §6.1.6).
///
/// `bending_stress` = `σm,d` contrainte de flexion de calcul (MPa),
/// `design_strength` = `fm,d` résistance de calcul en flexion (MPa) ; renvoie un
/// taux adimensionnel.
///
/// Panique si `bending_stress < 0` ou si `design_strength <= 0` (division par
/// zéro).
pub fn timber_utilisation(bending_stress: f64, design_strength: f64) -> f64 {
    assert!(
        bending_stress >= 0.0,
        "la contrainte de flexion σm,d doit être ≥ 0"
    );
    assert!(
        design_strength > 0.0,
        "la résistance de calcul fm,d doit être strictement positive"
    );
    bending_stress / design_strength
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn design_strength_reciprocity() {
        // Réciprocité : fm,d · γ_M / (kmod · kh) restitue la résistance
        // caractéristique fm,k.
        let (kmod, kh, fmk, gamma_m) = (0.8_f64, 1.0, 24.0, 1.3);
        let fmd = timber_design_bending_strength(kmod, kh, fmk, gamma_m);
        assert_relative_eq!(fmd * gamma_m / (kmod * kh), fmk, epsilon = 1e-9);
    }

    #[test]
    fn size_factor_unity_at_reference_depth() {
        // À hauteur égale à la référence, (h_ref/h)^s = 1^s = 1, donc kh = 1
        // (le plafond 1,3 n'intervient pas).
        let kh = timber_size_factor_depth(150.0, 150.0, 0.2, 1.3);
        assert_relative_eq!(kh, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn size_factor_capped_at_maximum() {
        // Section très basse (30 mm) : (150/30)^0,2 = 5^0,2 ≈ 1,3797 > 1,3,
        // donc kh est ramené au plafond kh,max = 1,3.
        let kh = timber_size_factor_depth(150.0, 30.0, 0.2, 1.3);
        assert_relative_eq!(kh, 1.3, epsilon = 1e-12);
        // Contrôle de la valeur non plafonnée : 5^0,2 ≈ 1,379729661.
        let raw = 5.0_f64.powf(0.2);
        assert_relative_eq!(raw, 1.379_729_661, epsilon = 1e-6);
        assert!(raw > 1.3);
    }

    #[test]
    fn bending_stress_proportionalities() {
        // σm,d = M_d / W : proportionnel au moment, inversement au module.
        // W = b·h²/6 = 100·200²/6 = 666 666,67 mm³ ; M_d = 5·10⁶ N·mm.
        let w = 100.0 * 200.0_f64.powi(2) / 6.0;
        let sigma = timber_bending_stress(5.0e6, w);
        assert_relative_eq!(sigma, 7.5, epsilon = 1e-9); // 5e6 / (4e6/6) = 7,5 MPa
        // Doubler le moment double la contrainte.
        let sigma2 = timber_bending_stress(10.0e6, w);
        assert_relative_eq!(sigma2, 2.0 * sigma, epsilon = 1e-9);
        // Doubler le module de flexion divise la contrainte par deux.
        let sigma_half = timber_bending_stress(5.0e6, 2.0 * w);
        assert_relative_eq!(sigma_half, sigma / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn utilisation_unity_when_stress_equals_strength() {
        // η = 1 quand σm,d = fm,d, et η = 0,5 à la moitié de la résistance.
        assert_relative_eq!(timber_utilisation(14.0, 14.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(timber_utilisation(7.0, 14.0), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn realistic_c24_solid_beam_case() {
        // Poutre bois massif C24, classe de service 1, charge de moyenne durée :
        //   kmod = 0,8 ; fm,k = 24 MPa ; γ_M = 1,3 (EN 1995-1-1, valeurs
        //   fournies par l'Eurocode 5 et l'Annexe Nationale).
        // Section rectangulaire b = 100 mm, h = 200 mm (h ≥ 150 mm → kh = 1).
        //   W    = b·h²/6 = 100·200²/6      = 666 666,67 mm³
        //   fm,d = 0,8·1,0·24 / 1,3         = 19,2 / 1,3 = 14,76923 MPa
        // Moment de calcul M_d = 5 kN·m = 5·10⁶ N·mm :
        //   σm,d = 5·10⁶ / 666 666,67       = 7,5 MPa
        //   η    = 7,5 / 14,76923 = 7,5·1,3 / 19,2 = 9,75 / 19,2 = 0,5078125
        let kh = timber_size_factor_depth(150.0, 200.0, 0.2, 1.3);
        // h = 200 mm ≥ h_ref = 150 mm : pas d'effet d'échelle, kh = 1.
        assert_relative_eq!(kh, 1.0, epsilon = 1e-12);

        let fmd = timber_design_bending_strength(0.8, kh, 24.0, 1.3);
        assert_relative_eq!(fmd, 14.769_230_769, epsilon = 1e-6);

        let w = 100.0 * 200.0_f64.powi(2) / 6.0;
        let sigma = timber_bending_stress(5.0e6, w);
        assert_relative_eq!(sigma, 7.5, epsilon = 1e-6);

        let eta = timber_utilisation(sigma, fmd);
        assert_relative_eq!(eta, 0.507_812_5, epsilon = 1e-6);
        assert!(eta < 1.0); // section vérifiée
    }

    #[test]
    fn size_factor_is_one_above_reference_depth() {
        // EN 1995-1-1 §3.2(3), éq. 3.1 : kh = min((150/h)^0,2 ; 1,3) ne vaut
        // que pour h < 150 mm ; au-delà, kh = 1. L'ancienne implémentation
        // renvoyait (150/h)^0,2 < 1, soit une minoration non prévue :
        //   h = 200 mm → 0,75^0,2 ≈ 0,9441 ; h = 600 mm → 0,25^0,2 ≈ 0,7579.
        for h in [150.0_f64, 150.000_001, 200.0, 600.0, 2000.0]
        {
            let kh = timber_size_factor_depth(150.0, h, 0.2, 1.3);
            assert_relative_eq!(kh, 1.0, max_relative = 1e-12);
        }
        // Lamellé-collé (§3.3, éq. 3.2 : h_ref = 600 mm, s = 0,1, plafond 1,1) :
        // h = 1200 mm → kh = 1 (et non 0,5^0,1 ≈ 0,9330).
        let kh_glulam = timber_size_factor_depth(600.0, 1200.0, 0.1, 1.1);
        assert_relative_eq!(kh_glulam, 1.0, max_relative = 1e-12);
        // Sous la hauteur de référence, la majoration est inchangée :
        // h = 100 mm → (150/100)^0,2 = 1,5^0,2 ≈ 1,084472.
        let kh_low = timber_size_factor_depth(150.0, 100.0, 0.2, 1.3);
        assert_relative_eq!(kh_low, 1.5_f64.powf(0.2), max_relative = 1e-12);
        assert!(kh_low > 1.0 && kh_low < 1.3);
    }

    #[test]
    #[should_panic(expected = "le coefficient partiel γ_M doit être strictement positif")]
    fn zero_gamma_m_panics() {
        timber_design_bending_strength(0.8, 1.0, 24.0, 0.0);
    }
}
