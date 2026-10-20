//! Dynamique de procédé du **premier ordre** et **réglage de régulateur** —
//! réponse indicielle d'un système du premier ordre, modèle du premier ordre à
//! retard pur (FOPDT) et réglage empirique PI par la méthode de la réponse
//! indicielle de Ziegler-Nichols (boucle ouverte).
//!
//! ```text
//! réponse indicielle 1er ordre  y(t) = K·A·(1 − exp(−t/τ))       [unité de y]
//! temps pour une fraction       t    = −τ·ln(1 − f)               `s`
//! gain proportionnel PI (Z-N)   Kc   = 0.9·τ / (Kp·θ)             `%CO/%VP`
//! temps intégral PI (Z-N)       Ti   = 3.33·θ                     `s`
//! réponse FOPDT                 y(t) = 0                si t < θ
//!                               y(t) = K·A·(1 − exp(−(t−θ)/τ))    si t ≥ θ
//! ```
//!
//! `y(t)` écart de la sortie par rapport à son état initial [unité de la variable
//! réglée], `K` gain statique du procédé [unité de y par unité d'échelon], `A`
//! amplitude de l'échelon appliqué à l'entrée [unité de l'entrée], `t` temps
//! écoulé depuis l'application de l'échelon `s`, `τ` constante de temps `s`, `f`
//! fraction de la valeur finale visée [sans dimension, dans `[0, 1[`] (`f = 0.632`
//! à `t = τ`), `Kp` gain statique identifié du procédé, `θ` retard pur (temps
//! mort) `s`, `Kc` gain proportionnel du régulateur, `Ti` temps intégral `s`.
//!
//! **Limite honnête** : le gain statique `Kp`/`K`, la constante de temps `τ` et le
//! retard pur `θ` sont des **paramètres identifiés sur la réponse mesurée du
//! procédé** (méthode de la tangente, du point d'inflexion, moindres carrés…) :
//! ils sont **fournis par l'appelant** et jamais inventés. Le modèle est
//! **linéaire du premier ordre (+ retard)**, valable au voisinage d'un point de
//! fonctionnement ; les coefficients de Ziegler-Nichols (`0.9`, `3.33`) sont des
//! **règles empiriques** de la réponse indicielle qui donnent un **point de départ
//! de réglage à affiner** (elles visent une décroissance en quart d'amplitude,
//! souvent trop agressive). Ces fonctions ne modélisent ni les non-linéarités, ni
//! les ordres supérieurs, ni la stabilité en boucle fermée.

/// Réponse indicielle d'un système du **premier ordre**
/// `y(t) = K·A·(1 − exp(−t/τ))` [unité de la variable réglée].
///
/// `gain` `K` gain statique du procédé, `time_constant` `τ` constante de temps
/// `s`, `time` `t` temps depuis l'application de l'échelon `s`, `step_amplitude`
/// `A` amplitude de l'échelon d'entrée. À `t = τ`, la sortie atteint `≈ 63.2 %`
/// de sa valeur finale `K·A`. Calculée sous la forme `−K·A·expm1(−t/τ)`, qui
/// garde la précision relative pour `t ≪ τ`.
///
/// Panique si `gain`, `time` ou `step_amplitude` n'est pas fini, si `time` est
/// négatif, ou si `time_constant` n'est pas strictement positif (division).
pub fn ford_step_response(gain: f64, time_constant: f64, time: f64, step_amplitude: f64) -> f64 {
    assert!(gain.is_finite(), "le gain statique doit être fini");
    assert!(
        time_constant > 0.0,
        "la constante de temps doit être strictement positive (s)"
    );
    assert!(
        time.is_finite() && time >= 0.0,
        "le temps doit être fini et positif ou nul (s)"
    );
    assert!(
        step_amplitude.is_finite(),
        "l'amplitude de l'échelon doit être finie"
    );
    let ratio: f64 = -time / time_constant;
    // `1 − exp(x)` écrit `−expm1(x)` : pas d'annulation pour `t ≪ τ`.
    gain * step_amplitude * -ratio.exp_m1()
}

/// Temps nécessaire pour atteindre une **fraction** `f` de la valeur finale d'une
/// réponse du premier ordre `t = −τ·ln(1 − f)` `s` — réciproque de
/// [`ford_step_response`] au niveau de la fraction.
///
/// `time_constant` `τ` constante de temps `s`, `fraction` `f` fraction de la
/// valeur finale visée [sans dimension, dans `[0, 1[`]. Cas usuels : `f = 0.632`
/// donne `t = τ`, `f = 0.95` donne `t ≈ 3·τ`, `f = 0.99` donne `t ≈ 4.6·τ`.
/// Calculé sous la forme `−τ·ln_1p(−f)`, qui garde la précision relative pour
/// `f ≪ 1` (et rend `+0` pour `f = 0`).
///
/// Panique si `time_constant` n'est pas strictement positif, ou si `fraction`
/// sort de `[0, 1[` (la borne 1 fait diverger le logarithme).
pub fn ford_time_to_fraction(time_constant: f64, fraction: f64) -> f64 {
    assert!(
        time_constant > 0.0,
        "la constante de temps doit être strictement positive (s)"
    );
    assert!(
        fraction.is_finite() && (0.0..1.0).contains(&fraction),
        "la fraction doit être comprise dans [0, 1[ (strictement inférieure à 1)"
    );
    // `ln(1 − f)` écrit `ln_1p(−f)` : pas d'arrondi de `1 − f` pour `f ≪ 1`.
    -time_constant * (-fraction).ln_1p()
}

/// Gain proportionnel d'un régulateur **PI** par la méthode de la réponse
/// indicielle de Ziegler-Nichols `Kc = 0.9·τ / (Kp·θ)` [gain sans dimension].
///
/// `process_gain` `Kp` gain statique identifié du procédé, `dead_time` `θ` retard
/// pur (temps mort) `s`, `time_constant` `τ` constante de temps identifiée `s`.
/// Le rapport `τ/θ` (contrôlabilité) gouverne l'agressivité du réglage.
///
/// Panique si `process_gain` n'est pas strictement positif, si `dead_time` n'est
/// pas strictement positif (division), ou si `time_constant` est négatif ou non
/// fini.
pub fn ford_zn_pi_gain(process_gain: f64, dead_time: f64, time_constant: f64) -> f64 {
    assert!(
        process_gain > 0.0,
        "le gain statique du procédé doit être strictement positif"
    );
    assert!(
        dead_time > 0.0,
        "le retard pur doit être strictement positif (s)"
    );
    assert!(
        time_constant.is_finite() && time_constant >= 0.0,
        "la constante de temps doit être finie et positive ou nulle (s)"
    );
    0.9 * time_constant / (process_gain * dead_time)
}

/// Temps intégral d'un régulateur **PI** par Ziegler-Nichols (réponse indicielle)
/// `Ti = 3.33·θ` `s`.
///
/// `dead_time` `θ` retard pur (temps mort) identifié `s`. Le temps intégral
/// s'exprime uniquement à partir du retard pur dans cette règle empirique.
///
/// Panique si `dead_time` est négatif ou non fini.
pub fn ford_zn_pi_integral_time(dead_time: f64) -> f64 {
    assert!(
        dead_time.is_finite() && dead_time >= 0.0,
        "le retard pur doit être fini et positif ou nul (s)"
    );
    3.33 * dead_time
}

/// Réponse indicielle d'un modèle du **premier ordre à retard pur** (FOPDT)
/// `y(t) = 0` si `t < θ`, sinon `y(t) = K·A·(1 − exp(−(t − θ)/τ))`
/// [unité de la variable réglée].
///
/// `gain` `K` gain statique, `time_constant` `τ` constante de temps `s`,
/// `dead_time` `θ` retard pur `s`, `time` `t` temps depuis l'échelon `s`,
/// `step_amplitude` `A` amplitude de l'échelon d'entrée. Tant que `t < θ`, la
/// sortie n'a pas encore réagi (retard de transport / temps mort). Calculée sous
/// la forme `−K·A·expm1(−(t − θ)/τ)`, précise juste après le retard.
///
/// Panique si `gain`, `time` ou `step_amplitude` n'est pas fini, si `time` ou
/// `dead_time` est négatif, ou si `time_constant` n'est pas strictement positif.
pub fn ford_first_order_plus_deadtime(
    gain: f64,
    time_constant: f64,
    dead_time: f64,
    time: f64,
    step_amplitude: f64,
) -> f64 {
    assert!(gain.is_finite(), "le gain statique doit être fini");
    assert!(
        time_constant > 0.0,
        "la constante de temps doit être strictement positive (s)"
    );
    assert!(
        dead_time.is_finite() && dead_time >= 0.0,
        "le retard pur doit être fini et positif ou nul (s)"
    );
    assert!(
        time.is_finite() && time >= 0.0,
        "le temps doit être fini et positif ou nul (s)"
    );
    assert!(
        step_amplitude.is_finite(),
        "l'amplitude de l'échelon doit être finie"
    );
    if time < dead_time
    {
        0.0
    }
    else
    {
        // `1 − exp(x)` écrit `−expm1(x)` : pas d'annulation juste après `θ`.
        gain * step_amplitude * -(-(time - dead_time) / time_constant).exp_m1()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::E;

    /// À `t = τ`, la réponse indicielle atteint exactement `1 − 1/e ≈ 63.2 %` de
    /// la valeur finale `K·A`. Cas chiffré : `K = 2`, `A = 5`, `τ = 10`, `t = 10`.
    /// Valeur finale `K·A = 10` ; `10·(1 − 1/e) = 10·0.6321205588 = 6.321205588`.
    /// (Recalcul indépendant : `1/e = 0.3678794412`, `1 − 0.3678794412 =
    /// 0.6321205588`, `× 10 = 6.321205588`.)
    #[test]
    fn step_response_atteint_63pct_a_tau() {
        let y = ford_step_response(2.0, 10.0, 10.0, 5.0);
        assert_relative_eq!(y, 6.321205588, epsilon = 1e-6);
        // Fraction exacte 1 − 1/e.
        assert_relative_eq!(y / (2.0 * 5.0), 1.0 - 1.0 / E, epsilon = 1e-9);
    }

    /// Réciprocité fraction ↔ temps : le temps rendu par [`ford_time_to_fraction`]
    /// pour une fraction `f` réinjecté dans [`ford_step_response`] redonne `f` de
    /// la valeur finale.
    #[test]
    fn reciprocite_fraction_temps() {
        let tau = 7.0;
        let f = 0.8;
        let t = ford_time_to_fraction(tau, f);
        let y = ford_step_response(1.0, tau, t, 1.0);
        assert_relative_eq!(y, f, epsilon = 1e-9);
    }

    /// `t = τ` correspond à la fraction `1 − 1/e`, et `t = 3τ` dépasse 95 %.
    #[test]
    fn time_to_fraction_points_reperes() {
        let tau = 4.0;
        assert_relative_eq!(
            ford_time_to_fraction(tau, 1.0 - 1.0 / E),
            tau,
            epsilon = 1e-9
        );
        // Repère classique : atteindre 95 % demande ≈ 3·τ (exactement −τ·ln 0,05 ≈ 2,996·τ).
        let t95 = ford_time_to_fraction(tau, 0.95);
        assert!(t95 > 2.9 * tau && t95 < 3.0 * tau);
    }

    /// Réglage PI de Ziegler-Nichols. Cas chiffré : `Kp = 2`, `θ = 3`, `τ = 30`.
    /// `Kc = 0.9·30/(2·3) = 27/6 = 4.5` ; `Ti = 3.33·3 = 9.99`.
    /// (Recalcul indépendant : `0.9·30 = 27`, `2·3 = 6`, `27/6 = 4.5` ;
    /// `3.33·3 = 9.99`.)
    #[test]
    fn reglage_zn_pi_cas_chiffre() {
        assert_relative_eq!(ford_zn_pi_gain(2.0, 3.0, 30.0), 4.5, epsilon = 1e-12);
        assert_relative_eq!(ford_zn_pi_integral_time(3.0), 9.99, epsilon = 1e-12);
    }

    /// Le gain PI est proportionnel à `τ` et inversement proportionnel à `Kp·θ` :
    /// doubler `τ` double `Kc`, doubler `θ` le divise par deux.
    #[test]
    fn zn_pi_gain_proportionnalites() {
        let base = ford_zn_pi_gain(2.0, 3.0, 30.0);
        assert_relative_eq!(ford_zn_pi_gain(2.0, 3.0, 60.0), 2.0 * base, epsilon = 1e-12);
        assert_relative_eq!(ford_zn_pi_gain(2.0, 6.0, 30.0), 0.5 * base, epsilon = 1e-12);
    }

    /// FOPDT : sortie nulle avant le retard, puis réponse identique à un premier
    /// ordre décalé de `θ`. Cas chiffré : `K = 2`, `τ = 10`, `θ = 3`, `A = 5`,
    /// `t = 13` ⇒ `t − θ = 10 = τ` ⇒ `y = 10·(1 − 1/e) = 6.321205588`.
    #[test]
    fn fopdt_retard_puis_premier_ordre() {
        assert_relative_eq!(
            ford_first_order_plus_deadtime(2.0, 10.0, 3.0, 2.9, 5.0),
            0.0,
            epsilon = 1e-12
        );
        let y = ford_first_order_plus_deadtime(2.0, 10.0, 3.0, 13.0, 5.0);
        assert_relative_eq!(y, 6.321205588, epsilon = 1e-6);
        // Équivalence au premier ordre décalé du retard.
        assert_relative_eq!(y, ford_step_response(2.0, 10.0, 10.0, 5.0), epsilon = 1e-12);
    }

    /// Petits `t/τ` : `1 − exp(−t/τ)` annule presque tous les chiffres
    /// significatifs (pour `t/τ = 1e-12`, l'ancienne formule rendait
    /// `9.999778782798785e-13`, erreur relative `2.2e-5` ; pour `t/τ = 1e-17`
    /// elle rendait `0`). Référence mpmath (50 chiffres) :
    /// `−expm1(−1e-12) = 9.99999999999500e-13`,
    /// `−expm1(−1e-17) = 1.00000000000000e-17` (série `x − x²/2 + …`).
    #[test]
    fn step_response_petit_t_sur_tau_garde_la_precision() {
        let y = ford_step_response(1.0, 1.0, 1.0e-12, 1.0);
        assert_relative_eq!(
            y,
            9.999_999_999_995e-13,
            epsilon = 0.0,
            max_relative = 1e-12
        );
        let y = ford_step_response(1.0, 1.0, 1.0e-17, 1.0);
        assert_relative_eq!(y, 1.0e-17, epsilon = 0.0, max_relative = 1e-12);
        // Mise à l'échelle : K·A = 10, τ = 1e3, t = 1e-9 ⇒ t/τ = 1e-12.
        let y = ford_step_response(2.0, 1.0e3, 1.0e-9, 5.0);
        assert_relative_eq!(
            y,
            10.0 * 9.999_999_999_995e-13,
            epsilon = 0.0,
            max_relative = 1e-12
        );
    }

    /// Même annulation pour le FOPDT juste après le retard pur : `θ = 0`,
    /// `τ = 1`, `t = 1e-12` doit coïncider avec la référence
    /// `−expm1(−1e-12) = 9.99999999999500e-13` (mpmath).
    #[test]
    fn fopdt_juste_apres_retard_garde_la_precision() {
        let y = ford_first_order_plus_deadtime(1.0, 1.0, 0.0, 1.0e-12, 1.0);
        assert_relative_eq!(
            y,
            9.999_999_999_995e-13,
            epsilon = 0.0,
            max_relative = 1e-12
        );
        let y = ford_first_order_plus_deadtime(1.0, 1.0, 0.0, 1.0e-17, 1.0);
        assert_relative_eq!(y, 1.0e-17, epsilon = 0.0, max_relative = 1e-12);
    }

    /// Petites fractions : `−ln(1 − f)` perd la précision car `1 − f` est
    /// arrondi (pour `f = 1e-12`, l'ancienne formule rendait
    /// `9.999778782803785e-13` ; pour `f = 1e-17` elle rendait `-0.0`).
    /// Référence mpmath (50 chiffres) : `−log1p(−1e-12) = 1.0000000000005e-12`,
    /// `−log1p(−1e-17) = 1.00000000000000e-17`.
    #[test]
    fn time_to_fraction_petite_fraction_garde_la_precision() {
        let t = ford_time_to_fraction(1.0, 1.0e-12);
        assert_relative_eq!(
            t,
            1.000_000_000_000_5e-12,
            epsilon = 0.0,
            max_relative = 1e-12
        );
        let t = ford_time_to_fraction(1.0, 1.0e-17);
        assert!(
            t > 0.0,
            "le temps doit être strictement positif, obtenu {t:e}"
        );
        assert_relative_eq!(t, 1.0e-17, epsilon = 0.0, max_relative = 1e-12);
        // Fraction nulle : temps nul (et non `-0.0`).
        assert!(ford_time_to_fraction(3.0, 0.0).is_sign_positive());
        // Réciprocité aux petites fractions.
        let f = 1.0e-15;
        let y = ford_step_response(1.0, 7.0, ford_time_to_fraction(7.0, f), 1.0);
        assert_relative_eq!(y, f, epsilon = 0.0, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le retard pur doit être strictement positif")]
    fn zn_pi_gain_retard_nul_panique() {
        let _ = ford_zn_pi_gain(2.0, 0.0, 30.0);
    }
}
