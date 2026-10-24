//! Interpolation spatiale par pondération inverse à la distance (IDW) —
//! avec un filtre d'aberrants explicite ([`crate::outlier_filter`]),
//! l'autre moitié du pipeline qui élimine la dépendance aux valeurs par
//! défaut propres à chaque logiciel SIG documentée comme source de
//! divergence entre cartes de rendement (voir [`crate::outlier_filter`]
//! pour la référence complète — Walczykova et al. 2018 y trouvent le
//! choix du filtre plus déterminant que celui de l'interpolation, mais
//! les deux doivent être explicites pour qu'un résultat soit
//! reproductible).
//!
//! `ẑ(x) = Σ wᵢ·zᵢ / Σ wᵢ`, `wᵢ = 1/dᵢᵖ` (`p` = puissance, typiquement 2),
//! calculé sur les `k_neighbors` points connus les plus proches du point
//! interrogé (IDW local, la variante la plus utilisée en pratique).
//!
//! Les poids sont évalués sous la forme équivalente `wᵢ = (d₀/dᵢ)ᵖ`, où
//! `d₀` est la distance du plus proche voisin : le facteur commun `d₀ᵖ` se
//! simplifie dans le rapport, le poids du plus proche voisin vaut 1 et les
//! autres sont dans `[0, 1]`. `dᵢᵖ` ne peut donc ni déborder (poids tous
//! nuls, `0/0`) ni sous-déborder (poids infini, `∞/∞`) pour de petites ou
//! grandes distances ou une grande puissance.

use crate::YieldPoint;

fn distance(query: (f64, f64), p: &YieldPoint) -> f64 {
    let (dx, dy) = (query.0 - p.x, query.1 - p.y);
    (dx * dx + dy * dy).sqrt()
}

/// Interpole la valeur de rendement au point `query` par IDW sur les
/// `k_neighbors` points connus les plus proches, avec départage
/// déterministe des ex æquo de distance par index croissant. Si `query`
/// coïncide avec un point connu (distance négligeable), renvoie sa
/// valeur directement plutôt que de diviser par un poids infini.
/// `None` si `known` est vide ou `k_neighbors == 0`.
pub fn idw_interpolate(
    known: &[YieldPoint],
    query: (f64, f64),
    power: f64,
    k_neighbors: usize,
) -> Option<f64> {
    if known.is_empty() || k_neighbors == 0
    {
        return None;
    }
    let mut dists: Vec<(f64, usize)> = known
        .iter()
        .enumerate()
        .map(|(i, p)| (distance(query, p), i))
        .collect();
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if let Some(&(d0, i0)) = dists.first()
    {
        if d0 < 1e-12
        {
            return Some(known[i0].yield_value);
        }
    }

    let k = k_neighbors.min(dists.len());
    // Weights relative to the nearest neighbour: (d0/d)^p equals
    // d0^p · (1/d^p), and the common factor d0^p cancels in the ratio. The
    // nearest weight is exactly 1, so the sum never overflows or vanishes.
    let d0 = dists[0].0;
    let mut weight_sum = 0.0;
    let mut weighted_value_sum = 0.0;
    for &(d, i) in &dists[..k]
    {
        let w = (d0 / d).powf(power);
        weight_sum += w;
        weighted_value_sum += w * known[i].yield_value;
    }
    Some(weighted_value_sum / weight_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn pt(x: f64, y: f64, yield_value: f64) -> YieldPoint {
        YieldPoint { x, y, yield_value }
    }

    #[test]
    fn equidistant_equal_values_return_that_value() {
        let known = vec![pt(-1.0, 0.0, 10.0), pt(1.0, 0.0, 10.0)];
        let z = idw_interpolate(&known, (0.0, 0.0), 2.0, 2).unwrap();
        assert_relative_eq!(z, 10.0, epsilon = 1e-9);
    }

    #[test]
    fn equidistant_unequal_values_average_evenly() {
        let known = vec![pt(-1.0, 0.0, 8.0), pt(1.0, 0.0, 12.0)];
        let z = idw_interpolate(&known, (0.0, 0.0), 2.0, 2).unwrap();
        assert_relative_eq!(z, 10.0, epsilon = 1e-9);
    }

    #[test]
    fn closer_point_dominates_the_weighted_average() {
        // known at distance 1 (yield 10) and distance 3 (yield 20), power=2:
        // w1=1, w2=1/9 -> (10*1 + 20/9) / (1 + 1/9) = (110/9)/(10/9) = 11.0 exactly.
        let known = vec![pt(0.0, 0.0, 10.0), pt(4.0, 0.0, 20.0)];
        let z = idw_interpolate(&known, (1.0, 0.0), 2.0, 2).unwrap();
        assert_relative_eq!(z, 11.0, epsilon = 1e-9);
    }

    #[test]
    fn query_at_a_known_point_returns_its_exact_value() {
        let known = vec![pt(0.0, 0.0, 7.0), pt(5.0, 5.0, 20.0)];
        let z = idw_interpolate(&known, (0.0, 0.0), 2.0, 2).unwrap();
        assert_relative_eq!(z, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn large_power_at_tiny_distances_stays_finite() {
        // d = 1e-10 and 2e-10 with power 40: d^40 underflows to 0, so the
        // naive 1/d^p weights were both +inf and the result was inf/inf = NaN.
        let known = vec![pt(1e-10, 0.0, 10.0), pt(-2e-10, 0.0, 20.0)];
        let z = idw_interpolate(&known, (0.0, 0.0), 40.0, 2).unwrap();
        let r = 0.5f64.powi(40);
        assert!(z.is_finite());
        assert_relative_eq!(z, (10.0 + 20.0 * r) / (1.0 + r), max_relative = 1e-12);
    }

    #[test]
    fn large_distances_stay_finite() {
        // Same geometry as closer_point_dominates_the_weighted_average,
        // scaled by 1e150 and with power 3: d^3 overflows to inf, so the
        // naive weights were both 0 and the result was 0/0 = NaN.
        // w1 = 1, w2 = 1/27 -> (10 + 20/27) / (1 + 1/27) = 290/28.
        let known = vec![pt(0.0, 0.0, 10.0), pt(4e150, 0.0, 20.0)];
        let z = idw_interpolate(&known, (1e150, 0.0), 3.0, 2).unwrap();
        assert!(z.is_finite());
        assert_relative_eq!(z, 290.0 / 28.0, max_relative = 1e-12);
    }

    #[test]
    fn empty_known_set_returns_none() {
        assert!(idw_interpolate(&[], (0.0, 0.0), 2.0, 2).is_none());
    }

    #[test]
    fn zero_k_neighbors_returns_none() {
        let known = vec![pt(0.0, 0.0, 7.0)];
        assert!(idw_interpolate(&known, (1.0, 1.0), 2.0, 0).is_none());
    }
}
