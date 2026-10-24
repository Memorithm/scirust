//! Filtrage des points aberrants sur une carte de rendement — les deux
//! filtres les plus largement cités dans la littérature de nettoyage de
//! cartes de rendement (Sudduth & Drummond, "Yield Editor: Software for
//! Removing Errors from Crop Yield Maps", Agronomy Journal 99(6), 2007 —
//! l'outil de référence USDA-ARS).
//!
//! **Pourquoi ce module existe** : Walczykova et al. (2018) montrent que
//! les mêmes données de rendement brutes, traitées par QGIS, Agro-Map ou
//! Farm Works avec leurs filtres par défaut respectifs, produisent des
//! cartes visiblement différentes — le choix du filtre d'aberrants (et
//! de son seuil) affecte le coefficient de variation résultant de 32 à
//! 46%, plus que le choix de la méthode d'interpolation. Ce module ne
//! prétend pas trancher le désaccord sur le "bon" seuil — il rend le
//! filtre *explicite et auditable* : mêmes points + mêmes paramètres ⇒
//! même résultat, byte pour byte, sur toute plateforme, plutôt qu'un
//! réglage implicite propre à l'outil.

use crate::YieldPoint;

/// Indices des points conservés après filtre global : rejette tout point
/// dont le rendement s'écarte de la moyenne globale de plus de `k_std`
/// écarts-types (Sudduth & Drummond, filtres "globaux").
///
/// Un rendement non fini (`NaN`, `±∞` — mesure absente ou capteur en
/// défaut) est toujours rejeté, et la moyenne et l'écart-type sont calculés
/// sur les seuls rendements finis : un point invalide ne peut donc pas faire
/// rejeter les autres.
pub fn global_filter(points: &[YieldPoint], k_std: f64) -> Vec<usize> {
    let finite: Vec<usize> = (0..points.len())
        .filter(|&i| points[i].yield_value.is_finite())
        .collect();
    if finite.is_empty()
    {
        return Vec::new();
    }
    // A field with no variation has nothing to filter. Test it exactly:
    // the computed mean of identical values is not always exact (e.g. 11
    // copies of 12345.678 average to 12345.678 - 1.8e-12), which gave a
    // spurious standard deviation of about 1.8e-12, above the 1e-12 cut-off,
    // and a z-score of exactly 1 for every point, so `k_std < 1` rejected the
    // whole uniform field.
    let first = points[finite[0]].yield_value;
    if finite.iter().all(|&i| points[i].yield_value == first)
    {
        return finite;
    }
    let n = finite.len() as f64;
    let mean = finite.iter().map(|&i| points[i].yield_value).sum::<f64>() / n;
    let variance = finite
        .iter()
        .map(|&i| (points[i].yield_value - mean).powi(2))
        .sum::<f64>()
        / n;
    let std = variance.sqrt();
    if std < 1e-12
    {
        return finite; // no variation: nothing to filter
    }
    finite
        .into_iter()
        .filter(|&i| ((points[i].yield_value - mean) / std).abs() <= k_std)
        .collect()
}

fn distance(a: &YieldPoint, b: &YieldPoint) -> f64 {
    let (dx, dy) = (a.x - b.x, a.y - b.y);
    (dx * dx + dy * dy).sqrt()
}

/// Indices des points conservés après filtre local (voisinage) :
/// rejette tout point dont le rendement s'écarte de la moyenne de ses
/// `k_neighbors` plus proches voisins de plus de `k_std` écarts-types
/// *locales* (calculées sur ce même voisinage) — attrape les anomalies
/// localisées (par ex. une bande de relevage de tête de champ) qu'un
/// filtre global au seuil généreux laisse passer, car elles restent
/// dans la plage globale tout en tranchant avec leur environnement
/// immédiat (Sudduth & Drummond, filtres "spatiaux").
///
/// Recherche de voisins par force brute (`O(n²)`), avec départage
/// déterministe des ex æquo de distance par index croissant — adapté aux
/// tailles de jeux de données de parcelle habituelles, pas à un flux à
/// très grande échelle.
///
/// Un point dont la position ou le rendement n'est pas fini (`NaN`, `±∞`)
/// est rejeté et n'entre dans le voisinage d'aucun autre point. Un point
/// valide dont le voisinage compte moins de deux points valides (par ex. un
/// jeu d'un ou deux points) ne peut pas être jugé localement — l'écart-type
/// d'une seule valeur est nul par construction — et est conservé. En
/// conséquence, `k_neighbors == 1` conserve tous les points valides (seuls
/// les points non finis sont rejetés) : il faut `k_neighbors >= 2` pour que
/// le filtre local agisse. `k_neighbors == 0` désactive le filtre et
/// renvoie tous les indices. Si les voisins ont tous le
/// même rendement (écart-type local nul), un point dont le rendement en
/// diffère est rejeté.
pub fn local_filter(points: &[YieldPoint], k_neighbors: usize, k_std: f64) -> Vec<usize> {
    let n = points.len();
    if n == 0 || k_neighbors == 0
    {
        return (0..n).collect();
    }
    let valid = |p: &YieldPoint| p.x.is_finite() && p.y.is_finite() && p.yield_value.is_finite();
    let mut kept = Vec::with_capacity(n);
    for i in 0..n
    {
        if !valid(&points[i])
        {
            continue;
        }
        let mut dists: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i && valid(&points[j]))
            .map(|j| (distance(&points[i], &points[j]), j))
            .collect();
        let k = k_neighbors.min(dists.len());
        if k < 2
        {
            // Fewer than two neighbours: no local spread can be estimated
            // (a single value has a standard deviation of 0 by
            // construction), hence no local evidence against the point.
            kept.push(i);
            continue;
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let neighbor_yields: Vec<f64> = dists[..k]
            .iter()
            .map(|&(_, j)| points[j].yield_value)
            .collect();
        // Identical neighbours are detected exactly and compared with their
        // common value: their computed mean can be off by a rounding error
        // (about 1.8e-12 for 11 copies of 12345.678), which used to produce a
        // spurious local standard deviation above the 1e-12 cut-off and a
        // z-score of exactly 1 for a point equal to its neighbours, so
        // `k_std < 1` rejected it.
        let first = neighbor_yields[0];
        let keep = if neighbor_yields.iter().all(|&y| y == first)
        {
            (points[i].yield_value - first).abs() < 1e-9
        }
        else
        {
            let mean = neighbor_yields.iter().sum::<f64>() / k as f64;
            let variance = neighbor_yields
                .iter()
                .map(|y| (y - mean).powi(2))
                .sum::<f64>()
                / k as f64;
            let std = variance.sqrt();
            if std < 1e-12
            {
                (points[i].yield_value - mean).abs() < 1e-9
            }
            else
            {
                ((points[i].yield_value - mean) / std).abs() <= k_std
            }
        };
        if keep
        {
            kept.push(i);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f64, y: f64, yield_value: f64) -> YieldPoint {
        YieldPoint { x, y, yield_value }
    }

    #[test]
    fn global_filter_rejects_a_gross_outlier() {
        let points = vec![
            pt(0.0, 0.0, 8.0),
            pt(1.0, 0.0, 9.0),
            pt(2.0, 0.0, 10.0),
            pt(3.0, 0.0, 9.5),
            pt(4.0, 0.0, 8.5),
            pt(5.0, 0.0, 50.0), // combine sensor glitch
        ];
        let kept = global_filter(&points, 2.0);
        assert!(!kept.contains(&5), "the 50.0 outlier should be rejected");
        assert_eq!(kept.len(), 5);
    }

    #[test]
    fn global_filter_keeps_everything_when_uniform() {
        let points = vec![pt(0.0, 0.0, 10.0); 5];
        assert_eq!(global_filter(&points, 1.0).len(), 5);
    }

    /// Deux zones de sol légitimement différentes (6 t/ha puis 12 t/ha),
    /// avec un point anormal à x=15 qui vaut 6.0 (comme la zone A) alors
    /// que ses voisins immédiats sont tous à 12.0 (zone B). Comme ce
    /// point anormal a *exactement* la même valeur qu'un point normal de
    /// zone A, un filtre global lui donne le même score-z que ce point
    /// légitime — il ne peut structurellement pas distinguer les deux.
    /// Le filtre local, lui, ne regarde que le voisinage immédiat et
    /// détecte la rupture. Valeurs vérifiées indépendamment (numpy)
    /// avant portage.
    #[test]
    fn local_filter_catches_what_global_filter_structurally_cannot() {
        let mut points = vec![];
        for i in 0..10
        {
            points.push(pt(i as f64, 0.0, 6.0)); // zone A, indices 0..9
        }
        for i in 10..20
        {
            let yield_value = if i == 15 { 6.0 } else { 12.0 }; // zone B, anomaly at index 15
            points.push(pt(i as f64, 0.0, yield_value));
        }

        let global_kept = global_filter(&points, 2.0);
        assert_eq!(
            global_kept.len(),
            points.len(),
            "a global filter cannot single out a value it also sees as legitimate elsewhere"
        );

        let local_kept = local_filter(&points, 3, 2.0);
        assert!(
            !local_kept.contains(&15),
            "the zone-B anomaly should be rejected locally"
        );
        assert_eq!(local_kept.len(), points.len() - 1);
    }

    /// Un rendement non fini (capteur en défaut : NaN, ou ±∞) ne doit pas
    /// contaminer la moyenne et l'écart-type globaux : avant correction, un
    /// seul NaN rendait `mean` et `std` NaN et **tous** les points étaient
    /// rejetés.
    #[test]
    fn global_filter_rejects_only_the_non_finite_point() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
        {
            let points = vec![
                pt(0.0, 0.0, 8.0),
                pt(1.0, 0.0, 9.0),
                pt(2.0, 0.0, bad),
                pt(3.0, 0.0, 10.0),
                pt(4.0, 0.0, 9.5),
            ];
            assert_eq!(global_filter(&points, 2.0), vec![0, 1, 3, 4], "bad = {bad}");
        }
    }

    /// Les statistiques globales sont celles des seuls points finis : le
    /// verdict sur les points valides est identique avec ou sans le point
    /// non fini.
    #[test]
    fn global_filter_statistics_ignore_non_finite_points() {
        let clean = vec![
            pt(0.0, 0.0, 8.0),
            pt(1.0, 0.0, 9.0),
            pt(2.0, 0.0, 10.0),
            pt(3.0, 0.0, 9.5),
            pt(4.0, 0.0, 8.5),
            pt(5.0, 0.0, 50.0),
        ];
        let mut dirty = clean.clone();
        dirty.push(pt(6.0, 0.0, f64::NAN));
        assert_eq!(global_filter(&clean, 2.0), global_filter(&dirty, 2.0));
        assert!(global_filter(&[pt(0.0, 0.0, f64::NAN)], 2.0).is_empty());
    }

    /// Un rendement NaN chez un voisin ne doit pas faire rejeter les points
    /// valides qui l'entourent (avant correction, la moyenne locale devenait
    /// NaN et les deux voisins du point NaN étaient rejetés avec lui).
    #[test]
    fn local_filter_ignores_non_finite_neighbours() {
        let mut points: Vec<YieldPoint> = (0..10).map(|i| pt(i as f64, 0.0, 10.0)).collect();
        points[5].yield_value = f64::NAN;
        let kept = local_filter(&points, 2, 2.0);
        let expected: Vec<usize> = (0..10).filter(|&i| i != 5).collect();
        assert_eq!(kept, expected);
    }

    /// Un point sans aucun voisin exploitable (jeu à un seul point) ne peut
    /// pas être jugé localement : il est conservé, comme avec
    /// `k_neighbors == 0`, au lieu d'être rejeté par une comparaison NaN
    /// (`0/0` sur un voisinage vide).
    #[test]
    fn local_filter_keeps_a_point_without_neighbours() {
        assert_eq!(local_filter(&[pt(0.0, 0.0, 7.0)], 3, 2.0), vec![0]);
        let points = vec![pt(0.0, 0.0, 7.0), pt(1.0, 0.0, f64::NAN)];
        assert_eq!(local_filter(&points, 3, 2.0), vec![0]);
    }

    /// Un voisinage d'un seul point n'a pas d'écart-type local (il vaut 0
    /// par construction) : avant correction, `k_neighbors == 1` — ou un jeu
    /// de deux points valides — rejetait tout point dont le rendement
    /// différait de celui de son voisin, quel que soit `k_std`, ce qui
    /// pouvait vider une carte à variation lisse.
    #[test]
    fn local_filter_keeps_points_judged_on_a_single_neighbour() {
        let points: Vec<YieldPoint> = (0..5)
            .map(|i| pt(i as f64, 0.0, 8.0 + 0.1 * i as f64))
            .collect();
        assert_eq!(local_filter(&points, 1, 2.0), vec![0, 1, 2, 3, 4]);
        // Only two valid points: the effective neighbourhood has one point
        // whatever `k_neighbors` is.
        let pair = vec![pt(0.0, 0.0, 8.0), pt(1.0, 0.0, 9.0)];
        assert_eq!(local_filter(&pair, 3, 2.0), vec![0, 1]);
        // With two or more identical neighbours, a differing point is still
        // rejected (infinite local z-score), as before.
        let mut flat: Vec<YieldPoint> = (0..6).map(|i| pt(i as f64, 0.0, 10.0)).collect();
        flat[3].yield_value = 11.0;
        assert!(!local_filter(&flat, 2, 2.0).contains(&3));
    }

    /// Un champ uniforme en kg/ha (11 points à 12 345,678) : la moyenne
    /// calculée de valeurs identiques n'est pas exacte (écart d'environ
    /// 1,8e-12), ce qui donnait avant correction un écart-type parasite
    /// supérieur au seuil de 1e-12 et un score-z de 1 pour chaque point —
    /// avec `k_std < 1`, tout le champ était rejeté (globalement) ou chaque
    /// point égal à ses voisins était rejeté (localement).
    #[test]
    fn uniform_field_is_kept_despite_mean_rounding() {
        let v = 12345.678;
        let n = 11;
        let mean = vec![v; n].iter().sum::<f64>() / n as f64;
        assert_ne!(mean, v, "test premise: the computed mean is not exact");
        let points: Vec<YieldPoint> = (0..n).map(|i| pt(i as f64, 0.0, v)).collect();
        let all: Vec<usize> = (0..n).collect();
        assert_eq!(global_filter(&points, 0.5), all);
        let points12: Vec<YieldPoint> = (0..12).map(|i| pt(i as f64, 0.0, v)).collect();
        assert_eq!(
            local_filter(&points12, 11, 0.5),
            (0..12).collect::<Vec<_>>()
        );
        // A differing point among identical neighbours is still rejected.
        let mut dirty = points12.clone();
        dirty[6].yield_value = v + 1.0;
        assert!(!local_filter(&dirty, 11, 0.5).contains(&6));
    }
}
