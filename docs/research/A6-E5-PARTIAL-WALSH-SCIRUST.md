# A6-E5 — SciRust SIGNAL : projections partielles de Walsh sur résidus logit

**Extension pure Rust, opt-in, sans mesure de modèle réel.** Le dépôt SciRust possède déjà `scirust-modalg::boolean::{walsh_hadamard,mobius_transform}` pour les tables complètes de fonctions booléennes sur un nombre borné de variables. **Ne pas dupliquer ces opérations**. Le nouvel opérateur `scirust_signal::partial_walsh::PartialWalsh` couvre un besoin distinct : la projection de résidus multiclasses continus sur une **liste explicitement bornée** de masques de parité sur des contextes physiques 32 bits.

- Étiquettes `truth` de 2..256 classes, probabilités de référence `p_y` normalisées et finies. Le résidu est `r_y=1[y=truth]-p_y`, soit le **négatif du gradient de la log-loss par rapport aux logits**.
- Accumulation exacte en `f64` de `Σ χ_m(x)r_y` pour au plus **96 masques, 8192 lignes et 256 classes**. Aucun tenseur `2^32`, aucun GPU SDK propriétaire, aucune allocation statique globale et aucun accès à `TEST` par ce noyau.
- L'amplitude `||R_m||₂` classe les masques, sans p-value ni garantie de récupération après sélection. Le service **ne prouve pas** que les probabilités externes proviennent réellement de prédictions hors-échantillon ; ce contrat est imposé au pipeline consommateur.
- Contrôles : XOR caché avec corrélations marginales nulles, résidus nuls sans faux pic, masse probabiliste invalide refusée sans changement d'état, doublons et épuisement des budgets refusés.
- Il complète `SML-GENIUS` sans lier l'inférence du modèle à une bibliothèque externe (autonomie du harnais). Le moteur de Beam Search et Möbius de SML-GENIUS reste distinct ; comparer leurs sorties sur des fixtures rigoureusement identiques avant d'affirmer une parité de mise en œuvre.
- Le workflow GitHub-hosted `.github/workflows/a6-e5-partial-walsh.yml` vise Rust nightly `2026-07-02` imposé par SciRust, `cargo fmt`, `check`, Clippy strict et tests `partial_walsh` ; **aucune mesure de bits/octet sur Flaubert ou Candide ne provient de ce PR**.

Noyau scientifique utile, mais non autorisé à remplacer la recherche préenregistrée A6-E4 ou à promouvoir SML-25B sans la comparaison real-data appariée A6-E5.
