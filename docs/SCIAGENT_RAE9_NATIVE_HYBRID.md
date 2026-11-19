# SciAgent RAE-9 — intégration réelle à l'attention numérique

**Statut : recherche, route explicitement activable uniquement sur CPU.**
La version standard de `SciAgentModel::forward`, la rétropropagation et les
parcours résidents WGPU M32/M33 sont inchangés.

## Chaîne effectivement raccordée

Le modèle SciAgent est dans `scirust-sciagent` : Transformer entraîné en
Rust, RoPE et GQA. Le checkpoint inclus `small-20M/final` est un modèle
entraîné du dépôt. Le nouveau `forward_research_rae9` réalise une vraie
inférence du modèle entier, en remplaçant **une seule sortie de tête du
premier bloc** au dernier jeton :

1. Les projections Q/K/V proviennent du vrai modèle, après RMSNorm et
   RoPE effectués par le ruban SciRust ; la tête K/V est la tête source
   `query_head / (n_heads / n_kv_heads)`, sans duplication artificielle.
2. Le chemin SciAgent normal calcule indépendamment l'attention dense
   causale, qui sert de référence numérique.
3. Un snapshot temporaire `RAE9QKV1 sciagent` est transmis à un
   exécutable Rust FLAT-ATTENTION **explicitement fourni par l'opérateur**.
   Le programme RAE-9 exécute RAE-5 (impulsions de changements de signe),
   RAE-7 (réparation bornée), RAE-8 (calcul QK/softmax/V) et renvoie une
   sortie de tête ou un repli dense.
4. Si RAE-9 est numériquement admissible, le dernier jeton de cette tête
   consomme **le vecteur FLAT calculé**. Le modèle reprend avec la projection
   de sortie, les résidus, les autres blocs et la tête de vocabulaire.
   En repli dense, la sortie native de SciAgent est conservée.
5. L'observation `Rae9HeadEvidence` donne la route, le nombre de clés
   sélectionnées, les produits QK logiques et l'écart à l'oracle natif.
   Le binaire `sciagent-rae9` compare les logits natifs et hybrides du
   vrai checkpoint et échoue en cas d'erreur numérique excessive.

Le ruban du **chemin expérimental** détache la tête injectée : il est
**interdit de le sélectionner pendant l'entraînement**. La route native
reste différentiable. Le snapshot est une copie hôte intégrale de Q/K/V,
les normes et bornes V sont calculées par un scan complet dans FLAT ;
aucune réduction de trafic physique, accélération, prise en charge des
séquences longues ou qualité du modèle entier n'est acquise.

## Vérification reproducible et contrôle

Épingler un exécutable construit à partir d'une révision de FLAT dont la
PR RAE-9 SciAgent est fusionnée et la CI verte. Avec un chemin `/chemin/...` correspondant au binaire **trusted** fourni par RemoteOps :

```bash
cargo test --locked -p scirust-sciagent rae9_research
cargo run --locked -p scirust-sciagent --bin sciagent-rae9 -- \
  --checkpoint scirust-sciagent/checkpoints/small-20M/final \
  --bridge /chemin/rae9_real_model_bridge \
  --head 1 --page-size 8
```

Pour un essai matériel : contrôle strict de la provenance du SHA SciRust et
FLAT, empreinte du checkpoint, disponibilité RAM/disque, exécution isolée et
bornée, préférence CPU Dell et **RemoteOps comme unique plan de contrôle**.
Ne pas lancer le code d'une PR non qualifiée sur un runner privilégié.
Les snapshots temporaires en permissions 0600 sont supprimés à la fin,
même en cas d'erreur. La sortie du test ne contient que des métriques ;
l'absence de résultat ou d'injection est un échec de qualification.

## Limites de portée

- Premier bloc, une tête, batch=1, non-paddé, requête du dernier jeton
  uniquement ; maximum 4 096 clés, 256 dimensions, 512 pages et 16 Mio
  de snapshot selon les limites RAE-9.
- Le chemin WGPU M32/M33 n'est **pas** encore un consommateur de masques
  impulsionnels ; la qualification réelle M32/M33 doit rester intacte.
- La présence d'une route `bounded_sparse` avec autant de clés exécutées
  que de clés originales correspond à une *intervention numérique*, **pas**
  à une optimisation creuse.
- Un gain de performances, l'apprentissage du routeur ou la comparaison
  de qualité doivent être mesurés séparément sur des requêtes réelles.

## Garde hybride locale sur la sortie de tête

L'admission d'une route `bounded_sparse` peut maintenant recevoir un **budget
d'erreur absolue indépendant** de la borne calculée par FLAT. Ce budget est
comparé à l'écart maximum composante par composante entre la sortie FLAT et
l'attention dense réelle du modèle. À budget dépassé, SciAgent conserve
l'attention native : l'injection ne se produit pas et l'observation
`native_budget_vetoed=true` le distingue d'un repli décidé par FLAT.

Activation explicite (choisir le budget en fonction d'une campagne
d'étalonnage, et non comme seuil de qualité supposé) :

```bash
cargo run --locked -p scirust-sciagent --bin sciagent-rae9 -- \
  --checkpoint scirust-sciagent/checkpoints/small-20M/final \
  --bridge /chemin/rae9_real_model_bridge \
  --head 1 --page-size 8 --max-native-head-delta 0.01
```

L'absence du paramètre conserve la route RAE-9 antérieure. Le budget doit
être fini et positif ou nul ; NaN, infini et valeurs négatives sont
refusés. Si le budget impose un repli dense, le CLI expérimental échoue
explicitement avec `RAE9_MODEL_HEAD_NOT_INJECTED` au lieu de prétendre
avoir expérimenté une injection. Cette tolérance ne garantit **ni**
l'identité de l'argmax final **ni** une qualité de réponse préservée. Les
observations numériques et les essais réels sur checkpoint doivent
toujours être qualifiés séparément.
