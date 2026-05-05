# Référence des KPIs et Métriques — PredictML

Ce document explique chaque indicateur affiché dans le dashboard Streamlit et retourné par l'API.

---

## Métriques de performance (classification)

### Accuracy (Précision globale)
**Ce que c'est** : proportion de prédictions correctes sur le total.

```
accuracy = (vrais positifs + vrais négatifs) / total
```

**Interprétation** :
- `1.0` = 100% de bonnes prédictions (parfait)
- `0.9` = 90% de bonnes prédictions
- Valeur naïve (prédire toujours la classe majoritaire) : proportion de la classe majoritaire

**Quand agir** : baisse de plus de 5% par rapport à la baseline → investigation. Baisse > 10% → ré-entraînement.

**Limites** : trompeuse sur les jeux déséquilibrés (ex : 98% de "non-fraude" → accuracy de 98% en prédisant toujours "non-fraude").

---

### F1 Score (Score F1)
**Ce que c'est** : moyenne harmonique de la précision et du rappel. Résiste aux classes déséquilibrées.

```
F1 = 2 × (precision × recall) / (precision + recall)
```

**Interprétation** :
- `1.0` = précision et rappel parfaits
- `0.8+` = bon pour la plupart des cas
- Plus utile que l'accuracy quand les classes sont déséquilibrées

**`f1_weighted`** : moyenne pondérée par le support de chaque classe — recommandé comme métrique principale.

---

### Precision (Précision)
**Ce que c'est** : parmi les prédictions positives, quelle proportion est réellement positive.

```
precision = vrais positifs / (vrais positifs + faux positifs)
```

**Quand privilégier** : quand un faux positif est coûteux (ex : alarme incendie déclenchée à tort).

---

### Recall (Rappel / Sensibilité)
**Ce que c'est** : parmi les cas positifs réels, quelle proportion est détectée.

```
recall = vrais positifs / (vrais positifs + faux négatifs)
```

**Quand privilégier** : quand un faux négatif est coûteux (ex : cancer non détecté).

---

### Taux d'erreur
**Ce que c'est** : proportion de prédictions incorrectes.

```
taux_erreur = 1 - accuracy
```

**Interprétation** : `0.05` = 5% des prédictions sont incorrectes.

---

## Métriques de performance (régression)

### MAE (Mean Absolute Error)
**Ce que c'est** : écart moyen absolu entre la prédiction et la valeur réelle.

```
MAE = mean(|y_pred - y_real|)
```

**Interprétation** : dans l'unité de la variable cible. Un MAE de 500 sur un prix d'appartement = erreur moyenne de 500 €.

---

### RMSE (Root Mean Square Error)
**Ce que c'est** : écart quadratique moyen — pénalise davantage les grandes erreurs.

```
RMSE = sqrt(mean((y_pred - y_real)²))
```

**Interprétation** : toujours supérieur ou égal au MAE. Si RMSE >> MAE, il y a des prédictions très éloignées.

---

### R² (Coefficient de détermination)
**Ce que c'est** : proportion de variance de la variable cible expliquée par le modèle.

```
R² = 1 - (SS_résiduel / SS_total)
```

**Interprétation** :
- `1.0` = modèle parfait (explique 100% de la variance)
- `0.0` = modèle équivalent à prédire la moyenne
- Négatif = le modèle est pire que prédire la moyenne

---

## Métriques de latence

### Latence moyenne
**Ce que c'est** : temps de réponse moyen de l'API pour une prédiction.

**Valeurs typiques** : 10–50 ms pour un modèle sklearn simple. Plus si le modèle est volumineux (RandomForest avec 500 arbres) ou si Redis est froid.

---

### Latence P95 (95e percentile)
**Ce que c'est** : 95% des prédictions sont traitées en moins de cette durée.

**Pourquoi P95 plutôt que la moyenne** : la moyenne masque les outliers. Si P95 = 200ms, les 5% de requêtes les plus lentes prennent plus de 200ms.

**Seuil recommandé** : définissez `max_latency_p95_ms` dans votre politique d'auto-promotion (ex : 500ms).

---

### Latence médiane (P50)
**Ce que c'est** : 50% des prédictions sont traitées en moins de cette durée.

---

## Métriques de drift (dérive)

### Z-Score (par feature)
**Ce que c'est** : écart entre la valeur moyenne en production et la baseline, exprimé en nombre d'écarts-types.

```
Z-score = (mean_production - mean_baseline) / std_baseline
```

**Seuils** :
- `|Z| < 2` → `ok`
- `2 ≤ |Z| < 3` → `warning` (dérive modérée)
- `|Z| ≥ 3` → `critical` (dérive forte)

**Interprétation** : un Z-score de 3 signifie que la distribution de production s'est décalée de 3 écarts-types — statistiquement très peu probable sans changement réel.

---

### PSI (Population Stability Index)
**Ce que c'est** : mesure le décalage de distribution entre deux populations (entraînement vs production).

```
PSI = Σ (% prod - % train) × ln(% prod / % train)
```

**Seuils standards** :
- `PSI < 0.1` → `ok` (distribution stable)
- `0.1 ≤ PSI < 0.2` → `warning` (légère dérive)
- `PSI ≥ 0.2` → `critical` (dérive significative — ré-entraînement recommandé)

---

### Null Rate (taux de valeurs manquantes)
**Ce que c'est** : proportion de prédictions récentes où cette feature est `null` ou absente.

**Seuil d'alerte** : configurable par modèle dans `alert_thresholds`. Défaut : `warning` si null rate > 5%, `critical` si > 10%.

---

### Output Drift (Label Shift / Dérive des sorties)
**Ce que c'est** : dérive de la distribution des classes prédites par rapport à la distribution de référence (issue de `training_stats.label_distribution`).

**Calcul** : PSI appliqué aux distributions de classes.

**Interprétation** : si votre modèle prédisait 33% de chaque classe Iris et prédit maintenant 80% de "setosa", la population de production a changé (ou le modèle est biaisé).

---

## Métriques A/B Testing

### P-value
**Ce que c'est** : probabilité d'observer un écart aussi grand entre les versions si elles étaient en réalité équivalentes.

**Interprétation** :
- `p < 0.05` → différence statistiquement significative (95% de confiance)
- `p < 0.01` → très forte évidence (99% de confiance)
- `p ≥ 0.05` → pas assez de données ou pas de différence réelle

**Attention** : ne promotez pas une version simplement parce qu'elle semble meilleure. Attendez que `p < confidence_level`.

---

### Niveau de confiance
**Ce que c'est** : seuil de signification statistique choisi (défaut : 95%).

**Signification** : avec 95% de confiance, on accepte 5% de risque de conclure à tort à une différence.

---

### Test statistique utilisé
| Condition | Test | Métrique testée |
|---|---|---|
| ≥ 1 erreur dans un des groupes | Chi-² (tableau de contingence) | Taux d'erreur |
| 0 erreur + temps de réponse disponibles | Mann-Whitney U | Temps de réponse (ms) |

---

### Winner
**Ce que c'est** : version avec la meilleure métrique (taux d'erreur plus bas ou latence plus faible), mais seulement si `significant: true`.

**Ne jamais promouvoir si `significant: false`** — il faut accumuler plus de données.

---

### min_samples_needed
**Ce que c'est** : nombre d'observations estimées nécessaires par version pour avoir 80% de puissance statistique au seuil configuré.

**Interprétation** : si vous avez 150 observations mais `min_samples_needed: 400`, attendez encore avant de conclure.

---

### Concordance (shadow)
**Ce que c'est** : proportion de cas où le modèle shadow et le modèle de production donnent la même prédiction.

**Interprétation** :
- `> 95%` → les deux modèles sont très similaires
- `80–95%` → différences notables, étude approfondie recommandée
- `< 80%` → comportements très différents

---

## Métriques de calibration

### Brier Score
**Ce que c'est** : mesure la qualité des probabilités prédites.

```
Brier = mean((p_pred - y_real)²)
```

**Interprétation** :
- `0` = probabilités parfaites
- `0.25` = ligne de base (prédire 50% pour tout)
- `< 0.1` = très bien calibré
- `0.1–0.25` = acceptable
- `> 0.25` = mauvaise calibration

---

### Gap de confiance
**Ce que c'est** : écart entre la confiance moyenne prédite et la précision réelle observée.

```
gap = confiance_moyenne_prédite - accuracy_réelle
```

**Interprétation** :
- `> 0` = **surconfiance** : le modèle surestime ses certitudes (prédit 90% de confiance mais n'a raison qu'à 75%)
- `< 0` = **sous-confiance** : le modèle est trop prudent
- `≈ 0` = bien calibré

---

### Statut de calibration
| Statut | Condition |
|---|---|
| `OK` | `|gap| < 0.05` |
| `Surconfiant` | `gap > 0.05` |
| `Sous-confiant` | `gap < -0.05` |

---

## Métriques de confiance

### Confiance moyenne
**Ce que c'est** : `max(probabilities)` moyen sur les prédictions récentes — à quel point le modèle est "sûr" en moyenne.

---

### Distribution de confiance
**Ce que c'est** : histogramme du niveau de confiance (`max(probabilities)`) des prédictions.

**Utilité** : identifier la proportion de prédictions incertaines. Si beaucoup de prédictions ont une confiance < 60%, affinez votre `confidence_threshold`.

---

## Métriques de tendance

### Tendance de performance
**Ce que c'est** : comparaison de l'accuracy entre la première et la seconde moitié de la période sélectionnée.

**Interprétation** : une baisse indique un drift potentiel du modèle ou un changement de la population.

---

## KPIs opérationnels

### Volume de prédictions
**Ce que c'est** : nombre de prédictions effectuées sur une période donnée.

**Suivi** : un pic inhabituel peut signaler un abus (scraping, boucle infinie), une baisse peut signaler une panne upstream.

---

### Quota journalier
**Ce que c'est** : nombre de prédictions autorisées par jour par utilisateur (`rate_limit_per_day`).

**Comportement** : si le quota est atteint, l'API retourne HTTP 429. Le quota se réinitialise à minuit UTC.

---

### Couverture ground truth
**Ce que c'est** : proportion de prédictions ayant un résultat observé associé.

```
couverture = nb_paires(prédiction, résultat_observé) / total_prédictions
```

**Interprétation** : une couverture de 100% signifie que vous pouvez calculer la performance réelle sur toutes les prédictions. En dessous de 20%, les métriques de performance réelle ne sont pas fiables.

---

## Guide de lecture rapide

| Métrique | Bon | Attention | Alerte |
|---|---|---|---|
| Accuracy | > 0.9 | 0.7–0.9 | < 0.7 |
| F1 Score | > 0.85 | 0.6–0.85 | < 0.6 |
| Taux d'erreur | < 0.05 | 0.05–0.15 | > 0.15 |
| Latence P95 | < 100ms | 100–500ms | > 500ms |
| Z-score drift | < 2 | 2–3 | > 3 |
| PSI | < 0.1 | 0.1–0.2 | > 0.2 |
| Brier Score | < 0.1 | 0.1–0.25 | > 0.25 |
| p-value A/B | < 0.05 = significatif | — | > 0.05 = pas concluant |
