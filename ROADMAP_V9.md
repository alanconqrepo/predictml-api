# ROADMAP V9 — Améliorations fonctionnelles & UX

> **Perspective** : Data Scientist / MLOps Engineer utilisant cette plateforme au quotidien pour
> gérer des modèles sklearn en production.
>
> **Périmètre** : Ce document couvre uniquement ce qui est réellement absent après V1–V8.
> Tout ce qui est déjà implémenté (A/B, SHAP, drift, retrain, auto-promotion, calibration,
> purge RGPD, leaderboard, monitoring…) n'est pas relisté.

---

## Tableau de synthèse

| # | Domaine | Fonctionnalité | Priorité | Difficulté |
|---|---------|----------------|----------|------------|
| 1 | UX | Configuration des seuils d'alerte dans l'UI | P1 | S |
| 2 | UX | Bouton "Promouvoir le gagnant" sur la page A/B | P1 | S |
| 3 | UX | Soumission inline d'un résultat observé | P1 | S |
| 4 | UX | Onglet curl/bash dans Code Example | P1 | XS |
| 5 | UX | Confidence threshold éditable dans l'UI | P1 | XS |
| 6 | UX | Recherche texte libre sur les modèles | P1 | XS |
| 7 | UX + API | Baseline auto-calculée à l'upload | P2 | S |
| 8 | UX | Delta d'importance des features avant/après retrain | P2 | S |
| 9 | UX | Export du rapport de supervision | P2 | S |
| 10 | API | Endpoint dédié `GET /models/{name}/retrain-history` | P2 | S |
| 11 | UX | Explorateur "What-if" (sliders → prédiction live) | P3 | M |
| 12 | API + UX | Model card export (résumé structuré stakeholders) | P3 | M |
| 13 | UX | Scatter plot multi-modèles accuracy vs latency | P3 | S |

Légende difficulté : **XS** < 1h · **S** 1–4h · **M** 4–8h · **L** > 1 jour

---

## P1 — Quick wins (valeur quotidienne, effort minimal)

---

### 1. Configuration des seuils d'alerte dans l'UI

#### Pourquoi

Le monitoring est actuellement passif : les métriques sont visibles dans la page Supervision
mais aucun utilisateur ne surveille le dashboard en permanence. Le champ `alert_thresholds`
existe déjà dans `ModelMetadata` (DB + schema), le service email (`src/services/email_service.py`)
et le supervision scheduler (`src/tasks/supervision_scheduler.py`) sont câblés — mais aucune
UI ne permet de configurer ces seuils par modèle.

Sans cette interface, la fonctionnalité d'alerte est invisible et inutilisée. Un modèle peut
dériver pendant des jours sans que personne ne le sache.

#### Comment

**Côté Streamlit — page "Supervision" (ou "Models" onglet alertes) :**
- Ajouter un formulaire d'édition des seuils sous la vue détail d'un modèle :
  - `error_rate_warning` (%) · `error_rate_critical` (%)
  - `latency_p95_warning_ms` · `latency_p95_critical_ms`
  - `accuracy_drop_warning` (delta absolu) · `accuracy_drop_critical`
- Appel à `PATCH /models/{name}/{version}` avec le champ `alert_thresholds` (déjà accepté
  par `ModelUpdateInput`).
- Afficher les seuils actifs sous forme de lignes horizontales colorées sur les graphiques
  de latence et d'error rate (Plotly `add_hline`).

**Fichiers à modifier :**
- `streamlit_app/pages/7_Supervision.py` — ajout du formulaire + overlay graphique
- `streamlit_app/utils/api_client.py` — `update_model()` existe déjà, rien à ajouter

**Aucune modification backend requise.**

---

### 2. Bouton "Promouvoir le gagnant" sur la page A/B

#### Pourquoi

La page A/B calcule la significativité statistique et indique clairement un gagnant (`winner`).
Mais l'action de promotion est découplée : l'admin doit quitter la page, aller dans "Models",
retrouver la version gagnante, et la passer en production manuellement. C'est une friction
inutile qui décourage de conclure les tests A/B rapidement.

L'objectif d'un test A/B est de prendre une décision — l'UI doit la faciliter au même endroit
où la décision est étayée.

#### Comment

**Côté Streamlit — page "A/B Testing" :**
- Quand `ab_significance.significant == True` et `ab_significance.winner` est renseigné,
  afficher un bandeau vert avec :
  - Le nom du gagnant et son p-value
  - Un bouton `st.button("🏆 Promouvoir en production")` (admin uniquement)
- Au clic :
  1. Appel `PATCH /models/{name}/{winner_version}` avec `{"is_production": true}` via
     `api_client.update_model()` (existant)
  2. Désactiver le mode A/B sur les autres versions : `PATCH` avec
     `{"deployment_mode": "production", "traffic_weight": 1.0}`
  3. Afficher un `st.success()` et recharger les métriques

**Fichiers à modifier :**
- `streamlit_app/pages/6_AB_Testing.py`
- Aucune modification backend requise.

---

### 3. Soumission inline d'un résultat observé depuis la vue prédiction

#### Pourquoi

Le workflow actuel est fractionné : l'utilisateur consulte une prédiction dans l'onglet
"Historique" de la page Predictions, connaît le résultat réel, mais doit basculer vers
l'onglet "Importer des résultats" pour le saisir. Ce flux brisé décourage la saisie de
ground truth au fil de l'eau, ce qui nuit à la couverture des `observed_results` (et donc
à la qualité des métriques de performance).

La couverture ground truth est le carburant de toutes les métriques de performance
(accuracy, drift de performance, auto-promotion). Toute friction qui la réduit dégrade
la qualité du monitoring.

#### Comment

**Côté Streamlit — page "Predictions", onglet Historique :**
- Dans la vue détail d'une prédiction (expander), ajouter une section :
  ```
  ── Résultat observé ──────────────────
  [Champ texte / number / select selon le type de modèle]
  [Bouton "Enregistrer le résultat réel"]
  ```
- Si un `observed_result` existe déjà pour ce `id_obs + model_name`, afficher la valeur
  existante en lecture seule avec un indicateur ✅.
- Au clic, appel `POST /observed-results` avec `id_obs`, `model_name`, `observed_result`
  via `api_client` (endpoint existant).
- Pas de rechargement global — juste un `st.success()` inline.

**Fichiers à modifier :**
- `streamlit_app/pages/3_Predictions.py`
- Aucune modification backend requise.

---

### 4. Onglet curl/bash dans la page Code Example

#### Pourquoi

La page "Code Example" ne contient que des exemples Python. Or le cas d'usage le plus
fréquent pour tester rapidement un endpoint est un simple `curl` depuis le terminal —
surtout pour les Data Engineers et DevOps qui intègrent l'API dans des scripts shell ou
des pipelines CI/CD. Aucune modification backend, coût quasi nul.

#### Comment

**Côté Streamlit — page "5_Code_Example.py" :**
- Ajouter un `st.tabs(["Python", "curl / bash", "JavaScript"])` autour des blocs
  de code existants.
- Onglet **curl / bash** : 4 blocs correspondant aux 4 étapes Python :
  1. Upload du modèle (`curl -F`)
  2. Prédiction (`curl -X POST -H "Authorization: Bearer …"`)
  3. Récupérer l'historique (`curl -G`)
  4. Soumettre un résultat observé (`curl -X POST`)
- Onglet **JavaScript** (optionnel) : `fetch()` avec `Authorization` header.
- Les URL et tokens sont pré-remplis depuis `st.session_state` (API URL + token déjà stockés).

**Fichiers à modifier :**
- `streamlit_app/pages/5_Code_Example.py`

---

### 5. Confidence threshold éditable dans l'UI

#### Pourquoi

Le champ `confidence_threshold` de `ModelMetadata` contrôle le flag `low_confidence` dans
chaque réponse de prédiction — un signal clé pour les équipes métier ("ne pas agir sur cette
prédiction"). Ce seuil est configurable via API (`PATCH /models/{name}`) mais aucun formulaire
Streamlit ne l'expose, ce qui force l'admin à utiliser curl pour un paramètre fréquemment ajusté.

#### Comment

**Côté Streamlit — page "Models", section "Modifier les métadonnées" :**
- Ajouter `st.slider("Confidence threshold", 0.0, 1.0, value=current_threshold, step=0.01)`.
- Inclure la valeur dans le payload de `PATCH /models/{name}/{version}` existant.
- Afficher la valeur actuelle dans la vue détail du modèle (déjà affichée en lecture seule,
  la rendre cliquable/éditable).

**Fichiers à modifier :**
- `streamlit_app/pages/2_Models.py`

---

### 6. Recherche texte libre sur les modèles

#### Pourquoi

Avec 20+ modèles en production, les filtres par tag et statut ne suffisent plus pour retrouver
rapidement un modèle par son nom partiel ou sa description. L'absence de recherche force à
faire défiler une liste qui grandit avec le projet.

#### Comment

**Côté Streamlit — page "Models" :**
- Ajouter un `st.text_input("🔍 Rechercher…")` en haut de la liste.
- Filtrer côté client (sur les données déjà chargées via `list_models()`) sur `name` et
  `description` avec un simple `str.lower() in str.lower()`.
- Combiner avec les filtres existants (tag, is_production, algorithm) de façon additive.

**Côté API (optionnel, si volume > 500 modèles) :**
- Ajouter un query param `?search=` à `GET /models` filtrant sur `name ILIKE` et
  `description ILIKE` en SQL.
- Fichier : `src/api/models.py` + `src/services/db_service.py`

**Pour la V9, le filtrage client est suffisant.**

---

## P2 — Effort moyen, valeur MLOps réelle

---

### 7. Baseline auto-calculée à l'upload du modèle

#### Pourquoi

La baseline de features (statistiques de distribution : mean, std, null rate par feature)
est la fondation de toute la détection de drift. Sans baseline, les endpoints `/drift`,
`/monitoring/model/{name}` et les alertes de drift n'ont aucune donnée à comparer.

Problème actuel : le calcul de baseline est une étape manuelle oubliée par la plupart des
utilisateurs. On le voit dans les dashboards où `drift_status: "no_baseline"` est fréquent.
Il faut cliquer sur "Calculer la baseline" dans la page Models, après l'upload, en sachant
que ça existe — ce que les nouveaux utilisateurs ignorent.

#### Comment

**Côté Streamlit — formulaire d'upload dans "Models" :**
- Ajouter une checkbox `☑ Calculer automatiquement la baseline depuis les données de production`
  (cochée par défaut si des prédictions existent pour ce modèle, décochée sinon).
- Valeur configurable : `baseline_days = st.number_input("Fenêtre (jours)", value=30)`.
- Flux post-upload :
  1. `POST /models` → upload réussi
  2. Si checkbox cochée → `POST /models/{name}/{version}/baseline?days={baseline_days}&dry_run=false`
     (endpoint existant : `compute_baseline`)
  3. Afficher le résultat : nombre de features, nombre de samples utilisés.

**Côté API (amélioration optionnelle) :**
- Ajouter un param `auto_baseline: bool = False` à `POST /models` qui déclenche le calcul
  en fin de handler si des prédictions existent pour ce nom de modèle.
- Fichiers : `src/api/models.py`, `src/services/db_service.py`

**Pour la V9, l'approche Streamlit en deux appels séquentiels est suffisante.**

---

### 8. Delta d'importance des features avant/après retrain

#### Pourquoi

Après un retrain, la question critique est : "Le modèle s'appuie-t-il sur les mêmes features
qu'avant ?" Un changement de feature importance indique soit une amélioration (le modèle
a appris de nouveaux patterns) soit un problème de data leakage ou de biais. Actuellement,
l'onglet "Historique des retrains" montre accuracy et F1 mais rien sur l'évolution des
features — une information au moins aussi importante pour un Data Scientist.

#### Comment

**Côté Streamlit — page "Retrain", onglet "Historique" :**
- Dans la vue détail d'un événement de retrain (source version → nouvelle version) :
  1. Appel `GET /models/{name}/feature-importance?version={source_version}` (baseline)
  2. Appel `GET /models/{name}/feature-importance?version={new_version}` (après retrain)
  3. Afficher un graphique à barres côte-à-côte (Plotly grouped bar) avec les top-10 features.
  4. Mettre en évidence les features dont l'importance a varié de plus de ±15% (couleur
     orange/rouge).
- Ajouter une métrique résumée : "Stabilité de l'importance : X% des features ont varié
  de moins de 10%".

**Côté API :**
- Aucune modification requise. `GET /models/{name}/feature-importance` accepte déjà
  un paramètre `version`.

**Fichiers à modifier :**
- `streamlit_app/pages/8_Retrain.py`
- `streamlit_app/utils/api_client.py` — `get_feature_importance(name, version, ...)` existe déjà

---

### 9. Export du rapport de supervision

#### Pourquoi

Les MLOps engineers produisent régulièrement des rapports de monitoring pour les équipes
produit ou la direction technique — "état de santé de nos modèles cette semaine". Actuellement,
toutes les données sont visibles dans le dashboard Supervision mais aucune n'est exportable :
il faut faire des screenshots ou recopier manuellement les métriques.

Un export CSV ou PDF de l'état courant du monitoring permet d'intégrer ce rapport dans
des outils tiers (Confluence, Notion, email hebdo).

#### Comment

**Côté Streamlit — page "Supervision" :**
- Bouton `📥 Exporter le rapport` en haut de la vue globale.
- Générer un CSV (via `pandas.DataFrame.to_csv()`) avec les colonnes :
  - model_name, status, predictions_7d, error_rate, latency_p95, drift_status,
    accuracy_7d, last_retrain, coverage_pct
- Pour chaque modèle, reprendre les données déjà chargées depuis `/monitoring/overview`
  (aucun appel API supplémentaire).
- Offrir aussi un résumé Markdown téléchargeable (`st.download_button`) qui liste
  les modèles en alerte, leurs métriques et les actions recommandées.

**Côté API :**
- Alternative : ajouter `GET /monitoring/overview?format=csv` qui retourne directement
  le CSV streamé — utile pour les intégrations programmatiques (scripts, CI).
- Fichier : `src/api/monitoring.py`

**Pour la V9, l'export client-side en Streamlit est la voie la plus rapide.**

---

### 10. Endpoint dédié `GET /models/{name}/retrain-history`

#### Pourquoi

L'historique des retrains est actuellement éparpillé dans le `ModelHistory` général
(filtrable par `action=retrain`) et dans le champ `training_stats` de chaque version.
Pour obtenir la progression des métriques à travers les retrains successifs, il faut
faire plusieurs appels et assembler les données côté client — ce qui est fait empiriquement
dans la page Retrain mais de façon fragile.

Un endpoint dédié simplifie l'accès, centralise la logique et rend ce flux accessible
à des clients tiers (scripts de monitoring, Grafana, etc.).

#### Comment

**Côté API :**
- Nouveau endpoint : `GET /models/{name}/retrain-history?limit=20&offset=0`
- Filtre dans `ModelHistory` sur `action IN ('retrain', 'auto_promoted')` + join sur
  `ModelMetadata` pour récupérer `training_stats`, `accuracy`, `f1_score`, `auto_promoted`.
- Réponse :
  ```json
  {
    "model_name": "iris",
    "history": [
      {
        "timestamp": "2026-04-01T03:00:00",
        "source_version": "1.0.0",
        "new_version": "1.1.0",
        "trained_by": "scheduler",
        "accuracy": 0.95,
        "f1_score": 0.94,
        "auto_promoted": true,
        "auto_promote_reason": "all criteria met",
        "n_rows": 12450,
        "train_start_date": "2026-03-01",
        "train_end_date": "2026-04-01"
      }
    ],
    "total": 8
  }
  ```
- Nouveau schéma `RetrainHistoryEntry`, `RetrainHistoryResponse` dans `src/schemas/model.py`.

**Fichiers à modifier :**
- `src/api/models.py` — nouveau handler
- `src/services/db_service.py` — `get_retrain_history(model_name, limit, offset)`
- `src/schemas/model.py` — 2 nouveaux schémas
- `streamlit_app/utils/api_client.py` — `get_retrain_history(name, limit)`
- `streamlit_app/pages/8_Retrain.py` — utiliser ce nouvel endpoint

---

## P3 — Engagé mais pertinent pour un projet mature

---

### 11. Explorateur "What-if" (sliders → prédiction live)

#### Pourquoi

Le Data Scientist a souvent besoin d'explorer de façon interactive le comportement d'un
modèle : "Que se passe-t-il si `sepal_length` passe de 5.1 à 7.0 ?" Actuellement, le
testeur de prédiction dans la page Models accepte un JSON manuel — utile mais lent pour
l'exploration. Un explorateur à sliders permet de répondre en quelques secondes à des
questions de sensibilité qui prennent sinon 10 minutes de scripts Python.

C'est l'un des outils les plus utilisés dans les démos et POCs ML auprès des équipes métier.

#### Comment

**Côté Streamlit — nouvel onglet "What-if" dans la page "Models", section détail modèle :**
- Prérequis : le modèle a une `feature_baseline` (pour connaître les features et leur plage).
- Pour chaque feature dans `feature_baseline` :
  - Si valeur numérique : `st.slider(feature, min_val, max_val, default=mean)`
  - Si valeur catégorielle détectée : `st.selectbox(feature, unique_values)`
- Bouton "Prédire" (ou rechargement automatique avec `st.session_state` debounce).
- Afficher : prédiction, probabilité, flag `low_confidence`, SHAP inline (si supporté).
- Visualiser l'évolution : maintenir un historique des combinaisons testées dans la session
  et tracer une courbe "probabilité vs valeur de feature X".

**Côté API :**
- Utilise `POST /predict` existant avec `?explain=true` pour le SHAP inline.
- Aucune modification requise.

**Fichiers à modifier :**
- `streamlit_app/pages/2_Models.py` — ajout d'un onglet `What-if`
- `streamlit_app/utils/api_client.py` — `predict()` existe déjà

**Contrainte :** nécessite que `feature_baseline` soit calculée. Afficher un message
d'invitation à calculer la baseline si elle est absente.

---

### 12. Model card export (résumé structuré pour stakeholders)

#### Pourquoi

Les équipes produit, compliance ou direction technique demandent périodiquement un résumé
d'un modèle en production : quelles données l'ont entraîné, quelles performances il atteint,
s'il dérive, quand il a été retrained. Aujourd'hui, ces informations sont éparpillées dans
5 pages différentes. Un "model card" — document standardisé introduit par Google et adopté
comme best practice MLOps — répond à ce besoin en un clic.

#### Comment

**Côté API — nouveau endpoint :**
```
GET /models/{name}/{version}/card
Accept: application/json | text/markdown
```
- Agrège en un seul appel : metadata du modèle, dernières métriques de performance,
  drift status, calibration score, feature importance top-5, dernière date de retrain,
  couverture observed_results.
- Retourne un JSON structuré ou un Markdown formaté selon le header `Accept`.
- Exemple Markdown généré :
  ```markdown
  # Model Card — iris v2.0.0
  **Algorithme** : RandomForestClassifier
  **Accuracy** : 0.95 | **F1** : 0.94
  **Drift** : ✅ Stable (last check 2026-04-27)
  **Dernier retrain** : 2026-04-01 (scheduler)
  **Features clés** : sepal_length (0.42), petal_width (0.31)…
  ```

**Côté Streamlit — page "Models", section détail modèle :**
- Bouton `📄 Exporter la model card` → `st.download_button` avec le Markdown généré.

**Fichiers à modifier :**
- `src/api/models.py` — nouveau handler `get_model_card()`
- `src/services/db_service.py` — aggrégation multi-source
- `src/schemas/model.py` — `ModelCardResponse`
- `streamlit_app/pages/2_Models.py` — bouton d'export

---

### 13. Scatter plot multi-modèles : accuracy vs latency

#### Pourquoi

Le leaderboard actuel (page Stats) classe les modèles sur une seule métrique à la fois.
Or le choix d'un modèle en production est toujours un compromis : un modèle très précis
mais lent peut être inacceptable pour un endpoint temps-réel, et inversement. Un scatter
plot "accuracy vs latency P95" avec une bulle par modèle (taille = volume de prédictions)
visualise ce front de Pareto immédiatement, sans nécessiter d'export Excel.

#### Comment

**Côté Streamlit — page "Stats", nouvel onglet "Comparaison" :**
- Données déjà disponibles via `GET /models/leaderboard` (accuracy, f1_score,
  latency_p95_ms, predictions_count pour chaque modèle production).
- Plotly `scatter` :
  - X : `latency_p95_ms`
  - Y : `accuracy` (ou `f1_score` selon sélecteur)
  - Taille de bulle : `predictions_count`
  - Couleur : drift status (vert/orange/rouge)
  - Label : `model_name`
- Ajouter des lignes de seuil configurables (latency SLA, accuracy minimum) pour
  visualiser les modèles hors SLA.
- Tooltip au survol : toutes les métriques du modèle.

**Côté API :**
- Aucune modification requise. `GET /models/leaderboard` fournit déjà toutes les données.

**Fichiers à modifier :**
- `streamlit_app/pages/4_Stats.py` — nouvel onglet avec scatter plot

---

## Ce qui a été délibérément exclu

| Fonctionnalité | Raison d'exclusion |
|---|---|
| Multi-armed bandit | Cas d'usage trop spécialisé ; A/B standard couvre 95% des besoins |
| Métriques de fairness / biais | Nécessite des données démographiques absentes du système ; hors scope |
| Orchestration pipeline (Airflow, Prefect) | Scope creep — la plateforme est une API, pas un orchestrateur |
| Intégration registre externe (HuggingFace, W&B) | Doublon avec MLflow déjà intégré |
| Async batch avec queue (Celery, Redis Queues) | Complexité opérationnelle élevée ; batch 120s couvre la majorité des cas |
| Commentaires / annotations collaboratifs | Utile mais hors du flux MLOps central ; 20% d'utilisateurs max |
| Gestion des groupes / équipes | Déjà géré via rôles (admin/user/readonly) ; suffisant pour la cible |
| UI dark mode | Nice-to-have, aucune valeur MLOps |
| Détection de biais temporel (concept drift) | Couvert partiellement par drift PSI ; le reste est applicatif |

---

## Ordre d'implémentation recommandé

Pour une sprint de 2 semaines :

**Semaine 1 — P1 (tout XS/S) :**
1. Items 4, 5, 6 (XS — 30 min chacun)
2. Items 1, 2, 3 (S — 2–3h chacun)

**Semaine 2 — P2 :**
3. Item 7 (baseline auto — S)
4. Item 8 (delta feature importance — S)
5. Item 9 (export supervision — S)
6. Item 10 (endpoint retrain-history — S, seul item avec modification backend)

**Sprint suivant — P3 (si pertinent) :**
7. Item 13 (scatter plot — S, valeur immédiate)
8. Item 11 (what-if explorer — M)
9. Item 12 (model card — M, nécessite endpoint API)
