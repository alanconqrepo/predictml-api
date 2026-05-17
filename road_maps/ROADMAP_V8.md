# ROADMAP V8 — predictml-api

> Perspective Data Scientist / MLOps Engineer  
> Date : 2026-04-26  
> Stack : FastAPI · PostgreSQL · MinIO · MLflow · Redis · Streamlit

---

## Sommaire

1. [Routes API existantes non exposées dans Streamlit](#1-routes-api-existantes-non-exposées-dans-streamlit)
2. [Nouvelles fonctionnalités API proposées](#2-nouvelles-fonctionnalités-api-proposées)
3. [Améliorations UX Streamlit](#3-améliorations-ux-streamlit)
4. [Ce qui est déjà bien couvert — Ne pas toucher](#4-ce-qui-est-déjà-bien-couvert--ne-pas-toucher)
5. [Tableau de synthèse des priorités](#5-tableau-de-synthèse-des-priorités)

---

## 1. Routes API existantes non exposées dans Streamlit

Ces fonctionnalités sont **entièrement implémentées côté API** mais absentes du dashboard Streamlit.  
Ce sont des **quick wins** : le travail backend est fait, seule l'UI manque.

---

### 1.1 `POST /models` — Formulaire d'upload de modèle

**Priorité :** 🔴 CRITIQUE  
**Effort UI estimé :** Moyen (3–4h)

**Pourquoi :**  
C'est le point d'entrée principal du produit — sans ça, il est impossible d'ajouter un modèle sans quitter le dashboard et écrire du code curl/Python. Tout nouvel utilisateur est bloqué dès son arrivée. L'API accepte un `.joblib` multipart + métadonnées optionnelles + un `train.py` optionnel, mais aucun formulaire n'existe dans Streamlit.

**Comment :**  
Dans `streamlit_app/pages/2_Models.py`, ajouter un expander "➕ Uploader un nouveau modèle" avec :
- `st.file_uploader` pour le fichier `.joblib` (pas de limite de taille imposée, streamer via `requests`)
- `st.file_uploader` optionnel pour le `train.py`
- Champs : `name` (text), `version` (text), `description` (text area), `algorithm` (selectbox), `accuracy` / `f1_score` (number inputs), `tags` (text multi-input)
- `st.progress` + spinner pendant l'upload (les .joblib peuvent dépasser 50 Mo)
- Réponse : `st.success()` avec le nom/version créés, ou `st.error()` avec le détail du 422

---

### 1.2 `POST /predict` — Test interactif de prédiction

**Priorité :** 🔴 CRITIQUE  
**Effort UI estimé :** Facile (2h)

**Pourquoi :**  
Un Data Scientist en exploration a besoin de tester une prédiction live immédiatement, sans sortir du dashboard pour écrire du Python. L'absence de ce formulaire oblige à lire la page "Code Example" et à construire une requête à la main, ce qui casse le flux de travail de validation post-déploiement.

**Comment :**  
Dans la vue détail d'un modèle (page 2_Models), ajouter un onglet **"🧪 Tester"** :
- Génération dynamique du formulaire depuis `feature_baseline` ou `expected_features` (récupérés via `/models/{name}/{version}`)
- `st.number_input` pour les features numériques, `st.text_input` pour les catégorielles
- Bouton "Prédire" → POST `/predict` avec `{"model_name": name, "model_version": version, "features": {...}}`
- Affichage : label prédit, tableau de probabilités par classe, latence en ms
- Checkbox optionnel "Avec explication SHAP" → paramètre `?explain=true` → affiche un bar chart des contributions SHAP via `POST /explain`

---

### 1.3 `DELETE /predictions/purge` — Interface de maintenance RGPD

**Priorité :** 🔴 CRITIQUE  
**Effort UI estimé :** Facile (1–2h)

**Pourquoi :**  
Sur un déploiement actif (1 000 prédictions/jour), la table `predictions` grossit sans contrôle. L'endpoint de purge existe et est sécurisé (dry_run par défaut), mais sans UI l'admin doit faire un curl pour nettoyer la base. Conformité RGPD et performances des requêtes analytiques en dépendent.

**Comment :**  
Dans `streamlit_app/pages/3_Predictions.py`, ajouter une section "🗑️ Maintenance" visible uniquement aux admins :
- `st.slider` "Purger les prédictions antérieures à X jours" (7 à 365)
- `st.selectbox` optionnel pour filtrer par modèle
- Bouton "Simuler (dry_run)" → DELETE `/predictions/purge?older_than_days=X&dry_run=true` → affiche `deleted_count`, `linked_observed_results_count` en avertissement si > 0
- Bouton "⚠️ Confirmer la purge" avec `st.dialog` de confirmation → même appel avec `dry_run=false`

---

### 1.4 `GET /models/{name}/feature-importance` — Graphique d'importance des features

**Priorité :** 🟠 HAUTE  
**Effort UI estimé :** Trivial (1h)

**Pourquoi :**  
L'importance des features (via SHAP agrégé sur les dernières prédictions) est disponible via endpoint mais n'est visualisée nulle part. C'est pourtant l'une des premières questions d'un Data Scientist : "quelles features comptent le plus pour ce modèle ?" Répondre à cette question depuis le dashboard est une attente de base.

**Comment :**  
Dans l'onglet "Métriques" de la vue détail du modèle (page 2_Models), ajouter un appel à `/models/{name}/feature-importance` et afficher un `st.bar_chart` horizontal (ou Plotly) des top-15 features triées par importance décroissante. Gérer le cas où SHAP n'a pas encore de prédictions à agréger (`st.info("Pas encore assez de prédictions pour calculer l'importance")`).

---

### 1.5 `POST /predict-batch` — Prédictions en lot via CSV

**Priorité :** 🟠 HAUTE  
**Effort UI estimé :** Moyen (3h)

**Pourquoi :**  
Le workflow typique d'un Data Scientist est : préparer un fichier CSV avec des observations, obtenir un CSV de prédictions en retour. Sans UI, cette opération nécessite du code Python + gestion du token. C'est un cas d'usage à fort volume (scoring de campagne marketing, scoring de crédit batch, etc.) que 80% des équipes data utilisent.

**Comment :**  
Dans `streamlit_app/pages/3_Predictions.py`, ajouter un onglet **"📦 Prédictions batch"** :
- `st.file_uploader` pour un CSV (avec exemple de format affiché)
- `st.selectbox` modèle cible + version
- Bouton "Lancer le scoring" → POST `/predict-batch` avec `timeout=120`
- Affichage du tableau résultat prévisualisé (50 premières lignes)
- `st.download_button` pour télécharger le CSV complet (colonnes originales + `prediction` + `probabilities`)

---

### 1.6 `POST /models/{name}/{version}/validate-input` — Validation de schéma

**Priorité :** 🟡 MOYENNE  
**Effort UI estimé :** Facile (1–2h)

**Pourquoi :**  
Les pannes silencieuses les plus courantes en ML production viennent de features mal nommées ou manquantes. L'endpoint de validation permet de diagnostiquer ces problèmes avant même de lancer une prédiction, mais l'absence d'UI force les développeurs à le tester via curl. L'intégrer dans le formulaire de test (1.2) ou en section dédiée réduit le temps de débogage de pipeline.

**Comment :**  
Dans l'onglet "🧪 Tester" (créé en 1.2), ajouter un sous-expander "Valider le schéma JSON" :
- `st.code_editor` (ou `st.text_area`) avec un exemple pré-rempli depuis `feature_baseline`
- Bouton "Valider" → POST `/models/{name}/{version}/validate-input`
- Affichage coloré : ✅ pour valid, ❌ par erreur (missing/unexpected feature), ⚠️ par warning (type coercion)
- Liste des `expected_features` sous forme de badge

---

### 1.7 `GET /models/{name}/{version}/download` — Téléchargement du fichier .joblib

**Priorité :** 🟡 MOYENNE  
**Effort UI estimé :** Trivial (30 min)

**Pourquoi :**  
Un Data Scientist ou un DevOps peut avoir besoin de récupérer le fichier modèle pour une analyse offline, un audit, ou un redéploiement manuel. Aujourd'hui, il faut accéder directement à la console MinIO. Un simple bouton de téléchargement suffit.

**Comment :**  
Dans la vue détail d'un modèle (page 2_Models), ajouter un `st.download_button` "⬇️ Télécharger le .joblib" qui appelle GET `/models/{name}/{version}/download` et streame le contenu binaire. Afficher la taille du fichier (depuis `file_size_bytes` dans les métadonnées) à côté du bouton.

---

### 1.8 `GET /users/{user_id}/usage` — Analytics par utilisateur

**Priorité :** 🟡 MOYENNE  
**Effort UI estimé :** Facile (1–2h)

**Pourquoi :**  
Un admin a besoin de comprendre qui consomme quoi : identifier les heavy users, détecter des comportements anormaux (burst soudain de prédictions), ou justifier la facturation. L'endpoint retourne déjà `predictions_by_model` et `predictions_by_day` mais n'est nulle part affiché.

**Comment :**  
Dans `streamlit_app/pages/1_Users.py`, dans la vue détail d'un utilisateur (admin uniquement), appeler GET `/users/{user_id}/usage?days=30` et afficher :
- Un bar chart "Prédictions par modèle (30 derniers jours)"
- Une courbe "Volume par jour" avec annotation du quota journalier
- Un KPI card "Consommation quota : X% du quota journalier utilisé aujourd'hui"

---

### 1.9 `GET /models/{name}/readiness` — Score de production-readiness

**Priorité :** 🟡 MOYENNE  
**Effort UI estimé :** Trivial (1h)

**Pourquoi :**  
Avant de promouvoir un modèle en production, l'admin doit vérifier 4 prérequis (fichier accessible, baseline calculée, pas de drift critique, is_production). L'endpoint retourne déjà ces 4 checks, mais l'UI oblige à promouvoir "à l'aveugle". Afficher le score de readiness dans la vue modèle réduit les mauvaises promotions.

**Comment :**  
Dans la vue détail du modèle, au-dessus du bouton "Promouvoir en production", afficher les 4 checks du `/models/{name}/readiness` sous forme de checklist colorée (✅/❌). Désactiver le bouton "Promouvoir" si `ready: false` et expliquer pourquoi.

---

### 1.10 `GET /predictions/{prediction_id}/explain` — Explication SHAP par prédiction

**Priorité :** 🟢 BASSE  
**Effort UI estimé :** Moyen (3h)

**Pourquoi :**  
Quand une prédiction étonne (ex. score de crédit refusé, anomalie détectée), l'utilisateur veut comprendre quelles features l'ont causée. L'endpoint SHAP post-hoc existe mais n'est accessible que via API. Dans l'historique des prédictions, un bouton "Expliquer" par ligne répondrait à ce besoin directement.

**Comment :**  
Dans le tableau de l'historique des prédictions (page 3_Predictions), ajouter un bouton "🔍 Expliquer" par ligne qui appelle GET `/predictions/{prediction_id}/explain` et affiche dans un `st.expander` :
- Un waterfall chart ou bar chart des contributions SHAP (valeur de chaque feature + sa contribution signée)
- La valeur de base (`shap_base_value`) et la prédiction finale
- Limiter aux 10 features les plus influentes pour la lisibilité


---

## 2. Nouvelles fonctionnalités API proposées

Ces fonctionnalités **n'existent pas encore** côté API. Classées par priorité décroissante et effort croissant. Filtre appliqué : utile pour ≥ 80% des utilisateurs MLOps, cohérent avec le périmètre du projet, pas d'usine à gaz.

---

### 2.1 `GET /models/leaderboard` — Classement global des performances

**Priorité :** 🟠 HAUTE  
**Difficulté :** Facile (2h)

**Pourquoi :**  
Avec plusieurs dizaines de modèles en production, il n'existe pas de vue agrégée "qui performe le mieux ?". Le `/monitoring/overview` donne la santé (erreurs, latence) mais pas le classement par performance ML (accuracy, F1). Un leaderboard permet à l'équipe de prendre des décisions rapides : quel modèle promouvoir ? lequel réentraîner en priorité ?

**Comment :**  
Nouvelle route `GET /models/leaderboard?metric=accuracy&days=30` dans `src/api/models.py` :
- Requête sur `ModelMetadata` filtrée sur `is_production=True` + `is_active=True`
- Jointure avec stats agrégées des 30 derniers jours (prédictions, erreurs, latence p95) depuis `DBService`
- Tri configurable par `metric` : `accuracy`, `f1_score`, `latency_p95_ms`, `predictions_count`
- Nouveau schéma `LeaderboardEntry` dans `src/schemas/model.py`
- Réponse example :
```json
[
  {"rank": 1, "name": "iris", "version": "2.0.0", "accuracy": 0.97, "f1_score": 0.96,
   "latency_p95_ms": 45, "drift_status": "ok", "predictions_30d": 12450},
  {"rank": 2, "name": "wine", "version": "1.1.0", "accuracy": 0.94, "f1_score": 0.93,
   "latency_p95_ms": 62, "drift_status": "warning", "predictions_30d": 8320}
]
```

---

### 2.2 `PATCH /models/{name}/{version}/deprecate` — Cycle de vie : dépréciation

**Priorité :** 🟡 MOYENNE  
**Difficulté :** Facile (2–3h — migration Alembic + endpoint + garde dans /predict)

**Pourquoi :**  
Aujourd'hui le cycle de vie d'un modèle est binaire : actif ou supprimé. Il manque un état intermédiaire "deprecated" pour signaler qu'un modèle ne doit plus recevoir de trafic **sans perdre son historique** (prédictions, observed_results, métriques restent consultables). C'est essentiel pour les audits de conformité et pour éviter des suppressions irréversibles accidentelles.

**Comment :**  
1. Ajouter `status: Enum("active", "deprecated", "archived")` dans `ModelMetadata` (migration Alembic `add_model_status`)
2. Nouvelle route `PATCH /models/{name}/{version}/deprecate` dans `src/api/models.py` → set `status="deprecated"`, `is_production=False`
3. Dans `POST /predict`, si le modèle sélectionné a `status="deprecated"`, renvoyer HTTP 410 Gone avec suggestion de la version production courante :
```json
{"detail": "Model iris/1.0.0 is deprecated. Current production: iris/2.0.0"}
```
4. `GET /models` filtre par défaut `status != "archived"` (les archived disparaissent des listes)

---

### 2.3 Export Parquet sur `GET /predictions/export`

**Priorité :** 🟡 MOYENNE  
**Difficulté :** Facile (2h — ajout format dans l'endpoint existant, pas de nouvelle route)

**Pourquoi :**  
L'export de prédictions existe en CSV et JSONL, mais le format Parquet est le standard des pipelines data (Spark, dbt, BigQuery, Snowflake). Un Data Engineer qui veut charger l'historique dans un data warehouse passe systématiquement par Parquet pour ses performances de compression et son typage fort. L'ajouter à l'endpoint existant ne casse rien.

**Comment :**  
Dans `src/api/predict.py`, route `GET /predictions/export` :
- Ajouter `format: Literal["csv", "jsonl", "parquet"] = "csv"` aux query params
- Quand `format="parquet"` : accumuler les lignes dans un `pd.DataFrame`, sérialiser via `df.to_parquet(buffer, index=False, engine="pyarrow")`, retourner en `StreamingResponse` avec `media_type="application/octet-stream"` et header `Content-Disposition: attachment; filename=predictions_{date}.parquet`
- pandas et pyarrow déjà présents dans les dépendances (sklearn/mlflow les tirent)

---

### 2.4 `GET /models/{name}/performance-report` — Rapport consolidé

**Priorité :** 🟢 BASSE  
**Difficulté :** Moyen (4–6h)

**Pourquoi :**  
Pour un bilan mensuel ou un audit, un MLOps doit aujourd'hui assembler manuellement les données de 4–5 endpoints (performance, drift, calibration, A/B, feature importance). Un endpoint consolidé évite N appels réseau depuis le dashboard, et peut servir de base pour générer un rapport PDF automatique en CI/CD.

**Comment :**  
Nouvelle route `GET /models/{name}/performance-report?format=json&days=30` dans `src/api/models.py` :
- Appel parallèle des services existants via `asyncio.gather` : `get_performance()`, `get_drift()`, `get_feature_importance()`, `get_calibration()`, `get_ab_significance()`
- Pas de nouvelle logique métier — uniquement orchestration de services existants
- `format=json` : réponse JSON avec toutes les sections
- `format=html` (optionnel, phase 2) : template Jinja2 minimal retourné en `text/html`

---

### 2.5 Webhooks sur événements modèle — Notifications proactives

**Priorité :** 🟢 BASSE  
**Difficulté :** Moyen (3–4h)

**Pourquoi :**  
Le champ `webhook_url` est déjà stocké dans `ModelMetadata` et des callbacks existent après chaque prédiction. Mais les événements critiques (drift détecté, retrain terminé, auto-promotion, taux d'erreur dépassé) ne déclenchent pas de notification proactive. Un DevOps qui ne surveille pas le dashboard en continu rate ces événements. Une intégration Slack/PagerDuty évite les régressions silencieuses.

**Comment :**  
Dans `src/services/webhook_service.py`, enrichir `send_webhook()` pour accepter un `event_type` et déclencher depuis :
- `retrain_service.py` fin de retrain → event `"retrain_completed"` (accuracy nouvelle version)
- `drift_service.py` quand drift passe en `"critical"` → event `"drift_critical"` (features concernées)
- `auto_promotion_service.py` lors d'une promotion → event `"model_promoted"`
- Scheduler de supervision quand taux d'erreur dépasse seuil → event `"error_rate_threshold"`

Payload unifié :
```json
{
  "event": "drift_critical",
  "model_name": "iris",
  "version": "1.0.0",
  "timestamp": "2026-04-26T14:32:00Z",
  "details": {"feature": "sepal_length", "psi": 0.28, "status": "critical"}
}
```
**Point de vigilance :** retry max 3 avec backoff exponentiel + timeout 5s pour ne pas bloquer les workflows sur une URL webhook indisponible.

---

### ❌ Fonctionnalités explicitement exclues

| Feature | Raison d'exclusion |
|---------|-------------------|
| Pipeline d'entraînement visuel (drag & drop) | Hors périmètre, complexité majeure |
| Marketplace de modèles | Hors périmètre, nécessite multi-tenant |
| Fine-tuning LLM intégré | Hors périmètre |
| Streaming de prédictions SSE/WebSocket | Cas d'usage < 20% des utilisateurs |
| Gestion multi-tenant / organisations | Refactoring architectural majeur |
| AutoML (sélection automatique d'algorithme) | Dépasse le rôle de la plateforme |

---

## 3. Améliorations UX Streamlit

Ces améliorations ne nécessitent **pas de nouvelles routes API** (sauf mention contraire). Elles fluidifient l'expérience existante.

---

### 3.1 Page 2_Models — Matrice de confusion (heatmap)

**Priorité :** 🟠 HAUTE  
**Effort :** Trivial (1h)

**Pourquoi :**  
La matrice de confusion est la première chose qu'un Data Scientist regarde pour évaluer un classificateur. L'endpoint `GET /models/{name}/performance` retourne **déjà** `confusion_matrix` + métriques par classe (precision/recall/F1 per label), mais ce bloc est ignoré dans l'UI qui n'affiche que l'accuracy globale. C'est une perte d'information majeure pour les modèles multi-classes.

**Comment :**  
Dans l'onglet "Métriques" de la vue détail du modèle (page 2_Models ou 4_Stats), parser le champ `confusion_matrix` de la réponse `/performance` et afficher une heatmap Plotly (`px.imshow`) avec les labels en axes X/Y. Ajouter un tableau "Métriques par classe" (precision, recall, F1 par label) en dessous.

---

### 3.2 Page 8_Retrain — Historique des retrains

**Priorité :** 🟠 HAUTE  
**Effort :** Facile (2h)

**Pourquoi :**  
Après plusieurs cycles de retrain (manuel ou schedulé), l'admin n'a aucune vue chronologique de "qui a entraîné quoi, quand, avec quels résultats". La page 8_Retrain affiche les schedules mais pas l'historique des versions créées par retrain. L'endpoint `GET /models/{name}/history` retourne déjà ces données (avec `trained_by="scheduler"` ou le user_id).

**Comment :**  
Dans la page `8_Retrain.py`, ajouter un onglet **"📜 Historique"** :
- Appel à `GET /models/{name}/history` pour le modèle sélectionné
- Tableau filtré sur `action_type = "retrain"` avec colonnes : date, version créée, trained_by, accuracy, f1_score, auto_promoted (oui/non)
- Graphique timeline de l'accuracy des versions retrainées pour visualiser la progression

---

### 3.3 Page 4_Stats — Leaderboard des modèles en production

**Priorité :** 🟠 HAUTE  
**Effort :** Facile (1h) — dépend de 2.1

**Pourquoi :**  
La page Stats actuelle montre des stats par modèle en tableaux séparés. Il manque une vue de classement qui permette de répondre immédiatement : "quel est mon meilleur modèle en ce moment ?". Un tableau classé avec code couleur (vert/orange/rouge) par accuracy donne une vision executive en un coup d'œil.

**Comment :**  
Dans `4_Stats.py`, ajouter un premier bloc "🏆 Leaderboard" qui appelle `GET /models/leaderboard` (route 2.1) et affiche un `st.dataframe` avec colonnes triables et coloration conditionnelle par `st.data_editor` ou Pandas Styler. Si la route 2.1 n'est pas encore implémentée, construire le leaderboard côté client depuis les données déjà chargées.

---

### 3.4 Navigation — Recherche et filtrage des modèles

**Priorité :** 🟡 MOYENNE  
**Effort :** Trivial (1h)

**Pourquoi :**  
Avec 20+ modèles, les `st.selectbox` deviennent difficiles à utiliser — il faut scroller dans une liste sans pouvoir taper pour filtrer. Streamlit 1.29+ supporte le paramètre `search` sur les selectbox. C'est une amélioration d'ergonomie fondamentale qui n'a aucune dépendance backend.

**Comment :**  
Dans toutes les pages qui ont un sélecteur de modèle (2_Models, 3_Predictions, 4_Stats, 6_AB_Testing, 7_Supervision, 8_Retrain), ajouter un `st.text_input("Filtrer par nom", key="model_search")` qui filtre la liste avant de l'afficher dans le `st.selectbox`. Alternativement, utiliser `st.selectbox(..., placeholder="Rechercher un modèle...")` si la version de Streamlit le supporte.

---

### 3.5 Feedback utilisateur — `st.toast()` pour les actions admin

**Priorité :** 🟡 MOYENNE  
**Effort :** Trivial (1–2h à appliquer sur toutes les pages)

**Pourquoi :**  
Actuellement, les confirmations d'actions admin (promotion, suppression, retrain lancé, token régénéré) s'affichent dans un `st.success()` ou `st.error()` qui disparaît au prochain rerender. Sur un dashboard avec plusieurs colonnes, l'utilisateur peut louper le feedback. `st.toast()` (notification non-bloquante en bas à droite) est plus visible et ne perturbe pas le layout.

**Comment :**  
Remplacer tous les `st.success("✅ Modèle promu")` / `st.error("❌ Erreur")` dans les callbacks d'action admin (promotion, suppression, retrain, purge, régénération token) par `st.toast("Modèle iris/2.0.0 promu en production", icon="✅")`. Conserver les `st.error()` pour les erreurs persistantes (formulaire invalide, etc.).

---

### 3.6 Page 3_Predictions — Export serveur-side (CSV/JSONL/Parquet)

**Priorité :** 🟡 MOYENNE  
**Effort :** Facile (1h)

**Pourquoi :**  
L'export actuel dans la page Prédictions construit un CSV côté client depuis le `pd.DataFrame` Streamlit — ce qui signifie que seules les 1 000 lignes chargées en mémoire sont exportées, pas toutes les prédictions qui correspondent aux filtres. L'endpoint `GET /predictions/export` supporte l'export streamé sans limite de volume, mais il n'est pas utilisé.

**Comment :**  
Dans `3_Predictions.py`, remplacer le `st.download_button` qui encode le DataFrame local par un appel à `GET /predictions/export?format=csv&model_name=X&start_date=Y...` avec les mêmes filtres que la vue courante. Streamer la réponse directement dans `st.download_button(data=response.content, ...)`. Ajouter un sélecteur `format` (CSV / JSONL / Parquet si 2.3 implémenté).

---

### 3.7 Page 7_Supervision — Lien direct vers la page Models

**Priorité :** 🟡 MOYENNE  
**Effort :** Trivial (30 min)

**Pourquoi :**  
Depuis la page Supervision, quand un modèle affiche une alerte (drift critique, taux d'erreur élevé), l'utilisateur veut naturellement aller sur la page Modèles pour agir (réentraîner, promouvoir une autre version, rollback). Aujourd'hui, il doit naviguer manuellement dans la sidebar. Un lien contextuel réduit les frictions.

**Comment :**  
Dans la vue détail d'un modèle sur `7_Supervision.py`, ajouter des `st.page_link` vers :
- `"pages/2_Models.py"` avec query param `?model=name` pour ouvrir directement la vue détail du modèle concerné
- `"pages/8_Retrain.py"` avec query param `?model=name` pour pré-sélectionner le modèle dans le formulaire de retrain
Utiliser `st.query_params` côté destination pour lire et pré-sélectionner le modèle automatiquement.

---

### 3.8 Performance — Mise en cache des listes de modèles

**Priorité :** 🟡 MOYENNE  
**Effort :** Facile (1h)

**Pourquoi :**  
Sur un dashboard Streamlit multi-pages, `GET /models` est appelé à chaque rerender de chaque page (sidebar, sélecteurs, tableaux). Sur un serveur distant avec 50+ modèles, cela génère N requêtes identiques par session. `@st.cache_data(ttl=30)` réduit la charge API sans impact fonctionnel (30 secondes de staleness est acceptable pour une liste de modèles).

**Comment :**  
Dans `streamlit_app/utils/api_client.py`, wrapper la fonction `get_models()` avec `@st.cache_data(ttl=30)`. Même chose pour `get_model_detail()` avec `ttl=10`. Ajouter un bouton "🔄 Rafraîchir" dans les pages critiques qui appelle `st.cache_data.clear()` pour forcer la mise à jour.

---

### 3.9 Page 1_Users — Indicateur de consommation quota visuel

**Priorité :** 🟢 BASSE  
**Effort :** Trivial (30 min)

**Pourquoi :**  
La page Users affiche `remaining_today` et `rate_limit_per_day` en texte brut. Un utilisateur qui approche de son quota ne le remarque pas facilement. Une barre de progression colorée (vert → orange → rouge) rend l'information immédiatement lisible.

**Comment :**  
Dans `1_Users.py`, remplacer l'affichage textuel du quota par un `st.progress(used/limit)` avec coloration conditionnelle via CSS inline :
- < 70% : vert (`st.success`)
- 70–90% : orange (`st.warning`)
- > 90% : rouge (`st.error`)
Afficher le tooltip "X / Y prédictions utilisées aujourd'hui — reset à minuit UTC".

---

### 3.10 Tooltips sur les métriques techniques

**Priorité :** 🟢 BASSE  
**Effort :** Trivial (2h à déployer sur toutes les pages)

**Pourquoi :**  
Les métriques avancées affichées dans le dashboard (Brier score, PSI, Cohen h, p-value, ECE) ne sont pas familières pour tous les utilisateurs du dashboard. Un product manager ou un responsable d'équipe qui consulte le dashboard ne sait pas interpréter un Brier score de 0.12. Des tooltips concis évitent des questions de support répétitives.

**Comment :**  
Utiliser le paramètre `help` des widgets Streamlit (`st.metric(..., help="Le Brier score mesure la précision des probabilités prédites. 0 = parfait, 1 = pire cas.")`) pour chaque métrique technique dans les pages 4_Stats, 6_AB_Testing, 7_Supervision. Créer un dictionnaire `METRIC_HELP` dans `streamlit_app/utils/` pour centraliser les définitions et les réutiliser dans toutes les pages.

---

## 4. Ce qui est déjà bien couvert — Ne pas toucher

Les fonctionnalités suivantes sont complètes, testées, et ne nécessitent pas d'amélioration majeure. Les modifier sans raison précise introduirait du risque sans valeur ajoutée.

| Fonctionnalité | Endpoints couverts | Streamlit |
|---|---|---|
| Versioning & historique des modèles | `GET /models/{name}/{version}/history`, rollback | ✅ Page 2_Models |
| A/B testing & shadow deployment | `GET /models/{name}/ab-compare`, routing automatique | ✅ Page 6_AB_Testing |
| Significativité statistique A/B | Chi², Mann-Whitney U, Cohen h/d, puissance statistique | ✅ Page 6_AB_Testing |
| Retraining planifié | APScheduler + cron + Redis lock multi-réplicas | ✅ Page 8_Retrain |
| Auto-promotion avec politique configurable | `PATCH /models/{name}/policy`, évaluation post-retrain | ✅ Page 8_Retrain |
| Drift detection (Z-score + PSI) | `GET /models/{name}/drift`, supervision automatique | ✅ Page 7_Supervision |
| Monitoring temps réel | `GET /monitoring/overview`, `GET /monitoring/model/{name}` | ✅ Page 7_Supervision |
| Calibration des probabilités | Brier score, diagramme de fiabilité | ✅ Page 7_Supervision |
| Validation du schéma d'entrée | `POST /validate-input`, mode strict `/predict` | ✅ (API seulement) |
| Auth Bearer token + rate limiting | Quota journalier, middleware async | ✅ Toutes les pages |
| Gestion complète des utilisateurs | CRUD + token + quota | ✅ Page 1_Users |
| Export streaming (CSV/JSONL) | `/predictions/export`, `/observed-results/export` | ✅ Page 3_Predictions |
| Purge RGPD | `DELETE /predictions/purge` avec dry_run | ✅ (API seulement) |
| Observed results & ground truth | Upload CSV, stats de couverture | ✅ Page 3_Predictions |

---

## 5. Tableau de synthèse des priorités

| # | Type | Fonctionnalité | Page / Fichier | Priorité | Difficulté | Valeur métier |
|---|------|----------------|---------------|----------|------------|---------------|
| 1 | UI | Upload modèle (POST /models) | 2_Models.py | 🔴 CRITIQUE | Moyen | Bloquant pour tout nouvel utilisateur |
| 2 | UI | Test interactif de prédiction (POST /predict) | 2_Models.py | 🔴 CRITIQUE | Facile | Essentiel pour validation post-déploiement |
| 3 | UI | Purge RGPD (DELETE /predictions/purge) | 3_Predictions.py | 🔴 CRITIQUE | Facile | Conformité + performances DB |
| 4 | UI | Feature importance (GET /feature-importance) | 2_Models.py | 🟠 HAUTE | Trivial | Explicabilité immédiate |
| 5 | UI | Matrice de confusion (données dans /performance) | 2_Models.py ou 4_Stats.py | 🟠 HAUTE | Trivial | Standard évaluation classification |
| 6 | UI | Batch prediction via CSV (POST /predict-batch) | 3_Predictions.py | 🟠 HAUTE | Moyen | Workflow DS à fort volume |
| 7 | UI | Historique des retrains | 8_Retrain.py | 🟠 HAUTE | Facile | Traçabilité des cycles d'entraînement |
| 8 | API | Leaderboard (GET /models/leaderboard) | Nouveau + 4_Stats.py | 🟠 HAUTE | Facile | Vision globale des perfs en production |
| 9 | UI | Export serveur-side (GET /predictions/export) | 3_Predictions.py | 🟡 MOYENNE | Facile | Cohérence export + volumes > 1 000 lignes |
| 10 | UI | Validate input schema UI | 2_Models.py | 🟡 MOYENNE | Facile | Débogage pipeline sans curl |
| 11 | UI | Download modèle .joblib | 2_Models.py | 🟡 MOYENNE | Trivial | Workflow DS offline |
| 12 | UI | Model readiness checklist | 2_Models.py | 🟡 MOYENNE | Trivial | Évite les promotions prématurées |
| 13 | UI | Analytics par utilisateur | 1_Users.py | 🟡 MOYENNE | Facile | Admin reporting |
| 14 | API | Deprecate model status | src/api/models.py | 🟡 MOYENNE | Facile | Cycle de vie explicite sans suppression |
| 15 | UX | st.toast() pour actions admin | Toutes les pages | 🟡 MOYENNE | Trivial | Feedback visible sans bloquer le layout |
| 16 | UX | Recherche/filtre modèles dans selectbox | Toutes les pages | 🟡 MOYENNE | Trivial | Ergonomie à 20+ modèles |
| 17 | UX | Liens directs Supervision → Models/Retrain | 7_Supervision.py | 🟡 MOYENNE | Trivial | Réduction des frictions dans le workflow |
| 18 | UX | Cache @st.cache_data(ttl=30) sur GET /models | utils/api_client.py | 🟡 MOYENNE | Facile | Réduction charge API, meilleure réactivité |
| 19 | API | Export Parquet (GET /predictions/export) | src/api/predict.py | 🟡 MOYENNE | Facile | Standard data pipeline |
| 20 | UI | SHAP par prédiction (GET /explain) | 3_Predictions.py | 🟡 MOYENNE | Moyen | Débogage prédiction individuelle |
| 21 | API | Webhooks événements modèle | src/services/webhook_service.py | 🟢 BASSE | Moyen | Intégration Slack/PagerDuty proactive |
| 22 | API | Performance report consolidé | src/api/models.py | 🟢 BASSE | Moyen | Reporting automatisé CI/CD |
| 23 | UX | Barre de progression quota | 1_Users.py | 🟢 BASSE | Trivial | UX lisibilité quota |
| 24 | UX | Tooltips métriques techniques | 4_Stats, 6_AB, 7_Supervision | 🟢 BASSE | Trivial | Réduction questions de support |

---

*ROADMAP_V8 générée le 2026-04-26 — à réviser après chaque sprint.*
