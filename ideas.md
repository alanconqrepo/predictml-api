# Ideas de fonctionnalités — predictml-api

> Perspective Data Scientist / MLOps Engineer  
> Critères : valeur ajoutée réelle, faisabilité sans réécriture majeure, cohérence avec l'existant

---

## Grille de lecture

| | Facile (< 1 jour) | Moyenne (1–3 jours) | Complexe (> 3 jours) |
|---|---|---|---|
| **Forte valeur** | ✅ Priorité 1 | 🔶 Priorité 2 | 🔴 Long terme |
| **Moyenne valeur** | 🟢 Quick wins | 🟡 Backlog | ⬜ À évaluer |
| **Faible valeur** | ⬜ Nice to have | ⬜ Skip | ⬜ Skip |

---

## Forte valeur ajoutée

### Facile à implémenter

#### 1. Enforcement du rate limiting ✅
**Quoi** : Le champ `rate_limit_per_day` existe dans la table `users` mais n'est pas vérifié dans l'API.  
**Comment** : Ajouter un middleware ou une dépendance FastAPI qui compte les prédictions du jour (`SELECT COUNT` sur `predictions` pour `user_id` + `timestamp >= today`) et lève une `429 Too Many Requests` si dépassé.  
**Fichiers** : `src/api/predict.py`, `src/core/security.py`  
**Impact** : Protection directe de l'infrastructure, respect des quotas par client.

---

#### 2. Endpoint batch prediction — `POST /predict-batch`
**Quoi** : Accepter une liste de feature dicts et retourner une liste de prédictions en une seule requête.  
**Comment** : Réutiliser la logique de `POST /predict`, boucler sur les inputs, persister toutes les prédictions en une transaction batch (`session.add_all`).  
**Schéma** :
```json
// Request
{"inputs": [{"feature1": 1.2, "feature2": 0.5}, ...], "model_name": "iris_model"}

// Response
{"predictions": [{"result": "setosa", "probabilities": [0.9, 0.05, 0.05]}, ...]}
```
**Fichiers** : `src/api/predict.py`, `src/schemas/`  
**Impact** : Débloque les cas d'usage scoring en masse (ex: batch nocturne).

---

#### 3. Endpoint performance modèle — `GET /models/{name}/performance`
**Quoi** : Calculer l'accuracy réelle en production en joinant `predictions` et `observed_results` via `id_obs`.  
**Comment** : Requête SQL : `JOIN observed_results ON predictions.id_obs = observed_results.id_obs AND predictions.model_name = observed_results.model_name`, comparer `prediction_result` vs `observed_result`, agréger par période.  
**Fichiers** : `src/api/models.py`, `src/services/db_service.py`  
**Impact** : Ferme la boucle train→serve→evaluate. Donne une accuracy live, pas juste les métriques de training.

---

#### 4. Seuil de confiance par modèle
**Quoi** : Ajouter un champ `confidence_threshold` (float, nullable) dans `model_metadata`. Si la proba max de la prédiction est en dessous du seuil, retourner un flag `low_confidence: true` dans la réponse.  
**Comment** : Ajouter la colonne DB, mettre à jour le schéma Pydantic, lire la valeur dans le service de prédiction.  
**Fichiers** : `src/db/models.py`, `src/schemas/`, `src/services/model_service.py`  
**Impact** : Permet aux consommateurs de l'API de déclencher un fallback (avis humain, modèle alternatif) sur les cas incertains.

---

#### 5. Filtre par `id_obs` dans l'historique de prédictions
**Quoi** : Ajouter `?id_obs=<value>` comme query param sur `GET /predictions`.  
**Comment** : Modifier la requête SQLAlchemy dans `db_service.py` pour filtrer si le param est fourni.  
**Fichiers** : `src/api/predict.py`, `src/services/db_service.py`  
**Impact** : Traçabilité complète — on peut retrouver toute la chaîne d'une observation depuis son identifiant métier.

---

#### 6. Export CSV des prédictions (dashboard Streamlit)
**Quoi** : Bouton "Télécharger en CSV" sur la page `3_Predictions.py`.  
**Comment** : `st.download_button(data=df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")`.  
**Fichiers** : `streamlit_app/pages/3_Predictions.py`  
**Impact** : Permet aux data scientists d'analyser les prédictions dans Excel/notebooks sans accès DB.

---

### Moyenne difficulté (1–3 jours)

#### 7. Suivi d'accuracy dans le temps (drift de performance)
**Quoi** : Courbe d'accuracy rolling (7j, 30j) calculée à partir des `observed_results` jointés aux `predictions`. Visualisation dans le dashboard page Stats.  
**Comment** : Requête SQL avec fenêtre temporelle + agrégat par jour. Plotly line chart dans `4_Stats.py`. Optionnel : alert si accuracy < seuil configurable sur le modèle.  
**Fichiers** : `src/services/db_service.py`, `streamlit_app/pages/4_Stats.py`  
**Impact** : Détection précoce de dégradation de performance en production sans outil externe.

---

#### 8. Détection de data drift (statistique)
**Quoi** : Comparer la distribution des features en production (issues des `predictions.input_features`) à une baseline enregistrée à l'upload du modèle.  
**Comment** : 
- À l'upload : stocker `feature_baseline` (moyenne + std par feature) dans `model_metadata.training_params`
- En production : calculer les stats sur la fenêtre glissante, appliquer un test KS ou PSI
- Endpoint `GET /models/{name}/drift` retourne un score par feature
**Dépendance** : `scipy.stats.ks_2samp` (déjà dans l'écosystème scipy)  
**Fichiers** : `src/api/models.py`, `src/services/model_service.py`, `src/db/models.py`  
**Impact** : Alerte sur les changements de distribution des données en entrée, avant que l'accuracy ne chute.

---

#### 9. Explainability locale — `POST /explain`
**Quoi** : Retourner les feature importances locales pour une prédiction donnée.  
**Comment** : Utiliser `shap.TreeExplainer` (compatible RandomForest, GBM) ou `shap.LinearExplainer` (compatible LogisticRegression). Endpoint accepte les mêmes inputs que `/predict`, retourne un dict `{feature: shap_value}`.  
**Dépendance** : `shap` (à ajouter dans `requirements.txt` et `pyproject.toml`)  
**Fichiers** : `src/api/predict.py`, `src/services/model_service.py`  
**Impact** : Répond aux exigences de transparence (RGPD Article 22, conformité secteur financier). Aide au debug des prédictions surprenantes.

---

#### 10. Migrations Alembic
**Quoi** : Remplacer `Base.metadata.create_all()` au démarrage par des migrations versionnées.  
**Comment** : `alembic init alembic`, configurer `env.py`, créer une migration initiale depuis les modèles existants. Les prochaines évolutions de schéma (`confidence_threshold`, `tags`, etc.) s'ajoutent via `alembic revision --autogenerate`.  
**Fichiers** : `alembic/`, `src/main.py`, `src/db/`  
**Impact** : Indispensable pour déployer des évolutions de schéma sans perdre les données. Actuellement tout changement de DB détruit les données.

---

#### 11. CI/CD GitHub Actions
**Quoi** : Pipeline automatique : `ruff check` + `black --check` + `pytest tests/ -v` sur chaque push/PR.  
**Comment** : Fichier `.github/workflows/ci.yml`, matrix Python 3.11+, cache pip.  
**Fichiers** : `.github/workflows/ci.yml`  
**Impact** : Évite de merger du code cassé. Prérequis pour ouvrir la contribution externe.

---

#### 12. Logging structuré (JSON)
**Quoi** : Remplacer les `print()` et logs non structurés par `structlog` ou `loguru` en format JSON.  
**Comment** : Configurer un logger global dans `src/core/config.py`, remplacer toutes les occurrences de `print()` (3 identifiées) et les `logger.info()` bruts.  
**Dépendance** : `structlog` ou `loguru`  
**Fichiers** : `src/core/config.py`, `src/api/*.py`, `src/services/*.py`  
**Impact** : Logs exploitables par ELK/Datadog/CloudWatch. Indispensable en production multi-instance.

---

### Complexe / Long terme

#### 13. A/B testing et déploiement canary
**Quoi** : Configurer un split de trafic entre deux versions d'un modèle (ex : 90% v1 / 10% v2) et comparer leurs métriques automatiquement.  
**Comment** : Nouveau champ `traffic_weight` dans `model_metadata`. Logique de routage dans `model_service.py`. Agrégation des métriques par version dans les stats.  
**Impact** : Valider une nouvelle version en production sans risque, avec données réelles.

---

#### 14. Alertes automatiques (webhook / email)
**Quoi** : Déclencher une notification si l'accuracy chute sous un seuil ou si un drift est détecté.  
**Comment** : Background task FastAPI (APScheduler ou Celery Beat) qui tourne toutes les X heures, évalue les métriques, et appelle un webhook configuré ou envoie un email via SMTP.  
**Impact** : Passe d'un système "pull" (il faut aller regarder) à "push" (on est alerté).

---

#### 15. Pipeline de retraining automatique
**Quoi** : Déclencher un retraining quand suffisamment de `observed_results` sont accumulés ou quand une dégradation est détectée.  
**Comment** : Orchestrateur léger (Prefect ou simple APScheduler), accès aux données via DB, log dans MLflow, upload automatique du nouveau modèle via l'API.  
**Impact** : Ferme la boucle MLOps complète : données → modèle → production → monitoring → retraining.

---

## Moyenne valeur ajoutée

### Facile à implémenter

#### 16. Upload de modèle dans Streamlit
**Quoi** : Formulaire dans le dashboard pour uploader un `.pkl` directement depuis le navigateur.  
**Comment** : `st.file_uploader()` + champs pour les métadonnées + appel à `POST /models` via `api_client.py`.  
**Fichiers** : `streamlit_app/pages/2_Models.py`, `streamlit_app/utils/api_client.py`  
**Impact** : Autonomie des data scientists sans accès CLI ni `curl`.

---

#### 17. Comparaison côte-à-côte de versions de modèle
**Quoi** : Sélecteur multi-version dans la page Models du dashboard, tableau comparatif `accuracy / F1 / precision / recall / nb prédictions / accuracy live`.  
**Comment** : Récupérer plusieurs modèles via `GET /models/{name}/{version}`, concaténer en DataFrame, afficher avec `st.dataframe` et highlighting conditionnel.  
**Fichiers** : `streamlit_app/pages/2_Models.py`  
**Impact** : Décision éclairée pour la promotion en production.

---

#### 18. Endpoint stats agrégées — `GET /predictions/stats`
**Quoi** : Retourner des agrégats pré-calculés (count, error_rate, p50/p95 response_time, accuracy live) par modèle et période, sans charger toutes les lignes.  
**Comment** : Requête SQL avec `GROUP BY`, `PERCENTILE_CONT` (PostgreSQL natif), filtre `?model_name=&days=30`.  
**Fichiers** : `src/api/predict.py`, `src/services/db_service.py`  
**Impact** : Dashboard plus rapide sur gros volumes, API exploitable par des outils tiers (Grafana, etc.).

---

#### 19. Tags personnalisés sur les modèles
**Quoi** : Champ JSON `tags` dans `model_metadata` (ex: `["production", "finance", "v2-experiment"]`). Filtre `?tag=` sur `GET /models`.  
**Comment** : Ajouter la colonne `tags JSONB` (PostgreSQL), mettre à jour schémas Pydantic, adapter la requête de listing.  
**Fichiers** : `src/db/models.py`, `src/schemas/`, `src/api/models.py`  
**Impact** : Organisation et découverte des modèles dans les projets avec beaucoup de versions.

---

#### 20. Validation de taille à l'upload de modèle
**Quoi** : Retourner `413 Request Entity Too Large` si le `.pkl` dépasse une taille configurable (ex : 500 MB).  
**Comment** : Lire `file.size` dans l'endpoint `POST /models`, comparer à `settings.MAX_MODEL_SIZE_MB`.  
**Fichiers** : `src/api/models.py`, `src/core/config.py`  
**Impact** : Évite les crashs mémoire lors du chargement de gros modèles.

---

#### 21. Marquage manuel d'une prédiction (correcte / incorrecte)
**Quoi** : Champ `is_correct` (bool, nullable) dans la table `predictions`, mise à jour via `PATCH /predictions/{id}`.  
**Comment** : Alternative légère à `observed_results` pour les cas simples (feedback binaire sans valeur exacte).  
**Fichiers** : `src/db/models.py`, `src/api/predict.py`, `src/schemas/`  
**Impact** : Collecte de feedback simplifié pour les utilisateurs finaux.

---

### Moyenne difficulté (1–3 jours)

#### 22. Profil de features baseline par modèle
**Quoi** : Stocker `{feature: {mean, std, min, max, dtype}}` à l'enregistrement du modèle (depuis les données d'entraînement). Comparer aux features reçues en production pour détecter les anomalies avant prédiction.  
**Comment** : Champ `feature_schema` JSON dans `model_metadata`. Validation optionnelle à chaque appel `/predict` (warning non bloquant).  
**Fichiers** : `src/db/models.py`, `src/services/model_service.py`  
**Impact** : Détection précoce de features out-of-range ou encodées différemment.

---

#### 23. Webhook sortant post-prédiction
**Quoi** : Configurer une URL de callback par modèle. L'API envoie un `POST` asynchrone avec le payload de la prédiction après chaque appel.  
**Comment** : Champ `webhook_url` dans `model_metadata`. Background task FastAPI (`BackgroundTasks`) pour l'appel HTTP.  
**Fichiers** : `src/db/models.py`, `src/api/predict.py`  
**Impact** : Intégration avec des systèmes tiers (Zapier, Slack, systèmes métier) sans polling.

---

#### 24. Cache Redis (remplacement du cache in-memory)
**Quoi** : Remplacer le dict Python `model_cache` par Redis pour partager le cache entre plusieurs instances de l'API.  
**Comment** : `redis-py` + sérialisation pickle des modèles. Clé `model:{name}:{version}`, TTL configurable.  
**Dépendance** : `redis`, service Redis dans `docker-compose.yml`  
**Fichiers** : `src/services/model_service.py`, `docker-compose.yml`  
**Impact** : Prérequis pour le passage en mode multi-pod (Kubernetes, scaling horizontal).

---

## Faible valeur ajoutée / Nice to have

### Facile à implémenter

#### 25. Copie du token API en un clic (dashboard)
**Quoi** : Bouton "Copier" à côté du token affiché dans la page Users.  
**Comment** : `st.code(token)` + `st.button("Copier")` avec `pyperclip` ou via JavaScript `navigator.clipboard`.  
**Fichiers** : `streamlit_app/pages/1_Users.py`

---

#### 26. Indication "last seen" sur les modèles
**Quoi** : Afficher la date/heure de la dernière prédiction pour chaque modèle dans le listing.  
**Comment** : `MAX(timestamp) GROUP BY model_name` dans la requête de listing.  
**Fichiers** : `src/services/db_service.py`, `src/api/models.py`

---

#### 27. Pagination cursor-based sur `/predictions`
**Quoi** : Remplacer `offset` par un curseur (dernier `id` vu) pour des résultats stables sur gros volumes.  
**Comment** : `WHERE id < :cursor ORDER BY id DESC LIMIT :limit`, retourner le cursor dans la réponse.  
**Fichiers** : `src/api/predict.py`, `src/services/db_service.py`  
**Impact** : Évite les doublons et sauts de pages quand de nouvelles prédictions arrivent pendant la navigation.

---

## Résumé priorisé

| Priorité | Fonctionnalité | Valeur | Difficulté |
|---|---|---|---|
| 🥇 1 | Enforcement rate limiting | Forte | Facile |
| 🥇 2 | Batch prediction `/predict-batch` | Forte | Facile |
| 🥇 3 | Performance réelle `/models/{name}/performance` | Forte | Facile |
| 🥇 4 | Seuil de confiance par modèle | Forte | Facile |
| 🥇 5 | Filtre `id_obs` sur `/predictions` | Forte | Facile |
| 🥈 6 | Suivi accuracy dans le temps | Forte | Moyenne |
| 🥈 7 | Migrations Alembic | Forte | Moyenne |
| 🥈 8 | CI/CD GitHub Actions | Forte | Moyenne |
| 🥈 9 | Explainability SHAP `/explain` | Forte | Moyenne |
| 🥈 10 | Logging structuré JSON | Forte | Moyenne |
| 🥉 11 | Upload modèle dans Streamlit | Moyenne | Facile |
| 🥉 12 | Comparaison de versions | Moyenne | Facile |
| 🥉 13 | Export CSV prédictions | Forte | Facile |
| 🥉 14 | Tags sur modèles | Moyenne | Facile |
| 🥉 15 | Détection data drift | Forte | Moyenne |
| ⏳ 16 | A/B testing canary | Forte | Complexe |
| ⏳ 17 | Alertes automatiques | Forte | Complexe |
| ⏳ 18 | Pipeline retraining auto | Forte | Complexe |
