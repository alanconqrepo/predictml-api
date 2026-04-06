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

## Liste des idées

| # | Fonctionnalité | Valeur | Difficulté | Fichiers clés | Fait | Refusé |
|---|---|---|---|---|:---:|:---:|
| 1 | **Enforcement rate limiting** — Vérifier `rate_limit_per_day` dans l'API, lever `429` si dépassé | Forte | Facile | `predict.py`, `security.py` | [ ] | [ ] |
| 2 | **Batch prediction `POST /predict-batch`** — Accepter une liste d'inputs, persister en transaction batch | Forte | Facile | `predict.py`, `schemas/` | [ ] | [ ] |
| 3 | **Performance réelle `GET /models/{name}/performance`** — Joindre `predictions` et `observed_results` pour accuracy live | Forte | Facile | `models.py`, `db_service.py` | [ ] | [ ] |
| 4 | **Seuil de confiance par modèle** — Champ `confidence_threshold`, flag `low_confidence` dans la réponse | Forte | Facile | `db/models.py`, `schemas/`, `model_service.py` | [ ] | [ ] |
| 5 | **Filtre `?id_obs=` sur `GET /predictions`** — Traçabilité complète depuis l'identifiant métier | Forte | Facile | `predict.py`, `db_service.py` | [ ] | [ ] |
| 6 | **Export CSV (dashboard)** — Bouton téléchargement sur la page Prédictions | Forte | Facile | `3_Predictions.py` | [ ] | [ ] |
| 7 | **Suivi accuracy dans le temps** — Courbe rolling 7j/30j, alerte si dégradation | Forte | Moyenne | `db_service.py`, `4_Stats.py` | [ ] | [ ] |
| 8 | **Détection data drift** — Comparaison KS/PSI distribution features prod vs baseline | Forte | Moyenne | `models.py`, `model_service.py`, `db/models.py` | [ ] | [ ] |
| 9 | **Explainability SHAP `POST /explain`** — Feature importances locales (RGPD Art. 22) | Forte | Moyenne | `predict.py`, `model_service.py` | [ ] | [ ] |
| 10 | **Migrations Alembic** — Remplacer `create_all()` par des migrations versionnées | Forte | Moyenne | `alembic/`, `main.py`, `src/db/` | [ ] | [ ] |
| 11 | **CI/CD GitHub Actions** — `ruff` + `black` + `pytest` sur chaque push/PR | Forte | Moyenne | `.github/workflows/ci.yml` | [ ] | [ ] |
| 12 | **Logging structuré JSON** — Remplacer `print()` par `structlog`/`loguru` | Forte | Moyenne | `config.py`, `src/api/`, `src/services/` | [ ] | [ ] |
| 13 | **A/B testing canary** — Split trafic entre versions, comparaison métriques auto | Forte | Complexe | `model_metadata`, `model_service.py` | [ ] | [ ] |
| 14 | **Alertes automatiques** — Webhook/email si accuracy < seuil ou drift détecté | Forte | Complexe | Background tasks, APScheduler | [ ] | [ ] |
| 15 | **Pipeline retraining automatique** — Retraining déclenché sur dégradation ou volume | Forte | Complexe | Prefect/APScheduler, MLflow, API | [ ] | [ ] |
| 16 | **Upload modèle dans Streamlit** — Formulaire `st.file_uploader()` + métadonnées | Moyenne | Facile | `2_Models.py`, `api_client.py` | [ ] | [ ] |
| 17 | **Comparaison côte-à-côte de versions** — Tableau multi-version avec highlighting | Moyenne | Facile | `2_Models.py` | [ ] | [ ] |
| 18 | **Stats agrégées `GET /predictions/stats`** — COUNT, error_rate, p50/p95 par modèle | Moyenne | Facile | `predict.py`, `db_service.py` | [ ] | [ ] |
| 19 | **Tags sur les modèles** — Champ JSONB `tags`, filtre `?tag=` sur `GET /models` | Moyenne | Facile | `db/models.py`, `schemas/`, `api/models.py` | [ ] | [ ] |
| 20 | **Validation taille à l'upload** — `413` si `.pkl` dépasse `MAX_MODEL_SIZE_MB` | Moyenne | Facile | `api/models.py`, `config.py` | [ ] | [ ] |
| 21 | **Marquage manuel prédiction** — Champ `is_correct`, `PATCH /predictions/{id}` | Moyenne | Facile | `db/models.py`, `predict.py`, `schemas/` | [ ] | [ ] |
| 22 | **Profil features baseline** — Stocker `{feature: {mean, std, min, max}}`, warning en prod | Moyenne | Moyenne | `db/models.py`, `model_service.py` | [ ] | [ ] |
| 23 | **Webhook sortant post-prédiction** — `POST` async vers URL callback par modèle | Moyenne | Moyenne | `db/models.py`, `predict.py` | [ ] | [ ] |
| 24 | **Cache Redis** — Remplacer le dict Python par Redis (multi-instance, Kubernetes) | Moyenne | Moyenne | `model_service.py`, `docker-compose.yml` | [ ] | [ ] |
| 25 | **Copie token en un clic** — Bouton "Copier" dans la page Users | Faible | Facile | `1_Users.py` | [ ] | [ ] |
| 26 | **Indication "last seen" sur les modèles** — `MAX(timestamp)` par modèle dans le listing | Faible | Facile | `db_service.py`, `api/models.py` | [ ] | [ ] |
| 27 | **Pagination cursor-based** — Remplacer `offset` par curseur sur `/predictions` | Faible | Facile | `predict.py`, `db_service.py` | [ ] | [ ] |
