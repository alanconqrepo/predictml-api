# ROADMAP V14 — Audit post-V13 & propositions finales

> **Perspective** : Data Scientist / MLOps Engineer utilisant cette plateforme au quotidien.
>
> **Périmètre** : Audit exhaustif du code source après V13. Ce document liste uniquement
> ce qui est factuellement absent ou incohérent — pas ce qui pourrait théoriquement exister.

---

## Bilan V13 : conclusions confirmées

ROADMAP_V13 concluait que le projet était "terminé". Après vérification indépendante du
code source, cette conclusion est correcte pour l'essentiel :

| Fonctionnalité V13 supposée présente | Statut réel |
|---|---|
| `min_golden_test_pass_rate` dans auto-promotion | ✅ Présent — `auto_promotion_service.py:134` |
| Retrain drift-triggered + cooldown | ✅ Présent — `supervision_reporter.py:258–287` |
| Page Streamlit Golden Tests complète | ✅ Présente — `9_Golden_Tests.py` |
| 50+ endpoints couvrant le cycle MLOps complet | ✅ Confirmé |
| Toutes les intégrations Docker (PG, MinIO, Redis, MLflow, OTEL) | ✅ Confirmé |

---

## Constat : deux lacunes réelles trouvées

Après audit ligne par ligne des endpoints et de la configuration Docker, deux points
concrets manquent. Ce ne sont pas des fonctionnalités nouvelles — ce sont des
**incohérences ou des manques opérationnels** dans ce qui est déjà en place.

---

## Propositions V14 (2 uniquement)

| # | Domaine | Item | Priorité | Difficulté |
|---|---------|------|----------|------------|
| 1 | API | `strict_validation` sur `/predict-batch` | P1 | S (30 min) |
| 2 | Infra | Dashboards Grafana pré-configurés | P2 | M (3–4h) |

Légende difficulté : **S** < 2h · **M** 2–8h

---

## P1 — `strict_validation` sur `/predict-batch`

### Pourquoi

`POST /predict` accepte `?strict_validation=true` depuis V11 : si des features inattendues
sont présentes, la requête est rejetée avec 422. `POST /predict-batch` n'a pas ce paramètre.

Conséquence concrète : un pipeline envoyant des features mal nommées sera rejeté en
prédiction unitaire mais accepté silencieusement en batch, produisant des résultats
potentiellement incorrects sur des milliers de lignes.

```python
# Ligne 714 de src/api/predict.py — présent sur /predict
strict_validation: bool = Query(
    False, description="Rejette les features inattendues avec 422"
)

# Ligne 971 — absent sur /predict-batch
async def predict_batch(input_data: BatchPredictionInput, ...):
    # Aucun paramètre strict_validation
```

### Ce qui manque

Ajouter `strict_validation: bool = Query(False, ...)` à `predict_batch` et appeler
`validate_input_features()` (déjà importé, même logique que `/predict`) sur chaque item
du batch, ou une fois avec les features du premier item comme proxy.

### Fichiers à modifier

| Fichier | Changement |
|---|---|
| `src/api/predict.py` | +`strict_validation` param + validation dans la boucle batch |
| `tests/test_predict_post.py` | 1–2 tests : batch avec features inattendues + `strict_validation=true` |

### Estimation

30 min. Zéro nouvelle dépendance. Zéro migration DB.

---

## P2 — Dashboards Grafana pré-configurés

### Pourquoi

Le service Grafana (`grafana/otel-lgtm`) est déjà dans `docker-compose.yml` sur le port
3000. L'API exporte des métriques Prometheus et des traces OTEL. Pourtant le répertoire
`monitoring/` ne contient qu'un `prometheus.yml` — aucun dashboard Grafana.

Un opérateur qui démarre le stack depuis zéro se retrouve avec Grafana vide : il doit
construire manuellement les dashboards pour surveiller le volume de prédictions, les
latences, les erreurs et le drift. C'est du travail répété à chaque nouveau déploiement.

### Ce qui manque

Trois fichiers JSON dans `monitoring/grafana/dashboards/` + un fichier de provisioning :

```
monitoring/
├── prometheus.yml              # déjà là
└── grafana/
    ├── provisioning.yaml       # déclare le datasource + le dossier de dashboards
    └── dashboards/
        ├── api-overview.json   # volume, latence P95, error rate, quota utilisateurs
        └── model-performance.json  # accuracy par modèle, drift status, retrain events
```

Le `docker-compose.yml` nécessite un volume mount supplémentaire :

```yaml
volumes:
  - ./monitoring/prometheus.yml:/otel-lgtm/prometheus.yaml:ro
  - ./monitoring/grafana/provisioning.yaml:/etc/grafana/provisioning/datasources/predictml.yaml:ro
  - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
```

Avec ce provisioning, les dashboards apparaissent automatiquement au `docker-compose up`
sans aucune action manuelle sur l'interface Grafana.

### Métriques disponibles (déjà collectées)

Les métriques FastAPI Prometheus et les traces OTEL couvrent sans modification :

- `http_requests_total{endpoint, method, status}` — volume par route
- `http_request_duration_seconds` — latence (histogramme, P50/P95/P99)
- `predictions_total`, `predictions_error_total` (compteurs custom si ajoutés)
- Traces OTEL : span par prédiction, retrain, drift check

### Estimation

3–4h. Zéro modification Python. Zéro migration DB.

---

## Idées évaluées et définitivement rejetées

Ces pistes ont été examinées pendant l'audit et écartées pour les raisons indiquées.
Elles ne méritent pas d'être réévaluées sauf changement de scope majeur.

| Idée | Raison du rejet |
|---|---|
| Dashboards Streamlit "santé infra" | `/health/dependencies` répond déjà. Un panneau Streamlit duplique sans valeur ajoutée pour un MLOps qui a les logs Docker. |
| Rate limit par modèle | Les quotas par utilisateur couvrent le cas. Ajouter un quota par modèle ajoute de la configuration sans résoudre une douleur réelle observée. |
| Clonage de version de modèle | Rare en pratique. L'upload + versioning couvre le cas sans complexité. |
| Feature importance historique (SHAP par prédiction) | Coût de stockage disproportionné. Le confidence trend et le feature drift input adressent la même question à moindre coût. |
| Ensemble / fusion multi-modèles | Haute complexité, cas d'usage de niche. L'A/B et le shadow couvrent la comparaison. |
| Export ONNX | Dépendance lourde (onnx, skl2onnx), cas d'usage edge/GPU rare. |
| Async batch scoring (job_id + polling) | `/predict-batch` synchrone couvre les volumes standards. Un système job async est une complexité infrastructure disproportionnée. |
| Métriques fairness / biais | Nécessite des attributs protégés et une définition métier. Hors scope généraliste. |
| Multi-tenancy / feature store | Hors scope du projet. |
| WebSocket / SSE streaming | < 20 % des cas. Le modèle request/response couvre les besoins. |
| Auto-reset baseline après retrain | Juste après un retrain, la nouvelle version n'a pas encore de prédictions réelles. Le calcul de baseline serait vide. |

---

## Conclusion

Le projet est fonctionnellement complet depuis V13. Les deux items ci-dessus sont des
**finitions** : l'un corrige une incohérence de comportement entre deux endpoints similaires,
l'autre rend opérationnel un service déjà déployé mais inutilisable sans configuration manuelle.

Au-delà de ces deux points, le projet entre en **phase de maintenance** :
- Mises à jour de dépendances (FastAPI, SQLAlchemy, APScheduler, MLflow…)
- Mises à jour de sécurité
- Évolutions guidées par des besoins opérationnels réels, pas par des fonctionnalités anticipées

> Une fonctionnalité n'est pertinente que si elle résout une douleur concrète sur votre
> déploiement. Si les deux propositions ci-dessus ne correspondent pas à une douleur réelle,
> il n'y a rien à faire.
