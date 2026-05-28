# ROADMAP V15 — PredictML API

**Date:** 2026-05-28  
**Status:** Maintenance phase — functionally complete since V13

---

## Honest assessment

After V14, the project covers the full MLOps lifecycle end-to-end:

- Model serving (single, batch, SHAP, anomalies, export)
- Drift detection (feature Z-score + PSI, output drift / label shift)
- Performance monitoring (classification + regression metrics, temporal granularity)
- A/B testing + shadow deployment + statistical significance
- Retraining (manual, scheduled cron, drift-triggered) with auto-promotion policy
- Golden test gates, input schema validation, model calibration, confidence analysis
- Full audit trail, rollback, model cards, leaderboard
- Alerting (drift, error spikes, performance drop, auto-demotion) via email + webhooks, every 6h via ARQ worker
- Grafana dashboards (API overview + model performance) pre-provisioned
- Streamlit dashboard with 13 pages, LLM chatbot, AutoTrain assistant

**The following proposals are the only two gaps identified that have genuine value for the majority of users. If neither is worth the effort, the project needs no new features.**

---

## P1 — `strict_validation` on `/predict-batch`

**Priority:** High  
**Effort:** ~30 min  
**Type:** Consistency fix (carry-over from V14)

### Why

`POST /predict` supports `?strict_validation=true`: it rejects requests with missing or unexpected features and returns a structured 422 with the list of violations. `POST /predict-batch` does not. A batch job sending 500 rows with a renamed column (e.g., `sepal_lenght` instead of `sepal_length`) produces 500 wrong predictions with no error raised — the most dangerous class of silent failure in ML.

### What

Add `strict_validation: bool = False` query parameter to `POST /predict-batch`. When enabled, apply the same `validate_input_features()` check from `src/services/input_validation_service.py` to each row before inference. Return 422 on the first violation with the row index and error details.

### Response on violation

```json
{
  "detail": "Strict validation failed on row 2",
  "errors": [
    { "type": "unexpected_feature", "feature": "sepal_lenght" },
    { "type": "missing_feature",    "feature": "sepal_length" }
  ],
  "row_index": 2
}
```

### Implementation

- File: `src/api/predict.py` — batch endpoint handler
- Service: `src/services/input_validation_service.py` — `validate_input_features()` (already used by `/predict`)
- Tests: 2-3 cases in `tests/test_predict_post.py` or a new `test_predict_batch_validation.py`

### Explicit non-goal

Do not fail the entire batch for warnings (type coercions). Only hard errors (`missing_feature`, `unexpected_feature`) block the request when `strict_validation=true`.

---

## P2 — Active labeling sampling endpoint

**Priority:** Medium  
**Effort:** ~2–3h  
**Type:** New endpoint + Streamlit panel

### Why

`GET /observed-results/stats` reports overall labeling coverage (e.g., "38% of predictions have ground truth"). It answers *how many* predictions are unlabeled but not *which ones to label next*. A Data Scientist who has 2 hours to annotate 50 predictions has no guidance on which ones will improve the model the most.

The standard answer from active learning: label what the model is most uncertain about (lowest max confidence = most informative to label). This is a single SQL query — no ML overhead.

### What

New endpoint:

```
GET /predictions/unlabeled
  ?strategy=uncertainty   # default: uncertainty|recent|random
  &model_name=iris        # optional filter
  &model_version=1.0.0   # optional filter
  &limit=50               # default 50, max 200
```

**Strategies:**

| Strategy | Sort order | Use case |
|----------|------------|----------|
| `uncertainty` | `max_confidence ASC` | Maximize model improvement per label |
| `recent` | `timestamp DESC` | Label latest predictions first (production monitoring) |
| `random` | `RANDOM()` | Unbiased sample for performance estimation |

**Response:**

```json
{
  "total_unlabeled": 1842,
  "returned": 50,
  "strategy": "uncertainty",
  "predictions": [
    {
      "id": "uuid",
      "id_obs": "obs-123",
      "model_name": "iris",
      "model_version": "1.0.0",
      "prediction_result": "versicolor",
      "max_confidence": 0.51,
      "timestamp": "2026-05-20T14:32:00"
    }
  ]
}
```

### Implementation

- SQL: `predictions LEFT OUTER JOIN observed_results ON (predictions.id_obs = observed_results.id_obs AND predictions.model_name = observed_results.model_name) WHERE observed_results.id IS NULL`
- New `DBService.get_unlabeled_predictions()` method in `src/services/db_service.py`
- New route in `src/api/predict.py` (or `observed_results.py`)
- Schema: `UnlabeledPredictionsResponse` in `src/schemas/prediction.py`

### Streamlit addition

Small panel at the bottom of `streamlit_app/pages/3_Predictions.py` (or in the observed results section):

- Counter: **"1 842 predictions awaiting labels"**
- Strategy selector (uncertainty / recent / random)
- "Export N predictions to label" button → downloads CSV with `id_obs`, `prediction_result`, `max_confidence`, `timestamp` + an empty `observed_result` column ready to fill
- After annotation, CSV is re-importable via the existing `POST /observed-results/upload-csv` endpoint

This closes the full active learning loop without any new infrastructure: label → upload → model improves.

### Tests

- `GET /predictions/unlabeled` nominal case (uncertainty strategy)
- Filter by model_name
- strategy=recent and strategy=random variants
- Empty result when all predictions are labeled
- Auth (401 without token)

---

## What was considered and excluded

| Idea | Why excluded |
|------|-------------|
| **Gradual canary rollout automation** | A/B with `traffic_weight` + `auto_promote` policy already covers the use case for 80% of users; full automation adds a state machine with non-trivial rollback logic |
| **Third Grafana dashboard (drift)** | Two dashboards already provisioned and functional; drift status is visible in model-performance dashboard |
| **Prediction confidence-based fallback routing** | Niche use case (requires defining a fallback model per model), adds complexity to the hot path |
| **Multi-tenant workspaces / organizations** | Out of scope for the core value proposition; role-based access already covers most team setups |
| **Custom evaluation metrics** | The existing set (accuracy, F1, precision, recall, AUC, MAE, RMSE, R²) covers all standard sklearn estimators |
| **Model ensemble / stacking** | Niche, outside the scope of a serving API |

---

## Summary

| # | Feature | Effort | Value |
|---|---------|--------|-------|
| P1 | `strict_validation` on `/predict-batch` | 30 min | Prevents silent failures at scale |
| P2 | Active labeling sampling endpoint | 2–3h | Closes the annotation feedback loop |

Both are small, self-contained, and add genuine operational value. Beyond these, the project is complete.
