# Translation Progress — French → English

## Method

- **One logical batch per session** — small, focused commits.
- Only translate: docstrings, inline comments, markdown text, UI labels.
- Do NOT change: variable names, function names, log keys, API field names, test assertions.
- After each batch: `ruff check` + `black --check` + baseline test run.
- Commit format: `i18n: translate <group> comments/docstrings to English`
- Branch: `claude/internationalize-to-english-Jnqxr`

---

## Batch 1 — Bootstrap + `src/core/` + `src/main.py`

- [x] Create `TRANSLATION_PROGRESS.md`
- [x] `src/main.py`
- [x] `src/core/config.py`
- [x] `src/core/security.py`
- [x] `src/core/logging.py`
- [x] `src/core/arq_pool.py`
- [x] `src/core/ml_metrics.py`
- [x] `src/core/platform_limits.py`
- [x] `src/core/telemetry.py`
- [x] `src/core/utils.py`
- [x] `src/core/__init__.py`

---

## Batch 2 — `src/api/`

- [x] `src/api/models.py` (heaviest)
- [x] `src/api/predict.py`
- [x] `src/api/monitoring.py`
- [x] `src/api/users.py`
- [x] `src/api/observed_results.py`
- [x] `src/api/jobs.py`
- [x] `src/api/account_requests.py`

---

## Batch 3 — `src/services/`

- [x] `src/services/db_service.py`
- [x] `src/services/retrain_service.py`
- [x] `src/services/model_service.py`
- [x] `src/services/auto_promotion_service.py`
- [x] `src/services/email_service.py`
- [x] `src/services/minio_service.py`
- [x] `src/services/mlflow_service.py`
- [x] `src/services/drift_service.py`
- [x] `src/services/metrics_service.py`
- [x] `src/services/shap_service.py`
- [x] `src/services/golden_test_service.py`
- [x] `src/services/ab_significance_service.py`
- [x] `src/services/input_validation_service.py`
- [x] `src/services/__init__.py`

---

## Batch 4 — `src/schemas/` + `src/tasks/` + `src/db/` + `src/workers/`

- [x] `src/schemas/prediction.py`
- [x] `src/schemas/model.py`
- [x] `src/schemas/observed_result.py`
- [x] `src/schemas/user.py`
- [x] `src/schemas/golden_test.py`
- [x] `src/schemas/task_run.py`
- [x] `src/schemas/monitoring.py`
- [x] `src/schemas/__init__.py`
- [x] `src/tasks/retrain_scheduler.py`
- [x] `src/tasks/arq_worker.py`
- [x] `src/tasks/supervision_reporter.py`
- [x] `src/db/database.py`
- [x] `src/workers/prediction_writer.py`

---

## Batch 5 — `streamlit_app/utils/`

- [x] `streamlit_app/utils/autotraining_tools.py` (heaviest)
- [x] `streamlit_app/utils/metrics_help.py`
- [x] `streamlit_app/utils/docs_loader.py`
- [x] `streamlit_app/utils/tools.py`
- [x] `streamlit_app/utils/api_client.py`
- [x] `streamlit_app/utils/auth.py`
- [x] `streamlit_app/utils/i18n.py`
- [x] `streamlit_app/utils/ui_helpers.py`

---

## Batch 6 — `streamlit_app/` pages + `app.py`

- [x] `streamlit_app/app.py`
- [x] `streamlit_app/pages/10_Aide.py`
- [x] `streamlit_app/pages/12_AutoTrain_Chatbot.py`
- [x] `streamlit_app/pages/2_Models.py`
- [x] `streamlit_app/pages/6_AB_Testing.py`
- [x] `streamlit_app/pages/4_Stats.py`
- [x] `streamlit_app/pages/7_Supervision.py`
- [x] `streamlit_app/pages/5_Code_Example.py`
- [x] `streamlit_app/pages/1_Users.py`
- [x] `streamlit_app/pages/3_Predictions.py`
- [x] `streamlit_app/pages/11_Services.py`
- [x] `streamlit_app/pages/9_Golden_Tests.py`
- [x] `streamlit_app/pages/8_Retrain.py`
- [x] `streamlit_app/pages/0_Demande_Acces.py`

---

## Batch 7 — `tests/` part A

- [x] `tests/conftest.py`
- [x] `tests/test_api.py`
- [x] `tests/test_predict_post.py`
- [x] `tests/test_predictions_get.py`
- [x] `tests/test_predictions_purge.py`
- [x] `tests/test_export_endpoint.py`
- [x] `tests/test_prediction_stats.py`
- [x] `tests/test_models_create.py`
- [x] `tests/test_models_get.py`
- [x] `tests/test_models_update.py`
- [x] `tests/test_models_delete.py`

---

## Batch 8 — `tests/` part B

- [x] `tests/test_retrain.py`
- [x] `tests/test_scheduled_retraining.py`
- [x] `tests/test_auto_promotion_policy.py`
- [x] `tests/test_ab_shadow.py`
- [x] `tests/test_ab_significance.py`
- [x] `tests/test_feature_importance.py`
- [x] `tests/test_observed_results.py`
- [x] `tests/test_users.py`
- [x] `tests/test_security.py`
- [x] `tests/test_rate_limit.py`
- [x] `tests/test_drift.py`

---

## Batch 9 — `tests/` part C (remaining)

- [ ] `tests/test_db_service_crud.py`
- [ ] `tests/test_monitoring_api.py`
- [ ] `tests/test_input_validation.py`
- [ ] All remaining test files (integration/, e2e/, unit/ subdirs if present)

---

## Batch 10 — Root markdown files

- [ ] `README.md`
- [ ] `CODING_STANDARDS.md`
- [ ] `CLAUDE.md`
- [ ] `CODE_QUALITY_PLAN.md`
- [ ] `CONTRIBUTING.md`
- [ ] `PROMPTS.md`
- [ ] `SKILLS.md`

---

## Batch 11 — `documentation/` markdown

- [ ] `documentation/API_REFERENCE.md` (heaviest)
- [ ] `documentation/KPIS_REFERENCE.md`
- [ ] `documentation/ARCHITECTURE.md`
- [ ] `documentation/TRAIN_SCRIPT_GUIDE.md`
- [ ] `documentation/BEGINNER_GUIDE.md`
- [ ] `documentation/DASHBOARD_GUIDE.md`
- [ ] `documentation/DATABASE.md`
- [ ] `documentation/DOCKER.md`
- [ ] `documentation/FAQ.md`
- [ ] `documentation/QUICKSTART.md`
- [ ] `documentation/GRAFANA_OTEL.md`
- [ ] `documentation/SECURITY_MODEL_VALIDATION.md`

---

## Batch 12 — `road_maps/` markdown (18 files)

- [ ] `road_maps/ROADMAP_V8.md`
- [ ] `road_maps/ROADMAP_V9.md`
- [ ] `road_maps/ROADMAP_V10.md`
- [ ] `road_maps/ROADMAP_V11.md`
- [ ] `road_maps/ROADMAP_V12.md`
- [ ] `road_maps/ROADMAP_V13.md`
- [ ] `road_maps/ROADMAP_V14.md`
- [ ] `road_maps/ROADMAP_V7.md`
- [ ] `road_maps/ROADMAP_V6.md`
- [ ] `road_maps/ROADMAP_V5.md`
- [ ] `road_maps/ROADMAP_V4.md`
- [ ] `road_maps/ROADMAP_V3.md`
- [ ] `road_maps/ROADMAP_SECURITY.md`
- [ ] `road_maps/ROADMAP_Scalability.md`
- [ ] `road_maps/roadmap_scalabilty_n2.md`
- [ ] `road_maps/ROADMAP.md`
- [ ] `road_maps/ROADMAP_4_PERPLEXITY.md`
- [ ] `road_maps/ideas.md`

---

## Batch 13 — `init_data/` + `smoke-tests/` + `alembic/` + `documentation/Scripts/`

- [ ] `init_data/example_train.py`
- [ ] `init_data/seed_sample_data.py`
- [ ] `init_data/resign_models.py`
- [ ] `init_data/create_multiple_advanced_models.py`
- [ ] `init_data/init_db.py`
- [ ] `init_data/create_multiple_models.py`
- [ ] `init_data/migrate_add_feature_baseline.py`
- [ ] `init_data/README.md`
- [ ] `smoke-tests/test_multimodel_api.py`
- [ ] `smoke-tests/README.md`
- [ ] `alembic/versions/*.py` (3 files)
- [ ] `documentation/Scripts/cancer/` (11 files)
- [ ] `documentation/Scripts/iris/` (11 files)
- [ ] `documentation/Scripts/titanic/` (5 files)
- [ ] `documentation/Scripts/wine/` (12 files)
- [ ] `Models/README.md`
