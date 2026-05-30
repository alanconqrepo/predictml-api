"""
Model management endpoints
"""

import ast
import asyncio
import io
import json
import math
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

import numpy as np
import structlog
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

import src.core.arq_pool as arq_pool_module
from src.core.audit import audit_log
from src.core.config import settings
from src.core.platform_limits import SUBPROCESS_PREEXEC_KWARGS
from src.core.rate_limit import limiter
from src.core.security import require_admin, verify_token
from src.db.database import AsyncSessionLocal, get_db, get_read_db
from src.db.models import HistoryActionType, ModelMetadata, User
from src.db.models.model_metadata import DeploymentMode
from src.db.models.task_run import TaskRun
from src.schemas.golden_test import GoldenTestCreate, GoldenTestResponse, GoldenTestRunResponse
from src.schemas.model import (
    ABCompareResponse,
    ABSignificance,
    ABVersionStats,
    CalibrationResponse,
    ComputeBaselineResponse,
    ConfidenceBin,
    ConfidenceDistributionResponse,
    ConfidenceTrendOverall,
    ConfidenceTrendPoint,
    ConfidenceTrendResponse,
    DeprecateModelResponse,
    CategoricalDriftResult,
    DriftReportResponse,
    FeatureDriftResult,
    FeatureImportanceItem,
    FeatureImportanceResponse,
    FeatureStats,
    LeaderboardEntry,
    ModelCardCalibrationSummary,
    ModelCardCoverage,
    ModelCardDriftSummary,
    ModelCardFeatureImportanceSummary,
    ModelCardPerformanceSummary,
    ModelCardResponse,
    ModelCardRetrainInfo,
    ModelCardTopFeature,
    ModelCompareResponse,
    ModelCreateResponse,
    ModelDeleteResponse,
    ModelGetResponse,
    ModelHistoryEntry,
    ModelHistoryResponse,
    ModelPerformanceResponse,
    ModelUpdateInput,
    ModelVersionSummary,
    OutputDriftResponse,
    PerClassMetrics,
    PerformanceReportResponse,
    PerformanceTimelineResponse,
    PeriodPerformance,
    PolicyUpdateResponse,
    PromotionPolicy,
    ReadinessCheck,
    ReadinessChecks,
    ReadinessResponse,
    ReliabilityBin,
    RetrainHistoryEntry,
    RetrainHistoryResponse,
    RetrainRequest,
    RetrainScheduleInput,
    RollbackResponse,
    ScheduleUpdateResponse,
    ShadowCompareResponse,
    ValidateInputResponse,
    VersionTimelineEntry,
    WarmupResponse,
)
from src.schemas.task_run import TaskRunEnqueued
from src.services import drift_service
from src.services.ab_significance_service import compute_ab_significance
from src.services.db_service import _ROLLBACK_FIELDS, DBService, _build_snapshot
from src.services.golden_test_service import GoldenTestService
from src.services.input_validation_service import resolve_expected_features, validate_input_features
from src.services.metrics_service import compute_auc, compute_roc_curve
from src.services.minio_service import minio_service
from src.services.mlflow_service import mlflow_service
from src.services.model_service import compute_model_hmac, model_service
from src.services.shap_service import compute_shap_explanation

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["models"])

_leaderboard_cache: dict = {}
_LEADERBOARD_TTL = 300  # 5 minutes

# Validation regex — prevent path traversal in MinIO object keys (e.g. "../admin")
NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
VERSION_RE = re.compile(r"^\d+\.\d+(\.\d+)?$")


def validate_model_name(name: str) -> str:
    if not NAME_RE.match(name):
        raise HTTPException(
            status_code=422,
            detail="Invalid model name (allowed characters: a-z A-Z 0-9 _ -)",
        )
    return name


def validate_version(version: str) -> str:
    if not VERSION_RE.match(version):
        raise HTTPException(
            status_code=422,
            detail="Invalid version (expected format: X.Y or X.Y.Z)",
        )
    return version


# Annotated path-parameter types — FastAPI enforces the pattern before the handler runs
ModelNamePath = Annotated[
    str,
    Path(pattern=r"^[a-zA-Z0-9_-]{1,64}$", description="Model name"),
]
ModelVersionPath = Annotated[
    str,
    Path(pattern=r"^\d+\.\d+(\.\d+)?$", description="Model version (X.Y or X.Y.Z)"),
]


async def _get_leaderboard_drift_status(
    db: AsyncSession,
    name: str,
    version: str,
    period_days: int,
    feature_baseline: Optional[dict],
    categorical_baseline: Optional[dict] = None,
) -> str:
    has_numeric = bool(feature_baseline)
    has_categorical = bool(categorical_baseline)
    if not has_numeric and not has_categorical:
        return "no_baseline"
    prod_stats, cat_prod_stats = await asyncio.gather(
        DBService.get_feature_production_stats(db, name, version, period_days),
        DBService.get_categorical_production_stats(db, name, version, period_days),
    )
    features = drift_service.compute_feature_drift(feature_baseline or {}, prod_stats, min_count=10)
    cat_features = drift_service.compute_categorical_drift(
        categorical_baseline or {}, cat_prod_stats, min_count=10
    )
    return drift_service.summarize_drift(
        features, baseline_available=has_numeric or has_categorical, categorical_features=cat_features
    )


_ALLOWED_IMPORT_MODULES = {
    "os",
    "sys",
    "json",
    "pickle",
    "joblib",
    "pandas",
    "numpy",
    "sklearn",
    "mlflow",
    "datetime",
    "pathlib",
    "math",
    "statistics",
    "collections",
    "typing",
    "warnings",
    "logging",
    "time",
    "copy",
    "functools",
    "itertools",
    "re",
    "io",
    "abc",
    "enum",
    "dataclasses",
    "csv",
    "dotenv",
    "boto3",
    "botocore",
    "importlib",
}


def _parse_json_field(value: str | None, field_name: str) -> dict | list | None:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"JSON invalide dans '{field_name}': {exc.msg}")


def _validate_train_script(source: str) -> Optional[str]:
    """
    Validates a train.py script against the required constraints.

    The script must:
    1. Be syntactically valid Python
    2. Only import modules from the allowed list
    3. Reference TRAIN_START_DATE (env variable)
    4. Reference TRAIN_END_DATE (env variable)
    5. Reference OUTPUT_MODEL_PATH (env variable)
    6. Contain a save call: joblib.dump or save_model

    Returns:
        None if the script is valid, otherwise an error message.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"Invalid Python syntax: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _ALLOWED_IMPORT_MODULES:
                    return (
                        f"Unauthorized import: '{alias.name}'. "
                        f"Allowed modules: {sorted(_ALLOWED_IMPORT_MODULES)}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in _ALLOWED_IMPORT_MODULES:
                    return (
                        f"Unauthorized import: '{node.module}'. "
                        f"Allowed modules: {sorted(_ALLOWED_IMPORT_MODULES)}"
                    )

    required_tokens = {
        "TRAIN_START_DATE": "must reference the TRAIN_START_DATE env variable",
        "TRAIN_END_DATE": "must reference the TRAIN_END_DATE env variable",
        "OUTPUT_MODEL_PATH": "must reference the OUTPUT_MODEL_PATH env variable",
    }
    for token, msg in required_tokens.items():
        if token not in source:
            return f"Le script {msg}"

    save_calls = ["joblib.dump", "save_model"]
    if not any(call in source for call in save_calls):
        return "The script must save the model with joblib.dump or save_model"

    return None


def _detect_task_type(model_bytes: bytes) -> Optional[str]:
    """
    Detects the task type from the sklearn model bytes (.joblib).
    Returns 'regression', 'classification_binary', 'classification_multiclass', or None.
    """
    try:
        import joblib

        obj = joblib.load(io.BytesIO(model_bytes))
        # Unwrap Pipeline: take the last estimator
        if hasattr(obj, "steps"):
            obj = obj.steps[-1][1]
        if hasattr(obj, "classes_"):
            n = len(obj.classes_)
            return "classification_binary" if n == 2 else "classification_multiclass"
        # Regression: no classes_, but predicts continuous values
        if hasattr(obj, "predict"):
            return "regression"
    except Exception:
        pass
    return None


def _extract_feature_importances(
    model_bytes: bytes,
    feature_names: Optional[list] = None,
) -> Optional[dict]:
    """
    Extract feature importances from a sklearn model (.joblib bytes).

    Supports:
    - tree-based models: feature_importances_ (RandomForest, GBT, XGBoost, etc.)
    - linear models: coef_ (LogisticRegression, LinearSVC, Ridge, etc.)
    - Pipelines: unwraps to the last estimator, uses feature names from the
      last transformer if available (ColumnTransformer, etc.)

    Returns {feature_name: score} sorted descending, or None if not supported.
    """
    try:
        import io as _io

        import joblib
        import numpy as np

        obj = joblib.load(_io.BytesIO(model_bytes))

        # Resolve feature names from Pipeline transformers if available
        pipeline_feature_names: Optional[list] = None
        if hasattr(obj, "steps"):
            # Try to get feature names from the last transformer before the estimator
            try:
                if hasattr(obj[:-1], "get_feature_names_out"):
                    pipeline_feature_names = list(obj[:-1].get_feature_names_out())
            except Exception:
                pass
            obj = obj.steps[-1][1]

        names = feature_names or pipeline_feature_names

        importances: Optional[np.ndarray] = None
        if hasattr(obj, "feature_importances_"):
            importances = np.array(obj.feature_importances_)
        elif hasattr(obj, "coef_"):
            coef = np.array(obj.coef_)
            # Multi-class: average abs coef across classes
            if coef.ndim > 1:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
        else:
            return None

        # Build feature name list
        if not names and hasattr(obj, "feature_names_in_"):
            names = list(obj.feature_names_in_)
        if not names:
            names = [f"feature_{i}" for i in range(len(importances))]

        if len(names) != len(importances):
            names = [f"feature_{i}" for i in range(len(importances))]

        total = importances.sum()
        if total > 0:
            importances = importances / total

        result = {name: round(float(score), 6) for name, score in zip(names, importances)}
        return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))

    except Exception:
        return None


_SAFE_UPLOAD_ENV_KEYS = {
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TEMP",
    "TMP",
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
    "VIRTUAL_ENV",
}


async def _run_train_subprocess(
    train_source: str,
) -> tuple[Optional[str], Optional[bytes]]:
    """
    Runs train.py in a subprocess with default dates (D-30 → today).

    Returns ``(req_txt, model_bytes)``:
    - ``req_txt``     : contents of requirements.txt (from "dependencies" in stdout JSON),
                        or None if the field is absent (caller falls back to AST).
    - ``model_bytes`` : contents of the .joblib produced by the script, or None if the subprocess
                        failed or produced no file (caller uses the uploaded file).

    Timeout: 120 s.
    """
    import asyncio
    import json as _json
    from datetime import datetime, timedelta

    from src.services.env_snapshot_service import dependencies_to_requirements_txt

    now = datetime.now()
    start_date = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "train.py")
        output_path = os.path.join(tmpdir, "output_model.joblib")

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(train_source)

        env = {k: v for k, v in os.environ.items() if k in _SAFE_UPLOAD_ENV_KEYS}
        env.update(
            {
                "TRAIN_START_DATE": start_date,
                "TRAIN_END_DATE": end_date,
                "OUTPUT_MODEL_PATH": output_path,
            }
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                script_path,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmpdir,
                **SUBPROCESS_PREEXEC_KWARGS,
            )
            try:
                raw_stdout, raw_stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                logger.warning("_run_train_subprocess : timeout 120 s")
                return None, None

            stdout_text = raw_stdout.decode("utf-8", errors="replace")

            # Read the produced .joblib (if present)
            model_bytes: Optional[bytes] = None
            if proc.returncode == 0 and os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    model_bytes = f.read()
            elif proc.returncode != 0:
                logger.warning(
                    "_run_train_subprocess: subprocess failed",
                    returncode=proc.returncode,
                    stderr=raw_stderr.decode("utf-8", errors="replace")[:300],
                )

            # Extract "dependencies" from the last JSON line of stdout
            req_txt: Optional[str] = None
            for line in reversed(stdout_text.strip().splitlines()):
                stripped = line.strip()
                if stripped.startswith("{"):
                    try:
                        data = _json.loads(stripped)
                        deps = data.get("dependencies")
                        if deps and isinstance(deps, dict):
                            req_txt = dependencies_to_requirements_txt(deps)
                    except _json.JSONDecodeError:
                        pass
                    break

            return req_txt, model_bytes

        except Exception as exc:
            logger.warning("_run_train_subprocess : exception", error=str(exc))
            return None, None


@router.get("/models", response_model=List[Dict[str, Any]])
async def list_models(
    tag: Optional[str] = Query(None, description="Filter by tag (e.g. production, finance)"),
    is_production: Optional[bool] = Query(None, description="Filter by is_production"),
    algorithm: Optional[str] = Query(
        None, description="Exact filter on algorithm (e.g. RandomForest)"
    ),
    min_accuracy: Optional[float] = Query(None, description="Filter accuracy >= value (e.g. 0.85)"),
    deployment_mode: Optional[Literal["production", "ab_test", "shadow"]] = Query(
        None, description="Deployment mode: production | ab_test | shadow"
    ),
    search: Optional[str] = Query(
        None, description="Text search on name and description (case-insensitive)"
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Lists all available models from the database.

    Optional filters (combinable with AND):
    - **tag**: filter by tag
    - **is_production**: filter by production status
    - **algorithm**: exact filter on algorithm
    - **min_accuracy**: minimum accuracy (>= value)
    - **deployment_mode**: deployment mode (`production`, `ab_test`, `shadow`)
    - **search**: text search on name and description (ILIKE)

    Returns:
        List of active models with their metadata
    """
    models = await model_service.get_available_models(
        db,
        is_production=is_production,
        algorithm=algorithm,
        min_accuracy=min_accuracy,
        deployment_mode=deployment_mode,
        search=search,
    )
    if tag:
        models = [m for m in models if m.get("tags") and tag in m["tags"]]
    return models


@router.get("/models/cached")
async def list_cached_models():
    """
    Lists models currently in memory cache.

    Returns:
        List of MinIO object keys in cache
    """
    cached = await model_service.get_cached_models()
    return {"cached_models": cached, "count": len(cached)}


@router.get("/models/leaderboard", response_model=List[LeaderboardEntry])
async def get_models_leaderboard(
    metric: Literal[
        "accuracy", "auc", "f1_score", "r2", "rmse", "latency_p95_ms", "predictions_count"
    ] = Query("accuracy", description="Metric to rank by"),
    days: int = Query(30, ge=1, le=365, description="Sliding window in days"),
    db: AsyncSession = Depends(get_read_db),
    _user: User = Depends(verify_token),
) -> List[LeaderboardEntry]:
    """
    Global ranking of production models by performance metric.

    Sorts all active production models by `metric` over the last `days` days.
    Result is cached for 5 minutes (TTL) to avoid recomputing drift on each call.
    """
    cache_key = f"{metric}:{days}"
    cached = _leaderboard_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < _LEADERBOARD_TTL:
        return cached["data"]

    models = await DBService.get_all_active_models(db, is_production=True)
    pred_stats_list = await DBService.get_prediction_stats(db, days=days)
    stats_by_name = {s["model_name"]: s for s in pred_stats_list}

    entries: List[LeaderboardEntry] = []
    for m in models:
        ps = stats_by_name.get(m.name, {})
        drift_status = await _get_leaderboard_drift_status(
            db, m.name, m.version, days, m.feature_baseline, m.categorical_baseline
        )
        tm = m.training_metrics or {}
        entries.append(
            LeaderboardEntry(
                rank=0,
                name=m.name,
                version=m.version,
                accuracy=m.accuracy,
                auc=m.auc,
                f1_score=m.f1_score,
                r2=tm.get("r2"),
                rmse=tm.get("rmse"),
                latency_p95_ms=ps.get("p95_response_time_ms"),
                drift_status=drift_status,
                predictions_count=ps.get("total_predictions", 0),
                deployment_mode=m.deployment_mode,
                is_production=bool(m.is_production),
            )
        )

    if metric == "latency_p95_ms":
        entries.sort(
            key=lambda e: e.latency_p95_ms if e.latency_p95_ms is not None else float("inf")
        )
    elif metric == "predictions_count":
        entries.sort(key=lambda e: e.predictions_count, reverse=True)
    elif metric == "auc":
        entries.sort(key=lambda e: e.auc if e.auc is not None else -1, reverse=True)
    elif metric == "f1_score":
        entries.sort(key=lambda e: e.f1_score if e.f1_score is not None else -1, reverse=True)
    elif metric == "r2":
        entries.sort(key=lambda e: e.r2 if e.r2 is not None else -float("inf"), reverse=True)
    elif metric == "rmse":
        # RMSE: lower is better → ascending order, None last
        entries.sort(key=lambda e: e.rmse if e.rmse is not None else float("inf"))
    else:  # accuracy
        entries.sort(key=lambda e: e.accuracy if e.accuracy is not None else -1, reverse=True)

    for i, entry in enumerate(entries, start=1):
        entry.rank = i

    _leaderboard_cache[cache_key] = {"ts": time.time(), "data": entries}
    return entries


# ---------------------------------------------------------------------------
# Helpers pour GET /models/{name}/performance
# ---------------------------------------------------------------------------


def _detect_model_type(metadata: Optional[ModelMetadata], pairs: list) -> str:
    """Detects whether the model is classification or regression."""
    if metadata and metadata.classes:
        return "classification"
    if any(row.probabilities for row in pairs):
        return "classification"
    pred_vals = [row.prediction_result for row in pairs if row.prediction_result is not None]
    if pred_vals and all(isinstance(v, (int, str, bool)) for v in pred_vals):
        return "classification"
    return "regression"


def _compute_classification_metrics(y_true: list, y_pred: list, classes: Optional[list]) -> tuple:
    y_true_s = [str(v) for v in y_true]
    y_pred_s = [str(v) for v in y_pred]
    data_labels = sorted(set(y_true_s + y_pred_s))
    if classes:
        candidate = sorted(set(str(c) for c in classes))
        labels = candidate if any(lbl in y_true_s for lbl in candidate) else data_labels
    else:
        labels = data_labels

    acc = accuracy_score(y_true_s, y_pred_s)
    prec = precision_score(y_true_s, y_pred_s, average="weighted", zero_division=0, labels=labels)
    rec = recall_score(y_true_s, y_pred_s, average="weighted", zero_division=0, labels=labels)
    f1 = f1_score(y_true_s, y_pred_s, average="weighted", zero_division=0, labels=labels)
    cm = confusion_matrix(y_true_s, y_pred_s, labels=labels).tolist()

    prec_per = precision_score(y_true_s, y_pred_s, average=None, zero_division=0, labels=labels)
    rec_per = recall_score(y_true_s, y_pred_s, average=None, zero_division=0, labels=labels)
    f1_per = f1_score(y_true_s, y_pred_s, average=None, zero_division=0, labels=labels)
    support_per = [y_true_s.count(lbl) for lbl in labels]

    per_class = {
        lbl: PerClassMetrics(
            precision=float(prec_per[i]),
            recall=float(rec_per[i]),
            f1_score=float(f1_per[i]),
            support=support_per[i],
        )
        for i, lbl in enumerate(labels)
    }
    return acc, prec, rec, f1, cm, labels, per_class


def _compute_regression_metrics(y_true: list, y_pred: list) -> tuple:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2


def _bucket_key(ts: datetime, granularity: str) -> str:
    if granularity == "day":
        return ts.strftime("%Y-%m-%d")
    if granularity == "week":
        return ts.strftime("%Y-W%W")
    if granularity == "month":
        return ts.strftime("%Y-%m")
    return ""


@router.get("/models/{name}/performance", response_model=ModelPerformanceResponse)
async def get_model_performance(
    name: ModelNamePath,
    start: Optional[datetime] = Query(None, description="Start of period (ISO 8601)"),
    end: Optional[datetime] = Query(None, description="End of period (ISO 8601)"),
    version: Optional[str] = Query(
        None, description="Model version (optional)", pattern=r"^\d+\.\d+(\.\d+)?$"
    ),
    granularity: Optional[Literal["day", "week", "month"]] = Query(
        None, description="Temporal aggregation (day, week, month)"
    ),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Computes real production performance metrics for a model.

    Joins predictions and observed results via `id_obs` to compute:
    - **Classification**: accuracy, precision/recall/f1 weighted, confusion matrix, per-class metrics
    - **Regression**: MAE, MSE, RMSE, R²

    Optional parameters:
    - **start** / **end**: time range
    - **version**: specific model version
    - **granularity**: break down metrics by day, week, or month

    Requires a valid Bearer token.
    """
    metadata = await DBService.get_model_metadata(db, name, version)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' not found.",
        )

    total = await DBService.count_predictions(db, name, start, end, version)
    _raw_pairs = await DBService.get_performance_pairs(db, name, start, end, version)
    pairs = [
        p for p in _raw_pairs if p.prediction_result is not None and p.observed_result is not None
    ]

    matched = len(pairs)
    model_type = _detect_model_type(metadata, pairs)

    if matched == 0:
        return ModelPerformanceResponse(
            model_name=name,
            model_version=metadata.version,
            period_start=start,
            period_end=end,
            total_predictions=total,
            matched_predictions=0,
            model_type=model_type,
        )

    y_true = [row.observed_result for row in pairs]
    y_pred = [row.prediction_result for row in pairs]
    y_prob = [row.probabilities for row in pairs]

    response = ModelPerformanceResponse(
        model_name=name,
        model_version=metadata.version,
        period_start=start,
        period_end=end,
        total_predictions=total,
        matched_predictions=matched,
        model_type=model_type,
    )

    if model_type == "classification":
        acc, prec, rec, f1, cm, classes, per_class = _compute_classification_metrics(
            y_true, y_pred, metadata.classes
        )
        response.accuracy = round(acc, 4)
        response.precision_weighted = round(float(prec), 4)
        response.recall_weighted = round(float(rec), 4)
        response.f1_weighted = round(float(f1), 4)
        response.confusion_matrix = cm
        response.classes = classes
        response.per_class_metrics = per_class
        # AUC and ROC curve (require probabilities)
        response.auc = compute_auc(y_true, y_prob, metadata.classes)
        fpr, tpr = compute_roc_curve(y_true, y_prob)
        response.roc_curve_fpr = fpr
        response.roc_curve_tpr = tpr
    else:
        y_true_f = [float(v) for v in y_true]
        y_pred_f = [float(v) for v in y_pred]
        mae, mse, rmse, r2 = _compute_regression_metrics(y_true_f, y_pred_f)
        response.mae = round(mae, 4)
        response.mse = round(mse, 4)
        response.rmse = round(rmse, 4)
        response.r2 = round(r2, 4)

    if granularity:
        buckets: Dict[str, List[int]] = {}
        for i, row in enumerate(pairs):
            key = _bucket_key(row.timestamp, granularity)
            buckets.setdefault(key, []).append(i)

        by_period = []
        for period_key in sorted(buckets):
            idxs = buckets[period_key]
            bt = [y_true[i] for i in idxs]
            bp = [y_pred[i] for i in idxs]
            bprob = [y_prob[i] for i in idxs]
            pp = PeriodPerformance(period=period_key, matched_count=len(idxs))
            if model_type == "classification":
                bt_s = [str(v) for v in bt]
                bp_s = [str(v) for v in bp]
                pp.accuracy = round(accuracy_score(bt_s, bp_s), 4)
                pp.f1_weighted = round(
                    float(f1_score(bt_s, bp_s, average="weighted", zero_division=0)), 4
                )
                pp.auc = compute_auc(bt, bprob, metadata.classes)
            else:
                bt_f = [float(v) for v in bt]
                bp_f = [float(v) for v in bp]
                pp.mae = round(mean_absolute_error(bt_f, bp_f), 4)
                pp.rmse = round(math.sqrt(mean_squared_error(bt_f, bp_f)), 4)
            by_period.append(pp)
        response.by_period = by_period

    return response


@router.get("/models/{name}/performance-timeline", response_model=PerformanceTimelineResponse)
async def get_model_performance_timeline(
    name: ModelNamePath,
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Returns the chronological evolution of performance metrics per model version.

    For each active version (sorted by deployment date), computes:
    - **Classification**: accuracy + F1 weighted
    - **Regression**: MAE

    Metrics are `null` if no observed_result is available for the version.
    `sample_count` indicates the number of (prediction, observed_result) pairs used.

    Requires a valid Bearer token.
    """
    metadata = await DBService.get_model_metadata(db, name)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' not found.",
        )

    timeline_data = await DBService.get_performance_timeline(db, name)

    return PerformanceTimelineResponse(
        model_name=name,
        timeline=[VersionTimelineEntry(**entry) for entry in timeline_data],
    )


@router.get("/models/{name}/drift", response_model=DriftReportResponse)
async def get_model_drift(
    name: ModelNamePath,
    version: Optional[str] = Query(
        None,
        description="Model version (default: production/latest)",
        pattern=r"^\d+\.\d+(\.\d+)?$",
    ),
    days: int = Query(7, ge=1, le=90, description="Time window in days"),
    min_predictions: int = Query(
        30, ge=5, description="Minimum number of predictions to compute drift"
    ),
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Data drift report for a model.

    Compares the distribution of features received in production (over `days` days)
    against the baseline profile stored at model upload time.

    - **Z-score**: `|prod_mean - baseline_mean| / baseline_std`
      - ok < 2 | warning 2–3 | critical ≥ 3
    - **PSI**: distribution divergence via normal bins
      - ok < 0.1 | warning 0.1–0.2 | critical ≥ 0.2

    Returns `drift_summary = "no_baseline"` if the baseline profile was not recorded.
    """
    metadata = await DBService.get_model_metadata(db, name, version)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' not found.",
        )

    production_stats, cat_prod_stats = await asyncio.gather(
        DBService.get_feature_production_stats(db, name, metadata.version, days),
        DBService.get_categorical_production_stats(db, name, metadata.version, days),
    )

    total_predictions = sum(v.get("count", 0) for v in production_stats.values())
    if not total_predictions:
        total_predictions = max(
            (v.get("_count", 0) for v in cat_prod_stats.values()), default=0
        )

    baseline = metadata.feature_baseline or {}
    categorical_baseline = metadata.categorical_baseline or {}
    baseline_available = bool(baseline) or bool(categorical_baseline)

    if not baseline_available:
        return DriftReportResponse(
            model_name=name,
            model_version=metadata.version,
            period_days=days,
            predictions_analyzed=total_predictions,
            baseline_available=False,
            drift_summary="no_baseline",
            features={
                feat: FeatureDriftResult(
                    production_mean=round(stats["mean"], 6),
                    production_std=round(stats["std"], 6),
                    production_count=stats["count"],
                    null_rate_production=(
                        round(stats["null_rate"], 6) if "null_rate" in stats else None
                    ),
                    drift_status="no_baseline",
                )
                for feat, stats in production_stats.items()
            },
            categorical_features={
                feat: CategoricalDriftResult(
                    baseline_distribution={},
                    production_distribution={k: v for k, v in stats.items() if k != "_count"},
                    production_count=stats.get("_count", 0),
                    drift_status="no_baseline",
                )
                for feat, stats in cat_prod_stats.items()
            },
        )

    features = drift_service.compute_feature_drift(baseline, production_stats, min_predictions)
    categorical_features = drift_service.compute_categorical_drift(
        categorical_baseline, cat_prod_stats, min_predictions
    )
    summary = drift_service.summarize_drift(
        features, baseline_available=True, categorical_features=categorical_features
    )

    return DriftReportResponse(
        model_name=name,
        model_version=metadata.version,
        period_days=days,
        predictions_analyzed=total_predictions,
        baseline_available=True,
        drift_summary=summary,
        features=features,
        categorical_features=categorical_features,
    )


@router.get("/models/{name}/output-drift", response_model=OutputDriftResponse)
async def get_model_output_drift(
    name: ModelNamePath,
    period_days: int = Query(7, ge=1, le=90, description="Time window in days"),
    model_version: Optional[str] = Query(
        None,
        description="Model version (default: production/latest)",
        pattern=r"^\d+\.\d+(\.\d+)?$",
    ),
    min_predictions: int = Query(
        30, ge=5, description="Minimum number of predictions to compute drift"
    ),
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Output distribution drift report (label shift) for a model.

    Compares the recent distribution of `prediction_result` (over `period_days` days)
    against the training distribution stored in `training_stats.label_distribution`.

    - **PSI**: ok < 0.1 | warning 0.1–0.2 | critical ≥ 0.2
    - Returns `status = "no_baseline"` if `label_distribution` is absent from `training_stats`.
    """
    metadata = await DBService.get_model_metadata(db, name, model_version)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' not found.",
        )

    return await drift_service.compute_output_drift(
        model_name=name,
        period_days=period_days,
        db=db,
        model_version=metadata.version,
        min_predictions=min_predictions,
    )


# ---------------------------------------------------------------------------
# Readiness gate
# ---------------------------------------------------------------------------


@router.get("/models/{name}/readiness")
async def get_model_readiness(
    name: ModelNamePath,
    version: str = Query(..., description="Model version to check", pattern=r"^\d+\.\d+(\.\d+)?$"),
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Checks whether a model version is operationally ready for traffic.

    Runs 4 checks:
    - **is_production**: the version is marked as production
    - **file_accessible**: the .joblib file is accessible in MinIO
    - **baseline_computed**: the feature baseline profile is computed
    - **no_critical_drift**: no critical drift in the last 24h

    Always HTTP 200 — `ready: false` is a state, not an error.
    """
    from fastapi.responses import JSONResponse

    model_meta = await DBService.get_model_metadata(db, name, version)
    if not model_meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' version '{version}' not found.",
        )

    prod_check = ReadinessCheck(
        pass_=model_meta.is_production,
        detail=None if model_meta.is_production else "is_production=False",
    )
    has_baseline = bool(model_meta.feature_baseline) or bool(model_meta.categorical_baseline)
    baseline_check = ReadinessCheck(
        pass_=has_baseline,
        detail=None if has_baseline else "feature_baseline and categorical_baseline are null",
    )

    async def _check_file() -> ReadinessCheck:
        info = await asyncio.to_thread(minio_service.get_object_info, model_meta.minio_object_key)
        if info is not None:
            return ReadinessCheck(pass_=True)
        return ReadinessCheck(pass_=False, detail="model file not found in MinIO")

    async def _check_drift() -> ReadinessCheck:
        if not model_meta.feature_baseline and not model_meta.categorical_baseline:
            return ReadinessCheck(pass_=True)
        prod_stats, cat_prod_stats = await asyncio.gather(
            DBService.get_feature_production_stats(db, name, version, days=1),
            DBService.get_categorical_production_stats(db, name, version, days=1),
        )
        features = drift_service.compute_feature_drift(
            model_meta.feature_baseline or {}, prod_stats, min_count=30
        )
        cat_features = drift_service.compute_categorical_drift(
            model_meta.categorical_baseline or {}, cat_prod_stats, min_count=30
        )
        drift_status = drift_service.summarize_drift(
            features, baseline_available=True, categorical_features=cat_features
        )
        if drift_status == "critical":
            return ReadinessCheck(pass_=False, detail=f"drift_status={drift_status}")
        return ReadinessCheck(pass_=True)

    file_check, drift_check = await asyncio.gather(_check_file(), _check_drift())

    checks = ReadinessChecks(
        is_production=prod_check,
        file_accessible=file_check,
        baseline_computed=baseline_check,
        no_critical_drift=drift_check,
    )
    ready = prod_check.pass_ and file_check.pass_ and baseline_check.pass_ and drift_check.pass_

    response = ReadinessResponse(
        model_name=name,
        version=version,
        ready=ready,
        checked_at=datetime.now(timezone.utc).replace(tzinfo=None),
        checks=checks,
    )
    return JSONResponse(content=response.model_dump(by_alias=True, mode="json"))


# ---------------------------------------------------------------------------
# Global feature importance (aggregated SHAP)
# IMPORTANT: declared BEFORE /models/{name}/{version} to prevent
# "feature-importance" from being interpreted as a `version` parameter.
# ---------------------------------------------------------------------------


@router.get("/models/{name}/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(
    name: ModelNamePath,
    version: Optional[str] = Query(
        None,
        description="Model version (default: production/latest)",
        pattern=r"^\d+\.\d+(\.\d+)?$",
    ),
    last_n: int = Query(100, ge=1, le=500, description="Number of predictions to sample"),
    days: int = Query(7, ge=1, le=90, description="Time window in days"),
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Global feature importance via aggregated SHAP values.

    Computes the mean of |SHAP| per feature over a sample of recent predictions
    to identify the most influential features of the model in production
    and detect behavioral drift.

    - **version**: target version; if absent, the `is_production=True` version
      is used, otherwise the most recent.
    - **last_n**: maximum number of predictions to sample (default 100, max 500).
    - **days**: time window in days (default 7).
    """
    metadata = await DBService.get_model_metadata(db, name, version)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' not found.",
        )

    try:
        model_data = await model_service.load_model(db, name, metadata.version)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model '{name}:{metadata.version}': {exc}",
        )

    model = model_data["model"]

    if not hasattr(model, "feature_names_in_"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Model '{name}:{metadata.version}' does not have 'feature_names_in_'. "
                "It must have been trained with a pandas DataFrame."
            ),
        )

    feature_names = list(model.feature_names_in_)
    feature_set = set(feature_names)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    end_naive = end.replace(tzinfo=None)
    start_naive = start.replace(tzinfo=None)

    predictions, _ = await DBService.get_predictions(
        db,
        model_name=name,
        start=start_naive,
        end=end_naive,
        model_version=metadata.version,
        limit=last_n,
    )

    predictions = list(predictions)[:last_n]

    if not predictions:
        return FeatureImportanceResponse(
            model_name=name,
            version=metadata.version,
            sample_size=0,
            feature_importance={},
        )

    shap_accumulator: dict[str, list[float]] = {f: [] for f in feature_names}

    for pred in predictions:
        input_features = pred.input_features
        if not isinstance(input_features, dict):
            continue
        if not feature_set.issubset(set(input_features.keys())):
            continue
        try:
            x = np.array([[input_features[f] for f in feature_names]], dtype=float)
            prediction_result = pred.prediction_result
            explanation = compute_shap_explanation(
                model=model,
                feature_names=feature_names,
                x=x,
                prediction_result=prediction_result,
                feature_baseline=metadata.feature_baseline,
            )
            for feat, val in explanation["shap_values"].items():
                if feat in shap_accumulator:
                    shap_accumulator[feat].append(abs(val))
        except Exception:
            continue

    processed = max((len(v) for v in shap_accumulator.values()), default=0)
    mean_abs = {
        feat: sum(vals) / len(vals) if vals else 0.0 for feat, vals in shap_accumulator.items()
    }
    ranked = sorted(mean_abs.items(), key=lambda kv: kv[1], reverse=True)
    feature_importance = {
        feat: FeatureImportanceItem(mean_abs_shap=round(val, 6), rank=rank + 1)
        for rank, (feat, val) in enumerate(ranked)
    }

    return FeatureImportanceResponse(
        model_name=name,
        version=metadata.version,
        sample_size=processed,
        feature_importance=feature_importance,
    )


# ---------------------------------------------------------------------------
# Model change history
# IMPORTANT: these routes must be declared BEFORE /models/{name}/{version}
# to prevent "history" from being interpreted as a `version` parameter.
# ---------------------------------------------------------------------------


@router.get("/models/{name}/history", response_model=ModelHistoryResponse)
async def list_model_history(
    name: ModelNamePath,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Returns the full history of all versions of a model (sorted by timestamp DESC).

    Requires a valid Bearer token.
    """
    entries, total = await DBService.get_model_history(
        db, name, model_version=None, limit=limit, offset=offset
    )
    return ModelHistoryResponse(
        model_name=name,
        version=None,
        entries=[ModelHistoryEntry.model_validate(e) for e in entries],
        total=total,
    )


@router.get("/models/{name}/{version}/history", response_model=ModelHistoryResponse)
async def list_model_version_history(
    name: ModelNamePath,
    version: ModelVersionPath,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Returns the history of a specific version of a model (sorted by timestamp DESC).

    Requires a valid Bearer token.
    """
    entries, total = await DBService.get_model_history(
        db, name, model_version=version, limit=limit, offset=offset
    )
    return ModelHistoryResponse(
        model_name=name,
        version=version,
        entries=[ModelHistoryEntry.model_validate(e) for e in entries],
        total=total,
    )


@router.post(
    "/models/{name}/{version}/rollback/{history_id}",
    response_model=RollbackResponse,
)
async def rollback_model(
    name: ModelNamePath,
    version: ModelVersionPath,
    history_id: int,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Restores a model's metadata to the state captured in a history entry.

    Only metadata fields are restored (not MinIO/MLflow artifact references).
    If the snapshot had `is_production=True`, other versions are automatically demoted.

    Reserved for administrators.
    """
    # Load the target model
    result = await db.execute(
        select(ModelMetadata)
        .options(selectinload(ModelMetadata.creator))
        .where(and_(ModelMetadata.name == name, ModelMetadata.version == version))
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' version '{version}' not found.",
        )

    # Load the history entry
    history_entry = await DBService.get_history_entry_by_id(db, history_id)
    if not history_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"History entry {history_id} not found.",
        )
    if history_entry.model_name != name or history_entry.model_version != version:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The history entry does not belong to this model/version.",
        )

    snapshot = history_entry.snapshot
    restored_fields = []

    # If the snapshot restores is_production=True → demote other versions
    if snapshot.get("is_production") is True:
        other_result = await db.execute(
            select(ModelMetadata).where(
                and_(
                    ModelMetadata.name == name,
                    ModelMetadata.version != version,
                    ModelMetadata.is_production.is_(True),
                )
            )
        )
        for other in other_result.scalars().all():
            other.is_production = False
            await DBService.log_model_history(
                db,
                other,
                HistoryActionType.SET_PRODUCTION,
                user.id,
                user.username,
                ["is_production"],
            )

    # Apply the snapshot to restorable fields
    for field in _ROLLBACK_FIELDS:
        if field in snapshot:
            value = snapshot[field]
            # Re-parse training_date if stored as ISO string
            if field == "training_date" and isinstance(value, str):
                value = datetime.fromisoformat(value)
            setattr(model, field, value)
            restored_fields.append(field)

    await db.flush()

    # Log the rollback as a new history entry
    new_entry = await DBService.log_model_history(
        db, model, HistoryActionType.ROLLBACK, user.id, user.username, restored_fields
    )

    await db.commit()
    _leaderboard_cache.clear()
    await db.refresh(model)
    audit_log(
        "model.rollback",
        actor_id=user.id,
        resource=f"{name}:{version}",
        details={"history_id": history_id},
    )

    return RollbackResponse(
        model_name=name,
        version=version,
        rolled_back_to_history_id=history_id,
        new_history_id=new_entry.id,
        restored_fields=restored_fields,
        snapshot=history_entry.snapshot,
    )


@router.patch("/models/{name}/{version}/deprecate", response_model=DeprecateModelResponse)
async def deprecate_model_version(
    name: ModelNamePath,
    version: ModelVersionPath,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Deprecates a model version (status → deprecated, is_production → False).

    The model remains viewable (history, metrics, past predictions) but
    calls to POST /predict will return HTTP 410 Gone with a suggestion of the
    current production version. Reversible via PATCH /models/{name}/{version}.

    Reserved for administrators.
    """
    # Load directly (without deprecated filter) to detect already-deprecated
    result = await db.execute(
        select(ModelMetadata).where(
            and_(
                ModelMetadata.name == name,
                ModelMetadata.version == version,
                ModelMetadata.is_active.is_(True),
            )
        )
    )
    meta = result.scalar_one_or_none()
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}/{version}' not found or inactive.",
        )
    if meta.status == "deprecated":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{name}/{version}' is already deprecated.",
        )

    meta = await DBService.deprecate_model(db, name, version)

    await DBService.log_model_history(
        db,
        meta,
        HistoryActionType.DEPRECATED,
        user.id,
        user.username,
        ["status", "is_production", "deprecated_at"],
    )

    logger.info("Model deprecated", model=name, version=version, by=user.username)
    audit_log("model.deprecate", actor_id=user.id, resource=f"{name}:{version}")

    return DeprecateModelResponse(
        name=meta.name,
        version=meta.version,
        status=meta.status,
        is_production=meta.is_production,
        deprecated_at=meta.deprecated_at,
        deprecated_by=user.username,
    )


@router.patch("/models/{name}/policy", response_model=PolicyUpdateResponse)
async def update_model_policy(
    name: ModelNamePath,
    payload: PromotionPolicy,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Sets or updates the auto-promotion policy after retraining for a model.

    The policy is persisted on all active versions of the model.
    On each retraining, if `auto_promote: true`, the new version will be
    automatically promoted to production when thresholds are met:

    - **min_accuracy**: minimum accuracy computed on the `min_sample_validation`
      most recent (prediction, observed result) pairs.
    - **max_latency_p95_ms**: maximum P95 latency of production predictions.
    - **min_sample_validation**: minimum number of validation pairs required
      (default: 10).
    - **auto_promote**: enable or disable auto-promotion (default: false).

    Reserved for administrators.
    """
    result = await db.execute(
        select(ModelMetadata).where(
            and_(ModelMetadata.name == name, ModelMetadata.is_active.is_(True))
        )
    )
    models = result.scalars().all()
    if not models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' not found or inactive.",
        )

    policy_dict = payload.model_dump()
    for model in models:
        model.promotion_policy = policy_dict

    await db.commit()

    logger.info(
        "Auto-promotion policy updated",
        model=name,
        policy=policy_dict,
        updated_by=user.username,
    )
    audit_log("model.policy_update", actor_id=user.id, resource=name)

    return PolicyUpdateResponse(
        model_name=name,
        promotion_policy=payload,
        updated_versions=len(models),
    )


@router.post(
    "/models/{name}/{version}/retrain",
    response_model=TaskRunEnqueued,
    status_code=202,
)
async def retrain_model(
    name: ModelNamePath,
    version: ModelVersionPath,
    payload: RetrainRequest,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Enqueues a model retraining — returns immediately 202 with a ``job_id``.

    The train.py script is executed in a separate ARQ worker (outside the API process).

    **Job tracking:**
    - `GET /jobs/{job_id}` — status and result
    - `GET /jobs/{job_id}/logs` — real-time logs (SSE)

    Reserved for administrators.
    """
    import uuid

    # 1. Verify the source model
    result = await db.execute(
        select(ModelMetadata)
        .options(selectinload(ModelMetadata.creator))
        .where(and_(ModelMetadata.name == name, ModelMetadata.version == version))
    )
    source_model = result.scalar_one_or_none()
    if not source_model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' version '{version}' not found.",
        )
    if not source_model.train_script_object_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Model '{name}' v{version} has no training script. "
                "Upload a train.py via POST /models using the train_file field."
            ),
        )

    # 2. Determine the new version and check uniqueness
    new_version = payload.new_version
    if not new_version:
        new_version = f"{version}-retrain-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    existing = await db.execute(
        select(ModelMetadata).where(
            and_(ModelMetadata.name == name, ModelMetadata.version == new_version)
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Version '{new_version}' already exists for model '{name}'.",
        )

    # 3. Extract source fields into memory (the worker needs them)
    source_fields = {
        "train_script_object_key": source_model.train_script_object_key,
        "description": source_model.description,
        "algorithm": source_model.algorithm,
        "features_count": source_model.features_count,
        "classes": source_model.classes,
        "model_task": source_model.model_task,
        "training_params": source_model.training_params,
        "hyperparameters": source_model.hyperparameters,
        "training_dataset": source_model.training_dataset,
        "feature_baseline": source_model.feature_baseline,
        "confidence_threshold": source_model.confidence_threshold,
        "tags": source_model.tags,
        "webhook_url": source_model.webhook_url,
        "promotion_policy": source_model.promotion_policy,
        "retrain_schedule": source_model.retrain_schedule,
        "accuracy": source_model.accuracy,
        "f1_score": source_model.f1_score,
    }

    # 4. Create the TaskRun entry in DB
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    job_id = uuid.uuid4()
    task_run = TaskRun(
        id=job_id,
        task_type="retrain",
        model_name=name,
        model_version=version,
        new_version=new_version,
        triggered_by=user.username,
        status="queued",
        enqueued_at=now,
    )
    db.add(task_run)
    await db.commit()

    # 5. Enqueue in ARQ
    try:
        arq_pool = await arq_pool_module.get_arq_pool()
        await arq_pool.enqueue_job(
            "retrain_task",
            job_id=str(job_id),
            model_name=name,
            source_version=version,
            new_version=new_version,
            start_date=str(payload.start_date),
            end_date=str(payload.end_date),
            set_production=payload.set_production,
            triggered_by=user.username,
            source_fields=source_fields,
            _job_id=str(job_id),  # ARQ job ID = our task_run ID to retrieve the result
        )
    except Exception as exc:
        # If ARQ is unavailable, mark the job as failed and inform the user
        logger.error("ARQ enqueue failed", job_id=str(job_id), error=str(exc))
        async with AsyncSessionLocal() as write_db:
            from sqlalchemy import update

            await write_db.execute(
                update(TaskRun)
                .where(TaskRun.id == job_id)
                .values(status="failed", error=f"ARQ enqueue failed: {exc}")
            )
            await write_db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ARQ worker unavailable. The job was created (id={job_id}) but could not be enqueued.",
        )

    audit_log(
        "retrain.enqueue",
        actor_id=user.id,
        resource=f"{name}:{version}",
        details={"job_id": str(job_id), "new_version": new_version},
    )
    logger.info(
        "Retrain enqueued",
        model=name,
        source_version=version,
        new_version=new_version,
        job_id=str(job_id),
        triggered_by=user.username,
    )

    return TaskRunEnqueued(
        job_id=job_id,
        status="queued",
        model_name=name,
        model_version=version,
        new_version=new_version,
        triggered_by=user.username,
        enqueued_at=now,
    )


@router.get(
    "/models/{name}/retrain-history",
    response_model=RetrainHistoryResponse,
)
async def get_retrain_history(
    name: ModelNamePath,
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Returns the retraining history for a model.

    Each entry corresponds to a version created by retraining (manual or scheduled).
    The `accuracy`, `f1_score`, `auto_promoted` and `auto_promote_reason` fields reflect
    the values at the time of retraining.

    Requires a valid Bearer token.
    """
    versions_exist = await db.execute(
        select(ModelMetadata).where(ModelMetadata.name == name).limit(1)
    )
    if not versions_exist.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' not found.",
        )

    records, total = await DBService.get_retrain_history(db, name, limit=limit, offset=offset)

    history = []
    for m in records:
        stats = m.training_stats or {}
        history.append(
            RetrainHistoryEntry(
                timestamp=m.created_at,
                source_version=m.parent_version,
                new_version=m.version,
                trained_by=m.trained_by,
                accuracy=m.accuracy,
                f1_score=m.f1_score,
                auto_promoted=stats.get("auto_promoted"),
                auto_promote_reason=stats.get("auto_promote_reason"),
                n_rows=stats.get("n_rows"),
                train_start_date=stats.get("train_start_date"),
                train_end_date=stats.get("train_end_date"),
            )
        )

    return RetrainHistoryResponse(model_name=name, history=history, total=total)


@router.patch(
    "/models/{name}/{version}/schedule",
    response_model=ScheduleUpdateResponse,
)
async def update_retrain_schedule(
    name: ModelNamePath,
    version: ModelVersionPath,
    payload: RetrainScheduleInput,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Sets or updates the automatic retraining schedule for a model version.

    - **cron**: 5-field cron expression (e.g. ``"0 3 * * 1"`` = every Monday at 03:00 UTC).
    - **lookback_days**: history window in days passed via ``TRAIN_START_DATE`` / ``TRAIN_END_DATE``.
    - **auto_promote**: if ``True``, evaluates the model's ``promotion_policy`` after each retrain.
    - **enabled**: ``False`` to pause the schedule without deleting it.

    Reserved for administrators.
    """
    from apscheduler.triggers.cron import CronTrigger

    from src.tasks.retrain_scheduler import (
        _compute_next_run_at,
        add_retrain_job,
        remove_retrain_job,
    )

    # 1. Verify the version exists
    result = await db.execute(
        select(ModelMetadata).where(
            and_(ModelMetadata.name == name, ModelMetadata.version == version)
        )
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' version '{version}' not found.",
        )

    # 2. Validate the cron expression
    cron = payload.cron
    next_run_at: Optional[datetime] = None
    if cron:
        try:
            CronTrigger.from_crontab(cron, timezone="UTC")
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid cron expression: {exc}",
            )
        next_run_at = _compute_next_run_at(cron)
    elif payload.enabled:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="A cron expression is required when enabled=True.",
        )

    # 3. Build the schedule dict (preserving existing last_run_at)
    existing = model.retrain_schedule or {}
    schedule_dict = {
        "cron": cron,
        "lookback_days": payload.lookback_days,
        "auto_promote": payload.auto_promote,
        "enabled": payload.enabled,
        "trigger_on_drift": payload.trigger_on_drift,
        "drift_retrain_cooldown_hours": payload.drift_retrain_cooldown_hours,
        "last_run_at": existing.get("last_run_at"),
        "next_run_at": next_run_at.isoformat() if next_run_at else None,
    }

    # 4. Persist
    model.retrain_schedule = schedule_dict
    await db.commit()
    await db.refresh(model)

    # 5. Update the live scheduler
    if payload.enabled and cron:
        add_retrain_job(name, version, schedule_dict)
    else:
        remove_retrain_job(name, version)

    logger.info(
        "Retraining schedule updated",
        model=name,
        version=version,
        cron=cron,
        enabled=payload.enabled,
        updated_by=user.username,
    )
    audit_log(
        "model.schedule_update",
        actor_id=user.id,
        resource=f"{name}:{version}",
        details={"cron": cron, "enabled": payload.enabled},
    )

    return ScheduleUpdateResponse(
        model_name=name,
        version=version,
        retrain_schedule=model.retrain_schedule,
    )


@router.get("/models/{name}/ab-compare", response_model=ABCompareResponse)
async def get_ab_comparison(
    name: ModelNamePath,
    days: int = Query(30, ge=1, le=90, description="Analysis window in days (max 90)"),
    metric: Optional[str] = Query(
        None,
        description="Metric for significance test: 'error_rate', 'mae', 'response_time_ms'. "
        "Default: automatic selection.",
        pattern=r"^(error_rate|mae|response_time_ms)$",
    ),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Side-by-side comparison of A/B and shadow versions of a model over a sliding window.

    For each version that generated predictions: total, shadow predictions, error rate,
    latency (avg / p95), label distribution, and shadow/production agreement rate (if id_obs).

    Requires a valid Bearer token.
    """
    # Verify the model exists
    all_metas_result = await db.execute(
        select(ModelMetadata).where(
            and_(ModelMetadata.name == name, ModelMetadata.is_active.is_(True))
        )
    )
    all_metas = all_metas_result.scalars().all()
    if not all_metas:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' not found or inactive.",
        )

    meta_by_version = {m.version: m for m in all_metas}

    raw_stats = await DBService.get_ab_comparison_stats(db, name, days=days)
    agreement_by_version = await DBService.get_shadow_agreement_rate(db, name, days=days)

    # Enrich each version with absolute prediction errors (for regression)
    def _try_abs_error(pred, obs) -> Optional[float]:
        try:
            return abs(float(pred) - float(obs))
        except (ValueError, TypeError):
            return None

    for s in raw_stats:
        pairs = await DBService.get_performance_pairs(db, name, model_version=s["version"])
        errors = [e for p, o, _, _ in pairs if (e := _try_abs_error(p, o)) is not None]
        s["prediction_errors"] = errors

    versions_out = []
    for s in raw_stats:
        ver = s["version"]
        m = meta_by_version.get(ver)
        versions_out.append(
            ABVersionStats(
                version=ver,
                deployment_mode=m.deployment_mode if m else None,
                traffic_weight=m.traffic_weight if m else None,
                total_predictions=s["total_predictions"],
                shadow_predictions=s["shadow_predictions"],
                error_rate=s["error_rate"],
                avg_response_time_ms=s["avg_response_time_ms"],
                p95_response_time_ms=s["p95_response_time_ms"],
                prediction_distribution=s["prediction_distribution"],
                agreement_rate=agreement_by_version.get(ver),
            )
        )

    significance_data = compute_ab_significance(raw_stats, metric=metric)
    ab_significance = ABSignificance(**significance_data) if significance_data else None

    return ABCompareResponse(
        model_name=name,
        period_days=days,
        versions=versions_out,
        ab_significance=ab_significance,
    )


# ---------------------------------------------------------------------------
# Shadow Deployment — comparaison enrichie
# ---------------------------------------------------------------------------


def _shadow_recommendation(
    n_comparable: int,
    shadow_accuracy: Optional[float],
    production_accuracy: Optional[float],
    shadow_confidence_delta: Optional[float],
) -> str:
    if n_comparable < 10:
        return "insufficient_data"
    if shadow_accuracy is not None and production_accuracy is not None:
        diff = shadow_accuracy - production_accuracy
        if diff > 0.02:
            return "shadow_better"
        if diff < -0.02:
            return "production_better"
        return "equivalent"
    if shadow_confidence_delta is not None:
        if shadow_confidence_delta >= 0.05:
            return "shadow_better"
        if shadow_confidence_delta <= -0.05:
            return "production_better"
    return "equivalent"


@router.get("/models/{name}/shadow-compare", response_model=ShadowCompareResponse)
async def get_shadow_comparison(
    name: ModelNamePath,
    period_days: int = Query(30, ge=1, le=90, description="Analysis window in days (max 90)"),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Enriched comparison between the shadow and production versions of a model.

    Returns: prediction agreement, confidence delta, latency delta,
    and shadow vs production accuracy if observed_results are available.
    The recommendation (shadow_better / production_better / equivalent / insufficient_data)
    is calculated automatically.

    Requires a valid Bearer token.
    """
    all_metas_result = await db.execute(
        select(ModelMetadata).where(
            and_(ModelMetadata.name == name, ModelMetadata.is_active.is_(True))
        )
    )
    all_metas = all_metas_result.scalars().all()
    if not all_metas:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' not found or inactive.",
        )

    stats = await DBService.get_shadow_comparison_stats(db, name, period_days=period_days)
    recommendation = _shadow_recommendation(
        n_comparable=stats["n_comparable"],
        shadow_accuracy=stats["shadow_accuracy"],
        production_accuracy=stats["production_accuracy"],
        shadow_confidence_delta=stats["shadow_confidence_delta"],
    )

    return ShadowCompareResponse(
        model_name=name,
        shadow_version=stats["shadow_version"],
        production_version=stats["production_version"],
        period_days=period_days,
        n_comparable=stats["n_comparable"],
        agreement_rate=stats["agreement_rate"],
        shadow_confidence_delta=stats["shadow_confidence_delta"],
        shadow_latency_delta_ms=stats["shadow_latency_delta_ms"],
        shadow_accuracy=stats["shadow_accuracy"],
        production_accuracy=stats["production_accuracy"],
        accuracy_available=stats["accuracy_available"],
        recommendation=recommendation,
    )


# ---------------------------------------------------------------------------
# Calibration des probabilités
# ---------------------------------------------------------------------------


@router.get("/models/{name}/calibration", response_model=CalibrationResponse)
async def get_model_calibration(
    name: ModelNamePath,
    version: Optional[str] = Query(
        None, description="Model version (all if absent)", pattern=r"^\d+\.\d+(\.\d+)?$"
    ),
    start: Optional[datetime] = Query(None, description="Start of the time range"),
    end: Optional[datetime] = Query(None, description="End of the time range"),
    n_bins: int = Query(10, ge=2, le=20, description="Number of buckets for the calibration curve"),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Analyses the probability calibration of a classification model.

    Returns the Brier score, confidence/accuracy gap, calibration status
    and the reliability diagram to diagnose over/under-confidence.
    """
    pairs = await DBService.get_performance_pairs(db, name, start, end, version)

    if not pairs:
        return CalibrationResponse(
            model_name=name,
            version=version,
            sample_size=0,
            calibration_status="insufficient_data",
        )

    # ── Model type detection ──────────────────────────────────────────────────
    # Regression: no pair has probabilities
    has_proba = any(row.probabilities for row in pairs)

    # ── Regression branch ────────────────────────────────────────────────────
    if not has_proba:
        reg_pairs: list[tuple[float, float]] = []
        for row in pairs:
            try:
                pred = float(row.prediction_result)
                obs = float(row.observed_result)
                reg_pairs.append((pred, obs))
            except (TypeError, ValueError):
                pass

        sample_size = len(reg_pairs)
        if sample_size < 2:
            return CalibrationResponse(
                model_name=name,
                version=version,
                sample_size=sample_size,
                model_type="regression",
                calibration_status="insufficient_data",
            )

        preds_arr = np.array([p for p, _ in reg_pairs], dtype=float)
        obs_arr = np.array([o for _, o in reg_pairs], dtype=float)
        residuals = preds_arr - obs_arr

        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals**2)))
        bias = float(np.mean(residuals))

        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((obs_arr - float(np.mean(obs_arr))) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None

        # Status: based on relative bias (in units of observed standard deviation)
        obs_std = float(np.std(obs_arr)) if sample_size > 1 else 1.0
        rel_bias = bias / obs_std if obs_std > 0 else 0.0
        if abs(rel_bias) < 0.10:
            reg_status = "ok"
        elif rel_bias > 0:
            reg_status = "biased_high"
        else:
            reg_status = "biased_low"

        # Scatter data — random sample ≤ 300 points
        rng = np.random.default_rng(42)
        idx = rng.choice(sample_size, min(300, sample_size), replace=False)
        scatter_data = [
            {"pred": round(float(preds_arr[i]), 4), "obs": round(float(obs_arr[i]), 4)}
            for i in sorted(idx)
        ]

        return CalibrationResponse(
            model_name=name,
            version=version,
            sample_size=sample_size,
            model_type="regression",
            calibration_status=reg_status,
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            r2=round(r2, 4) if r2 is not None else None,
            bias=round(bias, 4),
            scatter_data=scatter_data,
        )

    # ── Classification branch ─────────────────────────────────────────────────
    valid = [
        (row.prediction_result, row.observed_result, row.probabilities)
        for row in pairs
        if row.probabilities
    ]

    if not valid:
        return CalibrationResponse(
            model_name=name,
            version=version,
            sample_size=len(pairs),
            calibration_status="insufficient_data",
        )

    def _max_prob(p) -> float:
        return max(p.values()) if isinstance(p, dict) else max(p)

    confidences = np.array([_max_prob(p) for _, _, p in valid], dtype=float)
    corrects = np.array(
        [1.0 if str(pred) == str(obs) else 0.0 for pred, obs, _ in valid],
        dtype=float,
    )

    sample_size = len(valid)

    if sample_size < 30:
        return CalibrationResponse(
            model_name=name,
            version=version,
            sample_size=sample_size,
            calibration_status="insufficient_data",
        )

    brier_score = float(np.mean((confidences - corrects) ** 2))
    mean_confidence = float(np.mean(confidences))
    mean_accuracy = float(np.mean(corrects))
    gap = mean_confidence - mean_accuracy

    if abs(gap) < 0.05:
        calibration_status = "ok"
    elif gap > 0.05:
        calibration_status = "overconfident"
    else:
        calibration_status = "underconfident"

    # Reliability diagram — buckets [0, 1/n_bins), [1/n_bins, 2/n_bins), ...
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability: List[ReliabilityBin] = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        reliability.append(
            ReliabilityBin(
                confidence_bin=f"{lo:.1f}–{hi:.1f}",
                predicted_rate=round(float(np.mean(confidences[mask])), 4),
                observed_rate=round(float(np.mean(corrects[mask])), 4),
                count=count,
            )
        )

    return CalibrationResponse(
        model_name=name,
        version=version,
        sample_size=sample_size,
        model_type="classification",
        brier_score=round(brier_score, 4),
        calibration_status=calibration_status,
        mean_confidence=round(mean_confidence, 4),
        mean_accuracy=round(mean_accuracy, 4),
        overconfidence_gap=round(gap, 4),
        reliability=reliability,
    )


# ---------------------------------------------------------------------------
# Consolidated performance report — private helpers
# ---------------------------------------------------------------------------


def _build_performance_section(
    name: str,
    metadata: ModelMetadata,
    total: int,
    pairs: list,
) -> Optional[ModelPerformanceResponse]:
    try:
        matched = len(pairs)
        model_type = _detect_model_type(metadata, pairs)
        if matched == 0:
            return ModelPerformanceResponse(
                model_name=name,
                model_version=metadata.version,
                period_start=None,
                period_end=None,
                total_predictions=total,
                matched_predictions=0,
                model_type=model_type,
            )
        y_true = [row.observed_result for row in pairs]
        y_pred = [row.prediction_result for row in pairs]
        response = ModelPerformanceResponse(
            model_name=name,
            model_version=metadata.version,
            period_start=None,
            period_end=None,
            total_predictions=total,
            matched_predictions=matched,
            model_type=model_type,
        )
        if model_type == "classification":
            acc, prec, rec, f1, cm, classes, per_class = _compute_classification_metrics(
                y_true, y_pred, metadata.classes
            )
            response.accuracy = round(acc, 4)
            response.precision_weighted = round(float(prec), 4)
            response.recall_weighted = round(float(rec), 4)
            response.f1_weighted = round(float(f1), 4)
            response.confusion_matrix = cm
            response.classes = classes
            response.per_class_metrics = per_class
        else:
            y_true_f = [float(v) for v in y_true]
            y_pred_f = [float(v) for v in y_pred]
            mae, mse, rmse, r2 = _compute_regression_metrics(y_true_f, y_pred_f)
            response.mae = round(mae, 4)
            response.mse = round(mse, 4)
            response.rmse = round(rmse, 4)
            response.r2 = round(r2, 4)
        return response
    except Exception:
        return None


def _build_drift_section(
    name: str,
    metadata: ModelMetadata,
    days: int,
    production_stats: dict,
    categorical_production_stats: Optional[dict] = None,
) -> Optional[DriftReportResponse]:
    try:
        cat_prod_stats = categorical_production_stats or {}
        total_predictions = sum(v.get("count", 0) for v in production_stats.values())
        if not total_predictions:
            total_predictions = max(
                (v.get("_count", 0) for v in cat_prod_stats.values()), default=0
            )
        baseline = metadata.feature_baseline or {}
        categorical_baseline = metadata.categorical_baseline or {}
        baseline_available = bool(baseline) or bool(categorical_baseline)
        if not baseline_available:
            return DriftReportResponse(
                model_name=name,
                model_version=metadata.version,
                period_days=days,
                predictions_analyzed=total_predictions,
                baseline_available=False,
                drift_summary="no_baseline",
                features={
                    feat: FeatureDriftResult(
                        production_mean=round(stats["mean"], 6),
                        production_std=round(stats["std"], 6),
                        production_count=stats["count"],
                        null_rate_production=(
                            round(stats["null_rate"], 6) if "null_rate" in stats else None
                        ),
                        drift_status="no_baseline",
                    )
                    for feat, stats in production_stats.items()
                },
            )
        features = drift_service.compute_feature_drift(baseline, production_stats)
        categorical_features = drift_service.compute_categorical_drift(
            categorical_baseline, cat_prod_stats
        )
        summary = drift_service.summarize_drift(
            features, baseline_available=True, categorical_features=categorical_features
        )
        return DriftReportResponse(
            model_name=name,
            model_version=metadata.version,
            period_days=days,
            predictions_analyzed=total_predictions,
            baseline_available=True,
            drift_summary=summary,
            features=features,
            categorical_features=categorical_features,
        )
    except Exception:
        return None


def _build_calibration_section(
    name: str,
    version: Optional[str],
    pairs: list,
    n_bins: int = 10,
) -> Optional[CalibrationResponse]:
    try:
        if not pairs:
            return CalibrationResponse(
                model_name=name,
                version=version,
                sample_size=0,
                brier_score=None,
                calibration_status="insufficient_data",
                mean_confidence=None,
                mean_accuracy=None,
                overconfidence_gap=None,
                reliability=[],
            )
        valid = [
            (row.prediction_result, row.observed_result, row.probabilities)
            for row in pairs
            if row.probabilities
        ]
        if not valid:
            return None

        def _max_prob(p) -> float:
            return max(p.values()) if isinstance(p, dict) else max(p)

        confidences = np.array([_max_prob(p) for _, _, p in valid], dtype=float)
        corrects = np.array(
            [1.0 if str(pred) == str(obs) else 0.0 for pred, obs, _ in valid],
            dtype=float,
        )
        sample_size = len(valid)
        if sample_size < 30:
            return CalibrationResponse(
                model_name=name,
                version=version,
                sample_size=sample_size,
                brier_score=None,
                calibration_status="insufficient_data",
                mean_confidence=None,
                mean_accuracy=None,
                overconfidence_gap=None,
                reliability=[],
            )
        brier_score = float(np.mean((confidences - corrects) ** 2))
        mean_conf = float(np.mean(confidences))
        mean_acc = float(np.mean(corrects))
        gap = mean_conf - mean_acc
        if abs(gap) < 0.05:
            calibration_status = "ok"
        elif gap > 0.05:
            calibration_status = "overconfident"
        else:
            calibration_status = "underconfident"
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        reliability: List[ReliabilityBin] = []
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
            count = int(mask.sum())
            if count == 0:
                continue
            reliability.append(
                ReliabilityBin(
                    confidence_bin=f"{lo:.1f}–{hi:.1f}",
                    predicted_rate=round(float(np.mean(confidences[mask])), 4),
                    observed_rate=round(float(np.mean(corrects[mask])), 4),
                    count=count,
                )
            )
        return CalibrationResponse(
            model_name=name,
            version=version,
            sample_size=sample_size,
            brier_score=round(brier_score, 4),
            calibration_status=calibration_status,
            mean_confidence=round(mean_conf, 4),
            mean_accuracy=round(mean_acc, 4),
            overconfidence_gap=round(gap, 4),
            reliability=reliability,
        )
    except Exception:
        return None


def _build_ab_comparison_section(
    name: str,
    days: int,
    meta_by_version: Dict[str, Any],
    raw_stats: list,
    agreement_by_version: dict,
    pairs: list,
) -> Optional[ABCompareResponse]:
    try:
        if not raw_stats:
            return None
        for s in raw_stats:
            s.setdefault("prediction_errors", [])
        versions_out = []
        for s in raw_stats:
            ver = s["version"]
            m = meta_by_version.get(ver)
            versions_out.append(
                ABVersionStats(
                    version=ver,
                    deployment_mode=m.deployment_mode if m else None,
                    traffic_weight=m.traffic_weight if m else None,
                    total_predictions=s["total_predictions"],
                    shadow_predictions=s["shadow_predictions"],
                    error_rate=s["error_rate"],
                    avg_response_time_ms=s["avg_response_time_ms"],
                    p95_response_time_ms=s["p95_response_time_ms"],
                    prediction_distribution=s["prediction_distribution"],
                    agreement_rate=agreement_by_version.get(ver),
                )
            )
        significance_data = compute_ab_significance(raw_stats)
        ab_significance = ABSignificance(**significance_data) if significance_data else None
        return ABCompareResponse(
            model_name=name,
            period_days=days,
            versions=versions_out,
            ab_significance=ab_significance,
        )
    except Exception:
        return None


async def _build_feature_importance_section(
    db: AsyncSession,
    name: str,
    metadata: ModelMetadata,
    predictions: list,
    last_n: int = 100,
) -> Optional[FeatureImportanceResponse]:
    try:
        model_data = await model_service.load_model(db, name, metadata.version)
        model = model_data["model"]
        if not hasattr(model, "feature_names_in_"):
            return None
        feature_names = list(model.feature_names_in_)
        feature_set = set(feature_names)
        predictions = list(predictions)[:last_n]
        if not predictions:
            return FeatureImportanceResponse(
                model_name=name,
                version=metadata.version,
                sample_size=0,
                feature_importance={},
            )
        shap_accumulator: Dict[str, List[float]] = {f: [] for f in feature_names}
        for pred in predictions:
            input_features = pred.input_features
            if not isinstance(input_features, dict):
                continue
            if not feature_set.issubset(set(input_features.keys())):
                continue
            try:
                x = np.array([[input_features[f] for f in feature_names]], dtype=float)
                explanation = compute_shap_explanation(
                    model=model,
                    feature_names=feature_names,
                    x=x,
                    prediction_result=pred.prediction_result,
                    feature_baseline=metadata.feature_baseline,
                )
                for feat, val in explanation["shap_values"].items():
                    if feat in shap_accumulator:
                        shap_accumulator[feat].append(abs(val))
            except Exception:
                continue
        processed = max((len(v) for v in shap_accumulator.values()), default=0)
        mean_abs = {
            feat: sum(vals) / len(vals) if vals else 0.0 for feat, vals in shap_accumulator.items()
        }
        ranked = sorted(mean_abs.items(), key=lambda kv: kv[1], reverse=True)
        feature_importance = {
            feat: FeatureImportanceItem(mean_abs_shap=round(val, 6), rank=rank + 1)
            for rank, (feat, val) in enumerate(ranked)
        }
        return FeatureImportanceResponse(
            model_name=name,
            version=metadata.version,
            sample_size=processed,
            feature_importance=feature_importance,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Consolidated performance report — endpoint
# ---------------------------------------------------------------------------


@router.get("/models/{name}/performance-report", response_model=PerformanceReportResponse)
async def get_performance_report(
    name: ModelNamePath,
    days: int = Query(30, ge=1, le=365, description="Analysis window in days"),
    format: str = Query("json", description="Output format (json; html planned for phase 2)"),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Consolidated performance report for a model.

    Aggregates in a single call: actual performance, drift, feature importance (SHAP),
    probability calibration and A/B comparison.
    Each section is independently nullable — a missing section does not block the others.
    """
    metadata = await DBService.get_model_metadata(db, name)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' not found.",
        )

    version = metadata.version
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    now_naive = now.replace(tzinfo=None)
    start_naive = start.replace(tzinfo=None)

    # All DB I/O in one parallel gather (same pattern as compare_model_versions)
    (
        total_count,
        pairs,
        production_stats,
        cat_prod_stats,
        raw_ab_stats,
        agreement_by_version,
        predictions_result,
    ) = await asyncio.gather(
        DBService.count_predictions(db, name, start_naive, now_naive, version),
        DBService.get_performance_pairs(db, name, start_naive, now_naive, version),
        DBService.get_feature_production_stats(db, name, version, days),
        DBService.get_categorical_production_stats(db, name, version, days),
        DBService.get_ab_comparison_stats(db, name, days=days),
        DBService.get_shadow_agreement_rate(db, name, days=days),
        DBService.get_predictions(
            db, model_name=name, start=start_naive, end=now_naive, model_version=version, limit=100
        ),
    )

    predictions, _ = predictions_result

    all_metas_result = await db.execute(
        select(ModelMetadata).where(
            and_(ModelMetadata.name == name, ModelMetadata.is_active.is_(True))
        )
    )
    meta_by_version = {m.version: m for m in all_metas_result.scalars().all()}

    return PerformanceReportResponse(
        model_name=name,
        generated_at=now.replace(tzinfo=None),
        period_days=days,
        performance=_build_performance_section(name, metadata, total_count, pairs),
        drift=_build_drift_section(name, metadata, days, production_stats, cat_prod_stats),
        calibration=_build_calibration_section(name, version, pairs),
        ab_comparison=_build_ab_comparison_section(
            name, days, meta_by_version, raw_ab_stats, agreement_by_version, pairs
        ),
        feature_importance=await _build_feature_importance_section(db, name, metadata, predictions),
    )


@router.get("/models/{name}/confidence-trend", response_model=ConfidenceTrendResponse)
async def get_confidence_trend(
    name: ModelNamePath,
    version: Optional[str] = Query(
        None, description="Model version (all if absent)", pattern=r"^\d+\.\d+(\.\d+)?$"
    ),
    days: int = Query(30, ge=1, le=365, description="Sliding window in days"),
    granularity: str = Query("day", description="Temporal granularity (day)"),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Model confidence trend over a sliding window.

    confidence = max(probabilities) per prediction — early drift signal
    without requiring a baseline or observed_results.
    If the model does not return probabilities, the trend list is empty.
    """
    metadata = await DBService.get_model_metadata(db, name, version)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' not found.",
        )

    threshold = metadata.confidence_threshold if metadata.confidence_threshold is not None else 0.5

    result = await DBService.get_confidence_trend(
        db,
        model_name=name,
        version=version,
        days=days,
        confidence_threshold=threshold,
    )

    empty_overall = ConfidenceTrendOverall(
        mean_confidence=0.0,
        p25_confidence=0.0,
        p75_confidence=0.0,
        low_confidence_rate=0.0,
    )

    if not result["has_data"]:
        return ConfidenceTrendResponse(
            model_name=name,
            version=version,
            period_days=days,
            overall=empty_overall,
            trend=[],
        )

    return ConfidenceTrendResponse(
        model_name=name,
        version=version,
        period_days=days,
        overall=ConfidenceTrendOverall(**result["overall"]),
        trend=[ConfidenceTrendPoint(**p) for p in result["trend"]],
    )


@router.get(
    "/models/{name}/confidence-distribution",
    response_model=ConfidenceDistributionResponse,
)
async def get_confidence_distribution(
    name: ModelNamePath,
    version: Optional[str] = Query(
        None, description="Model version (all if absent)", pattern=r"^\d+\.\d+(\.\d+)?$"
    ),
    days: int = Query(7, ge=1, le=90, description="Sliding window in days"),
    high_threshold: float = Query(0.80, ge=0.5, le=1.0, description="High confidence threshold"),
    uncertain_threshold: float = Query(
        0.60, ge=0.5, le=1.0, description="Uncertainty alert threshold"
    ),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Model confidence histogram over a sliding window.

    confidence = max(probabilities) per prediction — no ground truth required.
    Returns 10 uniform bins in [0.5, 1.0] plus global metrics.
    If the model does not return probabilities, histogram is empty.
    """
    metadata = await DBService.get_model_metadata(db, name, version)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' not found.",
        )

    result = await DBService.get_confidence_distribution(
        db,
        model_name=name,
        version=version,
        days=days,
        high_threshold=high_threshold,
        uncertain_threshold=uncertain_threshold,
    )

    if not result["has_data"]:
        return ConfidenceDistributionResponse(
            model_name=name,
            version=version,
            period_days=days,
            sample_count=0,
            mean_confidence=0.0,
            pct_high_confidence=0.0,
            pct_uncertain=0.0,
            histogram=[],
        )

    return ConfidenceDistributionResponse(
        model_name=name,
        version=version,
        period_days=days,
        sample_count=result["sample_count"],
        mean_confidence=result["mean_confidence"],
        pct_high_confidence=result["pct_high_confidence"],
        pct_uncertain=result["pct_uncertain"],
        histogram=[ConfidenceBin(**b) for b in result["histogram"]],
    )


@router.get("/models/{name}/compare", response_model=ModelCompareResponse)
async def compare_model_versions(
    name: ModelNamePath,
    versions: Optional[str] = Query(
        None,
        description="Comma-separated versions (e.g. 1.0.0,2.0.0). All active versions if absent.",
    ),
    days: int = Query(
        7,
        ge=1,
        le=90,
        description="Time window for latency stats and live metrics (days)",
    ),
    start_date: Optional[str] = Query(
        None, description="Start date ISO (YYYY-MM-DD) — takes priority over days"
    ),
    end_date: Optional[str] = Query(
        None, description="End date ISO (YYYY-MM-DD) — takes priority over days"
    ),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Multi-version comparison of a model in a single call.

    Aggregates for each version: accuracy, F1, p50/p95 latency, drift status,
    Brier score, training date and number of training rows.

    If ?versions is omitted, returns all active versions of the model.
    Requires a valid Bearer token.
    """
    all_metas_result = await db.execute(
        select(ModelMetadata).where(
            and_(ModelMetadata.name == name, ModelMetadata.is_active.is_(True))
        )
    )
    all_metas = all_metas_result.scalars().all()
    if not all_metas:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' not found or inactive.",
        )

    if versions:
        requested = {v.strip() for v in versions.split(",") if v.strip()}
        filtered_metas = [m for m in all_metas if m.version in requested]
        if not filtered_metas:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"None of the requested versions ({versions}) is active for '{name}'.",
            )
    else:
        filtered_metas = list(all_metas)

    meta_by_version = {m.version: m for m in filtered_metas}

    # Resolve the time window: start_date/end_date take priority over days
    _period_start: Optional[datetime] = None
    _period_end: Optional[datetime] = None
    if start_date:
        try:
            _period_start = datetime.fromisoformat(start_date)
        except ValueError:
            pass
    if end_date:
        try:
            _period_end = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59)
        except ValueError:
            pass
    if _period_start is None and _period_end is None:
        _period_start = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        _period_end = datetime.now(timezone.utc).replace(tzinfo=None)

    # Latency stats — single DB query covering all versions, filtered on the same period
    ab_stats = await DBService.get_ab_comparison_stats(
        db, name, days=days, start=_period_start, end=_period_end
    )
    latency_by_version: dict = {}
    for s in ab_stats:
        times = sorted(s.get("response_times", []))
        n = len(times)
        p50 = round(times[max(0, int(n * 0.5) - 1)], 2) if n > 0 else None
        latency_by_version[s["version"]] = {
            "p50": p50,
            "p95": s.get("p95_response_time_ms"),
            "total": s.get("total_predictions", 0),
            "shadow": s.get("shadow_predictions", 0),
        }

    # Per-version extras (drift + live metrics) — parallel gather
    async def _version_extras(version: str, meta: ModelMetadata):
        prod_stats, cat_prod_stats_v, pairs = await asyncio.gather(
            DBService.get_feature_production_stats(
                db, name, version, days, start=_period_start, end=_period_end
            ),
            DBService.get_categorical_production_stats(db, name, version, days),
            DBService.get_performance_pairs(
                db, name, model_version=version, start=_period_start, end=_period_end
            ),
        )

        baseline = meta.feature_baseline or {}
        cat_baseline = meta.categorical_baseline or {}
        has_any_baseline = bool(baseline) or bool(cat_baseline)
        if has_any_baseline and (prod_stats or cat_prod_stats_v):
            feat_results = drift_service.compute_feature_drift(baseline, prod_stats, min_count=30)
            cat_results = drift_service.compute_categorical_drift(
                cat_baseline, cat_prod_stats_v, min_count=30
            )
            drift_status = drift_service.summarize_drift(
                feat_results, baseline_available=True, categorical_features=cat_results
            )
        elif has_any_baseline:
            drift_status = "insufficient_data"
        else:
            drift_status = "no_baseline"

        brier_score = None
        live_accuracy = None
        live_auc = None
        live_f1 = None
        live_mae = None
        live_rmse = None
        live_r2 = None
        valid_pairs = [(row[0], row[1], row[2]) for row in pairs if row[2]]
        all_pairs = [(row[0], row[1]) for row in pairs]
        all_probs = [row[2] for row in pairs]  # probabilities for each prediction

        # Determine model type: model_task takes priority, fallback to training_metrics
        if meta.model_task == "regression":
            _is_reg = True
        elif meta.model_task in ("classification_binary", "classification_multiclass"):
            _is_reg = False
        else:
            _tm = meta.training_metrics or {}
            _is_reg = any(k in _tm for k in ("mae", "rmse", "r2"))

        if all_pairs:
            if _is_reg:
                try:
                    preds_f = np.array([float(pred) for pred, _ in all_pairs])
                    obs_f = np.array([float(obs) for _, obs in all_pairs])
                    live_mae = round(float(mean_absolute_error(obs_f, preds_f)), 4)
                    live_rmse = round(float(np.sqrt(np.mean((preds_f - obs_f) ** 2))), 4)
                    ss_res = np.sum((obs_f - preds_f) ** 2)
                    ss_tot = np.sum((obs_f - obs_f.mean()) ** 2)
                    live_r2 = round(float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0, 4)
                except (ValueError, TypeError):
                    pass
            else:
                y_pred_s = [str(pred) for pred, _ in all_pairs]
                y_obs_s = [str(obs) for _, obs in all_pairs]
                correct = sum(p == o for p, o in zip(y_pred_s, y_obs_s))
                live_accuracy = round(correct / len(all_pairs), 4)
                try:
                    live_f1 = round(
                        float(f1_score(y_obs_s, y_pred_s, average="weighted", zero_division=0)), 4
                    )
                except Exception:
                    pass
                # Live AUC from stored probabilities
                live_auc = compute_auc([obs for _, obs in all_pairs], all_probs, meta.classes)

        if len(valid_pairs) >= 30:

            def _max_prob(p):
                return max(p.values()) if isinstance(p, dict) else max(p)

            confidences = np.array([_max_prob(probs) for _, _, probs in valid_pairs])
            corrects = np.array(
                [1.0 if str(pred) == str(obs) else 0.0 for pred, obs, _ in valid_pairs]
            )
            brier_score = round(float(np.mean((confidences - corrects) ** 2)), 4)

        return (
            drift_status,
            brier_score,
            live_accuracy,
            live_auc,
            live_f1,
            live_mae,
            live_rmse,
            live_r2,
        )

    ordered_versions = sorted(meta_by_version.keys())
    extras_list = await asyncio.gather(
        *[_version_extras(v, meta_by_version[v]) for v in ordered_versions]
    )
    extras_by_version = dict(zip(ordered_versions, extras_list))

    version_summaries = []
    for version in ordered_versions:
        meta = meta_by_version[version]
        (
            drift_status,
            brier_score,
            live_accuracy,
            live_auc,
            live_f1,
            live_mae,
            live_rmse,
            live_r2,
        ) = extras_by_version[version]
        lat = latency_by_version.get(version, {})
        training_stats = meta.training_stats or {}
        n_rows = training_stats.get("n_rows")
        training_metrics = meta.training_metrics or {}
        version_summaries.append(
            ModelVersionSummary(
                version=version,
                is_production=meta.is_production,
                model_task=meta.model_task,
                accuracy=meta.accuracy,
                auc=meta.auc,
                f1_score=meta.f1_score,
                latency_p50_ms=lat.get("p50"),
                latency_p95_ms=lat.get("p95"),
                drift_status=drift_status,
                brier_score=brier_score,
                trained_at=meta.created_at,
                n_rows_trained=n_rows,
                prediction_count=lat.get("total") or 0,
                shadow_prediction_count=lat.get("shadow") or 0,
                mae_eval=training_metrics.get("mae") if training_metrics else None,
                rmse_eval=training_metrics.get("rmse") if training_metrics else None,
                r2_eval=training_metrics.get("r2") if training_metrics else None,
                live_accuracy=live_accuracy,
                live_auc=live_auc,
                live_f1=live_f1,
                live_mae=live_mae,
                live_rmse=live_rmse,
                live_r2=live_r2,
            )
        )

    return ModelCompareResponse(
        model_name=name,
        compared_at=datetime.now(timezone.utc).replace(tzinfo=None),
        versions=version_summaries,
    )


# ---------------------------------------------------------------------------
# Model Card Export — helpers + endpoint
# ---------------------------------------------------------------------------

_DRIFT_EMOJI = {
    "ok": "✅",
    "warning": "⚠️",
    "critical": "❌",
    "no_baseline": "ℹ️",
    "insufficient_data": "ℹ️",
}


def _build_model_card_markdown(card: ModelCardResponse) -> str:
    lines: List[str] = [
        f"# Model Card — {card.model_name} v{card.version}",
        "",
        f"**Algorithm**: {card.algorithm or '—'}",
    ]
    # Performance metrics — prefer live section, fall back to training-time metadata
    if card.performance and card.performance.matched_predictions > 0:
        p = card.performance
        if p.accuracy is not None:
            lines.append(f"**Accuracy** : {p.accuracy} | **F1** : {p.f1_weighted}")
        elif p.mae is not None:
            lines.append(f"**MAE** : {p.mae} | **RMSE** : {p.rmse}")
    elif card.accuracy is not None:
        lines.append(f"**Accuracy** : {card.accuracy} | **F1** : {card.f1_score}")

    lines.append(f"**Production**: {'✅ Yes' if card.is_production else '❌ No'}")
    if card.created_at:
        lines.append(f"**Created at**: {card.created_at.strftime('%Y-%m-%d')}")
    if card.trained_by:
        lines.append(f"**Trained by**: {card.trained_by}")
    if card.training_dataset:
        lines.append(f"**Dataset**: {card.training_dataset}")
    if card.tags:
        lines.append(f"**Tags**: {', '.join(str(t) for t in card.tags)}")
    if card.classes:
        lines.append(f"**Classes**: {', '.join(str(c) for c in card.classes)}")
    if card.features_count:
        lines.append(f"**Feature count**: {card.features_count}")

    lines += ["", "---", ""]

    if card.drift:
        d = card.drift
        emoji = _DRIFT_EMOJI.get(d.drift_summary, "❓")
        last = d.last_check_at.strftime("%Y-%m-%d") if d.last_check_at else "—"
        lines.append(f"**Drift** : {emoji} {d.drift_summary.capitalize()} (last check {last})")
        if d.top_drifting_features:
            lines.append(f"  Drifting features: {', '.join(d.top_drifting_features)}")

    if card.retrain:
        r = card.retrain
        trained_by = f" ({r.trained_by})" if r.trained_by else ""
        date_str = r.last_retrain_date.strftime("%Y-%m-%d") if r.last_retrain_date else "—"
        lines.append(f"**Last retrain**: {date_str}{trained_by}")
        if r.n_rows_trained:
            lines.append(f"**Training data**: {r.n_rows_trained:,} rows")
        if r.next_run_at:
            lines.append(f"**Next retrain**: {r.next_run_at.strftime('%Y-%m-%d %H:%M')} UTC")

    if card.feature_importance and card.feature_importance.top_features:
        parts = [
            f"{f.feature} ({f.mean_abs_shap:.2f})" for f in card.feature_importance.top_features
        ]
        lines.append(f"**Key features**: {', '.join(parts)}")

    if card.calibration and card.calibration.brier_score is not None:
        lines.append(
            f"**Calibration** : {card.calibration.calibration_status}"
            f" (Brier : {card.calibration.brier_score:.4f})"
        )

    if card.coverage:
        cov = card.coverage
        pct = round(cov.coverage_rate * 100, 1)
        lines.append(f"**Coverage**: {pct}% ({cov.labeled_count}/{cov.total_predictions})")

    lines += ["", "---", f"_Generated on {card.generated_at.strftime('%Y-%m-%d %H:%M')} UTC_"]
    return "\n".join(lines)


@router.get("/models/{name}/{version}/card")
async def get_model_card(
    name: ModelNamePath,
    version: ModelVersionPath,
    request: Request,
    days: int = Query(30, ge=1, le=365, description="Analysis window in days"),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Model Card — structured summary of a model version.

    Aggregates in a single call: metadata, performance, drift, calibration,
    top-5 SHAP features, retrain info and ground truth coverage.

    Accept: application/json  → JSON (ModelCardResponse)
    Accept: text/markdown     → downloadable Markdown
    """
    metadata = await DBService.get_model_metadata(db, name, version)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' version '{version}' not found.",
        )

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    now_naive = now.replace(tzinfo=None)
    start_naive = start.replace(tzinfo=None)

    (
        total_count,
        pairs,
        production_stats,
        cat_prod_stats_card,
        coverage_stats,
        predictions_result,
    ) = await asyncio.gather(
        DBService.count_predictions(db, name, start_naive, now_naive, version),
        DBService.get_performance_pairs(db, name, start_naive, now_naive, version),
        DBService.get_feature_production_stats(db, name, version, days),
        DBService.get_categorical_production_stats(db, name, version, days),
        DBService.get_observed_results_stats(db, model_name=name),
        DBService.get_predictions(
            db, model_name=name, start=start_naive, end=now_naive, model_version=version, limit=50
        ),
    )
    predictions, _ = predictions_result

    # Performance
    performance_section: Optional[ModelCardPerformanceSummary] = None
    try:
        raw_perf = _build_performance_section(name, metadata, total_count, pairs)
        if raw_perf:
            performance_section = ModelCardPerformanceSummary(
                model_type=raw_perf.model_type,
                matched_predictions=raw_perf.matched_predictions,
                total_predictions=raw_perf.total_predictions,
                accuracy=raw_perf.accuracy,
                f1_weighted=raw_perf.f1_weighted,
                mae=raw_perf.mae,
                rmse=raw_perf.rmse,
            )
    except Exception:
        pass

    # Drift
    drift_section: Optional[ModelCardDriftSummary] = None
    try:
        raw_drift = _build_drift_section(name, metadata, days, production_stats, cat_prod_stats_card)
        if raw_drift:
            top_drifting: Optional[List[str]] = None
            if raw_drift.drift_summary in ("warning", "critical"):
                all_drifting = [
                    (feat, info.drift_status)
                    for feat, info in raw_drift.features.items()
                    if info.drift_status in ("warning", "critical")
                ] + [
                    (feat, info.drift_status)
                    for feat, info in raw_drift.categorical_features.items()
                    if info.drift_status in ("warning", "critical")
                ]
                sorted_feats = sorted(
                    all_drifting, key=lambda x: (0 if x[1] == "critical" else 1)
                )
                top_drifting = [f for f, _ in sorted_feats[:3]] or None
            drift_section = ModelCardDriftSummary(
                drift_summary=raw_drift.drift_summary,
                baseline_available=raw_drift.baseline_available,
                predictions_analyzed=raw_drift.predictions_analyzed,
                top_drifting_features=top_drifting,
                last_check_at=now_naive,
            )
    except Exception:
        pass

    # Calibration
    calibration_section: Optional[ModelCardCalibrationSummary] = None
    try:
        raw_cal = _build_calibration_section(name, version, pairs)
        if raw_cal:
            calibration_section = ModelCardCalibrationSummary(
                calibration_status=raw_cal.calibration_status,
                brier_score=raw_cal.brier_score,
                sample_size=raw_cal.sample_size,
            )
    except Exception:
        pass

    # Feature importance (SHAP — limited to last_n=50 for speed)
    feature_importance_section: Optional[ModelCardFeatureImportanceSummary] = None
    try:
        raw_fi = await _build_feature_importance_section(db, name, metadata, predictions, last_n=50)
        if raw_fi and raw_fi.feature_importance:
            ranked = sorted(raw_fi.feature_importance.items(), key=lambda kv: kv[1].rank)[:5]
            feature_importance_section = ModelCardFeatureImportanceSummary(
                top_features=[
                    ModelCardTopFeature(feature=feat, mean_abs_shap=item.mean_abs_shap)
                    for feat, item in ranked
                ],
                sample_size=raw_fi.sample_size,
            )
    except Exception:
        pass

    # Retrain info
    retrain_section: Optional[ModelCardRetrainInfo] = None
    try:
        ts = metadata.training_stats or {}
        last_retrain: Any = ts.get("trained_at") or metadata.training_date
        if isinstance(last_retrain, str):
            last_retrain = datetime.fromisoformat(last_retrain.replace("Z", "+00:00"))
        schedule = metadata.retrain_schedule or {}
        next_run_raw = schedule.get("next_run_at")
        next_run: Optional[datetime] = None
        if next_run_raw:
            if isinstance(next_run_raw, str):
                next_run = datetime.fromisoformat(next_run_raw.replace("Z", "+00:00"))
            elif isinstance(next_run_raw, datetime):
                next_run = next_run_raw
        retrain_section = ModelCardRetrainInfo(
            last_retrain_date=last_retrain,
            trained_by=metadata.trained_by,
            n_rows_trained=ts.get("n_rows"),
            next_run_at=next_run,
        )
    except Exception:
        pass

    # Coverage
    coverage_section: Optional[ModelCardCoverage] = None
    try:
        coverage_section = ModelCardCoverage(
            coverage_rate=coverage_stats["coverage_rate"],
            labeled_count=coverage_stats["labeled_count"],
            total_predictions=coverage_stats["total_predictions"],
        )
    except Exception:
        pass

    card = ModelCardResponse(
        model_name=name,
        version=version,
        generated_at=now_naive,
        algorithm=metadata.algorithm,
        accuracy=metadata.accuracy,
        f1_score=metadata.f1_score,
        tags=metadata.tags,
        classes=metadata.classes,
        features_count=metadata.features_count,
        trained_by=metadata.trained_by,
        training_dataset=metadata.training_dataset,
        created_at=metadata.created_at,
        is_production=metadata.is_production,
        performance=performance_section,
        drift=drift_section,
        calibration=calibration_section,
        feature_importance=feature_importance_section,
        retrain=retrain_section,
        coverage=coverage_section,
    )

    accept = request.headers.get("accept", "application/json")
    if "text/markdown" in accept:
        md_content = _build_model_card_markdown(card)
        filename = f"{name}_{version}_model_card.md"
        return Response(
            content=md_content,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    return Response(content=card.model_dump_json(), media_type="application/json")


@router.get(
    "/models/{name}/golden-tests",
    response_model=List[GoldenTestResponse],
)
async def list_golden_tests(
    name: ModelNamePath,
    _user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """Lists the registered golden test cases for a model."""
    tests = await GoldenTestService.get_tests(db, name)
    return [
        GoldenTestResponse(
            id=t.id,
            model_name=t.model_name,
            input_features=t.input_features,
            expected_output=t.expected_output,
            description=t.description,
            created_at=t.created_at,
            created_by_user_id=t.created_by_user_id,
        )
        for t in tests
    ]


@router.get("/models/{name}/{version}", response_model=ModelGetResponse)
async def get_model(
    name: ModelNamePath,
    version: ModelVersionPath,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the full metadata of a model (name + version).

    Attempts to load the model into memory (from MLflow or MinIO):
    - If loading succeeds: `model_loaded=true`, `model_type` and `feature_names` are set.
    - If loading fails: `model_loaded=false` and `load_instructions` contains
      the information needed to load the model manually in Python.
    """
    result = await db.execute(
        select(ModelMetadata)
        .options(selectinload(ModelMetadata.creator))
        .where(and_(ModelMetadata.name == name, ModelMetadata.version == version))
    )
    model_meta = result.scalar_one_or_none()

    if not model_meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' version '{version}' not found.",
        )

    # Attempt to load the model
    model_loaded = False
    model_type = None
    feature_names = None
    load_instructions = None

    try:
        cached = await model_service.load_model(db, name, version)
        ml_model = cached["model"]
        model_loaded = True
        model_type = type(ml_model).__name__
        if hasattr(ml_model, "feature_names_in_"):
            feature_names = list(ml_model.feature_names_in_)
    except Exception:
        # The model could not be loaded — build manual instructions
        if model_meta.mlflow_run_id:
            load_instructions = {
                "source": "mlflow",
                "run_id": model_meta.mlflow_run_id,
                "python_code": (
                    f"import mlflow.sklearn\n"
                    f"model = mlflow.sklearn.load_model('runs:/{model_meta.mlflow_run_id}/model')"
                ),
            }
        elif model_meta.minio_object_key:
            load_instructions = {
                "source": "minio",
                "bucket": model_meta.minio_bucket,
                "object_key": model_meta.minio_object_key,
                "python_code": (
                    f"from minio import Minio\n"
                    f"import io, joblib\n"
                    f"client = Minio('localhost:9002', access_key='...', secret_key='...', secure=False)\n"
                    f"response = client.get_object('{model_meta.minio_bucket}', '{model_meta.minio_object_key}')\n"
                    f"model = joblib.load(io.BytesIO(response.read()))"
                ),
            }

    return ModelGetResponse(
        id=model_meta.id,
        name=model_meta.name,
        version=model_meta.version,
        description=model_meta.description,
        algorithm=model_meta.algorithm,
        features_count=model_meta.features_count,
        classes=model_meta.classes,
        training_params=model_meta.training_params,
        training_metrics=model_meta.training_metrics,
        hyperparameters=model_meta.hyperparameters,
        training_dataset=model_meta.training_dataset,
        trained_by=model_meta.trained_by,
        training_date=model_meta.training_date,
        accuracy=model_meta.accuracy,
        f1_score=model_meta.f1_score,
        precision=model_meta.precision,
        recall=model_meta.recall,
        confidence_threshold=model_meta.confidence_threshold,
        feature_baseline=model_meta.feature_baseline,
        categorical_baseline=model_meta.categorical_baseline,
        feature_importances=model_meta.feature_importances,
        tags=model_meta.tags,
        webhook_url=model_meta.webhook_url,
        mlflow_run_id=model_meta.mlflow_run_id,
        minio_bucket=model_meta.minio_bucket,
        minio_object_key=model_meta.minio_object_key,
        file_size_bytes=model_meta.file_size_bytes,
        file_hash=model_meta.file_hash,
        user_id_creator=model_meta.user_id_creator,
        creator_username=model_meta.creator.username if model_meta.creator else None,
        is_active=model_meta.is_active,
        is_production=model_meta.is_production,
        traffic_weight=model_meta.traffic_weight,
        deployment_mode=model_meta.deployment_mode,
        promotion_policy=model_meta.promotion_policy,
        alert_thresholds=model_meta.alert_thresholds,
        created_at=model_meta.created_at,
        updated_at=model_meta.updated_at,
        deprecated_at=model_meta.deprecated_at,
        model_loaded=model_loaded,
        model_type=model_type,
        feature_names=feature_names,
        load_instructions=load_instructions,
    )


@router.post("/models", response_model=ModelCreateResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def create_model(
    request: Request,
    name: str = Form(..., description="Unique model name"),
    version: str = Form(..., description="Model version (e.g. 1.0.0)"),
    file: Optional[UploadFile] = File(
        None, description="Model file joblib/pkl (optional if mlflow_run_id is provided)"
    ),
    description: Optional[str] = Form(None),
    algorithm: Optional[str] = Form(None),
    mlflow_run_id: Optional[str] = Form(None),
    accuracy: Optional[float] = Form(None),
    auc: Optional[float] = Form(None, description="AUC-ROC (0–1, classification only)"),
    f1_score: Optional[float] = Form(None),
    features_count: Optional[int] = Form(None),
    classes: Optional[str] = Form(None, description="JSON array ex: [0, 1, 2]"),
    training_params: Optional[str] = Form(None, description="JSON object"),
    training_metrics: Optional[str] = Form(
        None,
        description="JSON object — training metrics (precision, recall, mae, rmse, r2…)",
    ),
    hyperparameters: Optional[str] = Form(
        None,
        description='JSON object — model hyperparameters (e.g. {"n_estimators": 200, "max_depth": 10})',
    ),
    training_dataset: Optional[str] = Form(None),
    feature_baseline: Optional[str] = Form(
        None,
        description='JSON: {"feature": {"mean": float, "std": float, "min": float, "max": float}}',
    ),
    tags: Optional[str] = Form(
        None, description='JSON array of tags e.g. ["production", "finance"]'
    ),
    webhook_url: Optional[str] = Form(None, description="POST callback URL after each prediction"),
    train_file: Optional[UploadFile] = File(
        None,
        description=(
            "Python retraining script (train.py). "
            "Must reference TRAIN_START_DATE, TRAIN_END_DATE, OUTPUT_MODEL_PATH "
            "and contain a joblib.dump/save_model call."
        ),
    ),
    parent_version: Optional[str] = Form(
        None,
        description="Parent version this model is derived from (lineage traceability).",
    ),
    auto_baseline: bool = Form(
        False,
        description=(
            "If True, automatically computes and saves the feature baseline "
            "from existing predictions for this model name (30-day window). "
            "Silently ignored if fewer than 100 predictions are available."
        ),
    ),
    run_training: bool = Form(
        False,
        description=(
            "If True, executes train.py in a subprocess during upload "
            "to generate a model consistent with the API's library versions. "
            "Default False: the uploaded file is stored as-is without retraining."
        ),
    ),
    local_dependencies: Optional[str] = Form(
        None,
        description=(
            "JSON {package: version} captured from the local training environment "
            '(e.g. {"scikit-learn": "1.6.1", "numpy": "2.2.5"}). '
            "Used to generate requirements.txt when run_training=False, "
            "reflecting the versions of the machine that produced the model."
        ),
    ),
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Creates a new model and registers it in the database.

    - **name** + **version** must be unique — returns 409 if the combination already exists.
    - **file**: model file (`.joblib`) required if `mlflow_run_id` is not provided.
      If `mlflow_run_id` is provided, MLflow already stores the model in MinIO — no duplication.
    - **mlflow_run_id**: MLflow run ID. Allows loading the model via MLflow at prediction time.
    - **train_file**: optional `train.py` script enabling automatic retraining.
      Must comply with the interface contract (TRAIN_START_DATE, TRAIN_END_DATE, OUTPUT_MODEL_PATH).

    Requires an admin Bearer token.
    """
    # Verify name + version uniqueness
    result = await db.execute(
        select(ModelMetadata).where(
            and_(ModelMetadata.name == name, ModelMetadata.version == version)
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A model '{name}' version '{version}' already exists.",
        )

    minio_bucket = None
    minio_object_key = None
    file_size_bytes = None
    model_hmac_signature = None

    model_bytes: Optional[bytes] = None
    if file is not None:
        # Read and validate the file (upload deferred after subprocess)
        model_bytes = await file.read()
        max_bytes = settings.MAX_MODEL_SIZE_MB * 1024 * 1024
        if len(model_bytes) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File exceeds the maximum allowed size "
                    f"({settings.MAX_MODEL_SIZE_MB} MB). "
                    f"Received size: {len(model_bytes) / 1024 / 1024:.1f} MB."
                ),
            )
        if not model_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The file is empty.",
            )
    elif not mlflow_run_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide a model file (.joblib) or a mlflow_run_id.",
        )

    # --- Training script processing (optional) ---
    train_script_object_key = None
    requirements_object_key = None
    subprocess_model_bytes: Optional[bytes] = None
    if train_file is not None:
        train_bytes = await train_file.read()
        if not train_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The train.py file is empty.",
            )
        try:
            train_source = train_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="train.py must be a valid UTF-8 text file.",
            )
        validation_error = _validate_train_script(train_source)
        if validation_error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid train.py script: {validation_error}",
            )
        train_object_name = f"{name}/{version}/train.py"
        _t0 = time.perf_counter()
        minio_service.upload_file_bytes(
            train_bytes, train_object_name, content_type="text/x-python"
        )
        logger.info(
            "train.py script uploaded",
            object_name=train_object_name,
            model=name,
            version=version,
            duration_ms=round((time.perf_counter() - _t0) * 1000),
        )
        train_script_object_key = train_object_name

        from src.services.env_snapshot_service import generate_requirements_txt

        if run_training:
            # Launch subprocess to capture dependencies and generate the Docker pkl.
            # If subprocess succeeds, its pkl replaces the user-uploaded pkl
            # (the Docker pkl is trained in the API environment — consistent library versions).
            req_txt, subprocess_model_bytes = await _run_train_subprocess(train_source)
            if req_txt is None:
                logger.info(
                    "AST fallback for requirements.txt — subprocess without 'dependencies'",
                    model=name,
                    version=version,
                )
                req_txt = generate_requirements_txt(train_source)
        else:
            logger.info(
                "run_training=False — subprocess skipped, uploaded model used as-is",
                model=name,
                version=version,
            )
            if local_dependencies:
                try:
                    local_deps_dict = json.loads(local_dependencies)
                    from src.services.env_snapshot_service import dependencies_to_requirements_txt

                    req_txt = dependencies_to_requirements_txt(local_deps_dict)
                    logger.info(
                        "requirements.txt generated from local_dependencies",
                        packages=list(local_deps_dict.keys()),
                        model=name,
                        version=version,
                    )
                except (json.JSONDecodeError, Exception):
                    req_txt = generate_requirements_txt(train_source)
            else:
                req_txt = generate_requirements_txt(train_source)

        req_object_name = f"{name}/{version}/requirements.txt"
        _t0 = time.perf_counter()
        minio_service.upload_file_bytes(
            req_txt.encode("utf-8"), req_object_name, content_type="text/plain"
        )
        logger.info(
            "requirements.txt uploaded",
            object_name=req_object_name,
            model=name,
            version=version,
            duration_ms=round((time.perf_counter() - _t0) * 1000),
        )
        # Fix : sauvegarder la clé MinIO du requirements.txt en DB
        requirements_object_key = req_object_name

    # If local_dependencies provided without train_file, still generate requirements.txt
    if local_dependencies and train_file is None and requirements_object_key is None:
        try:
            local_deps_dict = json.loads(local_dependencies)
            from src.services.env_snapshot_service import dependencies_to_requirements_txt

            req_txt = dependencies_to_requirements_txt(local_deps_dict)
            req_object_name = f"{name}/{version}/requirements.txt"
            minio_service.upload_file_bytes(
                req_txt.encode("utf-8"), req_object_name, content_type="text/plain"
            )
            requirements_object_key = req_object_name
            logger.info(
                "requirements.txt generated from local_dependencies (without train_file)",
                packages=list(local_deps_dict.keys()),
                model=name,
                version=version,
            )
        except Exception:
            pass

    # Choose which model bytes to store: subprocess takes priority, otherwise user file
    model_bytes_to_store = (
        subprocess_model_bytes if subprocess_model_bytes is not None else model_bytes
    )
    if subprocess_model_bytes is not None:
        logger.info("Subprocess model used (Docker env)", model=name, version=version)
    elif model_bytes is not None:
        logger.info("User model uploaded (no subprocess)", model=name, version=version)

    # Upload the final model to MinIO
    if model_bytes_to_store is not None:
        object_name = f"{name}/{version}/model.joblib"
        logger.info(
            "Starting model upload to MinIO",
            model=name,
            version=version,
            size_kb=round(len(model_bytes_to_store) / 1024),
        )
        _t0 = time.perf_counter()
        model_hmac_signature = compute_model_hmac(model_bytes_to_store)
        logger.info(
            "HMAC computed",
            model=name,
            version=version,
            duration_ms=round((time.perf_counter() - _t0) * 1000),
        )
        _t1 = time.perf_counter()
        upload_info = minio_service.upload_model_bytes(model_bytes_to_store, object_name)
        logger.info(
            "Model uploaded to MinIO",
            model=name,
            version=version,
            duration_ms=round((time.perf_counter() - _t1) * 1000),
        )
        minio_bucket = upload_info["bucket"]
        minio_object_key = object_name
        file_size_bytes = upload_info["size"]

    # Detect task type and extract feature importances from model bytes
    detected_task: Optional[str] = None
    extracted_importances: Optional[dict] = None
    if model_bytes_to_store is not None:
        detected_task = _detect_task_type(model_bytes_to_store)
        extracted_importances = _extract_feature_importances(model_bytes_to_store)
        if extracted_importances:
            logger.info(
                "Feature importances extracted",
                model=name,
                version=version,
                n_features=len(extracted_importances),
            )

    # Deserialize optional JSON fields
    classes_parsed = _parse_json_field(classes, "classes")
    training_params_parsed = _parse_json_field(training_params, "training_params")
    training_metrics_parsed = _parse_json_field(training_metrics, "training_metrics")
    hyperparameters_parsed = _parse_json_field(hyperparameters, "hyperparameters")
    feature_baseline_parsed = _parse_json_field(feature_baseline, "feature_baseline")
    tags_parsed = _parse_json_field(tags, "tags")

    # Create the database entry
    metadata = ModelMetadata(
        name=name,
        version=version,
        minio_bucket=minio_bucket,
        minio_object_key=minio_object_key,
        file_size_bytes=file_size_bytes,
        model_hmac_signature=model_hmac_signature,
        description=description,
        algorithm=algorithm,
        mlflow_run_id=mlflow_run_id,
        accuracy=accuracy,
        auc=auc if auc is not None else (
            training_metrics_parsed.get("roc_auc") if training_metrics_parsed else None
        ),
        f1_score=f1_score,
        features_count=features_count,
        classes=classes_parsed,
        training_params=training_params_parsed,
        training_metrics=training_metrics_parsed,
        hyperparameters=hyperparameters_parsed,
        training_dataset=training_dataset,
        feature_baseline=feature_baseline_parsed,
        feature_importances=extracted_importances,
        tags=tags_parsed,
        webhook_url=webhook_url,
        train_script_object_key=train_script_object_key,
        requirements_object_key=requirements_object_key,
        trained_by=user.username,
        user_id_creator=user.id,
        is_active=True,
        is_production=False,
        parent_version=parent_version,
        model_task=detected_task,
    )
    db.add(metadata)
    await db.flush()  # get the id before the snapshot
    await DBService.log_model_history(
        db, metadata, HistoryActionType.CREATED, user.id, user.username
    )
    await db.commit()
    await db.refresh(metadata)
    audit_log("model.upload", actor_id=user.id, resource=f"{name}:{version}")

    if auto_baseline:
        try:
            production_stats = await DBService.get_feature_production_stats(
                db, name, model_version=None, days=30
            )
            predictions_used = min((v["count"] for v in production_stats.values()), default=0)
            if predictions_used >= 100:
                computed_baseline = {
                    feat: {
                        "mean": round(s["mean"], 6),
                        "std": round(s["std"], 6),
                        "min": round(s["min"], 6),
                        "max": round(s["max"], 6),
                        "null_rate": round(s.get("null_rate", 0.0), 6),
                    }
                    for feat, s in production_stats.items()
                }
                metadata.feature_baseline = computed_baseline
                await DBService.log_model_history(
                    db,
                    metadata,
                    HistoryActionType.UPDATED,
                    user.id,
                    user.username,
                    ["feature_baseline"],
                )
                await db.commit()
                await db.refresh(metadata)
                logger.info(
                    "Baseline auto-computed at upload",
                    model=name,
                    version=version,
                    features=list(computed_baseline.keys()),
                    predictions_used=predictions_used,
                )
        except Exception:
            logger.warning("Auto-baseline failed at upload", model=name, version=version)

    return ModelCreateResponse(
        **{c.name: getattr(metadata, c.name) for c in metadata.__table__.columns},
        creator_username=user.username,
    )


@router.patch("/models/{name}/{version}", response_model=ModelCreateResponse)
async def update_model(
    name: ModelNamePath,
    version: ModelVersionPath,
    payload: ModelUpdateInput,
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Updates the metadata of a model (name + version).

    Editable fields: `description`, `is_production`, `accuracy`, `features_count`, `classes`.

    - If **is_production** is set to `true`, all other versions of the same model
      are automatically set to `false`.

    Requires a valid Bearer token.
    """
    # Retrieve the target model with its creator
    result = await db.execute(
        select(ModelMetadata)
        .options(selectinload(ModelMetadata.creator))
        .where(and_(ModelMetadata.name == name, ModelMetadata.version == version))
    )
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' version '{version}' not found.",
        )

    # Snapshot before modification (to detect actually changed fields)
    pre_snapshot = _build_snapshot(model)
    demoted_versions = []

    # If is_production becomes True → remove is_production from other versions,
    # except those in ab_test (they share production traffic simultaneously).
    # The current version inherits the deployment_mode from the payload or its current state.
    incoming_mode = (
        payload.deployment_mode if payload.deployment_mode is not None else model.deployment_mode
    )
    if payload.is_production is True:
        other_versions = await db.execute(
            select(ModelMetadata).where(
                and_(
                    ModelMetadata.name == name,
                    ModelMetadata.version != version,
                    ModelMetadata.is_production.is_(True),
                )
            )
        )
        for other in other_versions.scalars().all():
            # Keep is_production if the other version is in ab_test AND the current
            # version will also be in ab_test (legal A/B coexistence)
            if (
                other.deployment_mode == DeploymentMode.AB_TEST
                and incoming_mode == DeploymentMode.AB_TEST
            ):
                continue
            other.is_production = False
            demoted_versions.append(other)

    # Apply only provided fields (non-None)
    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if field == "training_stats" and isinstance(value, dict):
            # Merge with existing to avoid overwriting last retrain data
            merged = {**(model.training_stats or {}), **value}
            setattr(model, field, merged)
        else:
            setattr(model, field, value)

    await db.flush()

    # Validation: the sum of traffic_weight for the model's ab_test versions must stay ≤ 1.0
    if "deployment_mode" in update_data or "traffic_weight" in update_data:
        ab_result = await db.execute(
            select(ModelMetadata).where(
                and_(
                    ModelMetadata.name == name,
                    ModelMetadata.deployment_mode == DeploymentMode.AB_TEST,
                    ModelMetadata.is_active.is_(True),
                )
            )
        )
        ab_versions = ab_result.scalars().all()
        total_weight = sum((m.traffic_weight or 0.0) for m in ab_versions)
        if total_weight > 1.0 + 1e-9:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"The sum of traffic_weight for A/B versions of '{name}' "
                    f"exceeds 1.0 (current sum = {total_weight:.3f}). "
                    "Reduce the weight of some versions before increasing others."
                ),
            )

    # Determine actually changed fields
    post_snapshot = _build_snapshot(model)
    changed_fields = [k for k in update_data if pre_snapshot.get(k) != post_snapshot.get(k)]
    action = (
        HistoryActionType.SET_PRODUCTION
        if payload.is_production is True
        else HistoryActionType.UPDATED
    )

    await DBService.log_model_history(db, model, action, user.id, user.username, changed_fields)
    for demoted in demoted_versions:
        await DBService.log_model_history(
            db, demoted, HistoryActionType.SET_PRODUCTION, user.id, user.username, ["is_production"]
        )

    await db.commit()
    if "is_production" in update_data:
        _leaderboard_cache.clear()
    await db.refresh(model)

    return ModelCreateResponse(
        **{c.name: getattr(model, c.name) for c in model.__table__.columns},
        creator_username=model.creator.username if model.creator else None,
    )


# ---------------------------------------------------------------------------
# Helpers suppression
# ---------------------------------------------------------------------------


def _delete_minio_object(object_key: str) -> bool:
    """Deletes the MinIO object. Returns False if MinIO is unavailable."""
    try:
        return minio_service.delete_model(object_key)
    except Exception as e:
        logger.warning("MinIO deletion failed", object_key=object_key, error=str(e))
        return False


# ---------------------------------------------------------------------------
# DELETE routes
# ---------------------------------------------------------------------------


@router.delete("/models/{name}/{version}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_version(
    name: ModelNamePath,
    version: ModelVersionPath,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Deletes a specific version of a model.

    - Deletes the PostgreSQL database entry.
    - Deletes the associated MLflow run (if `mlflow_run_id` is set).
    - Deletes the `.joblib` object in MinIO.
    - **Cascade**: deletes all predictions for this version along with
      the observed_results linked to them.

    Returns **204 No Content** on success. Requires an admin Bearer token.
    """
    result = await db.execute(
        select(ModelMetadata).where(
            and_(ModelMetadata.name == name, ModelMetadata.version == version)
        )
    )
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' version '{version}' not found.",
        )

    if model.mlflow_run_id:
        mlflow_service.delete_run(model.mlflow_run_id)

    if model.minio_object_key:
        _delete_minio_object(model.minio_object_key)

    # Cascade: delete predictions + observed_results for this version
    cascade = await DBService.delete_predictions_for_version(db, name, version)

    # Log the deletion before deleting the ORM object
    was_production = model.is_production
    await DBService.log_model_history(db, model, HistoryActionType.DELETED, user.id, user.username)
    await db.delete(model)
    await db.commit()
    if was_production:
        _leaderboard_cache.clear()
    audit_log(
        "model.delete",
        actor_id=user.id,
        resource=f"{name}:{version}",
        details={
            "cascade_predictions": cascade["deleted_predictions"],
            "cascade_observed_results": cascade["deleted_observed_results"],
        },
    )


@router.post("/models/{name}/{version}/validate-input", response_model=ValidateInputResponse)
async def validate_model_input(
    name: ModelNamePath,
    version: ModelVersionPath,
    features: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    _auth: User = Depends(verify_token),
):
    """
    Validates input features against the expected schema of a model version.

    - Detects **missing features** (present in the model, absent from the request).
    - Detects **unexpected features** (present in the request, absent from the model).
    - Emits **coercion warnings** for string values convertible to float.

    The source of truth is, in priority order:
    1. `feature_names_in_` from the loaded sklearn model (trained on a pandas DataFrame).
    2. The keys of `feature_baseline` stored in the model metadata.

    If no schema is available, returns `expected_features: null` with `valid: true`
    (cannot validate without a reference).

    Requires a valid Bearer token.
    """
    # Retrieve metadata to access feature_baseline as fallback
    metadata = await DBService.get_model_metadata(db, name, version)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' version '{version}' not found.",
        )

    # Resolve expected features: loaded model takes priority, baseline as fallback
    loaded_model = None
    try:
        model_data = await model_service.load_model(db, name, version)
        loaded_model = model_data["model"]
    except HTTPException:
        pass

    expected_features = resolve_expected_features(loaded_model, metadata.feature_baseline)

    # No schema available — validation not possible
    if expected_features is None:
        return ValidateInputResponse(
            valid=True,
            errors=[],
            warnings=[],
            expected_features=None,
        )

    errors, warnings = validate_input_features(features, expected_features)

    return ValidateInputResponse(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        expected_features=sorted(expected_features),
    )


@router.post("/models/{name}/{version}/warmup", response_model=WarmupResponse)
async def warmup_model(
    name: ModelNamePath,
    version: ModelVersionPath,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Warms up the Redis cache for a model by loading it proactively.

    Eliminates cold-start latency (MinIO download + pickle deserialization) on
    the first prediction after a deployment or restart, thus avoiding
    a false latency asymmetry in A/B comparisons.

    Returns `already_cached: true` if the model was already in the memory cache.
    """
    cache_key = f"{name}:{version}"
    cached_before = await model_service.get_cached_models()
    already_cached = cache_key in cached_before

    t0 = time.monotonic()
    try:
        await model_service.load_model(db, name, version)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Model warmup error", model_name=name, version=version, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error warming up model '{name}' v'{version}'.",
        )
    load_time_ms = (time.monotonic() - t0) * 1000

    logger.info(
        "Model warmed up",
        model_name=name,
        version=version,
        already_cached=already_cached,
        load_time_ms=round(load_time_ms, 1),
    )

    return WarmupResponse(
        model_name=name,
        version=version,
        already_cached=already_cached,
        load_time_ms=round(load_time_ms, 1),
        cache_key=cache_key,
    )


@router.post(
    "/models/{name}/{version}/compute-baseline",
    response_model=ComputeBaselineResponse,
)
async def compute_model_baseline(
    name: ModelNamePath,
    version: ModelVersionPath,
    days: int = Query(30, ge=1, le=180, description="Time window in days"),
    dry_run: bool = Query(True, description="Compute without saving (default: True)"),
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Computes a feature baseline from recent production predictions.

    Uses the last `days` days of predictions to compute {mean, std, min, max}
    per numeric feature. The result can be saved as the model's `feature_baseline`
    to enable drift detection.

    - **dry_run=true** (default): computes and returns without saving.
    - **dry_run=false**: saves the baseline and activates drift monitoring.

    Raises a 422 error if the number of available predictions is below 100.

    Reserved for administrators.
    """
    metadata = await DBService.get_model_metadata(db, name, version)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' version '{version}' not found.",
        )

    production_stats = await DBService.get_feature_production_stats(db, name, version, days)

    predictions_used = min((v["count"] for v in production_stats.values()), default=0)
    if predictions_used < 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Insufficient data for a reliable baseline: "
                f"{predictions_used} predictions available (minimum required: 100). "
                f"Increase the time window with ?days=N or wait for more predictions."
            ),
        )

    baseline = {
        feat: {
            "mean": round(s["mean"], 6),
            "std": round(s["std"], 6),
            "min": round(s["min"], 6),
            "max": round(s["max"], 6),
            "null_rate": round(s.get("null_rate", 0.0), 6),
        }
        for feat, s in production_stats.items()
    }

    if not dry_run:
        metadata.feature_baseline = baseline
        await db.flush()
        await DBService.log_model_history(
            db, metadata, HistoryActionType.UPDATED, user.id, user.username, ["feature_baseline"]
        )
        await db.commit()
        logger.info(
            "Baseline computed and saved",
            model=name,
            version=version,
            features=list(baseline.keys()),
            predictions_used=predictions_used,
        )

    return ComputeBaselineResponse(
        model_name=name,
        version=version,
        predictions_used=predictions_used,
        dry_run=dry_run,
        baseline={feat: FeatureStats(**stats) for feat, stats in baseline.items()},
    )


@router.get("/models/{name}/{version}/download")
async def download_model(
    name: ModelNamePath,
    version: ModelVersionPath,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Downloads the .joblib file of a model.

    Admin only — the .joblib contains the model's internal logic.
    """
    result = await db.execute(
        select(ModelMetadata).where(
            and_(ModelMetadata.name == name, ModelMetadata.version == version)
        )
    )
    model_meta = result.scalar_one_or_none()

    if not model_meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' version '{version}' not found.",
        )

    if not model_meta.minio_object_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model file available for '{name}' version '{version}'.",
        )

    try:
        model_bytes = minio_service.download_file_bytes(model_meta.minio_object_key)
    except Exception as e:
        logger.error("Model download error", name=name, version=version, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error downloading the model.",
        )

    filename = f"{name}_{version}.joblib"
    return Response(
        content=model_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/models/{name}/{version}/download-dataset")
async def download_training_dataset(
    name: ModelNamePath,
    version: ModelVersionPath,
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Downloads the training dataset stored in MinIO for a model version.

    The `training_dataset` field must contain the MinIO path (e.g. `iris-classifier/datasets/…csv`).
    Returns 404 if no dataset is available or if the path is not a MinIO path.
    """
    result = await db.execute(
        select(ModelMetadata).where(
            and_(ModelMetadata.name == name, ModelMetadata.version == version)
        )
    )
    model_meta = result.scalar_one_or_none()

    if not model_meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' version '{version}' not found.",
        )

    dataset_path = model_meta.training_dataset
    if not dataset_path or "/" not in dataset_path or not dataset_path.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No training dataset available for this version (missing or invalid MinIO path).",
        )

    try:
        csv_bytes = await minio_service.async_download_file_bytes(dataset_path)
    except Exception as e:
        logger.error(
            "Dataset download error",
            name=name,
            version=version,
            path=dataset_path,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found in MinIO ({dataset_path}).",
        )

    filename = dataset_path.split("/")[-1]
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/models/{name}/{version}/download-script")
async def download_train_script(
    name: ModelNamePath,
    version: ModelVersionPath,
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Downloads the train.py script stored in MinIO for a model version.

    Returns 404 if no script was uploaded with this model.
    """
    result = await db.execute(
        select(ModelMetadata).where(
            and_(ModelMetadata.name == name, ModelMetadata.version == version)
        )
    )
    model_meta = result.scalar_one_or_none()

    if not model_meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' version '{version}' not found.",
        )

    script_key = model_meta.train_script_object_key
    if not script_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No training script available for this version.",
        )

    try:
        script_bytes = await minio_service.async_download_file_bytes(script_key)
    except Exception as e:
        logger.error(
            "Script download error",
            name=name,
            version=version,
            path=script_key,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Script not found in MinIO ({script_key}).",
        )

    filename = script_key.split("/")[-1]
    return Response(
        content=script_bytes,
        media_type="text/x-python",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/models/{name}", response_model=ModelDeleteResponse)
async def delete_model_all_versions(
    name: ModelNamePath,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Deletes all versions of a model.

    - Deletes all PostgreSQL entries for this name.
    - Deletes each associated MLflow run.
    - Deletes each `.joblib` object in MinIO.
    - **Cascade**: deletes predictions and observed_results for each version.

    Returns a summary of the deletions performed. Requires an admin Bearer token.
    """
    result = await db.execute(select(ModelMetadata).where(ModelMetadata.name == name))
    models = result.scalars().all()

    if not models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No model found with the name '{name}'.",
        )

    deleted_versions = []
    mlflow_runs_deleted = []
    minio_objects_deleted = []
    total_cascade_predictions = 0
    total_cascade_observed = 0

    for model in models:
        deleted_versions.append(model.version)

        if model.mlflow_run_id and mlflow_service.delete_run(model.mlflow_run_id):
            mlflow_runs_deleted.append(model.mlflow_run_id)

        if model.minio_object_key and _delete_minio_object(model.minio_object_key):
            minio_objects_deleted.append(model.minio_object_key)

        # Cascade per version
        cascade = await DBService.delete_predictions_for_version(db, name, model.version)
        total_cascade_predictions += cascade["deleted_predictions"]
        total_cascade_observed += cascade["deleted_observed_results"]

        await DBService.log_model_history(
            db, model, HistoryActionType.DELETED, user.id, user.username
        )
        await db.delete(model)

    await db.commit()
    audit_log(
        "model.delete_all",
        actor_id=user.id,
        resource=name,
        details={
            "versions": deleted_versions,
            "cascade_predictions": total_cascade_predictions,
            "cascade_observed_results": total_cascade_observed,
        },
    )

    return ModelDeleteResponse(
        name=name,
        deleted_versions=deleted_versions,
        mlflow_runs_deleted=mlflow_runs_deleted,
        minio_objects_deleted=minio_objects_deleted,
    )


# ---------------------------------------------------------------------------
# Golden Test Set
# ---------------------------------------------------------------------------


@router.post(
    "/models/{name}/golden-tests/upload-csv",
    status_code=status.HTTP_201_CREATED,
)
async def upload_golden_tests_csv(
    name: ModelNamePath,
    file: UploadFile = File(...),
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Imports a batch of golden test cases from a CSV (admin only).

    Expected CSV format: feature columns + ``expected_output`` (required) + ``description`` (optional).
    Example: ``sepal_length,sepal_width,petal_length,petal_width,expected_output,description``
    """
    content = await file.read()
    try:
        rows = GoldenTestService.parse_csv(content)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    created = 0
    errors = []
    for i, row in enumerate(rows):
        try:
            await GoldenTestService.create_test(
                db,
                model_name=name,
                input_features=row["input_features"],
                expected_output=row["expected_output"],
                description=row.get("description"),
                user_id=user.id,
            )
            created += 1
        except Exception as exc:
            errors.append({"row": i + 2, "error": str(exc)})

    await db.commit()
    return {"created": created, "errors": errors}


@router.post(
    "/models/{name}/golden-tests",
    response_model=GoldenTestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_golden_test(
    name: ModelNamePath,
    payload: GoldenTestCreate,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Creates a golden test case for a model (admin only)."""
    gt = await GoldenTestService.create_test(
        db,
        model_name=name,
        input_features=payload.input_features,
        expected_output=payload.expected_output,
        description=payload.description,
        user_id=user.id,
    )
    await db.commit()
    return GoldenTestResponse(
        id=gt.id,
        model_name=gt.model_name,
        input_features=gt.input_features,
        expected_output=gt.expected_output,
        description=gt.description,
        created_at=gt.created_at,
        created_by_user_id=gt.created_by_user_id,
    )


@router.delete(
    "/models/{name}/golden-tests/{test_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_golden_test(
    name: ModelNamePath,
    test_id: int,
    _user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Deletes a golden test case (admin only)."""
    deleted = await GoldenTestService.delete_test(db, test_id, name)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test case {test_id} not found for model '{name}'.",
        )
    await db.commit()


@router.post(
    "/models/{name}/{version}/golden-tests/run",
    response_model=GoldenTestRunResponse,
)
async def run_golden_tests(
    name: ModelNamePath,
    version: ModelVersionPath,
    _user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Runs all registered golden tests for a given model and version (admin only).

    Loads the model, predicts each test case and compares against the expected output.
    """
    return await GoldenTestService.run_tests(db, name, version)
