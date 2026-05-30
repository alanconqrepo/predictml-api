"""
Prediction endpoints
"""

import asyncio
import csv
import io
import json
import os
import random
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.ml_metrics import inference_duration_seconds, predictions_total
from src.core.rate_limit import limiter
from src.core.security import check_prediction_rate_limit, require_admin, verify_token
from src.db.database import AsyncSessionLocal, get_db, get_read_db
from src.db.models import Prediction, User
from src.db.models.user import UserRole
from src.schemas.prediction import (
    AnomaliesResponse,
    AnomalyFeatureDetail,
    AnomalyPredictionEntry,
    BatchPredictionInput,
    BatchPredictionOutput,
    BatchPredictionResultItem,
    ExplainInput,
    ExplainOutput,
    PredictionInput,
    PredictionOutput,
    PredictionResponse,
    PredictionsListResponse,
    PredictionStatsItem,
    PredictionStatsResponse,
    PurgeResponse,
    UnlabeledPredictionItem,
    UnlabeledPredictionsResponse,
)
from src.services.db_service import DBService
from src.services.input_validation_service import resolve_expected_features, validate_input_features
from src.services.model_service import model_service
from src.services.shap_service import compute_shap_explanation
from src.services.webhook_service import send_webhook

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["predictions"])

# Limits the number of concurrent shadow predictions to avoid saturating the connection pool
_SHADOW_SEMAPHORE = asyncio.Semaphore(int(os.getenv("SHADOW_CONCURRENCY", "20")))
# In batch mode, sample at most N items per shadow version
_MAX_SHADOW_ITEMS_PER_BATCH = int(os.getenv("MAX_SHADOW_ITEMS_PER_BATCH", "50"))


async def _run_shadow_prediction(
    model_name: str,
    shadow_version: str,
    features: dict,
    id_obs: Optional[str],
    user_id: int,
    client_ip: Optional[str],
    user_agent: Optional[str],
    timestamp: Optional[datetime] = None,
) -> None:
    """
    Runs the shadow model prediction in the background and persists it with is_shadow=True.
    All exceptions are caught and logged — never propagates to the client.
    """
    import time as _time

    async with _SHADOW_SEMAPHORE:
        start = _time.time()
        try:
            async with AsyncSessionLocal() as db:
                shadow_data = await model_service.load_model(db, model_name, shadow_version)
                shadow_model = shadow_data["model"]
                shadow_meta = shadow_data["metadata"]

                if not hasattr(shadow_model, "feature_names_in_"):
                    raise ValueError(
                        f"Shadow model '{model_name}:{shadow_version}' does not have feature_names_in_"
                    )

                x = pd.DataFrame([{n: features[n] for n in shadow_model.feature_names_in_}])
                raw = shadow_model.predict(x)[0]
                result = raw.item() if hasattr(raw, "item") else raw
                proba = (
                    shadow_model.predict_proba(x)[0].tolist()
                    if hasattr(shadow_model, "predict_proba")
                    else None
                )
                rt_ms = (_time.time() - start) * 1000

                _payload = {
                    "user_id": user_id,
                    "model_name": shadow_meta.name,
                    "model_version": shadow_meta.version,
                    "input_features": features,
                    "prediction_result": result,
                    "probabilities": proba,
                    "response_time_ms": rt_ms,
                    "client_ip": client_ip,
                    "user_agent": user_agent,
                    "status": "success",
                    "id_obs": id_obs,
                    "is_shadow": True,
                    "max_confidence": max(proba) if proba else None,
                    "timestamp": timestamp,
                }
                if settings.PREDICTION_STREAM_ENABLED:
                    _published = await _publish_prediction_to_stream(_payload)
                    if not _published:
                        await DBService.create_prediction(db=db, **_payload)
                else:
                    await DBService.create_prediction(db=db, **_payload)

                logger.info(
                    "Shadow prediction recorded",
                    model_name=model_name,
                    version=shadow_version,
                    result=result,
                )

        except Exception as e:
            logger.warning(
                "Shadow prediction failed (non-blocking)",
                model_name=model_name,
                version=shadow_version,
                error=str(e),
            )


async def _publish_prediction_to_stream(payload: dict) -> bool:
    """Publishes the prediction payload to Redis Stream. Returns True on success, False otherwise."""
    try:
        redis = await model_service._get_redis()
        flat: dict = {
            "user_id": str(payload["user_id"]),
            "model_name": payload["model_name"],
            "model_version": payload.get("model_version") or "",
            "id_obs": payload.get("id_obs") or "",
            "input_features": json.dumps(payload["input_features"], default=str),
            "prediction_result": json.dumps(payload["prediction_result"], default=str),
            "probabilities": (
                json.dumps(payload["probabilities"]) if payload.get("probabilities") else ""
            ),
            "response_time_ms": str(payload["response_time_ms"]),
            "client_ip": payload.get("client_ip") or "",
            "user_agent": payload.get("user_agent") or "",
            "status": payload.get("status", "success"),
            "error_message": payload.get("error_message") or "",
            "is_shadow": "true" if payload.get("is_shadow") else "false",
            "max_confidence": (
                str(payload["max_confidence"]) if payload.get("max_confidence") is not None else ""
            ),
        }
        await redis.xadd(
            settings.PREDICTION_STREAM_NAME,
            flat,
            maxlen=settings.PREDICTION_STREAM_MAXLEN,
            approximate=True,
        )
        return True
    except Exception as exc:
        logger.warning("Redis stream publish failed — falling back to sync", error=str(exc))
        return False


@router.get("/predictions/stats", response_model=PredictionStatsResponse)
async def get_prediction_stats(
    model_name: Optional[str] = Query(None, description="Filter by model name (optional)"),
    days: int = Query(30, ge=1, le=365, description="Time window in days (default: 30, max: 365)"),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Aggregated prediction statistics per model over a sliding window.

    - **model_name**: filter on a single model (optional)
    - **days**: time window in days (default 30, max 365)

    Returns for each model: total, errors, error rate, mean / p50 / p95 response time.

    Requires a valid Bearer token.
    """
    raw = await DBService.get_prediction_stats(db, days=days, model_name=model_name)
    return PredictionStatsResponse(
        days=days,
        model_name=model_name,
        stats=[PredictionStatsItem(**s) for s in raw],
    )


@router.get("/predictions", response_model=PredictionsListResponse)
async def get_predictions(
    name: str = Query(..., description="Model name"),
    start: datetime = Query(
        ..., description="Start of period (ISO 8601, e.g. 2024-01-01T00:00:00)"
    ),
    end: datetime = Query(..., description="End of period (ISO 8601, e.g. 2024-12-31T23:59:59)"),
    version: Optional[str] = Query(None, description="Model version (optional)"),
    user: Optional[str] = Query(None, description="Username (optional)"),
    id_obs: Optional[str] = Query(None, description="Observation identifier (optional)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    cursor: Optional[int] = Query(
        None, ge=1, description="Pagination cursor (id of the last seen prediction)"
    ),
    min_confidence: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum confidence (max of probabilities) — optional, classifiers only",
    ),
    max_confidence: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Maximum confidence (max of probabilities) — optional, classifiers only",
    ),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Returns the prediction history with filters (cursor-based pagination).

    - **name**: model name — required
    - **start** / **end**: datetime range — required
    - **version**: model version — optional
    - **user**: username — optional
    - **id_obs**: observation identifier — optional
    - **limit**: maximum number of results (default: 100, max: 1000)
    - **cursor**: id of the last seen prediction (for the next page, use `next_cursor` from the previous response)
    - **min_confidence**: filter on minimum confidence (max of probabilities), 0.0–1.0 — optional
    - **max_confidence**: filter on maximum confidence (max of probabilities), 0.0–1.0 — optional

    Requires a valid Bearer token.
    """
    if start > end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="'start' must be before 'end'.",
        )

    rows, total = await DBService.get_predictions(
        db=db,
        model_name=name,
        start=start,
        end=end,
        model_version=version,
        username=user,
        id_obs=id_obs,
        limit=limit,
        cursor=cursor,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )

    has_more = len(rows) > limit
    page = rows[:limit]
    next_cursor = page[-1].id if has_more and page else None

    return PredictionsListResponse(
        total=total,
        limit=limit,
        next_cursor=next_cursor,
        predictions=[
            PredictionResponse(
                id=p.id,
                model_name=p.model_name,
                model_version=p.model_version,
                id_obs=p.id_obs,
                input_features=p.input_features,
                prediction_result=p.prediction_result,
                probabilities=p.probabilities,
                max_confidence=p.max_confidence,
                response_time_ms=p.response_time_ms,
                timestamp=p.timestamp,
                status=p.status,
                error_message=p.error_message,
                username=p.user.username if p.user else None,
                is_shadow=p.is_shadow,
            )
            for p in page
        ],
    )


_EXPORT_PAGE_SIZE = 500


@router.get("/predictions/export")
async def export_predictions(
    model_name: Optional[str] = Query(None, description="Filter by model name (optional)"),
    start: datetime = Query(..., description="Start of period (ISO 8601)"),
    end: datetime = Query(..., description="End of period (ISO 8601)"),
    export_format: str = Query(
        "csv",
        alias="format",
        description="Export format: csv, jsonl or parquet (default: csv)",
    ),
    include_features: bool = Query(True, description="Include input_features in the export"),
    pred_status: Optional[str] = Query(
        None, alias="status", description="Filter by status: success or error"
    ),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Bulk export of predictions in CSV, JSONL or Parquet format via cursor-based streaming.

    - **model_name**: filter on a model (optional — all models if absent)
    - **start** / **end**: datetime range — required
    - **format**: `csv` (default), `jsonl` or `parquet`
    - **include_features**: include `input_features` in the export (default: true)
    - **status**: filter by status `success` or `error` (optional)

    Returns a file as a direct download (Content-Disposition: attachment).
    Cursor-based streaming avoids loading the entire history into memory (CSV/JSONL).
    Parquet format loads all rows into memory before serialization.

    Requires a valid Bearer token.
    """
    if start > end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="'start' must be before 'end'.",
        )
    if export_format not in ("csv", "jsonl", "parquet"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'format' parameter must be 'csv', 'jsonl' or 'parquet'.",
        )
    if pred_status is not None and pred_status not in ("success", "error"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'status' parameter must be 'success' or 'error'.",
        )

    fmt = export_format
    csv_cols = [
        "id",
        "timestamp",
        "model_name",
        "model_version",
        "username",
        "id_obs",
        "prediction_result",
        "observed_result",
        "probabilities",
        "response_time_ms",
        "status",
        "error_message",
        "is_shadow",
    ]
    if include_features:
        csv_cols.append("input_features")

    async def _generate():
        cursor: Optional[int] = None
        header_written = False

        while True:
            rows = await DBService.get_predictions_for_export(
                db=db,
                model_name=model_name,
                start=start,
                end=end,
                status_filter=pred_status,
                limit=_EXPORT_PAGE_SIZE,
                cursor=cursor,
            )

            if not rows:
                if fmt == "csv" and not header_written:
                    buf = io.StringIO()
                    csv.writer(buf).writerow(csv_cols)
                    yield buf.getvalue()
                break

            if fmt == "csv":
                if not header_written:
                    buf = io.StringIO()
                    csv.writer(buf).writerow(csv_cols)
                    yield buf.getvalue()
                    header_written = True
                for row, obs_result in rows:
                    buf = io.StringIO()
                    vals = [
                        row.id,
                        row.timestamp.isoformat() if row.timestamp else None,
                        row.model_name,
                        row.model_version,
                        row.user.username if row.user else None,
                        row.id_obs,
                        json.dumps(row.prediction_result),
                        obs_result,
                        json.dumps(row.probabilities),
                        row.response_time_ms,
                        row.status,
                        row.error_message,
                        row.is_shadow,
                    ]
                    if include_features:
                        vals.append(json.dumps(row.input_features))
                    csv.writer(buf).writerow(vals)
                    yield buf.getvalue()
            else:
                for row, obs_result in rows:
                    record: dict = {
                        "id": row.id,
                        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                        "model_name": row.model_name,
                        "model_version": row.model_version,
                        "username": row.user.username if row.user else None,
                        "id_obs": row.id_obs,
                        "prediction_result": row.prediction_result,
                        "observed_result": obs_result,
                        "probabilities": row.probabilities,
                        "response_time_ms": row.response_time_ms,
                        "status": row.status,
                        "error_message": row.error_message,
                        "is_shadow": row.is_shadow,
                    }
                    if include_features:
                        record["input_features"] = row.input_features
                    yield json.dumps(record) + "\n"

            if len(rows) < _EXPORT_PAGE_SIZE:
                break
            cursor = rows[-1][0].id

    if fmt == "parquet":
        all_rows = []
        cursor: Optional[int] = None
        while True:
            rows = await DBService.get_predictions_for_export(
                db=db,
                model_name=model_name,
                start=start,
                end=end,
                status_filter=pred_status,
                limit=_EXPORT_PAGE_SIZE,
                cursor=cursor,
            )
            if not rows:
                break
            for row, obs_result in rows:
                record: dict = {
                    "id": row.id,
                    "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                    "model_name": row.model_name,
                    "model_version": row.model_version,
                    "username": row.user.username if row.user else None,
                    "id_obs": row.id_obs,
                    "prediction_result": json.dumps(row.prediction_result),
                    # observed_result can be float (regression) or str (classification);
                    # serialize as JSON to avoid mixed-type columns in Parquet.
                    "observed_result": json.dumps(obs_result) if obs_result is not None else None,
                    "probabilities": json.dumps(row.probabilities),
                    "response_time_ms": row.response_time_ms,
                    "status": row.status,
                    "error_message": row.error_message,
                    "is_shadow": row.is_shadow,
                }
                if include_features:
                    record["input_features"] = json.dumps(row.input_features)
                all_rows.append(record)
            if len(rows) < _EXPORT_PAGE_SIZE:
                break
            cursor = rows[-1][0].id

        buf = io.BytesIO()
        pd.DataFrame(all_rows, columns=csv_cols).to_parquet(buf, index=False, engine="pyarrow")
        buf.seek(0)
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return Response(
            content=buf.read(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="predictions_{today}.parquet"'},
        )

    media_type = "text/csv" if fmt == "csv" else "application/x-ndjson"
    filename = f"predictions_export.{fmt}"
    return StreamingResponse(
        _generate(),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/predictions/anomalies", response_model=AnomaliesResponse)
async def get_anomalous_predictions(
    model_name: str = Query(..., description="Model name (required)"),
    days: int = Query(7, ge=1, le=90, description="Time window in days (default: 7)"),
    z_threshold: float = Query(
        3.0, ge=0.0, description="Z-score threshold for detection (default: 3.0)"
    ),
    limit: int = Query(
        200, ge=1, le=1000, description="Max predictions to analyze (default: 200, max: 1000)"
    ),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Predictions with anomalous features (z-score >= threshold).

    For each prediction in the time window, computes the z-score per feature
    by comparing to the model baseline: z = |value - baseline_mean| / baseline_std.
    Returns only predictions where at least one feature exceeds z_threshold.

    - **model_name**: model name — required
    - **days**: time window in days (default: 7, max: 90)
    - **z_threshold**: detection threshold (default: 3.0)
    - **limit**: max predictions to analyze (default: 200, max: 1000)

    Returns `error: "no_baseline"` if the model has no feature baseline.

    Requires a valid Bearer token.
    """
    metadata = await DBService.get_model_metadata(db, model_name)
    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found.",
        )

    baseline = metadata.feature_baseline
    if not baseline:
        return AnomaliesResponse(
            model_name=model_name,
            period_days=days,
            z_threshold=z_threshold,
            total_checked=0,
            anomalous_count=0,
            anomaly_rate=0.0,
            predictions=[],
            error="no_baseline",
        )

    predictions = await DBService.get_predictions_with_features(
        db=db,
        model_name=model_name,
        days=days,
        limit=limit,
    )

    total_checked = len(predictions)
    anomalous: list[AnomalyPredictionEntry] = []

    for pred, obs in predictions:
        features = pred.input_features
        if not isinstance(features, dict):
            continue

        anomalous_features: dict[str, AnomalyFeatureDetail] = {}

        for feat_name, feat_value in features.items():
            if not isinstance(feat_value, (int, float)) or isinstance(feat_value, bool):
                continue

            bl = baseline.get(feat_name)
            if bl is None:
                continue

            bl_mean = float(bl.get("mean", 0))
            bl_std = float(bl.get("std", 0))

            if bl_std <= 0:
                continue

            z = abs(float(feat_value) - bl_mean) / bl_std

            if z >= z_threshold:
                anomalous_features[feat_name] = AnomalyFeatureDetail(
                    value=float(feat_value),
                    z_score=round(z, 4),
                    baseline_mean=bl_mean,
                    baseline_std=bl_std,
                )

        if anomalous_features:
            max_confidence = None
            if pred.probabilities:
                max_confidence = round(max(pred.probabilities), 4)

            anomalous.append(
                AnomalyPredictionEntry(
                    prediction_id=pred.id,
                    timestamp=pred.timestamp,
                    prediction_result=pred.prediction_result,
                    max_confidence=max_confidence,
                    id_obs=pred.id_obs,
                    ground_truth=obs.observed_result if obs is not None else None,
                    anomalous_features=anomalous_features,
                )
            )

    anomalous_count = len(anomalous)
    anomaly_rate = round(anomalous_count / total_checked, 4) if total_checked > 0 else 0.0

    return AnomaliesResponse(
        model_name=model_name,
        period_days=days,
        z_threshold=z_threshold,
        total_checked=total_checked,
        anomalous_count=anomalous_count,
        anomaly_rate=anomaly_rate,
        predictions=anomalous,
    )


@router.get("/predictions/unlabeled", response_model=UnlabeledPredictionsResponse)
async def get_unlabeled_predictions(
    model_name: Optional[str] = Query(None, description="Filter by model name (optional)"),
    model_version: Optional[str] = Query(None, description="Filter by model version (optional)"),
    strategy: str = Query(
        "uncertainty",
        description=(
            "Sampling strategy: "
            "'uncertainty' (lowest confidence first — most informative for active learning), "
            "'recent' (newest predictions first), "
            "'random' (unbiased random sample)"
        ),
    ),
    limit: int = Query(
        50, ge=1, le=200, description="Max results to return (default: 50, max: 200)"
    ),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Returns predictions without an associated observed result, ordered by labeling value.

    Helps data teams prioritize annotation effort:
    - **uncertainty**: lowest max_confidence first — highest value for active learning
    - **recent**: newest predictions first — useful for production monitoring
    - **random**: unbiased sample — useful for performance estimation

    Only non-shadow, successful predictions with an id_obs are returned.
    The CSV export can be re-imported via POST /observed-results/upload-csv after annotation.

    Requires a valid Bearer token.
    """
    if strategy not in ("uncertainty", "recent", "random"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="'strategy' must be one of: uncertainty, recent, random",
        )

    predictions, total = await DBService.get_unlabeled_predictions(
        db=db,
        model_name=model_name,
        model_version=model_version,
        strategy=strategy,
        limit=limit,
    )

    return UnlabeledPredictionsResponse(
        total_unlabeled=total,
        returned=len(predictions),
        strategy=strategy,
        model_name=model_name,
        model_version=model_version,
        predictions=[
            UnlabeledPredictionItem(
                id=p.id,
                id_obs=p.id_obs,
                model_name=p.model_name,
                model_version=p.model_version,
                prediction_result=p.prediction_result,
                max_confidence=p.max_confidence,
                timestamp=p.timestamp,
            )
            for p in predictions
        ],
    )


@router.delete("/predictions/purge", response_model=PurgeResponse)
async def purge_predictions(
    older_than_days: int = Query(
        ...,
        ge=0,
        description="Delete predictions older than N days (0 = all)",
    ),
    model_name: Optional[str] = Query(
        None,
        description="Restrict the purge to a specific model (optional)",
    ),
    dry_run: bool = Query(
        True,
        description="Simulate without deleting (default: true — pass dry_run=false to actually delete)",
    ),
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Purges predictions and their associated observed_results.

    - **older_than_days**: retention threshold in days (0 = delete all)
    - **model_name**: restrict the purge to a single model (optional)
    - **dry_run**: `true` by default — simulation without deletion. Pass `dry_run=false` to actually delete.

    Cascades deletion to observed_results (ground truth) linked to purged predictions.

    Admin access only.
    """
    result = await DBService.purge_predictions(
        db=db,
        older_than_days=older_than_days,
        model_name=model_name,
        dry_run=dry_run,
    )
    return PurgeResponse(**result)


@router.delete("/predictions/{prediction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prediction(
    prediction_id: int,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Deletes a prediction by its ID (admin only)."""
    deleted = await DBService.delete_prediction(db, prediction_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found.")


@router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
async def get_prediction_by_id(
    prediction_id: int,
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Direct lookup of a prediction by its id.

    - Returns 404 if the prediction does not exist.
    - A standard user can only see their own predictions (403 otherwise).
    - An admin can see all predictions.
    """
    prediction = await DBService.get_prediction_by_id(db, prediction_id)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found.",
        )
    if user.role != UserRole.ADMIN and prediction.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: this prediction does not belong to you.",
        )
    return PredictionResponse(
        id=prediction.id,
        model_name=prediction.model_name,
        model_version=prediction.model_version,
        id_obs=prediction.id_obs,
        input_features=prediction.input_features,
        prediction_result=prediction.prediction_result,
        probabilities=prediction.probabilities,
        max_confidence=prediction.max_confidence,
        response_time_ms=prediction.response_time_ms,
        timestamp=prediction.timestamp,
        status=prediction.status,
        error_message=prediction.error_message,
        username=prediction.user.username if prediction.user else None,
        is_shadow=prediction.is_shadow,
    )


@router.get("/predictions/{prediction_id}/explain", response_model=ExplainOutput)
async def explain_prediction_by_id(
    prediction_id: int,
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the SHAP explanation of a stored prediction (post-hoc).

    - Returns 404 if the prediction does not exist.
    - A standard user can only see their own predictions (403 otherwise).
    - An admin can see all predictions.
    - Returns 422 if the prediction is in error (status != 'success') or if input_features is null.

    Uses input_features already stored in the database — no re-submission of features required.
    """
    prediction = await DBService.get_prediction_by_id(db, prediction_id)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found.",
        )
    if user.role != UserRole.ADMIN and prediction.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: this prediction does not belong to you.",
        )
    if prediction.status != "success" or prediction.input_features is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Cannot explain this prediction: "
                "status is not 'success' or input_features is absent."
            ),
        )

    model_data = await model_service.load_model(db, prediction.model_name, prediction.model_version)
    model = model_data["model"]
    metadata = model_data["metadata"]

    if not hasattr(model, "feature_names_in_"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Model '{prediction.model_name}' does not have the 'feature_names_in_' attribute. "
                "The model must have been trained with a pandas DataFrame."
            ),
        )

    feature_names = list(model.feature_names_in_)
    x = pd.DataFrame([{f: prediction.input_features[f] for f in feature_names}])

    explanation = compute_shap_explanation(
        model=model,
        feature_names=feature_names,
        x=x,
        prediction_result=prediction.prediction_result,
        feature_baseline=metadata.feature_baseline,
    )

    return ExplainOutput(
        model_name=metadata.name,
        model_version=metadata.version,
        prediction=prediction.prediction_result,
        shap_values=explanation["shap_values"],
        base_value=explanation["base_value"],
        model_type=explanation["model_type"],
    )


@router.post("/predict", response_model=PredictionOutput)
@limiter.limit("60/minute")
async def predict(
    input_data: PredictionInput,
    request: Request,
    background_tasks: BackgroundTasks,
    strict_validation: bool = Query(
        False,
        description=(
            "If true, rejects the request with 422 if the features do not match "
            "the model schema exactly (including unexpected features)."
        ),
    ),
    explain: bool = Query(
        False,
        description=(
            "If true, computes and returns local SHAP values inline in the response "
            "(shap_values, shap_base_value). Silent if the model type is not supported."
        ),
    ),
    store: bool = Query(
        True,
        description=(
            "If false, the prediction is not saved to the database "
            "(useful for interactive tests from the UI or debugging tools)."
        ),
    ),
    user: User = Depends(check_prediction_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """
    Makes a prediction with the specified sklearn model.

    - **model_name**: Name of the model to use
    - **model_version**: Target version (e.g. `1.0.0`). If absent, uses the `is_production=True`
      version; otherwise, the most recent version.
    - **id_obs**: Observation identifier (optional, stored in DB)
    - **features**: Features as a named dict `{"feature1": value, ...}`.
      The model must expose `feature_names_in_` (trained with a pandas DataFrame).
      Missing keys return a 422 error.
    - **strict_validation**: If `true`, also rejects unexpected features with 422.
    - **explain**: If `true`, returns local SHAP values (`shap_values`, `shap_base_value`)
      in the response. Silent if the model type is not supported by SHAP.

    Requires a Bearer token in the Authorization header.
    All predictions are logged to the database.
    """
    start_time = time.time()
    prediction_result = None
    probability = None
    error_message = None
    shadow_meta: list = []
    _metric_version = "unknown"
    _metric_mode = "production"
    structlog.contextvars.bind_contextvars(event_type="predict", model_name=input_data.model_name)

    try:
        # --- Routing: explicit version OR A/B/shadow routing ---
        if input_data.model_version is not None:
            # Explicit path → check deprecation BEFORE loading from MinIO
            explicit_meta = await DBService.get_model_metadata(
                db, input_data.model_name, input_data.model_version
            )
            if explicit_meta and getattr(explicit_meta, "status", "active") == "deprecated":
                prod_meta = await DBService.get_model_metadata(db, input_data.model_name)
                prod_hint = f"{input_data.model_name}/{prod_meta.version}" if prod_meta else "none"
                raise HTTPException(
                    status_code=status.HTTP_410_GONE,
                    detail=(
                        f"Model {input_data.model_name}/{input_data.model_version} is deprecated. "
                        f"Current production: {prod_hint}"
                    ),
                )
            model_data = await model_service.load_model(
                db, input_data.model_name, input_data.model_version
            )
            shadow_meta = []
        else:
            # Smart routing: A/B test or shadow if configured
            primary_meta, shadow_meta = await model_service.select_routing_versions(
                db, input_data.model_name
            )
            if primary_meta is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No active version found for model '{input_data.model_name}'.",
                )
            model_data = await model_service.load_model(
                db, input_data.model_name, primary_meta.version
            )

        model = model_data["model"]
        metadata = model_data["metadata"]

        # Convert the features dict to a numpy array
        if not hasattr(model, "feature_names_in_"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Model '{input_data.model_name}' does not have the 'feature_names_in_' attribute. "
                    "The model must have been trained with a pandas DataFrame "
                    "(column names are then automatically saved by sklearn)."
                ),
            )
        missing = set(model.feature_names_in_) - set(input_data.features.keys())
        if missing:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Missing features in the request: {sorted(missing)}. "
                    f"Expected features: {list(model.feature_names_in_)}"
                ),
            )

        # Strict mode: reject if unexpected features are present
        if strict_validation:
            expected = resolve_expected_features(model, getattr(metadata, "feature_baseline", None))
            if expected is not None:
                errors, _ = validate_input_features(input_data.features, expected)
                if errors:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail={
                            "message": "Strict validation failed: the input schema does not match.",
                            "valid": False,
                            "errors": [e.model_dump() for e in errors],
                            "expected_features": sorted(expected),
                        },
                    )
        x = pd.DataFrame([{name: input_data.features[name] for name in model.feature_names_in_}])

        # Run prediction
        prediction = model.predict(x)[0]
        prediction_result = prediction.item() if hasattr(prediction, "item") else prediction

        # Try to get probabilities if the model supports it
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(x)[0].tolist()

        # Compute low_confidence if a threshold is configured on the model
        low_confidence = None
        if probability is not None and metadata.confidence_threshold is not None:
            low_confidence = max(probability) < metadata.confidence_threshold

        response_time_ms = (time.time() - start_time) * 1000
        _metric_version = metadata.version
        _metric_mode = "explicit" if input_data.model_version is not None else "production"
        predictions_total.labels(
            model_name=metadata.name,
            version=_metric_version,
            mode=_metric_mode,
            status="success",
        ).inc()
        inference_duration_seconds.labels(
            model_name=metadata.name,
            version=_metric_version,
        ).observe(response_time_ms / 1000)

        # Log the successful prediction — skipped if store=False (UI tests)
        _saved_prediction = None
        if store:
            _prediction_payload = {
                "user_id": user.id,
                "model_name": metadata.name,
                "model_version": metadata.version,
                "id_obs": input_data.id_obs,
                "input_features": input_data.features,
                "prediction_result": prediction_result,
                "probabilities": probability,
                "response_time_ms": response_time_ms,
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "status": "success",
                "max_confidence": max(probability) if probability else None,
                "timestamp": input_data.timestamp,
            }
            if settings.PREDICTION_STREAM_ENABLED:
                _published = await _publish_prediction_to_stream(_prediction_payload)
                if not _published:
                    _saved_prediction = await DBService.create_prediction(
                        db=db, **_prediction_payload
                    )
            else:
                _saved_prediction = await DBService.create_prediction(db=db, **_prediction_payload)

            # --- Dispatch shadow in background (shadow + non-selected A/B versions) ---
            for _sm in shadow_meta:
                background_tasks.add_task(
                    _run_shadow_prediction,
                    model_name=metadata.name,
                    shadow_version=_sm.version,
                    features=input_data.features,
                    id_obs=input_data.id_obs,
                    user_id=user.id,
                    client_ip=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent"),
                    timestamp=input_data.timestamp,
                )

            # Fire webhook if configured on the model
            if metadata.webhook_url:
                background_tasks.add_task(
                    send_webhook,
                    metadata.webhook_url,
                    {
                        "model_name": metadata.name,
                        "model_version": metadata.version,
                        "id_obs": input_data.id_obs,
                        "prediction": prediction_result,
                        "probability": probability,
                        "low_confidence": low_confidence,
                    },
                )

        # selected_version is set only if A/B routing was used
        selected_version = metadata.version if input_data.model_version is None else None

        # Inline SHAP explanation (silently skip if model not supported or feature missing)
        shap_values_inline = None
        shap_base_value_inline = None
        if explain:
            try:
                from sklearn.pipeline import Pipeline as _Pipeline
                _feat_names = list(model.feature_names_in_)
                if isinstance(model, _Pipeline):
                    x_shap = pd.DataFrame([{n: input_data.features[n] for n in _feat_names}])
                else:
                    x_shap = np.array(
                        [[input_data.features[name] for name in _feat_names]], dtype=float
                    )
                explanation = compute_shap_explanation(
                    model=model,
                    feature_names=_feat_names,
                    x=x_shap,
                    prediction_result=prediction_result,
                    feature_baseline=metadata.feature_baseline,
                )
                shap_values_inline = explanation["shap_values"]
                shap_base_value_inline = explanation["base_value"]
            except Exception:
                logger.debug(
                    "Inline SHAP skipped (model not supported)",
                    model=metadata.name,
                    version=metadata.version,
                )

        return PredictionOutput(
            id=_saved_prediction.id if store and _saved_prediction is not None else None,
            model_name=metadata.name,
            model_version=metadata.version,
            id_obs=input_data.id_obs,
            prediction=prediction_result,
            probability=probability,
            low_confidence=low_confidence,
            selected_version=selected_version,
            shap_values=shap_values_inline,
            shap_base_value=shap_base_value_inline,
        )

    except HTTPException:
        # Re-raise HTTPExceptions (404, etc.)
        raise

    except Exception as e:
        # Log the error
        response_time_ms = (time.time() - start_time) * 1000
        error_message = str(e)

        predictions_total.labels(
            model_name=input_data.model_name,
            version=_metric_version,
            mode=_metric_mode,
            status="error",
        ).inc()

        if store:
            try:
                await DBService.create_prediction(
                    db=db,
                    user_id=user.id,
                    model_name=input_data.model_name,
                    model_version=None,
                    input_features=input_data.features,
                    prediction_result=None,
                    probabilities=None,
                    response_time_ms=response_time_ms,
                    client_ip=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent"),
                    status="error",
                    error_message=error_message,
                    id_obs=input_data.id_obs,
                    timestamp=input_data.timestamp,
                )
            except Exception as log_error:
                logger.error("Error logging the prediction", error=str(log_error))

        logger.error(
            "Internal error during prediction",
            model=input_data.model_name,
            error=error_message,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal prediction error. Check server logs.",
        )

    finally:
        structlog.contextvars.clear_contextvars()


MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "500"))


@router.post("/predict-batch", response_model=BatchPredictionOutput)
@limiter.limit("200/minute")
async def predict_batch(
    input_data: BatchPredictionInput,
    request: Request,
    background_tasks: BackgroundTasks,
    strict_validation: bool = Query(
        False,
        description=(
            "If true, rejects the request with 422 if unexpected features are present "
            "in any item of the batch."
        ),
    ),
    user: User = Depends(check_prediction_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """
    Makes batch predictions with the specified sklearn model.

    - **model_name**: Name of the model to use
    - **model_version**: Target version (optional)
    - **inputs**: List of observations, each with `features` and an optional `id_obs`

    The model is loaded once (shared cache), all predictions are persisted
    in a single transaction (`add_all`).

    Requires a Bearer token in the Authorization header.
    """
    batch_size = len(input_data.inputs)
    if batch_size > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch too large ({batch_size} items, max {MAX_BATCH_SIZE}).",
        )
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    # Check that the remaining quota covers the batch size
    today_count = await DBService.get_user_prediction_count_today(db, user.id)
    remaining = user.rate_limit_per_day - today_count
    if batch_size > remaining:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit insufficient for this batch ({batch_size} predictions requested, "
                f"{remaining} remaining today out of {user.rate_limit_per_day})."
            ),
        )

    try:
        # A/B / shadow routing if no explicit version
        batch_shadow_list: List = []
        if input_data.model_version is None:
            primary_meta, batch_shadow_list = await model_service.select_routing_versions(
                db, input_data.model_name
            )
            if primary_meta is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No active version found for model '{input_data.model_name}'.",
                )
            resolved_version = primary_meta.version
        else:
            resolved_version = input_data.model_version

        # Load the model once (shared cache)
        model_data = await model_service.load_model(db, input_data.model_name, resolved_version)
        model = model_data["model"]
        metadata = model_data["metadata"]

        # Validate that the model exposes feature_names_in_
        if not hasattr(model, "feature_names_in_"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Model '{input_data.model_name}' does not have the 'feature_names_in_' attribute. "
                    "The model must have been trained with a pandas DataFrame."
                ),
            )

        has_proba = hasattr(model, "predict_proba")
        confidence_threshold = metadata.confidence_threshold
        orm_objects: List[Prediction] = []
        results: List[BatchPredictionResultItem] = []

        for item in input_data.inputs:
            item_start = time.time()

            # Validate the features of this item
            missing = set(model.feature_names_in_) - set(item.features.keys())
            if missing:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"Missing features for observation '{item.id_obs}': "
                        f"{sorted(missing)}. Expected features: {list(model.feature_names_in_)}"
                    ),
                )

            # Strict mode: reject if unexpected features are present
            if strict_validation:
                expected = resolve_expected_features(
                    model, getattr(metadata, "feature_baseline", None)
                )
                if expected is not None:
                    errors, _ = validate_input_features(item.features, expected)
                    if errors:
                        raise HTTPException(
                            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail={
                                "message": f"Strict validation failed for observation '{item.id_obs}': the input schema does not match.",
                                "valid": False,
                                "errors": [e.model_dump() for e in errors],
                                "expected_features": sorted(expected),
                            },
                        )

            x = pd.DataFrame([{name: item.features[name] for name in model.feature_names_in_}])

            raw = model.predict(x)[0]
            prediction_result = raw.item() if hasattr(raw, "item") else raw
            probability = model.predict_proba(x)[0].tolist() if has_proba else None

            # Compute low_confidence if a threshold is configured on the model
            low_confidence = None
            if probability is not None and confidence_threshold is not None:
                low_confidence = max(probability) < confidence_threshold

            response_time_ms = (time.time() - item_start) * 1000

            orm_objects.append(
                Prediction(
                    user_id=user.id,
                    model_name=metadata.name,
                    model_version=metadata.version,
                    id_obs=item.id_obs,
                    input_features=item.features,
                    prediction_result=prediction_result,
                    probabilities=probability,
                    response_time_ms=response_time_ms,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    status="success",
                    max_confidence=max(probability) if probability else None,
                    **({"timestamp": item.timestamp} if item.timestamp is not None else {}),
                )
            )
            results.append(
                BatchPredictionResultItem(
                    id_obs=item.id_obs,
                    prediction=prediction_result,
                    probability=probability,
                    low_confidence=low_confidence,
                )
            )

        # Persist all predictions in a single transaction
        db.add_all(orm_objects)
        await db.commit()

        # Dispatch shadow predictions in the background (shadow + non-selected A/B versions)
        # Sample to avoid saturating the event loop on large batches
        if batch_shadow_list:
            shadow_items = (
                input_data.inputs
                if len(input_data.inputs) <= _MAX_SHADOW_ITEMS_PER_BATCH
                else random.sample(input_data.inputs, _MAX_SHADOW_ITEMS_PER_BATCH)
            )
            for _sm in batch_shadow_list:
                for item in shadow_items:
                    background_tasks.add_task(
                        _run_shadow_prediction,
                        input_data.model_name,
                        shadow_version=_sm.version,
                        features=item.features,
                        id_obs=item.id_obs,
                        user_id=user.id,
                        client_ip=client_ip,
                        user_agent=user_agent,
                        timestamp=item.timestamp,
                    )

        return BatchPredictionOutput(
            model_name=metadata.name,
            model_version=metadata.version,
            predictions=results,
        )

    except HTTPException:
        raise

    except Exception as e:
        error_message = str(e)
        response_time_ms = 0.0
        try:
            error_objects = [
                Prediction(
                    user_id=user.id,
                    model_name=input_data.model_name,
                    model_version=input_data.model_version,
                    id_obs=item.id_obs,
                    input_features=item.features,
                    prediction_result=None,
                    probabilities=None,
                    response_time_ms=response_time_ms,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    status="error",
                    error_message=error_message,
                )
                for item in input_data.inputs
            ]
            db.add_all(error_objects)
            await db.commit()
        except Exception as log_error:
            logger.error("Error logging the failed batch", error=str(log_error))

        logger.error(
            "Internal error during batch",
            model=input_data.model_name,
            error=error_message,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during batch processing. Check server logs.",
        )


@router.post("/explain", response_model=ExplainOutput)
async def explain(
    input_data: ExplainInput,
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns local SHAP importances for an observation.

    - **model_name** / **model_version**: same selection as `/predict`
    - **features**: same format as `/predict`

    Does not consume rate-limit quota and does not log to the database.

    **Supported models**:
    - Trees: RandomForest, GradientBoosting, DecisionTree, ExtraTrees, HistGradientBoosting
    - Linear: LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, SGD

    Returns a dict `{feature: shap_value}` indicating the contribution of each feature
    to the prediction, as well as the model's base value `E[f(X)]`.
    """
    model_data = await model_service.load_model(db, input_data.model_name, input_data.model_version)
    model = model_data["model"]
    metadata = model_data["metadata"]

    if not hasattr(model, "feature_names_in_"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Model '{input_data.model_name}' does not have the 'feature_names_in_' attribute. "
                "The model must have been trained with a pandas DataFrame."
            ),
        )

    feature_names = list(model.feature_names_in_)
    missing = set(feature_names) - set(input_data.features.keys())
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Missing features in the request: {sorted(missing)}. "
                f"Expected features: {feature_names}"
            ),
        )

    x = pd.DataFrame([{f: input_data.features[f] for f in feature_names}])

    raw = model.predict(x)[0]
    prediction_result = raw.item() if hasattr(raw, "item") else raw

    explanation = compute_shap_explanation(
        model=model,
        feature_names=feature_names,
        x=x,
        prediction_result=prediction_result,
        feature_baseline=metadata.feature_baseline,
    )

    return ExplainOutput(
        model_name=metadata.name,
        model_version=metadata.version,
        prediction=prediction_result,
        shap_values=explanation["shap_values"],
        base_value=explanation["base_value"],
        model_type=explanation["model_type"],
    )
