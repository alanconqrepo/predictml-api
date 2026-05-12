"""
Endpoints pour les prédictions
"""

import csv
import io
import json
import os
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.ml_metrics import inference_duration_seconds, predictions_total
from src.core.rate_limit import limiter
from src.core.security import check_prediction_rate_limit, require_admin, verify_token
from src.db.database import AsyncSessionLocal, get_db
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
)
from src.services.db_service import DBService
from src.services.input_validation_service import resolve_expected_features, validate_input_features
from src.services.model_service import model_service
from src.services.shap_service import compute_shap_explanation
from src.services.webhook_service import send_webhook

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["predictions"])


async def _run_shadow_prediction(
    model_name: str,
    shadow_version: str,
    features: dict,
    id_obs: Optional[str],
    user_id: int,
    client_ip: Optional[str],
    user_agent: Optional[str],
) -> None:
    """
    Exécute la prédiction du modèle shadow en background et la persiste avec is_shadow=True.
    Toutes les exceptions sont catchées et loggées — ne propage jamais rien au client.
    """
    import time as _time

    start = _time.time()
    try:
        async with AsyncSessionLocal() as db:
            shadow_data = await model_service.load_model(db, model_name, shadow_version)
            shadow_model = shadow_data["model"]
            shadow_meta = shadow_data["metadata"]

            if not hasattr(shadow_model, "feature_names_in_"):
                raise ValueError(
                    f"Shadow model '{model_name}:{shadow_version}' n'a pas feature_names_in_"
                )

            x = np.array([[features[n] for n in shadow_model.feature_names_in_]], dtype=object)
            raw = shadow_model.predict(x)[0]
            result = raw.item() if hasattr(raw, "item") else raw
            proba = (
                shadow_model.predict_proba(x)[0].tolist()
                if hasattr(shadow_model, "predict_proba")
                else None
            )
            rt_ms = (_time.time() - start) * 1000

            await DBService.create_prediction(
                db=db,
                user_id=user_id,
                model_name=shadow_meta.name,
                model_version=shadow_meta.version,
                input_features=features,
                prediction_result=result,
                probabilities=proba,
                response_time_ms=rt_ms,
                client_ip=client_ip,
                user_agent=user_agent,
                status="success",
                id_obs=id_obs,
                is_shadow=True,
                max_confidence=max(proba) if proba else None,
            )

            logger.info(
                "Shadow prediction enregistrée",
                model_name=model_name,
                version=shadow_version,
                result=result,
            )

    except Exception as e:
        logger.warning(
            "Échec de la prédiction shadow (non-bloquant)",
            model_name=model_name,
            version=shadow_version,
            error=str(e),
        )


@router.get("/predictions/stats", response_model=PredictionStatsResponse)
async def get_prediction_stats(
    model_name: Optional[str] = Query(None, description="Filtrer par nom de modèle (optionnel)"),
    days: int = Query(30, ge=1, le=365, description="Fenêtre en jours (défaut : 30, max : 365)"),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Statistiques agrégées des prédictions par modèle sur une fenêtre glissante.

    - **model_name** : filtrer sur un seul modèle (optionnel)
    - **days** : fenêtre temporelle en jours (défaut 30, max 365)

    Retourne pour chaque modèle : total, erreurs, taux d'erreur, temps de réponse moyen / p50 / p95.

    Nécessite un token Bearer valide.
    """
    raw = await DBService.get_prediction_stats(db, days=days, model_name=model_name)
    return PredictionStatsResponse(
        days=days,
        model_name=model_name,
        stats=[PredictionStatsItem(**s) for s in raw],
    )


@router.get("/predictions", response_model=PredictionsListResponse)
async def get_predictions(
    name: str = Query(..., description="Nom du modèle"),
    start: datetime = Query(
        ..., description="Début de la période (ISO 8601, ex: 2024-01-01T00:00:00)"
    ),
    end: datetime = Query(..., description="Fin de la période (ISO 8601, ex: 2024-12-31T23:59:59)"),
    version: Optional[str] = Query(None, description="Version du modèle (optionnel)"),
    user: Optional[str] = Query(None, description="Nom d'utilisateur (optionnel)"),
    id_obs: Optional[str] = Query(None, description="Identifiant de l'observation (optionnel)"),
    limit: int = Query(100, ge=1, le=1000, description="Nombre max de résultats"),
    cursor: Optional[int] = Query(
        None, ge=1, description="Curseur de pagination (id de la dernière prédiction vue)"
    ),
    min_confidence: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Confiance minimale (max des probabilités) — optionnel, classifieurs uniquement",
    ),
    max_confidence: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Confiance maximale (max des probabilités) — optionnel, classifieurs uniquement",
    ),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Retourne l'historique des prédictions avec filtres (pagination par curseur).

    - **name** : nom du modèle — obligatoire
    - **start** / **end** : plage datetime — obligatoire
    - **version** : version du modèle — optionnel
    - **user** : nom d'utilisateur — optionnel
    - **id_obs** : identifiant de l'observation — optionnel
    - **limit** : nombre max de résultats (défaut : 100, max : 1000)
    - **cursor** : id de la dernière prédiction vue (pour la page suivante, utiliser `next_cursor` de la réponse précédente)
    - **min_confidence** : filtre sur la confiance minimale (max des probabilités), 0.0–1.0 — optionnel
    - **max_confidence** : filtre sur la confiance maximale (max des probabilités), 0.0–1.0 — optionnel

    Nécessite un token Bearer valide.
    """
    if start > end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="'start' doit être antérieur à 'end'.",
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
    model_name: Optional[str] = Query(None, description="Filtrer par nom de modèle (optionnel)"),
    start: datetime = Query(..., description="Début de la période (ISO 8601)"),
    end: datetime = Query(..., description="Fin de la période (ISO 8601)"),
    export_format: str = Query(
        "csv",
        alias="format",
        description="Format d'export : csv, jsonl ou parquet (défaut : csv)",
    ),
    include_features: bool = Query(True, description="Inclure input_features dans l'export"),
    pred_status: Optional[str] = Query(
        None, alias="status", description="Filtrer par statut : success ou error"
    ),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Export bulk des prédictions au format CSV, JSONL ou Parquet via streaming par curseur.

    - **model_name** : filtrer sur un modèle (optionnel — tous les modèles si absent)
    - **start** / **end** : plage datetime — obligatoire
    - **format** : `csv` (défaut), `jsonl` ou `parquet`
    - **include_features** : inclure `input_features` dans l'export (défaut : true)
    - **status** : filtrer par statut `success` ou `error` (optionnel)

    Retourne un fichier en téléchargement direct (Content-Disposition: attachment).
    Le streaming par curseur évite de charger tout l'historique en mémoire (CSV/JSONL).
    Le format Parquet charge toutes les lignes en mémoire avant sérialisation.

    Nécessite un token Bearer valide.
    """
    if start > end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="'start' doit être antérieur à 'end'.",
        )
    if export_format not in ("csv", "jsonl", "parquet"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le paramètre 'format' doit être 'csv', 'jsonl' ou 'parquet'.",
        )
    if pred_status is not None and pred_status not in ("success", "error"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le paramètre 'status' doit être 'success' ou 'error'.",
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
                for row in rows:
                    buf = io.StringIO()
                    vals = [
                        row.id,
                        row.timestamp.isoformat() if row.timestamp else None,
                        row.model_name,
                        row.model_version,
                        row.user.username if row.user else None,
                        row.id_obs,
                        json.dumps(row.prediction_result),
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
                for row in rows:
                    record: dict = {
                        "id": row.id,
                        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                        "model_name": row.model_name,
                        "model_version": row.model_version,
                        "username": row.user.username if row.user else None,
                        "id_obs": row.id_obs,
                        "prediction_result": row.prediction_result,
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
            cursor = rows[-1].id

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
            for row in rows:
                record: dict = {
                    "id": row.id,
                    "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                    "model_name": row.model_name,
                    "model_version": row.model_version,
                    "username": row.user.username if row.user else None,
                    "id_obs": row.id_obs,
                    "prediction_result": json.dumps(row.prediction_result),
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
            cursor = rows[-1].id

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
    model_name: str = Query(..., description="Nom du modèle (requis)"),
    days: int = Query(7, ge=1, le=90, description="Fenêtre temporelle en jours (défaut : 7)"),
    z_threshold: float = Query(
        3.0, ge=0.0, description="Seuil z-score pour détection (défaut : 3.0)"
    ),
    limit: int = Query(
        200, ge=1, le=1000, description="Max prédictions à analyser (défaut : 200, max : 1000)"
    ),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Prédictions avec features aberrantes (z-score ≥ seuil).

    Pour chaque prédiction dans la fenêtre temporelle, calcule le z-score par feature
    en comparant à la baseline du modèle : z = |value - baseline_mean| / baseline_std.
    Retourne uniquement les prédictions où au moins une feature dépasse z_threshold.

    - **model_name** : nom du modèle — obligatoire
    - **days** : fenêtre temporelle en jours (défaut : 7, max : 90)
    - **z_threshold** : seuil de détection (défaut : 3.0)
    - **limit** : max prédictions à analyser (défaut : 200, max : 1000)

    Retourne `error: "no_baseline"` si le modèle n'a pas de baseline de features.

    Nécessite un token Bearer valide.
    """
    metadata = await DBService.get_model_metadata(db, model_name)
    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle '{model_name}' introuvable.",
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

    for pred in predictions:
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


@router.delete("/predictions/purge", response_model=PurgeResponse)
async def purge_predictions(
    older_than_days: int = Query(
        ...,
        ge=1,
        description="Supprimer les prédictions plus anciennes que N jours (ex: 90)",
    ),
    model_name: Optional[str] = Query(
        None,
        description="Restreindre la purge à un modèle spécifique (optionnel)",
    ),
    dry_run: bool = Query(
        True,
        description="Simuler sans supprimer (défaut : true — passer dry_run=false pour supprimer réellement)",
    ),
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Purge les prédictions plus anciennes que N jours (rétention RGPD).

    - **older_than_days** : seuil de rétention en jours — obligatoire (ex: 90)
    - **model_name** : restreindre la purge à un seul modèle (optionnel)
    - **dry_run** : `true` par défaut — simulation sans suppression. Passer `dry_run=false` pour supprimer réellement.

    La réponse inclut un avertissement (`linked_observed_results_count > 0`) si des prédictions
    supprimées sont liées à des observed_results (perte de données de performance).

    Accès réservé aux administrateurs.
    """
    result = await DBService.purge_predictions(
        db=db,
        older_than_days=older_than_days,
        model_name=model_name,
        dry_run=dry_run,
    )
    return PurgeResponse(**result)


@router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
async def get_prediction_by_id(
    prediction_id: int,
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Lookup direct d'une prédiction par son id.

    - Retourne 404 si la prédiction n'existe pas.
    - Un utilisateur standard ne voit que ses propres prédictions (403 sinon).
    - Un admin voit toutes les prédictions.
    """
    prediction = await DBService.get_prediction_by_id(db, prediction_id)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prédiction {prediction_id} introuvable.",
        )
    if user.role != UserRole.ADMIN and prediction.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès refusé : cette prédiction ne vous appartient pas.",
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
    Retourne l'explication SHAP d'une prédiction stockée (post-hoc).

    - Retourne 404 si la prédiction n'existe pas.
    - Un utilisateur standard ne voit que ses propres prédictions (403 sinon).
    - Un admin voit toutes les prédictions.
    - Retourne 422 si la prédiction est en erreur (status != 'success') ou si input_features est null.

    Utilise les input_features déjà stockées en base — aucune re-soumission des features requise.
    """
    prediction = await DBService.get_prediction_by_id(db, prediction_id)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prédiction {prediction_id} introuvable.",
        )
    if user.role != UserRole.ADMIN and prediction.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès refusé : cette prédiction ne vous appartient pas.",
        )
    if prediction.status != "success" or prediction.input_features is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Impossible d'expliquer cette prédiction : "
                "statut différent de 'success' ou input_features absent."
            ),
        )

    model_data = await model_service.load_model(db, prediction.model_name, prediction.model_version)
    model = model_data["model"]
    metadata = model_data["metadata"]

    if not hasattr(model, "feature_names_in_"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Le modèle '{prediction.model_name}' ne possède pas l'attribut "
                "'feature_names_in_'. Le modèle doit avoir été entraîné avec un DataFrame pandas."
            ),
        )

    feature_names = list(model.feature_names_in_)
    x = np.array([[prediction.input_features[f] for f in feature_names]], dtype=float)

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
            "Si true, rejette la requête avec 422 si les features ne correspondent pas "
            "exactement au schéma du modèle (features inattendues incluses)."
        ),
    ),
    explain: bool = Query(
        False,
        description=(
            "Si true, calcule et retourne les valeurs SHAP locales inline dans la réponse "
            "(shap_values, shap_base_value). Silencieux si le type de modèle n'est pas supporté."
        ),
    ),
    user: User = Depends(check_prediction_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """
    Fait une prédiction avec le modèle sklearn spécifié.

    - **model_name**: Nom du modèle à utiliser
    - **model_version**: Version cible (ex: `1.0.0`). Si absent, utilise la version
      `is_production=True` ; à défaut, la version la plus récente.
    - **id_obs**: Identifiant de l'observation (optionnel, stocké en DB)
    - **features**: Features sous forme de dict nommé `{"feature1": valeur, ...}`.
      Le modèle doit exposer `feature_names_in_` (entraîné avec un DataFrame pandas).
      Les clés manquantes retournent une erreur 422.
    - **strict_validation**: Si `true`, rejette également les features inattendues avec 422.
    - **explain**: Si `true`, retourne les valeurs SHAP locales (`shap_values`, `shap_base_value`)
      dans la réponse. Silencieux si le type de modèle n'est pas supporté par SHAP.

    Nécessite un token Bearer dans le header Authorization.
    Toutes les prédictions sont loggées dans la base de données.
    """
    start_time = time.time()
    prediction_result = None
    probability = None
    error_message = None
    shadow_meta = None
    _metric_version = "unknown"
    _metric_mode = "production"
    structlog.contextvars.bind_contextvars(event_type="predict", model_name=input_data.model_name)

    try:
        # --- Routage : version explicite OU routage A/B/shadow ---
        if input_data.model_version is not None:
            # Chemin explicite → vérifier la dépréciation AVANT de charger depuis MinIO
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
            shadow_meta = None
        else:
            # Routage intelligent : A/B test ou shadow si configuré
            primary_meta, shadow_meta = await model_service.select_routing_versions(
                db, input_data.model_name
            )
            if primary_meta is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Aucune version active trouvée pour le modèle '{input_data.model_name}'.",
                )
            model_data = await model_service.load_model(
                db, input_data.model_name, primary_meta.version
            )

        model = model_data["model"]
        metadata = model_data["metadata"]

        # Convertir le dict de features en array numpy
        if not hasattr(model, "feature_names_in_"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Le modèle '{input_data.model_name}' ne possède pas l'attribut "
                    "'feature_names_in_'. Le modèle doit avoir été entraîné avec un "
                    "DataFrame pandas (les noms de colonnes sont alors automatiquement "
                    "sauvegardés par sklearn)."
                ),
            )
        missing = set(model.feature_names_in_) - set(input_data.features.keys())
        if missing:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Features manquantes dans la requête : {sorted(missing)}. "
                    f"Features attendues : {list(model.feature_names_in_)}"
                ),
            )

        # Mode strict : rejeter si des features inattendues sont présentes
        if strict_validation:
            expected = resolve_expected_features(model, getattr(metadata, "feature_baseline", None))
            if expected is not None:
                errors, _ = validate_input_features(input_data.features, expected)
                if errors:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail={
                            "message": "Validation stricte échouée : le schéma d'entrée ne correspond pas.",
                            "valid": False,
                            "errors": [e.model_dump() for e in errors],
                            "expected_features": sorted(expected),
                        },
                    )
        x = np.array(
            [[input_data.features[name] for name in model.feature_names_in_]], dtype=object
        )

        # Faire la prédiction
        prediction = model.predict(x)[0]
        prediction_result = prediction.item() if hasattr(prediction, "item") else prediction

        # Essayer d'obtenir les probabilités si le modèle le supporte
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(x)[0].tolist()

        # Calculer low_confidence si un seuil est configuré sur le modèle
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

        # Logger la prédiction réussie dans la DB
        await DBService.create_prediction(
            db=db,
            user_id=user.id,
            model_name=metadata.name,
            model_version=metadata.version,
            input_features=input_data.features,
            prediction_result=prediction_result,
            probabilities=probability,
            response_time_ms=response_time_ms,
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="success",
            id_obs=input_data.id_obs,
            max_confidence=max(probability) if probability else None,
        )

        # --- Dispatch shadow en background (si une version shadow est active) ---
        if shadow_meta is not None:
            background_tasks.add_task(
                _run_shadow_prediction,
                model_name=metadata.name,
                shadow_version=shadow_meta.version,
                features=input_data.features,
                id_obs=input_data.id_obs,
                user_id=user.id,
                client_ip=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )

        # Déclencher le webhook si configuré sur le modèle
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

        # selected_version est renseigné uniquement si le routage A/B a été utilisé
        selected_version = metadata.version if input_data.model_version is None else None

        # Explication SHAP inline (silently skip si modèle non supporté ou feature manquante)
        shap_values_inline = None
        shap_base_value_inline = None
        if explain:
            try:
                x_float = np.array(
                    [[input_data.features[name] for name in model.feature_names_in_]], dtype=float
                )
                explanation = compute_shap_explanation(
                    model=model,
                    feature_names=list(model.feature_names_in_),
                    x=x_float,
                    prediction_result=prediction_result,
                    feature_baseline=metadata.feature_baseline,
                )
                shap_values_inline = explanation["shap_values"]
                shap_base_value_inline = explanation["base_value"]
            except Exception:
                logger.debug(
                    "SHAP inline ignoré (modèle non supporté)",
                    model=metadata.name,
                    version=metadata.version,
                )

        return PredictionOutput(
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
        # Re-raise les HTTPException (404, etc.)
        raise

    except Exception as e:
        # Logger l'erreur
        response_time_ms = (time.time() - start_time) * 1000
        error_message = str(e)

        predictions_total.labels(
            model_name=input_data.model_name,
            version=_metric_version,
            mode=_metric_mode,
            status="error",
        ).inc()

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
            )
        except Exception as log_error:
            logger.error("Erreur lors du logging de la prédiction", error=str(log_error))

        logger.error(
            "Erreur interne lors de la prédiction",
            model=input_data.model_name,
            error=error_message,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors de la prédiction. Consultez les logs serveur.",
        )

    finally:
        structlog.contextvars.clear_contextvars()


MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "500"))


@router.post("/predict-batch", response_model=BatchPredictionOutput)
@limiter.limit("10/minute")
async def predict_batch(
    input_data: BatchPredictionInput,
    request: Request,
    strict_validation: bool = Query(
        False,
        description=(
            "Si true, rejette la requête avec 422 si des features inattendues sont présentes "
            "dans l'un des items du batch."
        ),
    ),
    user: User = Depends(check_prediction_rate_limit),
    db: AsyncSession = Depends(get_db),
):
    """
    Fait des prédictions en batch avec le modèle sklearn spécifié.

    - **model_name**: Nom du modèle à utiliser
    - **model_version**: Version cible (optionnel)
    - **inputs**: Liste d'observations, chacune avec `features` et un `id_obs` optionnel

    Le modèle est chargé une seule fois (cache), toutes les prédictions sont persistées
    en une seule transaction (`add_all`).

    Nécessite un token Bearer dans le header Authorization.
    """
    batch_size = len(input_data.inputs)
    if batch_size > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch trop grand ({batch_size} items, max {MAX_BATCH_SIZE}).",
        )
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    # Vérifier que le quota restant couvre la taille du batch
    today_count = await DBService.get_user_prediction_count_today(db, user.id)
    remaining = user.rate_limit_per_day - today_count
    if batch_size > remaining:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit insuffisant pour ce batch ({batch_size} prédictions demandées, "
                f"{remaining} restantes aujourd'hui sur {user.rate_limit_per_day})."
            ),
        )

    try:
        # Charger le modèle une seule fois (cache partagé)
        model_data = await model_service.load_model(
            db, input_data.model_name, input_data.model_version
        )
        model = model_data["model"]
        metadata = model_data["metadata"]

        # Valider que le modèle expose feature_names_in_
        if not hasattr(model, "feature_names_in_"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Le modèle '{input_data.model_name}' ne possède pas l'attribut "
                    "'feature_names_in_'. Le modèle doit avoir été entraîné avec un "
                    "DataFrame pandas."
                ),
            )

        has_proba = hasattr(model, "predict_proba")
        confidence_threshold = metadata.confidence_threshold
        orm_objects: List[Prediction] = []
        results: List[BatchPredictionResultItem] = []

        for item in input_data.inputs:
            item_start = time.time()

            # Valider les features de cet item
            missing = set(model.feature_names_in_) - set(item.features.keys())
            if missing:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"Features manquantes pour l'observation '{item.id_obs}': "
                        f"{sorted(missing)}. Features attendues : {list(model.feature_names_in_)}"
                    ),
                )

            # Mode strict : rejeter si des features inattendues sont présentes
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
                                "message": f"Validation stricte échouée pour l'observation '{item.id_obs}' : le schéma d'entrée ne correspond pas.",
                                "valid": False,
                                "errors": [e.model_dump() for e in errors],
                                "expected_features": sorted(expected),
                            },
                        )

            x = np.array([[item.features[name] for name in model.feature_names_in_]], dtype=object)

            raw = model.predict(x)[0]
            prediction_result = raw.item() if hasattr(raw, "item") else raw
            probability = model.predict_proba(x)[0].tolist() if has_proba else None

            # Calculer low_confidence si un seuil est configuré sur le modèle
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

        # Persister toutes les prédictions en une seule transaction
        db.add_all(orm_objects)
        await db.commit()

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
            logger.error("Erreur lors du logging du batch en erreur", error=str(log_error))

        logger.error(
            "Erreur interne lors du batch",
            model=input_data.model_name,
            error=error_message,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors du traitement batch. Consultez les logs serveur.",
        )


@router.post("/explain", response_model=ExplainOutput)
async def explain(
    input_data: ExplainInput,
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Retourne les importances SHAP locales pour une observation.

    - **model_name** / **model_version** : même sélection que `/predict`
    - **features** : même format que `/predict`

    Ne consomme pas de quota rate-limit et ne logue pas en base de données.

    **Modèles supportés** :
    - Arbres : RandomForest, GradientBoosting, DecisionTree, ExtraTrees, HistGradientBoosting
    - Linéaires : LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, SGD

    Retourne un dict `{feature: shap_value}` indiquant la contribution de chaque feature
    à la prédiction, ainsi que la valeur de base `E[f(X)]` du modèle.
    """
    model_data = await model_service.load_model(db, input_data.model_name, input_data.model_version)
    model = model_data["model"]
    metadata = model_data["metadata"]

    if not hasattr(model, "feature_names_in_"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Le modèle '{input_data.model_name}' ne possède pas l'attribut "
                "'feature_names_in_'. Le modèle doit avoir été entraîné avec un DataFrame pandas."
            ),
        )

    feature_names = list(model.feature_names_in_)
    missing = set(feature_names) - set(input_data.features.keys())
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Features manquantes dans la requête : {sorted(missing)}. "
                f"Features attendues : {feature_names}"
            ),
        )

    x = np.array([[input_data.features[f] for f in feature_names]], dtype=float)

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
