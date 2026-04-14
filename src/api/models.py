"""
Endpoints pour la gestion des modèles
"""

import ast
import asyncio
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
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

from src.core.config import settings
from src.core.security import require_admin, verify_token
from src.db.database import get_db
from src.db.models import HistoryActionType, ModelMetadata, User
from src.db.models.model_metadata import DeploymentMode
from src.schemas.model import (
    ABCompareResponse,
    ABVersionStats,
    DriftReportResponse,
    FeatureDriftResult,
    ModelCreateResponse,
    ModelDeleteResponse,
    ModelGetResponse,
    ModelHistoryEntry,
    ModelHistoryResponse,
    ModelPerformanceResponse,
    ModelUpdateInput,
    PerClassMetrics,
    PeriodPerformance,
    RetrainRequest,
    RetrainResponse,
    RollbackResponse,
)
from src.services import drift_service
from src.services.db_service import _ROLLBACK_FIELDS, DBService, _build_snapshot
from src.services.minio_service import minio_service
from src.services.model_service import model_service

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["models"])


def _validate_train_script(source: str) -> Optional[str]:
    """
    Valide statiquement un script train.py contre les contraintes requises.

    Le script doit :
    1. Être du Python syntaxiquement valide
    2. Référencer TRAIN_START_DATE (variable d'env)
    3. Référencer TRAIN_END_DATE (variable d'env)
    4. Référencer OUTPUT_MODEL_PATH (variable d'env)
    5. Contenir un appel de sauvegarde : pickle.dump, joblib.dump ou save_model

    Returns:
        None si le script est valide, sinon un message d'erreur.
    """
    try:
        ast.parse(source)
    except SyntaxError as e:
        return f"Syntaxe Python invalide : {e}"

    required_tokens = {
        "TRAIN_START_DATE": "doit référencer la variable d'env TRAIN_START_DATE",
        "TRAIN_END_DATE": "doit référencer la variable d'env TRAIN_END_DATE",
        "OUTPUT_MODEL_PATH": "doit référencer la variable d'env OUTPUT_MODEL_PATH",
    }
    for token, msg in required_tokens.items():
        if token not in source:
            return f"Le script {msg}"

    save_calls = ["pickle.dump", "joblib.dump", "save_model"]
    if not any(call in source for call in save_calls):
        return "Le script doit sauvegarder le modèle avec pickle.dump, joblib.dump ou save_model"

    return None


@router.get("/models", response_model=List[Dict[str, Any]])
async def list_models(
    tag: Optional[str] = Query(None, description="Filtrer par tag (ex: production, finance)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Liste tous les modèles disponibles depuis la base de données.

    - **tag** : filtrer par tag (optionnel)

    Returns:
        Liste des modèles actifs avec leurs métadonnées
    """
    models = await model_service.get_available_models(db)
    if tag:
        models = [m for m in models if m.get("tags") and tag in m["tags"]]
    return models


@router.get("/models/cached")
async def list_cached_models():
    """
    Liste les modèles actuellement en cache mémoire

    Returns:
        Liste des object keys MinIO en cache
    """
    cached = await model_service.get_cached_models()
    return {"cached_models": cached, "count": len(cached)}


# ---------------------------------------------------------------------------
# Helpers pour GET /models/{name}/performance
# ---------------------------------------------------------------------------


def _detect_model_type(metadata: Optional[ModelMetadata], pairs: list) -> str:
    """Détecte si le modèle est de classification ou de régression."""
    if metadata and metadata.classes:
        return "classification"
    if any(row.probabilities for row in pairs):
        return "classification"
    pred_vals = [row.prediction_result for row in pairs if row.prediction_result is not None]
    if pred_vals and all(isinstance(v, (int, str, bool)) for v in pred_vals):
        return "classification"
    return "regression"


def _compute_classification_metrics(y_true: list, y_pred: list, classes: Optional[list]) -> tuple:
    labels = (
        sorted(set(str(c) for c in classes))
        if classes
        else sorted(set(str(v) for v in y_true + y_pred))
    )
    y_true_s = [str(v) for v in y_true]
    y_pred_s = [str(v) for v in y_pred]

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
    name: str,
    start: Optional[datetime] = Query(None, description="Début de période (ISO 8601)"),
    end: Optional[datetime] = Query(None, description="Fin de période (ISO 8601)"),
    version: Optional[str] = Query(None, description="Version du modèle (optionnel)"),
    granularity: Optional[Literal["day", "week", "month"]] = Query(
        None, description="Agrégation temporelle (day, week, month)"
    ),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Calcule les métriques de performance réelle en production pour un modèle.

    Joint les prédictions et les résultats observés via `id_obs` pour calculer :
    - **Classification** : accuracy, precision/recall/f1 weighted, confusion matrix, métriques par classe
    - **Régression** : MAE, MSE, RMSE, R²

    Paramètres optionnels :
    - **start** / **end** : plage temporelle
    - **version** : version spécifique du modèle
    - **granularity** : décomposer les métriques par jour, semaine ou mois

    Nécessite un token Bearer valide.
    """
    metadata = await DBService.get_model_metadata(db, name, version)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle '{name}' introuvable.",
        )

    total = await DBService.count_predictions(db, name, start, end, version)
    pairs = await DBService.get_performance_pairs(db, name, start, end, version)

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
            pp = PeriodPerformance(period=period_key, matched_count=len(idxs))
            if model_type == "classification":
                bt_s = [str(v) for v in bt]
                bp_s = [str(v) for v in bp]
                pp.accuracy = round(accuracy_score(bt_s, bp_s), 4)
                pp.f1_weighted = round(
                    float(f1_score(bt_s, bp_s, average="weighted", zero_division=0)), 4
                )
            else:
                bt_f = [float(v) for v in bt]
                bp_f = [float(v) for v in bp]
                pp.mae = round(mean_absolute_error(bt_f, bp_f), 4)
                pp.rmse = round(math.sqrt(mean_squared_error(bt_f, bp_f)), 4)
            by_period.append(pp)
        response.by_period = by_period

    return response


@router.get("/models/{name}/drift", response_model=DriftReportResponse)
async def get_model_drift(
    name: str,
    version: Optional[str] = Query(
        None, description="Version du modèle (défaut : production/dernière)"
    ),
    days: int = Query(7, ge=1, le=90, description="Fenêtre temporelle en jours"),
    min_predictions: int = Query(
        30, ge=5, description="Nombre minimum de prédictions pour calculer le drift"
    ),
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Rapport de data drift pour un modèle.

    Compare la distribution des features reçues en production (sur `days` jours)
    au profil baseline stocké lors de l'upload du modèle.

    - **Z-score** : `|prod_mean - baseline_mean| / baseline_std`
      - ok < 2 | warning 2–3 | critical ≥ 3
    - **PSI** : divergence de distribution via bins normaux
      - ok < 0.1 | warning 0.1–0.2 | critical ≥ 0.2

    Retourne `drift_summary = "no_baseline"` si le profil baseline n'a pas été enregistré.
    """
    metadata = await DBService.get_model_metadata(db, name, version)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle '{name}' introuvable.",
        )

    production_stats = await DBService.get_feature_production_stats(
        db, name, metadata.version, days
    )

    total_predictions = sum(v.get("count", 0) for v in production_stats.values())

    baseline = metadata.feature_baseline
    baseline_available = bool(baseline)

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
                    drift_status="no_baseline",
                )
                for feat, stats in production_stats.items()
            },
        )

    features = drift_service.compute_feature_drift(baseline, production_stats, min_predictions)
    summary = drift_service.summarize_drift(features, baseline_available=True)

    return DriftReportResponse(
        model_name=name,
        model_version=metadata.version,
        period_days=days,
        predictions_analyzed=total_predictions,
        baseline_available=True,
        drift_summary=summary,
        features=features,
    )


# ---------------------------------------------------------------------------
# Historique des changements de modèles
# IMPORTANT : ces routes doivent être déclarées AVANT /models/{name}/{version}
# pour éviter que "history" soit interprété comme un paramètre `version`.
# ---------------------------------------------------------------------------


@router.get("/models/{name}/history", response_model=ModelHistoryResponse)
async def list_model_history(
    name: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Retourne l'historique complet de toutes les versions d'un modèle (tri timestamp DESC).

    Nécessite un token Bearer valide.
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
    name: str,
    version: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Retourne l'historique d'une version spécifique d'un modèle (tri timestamp DESC).

    Nécessite un token Bearer valide.
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
    name: str,
    version: str,
    history_id: int,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Restaure les métadonnées d'un modèle à l'état capturé dans une entrée d'historique.

    Seuls les champs de métadonnées sont restaurés (pas les références artifacts MinIO/MLflow).
    Si le snapshot avait `is_production=True`, les autres versions sont automatiquement rétrogradées.

    Réservé aux administrateurs.
    """
    # Charger le modèle cible
    result = await db.execute(
        select(ModelMetadata)
        .options(selectinload(ModelMetadata.creator))
        .where(and_(ModelMetadata.name == name, ModelMetadata.version == version))
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle '{name}' version '{version}' introuvable.",
        )

    # Charger l'entrée d'historique
    history_entry = await DBService.get_history_entry_by_id(db, history_id)
    if not history_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entrée d'historique {history_id} introuvable.",
        )
    if history_entry.model_name != name or history_entry.model_version != version:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="L'entrée d'historique n'appartient pas à ce modèle/version.",
        )

    snapshot = history_entry.snapshot
    restored_fields = []

    # Si le snapshot restaure is_production=True → rétrograder les autres versions
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

    # Appliquer le snapshot sur les champs restaurables
    for field in _ROLLBACK_FIELDS:
        if field in snapshot:
            value = snapshot[field]
            # Re-parser training_date si stocké en ISO string
            if field == "training_date" and isinstance(value, str):
                value = datetime.fromisoformat(value)
            setattr(model, field, value)
            restored_fields.append(field)

    await db.flush()

    # Logger le rollback comme nouvelle entrée d'historique
    new_entry = await DBService.log_model_history(
        db, model, HistoryActionType.ROLLBACK, user.id, user.username, restored_fields
    )

    await db.commit()
    await db.refresh(model)

    return RollbackResponse(
        model_name=name,
        version=version,
        rolled_back_to_history_id=history_id,
        new_history_id=new_entry.id,
        restored_fields=restored_fields,
        snapshot=history_entry.snapshot,
    )


@router.post(
    "/models/{name}/{version}/retrain",
    response_model=RetrainResponse,
)
async def retrain_model(
    name: str,
    version: str,
    payload: RetrainRequest,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Ré-entraîne un modèle en exécutant son script train.py stocké dans MinIO.

    Le script reçoit via variables d'environnement :
    - **TRAIN_START_DATE** / **TRAIN_END_DATE** : plage de dates (YYYY-MM-DD)
    - **OUTPUT_MODEL_PATH** : chemin où déposer le `.pkl` produit
    - **MLFLOW_TRACKING_URI**, **MODEL_NAME** : optionnels

    Timeout d'exécution : 600 secondes.
    Une nouvelle version du modèle est créée dans MinIO et enregistrée en base.
    Si `set_production` est True, la nouvelle version est automatiquement mise en production.

    Réservé aux administrateurs.
    """
    # 1. Vérifier le modèle source
    result = await db.execute(
        select(ModelMetadata)
        .options(selectinload(ModelMetadata.creator))
        .where(and_(ModelMetadata.name == name, ModelMetadata.version == version))
    )
    source_model = result.scalar_one_or_none()
    if not source_model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle '{name}' version '{version}' introuvable.",
        )
    if not source_model.train_script_object_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Le modèle '{name}' v{version} ne possède pas de script d'entraînement. "
                "Uploadez un train.py via POST /models avec le champ train_file."
            ),
        )

    # 2. Déterminer la nouvelle version
    new_version = payload.new_version
    if not new_version:
        new_version = f"{version}-retrain-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    # Vérifier l'unicité de la nouvelle version
    existing = await db.execute(
        select(ModelMetadata).where(
            and_(ModelMetadata.name == name, ModelMetadata.version == new_version)
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"La version '{new_version}' existe déjà pour le modèle '{name}'.",
        )

    # 3. Télécharger train.py depuis MinIO
    try:
        script_bytes = minio_service.download_file_bytes(source_model.train_script_object_key)
    except Exception as e:
        logger.error(
            "Impossible de télécharger le script d'entraînement",
            object_key=source_model.train_script_object_key,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Impossible de télécharger le script train.py : {e}",
        )

    # 4. Exécuter le script dans un répertoire temporaire
    stdout_text = ""
    stderr_text = ""

    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "train.py")
        output_model_path = os.path.join(tmpdir, "output_model.pkl")

        with open(script_path, "wb") as f:
            f.write(script_bytes)

        env = {
            **os.environ,
            "TRAIN_START_DATE": payload.start_date,
            "TRAIN_END_DATE": payload.end_date,
            "OUTPUT_MODEL_PATH": output_model_path,
            "MLFLOW_TRACKING_URI": mlflow_tracking_uri,
            "MODEL_NAME": name,
        }

        try:
            proc = await asyncio.create_subprocess_exec(
                "python",
                script_path,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmpdir,
            )
            try:
                raw_stdout, raw_stderr = await asyncio.wait_for(proc.communicate(), timeout=600.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                logger.error("Timeout ré-entraînement", model=name, version=version)
                return RetrainResponse(
                    model_name=name,
                    source_version=version,
                    new_version=new_version,
                    success=False,
                    stdout="",
                    stderr="",
                    error="Timeout : le script d'entraînement a dépassé 600 secondes.",
                )

            stdout_text = raw_stdout.decode("utf-8", errors="replace")
            stderr_text = raw_stderr.decode("utf-8", errors="replace")

            logger.info(
                "Script d'entraînement exécuté",
                model=name,
                version=version,
                returncode=proc.returncode,
            )

            if proc.returncode != 0:
                logger.error(
                    "Ré-entraînement échoué",
                    model=name,
                    version=version,
                    returncode=proc.returncode,
                    stderr=stderr_text[:500],
                )
                return RetrainResponse(
                    model_name=name,
                    source_version=version,
                    new_version=new_version,
                    success=False,
                    stdout=stdout_text,
                    stderr=stderr_text,
                    error=f"Le script a terminé avec le code {proc.returncode}.",
                )

            # 5. Vérifier que le fichier modèle a été produit
            if not os.path.exists(output_model_path):
                return RetrainResponse(
                    model_name=name,
                    source_version=version,
                    new_version=new_version,
                    success=False,
                    stdout=stdout_text,
                    stderr=stderr_text,
                    error=(
                        f"Le script n'a pas produit de fichier modèle "
                        f"à OUTPUT_MODEL_PATH={output_model_path}"
                    ),
                )

            with open(output_model_path, "rb") as f:
                new_model_bytes = f.read()

        except RetrainResponse.__class__:
            raise
        except Exception as e:
            logger.error("Erreur inattendue lors du ré-entraînement", model=name, error=str(e))
            return RetrainResponse(
                model_name=name,
                source_version=version,
                new_version=new_version,
                success=False,
                stdout=stdout_text,
                stderr=stderr_text,
                error=f"Erreur d'exécution inattendue : {e}",
            )

    # 6. Extraire les métriques JSON depuis la dernière ligne stdout
    new_accuracy = source_model.accuracy
    new_f1 = source_model.f1_score
    for line in reversed(stdout_text.strip().splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                metrics = json.loads(stripped)
                new_accuracy = metrics.get("accuracy", new_accuracy)
                new_f1 = metrics.get("f1_score", new_f1)
            except json.JSONDecodeError:
                pass
            break

    # 7. Uploader le nouveau modèle dans MinIO
    new_object_key = f"{name}/v{new_version}.pkl"
    upload_info = minio_service.upload_model_bytes(new_model_bytes, new_object_key)

    # 8. Copier le script train.py pour la nouvelle version (pour chaîner les ré-entraînements)
    new_train_key = f"{name}/v{new_version}_train.py"
    minio_service.upload_file_bytes(script_bytes, new_train_key, content_type="text/x-python")

    # 9. Créer la nouvelle entrée ModelMetadata
    new_metadata = ModelMetadata(
        name=name,
        version=new_version,
        minio_bucket=upload_info["bucket"],
        minio_object_key=new_object_key,
        file_size_bytes=upload_info["size"],
        train_script_object_key=new_train_key,
        description=source_model.description,
        algorithm=source_model.algorithm,
        mlflow_run_id=None,
        accuracy=new_accuracy,
        f1_score=new_f1,
        features_count=source_model.features_count,
        classes=source_model.classes,
        training_params=source_model.training_params,
        training_dataset=(
            f"{source_model.training_dataset or name} "
            f"[{payload.start_date} → {payload.end_date}]"
        ),
        feature_baseline=source_model.feature_baseline,
        confidence_threshold=source_model.confidence_threshold,
        tags=source_model.tags,
        webhook_url=source_model.webhook_url,
        trained_by=user.username,
        training_date=datetime.now(timezone.utc),
        user_id_creator=user.id,
        is_active=True,
        is_production=False,
    )
    db.add(new_metadata)
    await db.flush()
    await DBService.log_model_history(
        db, new_metadata, HistoryActionType.CREATED, user.id, user.username
    )

    # 10. Passer en production si demandé
    if payload.set_production:
        other_result = await db.execute(
            select(ModelMetadata).where(
                and_(
                    ModelMetadata.name == name,
                    ModelMetadata.version != new_version,
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
        new_metadata.is_production = True
        await db.flush()
        await DBService.log_model_history(
            db,
            new_metadata,
            HistoryActionType.SET_PRODUCTION,
            user.id,
            user.username,
            ["is_production"],
        )

    await db.commit()
    await db.refresh(new_metadata)

    logger.info(
        "Ré-entraînement réussi",
        model=name,
        source_version=version,
        new_version=new_version,
        set_production=payload.set_production,
    )

    return RetrainResponse(
        model_name=name,
        source_version=version,
        new_version=new_version,
        success=True,
        stdout=stdout_text,
        stderr=stderr_text,
        new_model_metadata=ModelCreateResponse(
            **{c.name: getattr(new_metadata, c.name) for c in new_metadata.__table__.columns},
            creator_username=user.username,
        ),
    )


@router.get("/models/{name}/ab-compare", response_model=ABCompareResponse)
async def get_ab_comparison(
    name: str,
    days: int = Query(30, ge=1, le=90, description="Fenêtre d'analyse en jours (max 90)"),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Comparaison côte-à-côte des versions A/B et shadow d'un modèle sur une fenêtre glissante.

    Pour chaque version ayant généré des prédictions : total, prédictions shadow, taux d'erreur,
    latence (avg / p95), distribution des labels, et taux de concordance shadow/production (si id_obs).

    Nécessite un token Bearer valide.
    """
    # Vérifier que le modèle existe
    all_metas_result = await db.execute(
        select(ModelMetadata).where(
            and_(ModelMetadata.name == name, ModelMetadata.is_active.is_(True))
        )
    )
    all_metas = all_metas_result.scalars().all()
    if not all_metas:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle '{name}' introuvable ou inactif.",
        )

    meta_by_version = {m.version: m for m in all_metas}

    raw_stats = await DBService.get_ab_comparison_stats(db, name, days=days)
    agreement_by_version = await DBService.get_shadow_agreement_rate(db, name, days=days)

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

    return ABCompareResponse(
        model_name=name,
        period_days=days,
        versions=versions_out,
    )


@router.get("/models/{name}/{version}", response_model=ModelGetResponse)
async def get_model(
    name: str,
    version: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Retourne les métadonnées complètes d'un modèle (name + version).

    Tente de charger le modèle en mémoire (depuis MLflow ou MinIO) :
    - Si le chargement réussit : `model_loaded=true`, `model_type` et `feature_names` renseignés.
    - Si le chargement échoue : `model_loaded=false` et `load_instructions` contient
      les informations nécessaires pour charger le modèle manuellement en Python.
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
            detail=f"Modèle '{name}' version '{version}' introuvable.",
        )

    # Tenter de charger le modèle
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
        # Le modèle n'a pas pu être chargé — construire les instructions manuelles
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
                    f"import pickle\n"
                    f"client = Minio('localhost:9002', access_key='...', secret_key='...', secure=False)\n"
                    f"response = client.get_object('{model_meta.minio_bucket}', '{model_meta.minio_object_key}')\n"
                    f"model = pickle.loads(response.read())"
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
        training_dataset=model_meta.training_dataset,
        trained_by=model_meta.trained_by,
        training_date=model_meta.training_date,
        accuracy=model_meta.accuracy,
        f1_score=model_meta.f1_score,
        precision=model_meta.precision,
        recall=model_meta.recall,
        mlflow_run_id=model_meta.mlflow_run_id,
        minio_bucket=model_meta.minio_bucket,
        minio_object_key=model_meta.minio_object_key,
        file_size_bytes=model_meta.file_size_bytes,
        file_hash=model_meta.file_hash,
        user_id_creator=model_meta.user_id_creator,
        creator_username=model_meta.creator.username if model_meta.creator else None,
        is_active=model_meta.is_active,
        is_production=model_meta.is_production,
        created_at=model_meta.created_at,
        updated_at=model_meta.updated_at,
        deprecated_at=model_meta.deprecated_at,
        model_loaded=model_loaded,
        model_type=model_type,
        feature_names=feature_names,
        load_instructions=load_instructions,
    )


@router.post("/models", response_model=ModelCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_model(
    name: str = Form(..., description="Nom unique du modèle"),
    version: str = Form(..., description="Version du modèle (ex: 1.0.0)"),
    file: Optional[UploadFile] = File(
        None, description="Fichier .pkl (optionnel si mlflow_run_id fourni)"
    ),
    description: Optional[str] = Form(None),
    algorithm: Optional[str] = Form(None),
    mlflow_run_id: Optional[str] = Form(None),
    accuracy: Optional[float] = Form(None),
    f1_score: Optional[float] = Form(None),
    features_count: Optional[int] = Form(None),
    classes: Optional[str] = Form(None, description="JSON array ex: [0, 1, 2]"),
    training_params: Optional[str] = Form(None, description="JSON object"),
    training_dataset: Optional[str] = Form(None),
    feature_baseline: Optional[str] = Form(
        None,
        description='JSON: {"feature": {"mean": float, "std": float, "min": float, "max": float}}',
    ),
    tags: Optional[str] = Form(
        None, description='JSON array de tags ex: ["production", "finance"]'
    ),
    webhook_url: Optional[str] = Form(
        None, description="URL de callback POST après chaque prédiction"
    ),
    train_file: Optional[UploadFile] = File(
        None,
        description=(
            "Script Python de ré-entraînement (train.py). "
            "Doit référencer TRAIN_START_DATE, TRAIN_END_DATE, OUTPUT_MODEL_PATH "
            "et contenir un appel pickle.dump/joblib.dump/save_model."
        ),
    ),
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Crée un nouveau modèle et l'enregistre en base.

    - **name** + **version** doivent être uniques — retourne 409 si la combinaison existe déjà.
    - **file** : fichier `.pkl` requis si `mlflow_run_id` n'est pas fourni.
      Si `mlflow_run_id` est fourni, MLflow stocke déjà le modèle dans MinIO — pas de doublon.
    - **mlflow_run_id** : ID du run MLflow. Permet de charger le modèle via MLflow à la prédiction.
    - **train_file** : script `train.py` optionnel permettant le ré-entraînement automatique.
      Doit respecter le contrat d'interface (TRAIN_START_DATE, TRAIN_END_DATE, OUTPUT_MODEL_PATH).

    Nécessite un token Bearer valide.
    """
    # Vérifier l'unicité name + version
    result = await db.execute(
        select(ModelMetadata).where(
            and_(ModelMetadata.name == name, ModelMetadata.version == version)
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Un modèle '{name}' version '{version}' existe déjà.",
        )

    minio_bucket = None
    minio_object_key = None
    file_size_bytes = None

    if file is not None:
        # Lire et uploader le fichier vers MinIO
        model_bytes = await file.read()
        max_bytes = settings.MAX_MODEL_SIZE_MB * 1024 * 1024
        if len(model_bytes) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Le fichier dépasse la taille maximale autorisée "
                    f"({settings.MAX_MODEL_SIZE_MB} MB). "
                    f"Taille reçue : {len(model_bytes) / 1024 / 1024:.1f} MB."
                ),
            )
        if not model_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Le fichier est vide.",
            )
        object_name = f"{name}/v{version}.pkl"
        upload_info = minio_service.upload_model_bytes(model_bytes, object_name)
        minio_bucket = upload_info["bucket"]
        minio_object_key = object_name
        file_size_bytes = upload_info["size"]
    elif not mlflow_run_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Fournir un fichier .pkl ou un mlflow_run_id.",
        )

    # --- Traitement du script d'entraînement (optionnel) ---
    train_script_object_key = None
    if train_file is not None:
        train_bytes = await train_file.read()
        if not train_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Le fichier train.py est vide.",
            )
        try:
            train_source = train_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="train.py doit être un fichier texte UTF-8 valide.",
            )
        validation_error = _validate_train_script(train_source)
        if validation_error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Script train.py invalide : {validation_error}",
            )
        train_object_name = f"{name}/v{version}_train.py"
        minio_service.upload_file_bytes(
            train_bytes, train_object_name, content_type="text/x-python"
        )
        train_script_object_key = train_object_name
        logger.info(
            "Script train.py uploadé", object_name=train_object_name, model=name, version=version
        )

    # Désérialiser les champs JSON optionnels
    classes_parsed = json.loads(classes) if classes else None
    training_params_parsed = json.loads(training_params) if training_params else None
    feature_baseline_parsed = json.loads(feature_baseline) if feature_baseline else None
    tags_parsed = json.loads(tags) if tags else None

    # Créer l'entrée en base
    metadata = ModelMetadata(
        name=name,
        version=version,
        minio_bucket=minio_bucket,
        minio_object_key=minio_object_key,
        file_size_bytes=file_size_bytes,
        description=description,
        algorithm=algorithm,
        mlflow_run_id=mlflow_run_id,
        accuracy=accuracy,
        f1_score=f1_score,
        features_count=features_count,
        classes=classes_parsed,
        training_params=training_params_parsed,
        training_dataset=training_dataset,
        feature_baseline=feature_baseline_parsed,
        tags=tags_parsed,
        webhook_url=webhook_url,
        train_script_object_key=train_script_object_key,
        trained_by=user.username,
        user_id_creator=user.id,
        is_active=True,
        is_production=False,
    )
    db.add(metadata)
    await db.flush()  # obtenir l'id avant le snapshot
    await DBService.log_model_history(
        db, metadata, HistoryActionType.CREATED, user.id, user.username
    )
    await db.commit()
    await db.refresh(metadata)

    return ModelCreateResponse(
        **{c.name: getattr(metadata, c.name) for c in metadata.__table__.columns},
        creator_username=user.username,
    )


@router.patch("/models/{name}/{version}", response_model=ModelCreateResponse)
async def update_model(
    name: str,
    version: str,
    payload: ModelUpdateInput,
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Met à jour les métadonnées d'un modèle (name + version).

    Champs modifiables : `description`, `is_production`, `accuracy`, `features_count`, `classes`.

    - Si **is_production** passe à `true`, toutes les autres versions du même modèle
      passent automatiquement à `false`.

    Nécessite un token Bearer valide.
    """
    # Récupérer le modèle cible avec son créateur
    result = await db.execute(
        select(ModelMetadata)
        .options(selectinload(ModelMetadata.creator))
        .where(and_(ModelMetadata.name == name, ModelMetadata.version == version))
    )
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle '{name}' version '{version}' introuvable.",
        )

    # Snapshot avant modification (pour détecter les champs réellement modifiés)
    pre_snapshot = _build_snapshot(model)
    demoted_versions = []

    # Si is_production passe à True → retirer is_production des autres versions
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
            other.is_production = False
            demoted_versions.append(other)

    # Appliquer uniquement les champs fournis (non-None)
    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(model, field, value)

    await db.flush()

    # Validation : la somme des traffic_weight pour les versions ab_test du modèle doit rester ≤ 1.0
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
                    f"La somme des traffic_weight pour les versions A/B de '{name}' "
                    f"dépasse 1.0 (somme actuelle = {total_weight:.3f}). "
                    "Réduisez le poids de certaines versions avant d'en augmenter d'autres."
                ),
            )

    # Déterminer les champs réellement modifiés
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
    await db.refresh(model)

    return ModelCreateResponse(
        **{c.name: getattr(model, c.name) for c in model.__table__.columns},
        creator_username=model.creator.username if model.creator else None,
    )


# ---------------------------------------------------------------------------
# Helpers suppression
# ---------------------------------------------------------------------------


def _delete_mlflow_run(run_id: str) -> bool:
    """Supprime le run MLflow. Retourne False si MLflow est indisponible."""
    try:
        from mlflow.tracking import MlflowClient

        MlflowClient().delete_run(run_id)
        return True
    except Exception as e:
        logger.warning("MLflow suppression impossible", run_id=run_id, error=str(e))
        return False


def _delete_minio_object(object_key: str) -> bool:
    """Supprime l'objet MinIO. Retourne False si MinIO est indisponible."""
    try:
        return minio_service.delete_model(object_key)
    except Exception as e:
        logger.warning("MinIO suppression impossible", object_key=object_key, error=str(e))
        return False


# ---------------------------------------------------------------------------
# DELETE routes
# ---------------------------------------------------------------------------


@router.delete("/models/{name}/{version}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_version(
    name: str,
    version: str,
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Supprime une version spécifique d'un modèle.

    - Supprime l'entrée en base PostgreSQL.
    - Supprime le run MLflow associé (si `mlflow_run_id` renseigné).
    - Supprime l'objet `.pkl` dans MinIO.

    Retourne **204 No Content** en cas de succès.
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
            detail=f"Modèle '{name}' version '{version}' introuvable.",
        )

    if model.mlflow_run_id:
        _delete_mlflow_run(model.mlflow_run_id)

    if model.minio_object_key:
        _delete_minio_object(model.minio_object_key)

    # Logger la suppression avant de supprimer l'objet ORM
    await DBService.log_model_history(db, model, HistoryActionType.DELETED, user.id, user.username)
    await db.delete(model)
    await db.commit()


@router.delete("/models/{name}", response_model=ModelDeleteResponse)
async def delete_model_all_versions(
    name: str,
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Supprime toutes les versions d'un modèle.

    - Supprime toutes les entrées PostgreSQL pour ce nom.
    - Supprime chaque run MLflow associé.
    - Supprime chaque objet `.pkl` dans MinIO.

    Retourne un résumé des suppressions effectuées.
    """
    result = await db.execute(select(ModelMetadata).where(ModelMetadata.name == name))
    models = result.scalars().all()

    if not models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Aucun modèle trouvé avec le nom '{name}'.",
        )

    deleted_versions = []
    mlflow_runs_deleted = []
    minio_objects_deleted = []

    for model in models:
        deleted_versions.append(model.version)

        if model.mlflow_run_id and _delete_mlflow_run(model.mlflow_run_id):
            mlflow_runs_deleted.append(model.mlflow_run_id)

        if model.minio_object_key and _delete_minio_object(model.minio_object_key):
            minio_objects_deleted.append(model.minio_object_key)

        await DBService.log_model_history(
            db, model, HistoryActionType.DELETED, user.id, user.username
        )
        await db.delete(model)

    await db.commit()

    return ModelDeleteResponse(
        name=name,
        deleted_versions=deleted_versions,
        mlflow_runs_deleted=mlflow_runs_deleted,
        minio_objects_deleted=minio_objects_deleted,
    )
