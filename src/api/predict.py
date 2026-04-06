"""
Endpoints pour les prédictions
"""

import logging
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import check_prediction_rate_limit, verify_token
from src.db.database import get_db
from src.db.models import Prediction, User
from src.schemas.prediction import (
    BatchPredictionInput,
    BatchPredictionOutput,
    BatchPredictionResultItem,
    PredictionInput,
    PredictionOutput,
    PredictionResponse,
    PredictionsListResponse,
)
from src.services.db_service import DBService
from src.services.model_service import model_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predictions"])


@router.get("/predictions", response_model=PredictionsListResponse)
async def get_predictions(
    name: str = Query(..., description="Nom du modèle"),
    start: datetime = Query(
        ..., description="Début de la période (ISO 8601, ex: 2024-01-01T00:00:00)"
    ),
    end: datetime = Query(..., description="Fin de la période (ISO 8601, ex: 2024-12-31T23:59:59)"),
    version: Optional[str] = Query(None, description="Version du modèle (optionnel)"),
    user: Optional[str] = Query(None, description="Nom d'utilisateur (optionnel)"),
    limit: int = Query(100, ge=1, le=1000, description="Nombre max de résultats"),
    offset: int = Query(0, ge=0, description="Décalage pour la pagination"),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Retourne l'historique des prédictions avec filtres.

    - **name** : nom du modèle — obligatoire
    - **start** / **end** : plage datetime — obligatoire
    - **version** : version du modèle — optionnel
    - **user** : nom d'utilisateur — optionnel
    - **limit** / **offset** : pagination (défaut : 100 résultats, max 1000)

    Nécessite un token Bearer valide.
    """
    if start > end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="'start' doit être antérieur à 'end'.",
        )

    predictions, total = await DBService.get_predictions(
        db=db,
        model_name=name,
        start=start,
        end=end,
        model_version=version,
        username=user,
        limit=limit,
        offset=offset,
    )

    return PredictionsListResponse(
        total=total,
        limit=limit,
        offset=offset,
        predictions=[
            PredictionResponse(
                id=p.id,
                model_name=p.model_name,
                model_version=p.model_version,
                id_obs=p.id_obs,
                input_features=p.input_features,
                prediction_result=p.prediction_result,
                probabilities=p.probabilities,
                response_time_ms=p.response_time_ms,
                timestamp=p.timestamp,
                status=p.status,
                error_message=p.error_message,
                username=p.user.username if p.user else None,
            )
            for p in predictions
        ],
    )


@router.post("/predict", response_model=PredictionOutput)
async def predict(
    input_data: PredictionInput,
    request: Request,
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

    Nécessite un token Bearer dans le header Authorization.
    Toutes les prédictions sont loggées dans la base de données.
    """
    start_time = time.time()
    prediction_result = None
    probability = None
    error_message = None

    try:
        # Charger le modèle demandé — version explicite, sinon production, sinon plus récente
        model_data = await model_service.load_model(
            db, input_data.model_name, input_data.model_version
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
        )

        return PredictionOutput(
            model_name=metadata.name,
            model_version=metadata.version,
            id_obs=input_data.id_obs,
            prediction=prediction_result,
            probability=probability,
            low_confidence=low_confidence,
        )

    except HTTPException:
        # Re-raise les HTTPException (404, etc.)
        raise

    except Exception as e:
        # Logger l'erreur
        response_time_ms = (time.time() - start_time) * 1000
        error_message = str(e)

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
            logger.error("Erreur lors du logging de la prédiction: %s", log_error)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction avec '{input_data.model_name}': {error_message}",
        )


@router.post("/predict-batch", response_model=BatchPredictionOutput)
async def predict_batch(
    input_data: BatchPredictionInput,
    request: Request,
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
            logger.error("Erreur lors du logging du batch en erreur: %s", log_error)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du batch avec '{input_data.model_name}': {error_message}",
        )
