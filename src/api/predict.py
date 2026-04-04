"""
Endpoints pour les prédictions
"""
import time
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import verify_token
from src.db.database import get_db
from src.db.models import User
from src.schemas.prediction import PredictionInput, PredictionOutput
from src.services.model_service import model_service
from src.services.db_service import DBService

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=PredictionOutput)
async def predict(
    input_data: PredictionInput,
    request: Request,
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db)
):
    """
    Fait une prédiction avec le modèle sklearn spécifié.

    - **model_name**: Nom du modèle à utiliser
    - **id_obs**: Identifiant de l'observation (optionnel, stocké en DB)
    - **features**: Features pour la prédiction — deux formats acceptés :
        - **liste** `[5.1, 3.5, 1.4, 0.2]` : l'ordre doit correspondre à celui du modèle
        - **dict** `{"sepal_length": 5.1, ...}` : le modèle doit exposer `feature_names_in_`
          (entraîné avec un DataFrame pandas). Les clés manquantes retournent une erreur 422.

    Nécessite un token Bearer dans le header Authorization.
    Toutes les prédictions sont loggées dans la base de données.
    """
    start_time = time.time()
    prediction_result = None
    probability = None
    error_message = None
    status_str = "success"

    try:
        # Charger le modèle demandé (depuis MinIO via cache)
        model_data = await model_service.load_model(db, input_data.model_name)
        model = model_data["model"]
        metadata = model_data["metadata"]

        # Convertir les features en array numpy selon le format fourni
        if isinstance(input_data.features, dict):
            if not hasattr(model, 'feature_names_in_'):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"Le modèle '{input_data.model_name}' ne possède pas l'attribut "
                        "'feature_names_in_'. Pour utiliser le format dict, le modèle doit "
                        "avoir été entraîné avec un DataFrame pandas (les noms de colonnes "
                        "sont alors automatiquement sauvegardés par sklearn)."
                    )
                )
            missing = set(model.feature_names_in_) - set(input_data.features.keys())
            if missing:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"Features manquantes dans la requête : {sorted(missing)}. "
                        f"Features attendues : {list(model.feature_names_in_)}"
                    )
                )
            X = np.array(
                [[input_data.features[name] for name in model.feature_names_in_]],
                dtype=object
            )
        else:
            X = np.array([input_data.features], dtype=object)

        # Faire la prédiction
        prediction = model.predict(X)[0]
        prediction_result = prediction.item() if hasattr(prediction, 'item') else prediction

        # Essayer d'obtenir les probabilités si le modèle le supporte
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(X)[0].tolist()

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
            id_obs=input_data.id_obs
        )

        return PredictionOutput(
            model_name=metadata.name,
            id_obs=input_data.id_obs,
            prediction=prediction_result,
            probability=probability
        )

    except HTTPException:
        # Re-raise les HTTPException (404, etc.)
        raise

    except Exception as e:
        # Logger l'erreur
        response_time_ms = (time.time() - start_time) * 1000
        error_message = str(e)
        status_str = "error"

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
                id_obs=input_data.id_obs
            )
        except Exception as log_error:
            print(f"❌ Erreur lors du logging: {log_error}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction avec '{input_data.model_name}': {error_message}"
        )
