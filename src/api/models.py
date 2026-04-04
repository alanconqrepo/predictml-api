"""
Endpoints pour la gestion des modèles
"""
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import verify_token
from src.db.database import get_db
from src.db.models import ModelMetadata, User
from src.schemas.model import ModelCreateResponse, ModelUpdateInput
from src.services.minio_service import minio_service
from src.services.model_service import model_service

router = APIRouter(tags=["models"])


@router.get("/models", response_model=List[Dict[str, Any]])
async def list_models(db: AsyncSession = Depends(get_db)):
    """
    Liste tous les modèles disponibles depuis la base de données

    Returns:
        Liste des modèles actifs avec leurs métadonnées
    """
    models = await model_service.get_available_models(db)
    return models


@router.get("/models/cached")
async def list_cached_models():
    """
    Liste les modèles actuellement en cache mémoire

    Returns:
        Liste des object keys MinIO en cache
    """
    cached = model_service.get_cached_models()
    return {
        "cached_models": cached,
        "count": len(cached)
    }


@router.post("/models", response_model=ModelCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_model(
    file: UploadFile = File(..., description="Fichier .pkl du modèle sérialisé"),
    name: str = Form(..., description="Nom unique du modèle"),
    version: str = Form(..., description="Version du modèle (ex: 1.0.0)"),
    description: Optional[str] = Form(None),
    algorithm: Optional[str] = Form(None),
    mlflow_run_id: Optional[str] = Form(None),
    accuracy: Optional[float] = Form(None),
    f1_score: Optional[float] = Form(None),
    features_count: Optional[int] = Form(None),
    classes: Optional[str] = Form(None, description="JSON array ex: [0, 1, 2]"),
    training_params: Optional[str] = Form(None, description="JSON object"),
    training_dataset: Optional[str] = Form(None),
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Crée un nouveau modèle : upload vers MinIO + enregistrement en base.

    - **name** doit être unique — retourne 409 si le nom existe déjà.
    - **file** : fichier `.pkl` produit par `pickle.dumps(model)`.
    - **mlflow_run_id** : `mlflow.active_run().info.run_id` depuis le script d'entraînement.

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

    # Lire le contenu du fichier
    model_bytes = await file.read()
    if not model_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le fichier est vide.",
        )

    # Upload vers MinIO
    object_name = f"{name}/v{version}.pkl"
    upload_info = minio_service.upload_model_bytes(model_bytes, object_name)

    # Désérialiser les champs JSON optionnels
    classes_parsed = json.loads(classes) if classes else None
    training_params_parsed = json.loads(training_params) if training_params else None

    # Créer l'entrée en base
    metadata = ModelMetadata(
        name=name,
        version=version,
        minio_bucket=upload_info["bucket"],
        minio_object_key=object_name,
        file_size_bytes=upload_info["size"],
        description=description,
        algorithm=algorithm,
        mlflow_run_id=mlflow_run_id,
        accuracy=accuracy,
        f1_score=f1_score,
        features_count=features_count,
        classes=classes_parsed,
        training_params=training_params_parsed,
        training_dataset=training_dataset,
        trained_by=user.username,
        is_active=True,
        is_production=False,
    )
    db.add(metadata)
    await db.commit()
    await db.refresh(metadata)

    return metadata


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
    # Récupérer le modèle cible
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

    # Si is_production passe à True → retirer is_production des autres versions
    if payload.is_production is True:
        other_versions = await db.execute(
            select(ModelMetadata).where(
                and_(
                    ModelMetadata.name == name,
                    ModelMetadata.version != version,
                    ModelMetadata.is_production == True,
                )
            )
        )
        for other in other_versions.scalars().all():
            other.is_production = False

    # Appliquer uniquement les champs fournis (non-None)
    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(model, field, value)

    await db.commit()
    await db.refresh(model)

    return model
