"""
Endpoints pour la gestion des modèles
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.security import verify_token
from src.db.database import get_db
from src.db.models import ModelMetadata, User
from src.schemas.model import (
    ModelCreateResponse,
    ModelDeleteResponse,
    ModelGetResponse,
    ModelUpdateInput,
)
from src.services.minio_service import minio_service
from src.services.model_service import model_service

logger = logging.getLogger(__name__)

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
    return {"cached_models": cached, "count": len(cached)}


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
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Crée un nouveau modèle et l'enregistre en base.

    - **name** + **version** doivent être uniques — retourne 409 si la combinaison existe déjà.
    - **file** : fichier `.pkl` requis si `mlflow_run_id` n'est pas fourni.
      Si `mlflow_run_id` est fourni, MLflow stocke déjà le modèle dans MinIO — pas de doublon.
    - **mlflow_run_id** : ID du run MLflow. Permet de charger le modèle via MLflow à la prédiction.

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

    # Désérialiser les champs JSON optionnels
    classes_parsed = json.loads(classes) if classes else None
    training_params_parsed = json.loads(training_params) if training_params else None

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
        trained_by=user.username,
        user_id_creator=user.id,
        is_active=True,
        is_production=False,
    )
    db.add(metadata)
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

    # Appliquer uniquement les champs fournis (non-None)
    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(model, field, value)

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
        logger.warning("MLflow suppression impossible pour run %s: %s", run_id, e)
        return False


def _delete_minio_object(object_key: str) -> bool:
    """Supprime l'objet MinIO. Retourne False si MinIO est indisponible."""
    try:
        return minio_service.delete_model(object_key)
    except Exception as e:
        logger.warning("MinIO suppression impossible pour %s: %s", object_key, e)
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

        await db.delete(model)

    await db.commit()

    return ModelDeleteResponse(
        name=name,
        deleted_versions=deleted_versions,
        mlflow_runs_deleted=mlflow_runs_deleted,
        minio_objects_deleted=minio_objects_deleted,
    )
