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
from src.schemas.model import ModelCreateResponse, ModelDeleteResponse, ModelUpdateInput
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
    name: str = Form(..., description="Nom unique du modèle"),
    version: str = Form(..., description="Version du modèle (ex: 1.0.0)"),
    file: Optional[UploadFile] = File(None, description="Fichier .pkl (optionnel si mlflow_run_id fourni)"),
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
        print(f"⚠️  MLflow suppression impossible pour run {run_id}: {e}")
        return False


def _delete_minio_object(object_key: str) -> bool:
    """Supprime l'objet MinIO. Retourne False si MinIO est indisponible."""
    try:
        return minio_service.delete_model(object_key)
    except Exception as e:
        print(f"⚠️  MinIO suppression impossible pour {object_key}: {e}")
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
    result = await db.execute(
        select(ModelMetadata).where(ModelMetadata.name == name)
    )
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
