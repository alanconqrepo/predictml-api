"""
Endpoints pour les résultats observés
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import verify_token
from src.db.database import get_db
from src.db.models import User
from src.schemas.observed_result import (
    ObservedResultsUpsertRequest,
    ObservedResultsUpsertResponse,
    ObservedResultsListResponse,
    ObservedResultResponse,
)
from src.services.db_service import DBService

router = APIRouter(tags=["observed-results"])


@router.post(
    "/observed-results",
    response_model=ObservedResultsUpsertResponse,
    status_code=status.HTTP_200_OK,
)
async def upsert_observed_results(
    body: ObservedResultsUpsertRequest,
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Insère ou écrase des résultats réellement observés.

    - Chaque entrée est identifiée par la paire **(id_obs, model_name)**.
    - Si la paire existe déjà, la ligne est **écrasée** (observed_result, date_time).
    - Le `user_id` enregistré est celui du token Bearer utilisé.

    Nécessite un token Bearer valide.
    """
    records = [
        {
            "id_obs": item.id_obs,
            "model_name": item.model_name,
            "observed_result": item.observed_result,
            "date_time": item.date_time.replace(tzinfo=None),
            "user_id": user.id,
        }
        for item in body.data
    ]

    upserted = await DBService.upsert_observed_results(db, records)
    return ObservedResultsUpsertResponse(upserted=upserted)


@router.get("/observed-results", response_model=ObservedResultsListResponse)
async def get_observed_results(
    model_name: Optional[str] = Query(None, description="Filtrer par nom de modèle"),
    id_obs: Optional[str] = Query(None, description="Filtrer par identifiant d'observation"),
    start: Optional[datetime] = Query(None, description="Date de début (ISO 8601)"),
    end: Optional[datetime] = Query(None, description="Date de fin (ISO 8601)"),
    limit: int = Query(100, ge=1, le=1000, description="Nombre max de résultats"),
    offset: int = Query(0, ge=0, description="Décalage pour la pagination"),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Retourne les résultats observés avec filtres optionnels.

    - **model_name** : nom du modèle — optionnel
    - **id_obs** : identifiant d'observation — optionnel
    - **start** / **end** : plage datetime sur date_time — optionnel
    - **limit** / **offset** : pagination (défaut : 100 résultats, max 1000)

    Nécessite un token Bearer valide.
    """
    if start and end and start > end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'start' doit être antérieur à 'end'.",
        )

    results, total = await DBService.get_observed_results(
        db=db,
        model_name=model_name,
        id_obs=id_obs,
        start=start,
        end=end,
        limit=limit,
        offset=offset,
    )

    return ObservedResultsListResponse(
        total=total,
        limit=limit,
        offset=offset,
        results=[
            ObservedResultResponse(
                id=r.id,
                id_obs=r.id_obs,
                model_name=r.model_name,
                observed_result=r.observed_result,
                date_time=r.date_time,
                username=r.user.username if r.user else None,
            )
            for r in results
        ],
    )
