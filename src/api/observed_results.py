"""
Endpoints pour les résultats observés
"""

import csv
import io
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import verify_token
from src.db.database import get_db
from src.db.models import User
from src.schemas.observed_result import (
    CSVParseError,
    CSVUploadResponse,
    ObservedResultResponse,
    ObservedResultsListResponse,
    ObservedResultsStatsResponse,
    ObservedResultsUpsertRequest,
    ObservedResultsUpsertResponse,
)
from src.services.db_service import DBService

router = APIRouter(tags=["observed-results"])

_CSV_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)

_MAX_CSV_SIZE = 10 * 1024 * 1024  # 10 MB


def _parse_date(value: str) -> Optional[datetime]:
    for fmt in _CSV_DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


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


@router.post(
    "/observed-results/upload-csv",
    response_model=CSVUploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_observed_results_csv(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Importe des résultats observés depuis un fichier CSV (multipart/form-data).

    Format attendu : `id_obs, model_name, observed_result, date_time`

    - **model_name** (form) : écrase la colonne `model_name` du CSV si fourni
    - Taille max : 10 MB
    - Succès partiel : les lignes valides sont importées, les erreurs sont listées

    Nécessite un token Bearer valide.
    """
    content = await file.read()
    if len(content) > _MAX_CSV_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Fichier trop volumineux (max 10 MB)",
        )

    text = content.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    valid_records = []
    parse_errors: list[CSVParseError] = []

    for row_idx, row in enumerate(reader, start=2):
        id_obs = (row.get("id_obs") or "").strip()
        if not id_obs:
            parse_errors.append(CSVParseError(row=row_idx, reason="missing id_obs"))
            continue

        row_model = model_name or (row.get("model_name") or "").strip()
        if not row_model:
            parse_errors.append(CSVParseError(row=row_idx, reason="missing model_name"))
            continue

        raw_result = (row.get("observed_result") or "").strip()
        if not raw_result:
            parse_errors.append(CSVParseError(row=row_idx, reason="missing observed_result"))
            continue

        raw_dt = (row.get("date_time") or "").strip()
        if not raw_dt:
            parse_errors.append(CSVParseError(row=row_idx, reason="missing date_time"))
            continue

        dt = _parse_date(raw_dt)
        if dt is None:
            parse_errors.append(CSVParseError(row=row_idx, reason="invalid date format"))
            continue

        try:
            obs_val: float | int | str = int(raw_result)
        except ValueError:
            try:
                obs_val = float(raw_result)
            except ValueError:
                obs_val = raw_result

        valid_records.append(
            {
                "id_obs": id_obs,
                "model_name": row_model,
                "observed_result": obs_val,
                "date_time": dt,
                "user_id": user.id,
            }
        )

    upserted = 0
    if valid_records:
        upserted = await DBService.upsert_observed_results(db, valid_records)

    return CSVUploadResponse(
        upserted=upserted,
        skipped_rows=len(parse_errors),
        parse_errors=parse_errors,
        filename=file.filename or "",
    )


@router.get("/observed-results/stats", response_model=ObservedResultsStatsResponse)
async def get_observed_results_stats(
    model_name: Optional[str] = Query(None, description="Filtrer par modèle ; omis = global"),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Taux de couverture du ground truth : combien de prédictions ont un résultat observé.

    - **model_name** : si fourni, retourne les stats du modèle + breakdown par version.
    - Si omis, retourne les stats globales + breakdown par modèle.

    Nécessite un token Bearer valide.
    """
    stats = await DBService.get_observed_results_stats(db, model_name=model_name)
    return ObservedResultsStatsResponse(**stats)


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
