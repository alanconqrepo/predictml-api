"""
Endpoints de supervision des jobs ARQ.

GET  /jobs              — liste paginée des task_runs (admin)
GET  /jobs/{job_id}     — statut complet d'un job
GET  /jobs/{job_id}/logs — streaming SSE des logs en temps réel (ou logs archivés)
"""

import asyncio
import json
from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import require_admin
from src.db.database import get_db
from src.db.models.task_run import TaskRun
from src.schemas.task_run import TaskRunList, TaskRunStatus

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["jobs"])

_TERMINAL_STATUSES = {"success", "failed", "cancelled"}


# ---------------------------------------------------------------------------
# GET /jobs
# ---------------------------------------------------------------------------


@router.get("/jobs", response_model=TaskRunList)
async def list_jobs(
    model_name: Optional[str] = Query(None, description="Filtrer par nom de modèle"),
    task_type: Optional[str] = Query(
        None, description="Filtrer par type (retrain, scheduled_retrain)"
    ),
    job_status: Optional[str] = Query(None, alias="status", description="Filtrer par statut"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    _: object = Depends(require_admin),
):
    """
    Liste les task_runs avec filtres optionnels.
    Réservé aux administrateurs.
    """
    from sqlalchemy import func

    stmt = select(TaskRun)
    count_stmt = select(func.count()).select_from(TaskRun)

    if model_name:
        stmt = stmt.where(TaskRun.model_name == model_name)
        count_stmt = count_stmt.where(TaskRun.model_name == model_name)
    if task_type:
        stmt = stmt.where(TaskRun.task_type == task_type)
        count_stmt = count_stmt.where(TaskRun.task_type == task_type)
    if job_status:
        stmt = stmt.where(TaskRun.status == job_status)
        count_stmt = count_stmt.where(TaskRun.status == job_status)

    total = (await db.execute(count_stmt)).scalar_one()
    rows = (
        (await db.execute(stmt.order_by(TaskRun.enqueued_at.desc()).limit(limit).offset(offset)))
        .scalars()
        .all()
    )

    items = [_task_run_to_schema(r) for r in rows]
    return TaskRunList(items=items, total=total, limit=limit, offset=offset)


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}", response_model=TaskRunStatus)
async def get_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    _: object = Depends(require_admin),
):
    """
    Retourne le statut complet d'un job par son ID.
    Réservé aux administrateurs.
    """
    row = await db.get(TaskRun, job_id)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' introuvable.",
        )
    return _task_run_to_schema(row)


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/logs  — SSE streaming
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}/logs")
async def stream_job_logs(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    _: object = Depends(require_admin),
):
    """
    Stream des logs d'un job en Server-Sent Events (SSE).

    - **Job en cours** : les lignes arrivent au fil de l'exécution (polling 300 ms).
    - **Job terminé** : les logs archivés sont retournés en une seule passe.
    - **Job inexistant** : 404.

    Consommer en JavaScript :
    ```js
    const es = new EventSource('/jobs/<uuid>/logs');
    es.onmessage = (e) => console.log(e.data);
    ```

    Ou avec curl :
    ```bash
    curl -N -H "Authorization: Bearer <admin_token>" /jobs/<uuid>/logs
    ```
    """
    row = await db.get(TaskRun, job_id)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' introuvable.",
        )

    str_job_id = str(job_id)

    # Si le job est terminé et que les logs sont archivés, les retourner directement
    if row.status in _TERMINAL_STATUSES and row.logs:
        return StreamingResponse(
            _static_log_stream(row.logs),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Sinon, streamer depuis Redis en temps réel
    return StreamingResponse(
        _live_log_stream(str_job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _static_log_stream(logs: str):
    """Émet les logs archivés puis ferme le stream."""
    for line in logs.splitlines():
        yield f"data: {line}\n\n"
    yield "event: done\ndata: job completed\n\n"


async def _live_log_stream(job_id: str):
    """
    Tail des logs Redis en temps réel + polling DB pour le statut terminal.

    Algorithme :
    1. Lire les nouvelles lignes depuis la liste Redis (cursor → end)
    2. Émettre chaque ligne comme event SSE
    3. Attendre 300 ms
    4. Vérifier si le job est dans un état terminal
    5. Répéter jusqu'à terminal, puis émettre un event 'done'
    """
    from src.services.model_service import model_service

    cursor = 0
    max_iterations = 3600  # sécurité : 3600 × 300 ms = 18 min max

    try:
        redis = await model_service._get_redis()
    except Exception:
        yield "event: error\ndata: Redis indisponible\n\n"
        return

    from src.db.database import AsyncSessionLocal
    from src.db.models.task_run import TaskRun as _TaskRun

    for _ in range(max_iterations):
        # Lire les nouvelles lignes Redis
        try:
            lines = await redis.lrange(f"retrain_logs:{job_id}", cursor, -1)
            for raw_line in lines:
                decoded = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                yield f"data: {decoded}\n\n"
            cursor += len(lines)
        except Exception as exc:
            logger.warning("Erreur lecture logs Redis", job_id=job_id, error=str(exc))

        # Vérifier le statut dans la DB
        try:
            async with AsyncSessionLocal() as db:
                row = await db.get(_TaskRun, job_id)
            if row and row.status in _TERMINAL_STATUSES:
                # Lire une dernière fois pour ne pas rater les dernières lignes
                try:
                    lines = await redis.lrange(f"retrain_logs:{job_id}", cursor, -1)
                    for raw_line in lines:
                        decoded = (
                            raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                        )
                        yield f"data: {decoded}\n\n"
                except Exception:
                    pass
                # Émettre l'event de fin avec le statut
                yield f"event: done\ndata: {json.dumps({'status': row.status, 'new_version': row.new_version})}\n\n"
                return
        except Exception as exc:
            logger.warning("Erreur vérification statut DB", job_id=job_id, error=str(exc))

        await asyncio.sleep(0.3)

    # Timeout de sécurité
    yield "event: timeout\ndata: stream timeout après 18 min\n\n"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _task_run_to_schema(row: TaskRun) -> TaskRunStatus:
    duration = None
    if row.started_at and row.completed_at:
        duration = (row.completed_at - row.started_at).total_seconds()
    return TaskRunStatus(
        job_id=row.id,
        task_type=row.task_type,
        model_name=row.model_name,
        model_version=row.model_version,
        new_version=row.new_version,
        triggered_by=row.triggered_by,
        status=row.status,
        enqueued_at=row.enqueued_at,
        started_at=row.started_at,
        completed_at=row.completed_at,
        duration_seconds=duration,
        result=row.result,
        error=row.error,
    )
