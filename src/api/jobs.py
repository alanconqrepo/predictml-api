"""
ARQ job supervision endpoints.

GET  /jobs              — paginated list of task_runs (admin)
GET  /jobs/{job_id}     — full status of a job
GET  /jobs/{job_id}/logs — SSE streaming of real-time logs (or archived logs)
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
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    task_type: Optional[str] = Query(
        None, description="Filter by type (retrain, scheduled_retrain)"
    ),
    job_status: Optional[str] = Query(None, alias="status", description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    _: object = Depends(require_admin),
):
    """
    Lists task_runs with optional filters.
    Reserved for administrators.
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
    Returns the full status of a job by its ID.
    Reserved for administrators.
    """
    row = await db.get(TaskRun, job_id)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
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
    Streams job logs via Server-Sent Events (SSE).

    - **Running job**: lines arrive as execution progresses (300 ms polling).
    - **Completed job**: archived logs are returned in a single pass.
    - **Non-existent job**: 404.

    Consume in JavaScript:
    ```js
    const es = new EventSource('/jobs/<uuid>/logs');
    es.onmessage = (e) => console.log(e.data);
    ```

    Or with curl:
    ```bash
    curl -N -H "Authorization: Bearer <admin_token>" /jobs/<uuid>/logs
    ```
    """
    row = await db.get(TaskRun, job_id)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    str_job_id = str(job_id)

    # If the job is done and logs are archived, return them directly
    if row.status in _TERMINAL_STATUSES and row.logs:
        return StreamingResponse(
            _static_log_stream(row.logs),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Otherwise, stream from Redis in real time
    return StreamingResponse(
        _live_log_stream(str_job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _static_log_stream(logs: str):
    """Emits archived logs then closes the stream."""
    for line in logs.splitlines():
        yield f"data: {line}\n\n"
    yield "event: done\ndata: job completed\n\n"


async def _live_log_stream(job_id: str):
    """
    Tails Redis logs in real time + polls DB for terminal status.

    Algorithm:
    1. Read new lines from the Redis list (cursor → end)
    2. Emit each line as an SSE event
    3. Wait 300 ms
    4. Check if the job is in a terminal state
    5. Repeat until terminal, then emit a 'done' event
    """
    from src.services.model_service import model_service

    cursor = 0
    max_iterations = 3600  # safety: 3600 × 300 ms = 18 min max

    try:
        redis = await model_service._get_redis()
    except Exception:
        yield "event: error\ndata: Redis unavailable\n\n"
        return

    from src.db.database import AsyncSessionLocal
    from src.db.models.task_run import TaskRun as _TaskRun

    for _ in range(max_iterations):
        # Read new Redis lines
        try:
            lines = await redis.lrange(f"retrain_logs:{job_id}", cursor, -1)
            for raw_line in lines:
                decoded = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                yield f"data: {decoded}\n\n"
            cursor += len(lines)
        except Exception as exc:
            logger.warning("Redis log read error", job_id=job_id, error=str(exc))

        # Check status in DB
        try:
            async with AsyncSessionLocal() as db:
                row = await db.get(_TaskRun, job_id)
            if row and row.status in _TERMINAL_STATUSES:
                # Read one last time to avoid missing the last lines
                try:
                    lines = await redis.lrange(f"retrain_logs:{job_id}", cursor, -1)
                    for raw_line in lines:
                        decoded = (
                            raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                        )
                        yield f"data: {decoded}\n\n"
                except Exception:
                    pass
                # Emit the done event with the status
                yield f"event: done\ndata: {json.dumps({'status': row.status, 'new_version': row.new_version})}\n\n"
                return
        except Exception as exc:
            logger.warning("DB status check error", job_id=job_id, error=str(exc))

        await asyncio.sleep(0.3)

    # Safety timeout
    yield "event: timeout\ndata: stream timeout after 18 min\n\n"


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
