"""
Observed results endpoints
"""

import csv
import io
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import require_admin, verify_token
from src.db.database import get_db, get_read_db
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
    Inserts or overwrites actually observed results.

    - Each entry is identified by the pair **(id_obs, model_name)**.
    - If the pair already exists, the row is **overwritten** (observed_result, date_time).
    - The recorded `user_id` is that of the Bearer token used.

    Requires a valid Bearer token.
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
    Imports observed results from a CSV file (multipart/form-data).

    Expected format: `id_obs, model_name, observed_result, date_time`

    - **model_name** (form): overrides the `model_name` column in the CSV if provided
    - Max size: 10 MB
    - Partial success: valid rows are imported, errors are listed

    Requires a valid Bearer token.
    """
    content = await file.read()
    if len(content) > _MAX_CSV_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File too large (max 10 MB)",
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


_EXPORT_PAGE_SIZE = 500


@router.get("/observed-results/export")
async def export_observed_results(
    model_name: Optional[str] = Query(None, description="Filter by model name (optional)"),
    start: datetime = Query(..., description="Start of period (ISO 8601)"),
    end: datetime = Query(..., description="End of period (ISO 8601)"),
    export_format: str = Query(
        "csv",
        alias="format",
        description="Export format: csv or jsonl (default: csv)",
    ),
    _auth: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Bulk export of observed results as CSV or JSONL via cursor-based streaming.

    - **model_name**: filter by model (optional — all models if absent)
    - **start** / **end**: datetime range — required
    - **format**: `csv` (default) or `jsonl`

    Returns a file as a direct download (Content-Disposition: attachment).

    Admin only.
    """
    if start > end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'start' must be before 'end'.",
        )
    if export_format not in ("csv", "jsonl"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'format' parameter must be 'csv' or 'jsonl'.",
        )

    fmt = export_format
    csv_cols = ["id_obs", "model_name", "observed_result", "date_time"]

    async def _generate():
        cursor: Optional[int] = None
        header_written = False

        while True:
            rows = await DBService.get_observed_results_for_export(
                db=db,
                model_name=model_name,
                start=start,
                end=end,
                limit=_EXPORT_PAGE_SIZE,
                cursor=cursor,
            )

            if not rows:
                if fmt == "csv" and not header_written:
                    buf = io.StringIO()
                    csv.writer(buf).writerow(csv_cols)
                    yield buf.getvalue()
                break

            if fmt == "csv":
                if not header_written:
                    buf = io.StringIO()
                    csv.writer(buf).writerow(csv_cols)
                    yield buf.getvalue()
                    header_written = True
                for row in rows:
                    buf = io.StringIO()
                    csv.writer(buf).writerow(
                        [
                            row.id_obs,
                            row.model_name,
                            json.dumps(row.observed_result),
                            row.date_time.isoformat() if row.date_time else None,
                        ]
                    )
                    yield buf.getvalue()
            else:
                for row in rows:
                    yield json.dumps(
                        {
                            "id_obs": row.id_obs,
                            "model_name": row.model_name,
                            "observed_result": row.observed_result,
                            "date_time": row.date_time.isoformat() if row.date_time else None,
                        }
                    ) + "\n"

            if len(rows) < _EXPORT_PAGE_SIZE:
                break
            cursor = rows[-1].id

    media_type = "text/csv" if fmt == "csv" else "application/x-ndjson"
    filename = f"observed_results_export.{fmt}"
    return StreamingResponse(
        _generate(),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/observed-results/stats", response_model=ObservedResultsStatsResponse)
async def get_observed_results_stats(
    model_name: Optional[str] = Query(None, description="Filter by model; omitted = global"),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_read_db),
):
    """
    Ground truth coverage rate: how many predictions have an observed result.

    - **model_name**: if provided, returns model stats + breakdown by version.
    - If omitted, returns global stats + breakdown by model.

    Requires a valid Bearer token.
    """
    stats = await DBService.get_observed_results_stats(db, model_name=model_name)
    return ObservedResultsStatsResponse(**stats)


@router.get("/observed-results", response_model=ObservedResultsListResponse)
async def get_observed_results(
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    id_obs: Optional[str] = Query(None, description="Filter by observation identifier"),
    start: Optional[datetime] = Query(None, description="Start date (ISO 8601)"),
    end: Optional[datetime] = Query(None, description="End date (ISO 8601)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    _auth: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns observed results with optional filters.

    - **model_name**: model name — optional
    - **id_obs**: observation identifier — optional
    - **start** / **end**: datetime range on date_time — optional
    - **limit** / **offset**: pagination (default: 100 results, max 1000)

    Requires a valid Bearer token.
    """
    if start and end and start > end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'start' must be before 'end'.",
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
