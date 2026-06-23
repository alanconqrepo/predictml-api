"""
Endpoints for the model monitoring dashboard.

Two routes:
  GET /monitoring/overview          — global health of all models
  GET /monitoring/model/{name}      — full detail for one model
"""

import csv
import io
from datetime import datetime
from typing import Optional, Union

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import require_admin, verify_token
from src.db.database import get_read_db
from src.db.models import User
from src.schemas.alert_check import AlertCheckLogList, AlertCheckLogRead
from src.schemas.monitoring import (
    GlobalDashboard,
    GlobalStats,
    ModelDetailDashboard,
    ModelHealthSummary,
    MonitoringPeriod,
    TimeseriesPoint,
    VersionStats,
)
from src.services import drift_service
from src.services.db_service import DBService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Health thresholds derived from global stats
_ERROR_RATE_WARNING = 0.05  # 5% → warning
_ERROR_RATE_CRITICAL = 0.10  # 10% → critical

# Performance drift threshold (relative accuracy drop between 1st and 2nd half of period)
_PERF_DRIFT_WARNING = 0.05  # −5 pts → warning
_PERF_DRIFT_CRITICAL = 0.10  # −10 pts → critical


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _worst_health(*statuses: str) -> str:
    """Returns the most unfavourable status among ok/warning/critical/no_data."""
    order = {
        "ok": 0,
        "no_data": 0,
        "no_baseline": 0,
        "insufficient_data": 0,
        "warning": 1,
        "critical": 2,
    }
    return max(statuses, key=lambda s: order.get(s, 0))


def _error_rate_status(error_rate: float) -> str:
    if error_rate >= _ERROR_RATE_CRITICAL:
        return "critical"
    if error_rate >= _ERROR_RATE_WARNING:
        return "warning"
    return "ok"


def _performance_drift_status(perf_by_day: list[dict]) -> str:
    """
    Compares the 1st half of the period vs the 2nd half.
    Uses MAE for regression, accuracy for classification.
    Returns ok / warning / critical / no_data.
    """
    if len(perf_by_day) < 4:
        return "no_data"
    mid = len(perf_by_day) // 2
    use_mae = any(d.get("mae") is not None for d in perf_by_day)
    if use_mae:
        # For regression: rising MAE = degradation → invert to reuse the same logic
        first_half = [
            -d["mae"]
            for d in perf_by_day[:mid]
            if d["matched_count"] > 0 and d.get("mae") is not None
        ]
        second_half = [
            -d["mae"]
            for d in perf_by_day[mid:]
            if d["matched_count"] > 0 and d.get("mae") is not None
        ]
    else:
        first_half = [d["accuracy"] for d in perf_by_day[:mid] if d["matched_count"] > 0]
        second_half = [d["accuracy"] for d in perf_by_day[mid:] if d["matched_count"] > 0]
    if not first_half or not second_half:
        return "no_data"
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    drop = avg_first - avg_second  # positive = degradation
    if drop >= _PERF_DRIFT_CRITICAL:
        return "critical"
    if drop >= _PERF_DRIFT_WARNING:
        return "warning"
    return "ok"


async def _compute_feature_drift_status(
    db: AsyncSession,
    model_name: str,
    model_version: Optional[str],
    period_days: int,
    feature_baseline: Optional[dict],
) -> str:
    """Returns the feature drift status (ok/warning/critical/no_baseline/insufficient_data)."""
    if not feature_baseline:
        return "no_baseline"
    production_stats = await DBService.get_feature_production_stats(
        db, model_name, model_version, period_days
    )
    if not production_stats:
        return "no_data"
    features = drift_service.compute_feature_drift(feature_baseline, production_stats, min_count=10)
    return drift_service.summarize_drift(features, baseline_available=True)


async def _compute_output_drift_status(
    db: AsyncSession,
    model_name: str,
    model_version: Optional[str],
    period_days: int,
) -> str:
    """Returns the output drift status (ok/warning/critical/no_baseline/insufficient_data)."""
    report = await drift_service.compute_output_drift(
        model_name=model_name,
        period_days=period_days,
        db=db,
        model_version=model_version,
        min_predictions=10,
    )
    return report.status


# ---------------------------------------------------------------------------
# GET /monitoring/overview
# ---------------------------------------------------------------------------


_CSV_COLUMNS = [
    "model_name",
    "status",
    "predictions_7d",
    "error_rate",
    "latency_p95",
    "drift_status",
    "accuracy_7d",
    "last_retrain",
    "coverage_pct",
]


@router.get("/overview", response_model=GlobalDashboard)
async def monitoring_overview(
    start: datetime = Query(..., description="Start of the period (ISO 8601)"),
    end: datetime = Query(..., description="End of the period (ISO 8601)"),
    format: str = Query(
        default="json",
        pattern="^(json|csv)$",
        description="Output format: json (default) or csv",
    ),
    db: AsyncSession = Depends(get_read_db),
    _user: User = Depends(verify_token),
) -> Union[GlobalDashboard, StreamingResponse]:
    """
    Overview of the health of all models over a calendar range.

    For each active model that received predictions in the period:
    - Prediction volume (real + shadow), error rate, latency (avg/p50/p95)
    - Feature drift status (Z-score + PSI vs baseline)
    - Performance drift status (accuracy 1st half vs 2nd half)
    - Overall health: worst indicator (ok / warning / critical)
    """
    if end <= start:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'end' must be after 'start'.",
        )

    period_days = max(1, (end - start).days)

    # 1. Base stats per model
    raw_stats = await DBService.get_global_monitoring_stats(db, start, end)

    if not raw_stats:
        if format == "csv":
            buf = io.StringIO()
            csv.DictWriter(buf, fieldnames=_CSV_COLUMNS).writeheader()
            filename = f"supervision_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
            return StreamingResponse(
                iter([buf.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        return GlobalDashboard(
            period=MonitoringPeriod(start=start, end=end),
            global_stats=GlobalStats(
                total_predictions=0,
                total_shadow=0,
                total_errors=0,
                error_rate=0.0,
                avg_latency_ms=None,
                p95_latency_ms=None,
                active_models=0,
                models_critical=0,
                models_warning=0,
                models_ok=0,
            ),
            models=[],
        )

    # 2. Metadata for all active models (versions, deployment_modes, baselines)
    all_metadata = await DBService.get_all_active_models(db)
    meta_by_name: dict[str, list] = {}
    for m in all_metadata:
        meta_by_name.setdefault(m.name, []).append(m)

    # 3. Build health summary per model
    model_summaries: list[ModelHealthSummary] = []

    for raw in raw_stats:
        model_name = raw["model_name"]
        metas = meta_by_name.get(model_name, [])

        # Versions and deployment_modes from metadata
        versions = sorted({m.version for m in metas}) or raw["versions"]
        deployment_modes: dict[str, Optional[str]] = {m.version: m.deployment_mode for m in metas}
        # Baseline: use the production version if available, otherwise the first one
        prod_meta = next((m for m in metas if m.is_production), metas[0] if metas else None)

        # Feature drift status
        feature_drift_status = await _compute_feature_drift_status(
            db,
            model_name,
            prod_meta.version if prod_meta else None,
            period_days,
            prod_meta.feature_baseline if prod_meta else None,
        )

        # Output drift status
        output_drift_status = await _compute_output_drift_status(
            db,
            model_name,
            prod_meta.version if prod_meta else None,
            period_days,
        )

        # Performance drift status
        perf_by_day = await DBService.get_accuracy_drift(db, model_name, start, end)
        perf_drift_status = _performance_drift_status(perf_by_day)

        # Error status
        err_status = _error_rate_status(raw["error_rate"])

        # Overall health = worst indicator
        health = _worst_health(
            err_status, feature_drift_status, perf_drift_status, output_drift_status
        )

        model_summaries.append(
            ModelHealthSummary(
                model_name=model_name,
                versions=versions,
                deployment_modes=deployment_modes,
                total_predictions=raw["total_predictions"],
                shadow_predictions=raw["shadow_predictions"],
                error_count=raw["error_count"],
                error_rate=raw["error_rate"],
                avg_latency_ms=raw["avg_latency_ms"],
                p50_latency_ms=raw["p50_latency_ms"],
                p95_latency_ms=raw["p95_latency_ms"],
                feature_drift_status=feature_drift_status,
                performance_drift_status=perf_drift_status,
                output_drift_status=output_drift_status,
                last_prediction=raw["last_prediction"],
                health_status=health,
            )
        )

    # 4. Aggregated global stats
    total_pred = sum(m.total_predictions for m in model_summaries)
    total_shadow = sum(m.shadow_predictions for m in model_summaries)
    total_errors = sum(m.error_count for m in model_summaries)
    all_latencies = [m.avg_latency_ms for m in model_summaries if m.avg_latency_ms is not None]
    all_p95 = [m.p95_latency_ms for m in model_summaries if m.p95_latency_ms is not None]

    models_critical = sum(1 for m in model_summaries if m.health_status == "critical")
    models_warning = sum(1 for m in model_summaries if m.health_status == "warning")
    models_ok = len(model_summaries) - models_critical - models_warning

    global_stats = GlobalStats(
        total_predictions=total_pred,
        total_shadow=total_shadow,
        total_errors=total_errors,
        error_rate=round(total_errors / total_pred, 4) if total_pred > 0 else 0.0,
        avg_latency_ms=(
            round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else None
        ),
        p95_latency_ms=(round(sum(all_p95) / len(all_p95), 2) if all_p95 else None),
        active_models=len(model_summaries),
        models_critical=models_critical,
        models_warning=models_warning,
        models_ok=models_ok,
    )

    sorted_summaries = sorted(model_summaries, key=lambda m: m.model_name)

    if format == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for m in sorted_summaries:
            writer.writerow(
                {
                    "model_name": m.model_name,
                    "status": m.health_status,
                    "predictions_7d": m.total_predictions,
                    "error_rate": round(m.error_rate, 4),
                    "latency_p95": m.p95_latency_ms if m.p95_latency_ms is not None else "",
                    "drift_status": _worst_health(
                        m.feature_drift_status, m.performance_drift_status
                    ),
                    "accuracy_7d": "",
                    "last_retrain": "",
                    "coverage_pct": "",
                }
            )
        filename = f"supervision_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return GlobalDashboard(
        period=MonitoringPeriod(start=start, end=end),
        global_stats=global_stats,
        models=sorted_summaries,
    )


# ---------------------------------------------------------------------------
# GET /monitoring/model/{name}
# ---------------------------------------------------------------------------


@router.get("/model/{name}", response_model=ModelDetailDashboard)
async def monitoring_model_detail(
    name: str,
    start: datetime = Query(..., description="Start of the period (ISO 8601)"),
    end: datetime = Query(..., description="End of the period (ISO 8601)"),
    db: AsyncSession = Depends(get_read_db),
    _user: User = Depends(verify_token),
):
    """
    Detailed dashboard for a model over a calendar range.

    Returns:
    - Per-version statistics (predictions, errors, p50/p95 latency)
    - Daily time series (volume, errors, latency)
    - Performance drift (accuracy/MAE per day vs baseline)
    - Feature drift (Z-score + PSI per feature)
    - A/B comparison (if multiple versions in ab_test mode)
    - Latest distinct errors
    """
    if end <= start:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'end' must be after 'start'.",
        )

    period_days = max(1, (end - start).days)

    # Verify the model exists
    all_metas = await DBService.get_all_active_models(db)
    metas = [m for m in all_metas if m.name == name]
    if not metas:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' not found.",
        )

    # 1. Per-version stats over the period
    version_stats_raw = await DBService.get_model_version_stats_range(db, name, start, end)
    # Enrich with deployment_mode + traffic_weight from metadata
    meta_by_version = {m.version: m for m in metas}
    per_version_stats = [
        VersionStats(
            version=v["version"],
            deployment_mode=meta_by_version.get(v["version"], metas[0]).deployment_mode,
            traffic_weight=meta_by_version.get(v["version"], metas[0]).traffic_weight,
            total_predictions=v["total_predictions"],
            shadow_predictions=v["shadow_predictions"],
            error_count=v["error_count"],
            error_rate=v["error_rate"],
            avg_latency_ms=v["avg_latency_ms"],
            p50_latency_ms=v["p50_latency_ms"],
            p95_latency_ms=v["p95_latency_ms"],
        )
        for v in version_stats_raw
    ]

    # 2. Daily time series
    timeseries_raw = await DBService.get_model_predictions_timeseries(db, name, start, end)
    timeseries = [
        TimeseriesPoint(
            date=t["date"],
            total_predictions=t["total_predictions"],
            error_count=t["error_count"],
            error_rate=t["error_rate"],
            avg_latency_ms=t["avg_latency_ms"],
            p50_latency_ms=t["p50_latency_ms"],
            p95_latency_ms=t["p95_latency_ms"],
        )
        for t in timeseries_raw
    ]

    # 3. Performance per day (accuracy vs observed_results)
    performance_by_day = await DBService.get_accuracy_drift(db, name, start, end)

    # 4. Feature drift — over the sliding window equivalent to the period
    prod_meta = next((m for m in metas if m.is_production), metas[0])
    production_stats = await DBService.get_feature_production_stats(
        db, name, prod_meta.version, period_days
    )
    total_prod = sum(v.get("count", 0) for v in production_stats.values())
    baseline = prod_meta.feature_baseline
    baseline_available = bool(baseline)

    if not baseline_available:
        feature_drift: dict = {
            "baseline_available": False,
            "drift_summary": "no_baseline",
            "predictions_analyzed": total_prod,
            "features": {
                feat: {
                    "production_mean": round(stats["mean"], 4) if stats.get("mean") else None,
                    "production_count": stats.get("count", 0),
                    "drift_status": "no_baseline",
                }
                for feat, stats in production_stats.items()
            },
        }
    else:
        features_result = drift_service.compute_feature_drift(
            baseline, production_stats, min_count=10
        )
        summary = drift_service.summarize_drift(features_result, baseline_available=True)
        feature_drift = {
            "baseline_available": True,
            "drift_summary": summary,
            "predictions_analyzed": total_prod,
            "features": {
                feat: {k: v for k, v in res.model_dump().items() if v is not None}
                for feat, res in features_result.items()
            },
        }

    # 5. A/B/Shadow comparison — if multiple active versions
    ab_comparison: Optional[dict] = None
    has_ab = any(m.deployment_mode in ("ab_test", "shadow") for m in metas)
    if has_ab:
        ab_stats = await DBService.get_ab_comparison_stats(db, name, days=period_days)
        agreement = await DBService.get_shadow_agreement_rate(db, name, days=period_days)
        ab_comparison = {
            "versions": ab_stats,
            "shadow_agreement_rate": agreement,
        }

    # 6. Latest distinct errors
    recent_errors = await DBService.get_model_recent_errors(db, name, start, end, limit=5)

    return ModelDetailDashboard(
        model_name=name,
        period=MonitoringPeriod(start=start, end=end),
        per_version_stats=per_version_stats,
        timeseries=timeseries,
        performance_by_day=performance_by_day,
        feature_drift=feature_drift,
        ab_comparison=ab_comparison,
        recent_errors=recent_errors,
    )


# ---------------------------------------------------------------------------
# Alert check logs
# ---------------------------------------------------------------------------


@router.get("/alert-checks", response_model=AlertCheckLogList)
async def list_alert_checks(
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    check_type: Optional[str] = Query(
        None,
        description="Filter by check type (error_spike, auc, performance_drift, feature_drift, output_drift)",
    ),
    start: Optional[datetime] = Query(None, description="Start of period (ISO 8601)"),
    end: Optional[datetime] = Query(None, description="End of period (ISO 8601)"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_read_db),
    _: User = Depends(require_admin),
) -> AlertCheckLogList:
    """
    Return the history of alerting checks.

    Admin only. Each record represents one metric check for one model during a
    supervision run (every 6 h). The ``result`` field indicates whether the check
    found an anomaly and whether an email/webhook was sent.
    """

    rows, total = await DBService.get_alert_check_logs(
        db,
        model_name=model_name,
        check_type=check_type,
        start=start,
        end=end,
        limit=limit,
        offset=offset,
    )
    return AlertCheckLogList(
        items=[AlertCheckLogRead.model_validate(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )
