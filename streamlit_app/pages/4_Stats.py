"""
Usage statistics and charts
"""

import re
from datetime import datetime, timedelta, date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.api_client import get_models as get_models_cached
from utils.auth import get_client, require_auth
from utils.i18n import t

st.set_page_config(page_title=t("stats.page_title"), page_icon="📈", layout="wide")
require_auth()

col_title, col_refresh = st.columns([8, 1])
col_title.title(t("stats.title"))
if col_refresh.button(t("stats.btn_refresh"), key="stats_refresh", width='stretch'):
    st.cache_data.clear()
    st.rerun()
st.caption(t("stats.caption"))

client = get_client()

# --- Date filters (global) ---
col_d1, col_d2 = st.columns([1, 1])

date_start = col_d1.date_input(
    t("stats.date_start"),
    value=date.today() - timedelta(days=30),
    max_value=date.today(),
    key="stats_date_start",
)
date_end = col_d2.date_input(
    t("stats.date_end"),
    value=date.today(),
    min_value=date_start,
    max_value=date.today(),
    key="stats_date_end",
)

if date_end < date_start:
    st.error(t("stats.error_date_range"))
    st.stop()

days = max(1, (date_end - date_start).days)
start_dt = datetime.combine(date_start, datetime.min.time())
end_dt   = datetime.combine(date_end,   datetime.max.time().replace(microsecond=0))

try:
    models = get_models_cached(
        st.session_state.get("api_url"), st.session_state.get("api_token")
    )
    model_names = sorted({m["name"] for m in models})
except Exception:
    models = []
    model_names = []

_model_task_lookup = {
    (m["name"], m["version"]): m.get("model_task")
    for m in (models or [])
}
_model_weight_lookup = {
    (m["name"], m["version"]): m.get("traffic_weight")
    for m in (models or [])
}

if not model_names:
    st.warning(t("stats.no_models"))
    st.stop()

# --- Leaderboard helpers ---
_DRIFT_EMOJI = {
    "ok":               t("stats.drift.ok"),
    "warning":          t("stats.drift.warning"),
    "critical":         t("stats.drift.critical"),
    "no_baseline":      t("stats.drift.no_baseline"),
    "no_data":          t("stats.drift.no_data"),
    "insufficient_data": t("stats.drift.insufficient_data"),
    "unknown":          t("stats.drift.unknown"),
}


def _bg_accuracy(val):
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v >= 0.90:
        return "background-color: rgba(39, 174, 96, 0.25)"
    if v >= 0.70:
        return "background-color: rgba(241, 196, 15, 0.25)"
    return "background-color: rgba(231, 76, 60, 0.25)"


def _bg_r2(val):
    """R² coloring: green ≥ 0.90, yellow ≥ 0.70, red < 0.70, grey if None/NaN."""
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v >= 0.90:
        return "background-color: rgba(39, 174, 96, 0.25)"
    if v >= 0.70:
        return "background-color: rgba(241, 196, 15, 0.25)"
    return "background-color: rgba(231, 76, 60, 0.25)"


def _bg_error(val):
    """MAE/RMSE coloring — lower is better: green ≤ 0.10, yellow ≤ 0.50, red > 0.50."""
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v <= 0.10:
        return "background-color: rgba(39, 174, 96, 0.25)"
    if v <= 0.50:
        return "background-color: rgba(241, 196, 15, 0.25)"
    return "background-color: rgba(231, 76, 60, 0.25)"


def _build_leaderboard_fallback(models_list, stats_list, metric, n_days):
    """Build the leaderboard client-side if the API endpoint is unavailable."""
    stats_by_name = {s["model_name"]: s for s in stats_list}
    rows = [
        {
            "rank": 0,
            "name": m["name"],
            "version": m.get("version", ""),
            "accuracy": m.get("accuracy"),
            "auc": m.get("auc"),
            "f1_score": m.get("f1_score"),
            "mae": (m.get("training_metrics") or {}).get("mae"),
            "r2": (m.get("training_metrics") or {}).get("r2"),
            "rmse": (m.get("training_metrics") or {}).get("rmse"),
            "latency_p95_ms": stats_by_name.get(m["name"], {}).get("p95_response_time_ms"),
            "drift_status": "unknown",
            "predictions_count": stats_by_name.get(m["name"], {}).get("total_predictions", 0),
        }
        for m in models_list
        if m.get("is_production")
    ]
    if metric == "latency_p95_ms":
        rows.sort(
            key=lambda r: r["latency_p95_ms"] if r["latency_p95_ms"] is not None else float("inf")
        )
    elif metric == "predictions_count":
        rows.sort(key=lambda r: r["predictions_count"], reverse=True)
    elif metric == "auc":
        rows.sort(key=lambda r: r["auc"] if r["auc"] is not None else -1, reverse=True)
    elif metric == "f1_score":
        rows.sort(key=lambda r: r["f1_score"] if r["f1_score"] is not None else -1, reverse=True)
    elif metric == "r2":
        rows.sort(key=lambda r: r["r2"] if r["r2"] is not None else -float("inf"), reverse=True)
    elif metric == "mae":
        rows.sort(key=lambda r: r["mae"] if r["mae"] is not None else float("inf"))
    elif metric == "rmse":
        rows.sort(key=lambda r: r["rmse"] if r["rmse"] is not None else float("inf"))
    else:
        rows.sort(key=lambda r: r["accuracy"] if r["accuracy"] is not None else -1, reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


_DRIFT_COLOR = {
    "ok": "#2ECC71",
    "warning": "#F39C12",
    "critical": "#E74C3C",
    "no_baseline": "#95A5A6",
    "no_data": "#95A5A6",
    "insufficient_data": "#95A5A6",
    "unknown": "#95A5A6",
}


# ── Expander 1 : Leaderboard ─────────────────────────────────────────────────
with st.expander(t("stats.leaderboard.expander"), expanded=True):
    lb_col_metric, _ = st.columns([2, 3])

    _lb_sort_options = {
        "accuracy":         t("stats.leaderboard.sort_options.accuracy"),
        "auc":              t("stats.leaderboard.sort_options.auc"),
        "f1_score":         t("stats.leaderboard.sort_options.f1_score"),
        "mae":              t("stats.leaderboard.sort_options.mae"),
        "r2":               t("stats.leaderboard.sort_options.r2"),
        "rmse":             t("stats.leaderboard.sort_options.rmse"),
        "latency_p95_ms":   t("stats.leaderboard.sort_options.latency_p95_ms"),
        "predictions_count": t("stats.leaderboard.sort_options.predictions_count"),
    }

    lb_metric = lb_col_metric.selectbox(
        t("stats.leaderboard.sort_by"),
        options=list(_lb_sort_options.keys()),
        format_func=lambda x: _lb_sort_options.get(x, x),
        key="lb_metric",
    )

    try:
        leaderboard = client.get_leaderboard(metric=lb_metric, days=days)
    except Exception:
        try:
            _lb_models = get_models_cached(
                    st.session_state.get("api_url"), st.session_state.get("api_token")
                )
            _lb_stats = client.get_prediction_stats(days=days)
            leaderboard = _build_leaderboard_fallback(_lb_models, _lb_stats, lb_metric, days)
        except Exception:
            leaderboard = []

    # Model / version multiselect filters
    _lb_display = leaderboard
    if leaderboard:
        _lb_all_names = sorted({e["name"] for e in leaderboard})
        _lb_all_versions = sorted({e["version"] for e in leaderboard})
        _lb_fcol1, _lb_fcol2 = st.columns([1, 1])
        _lb_sel_models = _lb_fcol1.multiselect(
            t("stats.leaderboard.filter_model"),
            options=_lb_all_names,
            default=[],
            placeholder=t("stats.leaderboard.filter_all_models"),
            key="lb_filter_models",
        )
        _lb_sel_versions = _lb_fcol2.multiselect(
            t("stats.leaderboard.filter_version"),
            options=_lb_all_versions,
            default=[],
            placeholder=t("stats.leaderboard.filter_all_versions"),
            key="lb_filter_versions",
        )
        if _lb_sel_models or _lb_sel_versions:
            _lb_display = [
                e for e in leaderboard
                if (not _lb_sel_models or e["name"] in _lb_sel_models)
                and (not _lb_sel_versions or e["version"] in _lb_sel_versions)
            ]

    tab_table, tab_scatter = st.tabs([t("stats.leaderboard.tab_table"), t("stats.leaderboard.tab_scatter")])

    with tab_table:
        if _lb_display:
            # Column names used both for rename and column_config — define once
            _col_rank       = t("stats.leaderboard.col_rank")
            _col_model      = t("stats.leaderboard.col_model")
            _col_version    = t("stats.leaderboard.col_version")
            _col_status     = t("stats.leaderboard.col_status")
            _col_task       = t("stats.leaderboard.col_task")
            _col_accuracy   = t("stats.leaderboard.col_accuracy")
            _col_auc        = t("stats.leaderboard.col_auc")
            _col_f1         = t("stats.leaderboard.col_f1")
            _col_mae        = t("stats.leaderboard.col_mae")
            _col_r2         = t("stats.leaderboard.col_r2")
            _col_rmse       = t("stats.leaderboard.col_rmse")
            _col_latency        = t("stats.leaderboard.col_latency")
            _col_drift          = t("stats.leaderboard.col_drift")
            _col_preds          = t("stats.leaderboard.col_predictions", days=days)
            _col_ver_preds      = t("stats.leaderboard.col_version_predictions", days=days)
            _col_date_start     = t("stats.leaderboard.col_date_start")
            _col_date_end       = t("stats.leaderboard.col_date_end")

            _TASK_LABELS = {
                "classification_binary":     t("stats.leaderboard.col_task_binary"),
                "classification_multiclass": t("stats.leaderboard.col_task_multiclass"),
                "regression":                t("stats.leaderboard.col_task_regression"),
            }

            df_lb = pd.DataFrame(_lb_display)

            # Enrich with task type from models metadata
            df_lb["model_task_label"] = df_lb.apply(
                lambda r: _TASK_LABELS.get(
                    _model_task_lookup.get((r["name"], r["version"])),
                    t("stats.leaderboard.col_task_unknown"),
                ),
                axis=1,
            )

            # Compute localized status badge from raw deployment_mode / is_production fields
            def _lb_statut(row) -> str:
                mode = row.get("deployment_mode")
                weight = row.get("traffic_weight") or _model_weight_lookup.get((row.get("name"), row.get("version")))
                if mode == "ab_test":
                    return f"🟠 A/B ({float(weight):.0%})" if weight is not None else "🟠 A/B"
                if mode == "shadow":
                    return "🟣 Shadow"
                if row.get("is_production"):
                    return t("models.table.status_production")
                return t("models.table.status_active")

            df_lb[_col_status] = df_lb.apply(_lb_statut, axis=1)

            # When no predictions in the period, perf KPIs from training are misleading
            _zero_mask = df_lb["predictions_count"] == 0
            if _zero_mask.any():
                for _perf_col in ["accuracy", "auc", "f1_score", "mae", "r2", "rmse"]:
                    if _perf_col in df_lb.columns:
                        df_lb.loc[_zero_mask, _perf_col] = None

            df_display = df_lb.rename(
                columns={
                    "rank":                       _col_rank,
                    "name":                       _col_model,
                    "version":                    _col_version,
                    "model_task_label":           _col_task,
                    "accuracy":                   _col_accuracy,
                    "auc":                        _col_auc,
                    "f1_score":                   _col_f1,
                    "mae":                        _col_mae,
                    "r2":                         _col_r2,
                    "rmse":                       _col_rmse,
                    "latency_p95_ms":             _col_latency,
                    "drift_status":               _col_drift,
                    "predictions_count":          _col_preds,
                    "version_predictions_count":  _col_ver_preds,
                    "first_prediction_at":        _col_date_start,
                    "last_prediction_at":         _col_date_end,
                }
            )

            # Fix the 3 last columns: drop raw deployment_mode (replaced by _col_status),
            # rename is_production, then move Tâche just after Version
            df_display = df_display.drop(
                columns=["deployment_mode", "traffic_weight"], errors="ignore"
            )
            if "is_production" in df_display.columns:
                df_display = df_display.rename(columns={"is_production": t("models.analysis.col_production")})
                df_display[t("models.analysis.col_production")] = df_display[t("models.analysis.col_production")].map(
                    lambda x: "✅" if x else "—"
                )
            _col_is_prod = t("models.analysis.col_production")
            _cols = [c for c in df_display.columns if c != _col_task]
            _ins = _cols.index(_col_version) + 1 if _col_version in _cols else len(_cols)
            _cols.insert(_ins, _col_task)
            df_display = df_display[_cols]
            df_display[_col_drift] = df_display[_col_drift].map(lambda x: _DRIFT_EMOJI.get(x, x))

            # Force float64 on all metric columns (converts None/NaN → NaN)
            for _mc in [_col_accuracy, _col_auc, _col_f1, _col_mae, _col_r2, _col_rmse]:
                if _mc in df_display.columns:
                    df_display[_mc] = pd.to_numeric(df_display[_mc], errors="coerce")

            df_display[_col_latency] = df_display[_col_latency].apply(
                lambda x: f"{x:.0f} ms" if pd.notna(x) and x is not None else "—"
            )
            for _dc in (_col_date_start, _col_date_end):
                if _dc in df_display.columns:
                    df_display[_dc] = pd.to_datetime(df_display[_dc], errors="coerce").dt.date

            # Background coloring (NaN → "" via pd.isna guard in each function)
            _style_map = df_display.style.map(_bg_accuracy, subset=[_col_accuracy])
            if _col_auc in df_display.columns and df_display[_col_auc].notna().any():
                _style_map = _style_map.map(_bg_accuracy, subset=[_col_auc])
            if _col_f1 in df_display.columns and df_display[_col_f1].notna().any():
                _style_map = _style_map.map(_bg_accuracy, subset=[_col_f1])
            if _col_mae in df_display.columns and df_display[_col_mae].notna().any():
                _style_map = _style_map.map(_bg_error, subset=[_col_mae])
            if _col_r2 in df_display.columns and df_display[_col_r2].notna().any():
                _style_map = _style_map.map(_bg_r2, subset=[_col_r2])
            if _col_rmse in df_display.columns and df_display[_col_rmse].notna().any():
                _style_map = _style_map.map(_bg_error, subset=[_col_rmse])

            # Use Styler .format() to display "" for NaN — avoids "None" text
            _nan_fmt = lambda x: f"{x:.2f}" if pd.notna(x) else ""
            _fmt_map = {
                _mc: _nan_fmt
                for _mc in [_col_accuracy, _col_auc, _col_f1, _col_mae, _col_r2, _col_rmse]
                if _mc in df_display.columns
            }
            styled = _style_map.hide(axis="index").format(_fmt_map)

            _col_config = {
                _col_status: st.column_config.TextColumn(
                    _col_status,
                    help=t("stats.leaderboard.col_status_help"),
                ),
                _col_task: st.column_config.TextColumn(
                    _col_task,
                    help=t("stats.leaderboard.col_task_help"),
                ),
                _col_is_prod: st.column_config.TextColumn(_col_is_prod),
                _col_accuracy: st.column_config.TextColumn(
                    _col_accuracy,
                    help=t("stats.leaderboard.col_accuracy_help"),
                ),
                _col_auc: st.column_config.TextColumn(
                    _col_auc,
                    help=t("stats.leaderboard.col_auc_help"),
                ),
                _col_f1: st.column_config.TextColumn(
                    _col_f1,
                    help=t("stats.leaderboard.col_f1_help"),
                ),
                _col_mae: st.column_config.TextColumn(
                    _col_mae,
                    help=t("stats.leaderboard.col_mae_help"),
                ),
                _col_r2: st.column_config.TextColumn(
                    _col_r2,
                    help=t("stats.leaderboard.col_r2_help"),
                ),
                _col_rmse: st.column_config.TextColumn(
                    _col_rmse,
                    help=t("stats.leaderboard.col_rmse_help"),
                ),
                _col_latency: st.column_config.TextColumn(
                    _col_latency,
                    help=t("stats.leaderboard.col_latency_help"),
                ),
                _col_drift: st.column_config.TextColumn(
                    _col_drift,
                    help=t("stats.leaderboard.col_drift_help"),
                ),
                _col_ver_preds: st.column_config.NumberColumn(
                    _col_ver_preds,
                    help=t("stats.leaderboard.col_version_predictions_help", days=days),
                ),
                _col_preds: st.column_config.NumberColumn(
                    _col_preds,
                    help=t("stats.leaderboard.col_predictions_help", days=days),
                ),
                _col_date_start: st.column_config.DateColumn(
                    _col_date_start,
                    help=t("stats.leaderboard.col_date_start_help"),
                    format="DD/MM/YYYY",
                ),
                _col_date_end: st.column_config.DateColumn(
                    _col_date_end,
                    help=t("stats.leaderboard.col_date_end_help"),
                    format="DD/MM/YYYY",
                ),
            }
            if _zero_mask.any():
                st.caption(t("stats.leaderboard.kpis_hidden_no_pred"))
            st.dataframe(styled, width='stretch', column_config=_col_config)
        else:
            st.info(t("stats.leaderboard.no_models_prod"))

    with tab_scatter:
        if not _lb_display:
            st.info(t("stats.leaderboard.no_models_prod"))
        else:
            df_scatter = pd.DataFrame([
                {
                    "name": e["name"],
                    "version": e["version"],
                    "accuracy": e["accuracy"],
                    "auc": e.get("auc"),
                    "f1_score": e["f1_score"],
                    "mae": e.get("mae"),
                    "r2": e.get("r2"),
                    "rmse": e.get("rmse"),
                    "latency_p95_ms": e["latency_p95_ms"],
                    "drift_status": e["drift_status"],
                    "predictions_count": e["predictions_count"],
                }
                for e in _lb_display
            ])

            # Config for each metric: label, ratio [0-1] or not, step and max for threshold
            _METRIC_CFG = {
                "latency_p95_ms":    {"label": t("stats.leaderboard.scatter_label_latency"),  "is_ratio": False, "step": 10.0,  "max": None, "fmt_hover": lambda v: f"{v:.0f} ms"},
                "accuracy":          {"label": t("stats.leaderboard.scatter_label_accuracy"), "is_ratio": True,  "step": 0.05,  "max": 1.0,  "fmt_hover": lambda v: f"{v:.4f}"},
                "auc":               {"label": t("stats.leaderboard.scatter_label_auc"),      "is_ratio": True,  "step": 0.05,  "max": 1.0,  "fmt_hover": lambda v: f"{v:.4f}"},
                "f1_score":          {"label": t("stats.leaderboard.scatter_label_f1"),       "is_ratio": True,  "step": 0.05,  "max": 1.0,  "fmt_hover": lambda v: f"{v:.4f}"},
                "mae":               {"label": t("stats.leaderboard.scatter_label_mae"),      "is_ratio": False, "step": 0.01,  "max": None, "fmt_hover": lambda v: f"{v:.4f}"},
                "r2":                {"label": t("stats.leaderboard.scatter_label_r2"),       "is_ratio": True,  "step": 0.05,  "max": 1.0,  "fmt_hover": lambda v: f"{v:.4f}"},
                "rmse":              {"label": t("stats.leaderboard.scatter_label_rmse"),     "is_ratio": False, "step": 0.1,   "max": None, "fmt_hover": lambda v: f"{v:.4f}"},
                "predictions_count": {"label": t("stats.leaderboard.scatter_label_volume"),  "is_ratio": False, "step": 100.0, "max": None, "fmt_hover": lambda v: f"{int(v):,}"},
            }

            # Y axis = selected metric from "Sort by" (latency/count → fallback accuracy)
            _Y_FALLBACK = {"latency_p95_ms": "accuracy", "predictions_count": "accuracy"}
            scatter_y_metric = _Y_FALLBACK.get(lb_metric, lb_metric)

            # Dropdown X axis — all metrics except the one already on Y
            _x_options = [k for k in _METRIC_CFG if k != scatter_y_metric]
            _x_default = "latency_p95_ms" if "latency_p95_ms" in _x_options else _x_options[0]
            _x_default_idx = _x_options.index(_x_default)

            sc_col_x, sc_col_seuil_x, sc_col_seuil_y = st.columns([2, 2, 2])
            scatter_x_metric = sc_col_x.selectbox(
                t("stats.leaderboard.scatter_x_axis"),
                options=_x_options,
                index=_x_default_idx,
                format_func=lambda k: _METRIC_CFG[k]["label"],
                key="scatter_x_metric",
            )

            cfg_x = _METRIC_CFG[scatter_x_metric]
            cfg_y = _METRIC_CFG[scatter_y_metric]
            x_label = cfg_x["label"]
            y_label = cfg_y["label"]

            # X threshold (vertical line)
            _sx_is_lower_better = scatter_x_metric in ("rmse", "mae")
            _sx_label = (
                t("stats.leaderboard.scatter_threshold_max", label=x_label)
                if _sx_is_lower_better
                else t("stats.leaderboard.scatter_threshold_min", label=x_label)
            )
            x_threshold = sc_col_seuil_x.number_input(
                _sx_label,
                min_value=0.0,
                max_value=float(cfg_x["max"]) if cfg_x["max"] else None,
                value=0.0,
                step=float(cfg_x["step"]),
                format="%.2f",
                help=t("stats.leaderboard.scatter_threshold_x_help", label=x_label),
                key="scatter_x_threshold",
            )

            # Y threshold (horizontal line)
            _sy_is_lower_better = scatter_y_metric in ("rmse", "mae")
            _sy_label = (
                t("stats.leaderboard.scatter_threshold_max", label=y_label)
                if _sy_is_lower_better
                else t("stats.leaderboard.scatter_threshold_min", label=y_label)
            )
            y_threshold = sc_col_seuil_y.number_input(
                _sy_label,
                min_value=0.0,
                max_value=float(cfg_y["max"]) if cfg_y["max"] else None,
                value=0.0,
                step=float(cfg_y["step"]),
                format="%.2f",
                help=t("stats.leaderboard.scatter_threshold_y_help", label=y_label),
                key="scatter_y_threshold",
            )

            df_plot = df_scatter.dropna(subset=[scatter_x_metric, scatter_y_metric]).copy()
            _n_filtered = len(df_scatter) - len(df_plot)
            df_plot["color"] = df_plot["drift_status"].map(
                lambda s: _DRIFT_COLOR.get(s, _DRIFT_COLOR["unknown"])
            )
            df_plot["drift_label"] = df_plot["drift_status"].map(
                lambda s: _DRIFT_EMOJI.get(s, s)
            )
            df_plot["label"] = df_plot["name"] + " v" + df_plot["version"]

            if _n_filtered > 0 and not df_plot.empty:
                st.caption(t("stats.leaderboard.scatter_filtered", n=_n_filtered, x_label=x_label, y_label=y_label))

            if df_plot.empty:
                st.info(t("stats.leaderboard.scatter_no_data", x_label=x_label, y_label=y_label))
            else:
                fig = go.Figure()

                for _, row in df_plot.iterrows():
                    x_val   = row[scatter_x_metric]
                    y_val   = row[scatter_y_metric]
                    count   = int(row.get("predictions_count") or 0)
                    bubble  = max(10, min(60, count ** 0.5)) if count > 0 else 15

                    # Hover: X and Y first, then all other available metrics
                    hover_lines = [
                        f"<b>{row['label']}</b>",
                        f"{x_label} : {cfg_x['fmt_hover'](x_val)}",
                        f"{y_label} : {cfg_y['fmt_hover'](y_val)}",
                    ]
                    for key, cfg in _METRIC_CFG.items():
                        if key in (scatter_x_metric, scatter_y_metric, "predictions_count"):
                            continue
                        v = row.get(key)
                        if v is not None and not (isinstance(v, float) and v != v):
                            hover_lines.append(f"{cfg['label']} : {cfg['fmt_hover'](v)}")
                    hover_lines += [
                        t("stats.leaderboard.scatter_hover_predictions", count=f"{count:,}"),
                        t("stats.leaderboard.scatter_hover_drift", label=row["drift_label"]),
                    ]
                    hover_text = "<br>".join(hover_lines)

                    fig.add_trace(go.Scatter(
                        x=[x_val],
                        y=[y_val],
                        mode="markers+text",
                        marker=dict(
                            size=bubble,
                            color=row["color"],
                            opacity=0.75,
                            line=dict(width=1, color="white"),
                        ),
                        text=[row["name"]],
                        textposition="top center",
                        textfont=dict(size=11),
                        name=row["label"],
                        hovertemplate=hover_text + "<extra></extra>",
                        showlegend=False,
                    ))

                # Vertical line (X threshold)
                if x_threshold > 0:
                    _vline_label = (
                        t("stats.leaderboard.scatter_vline_max", label=x_label, value=f"{x_threshold:.2f}")
                        if _sx_is_lower_better
                        else t("stats.leaderboard.scatter_vline_min", label=x_label, value=f"{x_threshold:.0%}")
                        if cfg_x["is_ratio"]
                        else t("stats.leaderboard.scatter_vline_threshold", label=x_label, value=f"{x_threshold:.0f}")
                    )
                    fig.add_vline(
                        x=x_threshold,
                        line_dash="dash",
                        line_color="#E74C3C",
                        annotation_text=_vline_label,
                        annotation_position="top right",
                        annotation_font_color="#E74C3C",
                    )

                # Horizontal line (Y threshold)
                if y_threshold > 0:
                    _hline_label = (
                        t("stats.leaderboard.scatter_hline_max", label=y_label, value=f"{y_threshold:.2f}")
                        if _sy_is_lower_better
                        else t("stats.leaderboard.scatter_hline_min", label=y_label, value=f"{y_threshold:.0%}")
                        if cfg_y["is_ratio"]
                        else t("stats.leaderboard.scatter_hline_threshold", label=y_label, value=f"{y_threshold:.2f}")
                    )
                    fig.add_hline(
                        y=y_threshold,
                        line_dash="dash",
                        line_color="#E74C3C",
                        annotation_text=_hline_label,
                        annotation_position="bottom right",
                        annotation_font_color="#E74C3C",
                    )

                layout_kwargs = dict(
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    margin=dict(t=40, b=40),
                    hovermode="closest",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                )
                if cfg_x["is_ratio"]:
                    layout_kwargs["xaxis_range"] = [0, 1.05]
                    layout_kwargs["xaxis_tickformat"] = ".0%"
                if cfg_y["is_ratio"]:
                    layout_kwargs["yaxis_range"] = [0, 1.05]
                    layout_kwargs["yaxis_tickformat"] = ".0%"

                fig.update_layout(**layout_kwargs)
                st.plotly_chart(fig, width='stretch')


# ── Expander 2 : Aggregated statistics per model ────────────────────────────
with st.expander(t("stats.aggregated.expander"), expanded=False):
    # Local model filter for this expander
    _s2_col_search, _s2_col_model = st.columns([2, 2])
    with _s2_col_search:
        stats_search = st.text_input(
            t("stats.aggregated.filter_name"),
            key="stats_model_search",
            placeholder=t("stats.aggregated.filter_placeholder"),
        )
        stats_filtered = [n for n in model_names if stats_search.lower() in n.lower()] if stats_search else model_names
    with _s2_col_model:
        _model_all_label = t("stats.aggregated.model_all")
        model_filter = st.selectbox(
            t("stats.aggregated.model_select"),
            [_model_all_label] + (stats_filtered or model_names),
            key="stats_model_filter",
        )
    selected_model = None if model_filter == _model_all_label else model_filter

    # Load predictions for each model (or the selected model)
    all_preds = []
    fetch_models_list = [selected_model] if selected_model else model_names

    with st.spinner(t("stats.aggregated.loading")):
        for mname in fetch_models_list:
            try:
                data = client.get_predictions(
                    model_name=mname,
                    start=start_dt.isoformat(),
                    end=end_dt.isoformat(),
                    limit=1000,
                    offset=0,
                )
                preds = data.get("predictions", [])
                all_preds.extend(preds)
            except Exception:
                pass

    # Aggregated statistics (endpoint /predictions/stats)
    _agg_col_model    = t("stats.aggregated.col_model")
    _agg_col_total    = t("stats.aggregated.col_total")
    _agg_col_errors   = t("stats.aggregated.col_errors")
    _agg_col_err_rate = t("stats.aggregated.col_error_rate")
    _agg_col_avg_rt   = t("stats.aggregated.col_avg_rt")
    _agg_col_p50_rt   = t("stats.aggregated.col_p50_rt")
    _agg_col_p95_rt   = t("stats.aggregated.col_p95_rt")

    try:
        raw_stats = client.get_prediction_stats(days=days, model_name=selected_model)
        if raw_stats:
            df_stats = pd.DataFrame(raw_stats)
            df_stats = df_stats.rename(
                columns={
                    "model_name":           _agg_col_model,
                    "total_predictions":    _agg_col_total,
                    "error_count":          _agg_col_errors,
                    "error_rate":           _agg_col_err_rate,
                    "avg_response_time_ms": _agg_col_avg_rt,
                    "p50_response_time_ms": _agg_col_p50_rt,
                    "p95_response_time_ms": _agg_col_p95_rt,
                }
            )
            df_stats[_agg_col_err_rate] = (df_stats[_agg_col_err_rate] * 100).round(2).astype(str) + " %"
            st.dataframe(
                df_stats,
                width='stretch',
                hide_index=True,
                column_config={
                    _agg_col_model: st.column_config.TextColumn(
                        _agg_col_model,
                        help=t("stats.aggregated.col_model_help"),
                    ),
                    _agg_col_total: st.column_config.NumberColumn(
                        _agg_col_total,
                        help=t("stats.aggregated.col_total_help"),
                    ),
                    _agg_col_errors: st.column_config.NumberColumn(
                        _agg_col_errors,
                        help=t("stats.aggregated.col_errors_help"),
                    ),
                    _agg_col_err_rate: st.column_config.TextColumn(
                        _agg_col_err_rate,
                        help=t("stats.aggregated.col_error_rate_help"),
                    ),
                    _agg_col_avg_rt: st.column_config.NumberColumn(
                        _agg_col_avg_rt,
                        help=t("stats.aggregated.col_avg_rt_help"),
                        format="%.1f",
                    ),
                    _agg_col_p50_rt: st.column_config.NumberColumn(
                        _agg_col_p50_rt,
                        help=t("stats.aggregated.col_p50_rt_help"),
                        format="%.1f",
                    ),
                    _agg_col_p95_rt: st.column_config.NumberColumn(
                        _agg_col_p95_rt,
                        help=t("stats.aggregated.col_p95_rt_help"),
                        format="%.1f",
                    ),
                },
            )
        else:
            st.info(t("stats.aggregated.no_data"))
    except Exception as e:
        st.warning(t("stats.aggregated.load_error", error=e))

    if not all_preds:
        st.info(t("stats.aggregated.no_predictions"))
        st.stop()

    df = pd.DataFrame(all_preds)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.floor("h")
    df["is_error"] = df["status"] == "error"

    # --- KPI metrics ---
    total = len(df)
    error_rate = df["is_error"].mean() * 100
    median_rt = df["response_time_ms"].median() if "response_time_ms" in df.columns else 0
    n_models_used = df["model_name"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("stats.aggregated.kpi_total"), f"{total:,}")
    col2.metric(t("stats.aggregated.kpi_error_rate"), f"{error_rate:.1f}%", help=t("metrics.taux_erreur"))
    col3.metric(t("stats.aggregated.kpi_median_rt"), f"{median_rt:.1f} ms", help=t("metrics.latence_mediane"))
    col4.metric(t("stats.aggregated.kpi_models_used"), n_models_used)

    st.divider()

    # --- Charts ---
    _chart_model  = t("stats.aggregated.chart_label_model")
    _chart_nb_pred = t("stats.aggregated.chart_label_nb_pred")
    _chart_time_ms = t("stats.aggregated.chart_label_time_ms")
    _chart_date   = t("stats.aggregated.chart_label_date")

    row1_l, row1_r = st.columns(2)

    # Distribution by model
    with row1_l:
        st.subheader(t("stats.aggregated.chart_by_model"))
        model_counts = (
            df.groupby("model_name")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        fig = px.bar(
            model_counts,
            x="model_name",
            y="count",
            color="model_name",
            labels={"model_name": _chart_model, "count": _chart_nb_pred},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig, width='stretch')

    # Response time — density curve (KDE) per model
    with row1_r:
        st.subheader(t("stats.aggregated.chart_rt_dist"))
        if "response_time_ms" in df.columns:
            import numpy as np

            df_rt = df[~df["is_error"] & df["response_time_ms"].notna()].copy()
            colors = px.colors.qualitative.Set2
            fig = go.Figure()

            for i, (model, grp) in enumerate(df_rt.groupby("model_name")):
                vals = grp["response_time_ms"].values
                if len(vals) < 5:
                    continue
                # Clip to p99 to avoid squashing the curve with rare outliers
                p99 = float(np.percentile(vals, 99))
                vals_clipped = vals[vals <= p99]
                # Density-normalized histogram → 50 bins
                counts, edges = np.histogram(vals_clipped, bins=50, density=True)
                centers = (edges[:-1] + edges[1:]) / 2
                # Light Gaussian smoothing (convolution) without scipy
                sigma = 2
                k = np.arange(-3 * sigma, 3 * sigma + 1)
                kernel = np.exp(-0.5 * (k / sigma) ** 2)
                kernel /= kernel.sum()
                smoothed = np.convolve(counts, kernel, mode="same")

                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=centers,
                    y=smoothed,
                    mode="lines",
                    name=model,
                    line=dict(width=2, color=color),
                    fill="tozeroy",
                    opacity=0.6,
                ))

            fig.update_layout(
                xaxis_title=_chart_time_ms,
                yaxis_title=t("stats.aggregated.chart_label_density"),
                margin=dict(t=20),
                legend=dict(orientation="h", y=-0.25),
                hovermode="x unified",
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info(t("stats.aggregated.chart_rt_no_data"))

    row2_l, row2_r = st.columns(2)

    # Time series — predictions per day
    with row2_l:
        st.subheader(t("stats.aggregated.chart_daily"))
        daily = df.groupby(["date", "model_name"]).size().reset_index(name="count")
        fig = px.line(
            daily,
            x="date",
            y="count",
            color="model_name",
            labels={"date": _chart_date, "count": _chart_nb_pred, "model_name": _chart_model},
            markers=True,
        )
        fig.update_layout(margin=dict(t=20))
        st.plotly_chart(fig, width='stretch')

    # Errors per day per model
    with row2_r:
        st.subheader(t("stats.aggregated.chart_errors_daily"))
        _err_label   = t("stats.aggregated.col_errors")
        _model_label = _chart_model
        errors_daily = (
            df[df["is_error"]]
            .groupby(["date", "model_name"])
            .size()
            .reset_index(name=_err_label)
            .rename(columns={"date": _chart_date, "model_name": _model_label})
        )
        if errors_daily.empty:
            st.success(t("stats.aggregated.chart_no_errors"))
        else:
            fig = px.line(
                errors_daily,
                x=_chart_date,
                y=_err_label,
                color=_model_label,
                markers=True,
                labels={_chart_date: _chart_date, _err_label: t("stats.aggregated.chart_label_nb_errors"), _model_label: _model_label},
            )
            fig.update_layout(margin=dict(t=20), legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, width='stretch')

    # Response time box plot per model
    if "response_time_ms" in df.columns and df["response_time_ms"].notna().any():
        st.subheader(t("stats.aggregated.chart_boxplot"))
        fig = px.box(
            df[~df["is_error"]],
            x="model_name",
            y="response_time_ms",
            color="model_name",
            labels={"model_name": _chart_model, "response_time_ms": _chart_time_ms},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig, width='stretch')


# ── Expander 3 : Multi-model temporal accuracy ───────────────────────────
with st.expander(t("stats.accuracy_timeline.expander"), expanded=False):
    acc_col_models, acc_col_gran = st.columns([4, 1])

    with acc_col_models:
        _default_acc = model_names[:6] if len(model_names) > 6 else model_names
        acc_models = st.multiselect(
            t("stats.accuracy_timeline.models_label"),
            options=model_names,
            default=_default_acc,
            key="acc_models_multiselect",
        )

    with acc_col_gran:
        _acc_gran_options = {"day": t("stats.accuracy_timeline.gran_day"), "week": t("stats.accuracy_timeline.gran_week")}
        acc_gran = st.selectbox(
            t("stats.accuracy_timeline.granularity"),
            options=list(_acc_gran_options.keys()),
            format_func=lambda x: _acc_gran_options.get(x, x),
            key="acc_granularity",
        )

    if not acc_models:
        st.info(t("stats.accuracy_timeline.no_model_selected"))
    else:
        acc_rows: list[dict] = []
        acc_errors: list[str] = []
        with st.spinner(t("stats.accuracy_timeline.loading")):
            for mname in acc_models:
                try:
                    perf = client.get_model_performance(
                        model_name=mname,
                        start=start_dt.isoformat(),
                        end=end_dt.isoformat(),
                        granularity=acc_gran,
                    )
                    mtype = perf.get("model_type", "classification")
                    mcol = "accuracy" if mtype == "classification" else "mae"
                    for period in perf.get("by_period") or []:
                        v = period.get(mcol)
                        if v is not None:
                            acc_rows.append(
                                {
                                    "date": period["period"],
                                    t("stats.accuracy_timeline.col_model"): mname,
                                    "metric": float(v),
                                    "model_type": mtype,
                                    "matched_count": int(period.get("matched_count") or 0),
                                }
                            )
                except Exception:
                    acc_errors.append(mname)

        if acc_errors:
            st.caption(t("stats.accuracy_timeline.unavailable", models=", ".join(acc_errors)))

        if not acc_rows:
            st.info(t("stats.accuracy_timeline.no_data"))
        else:
            _acc_col_model = t("stats.accuracy_timeline.col_model")
            df_acc = pd.DataFrame(acc_rows)
            # The API returns "2026-W19" (week granularity) or "2026-05-11" (day).
            _sample = str(df_acc["date"].iloc[0]) if len(df_acc) else ""
            if re.match(r"^\d{4}-W\d{2}$", _sample):
                df_acc["date"] = pd.to_datetime(
                    df_acc["date"] + "-1", format="%G-W%V-%u"
                )
            else:
                df_acc["date"] = pd.to_datetime(df_acc["date"])
            df_acc = df_acc.sort_values("date")

            # Detect whether all models are of the same type
            _types = df_acc["model_type"].unique().tolist()
            _all_classif = all(tp == "classification" for tp in _types)
            _all_regress = all(tp == "regression" for tp in _types)
            if _all_classif:
                metric_label = t("stats.accuracy_timeline.metric_label_accuracy")
                y_fmt = ".0%"
                y_range = [0, 1.05]
                hover_fmt = ".1%"
            elif _all_regress:
                metric_label = t("stats.accuracy_timeline.metric_label_mae")
                y_fmt = None
                y_range = None
                hover_fmt = ".4f"
            else:
                metric_label = t("stats.accuracy_timeline.metric_label_mixed")
                y_fmt = None
                y_range = None
                hover_fmt = ".4f"

            colors = px.colors.qualitative.Set2
            fig_acc = go.Figure()
            _mae_suffix = t("stats.accuracy_timeline.legend_mae_suffix")
            for i, (model, grp) in enumerate(df_acc.groupby(_acc_col_model, sort=False)):
                grp = grp.sort_values("date")
                color = colors[i % len(colors)]
                mtype_grp = grp["model_type"].iloc[0]
                # "(MAE)" suffix if mixed types
                legend_label = model if len(_types) == 1 else (
                    f"{model} {_mae_suffix}" if mtype_grp == "regression" else model
                )
                h_fmt = ".1%" if mtype_grp == "classification" else ".4f"
                fig_acc.add_trace(
                    go.Scatter(
                        x=grp["date"],
                        y=grp["metric"],
                        mode="lines+markers",
                        name=legend_label,
                        line=dict(width=2, color=color),
                        marker=dict(size=6, color=color),
                        customdata=grp[["matched_count"]].values,
                        hovertemplate=(
                            f"<b>{model}</b><br>"
                            "%{x|%Y-%m-%d}<br>"
                            f"{metric_label} : %{{y:{h_fmt}}}<br>"
                            f"{t('stats.accuracy_timeline.hover_pairs')}<extra></extra>"
                        ),
                    )
                )

            fig_acc.update_layout(
                xaxis_title=t("stats.aggregated.chart_label_date"),
                yaxis_title=metric_label,
                yaxis_tickformat=y_fmt,
                yaxis_range=y_range,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=50, b=20),
                hovermode="x unified",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(200,200,200,0.12)"),
                yaxis=dict(gridcolor="rgba(200,200,200,0.12)"),
            )
            st.plotly_chart(fig_acc, width='stretch')

            # Summary table
            _col_obs_pairs = t("stats.accuracy_timeline.col_observed_pairs")
            _summary = []
            for model, grp in df_acc.groupby(_acc_col_model, sort=False):
                last = grp.sort_values("date").iloc[-1]
                _summary.append(
                    {
                        _acc_col_model: model,
                        metric_label: round(last["metric"], 4),
                        _col_obs_pairs: int(grp["matched_count"].sum()),
                    }
                )
            df_summary = (
                pd.DataFrame(_summary)
                .sort_values(metric_label, ascending=_all_regress)
                .reset_index(drop=True)
            )
            st.dataframe(
                df_summary,
                width='stretch',
                hide_index=True,
                column_config={
                    metric_label: st.column_config.NumberColumn(
                        metric_label,
                        help=t("stats.accuracy_timeline.col_metric_help"),
                        format="%.2f",
                    ),
                    _col_obs_pairs: st.column_config.NumberColumn(
                        _col_obs_pairs,
                        help=t("stats.accuracy_timeline.col_observed_pairs_help"),
                    ),
                },
            )


# ── Expander 4 : Multi-metric evolution — one model ───────────────────────
with st.expander(t("stats.multi_metric.expander"), expanded=False):

    # ── Controls ────────────────────────────────────────────────────────────
    _pm_col_model, _pm_col_ver, _pm_col_gran = st.columns([3, 2, 1])

    with _pm_col_model:
        perf_model = st.selectbox(
            t("stats.multi_metric.model_label"),
            options=model_names,
            key="pm_model_select",
        )

    with _pm_col_ver:
        _pm_ver_all = t("stats.multi_metric.version_all")
        def _parse_ver(v: str):
            parts = []
            for seg in v.split("."):
                try:
                    parts.append(int(seg))
                except ValueError:
                    parts.append(0)
            return parts

        _pm_versions = [_pm_ver_all] + sorted(
            {m["version"] for m in models if m["name"] == perf_model},
            key=_parse_ver,
            reverse=True,
        )
        perf_ver_sel = st.selectbox(t("stats.multi_metric.version_label"), _pm_versions, key="pm_ver_select")
        perf_ver_arg = None if perf_ver_sel == _pm_ver_all else perf_ver_sel

    with _pm_col_gran:
        _pm_gran_options = {
            "day":   t("stats.multi_metric.gran_day"),
            "week":  t("stats.multi_metric.gran_week"),
            "month": t("stats.multi_metric.gran_month"),
        }
        perf_gran = st.selectbox(
            t("stats.multi_metric.granularity"),
            options=list(_pm_gran_options.keys()),
            format_func=lambda x: _pm_gran_options.get(x, x),
            key="pm_gran_select",
        )

    # ── Loading ───────────────────────────────────────────────────────────
    try:
        _pm_perf = client.get_model_performance(
            model_name=perf_model,
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            version=perf_ver_arg,
            granularity=perf_gran,
        )
        _pm_mtype = _pm_perf.get("model_type", "classification")
        _pm_periods = _pm_perf.get("by_period") or []
    except Exception as _pm_exc:
        st.warning(t("stats.multi_metric.load_error", error=_pm_exc))
        _pm_mtype = "classification"
        _pm_periods = []

    # ── Catalogue of available metrics by model type ────────────────────
    _PM_METRICS_CLASSIF: dict[str, dict] = {
        "accuracy":      {"label": t("stats.multi_metric.metric_accuracy"), "ratio": True},
        "auc":           {"label": t("stats.multi_metric.metric_auc"),      "ratio": True},
        "f1_weighted":   {"label": t("stats.multi_metric.metric_f1"),       "ratio": True},
        "matched_count": {"label": t("stats.multi_metric.metric_pairs"),    "ratio": False},
    }
    _PM_METRICS_REGRESS: dict[str, dict] = {
        "mae":           {"label": t("stats.multi_metric.metric_mae"),   "ratio": False},
        "rmse":          {"label": t("stats.multi_metric.metric_rmse"),  "ratio": False},
        "matched_count": {"label": t("stats.multi_metric.metric_pairs"), "ratio": False},
    }
    _pm_catalog = _PM_METRICS_CLASSIF if _pm_mtype == "classification" else _PM_METRICS_REGRESS
    _pm_defaults = ["accuracy", "f1_weighted"] if _pm_mtype == "classification" else ["mae", "rmse"]

    # Reset multiselect when the model or its type changes to avoid empty selection
    _pm_model_key = f"{perf_model}__{_pm_mtype}"
    if st.session_state.get("_pm_prev_model_key") != _pm_model_key:
        st.session_state.pop("pm_metrics_select", None)
        st.session_state["_pm_prev_model_key"] = _pm_model_key

    perf_metrics = st.multiselect(
        t("stats.multi_metric.metrics_label"),
        options=list(_pm_catalog.keys()),
        default=_pm_defaults,
        format_func=lambda k: _pm_catalog[k]["label"],
        key="pm_metrics_select",
    )

    # ── Chart ─────────────────────────────────────────────────────────────
    if not perf_metrics:
        st.info(t("stats.multi_metric.no_metric_selected"))
    elif not _pm_periods:
        st.info(t("stats.multi_metric.no_data"))
    else:
        # Parse dates (day / ISO week / month)
        df_pm = pd.DataFrame(_pm_periods)
        _pm_sample = str(df_pm["period"].iloc[0])
        if re.match(r"^\d{4}-W\d{2}$", _pm_sample):
            df_pm["date"] = pd.to_datetime(df_pm["period"] + "-1", format="%G-W%V-%u")
        elif re.match(r"^\d{4}-\d{2}$", _pm_sample):
            df_pm["date"] = pd.to_datetime(df_pm["period"] + "-01", format="%Y-%m-%d")
        else:
            df_pm["date"] = pd.to_datetime(df_pm["period"])
        df_pm = df_pm.sort_values("date").reset_index(drop=True)

        # Primary metrics (left axis) vs matched_count (right axis, bars)
        _primary = [m for m in perf_metrics if m != "matched_count"]
        _show_count = "matched_count" in perf_metrics
        _all_ratio = bool(_primary) and all(_pm_catalog[m]["ratio"] for m in _primary)

        colors = px.colors.qualitative.Set2
        fig_pm = go.Figure()

        for i, metric in enumerate(_primary):
            cfg = _pm_catalog[metric]
            col = colors[i % len(colors)]
            vals = df_pm[metric] if metric in df_pm.columns else pd.Series([None] * len(df_pm))
            h_fmt = ".1%" if cfg["ratio"] else ".4f"
            fig_pm.add_trace(
                go.Scatter(
                    x=df_pm["date"],
                    y=vals,
                    mode="lines+markers",
                    name=cfg["label"],
                    line=dict(width=2, color=col),
                    marker=dict(size=7, color=col),
                    yaxis="y1",
                    hovertemplate=(
                        f"<b>{cfg['label']}</b><br>"
                        "%{x|%Y-%m-%d}<br>"
                        f"%{{y:{h_fmt}}}<extra></extra>"
                    ),
                )
            )

        _pairs_label = t("stats.multi_metric.bar_pairs")
        if _show_count:
            fig_pm.add_trace(
                go.Bar(
                    x=df_pm["date"],
                    y=df_pm["matched_count"],
                    name=_pairs_label,
                    yaxis="y2",
                    opacity=0.22,
                    marker_color="#999999",
                    hovertemplate=(
                        f"<b>{_pairs_label}</b><br>"
                        "%{x|%Y-%m-%d}<br>"
                        "%{y:d}<extra></extra>"
                    ),
                )
            )

        y1_title = " · ".join(_pm_catalog[m]["label"] for m in _primary) or ""
        fig_pm.update_layout(
            xaxis_title=t("stats.multi_metric.axis_date"),
            yaxis=dict(
                title=y1_title,
                tickformat=".0%" if _all_ratio else None,
                range=[0, 1.05] if _all_ratio else None,
                gridcolor="rgba(200,200,200,0.12)",
            ),
            yaxis2=dict(
                title=t("stats.multi_metric.axis_pairs"),
                overlaying="y",
                side="right",
                showgrid=False,
                rangemode="tozero",
            ) if _show_count else None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=55, b=20),
            hovermode="x unified",
            barmode="overlay",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(200,200,200,0.12)"),
        )
        st.plotly_chart(fig_pm, width='stretch')

        # ── Global summary for the period ───────────────────────────────────────
        _pm_global_metrics = {
            "accuracy": _pm_perf.get("accuracy"),
            "auc": _pm_perf.get("auc"),
            "f1_weighted": _pm_perf.get("f1_weighted"),
            "mae": _pm_perf.get("mae"),
            "rmse": _pm_perf.get("rmse"),
            "matched_count": _pm_perf.get("matched_predictions"),
        }
        _summary_cols = [m for m in perf_metrics if _pm_global_metrics.get(m) is not None]
        if _summary_cols:
            st.caption(t("stats.multi_metric.summary_caption"))
            _s_cols = st.columns(len(_summary_cols))
            for col_w, metric in zip(_s_cols, _summary_cols):
                cfg = _pm_catalog[metric]
                val = _pm_global_metrics[metric]
                fmt_val = f"{val:.1%}" if cfg["ratio"] else (
                    f"{int(val):,}" if metric == "matched_count" else f"{val:.4f}"
                )
                col_w.metric(cfg["label"], fmt_val)


# ── Expander 5 : Performance drift ────────────────────────────────────────
# Catalogue of metrics supported by the drift expander
_DRIFT_METRIC_CFG: dict[str, dict] = {
    "accuracy":    {"label": t("stats.multi_metric.metric_accuracy"), "ratio": True,  "higher_better": True,  "help_key": "accuracy"},
    "f1_weighted": {"label": t("stats.multi_metric.metric_f1"),       "ratio": True,  "higher_better": True,  "help_key": "f1"},
    "mae":         {"label": t("stats.multi_metric.metric_mae"),      "ratio": False, "higher_better": False, "help_key": "mae"},
    "rmse":        {"label": t("stats.multi_metric.metric_rmse"),     "ratio": False, "higher_better": False, "help_key": "rmse"},
}

with st.expander(t("stats.drift_performance.expander"), expanded=False):
    drift_col_model, drift_col_metric, drift_col_threshold, drift_col_dates = st.columns([2, 2, 2, 2])

    with drift_col_model:
        drift_search = st.text_input(
            t("stats.drift_performance.filter_name"),
            key="drift_model_search",
            placeholder=t("stats.drift_performance.filter_placeholder"),
        )
        drift_filtered = [n for n in model_names if drift_search.lower() in n.lower()] if drift_search else model_names
        drift_model = st.selectbox(
            t("stats.drift_performance.model_label"),
            drift_filtered or model_names,
            key="drift_model",
        )

    with drift_col_metric:
        drift_metric_key = st.selectbox(
            t("stats.drift_performance.metric_label"),
            options=list(_DRIFT_METRIC_CFG.keys()),
            format_func=lambda k: _DRIFT_METRIC_CFG[k]["label"],
            key="drift_metric_select",
        )
        _dcfg = _DRIFT_METRIC_CFG[drift_metric_key]
        metric_label = _dcfg["label"]
        _is_ratio = _dcfg["ratio"]
        _higher_better = _dcfg["higher_better"]

    with drift_col_threshold:
        alert_enabled = st.checkbox(t("stats.drift_performance.alert_checkbox"), value=True)
        if _is_ratio:
            threshold = st.slider(
                t("stats.drift_performance.slider_min", label=metric_label),
                min_value=0.0, max_value=1.0, value=0.7, step=0.05,
                disabled=not alert_enabled,
                help=t("stats.drift_performance.slider_help_min"),
            )
        else:
            threshold = st.number_input(
                t("stats.drift_performance.input_max", label=metric_label),
                min_value=0.0, value=0.5, step=0.05, format="%.3f",
                disabled=not alert_enabled,
                help=t("stats.drift_performance.input_help_max"),
            )

    with drift_col_dates:
        _default_end = datetime.utcnow().date()
        _default_start = _default_end - timedelta(days=45)
        drift_date_start = st.date_input(t("stats.drift_performance.date_start"), value=_default_start, key="drift_date_start")
        drift_date_end = st.date_input(t("stats.drift_performance.date_end"), value=_default_end, key="drift_date_end")

    drift_end = datetime.combine(drift_date_end, datetime.max.time())
    drift_start = datetime.combine(drift_date_start, datetime.min.time())
    _n_drift_days = (drift_date_end - drift_date_start).days

    if drift_start >= drift_end:
        st.warning(t("stats.drift_performance.date_error"))
        st.stop()

    try:
        perf_data = client.get_model_performance(
            model_name=drift_model,
            start=drift_start.isoformat(),
            end=drift_end.isoformat(),
            granularity="day",
        )
        by_period = perf_data.get("by_period") or []
        model_type = perf_data.get("model_type", "classification")
    except Exception as e:
        st.warning(t("stats.drift_performance.load_error", error=e))
        by_period = []
        model_type = "classification"

    if not by_period:
        st.info(t(
            "stats.drift_performance.no_data",
            date_start=drift_date_start,
            date_end=drift_date_end,
        ))
    else:
        drift_df = pd.DataFrame(by_period)
        drift_df["date"] = pd.to_datetime(drift_df["period"])
        drift_df = drift_df.sort_values("date").reset_index(drop=True)

        # Check metric availability for this model type
        _classif_metrics = {"accuracy", "f1_weighted"}
        _regress_metrics = {"mae", "rmse"}
        _wrong_type = (
            (drift_metric_key in _classif_metrics and model_type != "classification") or
            (drift_metric_key in _regress_metrics and model_type == "classification")
        )
        if _wrong_type or drift_metric_key not in drift_df.columns or drift_df[drift_metric_key].isna().all():
            _expected = (
                t("stats.drift_performance.expected_classif")
                if drift_metric_key in _classif_metrics
                else t("stats.drift_performance.expected_regress")
            )
            st.warning(t(
                "stats.drift_performance.wrong_type",
                metric=metric_label,
                model_type=model_type,
                expected=_expected,
            ))
        else:
            win_short = min(7, max(1, _n_drift_days // 6))
            win_long  = min(30, max(3, _n_drift_days // 2))
            drift_df["rolling_short"] = drift_df[drift_metric_key].rolling(win_short, min_periods=1).mean().round(4)
            drift_df["rolling_long"]  = drift_df[drift_metric_key].rolling(win_long,  min_periods=1).mean().round(4)
            roll_short_label = t("stats.drift_performance.rolling_short_label", n=win_short)
            roll_long_label  = t("stats.drift_performance.rolling_long_label",  n=win_long)

            last_val    = drift_df[drift_metric_key].iloc[-1]
            prev_val    = drift_df[drift_metric_key].iloc[-8] if len(drift_df) >= 8 else drift_df[drift_metric_key].iloc[0]
            delta       = round(last_val - prev_val, 4)
            matched_total = int(drift_df["matched_count"].sum())

            def _fmt(v: float) -> str:
                return f"{v:.1%}" if _is_ratio else f"{v:.4f}"

            m1, m2, m3 = st.columns(3)
            m1.metric(
                t("stats.drift_performance.kpi_last_day", metric=metric_label),
                _fmt(last_val),
                delta=f"{delta:+.1%}" if _is_ratio else f"{delta:+.4f}",
                delta_color="normal" if _higher_better else "inverse",
                help=t(f"metrics.{_dcfg['help_key']}"),
            )
            m2.metric(
                roll_short_label,
                _fmt(drift_df["rolling_short"].iloc[-1]),
                help=t(f"metrics.{_dcfg['help_key']}"),
            )
            m3.metric(
                t("stats.drift_performance.kpi_matched", days=_n_drift_days),
                f"{matched_total:,}",
            )

            # Threshold alert
            if alert_enabled:
                last_rolling = drift_df["rolling_short"].iloc[-1]
                _breach = (last_rolling < threshold) if _higher_better else (last_rolling > threshold)
                if _breach:
                    _direction = (
                        t("stats.drift_performance.direction_below")
                        if _higher_better
                        else t("stats.drift_performance.direction_above")
                    )
                    _seuil_fmt = f"{threshold:.0%}" if _is_ratio else f"{threshold:.3f}"
                    st.warning(t(
                        "stats.drift_performance.alert_breach",
                        n=win_short,
                        metric=metric_label,
                        current=_fmt(last_rolling),
                        direction=_direction,
                        threshold=_seuil_fmt,
                    ))

            _h_fmt = ".1%" if _is_ratio else ".4f"
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drift_df["date"], y=drift_df[drift_metric_key],
                mode="lines+markers",
                name=t("stats.drift_performance.daily_label", metric=metric_label),
                line=dict(color="#AAAAAA", width=1, dash="dot"),
                marker=dict(size=5), opacity=0.6,
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{metric_label} : %{{y:{_h_fmt}}}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=drift_df["date"], y=drift_df["rolling_short"],
                mode="lines", name=roll_short_label,
                line=dict(color="#636EFA", width=2),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{roll_short_label} : %{{y:{_h_fmt}}}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=drift_df["date"], y=drift_df["rolling_long"],
                mode="lines", name=roll_long_label,
                line=dict(color="#FF7F0E", width=2, dash="dash"),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{roll_long_label} : %{{y:{_h_fmt}}}<extra></extra>",
            ))
            if alert_enabled and threshold > 0:
                _seuil_annotation = (
                    t("stats.drift_performance.seuil_min", value=f"{threshold:.0%}")
                    if _is_ratio and _higher_better
                    else t("stats.drift_performance.seuil_max", value=f"{threshold:.0%}")
                    if _is_ratio
                    else t("stats.drift_performance.seuil_max", value=f"{threshold:.3f}")
                )
                fig.add_hline(
                    y=threshold,
                    line_dash="dot", line_color="#E74C3C",
                    annotation_text=_seuil_annotation,
                    annotation_position="bottom right",
                    annotation_font_color="#E74C3C",
                )
            fig.update_layout(
                xaxis_title=t("stats.aggregated.chart_label_date"),
                yaxis_title=metric_label,
                yaxis_tickformat=".0%" if _is_ratio else None,
                yaxis_range=[0, 1.05] if _is_ratio else None,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=40),
                hovermode="x unified",
            )
            st.plotly_chart(fig, width='stretch')
