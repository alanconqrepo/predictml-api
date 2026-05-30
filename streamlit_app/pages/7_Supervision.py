"""
ML model supervision dashboard.

Global view: health status of all models over a date range.
Detailed view: zoom in on a model (time series, drift, A/B testing).
"""

from datetime import date, datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.auth import get_client, require_auth
from utils.i18n import t

def _fmt_pred_result(value) -> str:
    """Format a prediction result: float → 3 decimal places, otherwise raw str."""
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


# Mode labels consistent with the Models and A/B Testing pages
_MODE_LABEL = {
    "production": "🟢 Production",
    "ab_test":    "🟠 A/B",
    "shadow":     "🟣 Shadow",
}

st.set_page_config(
    page_title="Supervision — PredictML",
    page_icon="🔍",
    layout="wide",
)
require_auth()

st.title(t("supervision.page_title"))

client = get_client()

# ---------------------------------------------------------------------------
# Filter bar: date range
# ---------------------------------------------------------------------------
col_start, col_end, col_refresh = st.columns([2, 2, 1])

default_end = date.today()
default_start = default_end - timedelta(days=7)

start_date = col_start.date_input(t("supervision.filters.from_label"), value=default_start)
end_date = col_end.date_input(t("supervision.filters.to_label"), value=default_end)

if end_date <= start_date:
    st.error(t("supervision.filters.date_order_error"))
    st.stop()

start_iso = datetime.combine(start_date, datetime.min.time()).isoformat()
end_iso = datetime.combine(end_date, datetime.max.time()).replace(microsecond=0).isoformat()

# ---------------------------------------------------------------------------
# Loading the overview
# ---------------------------------------------------------------------------
with st.spinner(t("supervision.loading.overview")):
    try:
        overview = client.get_monitoring_overview(start=start_iso, end=end_iso)
    except Exception as exc:
        st.error(t("supervision.errors.overview_load", error=exc))
        st.stop()

gs = overview.get("global_stats", {})
models_data = overview.get("models", [])

# ---------------------------------------------------------------------------
# Export preparation (computation only, buttons in the global tab)
# ---------------------------------------------------------------------------
_STATUS_SEVERITY = {
    "ok": 0, "no_data": 0, "no_baseline": 0, "insufficient_data": 0,
    "warning": 1, "critical": 2,
}

def _worst_drift(s1: str, s2: str) -> str:
    return s1 if _STATUS_SEVERITY.get(s1, 0) >= _STATUS_SEVERITY.get(s2, 0) else s2

_csv_bytes = None
_md_bytes = None

if models_data:
    _alert_models = sorted(
        [m for m in models_data if m.get("health_status") in ("critical", "warning")],
        key=lambda m: (m.get("health_status") != "critical", m["model_name"]),
    )
    _ok_models = sorted(
        [m for m in models_data if m.get("health_status") == "ok"],
        key=lambda m: m["model_name"],
    )
    _csv_rows = [
        {
            "model_name": m["model_name"],
            "status": m.get("health_status", ""),
            "predictions_7d": m.get("total_predictions", 0),
            "error_rate": round(m.get("error_rate", 0), 4),
            "latency_p95": m.get("p95_latency_ms") if m.get("p95_latency_ms") is not None else "",
            "drift_status": _worst_drift(
                _worst_drift(
                    m.get("feature_drift_status", "no_data"),
                    m.get("performance_drift_status", "no_data"),
                ),
                m.get("output_drift_status", "no_data"),
            ),
        }
        for m in models_data
    ]
    _csv_bytes = pd.DataFrame(_csv_rows).to_csv(index=False).encode("utf-8")
    _md_lines = [
        f"# Supervision report — {start_date} → {end_date}", "",
        "## Global summary", "",
        f"- **Production predictions**: {gs.get('total_predictions', 0):,}",
        f"- **Shadow predictions**: {gs.get('total_shadow', 0):,}",
        f"- **Execution error rate**: {gs.get('error_rate', 0) * 100:.1f} % "
        f"(server errors, excluding ML quality)",
        f"- **Average latency**: {gs.get('avg_latency_ms') or '—'} ms",
        f"- **Active models**: {gs.get('active_models', 0)}",
        f"- **Alerts**: 🔴 {gs.get('models_critical', 0)} critical · 🟡 {gs.get('models_warning', 0)} warning(s)",
        "",
    ]
    if _alert_models:
        _md_lines += ["## Models in alert", ""]
        for _m in _alert_models:
            _icon_md = "🔴" if _m.get("health_status") == "critical" else "🟡"
            _md_lines += [
                f"### {_icon_md} {_m['model_name']}", "",
                f"- **Status**: {_m.get('health_status', '—')}",
                f"- **Predictions**: {_m.get('total_predictions', 0):,}",
                f"- **Error rate**: {_m.get('error_rate', 0) * 100:.1f} %",
                f"- **Feature drift**: {_m.get('feature_drift_status', '—')}",
                f"- **Performance drift**: {_m.get('performance_drift_status', '—')}",
                "",
            ]
    _md_bytes = "\n".join(_md_lines).encode("utf-8")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
STATUS_ICON = {
    "ok": "🟢", "warning": "🟡", "critical": "🔴",
    "no_data": "⚪", "no_baseline": "⚪", "insufficient_data": "⚪",
}

def _icon(status: str) -> str:
    return STATUS_ICON.get(status, "⚪") + " " + status

# ---------------------------------------------------------------------------
# Global KPIs — always visible
# ---------------------------------------------------------------------------
st.divider()
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric(
    t("supervision.kpis.predictions_prod"),
    f"{gs.get('total_predictions', 0):,}",
    help=t("metrics.predictions_prod"),
)
k2.metric(
    t("supervision.kpis.predictions_shadow"),
    f"{gs.get('total_shadow', 0):,}",
    help=t("metrics.predictions_shadow"),
)
k3.metric(
    t("supervision.kpis.error_rate"),
    f"{gs.get('error_rate', 0) * 100:.1f} %",
    help=t("metrics.taux_erreur"),
)
k4.metric(
    t("supervision.kpis.avg_latency"),
    f"{gs.get('avg_latency_ms') or '—'} ms" if gs.get("avg_latency_ms") else "—",
    help=t("metrics.latence_avg"),
)
k5.metric(
    t("supervision.kpis.active_models"),
    gs.get("active_models", 0),
    help=t("metrics.modeles_actifs"),
)
_alerts = gs.get("models_critical", 0) + gs.get("models_warning", 0)
k6.metric(
    t("supervision.kpis.alerts"),
    f"🔴 {gs.get('models_critical', 0)} · 🟡 {gs.get('models_warning', 0)}",
    delta=t("supervision.kpis.alerts_delta_issues", count=_alerts) if _alerts else t("supervision.kpis.alerts_delta_ok"),
    delta_color="inverse" if _alerts else "normal",
    help=t("metrics.alertes_sante"),
)

if not models_data:
    st.info(t("supervision.errors.no_data"))
    st.stop()

# ---------------------------------------------------------------------------
# Main navigation: two tabs
# ---------------------------------------------------------------------------
st.divider()
_tab_global, _tab_detail = st.tabs([t("supervision.tab_global"), t("supervision.tab_detail")])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Global view
# ═══════════════════════════════════════════════════════════════════════════
with _tab_global:

    # ── Exports ──────────────────────────────────────────────────────────
    if _csv_bytes:
        _exp_col1, _exp_col2, _ = st.columns([2, 2, 4])
        _exp_col1.download_button(
            label=t("supervision.global.export_csv_btn"),
            data=_csv_bytes,
            file_name=f"supervision_{start_date}_{end_date}.csv",
            mime="text/csv",
            help=t("supervision.global.export_csv_help"),
        )
        _exp_col2.download_button(
            label=t("supervision.global.export_md_btn"),
            data=_md_bytes,
            file_name=f"supervision_{start_date}_{end_date}.md",
            mime="text/markdown",
            help=t("supervision.global.export_md_help"),
        )

    # ── Health table ─────────────────────────────────────────────────────
    st.subheader(t("supervision.global.health_subheader"))
    rows_table = []
    for m in models_data:
        rows_table.append({
            t("supervision.health_table.col_model"): m["model_name"],
            t("supervision.health_table.col_versions"): ", ".join(m.get("versions", [])),
            t("supervision.health_table.col_mode"): ", ".join(sorted(set(
                _MODE_LABEL.get(v, "⚪ —") for v in m.get("deployment_modes", {}).values() if v
            ))) or "—",
            t("supervision.health_table.col_predictions"): m["total_predictions"],
            t("supervision.health_table.col_shadow"): m["shadow_predictions"],
            t("supervision.health_table.col_errors"): f"{m['error_rate'] * 100:.1f} %",
            t("supervision.health_table.col_avg_latency"): f"{m['avg_latency_ms'] or '—'} ms" if m.get("avg_latency_ms") else "—",
            t("supervision.health_table.col_p95"): f"{m['p95_latency_ms']} ms" if m.get("p95_latency_ms") else "—",
            t("supervision.health_table.col_drift_features"): _icon(m.get("feature_drift_status", "no_data")),
            t("supervision.health_table.col_drift_perf"): _icon(m.get("performance_drift_status", "no_data")),
            t("supervision.health_table.col_drift_output"): _icon(m.get("output_drift_status", "no_data")),
            t("supervision.health_table.col_status"): _icon(m.get("health_status", "no_data")),
        })
    df_health = pd.DataFrame(rows_table)
    st.dataframe(
        df_health,
        width='stretch',
        hide_index=True,
        column_config={
            t("supervision.health_table.col_model"): st.column_config.TextColumn(t("supervision.health_table.col_model"), help=t("supervision.health_table.help_model")),
            t("supervision.health_table.col_versions"): st.column_config.TextColumn(t("supervision.health_table.col_versions"), help=t("supervision.health_table.help_versions")),
            t("supervision.health_table.col_mode"): st.column_config.TextColumn(
                t("supervision.health_table.col_mode"),
                help=t("supervision.health_table.help_mode"),
            ),
            t("supervision.health_table.col_predictions"): st.column_config.NumberColumn(
                t("supervision.health_table.col_predictions"),
                help=t("supervision.health_table.help_predictions"),
            ),
            t("supervision.health_table.col_shadow"): st.column_config.NumberColumn(
                t("supervision.health_table.col_shadow"),
                help=t("supervision.health_table.help_shadow"),
            ),
            t("supervision.health_table.col_errors"): st.column_config.TextColumn(
                t("supervision.health_table.col_errors"),
                help=t("supervision.health_table.help_errors"),
            ),
            t("supervision.health_table.col_avg_latency"): st.column_config.TextColumn(
                t("supervision.health_table.col_avg_latency"),
                help=t("supervision.health_table.help_avg_latency"),
            ),
            t("supervision.health_table.col_p95"): st.column_config.TextColumn(
                t("supervision.health_table.col_p95"),
                help=t("supervision.health_table.help_p95"),
            ),
            t("supervision.health_table.col_drift_features"): st.column_config.TextColumn(
                t("supervision.health_table.col_drift_features"),
                help=t("supervision.health_table.help_drift_features"),
            ),
            t("supervision.health_table.col_drift_perf"): st.column_config.TextColumn(
                t("supervision.health_table.col_drift_perf"),
                help=t("supervision.health_table.help_drift_perf"),
            ),
            t("supervision.health_table.col_drift_output"): st.column_config.TextColumn(
                t("supervision.health_table.col_drift_output"),
                help=t("supervision.health_table.help_drift_output"),
            ),
            t("supervision.health_table.col_status"): st.column_config.TextColumn(
                t("supervision.health_table.col_status"),
                help=t("supervision.health_table.help_status"),
            ),
        },
    )

    # ── Fetch timeseries per model (for trend charts) ────────────────────
    _ts_by_model: dict[str, list] = {}
    with st.spinner(t("supervision.loading.timeseries")):
        for _m in models_data:
            try:
                _det = client.get_monitoring_model(
                    name=_m["model_name"], start=start_iso, end=end_iso
                )
                _ts_by_model[_m["model_name"]] = _det.get("timeseries", [])
            except Exception:
                _ts_by_model[_m["model_name"]] = []

    _ts_rows = []
    for _mname, _ts in _ts_by_model.items():
        for _pt in _ts:
            _ts_rows.append({
                "date":            _pt["date"],
                "model_name":      _mname,
                "error_rate_pct":  round(_pt["error_rate"] * 100, 2),
                "avg_latency_ms":  _pt.get("avg_latency_ms"),
            })
    df_ts = pd.DataFrame(
        _ts_rows if _ts_rows
        else {"date": [], "model_name": [], "error_rate_pct": [], "avg_latency_ms": []}
    )
    if not df_ts.empty:
        df_ts["date"] = pd.to_datetime(df_ts["date"])
        df_ts = df_ts.sort_values("date")

    # ── Graphiques ────────────────────────────────────────────────────────
    df_models = pd.DataFrame([{
        "model_name":       m["model_name"],
        "total_predictions": m["total_predictions"],
        "error_rate_pct":   round(m["error_rate"] * 100, 2),
    } for m in models_data])

    col_pie, col_err_ts = st.columns([1, 2])

    fig_pie = px.pie(
        df_models,
        values="total_predictions",
        names="model_name",
        title=t("supervision.charts.pie_title"),
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pie.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{value:,} predictions<br>%{percent}<extra></extra>",
    )
    fig_pie.update_layout(showlegend=False, margin=dict(t=50, b=10, l=10, r=10))
    col_pie.plotly_chart(fig_pie, width='stretch')

    if not df_ts.empty:
        fig_err_ts = px.line(
            df_ts,
            x="date", y="error_rate_pct",
            color="model_name",
            title=t("supervision.charts.error_rate_ts_title"),
            labels={"date": t("supervision.charts.date_label"), "error_rate_pct": t("supervision.charts.errors_pct_label"), "model_name": t("supervision.charts.model_label")},
            markers=True,
        )
        fig_err_ts.add_hline(
            y=5, line_dash="dash", line_color="#e67e22",
            annotation_text=t("supervision.charts.warning_5pct"), annotation_position="top right",
        )
        fig_err_ts.add_hline(
            y=10, line_dash="dash", line_color="#c0392b",
            annotation_text=t("supervision.charts.critical_10pct"), annotation_position="top right",
        )
        fig_err_ts.update_layout(
            legend_title_text=t("supervision.charts.model_label"),
            yaxis_rangemode="tozero",
            yaxis_title=t("supervision.charts.errors_pct_label"),
        )
    else:
        fig_err_ts = go.Figure()
        fig_err_ts.update_layout(
            title=t("supervision.charts.error_rate_ts_title"),
            annotations=[dict(
                text=t("supervision.charts.no_timeseries_data"),
                showarrow=False, xref="paper", yref="paper",
                x=0.5, y=0.5, font_size=14, font_color="#888",
            )],
        )
    col_err_ts.plotly_chart(fig_err_ts, width='stretch')

    df_lat = df_ts[df_ts["avg_latency_ms"].notna()].copy() if not df_ts.empty else df_ts
    if not df_lat.empty:
        fig_lat = px.line(
            df_lat,
            x="date", y="avg_latency_ms",
            color="model_name",
            title=t("supervision.charts.latency_ts_title"),
            labels={
                "date": t("supervision.charts.date_label"),
                "avg_latency_ms": t("supervision.charts.latency_avg_label"),
                "model_name": t("supervision.charts.model_label"),
            },
            markers=True,
        )
        fig_lat.update_layout(
            legend_title_text=t("supervision.charts.model_label"),
            yaxis_rangemode="tozero",
            yaxis_title=t("supervision.charts.latency_avg_label"),
        )
    else:
        fig_lat = go.Figure()
        fig_lat.update_layout(
            title=t("supervision.charts.latency_ts_title"),
            annotations=[dict(
                text=t("supervision.charts.no_latency_data"),
                showarrow=False, xref="paper", yref="paper",
                x=0.5, y=0.5, font_size=14, font_color="#888",
            )],
        )
    st.plotly_chart(fig_lat, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Model detail
# ═══════════════════════════════════════════════════════════════════════════
with _tab_detail:

    # ── Selector — search + model on the same line ───────────────────────
    model_names = [m["model_name"] for m in models_data]
    _det_c1, _det_c2 = st.columns([1, 2])
    sup_search = _det_c1.text_input(t("supervision.detail.search_label"), key="sup_model_search", placeholder=t("supervision.detail.search_placeholder"))
    sup_filtered = [n for n in model_names if sup_search.lower() in n.lower()] if sup_search else model_names
    selected_model = _det_c2.selectbox(t("supervision.detail.select_model_label"), sup_filtered or model_names)

    if not selected_model:
        st.stop()

    st.session_state["_nav_model"] = selected_model

    _lc1, _lc2, _ = st.columns([1.5, 1.5, 3])
    with _lc1:
        st.page_link("pages/2_Models.py",  label=t("supervision.detail.link_manage_models"),  width='stretch')
    with _lc2:
        st.page_link("pages/8_Retrain.py", label=t("supervision.detail.link_retrain"),       width='stretch')

    # ── Loading the detail ───────────────────────────────────────────────
    with st.spinner(t("supervision.loading.detail", model=selected_model)):
        try:
            detail = client.get_monitoring_model(name=selected_model, start=start_iso, end=end_iso)
        except Exception as exc:
            st.error(t("supervision.errors.detail_load", error=exc))
            st.stop()

    per_version  = detail.get("per_version_stats", [])
    timeseries   = detail.get("timeseries", [])
    perf_by_day  = detail.get("performance_by_day", [])
    feature_drift = detail.get("feature_drift", {})
    ab_comparison = detail.get("ab_comparison")
    recent_errors = detail.get("recent_errors", [])

    _threshold_ver = next(
        (v["version"] for v in per_version if v.get("deployment_mode") == "production"),
        per_version[0]["version"] if per_version else None,
    )
    _overlay_thresholds: dict = {}
    if _threshold_ver:
        try:
            _overlay_thresholds = client.get_model(selected_model, _threshold_ver).get("alert_thresholds") or {}
        except Exception:
            pass
    _err_max: float | None = _overlay_thresholds.get("error_rate_max")
    _acc_min: float | None = _overlay_thresholds.get("accuracy_min")

    _feat_drift_status = feature_drift.get("drift_summary", "no_data")
    _has_drift_alert   = _feat_drift_status in ("warning", "critical")

    # ────────────────────────────────────────────────────────────────────
    # EXPANDER 1 — Versions & Traffic
    # ────────────────────────────────────────────────────────────────────
    with st.expander(t("supervision.detail.versions_expander"), expanded=True):
        if per_version:
            df_ver = pd.DataFrame([{
                t("supervision.versions_table.col_version"): v["version"],
                t("supervision.versions_table.col_mode"): _MODE_LABEL.get(v.get("deployment_mode") or "", "⚪ —"),
                t("supervision.versions_table.col_traffic_weight"): f"{v['traffic_weight']:.0%}" if v.get("traffic_weight") else "—",
                t("supervision.versions_table.col_predictions"): v["total_predictions"],
                t("supervision.versions_table.col_shadow"): v["shadow_predictions"],
                t("supervision.versions_table.col_errors"): f"{v['error_rate'] * 100:.1f} %",
                t("supervision.versions_table.col_avg_latency"): f"{v['avg_latency_ms'] or '—'} ms" if v.get("avg_latency_ms") else "—",
                t("supervision.versions_table.col_p50"): f"{v['p50_latency_ms']} ms" if v.get("p50_latency_ms") else "—",
                t("supervision.versions_table.col_p95"): f"{v['p95_latency_ms']} ms" if v.get("p95_latency_ms") else "—",
            } for v in per_version])
            st.dataframe(
                df_ver, width='stretch', hide_index=True,
                column_config={
                    t("supervision.versions_table.col_version"): st.column_config.TextColumn(t("supervision.versions_table.col_version"), help=t("supervision.versions_table.help_version")),
                    t("supervision.versions_table.col_mode"): st.column_config.TextColumn(t("supervision.versions_table.col_mode"), help=t("supervision.versions_table.help_mode")),
                    t("supervision.versions_table.col_traffic_weight"): st.column_config.TextColumn(t("supervision.versions_table.col_traffic_weight"), help=t("supervision.versions_table.help_traffic_weight")),
                    t("supervision.versions_table.col_predictions"): st.column_config.NumberColumn(t("supervision.versions_table.col_predictions"), help=t("supervision.versions_table.help_predictions")),
                    t("supervision.versions_table.col_shadow"): st.column_config.NumberColumn(t("supervision.versions_table.col_shadow"), help=t("supervision.versions_table.help_shadow")),
                    t("supervision.versions_table.col_errors"): st.column_config.TextColumn(t("supervision.versions_table.col_errors"), help=t("supervision.versions_table.help_errors")),
                    t("supervision.versions_table.col_avg_latency"): st.column_config.TextColumn(t("supervision.versions_table.col_avg_latency"), help=t("supervision.versions_table.help_avg_latency")),
                    t("supervision.versions_table.col_p50"): st.column_config.TextColumn(t("supervision.versions_table.col_p50"), help=t("supervision.versions_table.help_p50")),
                    t("supervision.versions_table.col_p95"): st.column_config.TextColumn(t("supervision.versions_table.col_p95"), help=t("supervision.versions_table.help_p95")),
                },
            )
        else:
            st.info(t("supervision.detail.no_version_active"))

    # ────────────────────────────────────────────────────────────────────
    # EXPANDER 2 — Activity & Latency
    # ────────────────────────────────────────────────────────────────────
    with st.expander(t("supervision.detail.activity_expander"), expanded=False):
        if timeseries:
            df_ts = pd.DataFrame(timeseries)
            df_ts["date"] = pd.to_datetime(df_ts["date"])

            col_ts1, col_ts2 = st.columns(2)

            fig_vol_ts = go.Figure()
            fig_vol_ts.add_trace(go.Scatter(
                x=df_ts["date"], y=df_ts["total_predictions"],
                name=t("supervision.charts.predictions_trace"), fill="tozeroy", line=dict(color="#2980b9"),
            ))
            fig_vol_ts.add_trace(go.Scatter(
                x=df_ts["date"], y=df_ts["error_count"],
                name=t("supervision.charts.errors_trace"), fill="tozeroy", line=dict(color="#c0392b"),
            ))
            fig_vol_ts.update_layout(
                title=t("supervision.charts.vol_ts_title"),
                xaxis_title=t("supervision.charts.date_label"), yaxis_title=t("supervision.charts.count_label"),
                hovermode="x unified", legend=dict(orientation="h", y=1.02),
            )
            col_ts1.plotly_chart(fig_vol_ts, width='stretch')

            if df_ts["avg_latency_ms"].notna().any():
                fig_lat = go.Figure()
                for col_name, label, color in [
                    ("avg_latency_ms", t("supervision.charts.latency_mean_trace"), "#3498db"),
                    ("p50_latency_ms", "p50", "#27ae60"),
                    ("p95_latency_ms", "p95", "#e67e22"),
                ]:
                    if col_name in df_ts and df_ts[col_name].notna().any():
                        fig_lat.add_trace(go.Scatter(
                            x=df_ts["date"], y=df_ts[col_name],
                            name=label, line=dict(color=color),
                        ))
                fig_lat.update_layout(
                    title=t("supervision.charts.latency_day_title"),
                    xaxis_title=t("supervision.charts.date_label"), yaxis_title="ms",
                    hovermode="x unified", legend=dict(orientation="h", y=1.02),
                )
                col_ts2.plotly_chart(fig_lat, width='stretch')

            fig_err_ts = px.area(
                df_ts, x="date", y="error_rate",
                title=t("supervision.charts.error_rate_day_title"),
                labels={"date": t("supervision.charts.date_label"), "error_rate": t("supervision.charts.error_rate_label")},
                color_discrete_sequence=["#e74c3c"],
            )
            if _err_max is not None:
                fig_err_ts.add_hline(y=_err_max, line_dash="dash", line_color="#c0392b",
                                     annotation_text=f"Threshold {_err_max * 100:.1f}%")
            else:
                fig_err_ts.add_hline(y=0.05, line_dash="dash", line_color="#e67e22", annotation_text="5%")
                fig_err_ts.add_hline(y=0.10, line_dash="dash", line_color="#c0392b", annotation_text="10%")
            fig_err_ts.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig_err_ts, width='stretch')
        else:
            st.info(t("supervision.detail.no_timeseries"))

    # ────────────────────────────────────────────────────────────────────
    # EXPANDER 3 — Drift & Anomalies
    # ────────────────────────────────────────────────────────────────────
    _drift_icon = STATUS_ICON.get(_feat_drift_status, "⚪")
    with st.expander(
        t("supervision.detail.drift_expander", icon=_drift_icon, status=_feat_drift_status),
        expanded=False,
    ):
        col_perf, col_feat = st.columns(2)

        with col_perf:
            st.markdown(t("supervision.drift.perf_drift_header"))
            if perf_by_day:
                df_perf = pd.DataFrame(perf_by_day)
                df_perf["date"] = pd.to_datetime(df_perf["date"])
                df_perf = df_perf[df_perf["matched_count"] > 0].sort_values("date")
                if not df_perf.empty:
                    _is_regression = (
                        "mae" in df_perf.columns
                        and df_perf["mae"].notna().any()
                    )
                    _has_r2 = (
                        _is_regression
                        and "r2" in df_perf.columns
                        and df_perf["r2"].notna().any()
                    )
                    if _has_r2:
                        _metric_col   = "r2"
                        _metric_label = t("supervision.drift.metric_r2_day")
                        _y_label      = "R²"
                        _tick_fmt     = ".3f"
                        _delta_fmt    = lambda d: t("supervision.drift.delta_fmt_r2", val=f"{-d:+.3f}")  # noqa: E731
                        _drop_invert  = False
                    elif _is_regression:
                        _metric_col   = "mae"
                        _metric_label = t("supervision.drift.metric_mae_day")
                        _y_label      = "MAE"
                        _tick_fmt     = ".3f"
                        _delta_fmt    = lambda d: t("supervision.drift.delta_fmt_mae", val=f"{-d:+.3f}")  # noqa: E731
                        _drop_invert  = True
                    else:
                        _metric_col   = "accuracy"
                        _metric_label = t("supervision.drift.metric_accuracy_day")
                        _y_label      = "Accuracy"
                        _tick_fmt     = ".0%"
                        _delta_fmt    = lambda d: t("supervision.drift.delta_fmt_accuracy", val=f"{-d:.1%}")  # noqa: E731
                        _drop_invert  = False

                    df_perf["rolling_3d"] = (
                        df_perf[_metric_col].rolling(window=3, min_periods=1).mean()
                    )
                    fig_perf = go.Figure()
                    fig_perf.add_trace(go.Scatter(
                        x=df_perf["date"], y=df_perf[_metric_col],
                        name=_metric_label, mode="markers+lines",
                        marker=dict(size=6), line=dict(color="#95a5a6"),
                    ))
                    fig_perf.add_trace(go.Scatter(
                        x=df_perf["date"], y=df_perf["rolling_3d"],
                        name=t("supervision.drift.rolling_3d_trace"), line=dict(color="#2980b9", width=2),
                    ))
                    if not _is_regression and _acc_min is not None:
                        fig_perf.add_hline(y=_acc_min, line_dash="dash", line_color="#c0392b",
                                           annotation_text=f"Min {_acc_min:.0%}",
                                           annotation_position="bottom right")
                    fig_perf.update_layout(
                        xaxis_title=t("supervision.charts.date_label"), yaxis_title=_y_label,
                        yaxis=dict(tickformat=_tick_fmt),
                        hovermode="x unified", legend=dict(orientation="h", y=1.02),
                    )
                    st.plotly_chart(fig_perf, width='stretch')
                    mid = len(df_perf) // 2
                    if mid > 0:
                        raw_drop = (
                            df_perf[_metric_col].iloc[:mid].mean()
                            - df_perf[_metric_col].iloc[mid:].mean()
                        )
                        drop = -raw_drop if _drop_invert else raw_drop
                        _status = (
                            t("supervision.drift.status_critical") if drop >= 0.10
                            else t("supervision.drift.status_warning") if drop >= 0.05
                            else t("supervision.drift.status_stable")
                        )
                        st.metric(
                            t("supervision.drift.perf_trend_metric"),
                            _status,
                            delta=_delta_fmt(raw_drop), delta_color="inverse",
                            help=t("metrics.tendance_performance"),
                        )
                else:
                    st.info(t("supervision.drift.no_obs_in_period"))
            else:
                st.info(t("supervision.drift.no_perf_data"))

        with col_feat:
            st.markdown(t("supervision.drift.feat_drift_header"))
            feat_summary  = feature_drift.get("drift_summary", "no_data")
            feat_baseline = feature_drift.get("baseline_available", False)
            feat_analyzed = feature_drift.get("predictions_analyzed", 0)
            st.markdown(t("supervision.drift.global_status", icon=STATUS_ICON.get(feat_summary, '⚪'), status=feat_summary))
            st.caption(t("supervision.drift.predictions_analyzed", count=feat_analyzed))
            features_dict = feature_drift.get("features", {})
            if features_dict and feat_baseline:
                rows_drift = [{
                    t("supervision.drift_table.col_feature"): fn,
                    t("supervision.drift_table.col_status"): _icon(fd.get("drift_status", "no_data")),
                    t("supervision.drift_table.col_prod_mean"): round(fd["production_mean"], 4) if fd.get("production_mean") is not None else "—",
                    t("supervision.drift_table.col_baseline_mean"): round(fd["baseline_mean"], 4) if fd.get("baseline_mean") is not None else "—",
                    t("supervision.drift_table.col_z_score"): round(fd["z_score"], 3) if fd.get("z_score") is not None else "—",
                    t("supervision.drift_table.col_psi"): round(fd["psi"], 4) if fd.get("psi") is not None else "—",
                    t("supervision.drift_table.col_n_prod"): fd.get("production_count", 0),
                } for fn, fd in features_dict.items()]
                st.dataframe(
                    pd.DataFrame(rows_drift), width='stretch', hide_index=True,
                    column_config={
                        t("supervision.drift_table.col_feature"): st.column_config.TextColumn(
                            t("supervision.drift_table.col_feature"),
                            help=t("supervision.drift_table.help_feature"),
                        ),
                        t("supervision.drift_table.col_status"): st.column_config.TextColumn(
                            t("supervision.drift_table.col_status"),
                            help=t("supervision.drift_table.help_status"),
                        ),
                        t("supervision.drift_table.col_prod_mean"): st.column_config.TextColumn(
                            t("supervision.drift_table.col_prod_mean"),
                            help=t("supervision.drift_table.help_prod_mean"),
                        ),
                        t("supervision.drift_table.col_baseline_mean"): st.column_config.TextColumn(
                            t("supervision.drift_table.col_baseline_mean"),
                            help=t("supervision.drift_table.help_baseline_mean"),
                        ),
                        t("supervision.drift_table.col_z_score"): st.column_config.TextColumn(
                            t("supervision.drift_table.col_z_score"),
                            help=t("supervision.drift_table.help_z_score"),
                        ),
                        t("supervision.drift_table.col_psi"): st.column_config.TextColumn(
                            t("supervision.drift_table.col_psi"),
                            help=t("supervision.drift_table.help_psi"),
                        ),
                        t("supervision.drift_table.col_n_prod"): st.column_config.NumberColumn(
                            t("supervision.drift_table.col_n_prod"),
                            help=t("supervision.drift_table.help_n_prod"),
                        ),
                    },
                )
            elif not feat_baseline:
                st.info(t("supervision.drift.no_baseline"))
            else:
                st.info(t("supervision.drift.no_drift_data"))

            # ── Categorical feature drift ──────────────────────────────────────
            cat_features_dict = feature_drift.get("categorical_features", {})
            if cat_features_dict:
                st.markdown(t("supervision.drift.cat_drift_header"))
                _STATUS_ORDER = {"critical": 0, "warning": 1, "ok": 2, "insufficient_data": 3, "no_baseline": 4}
                _CAT_PSI_COLORS = {"ok": "#27ae60", "warning": "#f39c12", "critical": "#e74c3c"}

                for feat_name, feat_data in sorted(
                    cat_features_dict.items(),
                    key=lambda x: _STATUS_ORDER.get(x[1].get("drift_status", "no_baseline"), 99),
                ):
                    drift_st = feat_data.get("drift_status", "no_data")
                    psi_val  = feat_data.get("psi")
                    prod_cnt = feat_data.get("production_count", 0)
                    bl_dist  = feat_data.get("baseline_distribution", {})
                    pr_dist  = feat_data.get("production_distribution", {})

                    icon = _icon(drift_st)
                    psi_str = f"PSI={psi_val:.4f}" if psi_val is not None else "PSI=—"
                    with st.expander(f"{icon} **{feat_name}** — {drift_st.upper()}  ·  {psi_str}  ·  n={prod_cnt}", expanded=(drift_st in ("warning", "critical"))):
                        if bl_dist and pr_dist:
                            all_cats = sorted(set(bl_dist.keys()) | set(pr_dist.keys()))
                            df_cat = pd.DataFrame({
                                t("supervision.drift_cat.col_category"): all_cats,
                                t("supervision.drift_cat.col_baseline"): [round(bl_dist.get(c, 0.0) * 100, 1) for c in all_cats],
                                t("supervision.drift_cat.col_production"): [round(pr_dist.get(c, 0.0) * 100, 1) for c in all_cats],
                            })
                            df_cat[t("supervision.drift_cat.col_delta")] = (
                                df_cat[t("supervision.drift_cat.col_production")]
                                - df_cat[t("supervision.drift_cat.col_baseline")]
                            ).round(1)

                            fig_cat = go.Figure()
                            fig_cat.add_trace(go.Bar(
                                name=t("supervision.drift_cat.legend_baseline"),
                                x=all_cats,
                                y=df_cat[t("supervision.drift_cat.col_baseline")],
                                marker_color="#5dade2",
                            ))
                            fig_cat.add_trace(go.Bar(
                                name=t("supervision.drift_cat.legend_production"),
                                x=all_cats,
                                y=df_cat[t("supervision.drift_cat.col_production")],
                                marker_color=_CAT_PSI_COLORS.get(drift_st, "#95a5a6"),
                            ))
                            fig_cat.update_layout(
                                barmode="group",
                                xaxis_title=feat_name,
                                yaxis_title=t("supervision.drift_cat.y_axis"),
                                height=280,
                                margin=dict(l=0, r=0, t=20, b=0),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            )
                            st.plotly_chart(fig_cat, use_container_width=True)

                            st.dataframe(df_cat, hide_index=True, use_container_width=True)
                        elif drift_st == "insufficient_data":
                            st.caption(t("supervision.drift.insufficient_data", count=prod_cnt))
                        else:
                            st.caption(t("supervision.drift.no_baseline"))

        st.divider()

        # Confidence trend
        st.markdown(t("supervision.confidence.header"))
        period_days = max(1, (end_date - start_date).days)
        try:
            conf_trend = client.get_confidence_trend(selected_model, days=period_days)
        except Exception:
            conf_trend = None

        if conf_trend is not None:
            trend_data = conf_trend.get("trend", [])
            overall = conf_trend.get("overall", {})
            if not trend_data:
                st.info(t("supervision.confidence.no_probabilities"))
            else:
                df_conf = pd.DataFrame(trend_data)
                df_conf["date"] = pd.to_datetime(df_conf["date"])
                df_conf = df_conf.sort_values("date")
                mean_conf = overall.get("mean_confidence", 0.0)
                low_rate  = overall.get("low_confidence_rate", 0.0)
                mid = len(df_conf) // 2
                delta_str = None
                if mid > 0:
                    dv = df_conf["mean_confidence"].iloc[mid:].mean() - df_conf["mean_confidence"].iloc[:mid].mean()
                    delta_str = t("supervision.confidence.delta_str", val=f"{dv:+.1%}")
                ct1, ct2, ct3 = st.columns(3)
                ct1.metric(t("supervision.confidence.mean_metric"), f"{mean_conf:.2%}", delta=delta_str, help=t("metrics.confiance_moyenne"))
                ct2.metric("P25", f"{overall.get('p25_confidence', 0):.2%}", help=t("metrics.p25_confiance"))
                ct3.metric("P75", f"{overall.get('p75_confidence', 0):.2%}", help=t("metrics.p75_confiance"))
                fig_conf = go.Figure()
                fig_conf.add_trace(go.Scatter(
                    x=pd.concat([df_conf["date"], df_conf["date"][::-1]]),
                    y=pd.concat([df_conf["p75"], df_conf["p25"][::-1]]),
                    fill="toself", fillcolor="rgba(41,128,185,0.15)",
                    line=dict(color="rgba(0,0,0,0)"), name="IQR (P25–P75)", hoverinfo="skip",
                ))
                fig_conf.add_trace(go.Scatter(
                    x=df_conf["date"], y=df_conf["mean_confidence"],
                    name=t("supervision.confidence.mean_trace"), mode="lines+markers",
                    line=dict(color="#2980b9", width=2), marker=dict(size=5),
                ))
                fig_conf.add_hline(y=0.5, line_dash="dot", line_color="#e74c3c",
                                   annotation_text=t("supervision.confidence.threshold_50"), annotation_position="bottom right")
                fig_conf.update_layout(
                    xaxis_title=t("supervision.charts.date_label"), yaxis_title=t("supervision.confidence.y_label"),
                    yaxis=dict(tickformat=".0%", range=[0, 1]),
                    hovermode="x unified", legend=dict(orientation="h", y=1.02),
                )
                st.plotly_chart(fig_conf, width='stretch')
                if low_rate > 0.15:
                    st.warning(t("supervision.confidence.low_rate_warning", pct=int(low_rate * 100)))
        else:
            st.info(t("supervision.confidence.unavailable"))

        st.divider()

        # Anomalous predictions
        st.markdown(t("supervision.anomalies.header"))
        st.caption(t("supervision.anomalies.caption"))
        _anom_col1, _anom_col2 = st.columns([1, 2])
        _anom_days = _anom_col1.number_input(
            t("supervision.anomalies.window_label"), min_value=1, max_value=90,
            value=min(max(period_days, 1), 90), step=1, key="anom_days",
        )
        _anom_z = _anom_col2.slider(
            t("supervision.anomalies.z_threshold_label"), min_value=1.0, max_value=6.0, value=3.0, step=0.1,
            format="%.1f", key="anom_z_threshold",
            help=t("supervision.anomalies.z_threshold_help"),
        )
        try:
            _anom_data = client.get_predictions_anomalies(
                model_name=selected_model, days=int(_anom_days),
                z_threshold=float(_anom_z), limit=200,
            )
        except Exception as _exc:
            _anom_data = None
            st.info(t("supervision.anomalies.unavailable", error=_exc))

        if _anom_data is not None:
            if _anom_data.get("error") == "no_baseline":
                st.info(t("supervision.anomalies.no_baseline"))
            else:
                _total = _anom_data.get("total_checked", 0)
                _count = _anom_data.get("anomalous_count", 0)
                _rate  = _anom_data.get("anomaly_rate", 0.0)
                _preds = _anom_data.get("predictions", [])
                _mc1, _mc2, _mc3 = st.columns(3)
                _mc1.metric(t("supervision.anomalies.metric_analyzed"), _total,
                            help=t("supervision.anomalies.metric_analyzed_help", days=_anom_days))
                _mc2.metric(t("supervision.anomalies.metric_anomalous"), _count,
                            help=t("supervision.anomalies.metric_anomalous_help", z=_anom_z))
                _mc3.metric(t("supervision.anomalies.metric_rate"), f"{_rate:.1%}",
                            help=t("supervision.anomalies.metric_rate_help"))
                if not _preds:
                    st.success(t("supervision.anomalies.none_detected", z=_anom_z, total=_total))
                else:
                    _preds_sorted = sorted(
                        _preds,
                        key=lambda x: -max(f["z_score"] for f in x["anomalous_features"].values()),
                    )
                    _has_confidence = any(_p.get("max_confidence") is not None for _p in _preds_sorted)

                    _rows_anom = []
                    for _p in _preds_sorted:
                        _feats = _p.get("anomalous_features", {})
                        _worst_z_val = max((_f["z_score"] for _f in _feats.values()), default=0.0)
                        _feat_names = ", ".join(
                            f"{_fn} (z={_fd['z_score']:.2f})"
                            for _fn, _fd in sorted(_feats.items(), key=lambda x: -x[1]["z_score"])
                        )
                        _row = {
                            t("supervision.anomaly_table.col_id"): _p["prediction_id"],
                            t("supervision.anomaly_table.col_timestamp"): _p["timestamp"][:19].replace("T", " "),
                            t("supervision.anomaly_table.col_result"): _fmt_pred_result(_p.get("prediction_result")),
                            t("supervision.anomaly_table.col_ground_truth"): _fmt_pred_result(_p.get("ground_truth")),
                            "id_obs": _p.get("id_obs") or "—",
                            t("supervision.anomaly_table.col_z_max"): round(_worst_z_val, 2),
                            t("supervision.anomaly_table.col_anomalous_features"): _feat_names,
                        }
                        if _has_confidence:
                            _row[t("supervision.anomaly_table.col_confidence")] = (
                                f"{_p['max_confidence']:.2%}"
                                if _p.get("max_confidence") is not None else "—"
                            )
                        _rows_anom.append(_row)

                    _cols_order = [
                        t("supervision.anomaly_table.col_id"),
                        t("supervision.anomaly_table.col_timestamp"),
                        t("supervision.anomaly_table.col_result"),
                        t("supervision.anomaly_table.col_ground_truth"),
                        "id_obs",
                    ]
                    if _has_confidence:
                        _cols_order.append(t("supervision.anomaly_table.col_confidence"))
                    _cols_order += [t("supervision.anomaly_table.col_z_max"), t("supervision.anomaly_table.col_anomalous_features")]
                    _df_anom = pd.DataFrame(_rows_anom)[_cols_order]

                    _col_cfg_anom = {
                        t("supervision.anomaly_table.col_id"): st.column_config.NumberColumn(
                            t("supervision.anomaly_table.col_id"),
                            help=t("supervision.anomaly_table.help_id"),
                        ),
                        t("supervision.anomaly_table.col_timestamp"): st.column_config.TextColumn(
                            t("supervision.anomaly_table.col_timestamp"),
                            help=t("supervision.anomaly_table.help_timestamp"),
                        ),
                        t("supervision.anomaly_table.col_result"): st.column_config.TextColumn(
                            t("supervision.anomaly_table.col_result"),
                            help=t("supervision.anomaly_table.help_result"),
                        ),
                        t("supervision.anomaly_table.col_ground_truth"): st.column_config.TextColumn(
                            t("supervision.anomaly_table.col_ground_truth"),
                            help=t("supervision.anomaly_table.help_ground_truth"),
                        ),
                        "id_obs": st.column_config.TextColumn(
                            "id_obs",
                            help=t("supervision.anomaly_table.help_id_obs"),
                        ),
                        t("supervision.anomaly_table.col_z_max"): st.column_config.NumberColumn(
                            t("supervision.anomaly_table.col_z_max"),
                            format="%.2f",
                            help=t("supervision.anomaly_table.help_z_max"),
                        ),
                        t("supervision.anomaly_table.col_anomalous_features"): st.column_config.TextColumn(
                            t("supervision.anomaly_table.col_anomalous_features"),
                            help=t("supervision.anomaly_table.help_anomalous_features"),
                        ),
                    }
                    if _has_confidence:
                        _col_cfg_anom[t("supervision.anomaly_table.col_confidence")] = st.column_config.TextColumn(
                            t("supervision.anomaly_table.col_confidence"),
                            help=t("supervision.anomaly_table.help_confidence"),
                        )
                    st.dataframe(_df_anom, width='stretch', hide_index=True, column_config=_col_cfg_anom)

                    # ── Prediction selector for detail view ─────────────────
                    st.markdown(t("supervision.anomalies.detail_header"))
                    _sel_labels = [
                        f"#{_p['prediction_id']}  ·  {_p['timestamp'][:16].replace('T', ' ')}  "
                        f"·  z_max = {max(f['z_score'] for f in _p['anomalous_features'].values()):.2f}"
                        f"  ({', '.join(sorted(_p['anomalous_features'].keys())[:2])}"
                        f"{'…' if len(_p['anomalous_features']) > 2 else ''})"
                        for _p in _preds_sorted
                    ]
                    _pred_by_label = dict(zip(_sel_labels, _preds_sorted))
                    _sel_label = st.selectbox(
                        t("supervision.anomalies.inspect_selectbox"),
                        options=_sel_labels,
                        index=0,
                        key="anom_detail_sel",
                        label_visibility="collapsed",
                        placeholder=t("supervision.anomalies.inspect_placeholder"),
                    )
                    if _sel_label:
                        _sel_pred = _pred_by_label[_sel_label]
                        _feats = _sel_pred.get("anomalous_features", {})
                        _feats_sorted = sorted(_feats.items(), key=lambda x: -x[1]["z_score"])

                        with st.container(border=True):
                            _hc1, _hc2, _hc3, _hc4 = st.columns([3, 1, 1, 1])
                            _hc1.markdown(
                                f"**{t('supervision.anomalies.pred_header', pred_id=_sel_pred['prediction_id'])}**  ·  "
                                f"{_sel_pred['timestamp'][:19].replace('T', ' ')}"
                                + (f"  ·  id_obs: `{_sel_pred['id_obs']}`" if _sel_pred.get("id_obs") else "")
                            )
                            _hc2.metric(t("supervision.anomaly_table.col_result"), _fmt_pred_result(_sel_pred.get("prediction_result")))
                            if _sel_pred.get("ground_truth") is not None:
                                _hc3.metric(t("supervision.anomaly_table.col_ground_truth"), _fmt_pred_result(_sel_pred.get("ground_truth")),
                                            help=t("supervision.anomalies.ground_truth_help"))
                            if _sel_pred.get("max_confidence") is not None:
                                _hc4.metric(t("supervision.anomaly_table.col_confidence"), f"{_sel_pred['max_confidence']:.2%}")

                            _dc1, _dc2 = st.columns([1, 1])

                            with _dc1:
                                st.caption(t("supervision.anomalies.z_scores_caption"))
                                _z_df = pd.DataFrame([
                                    {"Feature": fn, "Z-score": round(fd["z_score"], 2)}
                                    for fn, fd in _feats_sorted
                                ])
                                _z_max_val = _z_df["Z-score"].max()
                                _fig_z = px.bar(
                                    _z_df, x="Z-score", y="Feature", orientation="h",
                                    color="Z-score",
                                    color_continuous_scale=[
                                        [0.0, "#f39c12"], [0.5, "#e67e22"], [1.0, "#c0392b"]
                                    ],
                                    range_color=[_anom_z, max(_z_max_val, _anom_z + 0.1)],
                                    height=max(160, len(_feats_sorted) * 48 + 60),
                                    labels={"Z-score": "Z-score", "Feature": ""},
                                )
                                _fig_z.add_vline(
                                    x=_anom_z, line_dash="dash", line_color="#7f8c8d",
                                    annotation_text=t("supervision.anomalies.threshold_annotation", z=_anom_z),
                                    annotation_position="top right",
                                )
                                _fig_z.update_layout(
                                    margin=dict(l=0, r=10, t=10, b=30),
                                    yaxis=dict(autorange="reversed"),
                                    coloraxis_showscale=False,
                                    showlegend=False,
                                )
                                st.plotly_chart(_fig_z, width='stretch')

                            with _dc2:
                                st.caption(t("supervision.anomalies.values_vs_baseline_caption"))
                                _feat_detail_rows = [{
                                    t("supervision.feat_detail_table.col_feature"): fn,
                                    t("supervision.feat_detail_table.col_received_value"): round(fd["value"], 4),
                                    t("supervision.feat_detail_table.col_train_mean"): round(fd["baseline_mean"], 4),
                                    t("supervision.feat_detail_table.col_train_std"): round(fd["baseline_std"], 4),
                                    t("supervision.feat_detail_table.col_normal_range"): (
                                        f"[{fd['baseline_mean'] - 2*fd['baseline_std']:.3g}"
                                        f" – {fd['baseline_mean'] + 2*fd['baseline_std']:.3g}]"
                                    ),
                                    t("supervision.feat_detail_table.col_z_score"): round(fd["z_score"], 2),
                                } for fn, fd in _feats_sorted]
                                st.dataframe(
                                    pd.DataFrame(_feat_detail_rows),
                                    hide_index=True, width='stretch',
                                    column_config={
                                        t("supervision.feat_detail_table.col_feature"): st.column_config.TextColumn(t("supervision.feat_detail_table.col_feature")),
                                        t("supervision.feat_detail_table.col_received_value"): st.column_config.NumberColumn(
                                            t("supervision.feat_detail_table.col_received_value"), format="%.4f",
                                            help=t("supervision.feat_detail_table.help_received_value"),
                                        ),
                                        t("supervision.feat_detail_table.col_train_mean"): st.column_config.NumberColumn(
                                            t("supervision.feat_detail_table.col_train_mean_short"), format="%.4f",
                                            help=t("supervision.feat_detail_table.help_train_mean"),
                                        ),
                                        t("supervision.feat_detail_table.col_train_std"): st.column_config.NumberColumn(
                                            t("supervision.feat_detail_table.col_train_std_short"), format="%.4f",
                                            help=t("supervision.feat_detail_table.help_train_std"),
                                        ),
                                        t("supervision.feat_detail_table.col_normal_range"): st.column_config.TextColumn(
                                            t("supervision.feat_detail_table.col_normal_range_label"),
                                            help=t("supervision.feat_detail_table.help_normal_range"),
                                        ),
                                        t("supervision.feat_detail_table.col_z_score"): st.column_config.NumberColumn(
                                            t("supervision.feat_detail_table.col_z_score"), format="%.2f",
                                            help=t("supervision.feat_detail_table.help_z_score"),
                                        ),
                                    },
                                )

    # ────────────────────────────────────────────────────────────────────
    # EXPANDER 4 — Comparaison A/B & Shadow
    # ────────────────────────────────────────────────────────────────────
    if ab_comparison:
        ab_versions = ab_comparison.get("versions", [])
        agreement   = ab_comparison.get("shadow_agreement_rate", {})
        with st.expander(t("supervision.detail.ab_expander"), expanded=False):
            if ab_versions:
                df_ab = pd.DataFrame([{
                    t("supervision.ab_table.col_version"): v["version"],
                    t("supervision.ab_table.col_predictions"): v["total_predictions"],
                    t("supervision.ab_table.col_shadow"): v["shadow_predictions"],
                    t("supervision.ab_table.col_error_rate"): f"{v['error_rate'] * 100:.1f} %",
                    t("supervision.ab_table.col_avg_latency"): f"{v.get('avg_response_time_ms')} ms" if v.get("avg_response_time_ms") else "—",
                    t("supervision.ab_table.col_p95"): f"{v.get('p95_response_time_ms')} ms" if v.get("p95_response_time_ms") else "—",
                    t("supervision.ab_table.col_shadow_agreement"): f"{agreement.get(v['version'], 0) * 100:.1f} %" if v["version"] in agreement else "—",
                } for v in ab_versions])
                st.dataframe(
                    df_ab, width='stretch', hide_index=True,
                    column_config={
                        t("supervision.ab_table.col_version"): st.column_config.TextColumn(
                            t("supervision.ab_table.col_version"),
                            help=t("supervision.ab_table.help_version"),
                        ),
                        t("supervision.ab_table.col_predictions"): st.column_config.NumberColumn(
                            t("supervision.ab_table.col_predictions"),
                            help=t("supervision.ab_table.help_predictions"),
                        ),
                        t("supervision.ab_table.col_shadow"): st.column_config.NumberColumn(
                            t("supervision.ab_table.col_shadow"),
                            help=t("supervision.ab_table.help_shadow"),
                        ),
                        t("supervision.ab_table.col_error_rate"): st.column_config.TextColumn(
                            t("supervision.ab_table.col_error_rate"),
                            help=t("supervision.ab_table.help_error_rate"),
                        ),
                        t("supervision.ab_table.col_avg_latency"): st.column_config.TextColumn(
                            t("supervision.ab_table.col_avg_latency"),
                            help=t("supervision.ab_table.help_avg_latency"),
                        ),
                        t("supervision.ab_table.col_p95"): st.column_config.TextColumn(
                            t("supervision.ab_table.col_p95"),
                            help=t("supervision.ab_table.help_p95"),
                        ),
                        t("supervision.ab_table.col_shadow_agreement"): st.column_config.TextColumn(
                            t("supervision.ab_table.col_shadow_agreement"),
                            help=t("supervision.ab_table.help_shadow_agreement"),
                        ),
                    },
                )

                _all_dist_labels = sorted({
                    str(lbl)
                    for v in ab_versions
                    for lbl in v.get("prediction_distribution", {}).keys()
                })
                _is_reg_ab = any("." in lbl for lbl in _all_dist_labels)

                if _is_reg_ab:
                    st.markdown(t("supervision.ab_regression.header"))
                    _reg_rows = []
                    for v in ab_versions:
                        _dist = {
                            str(k): int(c)
                            for k, c in v.get("prediction_distribution", {}).items()
                        }
                        try:
                            _numeric = [(float(k), c) for k, c in _dist.items()]
                        except (ValueError, TypeError):
                            continue
                        _n = sum(c for _, c in _numeric)
                        if _n == 0:
                            continue
                        _mean = sum(val * c for val, c in _numeric) / _n
                        _var  = sum((val - _mean) ** 2 * c for val, c in _numeric) / _n
                        _reg_rows.append({
                            t("supervision.ab_table.col_version"): v["version"],
                            "N": _n,
                            t("supervision.ab_regression.col_mean_y"): round(_mean, 3),
                            t("supervision.ab_regression.col_std"): round(_var ** 0.5, 3),
                            t("supervision.ab_regression.col_min"): round(min(val for val, _ in _numeric), 3),
                            t("supervision.ab_regression.col_max"): round(max(val for val, _ in _numeric), 3),
                        })
                    if _reg_rows:
                        _df_reg = pd.DataFrame(_reg_rows)
                        if len(_reg_rows) == 2:
                            r0, r1 = _reg_rows[0], _reg_rows[1]
                            _df_reg = pd.concat([
                                _df_reg,
                                pd.DataFrame([{
                                    t("supervision.ab_table.col_version"): f"Δ  ({r1[t('supervision.ab_table.col_version')]} − {r0[t('supervision.ab_table.col_version')]})",
                                    "N": r1["N"] - r0["N"],
                                    t("supervision.ab_regression.col_mean_y"): round(r1[t("supervision.ab_regression.col_mean_y")] - r0[t("supervision.ab_regression.col_mean_y")], 3),
                                    t("supervision.ab_regression.col_std"): round(r1[t("supervision.ab_regression.col_std")] - r0[t("supervision.ab_regression.col_std")], 3),
                                    t("supervision.ab_regression.col_min"): round(r1[t("supervision.ab_regression.col_min")] - r0[t("supervision.ab_regression.col_min")], 3),
                                    t("supervision.ab_regression.col_max"): round(r1[t("supervision.ab_regression.col_max")] - r0[t("supervision.ab_regression.col_max")], 3),
                                }]),
                            ], ignore_index=True)
                        st.dataframe(
                            _df_reg, hide_index=True, width='stretch',
                            column_config={
                                t("supervision.ab_table.col_version"): st.column_config.TextColumn(
                                    t("supervision.ab_table.col_version"),
                                    help=t("supervision.ab_regression.help_version"),
                                ),
                                "N": st.column_config.NumberColumn(
                                    t("supervision.ab_regression.col_n_label"),
                                    help=t("supervision.ab_regression.help_n"),
                                ),
                                t("supervision.ab_regression.col_mean_y"): st.column_config.NumberColumn(
                                    t("supervision.ab_regression.col_mean_y"), format="%.3f",
                                    help=t("supervision.ab_regression.help_mean_y"),
                                ),
                                t("supervision.ab_regression.col_std"): st.column_config.NumberColumn(
                                    t("supervision.ab_regression.col_std"), format="%.3f",
                                    help=t("supervision.ab_regression.help_std"),
                                ),
                                t("supervision.ab_regression.col_min"): st.column_config.NumberColumn(
                                    t("supervision.ab_regression.col_min"), format="%.3f",
                                    help=t("supervision.ab_regression.help_min"),
                                ),
                                t("supervision.ab_regression.col_max"): st.column_config.NumberColumn(
                                    t("supervision.ab_regression.col_max"), format="%.3f",
                                    help=t("supervision.ab_regression.help_max"),
                                ),
                            },
                        )
                        if len(_reg_rows) >= 2:
                            _fig_reg = px.bar(
                                pd.DataFrame(_reg_rows),
                                x=t("supervision.ab_table.col_version"), y=t("supervision.ab_regression.col_mean_y"),
                                error_y=t("supervision.ab_regression.col_std"),
                                color=t("supervision.ab_table.col_version"),
                                text=t("supervision.ab_regression.col_mean_y"),
                                labels={t("supervision.ab_regression.col_mean_y"): t("supervision.ab_regression.y_axis_label")},
                                color_discrete_sequence=px.colors.qualitative.Set2,
                                title=t("supervision.ab_regression.chart_title"),
                            )
                            _fig_reg.update_traces(
                                texttemplate="%{text:.3f}", textposition="outside",
                            )
                            _fig_reg.update_layout(
                                showlegend=False, height=320,
                                yaxis_title=t("supervision.ab_regression.y_axis_label"),
                            )
                            st.plotly_chart(_fig_reg, width='stretch')

                elif _all_dist_labels:
                    st.markdown(t("supervision.ab_classification.header"))
                    _cls_rows_pct = []
                    for v in ab_versions:
                        _dist = {
                            str(k): int(c)
                            for k, c in v.get("prediction_distribution", {}).items()
                        }
                        _total = sum(_dist.values()) or 1
                        _row = {t("supervision.ab_table.col_version"): v["version"], t("supervision.ab_classification.col_total"): sum(_dist.values())}
                        for _lbl in _all_dist_labels:
                            _cnt = _dist.get(_lbl, 0)
                            _row[f"Cl. {_lbl}"] = f"{_cnt / _total:.1%}  ({_cnt})"
                        _cls_rows_pct.append(_row)

                    _cls_col_cfg = {
                        t("supervision.ab_table.col_version"): st.column_config.TextColumn(
                            t("supervision.ab_table.col_version"),
                            help=t("supervision.ab_classification.help_version"),
                        ),
                        t("supervision.ab_classification.col_total"): st.column_config.NumberColumn(
                            t("supervision.ab_classification.col_total"),
                            help=t("supervision.ab_classification.help_total"),
                        ),
                    }
                    for _lbl in _all_dist_labels:
                        _cls_col_cfg[f"Cl. {_lbl}"] = st.column_config.TextColumn(
                            t("supervision.ab_classification.class_col_label", lbl=_lbl),
                            help=t("supervision.ab_classification.class_col_help", lbl=_lbl),
                        )
                    st.dataframe(
                        pd.DataFrame(_cls_rows_pct), hide_index=True, width='stretch',
                        column_config=_cls_col_cfg,
                    )

                    _pct_plot = []
                    for v in ab_versions:
                        _dist = {
                            str(k): int(c)
                            for k, c in v.get("prediction_distribution", {}).items()
                        }
                        _total = sum(_dist.values()) or 1
                        for _lbl in _all_dist_labels:
                            _cnt = _dist.get(_lbl, 0)
                            _pct_plot.append({
                                t("supervision.ab_table.col_version"): v["version"],
                                t("supervision.ab_classification.class_label"): _lbl,
                                "%": round(_cnt / _total * 100, 2),
                                "N": _cnt,
                            })
                    _fig_pct = px.bar(
                        pd.DataFrame(_pct_plot),
                        x=t("supervision.ab_table.col_version"), y="%", color=t("supervision.ab_classification.class_label"),
                        barmode="stack",
                        text=pd.DataFrame(_pct_plot)["%"].apply(lambda v: f"{v:.1f}%"),
                        labels={"%": t("supervision.ab_classification.pct_label"), t("supervision.ab_classification.class_label"): t("supervision.ab_classification.predicted_class_label")},
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        custom_data=["N"],
                        title=t("supervision.ab_classification.chart_title"),
                    )
                    _fig_pct.update_traces(
                        hovertemplate=(
                            t("supervision.ab_classification.hover_template")
                        ),
                    )
                    _fig_pct.update_layout(
                        yaxis=dict(ticksuffix="%", range=[0, 108]),
                        legend_title=t("supervision.ab_classification.predicted_class_label"),
                        height=380,
                    )
                    st.plotly_chart(_fig_pct, width='stretch')

    # ────────────────────────────────────────────────────────────────────
    # EXPANDER 5 — Calibration des probabilités
    # ────────────────────────────────────────────────────────────────────
    with st.expander(t("supervision.detail.calibration_expander"), expanded=False):
        try:
            calib = client.get_model_calibration(model_name=selected_model, start=start_iso, end=end_iso)
        except Exception as exc:
            st.info(t("supervision.calibration.unavailable", error=exc))
            calib = None

        if calib:
            calib_model_type = calib.get("model_type", "classification")
            calib_status = calib.get("calibration_status", "insufficient_data")

            if calib_model_type == "regression":
                if calib_status == "insufficient_data":
                    st.info(t("supervision.calibration.regression_insufficient", count=calib.get('sample_size', 0)))
                else:
                    st.caption(t("supervision.calibration.regression_caption"))
                    STATUS_REG = {
                        "ok":           t("supervision.calibration.status_ok"),
                        "biased_high":  t("supervision.calibration.status_biased_high"),
                        "biased_low":   t("supervision.calibration.status_biased_low"),
                    }
                    _calib_mae   = calib.get("mae")
                    _calib_rmse  = calib.get("rmse")
                    _calib_r2    = calib.get("r2")
                    _calib_bias  = calib.get("bias")
                    _bias_status = STATUS_REG.get(calib_status, calib_status)

                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("MAE", f"{_calib_mae:.3f}" if _calib_mae is not None else "—", help=t("metrics.calib_mae"))
                    rc2.metric("RMSE", f"{_calib_rmse:.3f}" if _calib_rmse is not None else "—", help=t("metrics.calib_rmse"))
                    rc3.metric("R²", f"{_calib_r2:.3f}" if _calib_r2 is not None else "—", help=t("metrics.calib_r2"))
                    rc4.metric(t("supervision.calibration.mean_bias_metric"), f"{_calib_bias:+.3f}" if _calib_bias is not None else "—",
                               delta=_bias_status, delta_color="off", help=t("metrics.calib_biais"))

                    scatter_data = calib.get("scatter_data") or []
                    if scatter_data:
                        _preds = [pt["pred"] for pt in scatter_data]
                        _obs   = [pt["obs"]  for pt in scatter_data]

                        _all_vals  = _preds + _obs
                        _axis_min  = min(_all_vals)
                        _axis_max  = max(_all_vals)
                        _pad       = (_axis_max - _axis_min) * 0.05 or 0.1
                        _diag_vals = [_axis_min - _pad, _axis_max + _pad]

                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(go.Scatter(
                            x=_diag_vals, y=_diag_vals,
                            mode="lines",
                            name=t("supervision.calibration.perfect_calib_trace"),
                            line=dict(color="grey", dash="dot", width=1),
                            hoverinfo="skip",
                        ))
                        fig_scatter.add_trace(go.Scatter(
                            x=_obs, y=_preds,
                            mode="markers",
                            name=t("supervision.calibration.predictions_trace"),
                            marker=dict(color="#2980b9", size=5, opacity=0.6),
                            hovertemplate=t("supervision.calibration.scatter_hover"),
                        ))
                        fig_scatter.update_layout(
                            title=t("supervision.calibration.scatter_title", n=len(scatter_data)),
                            xaxis_title=t("supervision.calibration.scatter_x"),
                            yaxis_title=t("supervision.calibration.scatter_y"),
                            xaxis=dict(range=[_axis_min - _pad, _axis_max + _pad]),
                            yaxis=dict(range=[_axis_min - _pad, _axis_max + _pad]),
                            hovermode="closest",
                            legend=dict(orientation="h", y=1.05),
                        )
                        st.plotly_chart(fig_scatter, width='stretch')

                        _residuals = [p - o for p, o in zip(_preds, _obs)]
                        _res_mean  = sum(_residuals) / len(_residuals)
                        fig_res = go.Figure()
                        fig_res.add_vline(x=0, line=dict(color="grey", dash="dot", width=1))
                        fig_res.add_vline(x=_res_mean, line=dict(color="#e74c3c", dash="dash", width=1.5),
                                          annotation_text=t("supervision.calibration.bias_annotation", val=_res_mean),
                                          annotation_position="top right")
                        fig_res.add_trace(go.Histogram(
                            x=_residuals,
                            name=t("supervision.calibration.residuals_trace"),
                            marker_color="#2980b9",
                            opacity=0.8,
                            hovertemplate=t("supervision.calibration.residuals_hover"),
                        ))
                        fig_res.update_layout(
                            title=t("supervision.calibration.residuals_title"),
                            xaxis_title=t("supervision.calibration.residuals_x"),
                            yaxis_title=t("supervision.calibration.residuals_y"),
                            bargap=0.05,
                        )
                        st.plotly_chart(fig_res, width='stretch')

                    if calib_status in ("biased_high", "biased_low"):
                        _bias_dir = t("supervision.calibration.overestimates") if calib_status == "biased_high" else t("supervision.calibration.underestimates")
                        st.warning(t("supervision.calibration.bias_warning", direction=_bias_dir))

            else:
                if calib_status == "insufficient_data":
                    st.info(t("supervision.calibration.classification_insufficient", count=calib.get('sample_size', 0)))
                else:
                    STATUS_CALIB = {
                        "ok": t("supervision.calibration.status_ok"),
                        "overconfident": t("supervision.calibration.status_overconfident"),
                        "underconfident": t("supervision.calibration.status_underconfident"),
                    }
                    brier = calib.get("brier_score")
                    gap   = calib.get("overconfidence_gap")
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric(t("supervision.calibration.brier_metric"), f"{brier:.4f}" if brier is not None else "—", help=t("metrics.brier_score"))
                    cc2.metric(t("supervision.calibration.gap_metric"), f"{gap:+.2%}" if gap is not None else "—", help=t("metrics.gap_confiance"))
                    cc3.metric(t("supervision.calibration.status_metric"), STATUS_CALIB.get(calib_status, calib_status), help=t("metrics.statut_calibration"))
                    reliability = calib.get("reliability", [])
                    if reliability:
                        bins       = [b["confidence_bin"]  for b in reliability]
                        obs_rates  = [b["observed_rate"]   for b in reliability]
                        counts     = [b["count"]            for b in reliability]
                        diag_vals  = [(float(b.split("–")[0]) + float(b.split("–")[1])) / 2 for b in bins]
                        fig_cal = go.Figure()
                        fig_cal.add_trace(go.Scatter(x=bins, y=diag_vals, name=t("supervision.calibration.perfect_calib_trace"),
                                                     line=dict(color="grey", dash="dot", width=1), mode="lines"))
                        fig_cal.add_trace(go.Scatter(
                            x=bins + bins[::-1], y=obs_rates + diag_vals[::-1],
                            fill="toself", fillcolor="rgba(200,200,200,0.25)",
                            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
                        ))
                        fig_cal.add_trace(go.Scatter(
                            x=bins, y=obs_rates, name=t("supervision.calibration.observed_rate_trace"),
                            mode="markers+lines",
                            marker=dict(size=[max(8, min(24, c // 5)) for c in counts], color="#2980b9"),
                            customdata=counts,
                            hovertemplate=t("supervision.calibration.reliability_hover"),
                        ))
                        fig_cal.update_layout(
                            title=t("supervision.calibration.reliability_title"),
                            xaxis_title=t("supervision.calibration.reliability_x"), yaxis_title=t("supervision.calibration.reliability_y"),
                            yaxis=dict(tickformat=".0%", range=[0, 1]),
                            hovermode="x unified", legend=dict(orientation="h", y=1.05),
                        )
                        st.plotly_chart(fig_cal, width='stretch')
                    if calib_status == "overconfident":
                        st.warning(t("supervision.calibration.overconfident_advice"))

    # ────────────────────────────────────────────────────────────────────
    # EXPANDER 6 — Erreurs récentes
    # ────────────────────────────────────────────────────────────────────
    if recent_errors:
        with st.expander(t("supervision.detail.recent_errors_expander", count=len(recent_errors)), expanded=False):
            for i, err in enumerate(recent_errors, 1):
                st.markdown(f"`{i}.` {err}")

    # ────────────────────────────────────────────────────────────────────
    # EXPANDER 7 — Configuration (seuils + politique) — admin
    # ────────────────────────────────────────────────────────────────────
    _is_sup_admin = st.session_state.get("is_admin", False)

    _prod_ver_for_policy = next(
        (v["version"] for v in per_version if v.get("deployment_mode") == "production"),
        per_version[0]["version"] if per_version else None,
    )
    _policy: dict = {}
    if _prod_ver_for_policy:
        try:
            _policy = client.get_model(_prod_ver_for_policy, _prod_ver_for_policy).get("promotion_policy") or {}
        except Exception:
            try:
                _policy = client.get_model(selected_model, _prod_ver_for_policy).get("promotion_policy") or {}
            except Exception:
                _policy = {}

    _auto_promote_on = _policy.get("auto_promote", False)
    _auto_demote_on  = _policy.get("auto_demote",  False)
    _seuils_set      = bool(_overlay_thresholds)

    _conf_label = t("supervision.config.expander_base_label")
    _conf_badges = []
    if _seuils_set:      _conf_badges.append("🔔")
    if _auto_promote_on: _conf_badges.append("🚀")
    if _auto_demote_on:  _conf_badges.append("⚡")
    if _conf_badges:
        _conf_label += "  " + " ".join(_conf_badges)

    with st.expander(_conf_label, expanded=False):
        _conf_tab_seuils, _conf_tab_promo, _conf_tab_cb = st.tabs([
            t("supervision.config.tab_thresholds"),
            t("supervision.config.tab_auto_promote"),
            t("supervision.config.tab_circuit_breaker"),
        ])

        # ── Alert thresholds ─────────────────────────────────────────────
        with _conf_tab_seuils:
            if not per_version:
                st.info(t("supervision.config.no_version"))
            else:
                _ver_options = [v["version"] for v in per_version]
                _default_ver = next(
                    (v["version"] for v in per_version if v.get("deployment_mode") == "production"),
                    _ver_options[0],
                )
                _sel_ver = st.selectbox(
                    t("supervision.config.version_to_configure"), _ver_options,
                    index=_ver_options.index(_default_ver), key="alert_version_select",
                )
                try:
                    _full = client.get_model(selected_model, _sel_ver)
                    _cur  = _full.get("alert_thresholds") or {}
                except Exception:
                    _cur = {}

                _err_val   = _cur.get("error_rate_max")
                _acc_val   = _cur.get("accuracy_min")
                _drift_val = _cur.get("drift_auto_alert") or False

                if _cur:
                    _ci1, _ci2, _ci3 = st.columns(3)
                    _ci1.metric(t("supervision.config.threshold_error_rate_max"), f"{_err_val * 100:.1f} %" if _err_val is not None else "—")
                    _ci2.metric(t("supervision.config.threshold_accuracy_min"), f"{_acc_val:.0%}" if _acc_val is not None else "—")
                    _ci3.metric(t("supervision.config.threshold_drift_alert"), t("supervision.config.drift_alert_enabled") if _drift_val else t("supervision.config.drift_alert_disabled"))

                with st.form("alert_thresholds_form"):
                    _col1, _col2 = st.columns(2)
                    new_error_rate_pct = _col1.number_input(
                        t("supervision.config.form_error_rate_max"), min_value=0.0, max_value=100.0,
                        value=round(_err_val * 100, 2) if _err_val is not None else 10.0,
                        step=0.5, help=t("supervision.config.form_error_rate_help"),
                    )
                    new_accuracy_min = _col2.number_input(
                        t("supervision.config.form_accuracy_min"), min_value=0.0, max_value=1.0,
                        value=float(_acc_val) if _acc_val is not None else 0.80,
                        step=0.01, format="%.2f", help=t("supervision.config.form_accuracy_help"),
                    )
                    new_drift_auto = st.checkbox(t("supervision.config.form_drift_auto_alert"), value=bool(_drift_val))
                    _submitted = st.form_submit_button(t("supervision.config.form_save_btn"), type="primary")
                    if _submitted:
                        try:
                            client.update_model(selected_model, _sel_ver, {"alert_thresholds": {
                                "error_rate_max": new_error_rate_pct / 100,
                                "accuracy_min": new_accuracy_min,
                                "drift_auto_alert": new_drift_auto,
                            }})
                            st.success(t("supervision.config.thresholds_saved", model=selected_model, version=_sel_ver))
                            st.cache_data.clear()
                        except Exception as _exc:
                            st.error(t("supervision.errors.generic", error=_exc))

        # ── Auto-promotion ────────────────────────────────────────────────
        with _conf_tab_promo:
            st.caption(t("supervision.config.auto_promote_caption", model=selected_model))
            _ap1, _ap2 = st.columns(2)
            with _ap1:
                new_auto_promote = st.checkbox(
                    t("supervision.config.auto_promote_enable"), value=_auto_promote_on,
                    key="sup_auto_promote", disabled=not _is_sup_admin,
                )
                _min_acc = _policy.get("min_accuracy")
                new_min_accuracy = st.number_input(
                    t("supervision.config.min_accuracy_label"),
                    min_value=0.0, max_value=1.0, step=0.01,
                    value=float(_min_acc) if _min_acc is not None else 0.0,
                    key="sup_min_accuracy", disabled=not _is_sup_admin,
                    help=t("supervision.config.min_accuracy_help"),
                )
                _min_golden = _policy.get("min_golden_test_pass_rate")
                new_min_golden = st.number_input(
                    t("supervision.config.min_golden_label"),
                    min_value=0.0, max_value=1.0, step=0.01,
                    value=float(_min_golden) if _min_golden is not None else 0.0,
                    key="sup_min_golden", disabled=not _is_sup_admin,
                )
            with _ap2:
                _max_mae = _policy.get("max_mae")
                new_max_mae = st.number_input(
                    t("supervision.config.max_mae_label"),
                    min_value=0.0, step=0.01,
                    value=float(_max_mae) if _max_mae is not None else 0.0,
                    key="sup_max_mae", disabled=not _is_sup_admin,
                    help=t("supervision.config.max_mae_help"),
                )
                _max_lat = _policy.get("max_latency_p95_ms")
                new_max_latency = st.number_input(
                    t("supervision.config.max_latency_label"),
                    min_value=0.0, step=10.0,
                    value=float(_max_lat) if _max_lat is not None else 0.0,
                    key="sup_max_latency", disabled=not _is_sup_admin,
                )
                new_min_samples = st.number_input(
                    t("supervision.config.min_samples_label"),
                    min_value=1, step=1,
                    value=int(_policy.get("min_sample_validation", 10)),
                    key="sup_min_samples", disabled=not _is_sup_admin,
                )

            if _is_sup_admin and st.button(t("supervision.config.save_auto_promote_btn"), key="save_auto_promote"):
                try:
                    client.set_policy(
                        selected_model,
                        auto_promote=new_auto_promote,
                        min_accuracy=new_min_accuracy if new_min_accuracy > 0 else None,
                        max_mae=new_max_mae if new_max_mae > 0 else None,
                        max_latency_p95_ms=new_max_latency if new_max_latency > 0 else None,
                        min_sample_validation=new_min_samples,
                        min_golden_test_pass_rate=new_min_golden if new_min_golden > 0 else None,
                        auto_demote=_policy.get("auto_demote", False),
                        demote_on_drift=_policy.get("demote_on_drift", "critical"),
                        demote_on_accuracy_below=_policy.get("demote_on_accuracy_below"),
                        demote_cooldown_hours=_policy.get("demote_cooldown_hours", 24),
                    )
                    st.toast(t("supervision.config.auto_promote_saved"), icon="✅")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as _exc:
                    st.error(t("supervision.errors.generic", error=_exc))
            elif not _is_sup_admin:
                st.caption(t("supervision.config.admin_only"))

        # ── Circuit breaker ───────────────────────────────────────────────
        with _conf_tab_cb:
            st.caption(t("supervision.config.circuit_breaker_caption"))
            _cb1, _cb2 = st.columns(2)
            with _cb1:
                new_auto_demote = st.checkbox(
                    t("supervision.config.circuit_breaker_enable"), value=_auto_demote_on,
                    key="sup_auto_demote", disabled=not _is_sup_admin,
                )
                new_demote_on_drift = st.selectbox(
                    t("supervision.config.demote_drift_trigger"),
                    ["warning", "critical"],
                    index=0 if _policy.get("demote_on_drift", "critical") == "warning" else 1,
                    key="sup_demote_on_drift",
                    format_func=lambda x: t(f"supervision.config.drift_level_{x}"),
                    disabled=not _is_sup_admin,
                )
            with _cb2:
                _acc_thr = _policy.get("demote_on_accuracy_below")
                new_demote_accuracy = st.number_input(
                    t("supervision.config.demote_accuracy_label"),
                    min_value=0.0, max_value=1.0, step=0.01,
                    value=float(_acc_thr) if _acc_thr is not None else 0.0,
                    key="sup_demote_accuracy", disabled=not _is_sup_admin,
                    help=t("supervision.config.demote_accuracy_help"),
                )
                new_cooldown = st.number_input(
                    t("supervision.config.cooldown_label"),
                    min_value=0, step=1,
                    value=int(_policy.get("demote_cooldown_hours", 24)),
                    key="sup_cooldown", disabled=not _is_sup_admin,
                    help=t("supervision.config.cooldown_help"),
                )

            if _is_sup_admin and st.button(t("supervision.config.save_circuit_breaker_btn"), key="save_circuit_breaker"):
                try:
                    client.set_policy(
                        selected_model,
                        auto_promote=_policy.get("auto_promote", False),
                        min_accuracy=_policy.get("min_accuracy"),
                        max_mae=_policy.get("max_mae"),
                        max_latency_p95_ms=_policy.get("max_latency_p95_ms"),
                        min_sample_validation=_policy.get("min_sample_validation", 10),
                        min_golden_test_pass_rate=_policy.get("min_golden_test_pass_rate"),
                        auto_demote=new_auto_demote,
                        demote_on_drift=new_demote_on_drift,
                        demote_on_accuracy_below=new_demote_accuracy if new_demote_accuracy > 0 else None,
                        demote_cooldown_hours=new_cooldown,
                    )
                    st.toast(t("supervision.config.circuit_breaker_saved"), icon="✅")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as _exc:
                    st.error(t("supervision.errors.generic", error=_exc))
            elif not _is_sup_admin:
                st.caption(t("supervision.config.admin_only"))

            try:
                _history_data    = client.get_model_history(selected_model)
                _history_entries = _history_data.get("history", [])
                _last_demote     = next((e for e in _history_entries if e.get("action") == "auto_demote"), None)
            except Exception:
                _last_demote = None

            if _last_demote:
                _demote_ts     = _last_demote.get("timestamp", "")[:16].replace("T", " ")
                _demote_reason = (_last_demote.get("snapshot") or {}).get("auto_demote_reason", "—")
                st.error(t("supervision.config.last_demote_error", ts=_demote_ts, reason=_demote_reason), icon="🚨")
            elif _auto_demote_on:
                st.success(t("supervision.config.no_recent_demote"))
