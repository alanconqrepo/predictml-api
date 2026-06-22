"""
Alerting history — visualisation des checks d'alerting de supervision.

Chaque ligne correspond à un check d'une métrique pour un modèle lors d'une
exécution de la tâche de supervision (toutes les 6 h). Le champ ``result``
indique si une anomalie a été détectée et si un email/webhook a été envoyé.
"""

from datetime import date, datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.auth import get_client, require_auth
from utils.i18n import t

st.set_page_config(
    page_title="Alerting History — PredictML",
    page_icon="🔔",
    layout="wide",
)
require_auth()

st.title(t("alert_history.page_title"))
st.caption(t("alert_history.caption"))

client = get_client()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
col_start, col_end, col_model, col_type = st.columns([2, 2, 2, 2])

default_end = date.today()
default_start = default_end - timedelta(days=7)

start_date = col_start.date_input(t("alert_history.filters.from_label"), value=default_start)
end_date = col_end.date_input(t("alert_history.filters.to_label"), value=default_end)

if end_date <= start_date:
    st.error(t("alert_history.filters.date_order_error"))
    st.stop()

start_iso = datetime.combine(start_date, datetime.min.time()).isoformat()
end_iso = datetime.combine(end_date, datetime.max.time()).replace(microsecond=0).isoformat()

_CHECK_TYPE_OPTIONS = [
    "error_spike",
    "auc",
    "performance_drift",
    "feature_drift",
    "output_drift",
]

selected_model = col_model.text_input(
    t("alert_history.filters.model_label"),
    value="",
    placeholder=t("alert_history.filters.model_placeholder"),
)
selected_type = col_type.selectbox(
    t("alert_history.filters.type_label"),
    options=[""] + _CHECK_TYPE_OPTIONS,
    format_func=lambda x: t(f"alert_history.check_types.{x}") if x else t("alert_history.filters.all_types"),
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with st.spinner(t("alert_history.loading")):
    try:
        payload = client.get_alert_checks(
            model_name=selected_model.strip() or None,
            check_type=selected_type or None,
            start=start_iso,
            end=end_iso,
            limit=500,
        )
    except Exception as exc:
        st.error(t("alert_history.errors.load", error=exc))
        st.stop()

items = payload.get("items", [])
total = payload.get("total", 0)

if not items:
    st.info(t("alert_history.no_data"))
    st.stop()

df = pd.DataFrame(items)
df["checked_at"] = pd.to_datetime(df["checked_at"])
df["date"] = df["checked_at"].dt.date

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

n_total = len(df)
n_alerts = int((df["result"] == "alert_triggered").sum())
n_sent = int(df["alert_sent"].sum())
n_skipped = int((df["result"] == "skipped_no_predictions").sum())

kpi1.metric(t("alert_history.kpis.total"), n_total)
kpi2.metric(t("alert_history.kpis.alerts"), n_alerts)
kpi3.metric(t("alert_history.kpis.sent"), n_sent)
kpi4.metric(t("alert_history.kpis.skipped"), n_skipped)

if total > len(items):
    st.caption(t("alert_history.truncated_warning", shown=len(items), total=total))

st.divider()

# ---------------------------------------------------------------------------
# Chart: stacked bars per day + check type
# ---------------------------------------------------------------------------
_RESULT_COLOR = {
    "ok": "#4CAF50",
    "alert_triggered": "#f44336",
    "skipped_no_predictions": "#9E9E9E",
    "error": "#FF9800",
}

tab_chart, tab_table = st.tabs([
    t("alert_history.tab_chart"),
    t("alert_history.tab_table"),
])

with tab_chart:
    st.subheader(t("alert_history.chart.title"))

    # Stacked bar: count by date and result
    df_daily = (
        df.groupby(["date", "result"])
        .size()
        .reset_index(name="count")
    )
    df_daily["result_label"] = df_daily["result"].map(
        lambda r: t(f"alert_history.results.{r}")
    )

    if not df_daily.empty:
        fig_bar = px.bar(
            df_daily,
            x="date",
            y="count",
            color="result",
            color_discrete_map=_RESULT_COLOR,
            labels={
                "date": t("alert_history.chart.date_label"),
                "count": t("alert_history.chart.count_label"),
                "result": t("alert_history.chart.result_label"),
            },
            title=t("alert_history.chart.daily_title"),
        )
        fig_bar.update_layout(barmode="stack", legend_title_text=t("alert_history.chart.result_label"))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Breakdown by check type (pie)
    col_pie, col_model_bar = st.columns(2)

    with col_pie:
        df_type = (
            df[df["result"] == "alert_triggered"]
            .groupby("check_type")
            .size()
            .reset_index(name="count")
        )
        if not df_type.empty:
            df_type["type_label"] = df_type["check_type"].map(
                lambda c: t(f"alert_history.check_types.{c}")
            )
            fig_pie = px.pie(
                df_type,
                names="type_label",
                values="count",
                title=t("alert_history.chart.by_type_title"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info(t("alert_history.chart.no_alerts"))

    with col_model_bar:
        df_model = (
            df[df["result"] == "alert_triggered"]
            .groupby("model_name")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=True)
        )
        if not df_model.empty:
            fig_model = px.bar(
                df_model,
                x="count",
                y="model_name",
                orientation="h",
                title=t("alert_history.chart.by_model_title"),
                labels={
                    "model_name": t("alert_history.chart.model_label"),
                    "count": t("alert_history.chart.alerts_count_label"),
                },
                color_discrete_sequence=["#f44336"],
            )
            fig_model.update_layout(yaxis_title="")
            st.plotly_chart(fig_model, use_container_width=True)
        else:
            st.info(t("alert_history.chart.no_alerts"))

with tab_table:
    st.subheader(t("alert_history.table.title"))

    # Result filter
    result_filter = st.multiselect(
        t("alert_history.filters.result_label"),
        options=["ok", "alert_triggered", "skipped_no_predictions", "error"],
        format_func=lambda r: t(f"alert_history.results.{r}"),
        default=[],
        key="result_filter",
    )

    df_display = df.copy()
    if result_filter:
        df_display = df_display[df_display["result"].isin(result_filter)]

    # Build display dataframe
    _RESULT_EMOJI = {
        "ok": "✅ OK",
        "alert_triggered": "🔴 Alerte",
        "skipped_no_predictions": "⏭️ Sauté",
        "error": "⚠️ Erreur",
    }

    def _fmt_result(r: str) -> str:
        return _RESULT_EMOJI.get(r, r)

    def _fmt_details(d) -> str:
        if not d or not isinstance(d, dict):
            return ""
        parts = []
        for k, v in d.items():
            if v is not None:
                parts.append(f"{k}={v}" if not isinstance(v, float) else f"{k}={v:.4f}")
        return ", ".join(parts[:4])

    df_table = pd.DataFrame(
        {
            t("alert_history.table.col_date"): df_display["checked_at"].dt.strftime("%Y-%m-%d %H:%M"),
            t("alert_history.table.col_model"): df_display["model_name"],
            t("alert_history.table.col_version"): df_display["model_version"].fillna("—"),
            t("alert_history.table.col_type"): df_display["check_type"].map(
                lambda c: t(f"alert_history.check_types.{c}")
            ),
            t("alert_history.table.col_result"): df_display["result"].map(_fmt_result),
            t("alert_history.table.col_mail"): df_display["alert_sent"].map(lambda x: "✅" if x else ""),
            t("alert_history.table.col_webhook"): df_display["webhook_sent"].map(lambda x: "✅" if x else ""),
            t("alert_history.table.col_new_preds"): df_display["new_predictions_count"].fillna("—"),
            t("alert_history.table.col_details"): df_display["details"].map(_fmt_details),
        }
    )

    st.dataframe(df_table, use_container_width=True, hide_index=True)
    st.caption(t("alert_history.table.row_count", count=len(df_table)))
