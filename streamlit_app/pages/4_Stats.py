"""
Statistiques et graphiques d'utilisation
"""

from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.api_client import get_models as get_models_cached
from utils.auth import get_client, require_auth
from utils.metrics_help import METRIC_HELP

st.set_page_config(page_title="Stats — PredictML", page_icon="📈", layout="wide")
require_auth()

col_title, col_refresh = st.columns([8, 1])
col_title.title("📈 Statistiques")
if col_refresh.button("🔄 Rafraîchir", key="stats_refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

client = get_client()

# --- Sélecteur de période ---
col_period, col_model = st.columns([2, 2])
period = col_period.radio("Période", ["7 jours", "30 jours", "90 jours"], horizontal=True)
days_map = {"7 jours": 7, "30 jours": 30, "90 jours": 90}
days = days_map[period]

try:
    models = get_models_cached(
        st.session_state.get("api_url"), st.session_state.get("api_token")
    )
    model_names = sorted({m["name"] for m in models})
except Exception:
    model_names = []

with col_model:
    stats_search = st.text_input("Filtrer par nom", key="stats_model_search", placeholder="Rechercher…")
    stats_filtered = [n for n in model_names if stats_search.lower() in n.lower()] if stats_search else model_names
    model_filter = st.selectbox("Modèle", ["(tous)"] + (stats_filtered or model_names))
selected_model = None if model_filter == "(tous)" else model_filter

# Utiliser le premier modèle disponible si on veut filtrer
if not model_names:
    st.warning("Aucun modèle disponible.")
    st.stop()

end_dt = datetime.utcnow()
start_dt = end_dt - timedelta(days=days)

# --- 🏆 Leaderboard ---
st.subheader("🏆 Leaderboard — Modèles en production")

lb_col_metric, _ = st.columns([2, 3])
lb_metric = lb_col_metric.selectbox(
    "Trier par",
    options=["accuracy", "f1_score", "latency_p95_ms", "predictions_count"],
    format_func=lambda x: {
        "accuracy": "Accuracy",
        "f1_score": "F1 Score",
        "latency_p95_ms": "Latence p95",
        "predictions_count": "Volume de prédictions",
    }.get(x, x),
    key="lb_metric",
)

_DRIFT_EMOJI = {
    "ok": "🟢 ok",
    "warning": "🟡 warning",
    "critical": "🔴 critique",
    "no_baseline": "⚪ pas de baseline",
    "no_data": "⚪ pas de données",
    "insufficient_data": "⚪ données insuffisantes",
    "unknown": "⚪ inconnu",
}


def _bg_accuracy(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v >= 0.90:
        return "background-color: rgba(39, 174, 96, 0.25)"
    if v >= 0.70:
        return "background-color: rgba(241, 196, 15, 0.25)"
    return "background-color: rgba(231, 76, 60, 0.25)"


def _build_leaderboard_fallback(models_list, stats_list, metric, n_days):
    """Construit le leaderboard côté client si l'endpoint API n'est pas disponible."""
    stats_by_name = {s["model_name"]: s for s in stats_list}
    rows = [
        {
            "rank": 0,
            "name": m["name"],
            "version": m.get("version", ""),
            "accuracy": m.get("accuracy"),
            "f1_score": m.get("f1_score"),
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
    elif metric == "f1_score":
        rows.sort(key=lambda r: r["f1_score"] if r["f1_score"] is not None else -1, reverse=True)
    else:
        rows.sort(key=lambda r: r["accuracy"] if r["accuracy"] is not None else -1, reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


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

_DRIFT_COLOR = {
    "ok": "#2ECC71",
    "warning": "#F39C12",
    "critical": "#E74C3C",
    "no_baseline": "#95A5A6",
    "no_data": "#95A5A6",
    "insufficient_data": "#95A5A6",
    "unknown": "#95A5A6",
}

tab_table, tab_scatter = st.tabs(["Tableau", "Comparaison"])

with tab_table:
    if leaderboard:
        df_lb = pd.DataFrame(leaderboard)
        df_display = df_lb.rename(
            columns={
                "rank": "Rang",
                "name": "Modèle",
                "version": "Version",
                "accuracy": "Accuracy",
                "f1_score": "F1 Score",
                "latency_p95_ms": "Latence p95 (ms)",
                "drift_status": "Drift",
                "predictions_count": f"Prédictions ({days}j)",
            }
        )
        df_display["Drift"] = df_display["Drift"].map(lambda x: _DRIFT_EMOJI.get(x, x))
        df_display["Accuracy"] = df_display["Accuracy"].round(4)
        df_display["F1 Score"] = df_display["F1 Score"].round(4)
        df_display["Latence p95 (ms)"] = df_display["Latence p95 (ms)"].apply(
            lambda x: f"{x:.0f} ms" if pd.notna(x) and x is not None else "—"
        )
        styled = df_display.style.map(_bg_accuracy, subset=["Accuracy"]).hide(axis="index")
        st.dataframe(styled, use_container_width=True)
    else:
        st.info("Aucun modèle en production trouvé.")

with tab_scatter:
    if not leaderboard:
        st.info("Aucun modèle en production trouvé.")
    else:
        df_scatter = pd.DataFrame([
            {
                "name": e["name"],
                "version": e["version"],
                "accuracy": e["accuracy"],
                "f1_score": e["f1_score"],
                "latency_p95_ms": e["latency_p95_ms"],
                "drift_status": e["drift_status"],
                "predictions_count": e["predictions_count"],
            }
            for e in leaderboard
        ])

        scatter_col_metric, scatter_col_sla, scatter_col_acc = st.columns([2, 2, 2])
        scatter_y_metric = scatter_col_metric.selectbox(
            "Métrique Y",
            options=["accuracy", "f1_score"],
            format_func=lambda x: {"accuracy": "Accuracy", "f1_score": "F1 Score"}.get(x, x),
            key="scatter_y_metric",
        )
        sla_latency = scatter_col_sla.number_input(
            "SLA latence (ms)",
            min_value=0,
            max_value=10000,
            value=0,
            step=10,
            help="Affiche une ligne verticale de seuil. 0 = désactivé.",
            key="scatter_sla_latency",
        )
        min_accuracy = scatter_col_acc.number_input(
            "Accuracy minimum",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            format="%.2f",
            help="Affiche une ligne horizontale de seuil. 0 = désactivé.",
            key="scatter_min_accuracy",
        )

        y_label = "Accuracy" if scatter_y_metric == "accuracy" else "F1 Score"

        df_plot = df_scatter.dropna(subset=["latency_p95_ms", scatter_y_metric]).copy()
        df_plot["bubble_size"] = df_plot["predictions_count"].clip(lower=1)
        df_plot["color"] = df_plot["drift_status"].map(
            lambda s: _DRIFT_COLOR.get(s, _DRIFT_COLOR["unknown"])
        )
        df_plot["drift_label"] = df_plot["drift_status"].map(
            lambda s: _DRIFT_EMOJI.get(s, s)
        )
        df_plot["label"] = df_plot["name"] + " v" + df_plot["version"]

        if df_plot.empty:
            st.info(
                "Données insuffisantes pour le scatter plot : les modèles doivent avoir "
                "une latence p95 et une accuracy renseignées."
            )
        else:
            fig = go.Figure()

            for _, row in df_plot.iterrows():
                acc_val = row[scatter_y_metric]
                lat_val = row["latency_p95_ms"]
                count = int(row["predictions_count"])
                bubble = max(10, min(60, count ** 0.5))

                acc_str = f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "—"
                f1_str = f"{row['f1_score']:.4f}" if row["f1_score"] is not None else "—"
                hover_text = (
                    f"<b>{row['label']}</b><br>"
                    f"Accuracy : {acc_str}<br>"
                    f"F1 Score : {f1_str}<br>"
                    f"Latence p95 : {lat_val:.0f} ms<br>"
                    f"Prédictions : {count:,}<br>"
                    f"Drift : {row['drift_label']}"
                )

                fig.add_trace(go.Scatter(
                    x=[lat_val],
                    y=[acc_val],
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

            if sla_latency > 0:
                fig.add_vline(
                    x=sla_latency,
                    line_dash="dash",
                    line_color="#E74C3C",
                    annotation_text=f"SLA {sla_latency} ms",
                    annotation_position="top right",
                    annotation_font_color="#E74C3C",
                )
            if min_accuracy > 0:
                fig.add_hline(
                    y=min_accuracy,
                    line_dash="dash",
                    line_color="#E74C3C",
                    annotation_text=f"Min {y_label} {min_accuracy:.0%}",
                    annotation_position="bottom right",
                    annotation_font_color="#E74C3C",
                )

            fig.update_layout(
                xaxis_title="Latence p95 (ms)",
                yaxis_title=y_label,
                yaxis_range=[0, 1.05],
                yaxis_tickformat=".0%",
                margin=dict(t=40, b=40),
                hovermode="closest",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Taille des bulles proportionnelle au volume de prédictions. "
                "Couleur : 🟢 ok · 🟡 warning · 🔴 critique · ⚪ pas de baseline / données."
            )

st.divider()

# Charger les prédictions pour chaque modèle (ou le modèle sélectionné)
all_preds = []

fetch_models_list = [selected_model] if selected_model else model_names

with st.spinner("Chargement des statistiques..."):
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

# --- Statistiques agrégées par modèle (endpoint /predictions/stats) ---
st.divider()
st.subheader("📊 Statistiques agrégées par modèle")
try:
    raw_stats = client.get_prediction_stats(days=days, model_name=selected_model)
    if raw_stats:
        df_stats = pd.DataFrame(raw_stats)
        df_stats = df_stats.rename(
            columns={
                "model_name": "Modèle",
                "total_predictions": "Total",
                "error_count": "Erreurs",
                "error_rate": "Taux d'erreur",
                "avg_response_time_ms": "Moy. RT (ms)",
                "p50_response_time_ms": "p50 RT (ms)",
                "p95_response_time_ms": "p95 RT (ms)",
            }
        )
        df_stats["Taux d'erreur"] = (df_stats["Taux d'erreur"] * 100).round(2).astype(str) + " %"
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune donnée pour cette période.")
except Exception as e:
    st.warning(f"Impossible de charger les statistiques agrégées : {e}")

if not all_preds:
    st.info("Aucune prédiction dans la période sélectionnée.")
    st.stop()

df = pd.DataFrame(all_preds)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.floor("h")
df["is_error"] = df["status"] == "error"

# --- Métriques ---
st.divider()
total = len(df)
error_rate = df["is_error"].mean() * 100
median_rt = df["response_time_ms"].median() if "response_time_ms" in df.columns else 0
n_models_used = df["model_name"].nunique()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total prédictions", f"{total:,}")
col2.metric("Taux d'erreur", f"{error_rate:.1f}%", help=METRIC_HELP["taux_erreur"])
col3.metric("Temps de réponse médian", f"{median_rt:.1f} ms", help=METRIC_HELP["latence_mediane"])
col4.metric("Modèles utilisés", n_models_used)

st.divider()

# --- Graphiques ---
row1_l, row1_r = st.columns(2)

# Distribution par modèle
with row1_l:
    st.subheader("Distribution par modèle")
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
        labels={"model_name": "Modèle", "count": "Nb prédictions"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(showlegend=False, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

# Temps de réponse — histogramme
with row1_r:
    st.subheader("Distribution des temps de réponse")
    if "response_time_ms" in df.columns:
        fig = px.histogram(
            df[~df["is_error"]],
            x="response_time_ms",
            nbins=40,
            labels={"response_time_ms": "Temps (ms)", "count": "Nb"},
            color_discrete_sequence=["#636EFA"],
        )
        fig.update_layout(margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Données de temps de réponse non disponibles.")

row2_l, row2_r = st.columns(2)

# Série temporelle — prédictions par jour
with row2_l:
    st.subheader("Prédictions par jour")
    daily = df.groupby(["date", "model_name"]).size().reset_index(name="count")
    fig = px.line(
        daily,
        x="date",
        y="count",
        color="model_name",
        labels={"date": "Date", "count": "Nb prédictions", "model_name": "Modèle"},
        markers=True,
    )
    fig.update_layout(margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

# Succès vs Erreurs par jour
with row2_r:
    st.subheader("Succès vs Erreurs par jour")
    status_daily = df.groupby(["date", "status"]).size().reset_index(name="count")
    fig = px.area(
        status_daily,
        x="date",
        y="count",
        color="status",
        labels={"date": "Date", "count": "Nb", "status": "Statut"},
        color_discrete_map={"success": "#00CC96", "error": "#EF553B"},
    )
    fig.update_layout(margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

# Boîte à moustaches temps de réponse par modèle
if "response_time_ms" in df.columns and df["response_time_ms"].notna().any():
    st.subheader("Temps de réponse par modèle (boîte à moustaches)")
    fig = px.box(
        df[~df["is_error"]],
        x="model_name",
        y="response_time_ms",
        color="model_name",
        labels={"model_name": "Modèle", "response_time_ms": "Temps (ms)"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(showlegend=False, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

# --- Drift de performance (accuracy rolling) ---
st.divider()
st.subheader("Drift de performance — accuracy rolling (30j)")

drift_col_model, drift_col_threshold = st.columns([2, 2])
with drift_col_model:
    drift_search = st.text_input("Filtrer par nom", key="drift_model_search", placeholder="Rechercher…")
    drift_filtered = [n for n in model_names if drift_search.lower() in n.lower()] if drift_search else model_names
    drift_model = st.selectbox("Modèle (drift)", drift_filtered or model_names, key="drift_model")
alert_enabled = drift_col_threshold.checkbox("Activer alerte seuil", value=True)
threshold = drift_col_threshold.slider(
    "Seuil d'alerte accuracy",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05,
    disabled=not alert_enabled,
)

drift_end = datetime.utcnow()
drift_start = drift_end - timedelta(days=30)

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
    st.warning(f"Impossible de charger les métriques de performance : {e}")
    by_period = []
    model_type = "classification"

if not by_period:
    st.info(
        "Pas assez de données observées pour ce modèle sur les 30 derniers jours. "
        "Soumettez des résultats via POST /observed-results pour activer le suivi."
    )
else:
    metric_col = "accuracy" if model_type == "classification" else "mae"
    metric_label = "Accuracy" if model_type == "classification" else "MAE"

    drift_df = pd.DataFrame(by_period)
    drift_df["date"] = pd.to_datetime(drift_df["period"])
    drift_df = drift_df.sort_values("date").reset_index(drop=True)

    if metric_col in drift_df.columns and drift_df[metric_col].notna().any():
        drift_df["rolling_7d"] = drift_df[metric_col].rolling(7, min_periods=1).mean().round(4)
        drift_df["rolling_30d"] = drift_df[metric_col].rolling(30, min_periods=1).mean().round(4)

        # Métriques résumé
        last_val = drift_df[metric_col].iloc[-1]
        prev_7d_val = (
            drift_df[metric_col].iloc[-8] if len(drift_df) >= 8 else drift_df[metric_col].iloc[0]
        )
        delta = round(last_val - prev_7d_val, 4)
        matched_total = int(drift_df["matched_count"].sum())

        m1, m2, m3 = st.columns(3)
        _perf_help = METRIC_HELP["accuracy"] if model_type == "classification" else METRIC_HELP["mae"]
        m1.metric(
            f"{metric_label} (dernier jour)",
            f"{last_val:.1%}" if model_type == "classification" else f"{last_val:.4f}",
            delta=f"{delta:+.1%}" if model_type == "classification" else f"{delta:+.4f}",
            help=_perf_help,
        )
        m2.metric(
            "Moyenne mobile 7j",
            (
                f"{drift_df['rolling_7d'].iloc[-1]:.1%}"
                if model_type == "classification"
                else f"{drift_df['rolling_7d'].iloc[-1]:.4f}"
            ),
            help=_perf_help,
        )
        m3.metric("Prédictions avec résultat observé (30j)", f"{matched_total:,}")

        # Alerte drift
        if alert_enabled and model_type == "classification":
            last_rolling_7d = drift_df["rolling_7d"].iloc[-1]
            if last_rolling_7d < threshold:
                st.warning(
                    f"Drift détecté : la moyenne mobile 7j de l'accuracy ({last_rolling_7d:.1%}) "
                    f"est en dessous du seuil configuré ({threshold:.0%})."
                )

        # Graphique Plotly
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=drift_df["date"],
                y=drift_df[metric_col],
                mode="lines+markers",
                name=f"{metric_label} journalier",
                line=dict(color="#AAAAAA", width=1, dash="dot"),
                marker=dict(size=5),
                opacity=0.6,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=drift_df["date"],
                y=drift_df["rolling_7d"],
                mode="lines",
                name="Moyenne mobile 7j",
                line=dict(color="#636EFA", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=drift_df["date"],
                y=drift_df["rolling_30d"],
                mode="lines",
                name="Moyenne mobile 30j",
                line=dict(color="#FF7F0E", width=2, dash="dash"),
            )
        )
        if alert_enabled and model_type == "classification":
            fig.add_hline(
                y=threshold,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Seuil {threshold:.0%}",
                annotation_position="bottom right",
            )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=metric_label,
            yaxis_tickformat=".0%" if model_type == "classification" else None,
            yaxis_range=[0, 1.05] if model_type == "classification" else None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Métrique '{metric_col}' non disponible pour ce modèle (type : {model_type}).")
