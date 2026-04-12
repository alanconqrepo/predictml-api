"""
Statistiques et graphiques d'utilisation
"""
from datetime import datetime, timedelta, date
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.auth import require_auth, get_client

st.set_page_config(page_title="Stats — PredictML", page_icon="📈", layout="wide")
require_auth()

st.title("📈 Statistiques")

client = get_client()

# --- Sélecteur de période ---
col_period, col_model = st.columns([2, 2])
period = col_period.radio("Période", ["7 jours", "30 jours", "90 jours"], horizontal=True)
days_map = {"7 jours": 7, "30 jours": 30, "90 jours": 90}
days = days_map[period]

try:
    models = client.list_models()
    model_names = sorted({m["name"] for m in models})
except Exception:
    model_names = []

model_filter = col_model.selectbox("Modèle", ["(tous)"] + model_names)
selected_model = None if model_filter == "(tous)" else model_filter

# Utiliser le premier modèle disponible si on veut filtrer
if not model_names:
    st.warning("Aucun modèle disponible.")
    st.stop()

end_dt = datetime.utcnow()
start_dt = end_dt - timedelta(days=days)

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
        df_stats = df_stats.rename(columns={
            "model_name": "Modèle",
            "total_predictions": "Total",
            "error_count": "Erreurs",
            "error_rate": "Taux d'erreur",
            "avg_response_time_ms": "Moy. RT (ms)",
            "p50_response_time_ms": "p50 RT (ms)",
            "p95_response_time_ms": "p95 RT (ms)",
        })
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
col2.metric("Taux d'erreur", f"{error_rate:.1f}%")
col3.metric("Temps de réponse médian", f"{median_rt:.1f} ms")
col4.metric("Modèles utilisés", n_models_used)

st.divider()

# --- Graphiques ---
row1_l, row1_r = st.columns(2)

# Distribution par modèle
with row1_l:
    st.subheader("Distribution par modèle")
    model_counts = df.groupby("model_name").size().reset_index(name="count").sort_values("count", ascending=False)
    fig = px.bar(model_counts, x="model_name", y="count", color="model_name",
                 labels={"model_name": "Modèle", "count": "Nb prédictions"},
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

# Temps de réponse — histogramme
with row1_r:
    st.subheader("Distribution des temps de réponse")
    if "response_time_ms" in df.columns:
        fig = px.histogram(
            df[df["is_error"] == False],
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
        daily, x="date", y="count", color="model_name",
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
        status_daily, x="date", y="count", color="status",
        labels={"date": "Date", "count": "Nb", "status": "Statut"},
        color_discrete_map={"success": "#00CC96", "error": "#EF553B"},
    )
    fig.update_layout(margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

# Boîte à moustaches temps de réponse par modèle
if "response_time_ms" in df.columns and df["response_time_ms"].notna().any():
    st.subheader("Temps de réponse par modèle (boîte à moustaches)")
    fig = px.box(
        df[df["is_error"] == False],
        x="model_name", y="response_time_ms",
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
drift_model = drift_col_model.selectbox(
    "Modèle (drift)",
    model_names,
    key="drift_model",
)
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
        drift_df["rolling_7d"] = (
            drift_df[metric_col].rolling(7, min_periods=1).mean().round(4)
        )
        drift_df["rolling_30d"] = (
            drift_df[metric_col].rolling(30, min_periods=1).mean().round(4)
        )

        # Métriques résumé
        last_val = drift_df[metric_col].iloc[-1]
        prev_7d_val = drift_df[metric_col].iloc[-8] if len(drift_df) >= 8 else drift_df[metric_col].iloc[0]
        delta = round(last_val - prev_7d_val, 4)
        matched_total = int(drift_df["matched_count"].sum())

        m1, m2, m3 = st.columns(3)
        m1.metric(f"{metric_label} (dernier jour)", f"{last_val:.1%}" if model_type == "classification" else f"{last_val:.4f}", delta=f"{delta:+.1%}" if model_type == "classification" else f"{delta:+.4f}")
        m2.metric("Moyenne mobile 7j", f"{drift_df['rolling_7d'].iloc[-1]:.1%}" if model_type == "classification" else f"{drift_df['rolling_7d'].iloc[-1]:.4f}")
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
        fig.add_trace(go.Scatter(
            x=drift_df["date"],
            y=drift_df[metric_col],
            mode="lines+markers",
            name=f"{metric_label} journalier",
            line=dict(color="#AAAAAA", width=1, dash="dot"),
            marker=dict(size=5),
            opacity=0.6,
        ))
        fig.add_trace(go.Scatter(
            x=drift_df["date"],
            y=drift_df["rolling_7d"],
            mode="lines",
            name="Moyenne mobile 7j",
            line=dict(color="#636EFA", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=drift_df["date"],
            y=drift_df["rolling_30d"],
            mode="lines",
            name="Moyenne mobile 30j",
            line=dict(color="#FF7F0E", width=2, dash="dash"),
        ))
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
