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
