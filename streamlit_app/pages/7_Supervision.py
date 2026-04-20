"""
Tableau de bord de supervision des modèles ML.

Vue globale : état de santé de tous les modèles sur une plage calendaire.
Vue détaillée : zoom sur un modèle (série temporelle, drift, A/B testing).
"""

from datetime import datetime, timedelta, date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.auth import get_client, require_auth

st.set_page_config(
    page_title="Supervision — PredictML",
    page_icon="🔍",
    layout="wide",
)
require_auth()

st.title("🔍 Supervision des modèles")

client = get_client()

# ---------------------------------------------------------------------------
# Barre de filtres : plage de dates
# ---------------------------------------------------------------------------
col_start, col_end, col_refresh = st.columns([2, 2, 1])

default_end = date.today()
default_start = default_end - timedelta(days=7)

start_date = col_start.date_input("Du", value=default_start)
end_date = col_end.date_input("Au", value=default_end)

if end_date <= start_date:
    st.error("La date de fin doit être postérieure à la date de début.")
    st.stop()

start_iso = datetime.combine(start_date, datetime.min.time()).isoformat()
end_iso = datetime.combine(end_date, datetime.max.time()).replace(microsecond=0).isoformat()

# ---------------------------------------------------------------------------
# Chargement de la vue d'ensemble
# ---------------------------------------------------------------------------
with st.spinner("Chargement de la vue d'ensemble…"):
    try:
        overview = client.get_monitoring_overview(start=start_iso, end=end_iso)
    except Exception as exc:
        st.error(f"Impossible de charger la vue d'ensemble : {exc}")
        st.stop()

gs = overview.get("global_stats", {})
models_data = overview.get("models", [])

# ---------------------------------------------------------------------------
# Section 1 — KPIs globaux
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📊 Vue d'ensemble")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Prédictions", f"{gs.get('total_predictions', 0):,}")
k2.metric(
    "Taux d'erreur",
    f"{gs.get('error_rate', 0) * 100:.1f} %",
    delta=None,
)
k3.metric(
    "Latence moy.",
    f"{gs.get('avg_latency_ms') or '—'} ms" if gs.get("avg_latency_ms") else "—",
)
k4.metric("Modèles actifs", gs.get("active_models", 0))

alerts = gs.get("models_critical", 0) + gs.get("models_warning", 0)
k5.metric(
    "Alertes",
    f"🔴 {gs.get('models_critical', 0)} · 🟡 {gs.get('models_warning', 0)}",
    delta=f"{alerts} modèle(s) à surveiller" if alerts else "Tout est OK",
    delta_color="inverse" if alerts else "normal",
)

# ---------------------------------------------------------------------------
# Section 2 — Tableau de santé par modèle
# ---------------------------------------------------------------------------
if not models_data:
    st.info("Aucune prédiction dans la période sélectionnée.")
    st.stop()

st.divider()
st.subheader("🏥 Santé par modèle")

STATUS_ICON = {
    "ok": "🟢",
    "warning": "🟡",
    "critical": "🔴",
    "no_data": "⚪",
    "no_baseline": "⚪",
    "insufficient_data": "⚪",
}


def _icon(status: str) -> str:
    return STATUS_ICON.get(status, "⚪") + " " + status


rows_table = []
for m in models_data:
    rows_table.append(
        {
            "Modèle": m["model_name"],
            "Versions": ", ".join(m.get("versions", [])),
            "Mode": ", ".join(
                sorted(set(v for v in m.get("deployment_modes", {}).values() if v))
            ) or "—",
            "Prédictions": m["total_predictions"],
            "Shadow": m["shadow_predictions"],
            "Erreurs": f"{m['error_rate'] * 100:.1f} %",
            "Latence moy.": f"{m['avg_latency_ms'] or '—'} ms" if m.get("avg_latency_ms") else "—",
            "p95": f"{m['p95_latency_ms']} ms" if m.get("p95_latency_ms") else "—",
            "Drift features": _icon(m.get("feature_drift_status", "no_data")),
            "Drift perf.": _icon(m.get("performance_drift_status", "no_data")),
            "Statut": _icon(m.get("health_status", "no_data")),
        }
    )

df_health = pd.DataFrame(rows_table)
st.dataframe(df_health, use_container_width=True, hide_index=True)

# Graphiques côte à côte : volume et erreurs
col_vol, col_err = st.columns(2)

df_models = pd.DataFrame(
    [
        {
            "model_name": m["model_name"],
            "total_predictions": m["total_predictions"],
            "error_rate_pct": round(m["error_rate"] * 100, 2),
        }
        for m in models_data
    ]
)

fig_vol = px.bar(
    df_models.sort_values("total_predictions", ascending=False),
    x="model_name",
    y="total_predictions",
    title="Volume de prédictions par modèle",
    labels={"model_name": "Modèle", "total_predictions": "Prédictions"},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig_vol.update_layout(showlegend=False)
col_vol.plotly_chart(fig_vol, use_container_width=True)

fig_err = px.bar(
    df_models.sort_values("error_rate_pct", ascending=False),
    x="model_name",
    y="error_rate_pct",
    title="Taux d'erreur par modèle (%)",
    labels={"model_name": "Modèle", "error_rate_pct": "Erreurs (%)"},
    color="error_rate_pct",
    color_continuous_scale=["#27ae60", "#e67e22", "#c0392b"],
)
fig_err.add_hline(y=5, line_dash="dash", line_color="#e67e22", annotation_text="Seuil warning 5%")
fig_err.add_hline(y=10, line_dash="dash", line_color="#c0392b", annotation_text="Seuil critique 10%")
fig_err.update_layout(showlegend=False, coloraxis_showscale=False)
col_err.plotly_chart(fig_err, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3 — Détail par modèle
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🔎 Détail d'un modèle")

model_names = [m["model_name"] for m in models_data]
selected_model = st.selectbox("Sélectionner un modèle", model_names)

if not selected_model:
    st.stop()

with st.spinner(f"Chargement du détail pour {selected_model}…"):
    try:
        detail = client.get_monitoring_model(
            name=selected_model, start=start_iso, end=end_iso
        )
    except Exception as exc:
        st.error(f"Impossible de charger le détail : {exc}")
        st.stop()

per_version = detail.get("per_version_stats", [])
timeseries = detail.get("timeseries", [])
perf_by_day = detail.get("performance_by_day", [])
feature_drift = detail.get("feature_drift", {})
ab_comparison = detail.get("ab_comparison")
recent_errors = detail.get("recent_errors", [])

# --- Stats par version ---
if per_version:
    st.markdown("#### 📋 Statistiques par version")
    df_ver = pd.DataFrame(
        [
            {
                "Version": v["version"],
                "Mode": v.get("deployment_mode") or "—",
                "Poids trafic": f"{v['traffic_weight']:.0%}" if v.get("traffic_weight") else "—",
                "Prédictions": v["total_predictions"],
                "Shadow": v["shadow_predictions"],
                "Erreurs": f"{v['error_rate'] * 100:.1f} %",
                "Latence moy.": f"{v['avg_latency_ms'] or '—'} ms" if v.get("avg_latency_ms") else "—",
                "p50": f"{v['p50_latency_ms']} ms" if v.get("p50_latency_ms") else "—",
                "p95": f"{v['p95_latency_ms']} ms" if v.get("p95_latency_ms") else "—",
            }
            for v in per_version
        ]
    )
    st.dataframe(df_ver, use_container_width=True, hide_index=True)

# --- Série temporelle ---
if timeseries:
    df_ts = pd.DataFrame(timeseries)
    df_ts["date"] = pd.to_datetime(df_ts["date"])

    col_ts1, col_ts2 = st.columns(2)

    # Volume + erreurs
    fig_vol_ts = go.Figure()
    fig_vol_ts.add_trace(
        go.Scatter(
            x=df_ts["date"],
            y=df_ts["total_predictions"],
            name="Prédictions",
            fill="tozeroy",
            line=dict(color="#2980b9"),
        )
    )
    fig_vol_ts.add_trace(
        go.Scatter(
            x=df_ts["date"],
            y=df_ts["error_count"],
            name="Erreurs",
            fill="tozeroy",
            line=dict(color="#c0392b"),
        )
    )
    fig_vol_ts.update_layout(
        title="Prédictions & erreurs par jour",
        xaxis_title="Date",
        yaxis_title="Nombre",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
    )
    col_ts1.plotly_chart(fig_vol_ts, use_container_width=True)

    # Latence
    if df_ts["avg_latency_ms"].notna().any():
        fig_lat = go.Figure()
        for col_name, label, color in [
            ("avg_latency_ms", "Moyenne", "#3498db"),
            ("p50_latency_ms", "p50", "#27ae60"),
            ("p95_latency_ms", "p95", "#e67e22"),
        ]:
            if col_name in df_ts and df_ts[col_name].notna().any():
                fig_lat.add_trace(
                    go.Scatter(
                        x=df_ts["date"],
                        y=df_ts[col_name],
                        name=label,
                        line=dict(color=color),
                    )
                )
        fig_lat.update_layout(
            title="Latence par jour (ms)",
            xaxis_title="Date",
            yaxis_title="ms",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02),
        )
        col_ts2.plotly_chart(fig_lat, use_container_width=True)

    # Taux d'erreur par jour (area)
    fig_err_ts = px.area(
        df_ts,
        x="date",
        y="error_rate",
        title="Taux d'erreur par jour",
        labels={"date": "Date", "error_rate": "Taux d'erreur"},
        color_discrete_sequence=["#e74c3c"],
    )
    fig_err_ts.add_hline(
        y=0.05, line_dash="dash", line_color="#e67e22", annotation_text="5%"
    )
    fig_err_ts.add_hline(
        y=0.10, line_dash="dash", line_color="#c0392b", annotation_text="10%"
    )
    fig_err_ts.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig_err_ts, use_container_width=True)

# --- Drift de performance ---
st.divider()
col_perf, col_feat = st.columns(2)

with col_perf:
    st.markdown("#### 📈 Drift de performance")
    if perf_by_day:
        df_perf = pd.DataFrame(perf_by_day)
        df_perf["date"] = pd.to_datetime(df_perf["date"])
        df_perf = df_perf[df_perf["matched_count"] > 0]

        if not df_perf.empty:
            # Moyenne mobile 3 jours
            df_perf = df_perf.sort_values("date")
            df_perf["rolling_3d"] = df_perf["accuracy"].rolling(window=3, min_periods=1).mean()

            fig_perf = go.Figure()
            fig_perf.add_trace(
                go.Scatter(
                    x=df_perf["date"],
                    y=df_perf["accuracy"],
                    name="Accuracy / jour",
                    mode="markers+lines",
                    marker=dict(size=6),
                    line=dict(color="#95a5a6"),
                )
            )
            fig_perf.add_trace(
                go.Scatter(
                    x=df_perf["date"],
                    y=df_perf["rolling_3d"],
                    name="Moyenne 3j",
                    line=dict(color="#2980b9", width=2),
                )
            )
            fig_perf.update_layout(
                xaxis_title="Date",
                yaxis_title="Accuracy",
                yaxis=dict(tickformat=".0%"),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_perf, use_container_width=True)

            # Indicateur de drift
            mid = len(df_perf) // 2
            if mid > 0:
                avg1 = df_perf["accuracy"].iloc[:mid].mean()
                avg2 = df_perf["accuracy"].iloc[mid:].mean()
                drop = avg1 - avg2
                drift_label = (
                    "🔴 Critique" if drop >= 0.10
                    else "🟡 Attention" if drop >= 0.05
                    else "🟢 Stable"
                )
                st.metric(
                    "Tendance performance",
                    drift_label,
                    delta=f"{-drop:.1%} (1re vs 2e moitié)",
                    delta_color="inverse",
                )
        else:
            st.info("Aucune observation liée (id_obs) dans la période.")
    else:
        st.info("Aucune donnée de performance disponible.")

# --- Drift des features ---
with col_feat:
    st.markdown("#### 🌡️ Drift des features")
    feat_summary = feature_drift.get("drift_summary", "no_data")
    feat_baseline = feature_drift.get("baseline_available", False)
    feat_analyzed = feature_drift.get("predictions_analyzed", 0)

    summary_icon = STATUS_ICON.get(feat_summary, "⚪")
    st.markdown(f"**Statut global** : {summary_icon} `{feat_summary}`")
    st.caption(f"{feat_analyzed} prédictions analysées")

    features_dict = feature_drift.get("features", {})
    if features_dict and feat_baseline:
        rows_drift = []
        for feat_name, feat_data in features_dict.items():
            drift_status = feat_data.get("drift_status", "no_data")
            rows_drift.append(
                {
                    "Feature": feat_name,
                    "Statut": _icon(drift_status),
                    "Moy. prod.": (
                        round(feat_data["production_mean"], 4)
                        if feat_data.get("production_mean") is not None
                        else "—"
                    ),
                    "Moy. baseline": (
                        round(feat_data["baseline_mean"], 4)
                        if feat_data.get("baseline_mean") is not None
                        else "—"
                    ),
                    "Z-score": (
                        round(feat_data["z_score"], 3)
                        if feat_data.get("z_score") is not None
                        else "—"
                    ),
                    "PSI": (
                        round(feat_data["psi"], 4)
                        if feat_data.get("psi") is not None
                        else "—"
                    ),
                    "N prod.": feat_data.get("production_count", 0),
                }
            )
        df_drift = pd.DataFrame(rows_drift)
        st.dataframe(df_drift, use_container_width=True, hide_index=True)
    elif not feat_baseline:
        st.info("Pas de baseline de features enregistrée pour ce modèle.")
    else:
        st.info("Aucune donnée de drift disponible.")

# --- Calibration des probabilités ---
st.divider()
st.markdown("#### 📏 Calibration")

try:
    calib = client.get_model_calibration(
        model_name=selected_model,
        start=start_iso,
        end=end_iso,
    )
except Exception as exc:
    st.info(f"Calibration non disponible : {exc}")
    calib = None

if calib:
    calib_status = calib.get("calibration_status", "insufficient_data")

    if calib_status == "insufficient_data":
        st.info(
            "Soumettez des observed_results pour activer la calibration "
            f"(échantillon actuel : {calib.get('sample_size', 0)} paires)."
        )
    else:
        STATUS_CALIB = {"ok": "🟢 OK", "overconfident": "🟡 Sur-confiant", "underconfident": "🔴 Sous-confiant"}
        label_status = STATUS_CALIB.get(calib_status, calib_status)

        cc1, cc2, cc3 = st.columns(3)
        brier = calib.get("brier_score")
        gap = calib.get("overconfidence_gap")
        cc1.metric("Brier score", f"{brier:.4f}" if brier is not None else "—")
        cc2.metric(
            "Gap confiance/précision",
            f"{gap:+.2%}" if gap is not None else "—",
        )
        cc3.metric("Statut", label_status)

        reliability = calib.get("reliability", [])
        if reliability:
            bins = [b["confidence_bin"] for b in reliability]
            pred_rates = [b["predicted_rate"] for b in reliability]
            obs_rates = [b["observed_rate"] for b in reliability]
            counts = [b["count"] for b in reliability]

            fig_cal = go.Figure()

            # Zone grisée entre courbe réelle et diagonale
            diag_vals = [(float(b.split("–")[0]) + float(b.split("–")[1])) / 2 for b in bins]
            fig_cal.add_trace(
                go.Scatter(
                    x=bins,
                    y=diag_vals,
                    name="Calibration parfaite",
                    line=dict(color="grey", dash="dot", width=1),
                    mode="lines",
                )
            )
            fig_cal.add_trace(
                go.Scatter(
                    x=bins + bins[::-1],
                    y=obs_rates + diag_vals[::-1],
                    fill="toself",
                    fillcolor="rgba(200,200,200,0.25)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            marker_sizes = [max(8, min(24, c // 5)) for c in counts]
            fig_cal.add_trace(
                go.Scatter(
                    x=bins,
                    y=obs_rates,
                    name="Taux observé",
                    mode="markers+lines",
                    marker=dict(size=marker_sizes, color="#2980b9"),
                    customdata=counts,
                    hovertemplate="Bucket : %{x}<br>Taux observé : %{y:.2%}<br>N : %{customdata}<extra></extra>",
                )
            )
            fig_cal.update_layout(
                title="Courbe de calibration (reliability diagram)",
                xaxis_title="Confiance prédite",
                yaxis_title="Taux observé",
                yaxis=dict(tickformat=".0%", range=[0, 1]),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig_cal, use_container_width=True)

        if calib_status == "overconfident":
            st.warning(
                "Envisagez `CalibratedClassifierCV(method='isotonic')` lors du prochain retrain "
                "pour corriger la sur-confiance du modèle."
            )

# --- Comparaison A/B & Shadow ---
if ab_comparison:
    st.divider()
    st.markdown("#### ⚖️ Comparaison A/B & Shadow")

    ab_versions = ab_comparison.get("versions", [])
    agreement = ab_comparison.get("shadow_agreement_rate", {})

    if ab_versions:
        df_ab = pd.DataFrame(
            [
                {
                    "Version": v["version"],
                    "Prédictions": v["total_predictions"],
                    "Shadow": v["shadow_predictions"],
                    "Taux d'erreur": f"{v['error_rate'] * 100:.1f} %",
                    "Latence moy.": (
                        f"{v.get('avg_response_time_ms')} ms"
                        if v.get("avg_response_time_ms")
                        else "—"
                    ),
                    "p95": (
                        f"{v.get('p95_response_time_ms')} ms"
                        if v.get("p95_response_time_ms")
                        else "—"
                    ),
                    "Accord shadow": (
                        f"{agreement.get(v['version'], 0) * 100:.1f} %"
                        if v["version"] in agreement
                        else "—"
                    ),
                }
                for v in ab_versions
            ]
        )
        st.dataframe(df_ab, use_container_width=True, hide_index=True)

        # Distribution des prédictions par version
        dist_data = []
        for v in ab_versions:
            dist = v.get("prediction_distribution", {})
            for label, count in dist.items():
                dist_data.append(
                    {
                        "version": v["version"],
                        "label": str(label),
                        "count": count,
                    }
                )

        if dist_data:
            df_dist = pd.DataFrame(dist_data)
            fig_dist = px.bar(
                df_dist,
                x="label",
                y="count",
                color="version",
                barmode="group",
                title="Distribution des prédictions par version et classe",
                labels={"label": "Classe prédite", "count": "Nombre", "version": "Version"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

# --- Erreurs récentes ---
if recent_errors:
    st.divider()
    with st.expander(f"⚠️ Dernières erreurs ({len(recent_errors)})", expanded=False):
        for i, err in enumerate(recent_errors, 1):
            st.markdown(f"`{i}.` {err}")
