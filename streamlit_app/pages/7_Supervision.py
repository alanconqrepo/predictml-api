"""
Tableau de bord de supervision des modèles ML.

Vue globale : état de santé de tous les modèles sur une plage calendaire.
Vue détaillée : zoom sur un modèle (série temporelle, drift, A/B testing).
"""

from datetime import date, datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.auth import get_client, require_auth
from utils.metrics_help import METRIC_HELP

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
# Boutons d'export
# ---------------------------------------------------------------------------
if models_data:
    _STATUS_SEVERITY = {
        "ok": 0,
        "no_data": 0,
        "no_baseline": 0,
        "insufficient_data": 0,
        "warning": 1,
        "critical": 2,
    }

    def _worst_drift(s1: str, s2: str) -> str:
        return s1 if _STATUS_SEVERITY.get(s1, 0) >= _STATUS_SEVERITY.get(s2, 0) else s2

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
            "accuracy_7d": "",
            "last_retrain": "",
            "coverage_pct": "",
        }
        for m in models_data
    ]
    _csv_bytes = pd.DataFrame(_csv_rows).to_csv(index=False).encode("utf-8")

    _alert_models = sorted(
        [m for m in models_data if m.get("health_status") in ("critical", "warning")],
        key=lambda m: (m.get("health_status") != "critical", m["model_name"]),
    )
    _ok_models = sorted(
        [m for m in models_data if m.get("health_status") == "ok"],
        key=lambda m: m["model_name"],
    )
    _md_lines = [
        f"# Rapport de supervision — {start_date} → {end_date}",
        "",
        "## Résumé global",
        "",
        f"- **Prédictions** : {gs.get('total_predictions', 0):,}",
        f"- **Taux d'erreur** : {gs.get('error_rate', 0) * 100:.1f} %",
        f"- **Latence moyenne** : {gs.get('avg_latency_ms') or '—'} ms",
        f"- **Modèles actifs** : {gs.get('active_models', 0)}",
        f"- **Alertes** : 🔴 {gs.get('models_critical', 0)} critique(s) · 🟡 {gs.get('models_warning', 0)} avertissement(s)",
        "",
    ]
    if _alert_models:
        _md_lines += ["## Modèles en alerte", ""]
        for _m in _alert_models:
            _icon_md = "🔴" if _m.get("health_status") == "critical" else "🟡"
            _md_lines += [
                f"### {_icon_md} {_m['model_name']}",
                "",
                f"- **Statut** : {_m.get('health_status', '—')}",
                f"- **Prédictions (période)** : {_m.get('total_predictions', 0):,}",
                f"- **Taux d'erreur** : {_m.get('error_rate', 0) * 100:.1f} %",
            ]
            if _m.get("p95_latency_ms") is not None:
                _md_lines.append(f"- **Latence p95** : {_m['p95_latency_ms']} ms")
            _md_lines += [
                f"- **Drift features** : {_m.get('feature_drift_status', '—')}",
                f"- **Drift performance** : {_m.get('performance_drift_status', '—')}",
                f"- **Drift sortie** : {_m.get('output_drift_status', '—')}",
            ]
            _actions = []
            if _m.get("error_rate", 0) >= 0.10:
                _actions.append("Taux d'erreur critique — analyser les logs et les données d'entrée.")
            elif _m.get("error_rate", 0) >= 0.05:
                _actions.append("Taux d'erreur élevé — surveiller l'évolution.")
            if _m.get("feature_drift_status") in ("warning", "critical"):
                _actions.append("Drift de features détecté — recalculer le baseline ou ré-entraîner.")
            if _m.get("performance_drift_status") in ("warning", "critical"):
                _actions.append("Drift de performance détecté — envisager un ré-entraînement.")
            if _m.get("output_drift_status") in ("warning", "critical"):
                _actions.append("Drift de sortie (label shift) détecté — vérifier la distribution des prédictions.")
            if _actions:
                _md_lines += ["", "**Actions recommandées :**"]
                _md_lines += [f"- {_a}" for _a in _actions]
            _md_lines.append("")
    if _ok_models:
        _md_lines += [
            "## Modèles en bonne santé",
            "",
            ", ".join(f"`{_m['model_name']}`" for _m in _ok_models),
            "",
        ]
    _md_bytes = "\n".join(_md_lines).encode("utf-8")

    _exp_col1, _exp_col2, _ = st.columns([2, 2, 4])
    _exp_col1.download_button(
        label="📥 Exporter CSV",
        data=_csv_bytes,
        file_name=f"supervision_{start_date}_{end_date}.csv",
        mime="text/csv",
        help="Télécharger les métriques de supervision au format CSV",
    )
    _exp_col2.download_button(
        label="📄 Exporter Markdown",
        data=_md_bytes,
        file_name=f"supervision_{start_date}_{end_date}.md",
        mime="text/markdown",
        help="Télécharger un résumé narratif des alertes et recommandations",
    )

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
    help=METRIC_HELP["taux_erreur"],
)
k3.metric(
    "Latence moy.",
    f"{gs.get('avg_latency_ms') or '—'} ms" if gs.get("avg_latency_ms") else "—",
    help=METRIC_HELP["latence_avg"],
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
            "Mode": ", ".join(sorted(set(v for v in m.get("deployment_modes", {}).values() if v)))
            or "—",
            "Prédictions": m["total_predictions"],
            "Shadow": m["shadow_predictions"],
            "Erreurs": f"{m['error_rate'] * 100:.1f} %",
            "Latence moy.": f"{m['avg_latency_ms'] or '—'} ms" if m.get("avg_latency_ms") else "—",
            "p95": f"{m['p95_latency_ms']} ms" if m.get("p95_latency_ms") else "—",
            "Drift features": _icon(m.get("feature_drift_status", "no_data")),
            "Drift perf.": _icon(m.get("performance_drift_status", "no_data")),
            "Drift sortie": _icon(m.get("output_drift_status", "no_data")),
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
fig_err.add_hline(
    y=10, line_dash="dash", line_color="#c0392b", annotation_text="Seuil critique 10%"
)
fig_err.update_layout(showlegend=False, coloraxis_showscale=False)
col_err.plotly_chart(fig_err, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3 — Détail par modèle
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🔎 Détail d'un modèle")

model_names = [m["model_name"] for m in models_data]
sup_search = st.text_input("Filtrer par nom", key="sup_model_search", placeholder="Rechercher un modèle…")
sup_filtered = [n for n in model_names if sup_search.lower() in n.lower()] if sup_search else model_names
selected_model = st.selectbox("Sélectionner un modèle", sup_filtered or model_names)

if not selected_model:
    st.stop()

# Store for cross-page navigation (read by Models and Retrain pages)
st.session_state["_nav_model"] = selected_model

# Quick links to act on the selected model
_lc1, _lc2, _ = st.columns([1.5, 1.5, 3])
with _lc1:
    st.page_link("pages/2_Models.py", label="🤖 Gérer dans Models", use_container_width=True)
with _lc2:
    st.page_link("pages/8_Retrain.py", label="🔄 Ré-entraîner", use_container_width=True)

with st.spinner(f"Chargement du détail pour {selected_model}…"):
    try:
        detail = client.get_monitoring_model(name=selected_model, start=start_iso, end=end_iso)
    except Exception as exc:
        st.error(f"Impossible de charger le détail : {exc}")
        st.stop()

per_version = detail.get("per_version_stats", [])
timeseries = detail.get("timeseries", [])
perf_by_day = detail.get("performance_by_day", [])
feature_drift = detail.get("feature_drift", {})
ab_comparison = detail.get("ab_comparison")
recent_errors = detail.get("recent_errors", [])

# Load alert thresholds from production (or first) version for chart overlays
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
                "Latence moy.": (
                    f"{v['avg_latency_ms'] or '—'} ms" if v.get("avg_latency_ms") else "—"
                ),
                "p50": f"{v['p50_latency_ms']} ms" if v.get("p50_latency_ms") else "—",
                "p95": f"{v['p95_latency_ms']} ms" if v.get("p95_latency_ms") else "—",
            }
            for v in per_version
        ]
    )
    st.dataframe(df_ver, use_container_width=True, hide_index=True)

# --- Seuils d'alerte ---
st.divider()
with st.expander("🔔 Seuils d'alerte", expanded=bool(_overlay_thresholds)):
    if not per_version:
        st.info("Aucune version disponible pour configurer les seuils.")
    else:
        _ver_options = [v["version"] for v in per_version]
        _default_ver = next(
            (v["version"] for v in per_version if v.get("deployment_mode") == "production"),
            _ver_options[0],
        )
        _sel_ver = st.selectbox(
            "Version à configurer",
            _ver_options,
            index=_ver_options.index(_default_ver),
            key="alert_version_select",
        )
        try:
            _full = client.get_model(selected_model, _sel_ver)
            _cur = _full.get("alert_thresholds") or {}
        except Exception:
            _cur = {}

        _err_val = _cur.get("error_rate_max")
        _acc_val = _cur.get("accuracy_min")
        _drift_val = _cur.get("drift_auto_alert") or False

        if _cur:
            _ci1, _ci2, _ci3 = st.columns(3)
            _ci1.metric(
                "Taux d'erreur max",
                f"{_err_val * 100:.1f} %" if _err_val is not None else "—",
            )
            _ci2.metric(
                "Accuracy min",
                f"{_acc_val:.0%}" if _acc_val is not None else "—",
            )
            _ci3.metric(
                "Alerte drift",
                "✅ Activée" if _drift_val else "⬜ Désactivée",
            )

        with st.form("alert_thresholds_form"):
            _col1, _col2 = st.columns(2)
            new_error_rate_pct = _col1.number_input(
                "Taux d'erreur max (%)",
                min_value=0.0,
                max_value=100.0,
                value=round(_err_val * 100, 2) if _err_val is not None else 10.0,
                step=0.5,
                help="Déclenche une alerte si le taux d'erreur dépasse ce seuil.",
            )
            new_accuracy_min = _col2.number_input(
                "Accuracy minimale",
                min_value=0.0,
                max_value=1.0,
                value=float(_acc_val) if _acc_val is not None else 0.80,
                step=0.01,
                format="%.2f",
                help="Déclenche une alerte si l'accuracy tombe sous ce seuil.",
            )
            new_drift_auto = st.checkbox(
                "Alerte automatique sur drift de features",
                value=bool(_drift_val),
            )
            _submitted = st.form_submit_button("💾 Enregistrer les seuils", type="primary")
            if _submitted:
                _payload = {
                    "alert_thresholds": {
                        "error_rate_max": new_error_rate_pct / 100,
                        "accuracy_min": new_accuracy_min,
                        "drift_auto_alert": new_drift_auto,
                    }
                }
                try:
                    client.update_model(selected_model, _sel_ver, _payload)
                    st.success(f"✅ Seuils mis à jour pour {selected_model} v{_sel_ver}.")
                    st.cache_data.clear()
                except Exception as _exc:
                    st.error(f"Erreur lors de la mise à jour : {_exc}")

# --- Circuit breaker auto-demotion ---
st.divider()
_prod_ver_for_policy = next(
    (v["version"] for v in per_version if v.get("deployment_mode") == "production"),
    per_version[0]["version"] if per_version else None,
)
if _prod_ver_for_policy:
    try:
        _policy_full = client.get_model(selected_model, _prod_ver_for_policy)
        _policy = _policy_full.get("promotion_policy") or {}
    except Exception:
        _policy = {}

    _auto_demote_on = _policy.get("auto_demote", False)
    with st.expander(
        f"⚡ Circuit breaker (auto-demotion) {'🔴 Activé' if _auto_demote_on else '⬜ Inactif'}",
        expanded=_auto_demote_on,
    ):
        if not _auto_demote_on:
            st.info(
                "L'auto-demotion est désactivée. Configurez-la via "
                "`PATCH /models/{name}/policy` avec `auto_demote: true`."
            )
        else:
            _cd1, _cd2, _cd3 = st.columns(3)
            _cd1.metric(
                "Niveau de drift déclencheur",
                _policy.get("demote_on_drift", "critical").capitalize(),
            )
            _acc_thr = _policy.get("demote_on_accuracy_below")
            _cd2.metric(
                "Accuracy minimale",
                f"{_acc_thr:.0%}" if _acc_thr is not None else "—",
            )
            _cd3.metric(
                "Cooldown",
                f"{_policy.get('demote_cooldown_hours', 24)} h",
            )

        # Dernière auto-demotion dans l'historique
        try:
            _history_data = client.get_model_history(selected_model)
            _history_entries = _history_data.get("history", [])
            _last_demote = next(
                (
                    e
                    for e in _history_entries
                    if e.get("action") == "auto_demote"
                ),
                None,
            )
        except Exception:
            _last_demote = None

        if _last_demote:
            _demote_ts = _last_demote.get("timestamp", "")[:16].replace("T", " ")
            _demote_reason = (_last_demote.get("snapshot") or {}).get(
                "auto_demote_reason", "—"
            )
            st.error(
                f"🔴 **Auto-démis le {_demote_ts} UTC** — {_demote_reason}",
                icon="🚨",
            )
        elif _auto_demote_on:
            st.success("🟢 Aucune auto-demotion récente détectée.")

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
    if _err_max is not None:
        fig_err_ts.add_hline(
            y=_err_max,
            line_dash="dash",
            line_color="#c0392b",
            annotation_text=f"Seuil {_err_max * 100:.1f}%",
        )
    else:
        fig_err_ts.add_hline(y=0.05, line_dash="dash", line_color="#e67e22", annotation_text="5%")
        fig_err_ts.add_hline(y=0.10, line_dash="dash", line_color="#c0392b", annotation_text="10%")
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
            if _acc_min is not None:
                fig_perf.add_hline(
                    y=_acc_min,
                    line_dash="dash",
                    line_color="#c0392b",
                    annotation_text=f"Min {_acc_min:.0%}",
                    annotation_position="bottom right",
                )
            st.plotly_chart(fig_perf, use_container_width=True)

            # Indicateur de drift
            mid = len(df_perf) // 2
            if mid > 0:
                avg1 = df_perf["accuracy"].iloc[:mid].mean()
                avg2 = df_perf["accuracy"].iloc[mid:].mean()
                drop = avg1 - avg2
                drift_label = (
                    "🔴 Critique"
                    if drop >= 0.10
                    else "🟡 Attention" if drop >= 0.05 else "🟢 Stable"
                )
                st.metric(
                    "Tendance performance",
                    drift_label,
                    delta=f"{-drop:.1%} (1re vs 2e moitié)",
                    delta_color="inverse",
                    help=METRIC_HELP["tendance_performance"],
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
                        round(feat_data["psi"], 4) if feat_data.get("psi") is not None else "—"
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

# --- Tendance de confiance ---
st.divider()
st.markdown("#### 📉 Tendance de confiance")

period_days = max(1, (end_date - start_date).days)
try:
    conf_trend = client.get_confidence_trend(selected_model, days=period_days)
except Exception as exc:
    conf_trend = None
    st.info(f"Tendance de confiance non disponible : {exc}")

if conf_trend is not None:
    trend_data = conf_trend.get("trend", [])
    overall = conf_trend.get("overall", {})

    if not trend_data:
        st.info("Ce modèle ne retourne pas de probabilités — tendance de confiance indisponible.")
    else:
        df_conf = pd.DataFrame(trend_data)
        df_conf["date"] = pd.to_datetime(df_conf["date"])
        df_conf = df_conf.sort_values("date")

        threshold_val = 0.5  # default; endpoint uses model's confidence_threshold
        mean_conf = overall.get("mean_confidence", 0.0)
        low_rate = overall.get("low_confidence_rate", 0.0)

        # Delta: dernière semaine vs semaine précédente
        mid = len(df_conf) // 2
        delta_str = None
        if mid > 0:
            avg_first = df_conf["mean_confidence"].iloc[:mid].mean()
            avg_second = df_conf["mean_confidence"].iloc[mid:].mean()
            delta_val = avg_second - avg_first
            delta_str = f"{delta_val:+.1%} (1re vs 2e moitié)"

        ct1, ct2, ct3 = st.columns(3)
        ct1.metric(
            "Confiance moyenne",
            f"{mean_conf:.2%}",
            delta=delta_str,
            delta_color="normal",
            help=METRIC_HELP["confiance_moyenne"],
        )
        ct2.metric("P25", f"{overall.get('p25_confidence', 0):.2%}", help=METRIC_HELP["p25_confiance"])
        ct3.metric("P75", f"{overall.get('p75_confidence', 0):.2%}", help=METRIC_HELP["p75_confiance"])

        # Graphique : ligne mean + bande IQR + seuil
        fig_conf = go.Figure()

        # Bande IQR semi-transparente
        fig_conf.add_trace(
            go.Scatter(
                x=pd.concat([df_conf["date"], df_conf["date"][::-1]]),
                y=pd.concat([df_conf["p75"], df_conf["p25"][::-1]]),
                fill="toself",
                fillcolor="rgba(41, 128, 185, 0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="IQR (P25–P75)",
                hoverinfo="skip",
            )
        )

        # Ligne mean_confidence
        fig_conf.add_trace(
            go.Scatter(
                x=df_conf["date"],
                y=df_conf["mean_confidence"],
                name="Confiance moyenne",
                mode="lines+markers",
                line=dict(color="#2980b9", width=2),
                marker=dict(size=5),
            )
        )

        # Seuil de confiance en pointillé
        fig_conf.add_hline(
            y=threshold_val,
            line_dash="dot",
            line_color="#e74c3c",
            annotation_text=f"Seuil {threshold_val:.0%}",
            annotation_position="bottom right",
        )

        fig_conf.update_layout(
            xaxis_title="Date",
            yaxis_title="Confiance",
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig_conf, use_container_width=True)

        if low_rate > 0.15:
            st.warning(
                f"{low_rate * 100:.0f} % des prédictions sont sous le seuil de confiance "
                f"— signal possible de drift d'entrée."
            )

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
        STATUS_CALIB = {
            "ok": "🟢 OK",
            "overconfident": "🟡 Sur-confiant",
            "underconfident": "🔴 Sous-confiant",
        }
        label_status = STATUS_CALIB.get(calib_status, calib_status)

        cc1, cc2, cc3 = st.columns(3)
        brier = calib.get("brier_score")
        gap = calib.get("overconfidence_gap")
        cc1.metric("Brier score", f"{brier:.4f}" if brier is not None else "—", help=METRIC_HELP["brier_score"])
        cc2.metric(
            "Gap confiance/précision",
            f"{gap:+.2%}" if gap is not None else "—",
            help=METRIC_HELP["gap_confiance"],
        )
        cc3.metric("Statut", label_status, help=METRIC_HELP["statut_calibration"])

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

# --- Prédictions anomales ---
st.divider()
st.markdown("#### 🚨 Prédictions anomales")

_anom_col1, _anom_col2 = st.columns([1, 2])
_anom_days = _anom_col1.number_input(
    "Fenêtre (jours)",
    min_value=1,
    max_value=90,
    value=min(max(period_days, 1), 90),
    step=1,
    key="anom_days",
)
_anom_z = _anom_col2.slider(
    "Seuil z-score",
    min_value=1.0,
    max_value=6.0,
    value=3.0,
    step=0.1,
    format="%.1f",
    key="anom_z_threshold",
    help="Features dont |z| ≥ seuil sont considérées aberrantes.",
)

try:
    _anom_data = client.get_predictions_anomalies(
        model_name=selected_model,
        days=int(_anom_days),
        z_threshold=float(_anom_z),
        limit=200,
    )
except Exception as _exc:
    _anom_data = None
    st.info(f"Prédictions anomales non disponibles : {_exc}")

if _anom_data is not None:
    _anom_error = _anom_data.get("error")
    if _anom_error == "no_baseline":
        st.info(
            "Pas de baseline de features enregistrée pour ce modèle — "
            "impossible de calculer les z-scores. Fournissez `feature_baseline` à l'upload."
        )
    else:
        _total = _anom_data.get("total_checked", 0)
        _count = _anom_data.get("anomalous_count", 0)
        _rate = _anom_data.get("anomaly_rate", 0.0)
        _preds = _anom_data.get("predictions", [])

        _mc1, _mc2, _mc3 = st.columns(3)
        _mc1.metric("Prédictions analysées", _total)
        _mc2.metric("Prédictions anomales", _count)
        _mc3.metric(
            "Taux d'anomalie",
            f"{_rate:.1%}",
            help="anomalous_count / total_checked",
        )

        if not _preds:
            st.success(
                f"Aucune prédiction anomale détectée (z ≥ {_anom_z:.1f}) sur {_total} prédiction(s)."
            )
        else:
            _rows_anom = []
            for _p in _preds:
                _feats = _p.get("anomalous_features", {})
                _worst_z = max((_f["z_score"] for _f in _feats.values()), default=0.0)
                _feat_names = ", ".join(
                    f"{_fn} (z={_fd['z_score']:.2f})" for _fn, _fd in _feats.items()
                )
                _rows_anom.append(
                    {
                        "ID": _p["prediction_id"],
                        "Timestamp": _p["timestamp"],
                        "Résultat": str(_p.get("prediction_result", "")),
                        "Confiance": (
                            f"{_p['max_confidence']:.2%}"
                            if _p.get("max_confidence") is not None
                            else "—"
                        ),
                        "Z-score max": round(_worst_z, 2),
                        "Features aberrantes": _feat_names,
                    }
                )

            _df_anom = pd.DataFrame(_rows_anom).sort_values("Z-score max", ascending=False)
            st.dataframe(_df_anom, use_container_width=True, hide_index=True)

            with st.expander("Détail des features par prédiction", expanded=False):
                for _p in sorted(
                    _preds,
                    key=lambda x: -max(f["z_score"] for f in x["anomalous_features"].values()),
                ):
                    _feats = _p.get("anomalous_features", {})
                    st.markdown(
                        f"**Prédiction #{_p['prediction_id']}** — {_p['timestamp'][:19]}  "
                        f"résultat : `{_p.get('prediction_result')}`"
                    )
                    _feat_rows = [
                        {
                            "Feature": _fn,
                            "Valeur": round(_fd["value"], 4),
                            "Z-score": round(_fd["z_score"], 4),
                            "Baseline μ": round(_fd["baseline_mean"], 4),
                            "Baseline σ": round(_fd["baseline_std"], 4),
                        }
                        for _fn, _fd in sorted(_feats.items(), key=lambda x: -x[1]["z_score"])
                    ]
                    st.dataframe(pd.DataFrame(_feat_rows), use_container_width=True, hide_index=True)
                    st.divider()
