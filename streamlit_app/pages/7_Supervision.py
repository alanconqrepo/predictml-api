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

# Labels de mode cohérents avec les pages Modèles et A/B Testing
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
# Préparation des exports (calcul uniquement, boutons dans l'onglet global)
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
        f"# Rapport de supervision — {start_date} → {end_date}", "",
        "## Résumé global", "",
        f"- **Prédictions production** : {gs.get('total_predictions', 0):,}",
        f"- **Prédictions shadow** : {gs.get('total_shadow', 0):,}",
        f"- **Taux d'erreur exécution** : {gs.get('error_rate', 0) * 100:.1f} % "
        f"(erreurs serveur, hors qualité ML)",
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
                f"### {_icon_md} {_m['model_name']}", "",
                f"- **Statut** : {_m.get('health_status', '—')}",
                f"- **Prédictions** : {_m.get('total_predictions', 0):,}",
                f"- **Taux d'erreur** : {_m.get('error_rate', 0) * 100:.1f} %",
                f"- **Drift features** : {_m.get('feature_drift_status', '—')}",
                f"- **Drift performance** : {_m.get('performance_drift_status', '—')}",
                "",
            ]
    _md_bytes = "\n".join(_md_lines).encode("utf-8")

# ---------------------------------------------------------------------------
# Helpers partagés
# ---------------------------------------------------------------------------
STATUS_ICON = {
    "ok": "🟢", "warning": "🟡", "critical": "🔴",
    "no_data": "⚪", "no_baseline": "⚪", "insufficient_data": "⚪",
}

def _icon(status: str) -> str:
    return STATUS_ICON.get(status, "⚪") + " " + status

# ---------------------------------------------------------------------------
# KPIs globaux — toujours visibles
# ---------------------------------------------------------------------------
st.divider()
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric(
    "Préd. production",
    f"{gs.get('total_predictions', 0):,}",
    help=METRIC_HELP["predictions_prod"],
)
k2.metric(
    "Préd. shadow",
    f"{gs.get('total_shadow', 0):,}",
    help=METRIC_HELP["predictions_shadow"],
)
k3.metric(
    "Taux d'erreur",
    f"{gs.get('error_rate', 0) * 100:.1f} %",
    help=METRIC_HELP["taux_erreur"],
)
k4.metric(
    "Latence moy.",
    f"{gs.get('avg_latency_ms') or '—'} ms" if gs.get("avg_latency_ms") else "—",
    help=METRIC_HELP["latence_avg"],
)
k5.metric(
    "Modèles actifs",
    gs.get("active_models", 0),
    help=METRIC_HELP["modeles_actifs"],
)
_alerts = gs.get("models_critical", 0) + gs.get("models_warning", 0)
k6.metric(
    "Alertes",
    f"🔴 {gs.get('models_critical', 0)} · 🟡 {gs.get('models_warning', 0)}",
    delta=f"{_alerts} modèle(s) à surveiller" if _alerts else "Tout est OK",
    delta_color="inverse" if _alerts else "normal",
    help=METRIC_HELP["alertes_sante"],
)

if not models_data:
    st.info("Aucune prédiction dans la période sélectionnée.")
    st.stop()

# ---------------------------------------------------------------------------
# Navigation principale : deux onglets
# ---------------------------------------------------------------------------
st.divider()
_tab_global, _tab_detail = st.tabs(["🌍 Vue globale", "🔎 Zoom sur un modèle"])


# ═══════════════════════════════════════════════════════════════════════════
# ONGLET 1 — Vue globale
# ═══════════════════════════════════════════════════════════════════════════
with _tab_global:

    # ── Exports ──────────────────────────────────────────────────────────
    if _csv_bytes:
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

    # ── Tableau de santé ─────────────────────────────────────────────────
    st.subheader("🏥 Santé par modèle")
    rows_table = []
    for m in models_data:
        rows_table.append({
            "Modèle": m["model_name"],
            "Versions": ", ".join(m.get("versions", [])),
            "Mode": ", ".join(sorted(set(
                _MODE_LABEL.get(v, "⚪ —") for v in m.get("deployment_modes", {}).values() if v
            ))) or "—",
            "Prédictions": m["total_predictions"],
            "Shadow": m["shadow_predictions"],
            "Erreurs": f"{m['error_rate'] * 100:.1f} %",
            "Latence moy.": f"{m['avg_latency_ms'] or '—'} ms" if m.get("avg_latency_ms") else "—",
            "p95": f"{m['p95_latency_ms']} ms" if m.get("p95_latency_ms") else "—",
            "Drift features": _icon(m.get("feature_drift_status", "no_data")),
            "Drift perf.": _icon(m.get("performance_drift_status", "no_data")),
            "Drift sortie": _icon(m.get("output_drift_status", "no_data")),
            "Statut": _icon(m.get("health_status", "no_data")),
        })
    df_health = pd.DataFrame(rows_table)
    st.dataframe(
        df_health,
        width='stretch',
        hide_index=True,
        column_config={
            "Modèle": st.column_config.TextColumn("Modèle", help="Nom du modèle ML déployé."),
            "Versions": st.column_config.TextColumn("Versions", help="Versions actives (is_active=True)."),
            "Mode": st.column_config.TextColumn(
                "Mode",
                help=(
                    "Mode(s) de déploiement des versions actives :\n"
                    "🟢 Production · 🟠 A/B · 🟣 Shadow · ⚪ — (aucun routage)"
                ),
            ),
            "Prédictions": st.column_config.NumberColumn(
                "Prédictions",
                help=(
                    "Prédictions production retournées aux clients (is_shadow=False) "
                    "sur la période sélectionnée."
                ),
            ),
            "Shadow": st.column_config.NumberColumn(
                "Shadow",
                help=(
                    "Prédictions silencieuses (is_shadow=True) calculées en arrière-plan "
                    "et non retournées au client.\n"
                    "Utilisées pour comparer de nouvelles versions sans impacter le trafic réel."
                ),
            ),
            "Erreurs": st.column_config.TextColumn(
                "Erreurs",
                help=(
                    "Taux d'erreurs d'exécution : COUNT(status ≠ 'success') / COUNT(*)\n\n"
                    "⚠️ Erreurs serveur uniquement (exception, modèle non chargé, timeout…). "
                    "Ce n'est PAS un indicateur de qualité ML.\n\n"
                    "🟡 Warning : ≥ 5 %\n"
                    "🔴 Critical : ≥ 10 %"
                ),
            ),
            "Latence moy.": st.column_config.TextColumn(
                "Latence moy.",
                help=(
                    "Temps de réponse moyen (ms) calculé sur les requêtes réussies "
                    "(status = 'success') sur la période."
                ),
            ),
            "p95": st.column_config.TextColumn(
                "p95",
                help=(
                    "95e percentile de latence : 95 % des requêtes réussies sont traitées "
                    "en moins de ce temps. Indicateur clé pour détecter les pics de lenteur."
                ),
            ),
            "Drift features": st.column_config.TextColumn(
                "Drift features",
                help=(
                    "Détection de dérive sur chaque feature numérique. "
                    "Statut = pire parmi 3 métriques :\n\n"
                    "• Z-score = |moy_prod − moy_baseline| / σ_baseline\n"
                    "  🟡 Warning : ≥ 2  ·  🔴 Critical : ≥ 3\n\n"
                    "• PSI (Population Stability Index) sur 10 bins normaux\n"
                    "  🟡 Warning : ≥ 0.1  ·  🔴 Critical : ≥ 0.2\n\n"
                    "• Null rate : écart du taux de valeurs manquantes vs baseline\n"
                    "  🟡 Warning : écart ≥ 5 pts  ·  🔴 Critical : écart ≥ 15 pts ou null > 30 %\n\n"
                    "⚪ no_baseline = aucune baseline stockée pour ce modèle.\n"
                    "⚪ insufficient_data = moins de 10 prédictions sur la période."
                ),
            ),
            "Drift perf.": st.column_config.TextColumn(
                "Drift perf.",
                help=(
                    "Détection de dégradation de performance dans le temps.\n\n"
                    "Méthode : compare la performance moyenne entre la 1ère moitié "
                    "et la 2ème moitié de la période sélectionnée.\n"
                    "• Classification : utilise l'accuracy (résultats observés liés requis)\n"
                    "• Régression : utilise le MAE (une hausse = dégradation)\n\n"
                    "🟡 Warning : baisse ≥ 5 points\n"
                    "🔴 Critical : baisse ≥ 10 points\n\n"
                    "⚪ no_data = moins de 4 jours de données ou aucun résultat observé lié."
                ),
            ),
            "Drift sortie": st.column_config.TextColumn(
                "Drift sortie",
                help=(
                    "Dérive de la distribution des valeurs prédites par le modèle.\n\n"
                    "Méthode : PSI (Population Stability Index) entre la distribution "
                    "de prédictions actuelle et la baseline :\n"
                    "• Classification : distribution des classes prédites\n"
                    "• Régression : distribution via quartiles d'entraînement\n\n"
                    "🟡 Warning : PSI ≥ 0.1\n"
                    "🔴 Critical : PSI ≥ 0.2\n\n"
                    "⚪ no_baseline = aucun label_distribution stocké pour ce modèle."
                ),
            ),
            "Statut": st.column_config.TextColumn(
                "Statut",
                help=(
                    "Statut de santé global = pire des 4 indicateurs :\n"
                    "① Taux d'erreur exécution\n"
                    "② Drift features (Z-score + PSI + null rate)\n"
                    "③ Drift performance (baisse accuracy/MAE)\n"
                    "④ Drift sortie (PSI distribution des prédictions)\n\n"
                    "🟢 ok · 🟡 warning · 🔴 critical\n"
                    "⚪ no_data / no_baseline / insufficient_data = données insuffisantes "
                    "(ne dégrade pas le statut global)"
                ),
            ),
        },
    )

    # ── Fetch timeseries par modèle (pour les graphiques d'évolution) ────
    _ts_by_model: dict[str, list] = {}
    with st.spinner("Chargement des séries temporelles…"):
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

    # — Camembert : répartition des prédictions production ——————————————
    fig_pie = px.pie(
        df_models,
        values="total_predictions",
        names="model_name",
        title="Répartition des prédictions production",
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pie.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{value:,} prédictions<br>%{percent}<extra></extra>",
    )
    fig_pie.update_layout(showlegend=False, margin=dict(t=50, b=10, l=10, r=10))
    col_pie.plotly_chart(fig_pie, use_container_width=True)

    # — Évolution du taux d'erreur ——————————————————————————————————————
    if not df_ts.empty:
        fig_err_ts = px.line(
            df_ts,
            x="date", y="error_rate_pct",
            color="model_name",
            title="Évolution du taux d'erreur d'exécution (%)",
            labels={"date": "Date", "error_rate_pct": "Erreurs (%)", "model_name": "Modèle"},
            markers=True,
        )
        fig_err_ts.add_hline(
            y=5, line_dash="dash", line_color="#e67e22",
            annotation_text="Warning 5 %", annotation_position="top right",
        )
        fig_err_ts.add_hline(
            y=10, line_dash="dash", line_color="#c0392b",
            annotation_text="Critique 10 %", annotation_position="top right",
        )
        fig_err_ts.update_layout(
            legend_title_text="Modèle",
            yaxis_rangemode="tozero",
            yaxis_title="Erreurs (%)",
        )
    else:
        fig_err_ts = go.Figure()
        fig_err_ts.update_layout(
            title="Évolution du taux d'erreur d'exécution (%)",
            annotations=[dict(
                text="Pas de données temporelles disponibles",
                showarrow=False, xref="paper", yref="paper",
                x=0.5, y=0.5, font_size=14, font_color="#888",
            )],
        )
    col_err_ts.plotly_chart(fig_err_ts, use_container_width=True)

    # — Évolution de la latence moyenne ————————————————————————————————
    df_lat = df_ts[df_ts["avg_latency_ms"].notna()].copy() if not df_ts.empty else df_ts
    if not df_lat.empty:
        fig_lat = px.line(
            df_lat,
            x="date", y="avg_latency_ms",
            color="model_name",
            title="Évolution de la latence moyenne de réponse (ms)",
            labels={
                "date": "Date",
                "avg_latency_ms": "Latence moy. (ms)",
                "model_name": "Modèle",
            },
            markers=True,
        )
        fig_lat.update_layout(
            legend_title_text="Modèle",
            yaxis_rangemode="tozero",
            yaxis_title="Latence moy. (ms)",
        )
    else:
        fig_lat = go.Figure()
        fig_lat.update_layout(
            title="Évolution de la latence moyenne de réponse (ms)",
            annotations=[dict(
                text="Pas de données de latence disponibles",
                showarrow=False, xref="paper", yref="paper",
                x=0.5, y=0.5, font_size=14, font_color="#888",
            )],
        )
    st.plotly_chart(fig_lat, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# ONGLET 2 — Zoom sur un modèle
# ═══════════════════════════════════════════════════════════════════════════
with _tab_detail:

    # ── Sélecteur — recherche + modèle sur la même ligne ─────────────────
    model_names = [m["model_name"] for m in models_data]
    _det_c1, _det_c2 = st.columns([1, 2])
    sup_search = _det_c1.text_input("Filtrer par nom", key="sup_model_search", placeholder="Rechercher un modèle…")
    sup_filtered = [n for n in model_names if sup_search.lower() in n.lower()] if sup_search else model_names
    selected_model = _det_c2.selectbox("Sélectionner un modèle", sup_filtered or model_names)

    if not selected_model:
        st.stop()

    st.session_state["_nav_model"] = selected_model

    _lc1, _lc2, _ = st.columns([1.5, 1.5, 3])
    with _lc1:
        st.page_link("pages/2_Models.py",  label="🤖 Gérer dans Models",  width='stretch')
    with _lc2:
        st.page_link("pages/8_Retrain.py", label="🔄 Ré-entraîner",       width='stretch')

    # ── Chargement du détail ──────────────────────────────────────────────
    with st.spinner(f"Chargement du détail pour {selected_model}…"):
        try:
            detail = client.get_monitoring_model(name=selected_model, start=start_iso, end=end_iso)
        except Exception as exc:
            st.error(f"Impossible de charger le détail : {exc}")
            st.stop()

    per_version  = detail.get("per_version_stats", [])
    timeseries   = detail.get("timeseries", [])
    perf_by_day  = detail.get("performance_by_day", [])
    feature_drift = detail.get("feature_drift", {})
    ab_comparison = detail.get("ab_comparison")
    recent_errors = detail.get("recent_errors", [])

    # Seuils d'alerte pour les overlays graphiques
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

    # État du drift pour pilote d'expansion
    _feat_drift_status = feature_drift.get("drift_summary", "no_data")
    _has_drift_alert   = _feat_drift_status in ("warning", "critical")

    # ────────────────────────────────────────────────────────────────────
    # EXPANDER 1 — Versions & Trafic
    # ────────────────────────────────────────────────────────────────────
    with st.expander("📋 Versions & Trafic", expanded=True):
        if per_version:
            df_ver = pd.DataFrame([{
                "Version": v["version"],
                "Mode": _MODE_LABEL.get(v.get("deployment_mode") or "", "⚪ —"),
                "Poids trafic": f"{v['traffic_weight']:.0%}" if v.get("traffic_weight") else "—",
                "Prédictions": v["total_predictions"],
                "Shadow": v["shadow_predictions"],
                "Erreurs": f"{v['error_rate'] * 100:.1f} %",
                "Latence moy.": f"{v['avg_latency_ms'] or '—'} ms" if v.get("avg_latency_ms") else "—",
                "p50": f"{v['p50_latency_ms']} ms" if v.get("p50_latency_ms") else "—",
                "p95": f"{v['p95_latency_ms']} ms" if v.get("p95_latency_ms") else "—",
            } for v in per_version])
            st.dataframe(
                df_ver, width='stretch', hide_index=True,
                column_config={
                    "Version": st.column_config.TextColumn("Version", help="Numéro de version."),
                    "Mode": st.column_config.TextColumn("Mode", help="production · ab_test · shadow."),
                    "Poids trafic": st.column_config.TextColumn("Poids trafic", help="Part du trafic en mode A/B."),
                    "Prédictions": st.column_config.NumberColumn("Prédictions", help="Prédictions production sur la période."),
                    "Shadow": st.column_config.NumberColumn("Shadow", help="Prédictions shadow sur la période."),
                    "Erreurs": st.column_config.TextColumn("Erreurs", help="Taux d'erreur de cette version."),
                    "Latence moy.": st.column_config.TextColumn("Latence moy.", help="Temps de réponse moyen (ms)."),
                    "p50": st.column_config.TextColumn("p50", help="Médiane du temps de réponse."),
                    "p95": st.column_config.TextColumn("p95", help="95e percentile — indicateur SLA."),
                },
            )
        else:
            st.info("Aucune version active sur la période.")

    # ────────────────────────────────────────────────────────────────────
    # EXPANDER 2 — Activité & Latence
    # ────────────────────────────────────────────────────────────────────
    with st.expander("📈 Activité & Latence", expanded=False):
        if timeseries:
            df_ts = pd.DataFrame(timeseries)
            df_ts["date"] = pd.to_datetime(df_ts["date"])

            col_ts1, col_ts2 = st.columns(2)

            fig_vol_ts = go.Figure()
            fig_vol_ts.add_trace(go.Scatter(
                x=df_ts["date"], y=df_ts["total_predictions"],
                name="Prédictions", fill="tozeroy", line=dict(color="#2980b9"),
            ))
            fig_vol_ts.add_trace(go.Scatter(
                x=df_ts["date"], y=df_ts["error_count"],
                name="Erreurs", fill="tozeroy", line=dict(color="#c0392b"),
            ))
            fig_vol_ts.update_layout(
                title="Prédictions & erreurs par jour",
                xaxis_title="Date", yaxis_title="Nombre",
                hovermode="x unified", legend=dict(orientation="h", y=1.02),
            )
            col_ts1.plotly_chart(fig_vol_ts, width='stretch')

            if df_ts["avg_latency_ms"].notna().any():
                fig_lat = go.Figure()
                for col_name, label, color in [
                    ("avg_latency_ms", "Moyenne", "#3498db"),
                    ("p50_latency_ms", "p50", "#27ae60"),
                    ("p95_latency_ms", "p95", "#e67e22"),
                ]:
                    if col_name in df_ts and df_ts[col_name].notna().any():
                        fig_lat.add_trace(go.Scatter(
                            x=df_ts["date"], y=df_ts[col_name],
                            name=label, line=dict(color=color),
                        ))
                fig_lat.update_layout(
                    title="Latence par jour (ms)",
                    xaxis_title="Date", yaxis_title="ms",
                    hovermode="x unified", legend=dict(orientation="h", y=1.02),
                )
                col_ts2.plotly_chart(fig_lat, width='stretch')

            fig_err_ts = px.area(
                df_ts, x="date", y="error_rate",
                title="Taux d'erreur par jour",
                labels={"date": "Date", "error_rate": "Taux d'erreur"},
                color_discrete_sequence=["#e74c3c"],
            )
            if _err_max is not None:
                fig_err_ts.add_hline(y=_err_max, line_dash="dash", line_color="#c0392b",
                                     annotation_text=f"Seuil {_err_max * 100:.1f}%")
            else:
                fig_err_ts.add_hline(y=0.05, line_dash="dash", line_color="#e67e22", annotation_text="5%")
                fig_err_ts.add_hline(y=0.10, line_dash="dash", line_color="#c0392b", annotation_text="10%")
            fig_err_ts.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig_err_ts, width='stretch')
        else:
            st.info("Aucune série temporelle disponible sur la période.")

    # ────────────────────────────────────────────────────────────────────
    # EXPANDER 3 — Drift & Anomalies
    # ────────────────────────────────────────────────────────────────────
    _drift_icon = STATUS_ICON.get(_feat_drift_status, "⚪")
    with st.expander(
        f"🌡️ Drift & Anomalies  {_drift_icon} {_feat_drift_status}",
        expanded=False,
    ):
        # Drift de performance + drift features côte à côte
        col_perf, col_feat = st.columns(2)

        with col_perf:
            st.markdown("**📈 Drift de performance**")
            if perf_by_day:
                df_perf = pd.DataFrame(perf_by_day)
                df_perf["date"] = pd.to_datetime(df_perf["date"])
                df_perf = df_perf[df_perf["matched_count"] > 0].sort_values("date")
                if not df_perf.empty:
                    # Detect regression via mae (available even on older API versions)
                    _is_regression = (
                        "mae" in df_perf.columns
                        and df_perf["mae"].notna().any()
                    )
                    # Choose best metric: R² if available, MAE as fallback for regression
                    _has_r2 = (
                        _is_regression
                        and "r2" in df_perf.columns
                        and df_perf["r2"].notna().any()
                    )
                    if _has_r2:
                        _metric_col   = "r2"
                        _metric_label = "R² / jour"
                        _y_label      = "R²"
                        _tick_fmt     = ".3f"
                        _delta_fmt    = lambda d: f"{-d:+.3f} (1re vs 2e moitié)"  # noqa: E731
                        _drop_invert  = False   # drop > 0 = R² fell = bad
                    elif _is_regression:
                        # MAE fallback: lower is better so invert sign for trend
                        _metric_col   = "mae"
                        _metric_label = "MAE / jour"
                        _y_label      = "MAE"
                        _tick_fmt     = ".3f"
                        _delta_fmt    = lambda d: f"{-d:+.3f} (1re vs 2e moitié)"  # noqa: E731
                        _drop_invert  = True    # drop > 0 = MAE increased = bad (same logic)
                    else:
                        _metric_col   = "accuracy"
                        _metric_label = "Accuracy / jour"
                        _y_label      = "Accuracy"
                        _tick_fmt     = ".0%"
                        _delta_fmt    = lambda d: f"{-d:.1%} (1re vs 2e moitié)"  # noqa: E731
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
                        name="Moyenne 3j", line=dict(color="#2980b9", width=2),
                    ))
                    if not _is_regression and _acc_min is not None:
                        fig_perf.add_hline(y=_acc_min, line_dash="dash", line_color="#c0392b",
                                           annotation_text=f"Min {_acc_min:.0%}",
                                           annotation_position="bottom right")
                    fig_perf.update_layout(
                        xaxis_title="Date", yaxis_title=_y_label,
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
                        # For MAE: increase = bad, so invert sign to match "drop" semantics
                        drop = -raw_drop if _drop_invert else raw_drop
                        _status = (
                            "🔴 Critique" if drop >= 0.10
                            else "🟡 Attention" if drop >= 0.05
                            else "🟢 Stable"
                        )
                        st.metric(
                            "Tendance performance",
                            _status,
                            delta=_delta_fmt(raw_drop), delta_color="inverse",
                            help=METRIC_HELP["tendance_performance"],
                        )
                else:
                    st.info("Aucune observation liée dans la période.")
            else:
                st.info("Aucune donnée de performance disponible.")

        with col_feat:
            st.markdown("**🌡️ Drift des features**")
            feat_summary  = feature_drift.get("drift_summary", "no_data")
            feat_baseline = feature_drift.get("baseline_available", False)
            feat_analyzed = feature_drift.get("predictions_analyzed", 0)
            st.markdown(f"**Statut global** : {STATUS_ICON.get(feat_summary, '⚪')} `{feat_summary}`")
            st.caption(f"{feat_analyzed} prédictions analysées")
            features_dict = feature_drift.get("features", {})
            if features_dict and feat_baseline:
                rows_drift = [{
                    "Feature": fn,
                    "Statut": _icon(fd.get("drift_status", "no_data")),
                    "Moy. prod.": round(fd["production_mean"], 4) if fd.get("production_mean") is not None else "—",
                    "Moy. baseline": round(fd["baseline_mean"], 4) if fd.get("baseline_mean") is not None else "—",
                    "Z-score": round(fd["z_score"], 3) if fd.get("z_score") is not None else "—",
                    "PSI": round(fd["psi"], 4) if fd.get("psi") is not None else "—",
                    "N prod.": fd.get("production_count", 0),
                } for fn, fd in features_dict.items()]
                st.dataframe(
                    pd.DataFrame(rows_drift), width='stretch', hide_index=True,
                    column_config={
                        "Feature": st.column_config.TextColumn(
                            "Feature",
                            help=(
                                "Nom de la feature d'entrée du modèle.\n\n"
                                "Les lignes sont triées par ordre alphabétique. "
                                "Triez sur « Statut » pour faire remonter les features "
                                "en alerte."
                            ),
                        ),
                        "Statut": st.column_config.TextColumn(
                            "Statut",
                            help=(
                                "Statut de drift de la feature — pire parmi 3 indicateurs :\n\n"
                                "• Z-score (déplacement de la moyenne)\n"
                                "• PSI (changement de distribution)\n"
                                "• Taux de valeurs nulles (données manquantes)\n\n"
                                "🟢 ok — distribution stable\n"
                                "🟡 warning — dérive modérée à surveiller\n"
                                "🔴 critical — dérive forte, action recommandée\n"
                                "⚪ no_baseline / insufficient_data — pas de référence "
                                "ou moins de 10 prédictions"
                            ),
                        ),
                        "Moy. prod.": st.column_config.TextColumn(
                            "Moy. prod.",
                            help=(
                                "Moyenne des valeurs reçues en production sur la période "
                                "sélectionnée.\n\n"
                                "Calculée sur les N prédictions récentes (colonne N prod.). "
                                "Comparez avec « Moy. baseline » pour détecter un déplacement "
                                "de la distribution d'entrée — par exemple un capteur qui dérive "
                                "ou une population de clients qui change."
                            ),
                        ),
                        "Moy. baseline": st.column_config.TextColumn(
                            "Moy. baseline",
                            help=(
                                "Moyenne des valeurs de cette feature dans le dataset "
                                "d'entraînement (feature_baseline stockée avec le modèle).\n\n"
                                "C'est la référence de normalité : si « Moy. prod. » s'en "
                                "éloigne significativement, le modèle reçoit des données "
                                "différentes de celles sur lesquelles il a été entraîné."
                            ),
                        ),
                        "Z-score": st.column_config.TextColumn(
                            "Z-score",
                            help=(
                                "Déplacement de la moyenne, exprimé en unités d'écart-type "
                                "d'entraînement :\n\n"
                                "Z = |μ_prod − μ_baseline| / σ_baseline\n\n"
                                "Interprétation :\n"
                                "• Z = 1 → la moyenne a bougé d'1 écart-type (normal)\n"
                                "• Z = 2 → déplacement notable — la moitié des distributions "
                                "normales ne se chevauchent plus qu'à ~5 %\n"
                                "• Z = 3 → déplacement fort — signal de drift avéré\n\n"
                                "🟡 Warning : Z ≥ 2  ·  🔴 Critical : Z ≥ 3\n\n"
                                "⚠️ N'indique que le déplacement de la moyenne, "
                                "pas les changements de forme de la distribution "
                                "(compléter avec PSI)."
                            ),
                        ),
                        "PSI": st.column_config.TextColumn(
                            "PSI",
                            help=(
                                "Population Stability Index — mesure la différence globale "
                                "entre la distribution en production et la distribution "
                                "d'entraînement, sur 10 bins de largeur égale :\n\n"
                                "PSI = Σ (P_i − Q_i) × ln(P_i / Q_i)\n\n"
                                "Interprétation :\n"
                                "• PSI < 0.1 → distribution stable\n"
                                "• PSI 0.1–0.2 → dérive modérée — à surveiller\n"
                                "• PSI ≥ 0.2 → dérive forte — recalibration recommandée\n\n"
                                "🟡 Warning : ≥ 0.1  ·  🔴 Critical : ≥ 0.2\n\n"
                                "Contrairement au Z-score, le PSI détecte aussi les changements "
                                "de forme (asymétrie, queues plus lourdes, bimodalité…)."
                            ),
                        ),
                        "N prod.": st.column_config.NumberColumn(
                            "N prod.",
                            help=(
                                "Nombre de prédictions récentes utilisées pour calculer "
                                "les statistiques de production (Moy. prod., Z-score, PSI).\n\n"
                                "Un N faible (< 30) rend les estimations moins fiables : "
                                "le Z-score et le PSI peuvent être bruyants. "
                                "Le seuil minimal requis est 10 (en dessous : ⚪ insufficient_data)."
                            ),
                        ),
                    },
                )
            elif not feat_baseline:
                st.info("Pas de baseline de features enregistrée pour ce modèle.")
            else:
                st.info("Aucune donnée de drift disponible.")

        st.divider()

        # Tendance de confiance
        st.markdown("**📉 Tendance de confiance**")
        period_days = max(1, (end_date - start_date).days)
        try:
            conf_trend = client.get_confidence_trend(selected_model, days=period_days)
        except Exception:
            conf_trend = None

        if conf_trend is not None:
            trend_data = conf_trend.get("trend", [])
            overall = conf_trend.get("overall", {})
            if not trend_data:
                st.info("Ce modèle ne retourne pas de probabilités — tendance de confiance indisponible.")
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
                    delta_str = f"{dv:+.1%} (1re vs 2e moitié)"
                ct1, ct2, ct3 = st.columns(3)
                ct1.metric("Confiance moyenne", f"{mean_conf:.2%}", delta=delta_str, help=METRIC_HELP["confiance_moyenne"])
                ct2.metric("P25", f"{overall.get('p25_confidence', 0):.2%}", help=METRIC_HELP["p25_confiance"])
                ct3.metric("P75", f"{overall.get('p75_confidence', 0):.2%}", help=METRIC_HELP["p75_confiance"])
                fig_conf = go.Figure()
                fig_conf.add_trace(go.Scatter(
                    x=pd.concat([df_conf["date"], df_conf["date"][::-1]]),
                    y=pd.concat([df_conf["p75"], df_conf["p25"][::-1]]),
                    fill="toself", fillcolor="rgba(41,128,185,0.15)",
                    line=dict(color="rgba(0,0,0,0)"), name="IQR (P25–P75)", hoverinfo="skip",
                ))
                fig_conf.add_trace(go.Scatter(
                    x=df_conf["date"], y=df_conf["mean_confidence"],
                    name="Confiance moyenne", mode="lines+markers",
                    line=dict(color="#2980b9", width=2), marker=dict(size=5),
                ))
                fig_conf.add_hline(y=0.5, line_dash="dot", line_color="#e74c3c",
                                   annotation_text="Seuil 50%", annotation_position="bottom right")
                fig_conf.update_layout(
                    xaxis_title="Date", yaxis_title="Confiance",
                    yaxis=dict(tickformat=".0%", range=[0, 1]),
                    hovermode="x unified", legend=dict(orientation="h", y=1.02),
                )
                st.plotly_chart(fig_conf, width='stretch')
                if low_rate > 0.15:
                    st.warning(f"{low_rate * 100:.0f} % des prédictions sous le seuil — signal possible de drift d'entrée.")
        else:
            st.info("Tendance de confiance non disponible.")

        st.divider()

        # Prédictions anomales
        st.markdown("**🚨 Prédictions anomales**")
        st.caption(
            "Une prédiction est marquée **anomale** lorsqu'au moins une de ses features d'entrée "
            "présente un Z-score ≥ seuil par rapport aux données d'entraînement. "
            "Ces prédictions ont été reçues sur des entrées inhabituelles — "
            "le modèle a potentiellement extrapolé hors de sa zone de confiance."
        )
        _anom_col1, _anom_col2 = st.columns([1, 2])
        _anom_days = _anom_col1.number_input(
            "Fenêtre (jours)", min_value=1, max_value=90,
            value=min(max(period_days, 1), 90), step=1, key="anom_days",
        )
        _anom_z = _anom_col2.slider(
            "Seuil z-score", min_value=1.0, max_value=6.0, value=3.0, step=0.1,
            format="%.1f", key="anom_z_threshold",
            help=(
                "Seuil de détection : une feature est considérée aberrante si\n\n"
                "Z = |valeur reçue − moyenne entraînement| / écart-type entraînement ≥ seuil\n\n"
                "• Z = 2 → valeur à 2 écarts-types de la moyenne (5 % d'occurrence si distribution normale)\n"
                "• Z = 3 → valeur à 3 écarts-types (0.3 % d'occurrence) — anomalie forte\n\n"
                "Abaisser le seuil remonte plus d'anomalies (moins strict). "
                "Le remonter filtre uniquement les cas extrêmes."
            ),
        )
        try:
            _anom_data = client.get_predictions_anomalies(
                model_name=selected_model, days=int(_anom_days),
                z_threshold=float(_anom_z), limit=200,
            )
        except Exception as _exc:
            _anom_data = None
            st.info(f"Prédictions anomales non disponibles : {_exc}")

        if _anom_data is not None:
            if _anom_data.get("error") == "no_baseline":
                st.info("Pas de baseline de features — impossible de calculer les z-scores.")
            else:
                _total = _anom_data.get("total_checked", 0)
                _count = _anom_data.get("anomalous_count", 0)
                _rate  = _anom_data.get("anomaly_rate", 0.0)
                _preds = _anom_data.get("predictions", [])
                _mc1, _mc2, _mc3 = st.columns(3)
                _mc1.metric("Prédictions analysées", _total,
                            help=f"Prédictions récentes analysées sur les {_anom_days} derniers jours.")
                _mc2.metric("Prédictions anomales", _count,
                            help=f"Prédictions avec au moins une feature dont |Z| ≥ {_anom_z:.1f}.")
                _mc3.metric("Taux d'anomalie", f"{_rate:.1%}",
                            help="Part des prédictions anomales parmi les prédictions analysées.")
                if not _preds:
                    st.success(f"Aucune anomalie détectée (z ≥ {_anom_z:.1f}) sur {_total} prédiction(s).")
                else:
                    # Trier par z-score max décroissant — ordre stable pour le sélecteur
                    _preds_sorted = sorted(
                        _preds,
                        key=lambda x: -max(f["z_score"] for f in x["anomalous_features"].values()),
                    )
                    _rows_anom = []
                    for _p in _preds_sorted:
                        _feats = _p.get("anomalous_features", {})
                        _worst_z_val = max((_f["z_score"] for _f in _feats.values()), default=0.0)
                        _feat_names = ", ".join(
                            f"{_fn} (z={_fd['z_score']:.2f})"
                            for _fn, _fd in sorted(_feats.items(), key=lambda x: -x[1]["z_score"])
                        )
                        _rows_anom.append({
                            "ID": _p["prediction_id"],
                            "Timestamp": _p["timestamp"][:19].replace("T", " "),
                            "Résultat": str(_p.get("prediction_result", "")),
                            "Confiance": (
                                f"{_p['max_confidence']:.2%}"
                                if _p.get("max_confidence") is not None else "—"
                            ),
                            "Z-score max": round(_worst_z_val, 2),
                            "Features aberrantes": _feat_names,
                        })
                    _df_anom = pd.DataFrame(_rows_anom)
                    st.dataframe(
                        _df_anom, width='stretch', hide_index=True,
                        column_config={
                            "ID": st.column_config.NumberColumn(
                                "ID",
                                help="Identifiant unique de la prédiction dans la base.",
                            ),
                            "Timestamp": st.column_config.TextColumn(
                                "Timestamp",
                                help="Date et heure de la prédiction.",
                            ),
                            "Résultat": st.column_config.TextColumn(
                                "Résultat",
                                help=(
                                    "Valeur prédite par le modèle pour cette requête. "
                                    "Peut être moins fiable si plusieurs features sont aberrantes."
                                ),
                            ),
                            "Confiance": st.column_config.TextColumn(
                                "Confiance",
                                help=(
                                    "Probabilité max retournée par le modèle (classifieurs uniquement). "
                                    "— pour les modèles de régression. "
                                    "Une confiance haute sur une prédiction anomale est un signal fort "
                                    "de sur-confiance du modèle hors de sa zone d'entraînement."
                                ),
                            ),
                            "Z-score max": st.column_config.NumberColumn(
                                "Z-score max",
                                format="%.2f",
                                help=(
                                    "Z-score le plus élevé parmi toutes les features aberrantes "
                                    "de cette prédiction.\n\n"
                                    "Z = |valeur reçue − μ entraînement| / σ entraînement\n\n"
                                    "Plus Z est élevé, plus l'entrée est atypique par rapport "
                                    "aux données vues à l'entraînement."
                                ),
                            ),
                            "Features aberrantes": st.column_config.TextColumn(
                                "Features aberrantes",
                                help=(
                                    "Features dont |Z| ≥ seuil, classées par Z-score décroissant. "
                                    "Format : nom_feature (z=X.XX). "
                                    "Sélectionnez une ligne ci-dessous pour voir le détail complet."
                                ),
                            ),
                        },
                    )

                    # ── Sélecteur de prédiction pour le détail ──────────────
                    st.markdown("##### 🔍 Détail d'une prédiction")
                    _sel_labels = [
                        f"#{_p['prediction_id']}  ·  {_p['timestamp'][:16].replace('T', ' ')}  "
                        f"·  z_max = {max(f['z_score'] for f in _p['anomalous_features'].values()):.2f}"
                        f"  ({', '.join(sorted(_p['anomalous_features'].keys())[:2])}"
                        f"{'…' if len(_p['anomalous_features']) > 2 else ''})"
                        for _p in _preds_sorted
                    ]
                    _pred_by_label = dict(zip(_sel_labels, _preds_sorted))
                    _sel_label = st.selectbox(
                        "Prédiction à inspecter",
                        options=_sel_labels,
                        index=0,
                        key="anom_detail_sel",
                        label_visibility="collapsed",
                        placeholder="Sélectionnez une prédiction…",
                    )
                    if _sel_label:
                        _sel_pred = _pred_by_label[_sel_label]
                        _feats = _sel_pred.get("anomalous_features", {})
                        _feats_sorted = sorted(_feats.items(), key=lambda x: -x[1]["z_score"])

                        with st.container(border=True):
                            # En-tête
                            _hc1, _hc2, _hc3 = st.columns([3, 1, 1])
                            _hc1.markdown(
                                f"**Prédiction #{_sel_pred['prediction_id']}**  ·  "
                                f"{_sel_pred['timestamp'][:19].replace('T', ' ')}"
                            )
                            _hc2.metric("Résultat", str(_sel_pred.get("prediction_result", "")))
                            if _sel_pred.get("max_confidence") is not None:
                                _hc3.metric("Confiance", f"{_sel_pred['max_confidence']:.2%}")

                            # Graphique + tableau côte à côte
                            _dc1, _dc2 = st.columns([1, 1])

                            with _dc1:
                                st.caption("Z-scores des features aberrantes")
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
                                    annotation_text=f"Seuil {_anom_z:.1f}",
                                    annotation_position="top right",
                                )
                                _fig_z.update_layout(
                                    margin=dict(l=0, r=10, t=10, b=30),
                                    yaxis=dict(autorange="reversed"),
                                    coloraxis_showscale=False,
                                    showlegend=False,
                                )
                                st.plotly_chart(_fig_z, use_container_width=True)

                            with _dc2:
                                st.caption("Valeurs reçues vs baseline d'entraînement")
                                _feat_detail_rows = [{
                                    "Feature": fn,
                                    "Valeur reçue": round(fd["value"], 4),
                                    "Moy. entraîn. (μ)": round(fd["baseline_mean"], 4),
                                    "Écart-type (σ)": round(fd["baseline_std"], 4),
                                    "Plage normale": (
                                        f"[{fd['baseline_mean'] - 2*fd['baseline_std']:.3g}"
                                        f" – {fd['baseline_mean'] + 2*fd['baseline_std']:.3g}]"
                                    ),
                                    "Z-score": round(fd["z_score"], 2),
                                } for fn, fd in _feats_sorted]
                                st.dataframe(
                                    pd.DataFrame(_feat_detail_rows),
                                    hide_index=True,
                                    use_container_width=True,
                                    column_config={
                                        "Feature": st.column_config.TextColumn("Feature"),
                                        "Valeur reçue": st.column_config.NumberColumn(
                                            "Valeur reçue", format="%.4f",
                                            help="Valeur reçue dans la requête de prédiction.",
                                        ),
                                        "Moy. entraîn. (μ)": st.column_config.NumberColumn(
                                            "μ entraîn.", format="%.4f",
                                            help="Moyenne de cette feature dans le dataset d'entraînement.",
                                        ),
                                        "Écart-type (σ)": st.column_config.NumberColumn(
                                            "σ entraîn.", format="%.4f",
                                            help="Écart-type de cette feature dans le dataset d'entraînement.",
                                        ),
                                        "Plage normale": st.column_config.TextColumn(
                                            "Plage normale [μ±2σ]",
                                            help=(
                                                "Intervalle μ ± 2σ — 95 % des valeurs d'entraînement "
                                                "se situaient dans cette plage. "
                                                "Une valeur hors de cette plage est statistiquement rare."
                                            ),
                                        ),
                                        "Z-score": st.column_config.NumberColumn(
                                            "Z-score", format="%.2f",
                                            help=(
                                                "Z = |valeur reçue − μ| / σ\n\n"
                                                "Nombre d'écarts-types séparant la valeur reçue "
                                                "de la moyenne d'entraînement."
                                            ),
                                        ),
                                    },
                                )

    # ────────────────────────────────────────────────────────────────────
    # EXPANDER 4 — Comparaison A/B & Shadow
    # ────────────────────────────────────────────────────────────────────
    if ab_comparison:
        ab_versions = ab_comparison.get("versions", [])
        agreement   = ab_comparison.get("shadow_agreement_rate", {})
        with st.expander("⚖️ Comparaison A/B & Shadow", expanded=False):
            if ab_versions:
                df_ab = pd.DataFrame([{
                    "Version": v["version"],
                    "Prédictions": v["total_predictions"],
                    "Shadow": v["shadow_predictions"],
                    "Taux d'erreur": f"{v['error_rate'] * 100:.1f} %",
                    "Latence moy.": f"{v.get('avg_response_time_ms')} ms" if v.get("avg_response_time_ms") else "—",
                    "p95": f"{v.get('p95_response_time_ms')} ms" if v.get("p95_response_time_ms") else "—",
                    "Accord shadow": f"{agreement.get(v['version'], 0) * 100:.1f} %" if v["version"] in agreement else "—",
                } for v in ab_versions])
                st.dataframe(df_ab, width='stretch', hide_index=True)

                # ── Distribution des sorties prédites ─────────────────────
                _all_dist_labels = sorted({
                    str(lbl)
                    for v in ab_versions
                    for lbl in v.get("prediction_distribution", {}).keys()
                })
                _is_reg_ab = any("." in lbl for lbl in _all_dist_labels)

                if _is_reg_ab:
                    # ── Régression : stats pondérées depuis la distribution ──
                    st.markdown("**📊 Distribution des sorties prédites (régression)**")
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
                            "Version": v["version"],
                            "N": _n,
                            "Moyenne ŷ": round(_mean, 3),
                            "Écart-type σ": round(_var ** 0.5, 3),
                            "Min": round(min(val for val, _ in _numeric), 3),
                            "Max": round(max(val for val, _ in _numeric), 3),
                        })
                    if _reg_rows:
                        _df_reg = pd.DataFrame(_reg_rows)
                        # Ligne delta si exactement 2 versions
                        if len(_reg_rows) == 2:
                            r0, r1 = _reg_rows[0], _reg_rows[1]
                            _df_reg = pd.concat([
                                _df_reg,
                                pd.DataFrame([{
                                    "Version": f"Δ  ({r1['Version']} − {r0['Version']})",
                                    "N": r1["N"] - r0["N"],
                                    "Moyenne ŷ": round(r1["Moyenne ŷ"] - r0["Moyenne ŷ"], 3),
                                    "Écart-type σ": round(r1["Écart-type σ"] - r0["Écart-type σ"], 3),
                                    "Min": round(r1["Min"] - r0["Min"], 3),
                                    "Max": round(r1["Max"] - r0["Max"], 3),
                                }]),
                            ], ignore_index=True)
                        st.dataframe(
                            _df_reg, hide_index=True, width='stretch',
                            column_config={
                                "Version": st.column_config.TextColumn("Version"),
                                "N": st.column_config.NumberColumn(
                                    "N prédictions",
                                    help="Nombre de prédictions production sur la période.",
                                ),
                                "Moyenne ŷ": st.column_config.NumberColumn(
                                    "Moyenne ŷ", format="%.3f",
                                    help=(
                                        "Valeur prédite moyenne. "
                                        "Un écart entre versions indique un biais systématique — "
                                        "les deux modèles ne prédisent pas la même plage de valeurs."
                                    ),
                                ),
                                "Écart-type σ": st.column_config.NumberColumn(
                                    "Écart-type σ", format="%.3f",
                                    help=(
                                        "Dispersion des prédictions autour de la moyenne. "
                                        "Un σ très différent entre versions peut indiquer "
                                        "un comportement divergent (une version est plus "
                                        "« prudente » que l'autre)."
                                    ),
                                ),
                                "Min": st.column_config.NumberColumn(
                                    "Min", format="%.3f",
                                    help="Valeur prédite minimale — utile pour détecter des extrapolations basses.",
                                ),
                                "Max": st.column_config.NumberColumn(
                                    "Max", format="%.3f",
                                    help="Valeur prédite maximale — utile pour détecter des extrapolations hautes.",
                                ),
                            },
                        )
                        # Graphique Moyenne ± σ
                        if len(_reg_rows) >= 2:
                            _fig_reg = px.bar(
                                pd.DataFrame(_reg_rows),
                                x="Version", y="Moyenne ŷ",
                                error_y="Écart-type σ",
                                color="Version",
                                text="Moyenne ŷ",
                                labels={"Moyenne ŷ": "Valeur prédite ŷ"},
                                color_discrete_sequence=px.colors.qualitative.Set2,
                                title="Moyenne ŷ ± σ par version",
                            )
                            _fig_reg.update_traces(
                                texttemplate="%{text:.3f}", textposition="outside",
                            )
                            _fig_reg.update_layout(
                                showlegend=False, height=320,
                                yaxis_title="Valeur prédite ŷ",
                            )
                            st.plotly_chart(_fig_reg, use_container_width=True)

                elif _all_dist_labels:
                    # ── Classification : barres 100 % empilées ──────────────
                    st.markdown("**📊 Répartition des classes prédites**")
                    # Table % + count
                    _cls_rows_pct = []
                    for v in ab_versions:
                        _dist = {
                            str(k): int(c)
                            for k, c in v.get("prediction_distribution", {}).items()
                        }
                        _total = sum(_dist.values()) or 1
                        _row = {"Version": v["version"], "Total": sum(_dist.values())}
                        for _lbl in _all_dist_labels:
                            _cnt = _dist.get(_lbl, 0)
                            _row[f"Cl. {_lbl}"] = f"{_cnt / _total:.1%}  ({_cnt})"
                        _cls_rows_pct.append(_row)
                    st.dataframe(
                        pd.DataFrame(_cls_rows_pct), hide_index=True, width='stretch',
                    )
                    # Graphique 100 % empilé
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
                                "Version": v["version"],
                                "Classe": _lbl,
                                "%": round(_cnt / _total * 100, 2),
                                "N": _cnt,
                            })
                    _fig_pct = px.bar(
                        pd.DataFrame(_pct_plot),
                        x="Version", y="%", color="Classe",
                        barmode="stack",
                        text=pd.DataFrame(_pct_plot)["%"].apply(lambda v: f"{v:.1f}%"),
                        labels={"%": "Part (%)", "Classe": "Classe prédite"},
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        custom_data=["N"],
                        title="Répartition des classes prédites (%)",
                    )
                    _fig_pct.update_traces(
                        hovertemplate=(
                            "Classe %{fullData.name}<br>"
                            "%{y:.1f}%  (%{customdata[0]} prédictions)"
                            "<extra></extra>"
                        ),
                    )
                    _fig_pct.update_layout(
                        yaxis=dict(ticksuffix="%", range=[0, 108]),
                        legend_title="Classe prédite",
                        height=380,
                    )
                    st.plotly_chart(_fig_pct, width='stretch')

    # ────────────────────────────────────────────────────────────────────
    # EXPANDER 5 — Calibration des probabilités
    # ────────────────────────────────────────────────────────────────────
    with st.expander("📏 Calibration des probabilités", expanded=False):
        try:
            calib = client.get_model_calibration(model_name=selected_model, start=start_iso, end=end_iso)
        except Exception as exc:
            st.info(f"Calibration non disponible : {exc}")
            calib = None

        if calib:
            calib_status = calib.get("calibration_status", "insufficient_data")
            if calib_status == "insufficient_data":
                st.info(f"Soumettez des observed_results pour activer la calibration (échantillon : {calib.get('sample_size', 0)} paires).")
            else:
                STATUS_CALIB = {"ok": "🟢 OK", "overconfident": "🟡 Sur-confiant", "underconfident": "🔴 Sous-confiant"}
                brier = calib.get("brier_score")
                gap   = calib.get("overconfidence_gap")
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Brier score", f"{brier:.4f}" if brier is not None else "—", help=METRIC_HELP["brier_score"])
                cc2.metric("Gap confiance/précision", f"{gap:+.2%}" if gap is not None else "—", help=METRIC_HELP["gap_confiance"])
                cc3.metric("Statut", STATUS_CALIB.get(calib_status, calib_status), help=METRIC_HELP["statut_calibration"])
                reliability = calib.get("reliability", [])
                if reliability:
                    bins       = [b["confidence_bin"]  for b in reliability]
                    obs_rates  = [b["observed_rate"]   for b in reliability]
                    counts     = [b["count"]            for b in reliability]
                    diag_vals  = [(float(b.split("–")[0]) + float(b.split("–")[1])) / 2 for b in bins]
                    fig_cal = go.Figure()
                    fig_cal.add_trace(go.Scatter(x=bins, y=diag_vals, name="Calibration parfaite",
                                                 line=dict(color="grey", dash="dot", width=1), mode="lines"))
                    fig_cal.add_trace(go.Scatter(
                        x=bins + bins[::-1], y=obs_rates + diag_vals[::-1],
                        fill="toself", fillcolor="rgba(200,200,200,0.25)",
                        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
                    ))
                    fig_cal.add_trace(go.Scatter(
                        x=bins, y=obs_rates, name="Taux observé",
                        mode="markers+lines",
                        marker=dict(size=[max(8, min(24, c // 5)) for c in counts], color="#2980b9"),
                        customdata=counts,
                        hovertemplate="Bucket : %{x}<br>Taux observé : %{y:.2%}<br>N : %{customdata}<extra></extra>",
                    ))
                    fig_cal.update_layout(
                        title="Courbe de calibration (reliability diagram)",
                        xaxis_title="Confiance prédite", yaxis_title="Taux observé",
                        yaxis=dict(tickformat=".0%", range=[0, 1]),
                        hovermode="x unified", legend=dict(orientation="h", y=1.05),
                    )
                    st.plotly_chart(fig_cal, width='stretch')
                if calib_status == "overconfident":
                    st.warning("Envisagez `CalibratedClassifierCV(method='isotonic')` lors du prochain retrain.")

    # ────────────────────────────────────────────────────────────────────
    # EXPANDER 6 — Erreurs récentes
    # ────────────────────────────────────────────────────────────────────
    if recent_errors:
        with st.expander(f"🔔 Erreurs récentes ({len(recent_errors)})", expanded=False):
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

    _conf_label = "⚙️ Configuration"
    _conf_badges = []
    if _seuils_set:      _conf_badges.append("🔔")
    if _auto_promote_on: _conf_badges.append("🚀")
    if _auto_demote_on:  _conf_badges.append("⚡")
    if _conf_badges:
        _conf_label += "  " + " ".join(_conf_badges)

    with st.expander(_conf_label, expanded=False):
        _conf_tab_seuils, _conf_tab_promo, _conf_tab_cb = st.tabs([
            "🔔 Seuils d'alerte",
            "🚀 Auto-promotion",
            "⚡ Circuit breaker",
        ])

        # ── Seuils d'alerte ───────────────────────────────────────────────
        with _conf_tab_seuils:
            if not per_version:
                st.info("Aucune version disponible.")
            else:
                _ver_options = [v["version"] for v in per_version]
                _default_ver = next(
                    (v["version"] for v in per_version if v.get("deployment_mode") == "production"),
                    _ver_options[0],
                )
                _sel_ver = st.selectbox(
                    "Version à configurer", _ver_options,
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
                    _ci1.metric("Taux d'erreur max", f"{_err_val * 100:.1f} %" if _err_val is not None else "—")
                    _ci2.metric("Accuracy min", f"{_acc_val:.0%}" if _acc_val is not None else "—")
                    _ci3.metric("Alerte drift", "✅ Activée" if _drift_val else "⬜ Désactivée")

                with st.form("alert_thresholds_form"):
                    _col1, _col2 = st.columns(2)
                    new_error_rate_pct = _col1.number_input(
                        "Taux d'erreur max (%)", min_value=0.0, max_value=100.0,
                        value=round(_err_val * 100, 2) if _err_val is not None else 10.0,
                        step=0.5, help="Déclenche une alerte si le taux d'erreur dépasse ce seuil.",
                    )
                    new_accuracy_min = _col2.number_input(
                        "Accuracy minimale", min_value=0.0, max_value=1.0,
                        value=float(_acc_val) if _acc_val is not None else 0.80,
                        step=0.01, format="%.2f", help="Déclenche une alerte si l'accuracy tombe sous ce seuil.",
                    )
                    new_drift_auto = st.checkbox("Alerte automatique sur drift de features", value=bool(_drift_val))
                    _submitted = st.form_submit_button("💾 Enregistrer les seuils", type="primary")
                    if _submitted:
                        try:
                            client.update_model(selected_model, _sel_ver, {"alert_thresholds": {
                                "error_rate_max": new_error_rate_pct / 100,
                                "accuracy_min": new_accuracy_min,
                                "drift_auto_alert": new_drift_auto,
                            }})
                            st.success(f"✅ Seuils mis à jour pour {selected_model} v{_sel_ver}.")
                            st.cache_data.clear()
                        except Exception as _exc:
                            st.error(f"Erreur : {_exc}")

        # ── Auto-promotion ────────────────────────────────────────────────
        with _conf_tab_promo:
            st.caption(
                f"Promeut automatiquement le modèle ré-entraîné si tous les critères sont satisfaits. "
                f"Appliquée à toutes les versions de **{selected_model}**. 0 = critère désactivé."
            )
            _ap1, _ap2 = st.columns(2)
            with _ap1:
                new_auto_promote = st.checkbox(
                    "Activer l'auto-promotion", value=_auto_promote_on,
                    key="sup_auto_promote", disabled=not _is_sup_admin,
                )
                _min_acc = _policy.get("min_accuracy")
                new_min_accuracy = st.number_input(
                    "Accuracy minimale (0 = désactivé)",
                    min_value=0.0, max_value=1.0, step=0.01,
                    value=float(_min_acc) if _min_acc is not None else 0.0,
                    key="sup_min_accuracy", disabled=not _is_sup_admin,
                    help="Accuracy requise sur les paires de validation pour promotion.",
                )
                _min_golden = _policy.get("min_golden_test_pass_rate")
                new_min_golden = st.number_input(
                    "Taux Golden Tests min (0 = désactivé)",
                    min_value=0.0, max_value=1.0, step=0.01,
                    value=float(_min_golden) if _min_golden is not None else 0.0,
                    key="sup_min_golden", disabled=not _is_sup_admin,
                )
            with _ap2:
                _max_mae = _policy.get("max_mae")
                new_max_mae = st.number_input(
                    "MAE maximale (0 = désactivé)",
                    min_value=0.0, step=0.01,
                    value=float(_max_mae) if _max_mae is not None else 0.0,
                    key="sup_max_mae", disabled=not _is_sup_admin,
                    help="MAE maximale autorisée (régression uniquement).",
                )
                _max_lat = _policy.get("max_latency_p95_ms")
                new_max_latency = st.number_input(
                    "Latence P95 max (ms, 0 = désactivé)",
                    min_value=0.0, step=10.0,
                    value=float(_max_lat) if _max_lat is not None else 0.0,
                    key="sup_max_latency", disabled=not _is_sup_admin,
                )
                new_min_samples = st.number_input(
                    "Échantillons de validation min",
                    min_value=1, step=1,
                    value=int(_policy.get("min_sample_validation", 10)),
                    key="sup_min_samples", disabled=not _is_sup_admin,
                )

            if _is_sup_admin and st.button("💾 Sauvegarder l'auto-promotion", key="save_auto_promote"):
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
                    st.toast("Politique d'auto-promotion mise à jour.", icon="✅")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as _exc:
                    st.error(f"Erreur : {_exc}")
            elif not _is_sup_admin:
                st.caption("🔒 Réservé aux administrateurs.")

        # ── Circuit breaker ───────────────────────────────────────────────
        with _conf_tab_cb:
            st.caption(
                "Retire automatiquement du production un modèle dont les performances se dégradent. "
                "Appliqué à toutes les versions."
            )
            _cb1, _cb2 = st.columns(2)
            with _cb1:
                new_auto_demote = st.checkbox(
                    "Activer le circuit breaker", value=_auto_demote_on,
                    key="sup_auto_demote", disabled=not _is_sup_admin,
                )
                new_demote_on_drift = st.selectbox(
                    "Niveau de drift déclencheur",
                    ["warning", "critical"],
                    index=0 if _policy.get("demote_on_drift", "critical") == "warning" else 1,
                    key="sup_demote_on_drift",
                    format_func=lambda x: "⚠️ Warning" if x == "warning" else "🔴 Critical",
                    disabled=not _is_sup_admin,
                )
            with _cb2:
                _acc_thr = _policy.get("demote_on_accuracy_below")
                new_demote_accuracy = st.number_input(
                    "Accuracy minimale (0 = désactivé)",
                    min_value=0.0, max_value=1.0, step=0.01,
                    value=float(_acc_thr) if _acc_thr is not None else 0.0,
                    key="sup_demote_accuracy", disabled=not _is_sup_admin,
                    help="Descend sous ce seuil → demotion automatique.",
                )
                new_cooldown = st.number_input(
                    "Cooldown (heures)",
                    min_value=0, step=1,
                    value=int(_policy.get("demote_cooldown_hours", 24)),
                    key="sup_cooldown", disabled=not _is_sup_admin,
                    help="Délai minimal entre deux auto-demotions.",
                )

            if _is_sup_admin and st.button("💾 Sauvegarder le circuit breaker", key="save_circuit_breaker"):
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
                    st.toast("Circuit breaker mis à jour.", icon="✅")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as _exc:
                    st.error(f"Erreur : {_exc}")
            elif not _is_sup_admin:
                st.caption("🔒 Réservé aux administrateurs.")

            # Historique de la dernière auto-demotion
            try:
                _history_data    = client.get_model_history(selected_model)
                _history_entries = _history_data.get("history", [])
                _last_demote     = next((e for e in _history_entries if e.get("action") == "auto_demote"), None)
            except Exception:
                _last_demote = None

            if _last_demote:
                _demote_ts     = _last_demote.get("timestamp", "")[:16].replace("T", " ")
                _demote_reason = (_last_demote.get("snapshot") or {}).get("auto_demote_reason", "—")
                st.error(f"🔴 **Auto-démis le {_demote_ts} UTC** — {_demote_reason}", icon="🚨")
            elif _auto_demote_on:
                st.success("🟢 Aucune auto-demotion récente détectée.")
