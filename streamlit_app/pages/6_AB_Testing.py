"""
Dashboard A/B Testing & Shadow Deployment
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.auth import require_auth, require_admin, get_client

st.set_page_config(page_title="A/B Testing — PredictML", page_icon="🧪", layout="wide")
require_auth()

st.title("🧪 A/B Testing & Shadow Deployment")
st.caption(
    "Configurez des splits de trafic entre versions d'un même modèle et comparez leurs métriques en temps réel."
)

client = get_client()
is_admin = st.session_state.get("is_admin", False)

# --- Charger la liste des modèles ---
try:
    all_models = client.list_models()
except Exception as e:
    st.error(f"Impossible de charger les modèles : {e}")
    st.stop()

if not all_models:
    st.info("Aucun modèle disponible.")
    st.stop()

# Grouper les versions par nom de modèle
from collections import defaultdict

model_groups: dict = defaultdict(list)
for m in all_models:
    model_groups[m["name"]].append(m)

model_names = sorted(model_groups.keys())
selected_model = st.selectbox("🔍 Modèle", model_names, key="ab_model_select")
versions_for_model = model_groups[selected_model]

st.divider()

# ===========================================================================
# SECTION 1 — Configuration du déploiement (admin seulement)
# ===========================================================================
if is_admin:
    with st.expander("⚙️ Configuration A/B / Shadow", expanded=True):
        st.markdown(
            "Configurez le mode de déploiement et le poids de trafic de chaque version. "
            "La somme des poids A/B doit être **≤ 1.0**."
        )

        MODES = ["(inchangé)", "ab_test", "shadow", "production"]
        configs: dict = {}

        cols_header = st.columns([2, 2, 2, 1])
        cols_header[0].markdown("**Version**")
        cols_header[1].markdown("**Mode actuel**")
        cols_header[2].markdown("**Nouveau mode**")
        cols_header[3].markdown("**Poids**")

        for v in versions_for_model:
            ver = v["version"]
            mode_current = v.get("deployment_mode") or "—"
            weight_current = v.get("traffic_weight")

            row_cols = st.columns([2, 2, 2, 1])
            row_cols[0].markdown(f"`{ver}`")

            # Badge mode actuel
            badge = {"ab_test": "🟠 A/B", "shadow": "🟣 Shadow", "production": "🟢 Prod"}.get(
                mode_current, f"⚪ {mode_current}"
            )
            row_cols[1].markdown(badge)

            new_mode = row_cols[2].selectbox(
                "Mode",
                MODES,
                key=f"mode_{ver}",
                label_visibility="collapsed",
            )

            new_weight = None
            if new_mode == "ab_test":
                new_weight = row_cols[3].number_input(
                    "Poids",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    value=float(weight_current or 0.5),
                    key=f"weight_{ver}",
                    label_visibility="collapsed",
                )
            else:
                row_cols[3].markdown("—")

            configs[ver] = {"mode": new_mode, "weight": new_weight, "current": v}

        # Somme des poids preview
        total_weight = sum(
            cfg["weight"]
            for cfg in configs.values()
            if cfg["mode"] == "ab_test" and cfg["weight"] is not None
        )
        # Prendre en compte aussi les versions non modifiées (mode=(inchangé)) déjà en ab_test
        for cfg in configs.values():
            if cfg["mode"] == "(inchangé)" and cfg["current"].get("deployment_mode") == "ab_test":
                total_weight += cfg["current"].get("traffic_weight") or 0.0

        weight_color = "🟢" if total_weight <= 1.0 else "🔴"
        st.markdown(f"**Somme des poids A/B :** {weight_color} `{total_weight:.2f}` / 1.0")
        if total_weight > 1.0:
            st.warning("⚠️ La somme des poids dépasse 1.0 — corrigez avant d'appliquer.")

        if st.button("✅ Appliquer la configuration", type="primary", disabled=total_weight > 1.0):
            errors = []
            updated = 0
            for ver, cfg in configs.items():
                if cfg["mode"] == "(inchangé)":
                    continue
                try:
                    client.update_model_deployment(
                        name=selected_model,
                        version=ver,
                        deployment_mode=cfg["mode"],
                        traffic_weight=cfg["weight"],
                    )
                    updated += 1
                except Exception as e:
                    errors.append(f"`{ver}` : {e}")

            if errors:
                for err in errors:
                    st.error(err)
            if updated:
                st.success(f"{updated} version(s) mise(s) à jour.")
                st.cache_data.clear()
                st.rerun()

else:
    st.info("🔒 La configuration A/B est réservée aux administrateurs.")

st.divider()

# ===========================================================================
# SECTION 2 — Dashboard de comparaison
# ===========================================================================
st.subheader("📊 Comparaison des versions")

period_label = st.radio(
    "Période d'analyse",
    ["7 jours", "14 jours", "30 jours"],
    horizontal=True,
    key="ab_period",
)
days_map = {"7 jours": 7, "14 jours": 14, "30 jours": 30}
days = days_map[period_label]

try:
    ab_data = client.get_ab_comparison(selected_model, days=days)
    versions_stats = ab_data.get("versions", [])
except Exception as e:
    st.error(f"Impossible de charger les données de comparaison : {e}")
    versions_stats = []

if not versions_stats:
    st.info("Aucune prédiction enregistrée pour ce modèle sur la période sélectionnée.")
else:
    # KPI cards par version
    num_versions = len(versions_stats)
    kpi_cols = st.columns(num_versions)
    for i, vs in enumerate(versions_stats):
        mode = vs.get("deployment_mode") or "legacy"
        mode_badge = {"ab_test": "🟠 A/B", "shadow": "🟣 Shadow", "production": "🟢 Prod"}.get(
            mode, f"⚪ {mode}"
        )
        weight = vs.get("traffic_weight")
        weight_str = f" — `{weight:.0%}`" if weight is not None else ""
        with kpi_cols[i]:
            st.markdown(f"**v{vs['version']}** {mode_badge}{weight_str}")
            st.metric("Prédictions (prod)", vs["total_predictions"])
            st.metric("Shadow", vs["shadow_predictions"])
            err_rate = vs.get("error_rate", 0)
            st.metric("Taux d'erreur", f"{err_rate:.1%}")
            avg_rt = vs.get("avg_response_time_ms")
            p95_rt = vs.get("p95_response_time_ms")
            st.metric("Latence avg (ms)", f"{avg_rt:.1f}" if avg_rt is not None else "—")
            st.metric("Latence p95 (ms)", f"{p95_rt:.1f}" if p95_rt is not None else "—")
            agree = vs.get("agreement_rate")
            if agree is not None:
                st.metric("Concordance shadow", f"{agree:.1%}")

    st.divider()

    # Graphique : distribution du trafic par version
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**Répartition du trafic (prédictions production)**")
        traffic_df = pd.DataFrame(
            [
                {"Version": vs["version"], "Prédictions": vs["total_predictions"]}
                for vs in versions_stats
            ]
        )
        if traffic_df["Prédictions"].sum() > 0:
            fig_traffic = px.bar(
                traffic_df,
                x="Version",
                y="Prédictions",
                color="Version",
                text_auto=True,
            )
            fig_traffic.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_traffic, use_container_width=True)
        else:
            st.info("Pas de prédictions production enregistrées.")

    with col_chart2:
        st.markdown("**Distribution des labels par version (production)**")
        dist_rows = []
        for vs in versions_stats:
            for label, count in vs.get("prediction_distribution", {}).items():
                dist_rows.append({"Version": vs["version"], "Label": str(label), "Count": count})

        if dist_rows:
            dist_df = pd.DataFrame(dist_rows)
            fig_dist = px.bar(
                dist_df,
                x="Label",
                y="Count",
                color="Version",
                barmode="group",
            )
            fig_dist.update_layout(height=300)
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Pas de distribution disponible.")

    # Taux de concordance shadow
    shadow_versions = [vs for vs in versions_stats if vs.get("agreement_rate") is not None]
    if shadow_versions:
        st.divider()
        st.markdown("**🔮 Concordance shadow vs production**")
        st.caption(
            "Fraction des prédictions shadow qui correspondent à la prédiction production "
            "pour le même `id_obs`. Nécessite que `id_obs` soit renseigné dans les requêtes."
        )
        agree_data = [
            {
                "Version shadow": vs["version"],
                "Concordance": f"{vs['agreement_rate']:.1%}",
                "Score": vs["agreement_rate"],
            }
            for vs in shadow_versions
        ]
        agree_df = pd.DataFrame(agree_data)
        fig_agree = px.bar(
            agree_df,
            x="Version shadow",
            y="Score",
            text="Concordance",
            color_discrete_sequence=["#7c3aed"],
            range_y=[0, 1],
        )
        fig_agree.update_layout(height=250, yaxis_tickformat=".0%")
        st.plotly_chart(fig_agree, use_container_width=True)
    else:
        st.info(
            "💡 Le taux de concordance shadow/production sera calculé automatiquement "
            "dès que des prédictions avec `id_obs` seront enregistrées pour les deux versions."
        )

st.divider()

# ===========================================================================
# SECTION 3 — Log des prédictions shadow récentes
# ===========================================================================
st.subheader("🔮 Prédictions shadow récentes")

end_dt = datetime.utcnow()
start_dt = end_dt - timedelta(days=days)

try:
    pred_data = client.get_predictions(
        model_name=selected_model,
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
        limit=200,
    )
    all_preds = pred_data.get("predictions", [])
    shadow_preds = [p for p in all_preds if p.get("is_shadow")]
except Exception as e:
    st.warning(f"Impossible de charger les prédictions : {e}")
    shadow_preds = []

if not shadow_preds:
    st.info("Aucune prédiction shadow enregistrée sur cette période.")
else:
    shadow_rows = []
    for p in shadow_preds:
        shadow_rows.append(
            {
                "ID": p.get("id"),
                "Timestamp": pd.to_datetime(p.get("timestamp")).strftime("%Y-%m-%d %H:%M:%S")
                if p.get("timestamp")
                else "—",
                "Version": p.get("model_version") or "—",
                "id_obs": p.get("id_obs") or "—",
                "Résultat": str(p.get("prediction_result", "")),
                "Latence (ms)": f"{p['response_time_ms']:.1f}"
                if p.get("response_time_ms") is not None
                else "—",
                "Statut": "✅" if p.get("status") == "success" else "❌",
            }
        )

    st.caption(f"{len(shadow_preds)} prédiction(s) shadow")
    st.dataframe(pd.DataFrame(shadow_rows), use_container_width=True, hide_index=True)
