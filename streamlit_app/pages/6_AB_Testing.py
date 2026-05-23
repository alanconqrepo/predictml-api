"""
Dashboard A/B Testing & Shadow Deployment
"""

from collections import defaultdict
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.api_client import get_models as get_models_cached
from utils.auth import get_client, require_auth
from utils.metrics_help import METRIC_HELP

st.set_page_config(page_title="A/B Testing — PredictML", page_icon="🧪", layout="wide")
require_auth()

col_title, col_refresh = st.columns([8, 1])
col_title.title("🧪 A/B Testing & Shadow Deployment")
if col_refresh.button("🔄 Rafraîchir", key="ab_refresh", width='stretch'):
    st.cache_data.clear()
    st.rerun()
st.caption(
    "Configurez des splits de trafic entre versions d'un même modèle et comparez leurs métriques en temps réel."
)

client = get_client()
is_admin = st.session_state.get("is_admin", False)

# --- Charger la liste des modèles ---
try:
    all_models = get_models_cached(
        st.session_state.get("api_url"), st.session_state.get("api_token")
    )
except Exception as e:
    st.error(f"Impossible de charger les modèles : {e}")
    st.stop()

if not all_models:
    st.info("Aucun modèle disponible.")
    st.stop()

# Grouper les versions par nom de modèle
model_groups: dict = defaultdict(list)
for m in all_models:
    model_groups[m["name"]].append(m)

model_names = sorted(model_groups.keys())
_ab_col1, _ab_col2 = st.columns([2, 3])
ab_search = _ab_col1.text_input("Filtrer par nom", key="ab_model_search", placeholder="Rechercher un modèle…")
ab_filtered = [n for n in model_names if ab_search.lower() in n.lower()] if ab_search else model_names
selected_model = _ab_col2.selectbox("🔍 Modèle", ab_filtered or model_names, key="ab_model_select")
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

        # Mapping label ↔ valeur API
        # "—" = aucun changement (utilisé quand le mode courant n'est pas dans la liste)
        _MODE_TO_LABEL = {
            "ab_test":    "🟠 A/B",
            "shadow":     "🟣 Shadow",
            "production": "🟢 Prod",
        }
        _LABEL_TO_MODE = {v: k for k, v in _MODE_TO_LABEL.items()}
        _MODE_LABELS = ["—"] + list(_MODE_TO_LABEL.values())  # ["—", "🟠 A/B", "🟣 Shadow", "🟢 Prod"]
        configs: dict = {}

        cols_header = st.columns([2, 2, 1, 2, 1])
        cols_header[0].markdown("**Version**")
        cols_header[1].markdown("**Mode actuel**")
        cols_header[2].markdown("**Poids actuel**")
        cols_header[3].markdown("**Nouveau mode**")
        cols_header[4].markdown("**Nouveau poids**")

        for v in versions_for_model:
            ver = v["version"]
            mode_current = v.get("deployment_mode") or ""
            weight_current = v.get("traffic_weight")

            row_cols = st.columns([2, 2, 1, 2, 1])
            row_cols[0].markdown(f"`{ver}`")

            # Mode actuel (badge)
            row_cols[1].markdown(_MODE_TO_LABEL.get(mode_current, f"⚪ {mode_current or '—'}"))

            # Poids actuel
            row_cols[2].markdown(
                f"`{weight_current:.0%}`" if weight_current is not None else "—"
            )

            # Nouveau mode — badge pré-rempli ; "—" si mode inconnu (ex: uploaded)
            _cur_label = _MODE_TO_LABEL.get(mode_current)   # None si mode inconnu
            _default_idx = _MODE_LABELS.index(_cur_label) if _cur_label else 0  # 0 = "—"
            new_mode_label = row_cols[3].selectbox(
                "Mode",
                _MODE_LABELS,
                index=_default_idx,
                key=f"mode_{ver}",
                label_visibility="collapsed",
            )
            new_mode = _LABEL_TO_MODE.get(new_mode_label)  # None si "—"

            # Nouveau poids — visible uniquement en mode A/B, pré-rempli
            new_weight = None
            if new_mode == "ab_test":
                new_weight = row_cols[4].number_input(
                    "Poids",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    value=float(weight_current if weight_current is not None else 0.5),
                    key=f"weight_{ver}",
                    label_visibility="collapsed",
                )
            else:
                row_cols[4].markdown("—")

            configs[ver] = {"mode": new_mode, "weight": new_weight, "current": v}

        # Somme des poids A/B — inclut les versions sélectionnées ET celles inchangées déjà en A/B
        total_weight = sum(
            cfg["weight"]
            for cfg in configs.values()
            if cfg["mode"] == "ab_test" and cfg["weight"] is not None
        )
        for cfg in configs.values():
            if cfg["mode"] is None and cfg["current"].get("deployment_mode") == "ab_test":
                total_weight += cfg["current"].get("traffic_weight") or 0.0

        weight_color = "🟢" if total_weight <= 1.0 else "🔴"
        st.markdown(f"**Somme des poids A/B :** {weight_color} `{total_weight:.2f}` / 1.0")
        if total_weight > 1.0:
            st.warning("⚠️ La somme des poids dépasse 1.0 — corrigez avant d'appliquer.")

        if st.button("✅ Appliquer la configuration", type="primary", disabled=total_weight > 1.0):
            errors = []
            updated = 0
            for ver, cfg in configs.items():
                # Ignorer "—" (aucun changement demandé)
                if cfg["mode"] is None:
                    continue
                # Ignorer si mode et poids identiques à l'état actuel
                cur = cfg["current"]
                if (
                    cfg["mode"] == cur.get("deployment_mode")
                    and cfg["weight"] == cur.get("traffic_weight")
                ):
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
                    st.toast(err, icon="❌")
            if updated:
                st.toast(f"{updated} version(s) mise(s) à jour.", icon="✅")
                st.cache_data.clear()
                st.rerun()

else:
    st.info("🔒 La configuration A/B est réservée aux administrateurs.")

st.divider()

# ===========================================================================
# SECTION 2 — Dashboard de comparaison
# ===========================================================================
st.subheader("📊 Comparaison des versions")

_cmp_c1, _cmp_c2, _cmp_c3 = st.columns([2, 2, 3])
_ab_start = _cmp_c1.date_input(
    "Date début", value=date.today() - timedelta(days=30), key="ab_start_date"
)
_ab_end = _cmp_c2.date_input("Date fin", value=date.today(), key="ab_end_date")
days = max((_ab_end - _ab_start).days, 1)

_METRIC_OPTIONS = {
    "Auto (sélection intelligente)": None,
    "Taux d'erreur — Chi-²":         "error_rate",
    "MAE prédiction — Mann-Whitney":  "mae",
    "Latence réponse — Mann-Whitney": "response_time_ms",
}
_sig_metric_label = _cmp_c3.selectbox(
    "Métrique du test de significativité",
    list(_METRIC_OPTIONS.keys()),
    key="ab_sig_metric",
)
_sig_metric = _METRIC_OPTIONS[_sig_metric_label]

try:
    ab_data = client.get_ab_comparison(selected_model, days=days, metric=_sig_metric)
    versions_stats = ab_data.get("versions", [])
    ab_significance = ab_data.get("ab_significance")
except Exception as e:
    st.error(f"Impossible de charger les données de comparaison : {e}")
    versions_stats = []
    ab_significance = None

if not versions_stats:
    st.info("Aucune prédiction enregistrée pour ce modèle sur la période sélectionnée.")
else:
    # --- Tableau de comparaison ---
    _BADGE_CMP = {"ab_test": "🟠 A/B", "shadow": "🟣 Shadow", "production": "🟢 Prod"}
    _vfm_lookup = {v["version"]: v for v in versions_for_model}

    _rows = []
    for vs in versions_stats:
        ver = vs["version"]
        meta = _vfm_lookup.get(ver, {})
        mode = vs.get("deployment_mode") or ""
        weight = vs.get("traffic_weight")
        created_raw = meta.get("created_at") or meta.get("upload_date")
        try:
            created_str = datetime.fromisoformat(
                created_raw.replace("Z", "+00:00")
            ).strftime("%Y-%m-%d") if created_raw else "—"
        except Exception:
            created_str = str(created_raw)[:10] if created_raw else "—"

        _rows.append({
            "Version":       ver,
            "Mode":          _BADGE_CMP.get(mode, f"⚪ {mode or '—'}"),
            "Poids":         f"{weight:.0%}" if weight is not None else "—",
            "Algorithme":    meta.get("algorithm") or "—",
            "Créé le":       created_str,
            "Créateur":      meta.get("creator_username") or "—",
            "Préd. (prod)":  vs.get("total_predictions", 0),
            "Shadow":        vs.get("shadow_predictions", 0),
            "Err. (%)":      round(vs.get("error_rate", 0) * 100, 2) if vs.get("error_rate") is not None else None,
            "Lat. avg (ms)": round(vs["avg_response_time_ms"], 1) if vs.get("avg_response_time_ms") is not None else None,
            "Lat. p95 (ms)": round(vs["p95_response_time_ms"], 1) if vs.get("p95_response_time_ms") is not None else None,
            "Concordance":   round(vs["agreement_rate"] * 100, 1) if vs.get("agreement_rate") is not None else None,
            "Accuracy":      meta.get("accuracy"),
            "F1":            meta.get("f1_score"),
            "R²":            meta.get("r2_score"),
            "RMSE":          meta.get("rmse"),
        })

    _all_col_config = {
        "Version":       st.column_config.TextColumn("Version"),
        "Mode":          st.column_config.TextColumn("Mode"),
        "Poids":         st.column_config.TextColumn("Poids", help="Part de trafic allouée à cette version."),
        "Algorithme":    st.column_config.TextColumn("Algorithme"),
        "Créé le":       st.column_config.TextColumn("Créé le"),
        "Créateur":      st.column_config.TextColumn("Créateur"),
        "Préd. (prod)":  st.column_config.NumberColumn("Préd. (prod)", help=METRIC_HELP.get("predictions_prod", "Prédictions hors shadow.")),
        "Shadow":        st.column_config.NumberColumn("Shadow", help=METRIC_HELP.get("shadow_predictions", "Prédictions shadow (non exposées au client).")),
        "Err. (%)":      st.column_config.NumberColumn("Err. (%)", format="%.2f %%", help=METRIC_HELP.get("taux_erreur", "Taux d'erreur API sur la période.")),
        "Lat. avg (ms)": st.column_config.NumberColumn("Lat. avg (ms)", format="%.1f", help=METRIC_HELP.get("latence_avg", "Latence moyenne de réponse.")),
        "Lat. p95 (ms)": st.column_config.NumberColumn("Lat. p95 (ms)", format="%.1f", help=METRIC_HELP.get("latence_p95", "95e percentile de latence.")),
        "Concordance":   st.column_config.NumberColumn("Concordance (%)", format="%.1f %%", help=METRIC_HELP.get("concordance_shadow", "Taux d'accord entre shadow et prod.")),
        "Accuracy":      st.column_config.NumberColumn("Accuracy", format="%.3f", help="Accuracy sur le jeu de test à l'entraînement."),
        "F1":            st.column_config.NumberColumn("F1", format="%.3f", help="F1-score sur le jeu de test à l'entraînement."),
        "R²":            st.column_config.NumberColumn("R²", format="%.3f", help="Coefficient de détermination (régression)."),
        "RMSE":          st.column_config.NumberColumn("RMSE", format="%.4f", help="Root Mean Squared Error (régression)."),
    }

    # Colonnes toujours visibles
    _base_cols = ["Version", "Mode", "Algorithme", "Créé le", "Créateur",
                  "Préd. (prod)", "Shadow"]
    # Colonnes de métriques : masquées si toutes les valeurs sont null
    _metric_cols = ["Err. (%)", "Lat. avg (ms)", "Lat. p95 (ms)", "Concordance",
                    "Accuracy", "F1", "R²", "RMSE"]

    _df = pd.DataFrame(_rows)
    _visible_metrics = [c for c in _metric_cols if _df[c].notna().any()]
    _df = _df[_base_cols + _visible_metrics]
    _col_config = {k: v for k, v in _all_col_config.items() if k in _df.columns}

    st.dataframe(_df, width="stretch", hide_index=True, column_config=_col_config)

    # ===========================================================================
    # Bloc significativité statistique
    # ===========================================================================
    st.divider()
    st.subheader("🔬 Significativité statistique")

    if ab_significance is None:
        _no_data_reasons = {
            "error_rate":       "le Chi-² requiert au moins une erreur observée dans l'un des groupes",
            "mae":              "Mann-Whitney sur MAE requiert des résidus de prédiction (modèle de régression avec observed-results)",
            "response_time_ms": "Mann-Whitney sur la latence requiert au moins 2 prédictions par version",
        }
        if _sig_metric:
            st.warning(
                f"⚠️ Pas assez de données pour la métrique **{_sig_metric_label}** "
                f"sur cette période ({_no_data_reasons.get(_sig_metric, '')})."
            )
        else:
            st.info(
                "💡 Le test de significativité sera disponible dès que deux versions "
                "auront accumulé des prédictions sur la période sélectionnée."
            )
    else:
        sig = ab_significance
        is_significant = sig.get("significant", False)
        p_value = sig.get("p_value", 1.0)
        confidence = sig.get("confidence_level", 0.95)
        winner = sig.get("winner")
        metric = sig.get("metric", "")
        test = sig.get("test", "")
        # Avertir si la métrique réellement utilisée diffère de celle demandée
        if _sig_metric and metric != _sig_metric:
            st.warning(
                f"⚠️ La métrique **{_sig_metric_label}** n'était pas disponible — "
                f"test effectué sur : **{metric}**."
            )
        min_needed = sig.get("min_samples_needed", 0)
        current_samples: dict = sig.get("current_samples", {})

        # --- Bannière verdict ---
        if is_significant:
            if winner:
                st.success(
                    f"✅ **Différence statistiquement significative** — "
                    f"la version **{winner}** est meilleure sur la métrique *{metric}*"
                )
            else:
                st.success(
                    "✅ **Différence statistiquement significative** entre les deux versions."
                )
        else:
            st.warning(
                f"⚠️ **Différence non significative** — "
                f"impossible de conclure avec les données actuelles "
                f"(p = {p_value:.4f}, seuil = {1 - confidence:.2f})"
            )

        # --- Bandeau de promotion du gagnant (admin uniquement) ---
        if is_significant and winner and is_admin:
            with st.container(border=True):
                promo_col1, promo_col2 = st.columns([3, 1])
                promo_col1.markdown(
                    f"🟢 **Gagnant identifié : `{winner}`** — "
                    f"p-value = **{p_value:.4f}** · confiance {confidence:.0%}"
                )
                if promo_col2.button(
                    "🏆 Promouvoir en production",
                    type="primary",
                    key="ab_promote_winner",
                    width='stretch',
                ):
                    errors_promo = []
                    try:
                        client.update_model(selected_model, winner, {"is_production": True})
                    except Exception as e:
                        errors_promo.append(f"Promotion `{winner}` : {e}")
                    for v in versions_for_model:
                        if v["version"] == winner:
                            continue
                        try:
                            client.update_model(
                                selected_model,
                                v["version"],
                                {"deployment_mode": "production", "traffic_weight": 1.0},
                            )
                        except Exception as e:
                            errors_promo.append(f"Version `{v['version']}` : {e}")
                    if errors_promo:
                        for err in errors_promo:
                            st.error(err)
                    else:
                        st.success(f"✅ `{winner}` promu en production avec succès.")
                        st.cache_data.clear()
                        st.rerun()

        # --- KPI de significativité ---
        sig_cols = st.columns(4)

        test_label = {"chi2": "Chi-²", "mann_whitney_u": "Mann-Whitney U"}.get(test, test)
        metric_label = {"error_rate": "Taux d'erreur", "response_time_ms": "Latence (ms)"}.get(
            metric, metric
        )

        sig_cols[0].metric("Test statistique", test_label, help=METRIC_HELP["test_statistique"])
        sig_cols[1].metric("Métrique analysée", metric_label, help=METRIC_HELP["metrique_analysee"])
        sig_cols[2].metric(
            "p-value",
            f"{p_value:.4f}",
            delta=f"seuil {1 - confidence:.2f}",
            delta_color="off",
            help=METRIC_HELP["p_value"],
        )
        sig_cols[3].metric("Niveau de confiance", f"{confidence:.0%}", help=METRIC_HELP["niveau_confiance"])

        # --- Jauge p-value ---
        threshold = 1.0 - confidence
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=p_value,
                number={"valueformat": ".4f", "suffix": ""},
                gauge={
                    "axis": {"range": [0, 0.2], "tickformat": ".2f"},
                    "bar": {"color": "#16a34a" if is_significant else "#f59e0b"},
                    "steps": [
                        {"range": [0, threshold], "color": "#dcfce7"},
                        {"range": [threshold, 0.2], "color": "#fef3c7"},
                    ],
                    "threshold": {
                        "line": {"color": "#dc2626", "width": 3},
                        "thickness": 0.85,
                        "value": threshold,
                    },
                },
                title={"text": f"p-value (seuil α = {threshold:.2f})"},
            )
        )
        fig_gauge.update_layout(height=240, margin=dict(t=40, b=10, l=20, r=20))

        # --- Jauge puissance (échantillons actuels vs nécessaires) ---
        total_current = sum(current_samples.values())
        total_needed = min_needed * len(current_samples) if min_needed > 0 else total_current
        power_pct = min(total_current / total_needed, 1.0) * 100 if total_needed > 0 else 100.0

        fig_power = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=power_pct,
                number={"valueformat": ".0f", "suffix": "%"},
                delta={
                    "reference": 100,
                    "valueformat": ".0f",
                    "suffix": "%",
                    "increasing": {"color": "#16a34a"},
                },
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#16a34a" if power_pct >= 100 else "#f59e0b"},
                    "steps": [
                        {"range": [0, 50], "color": "#fee2e2"},
                        {"range": [50, 80], "color": "#fef3c7"},
                        {"range": [80, 100], "color": "#dcfce7"},
                    ],
                },
                title={"text": "Puissance statistique (données actuelles vs requises)"},
            )
        )
        fig_power.update_layout(height=240, margin=dict(t=40, b=10, l=20, r=20))

        gauge_col1, gauge_col2 = st.columns(2)
        with gauge_col1:
            st.plotly_chart(fig_gauge, width='stretch')
        with gauge_col2:
            st.plotly_chart(fig_power, width='stretch')

        # --- Tableau des échantillons ---
        if current_samples:
            st.markdown("**Observations disponibles par version**")
            sample_rows = []
            for ver, n in current_samples.items():
                enough = n >= min_needed if min_needed > 0 else True
                sample_rows.append(
                    {
                        "Version": ver,
                        "Observations actuelles": n,
                        "Minimum recommandé": min_needed if min_needed > 0 else "—",
                        "Suffisant ?": (
                            "✅ Oui" if enough else f"❌ Non ({min_needed - n} manquantes)"
                        ),
                    }
                )
            st.dataframe(
                pd.DataFrame(sample_rows),
                width='stretch',
                hide_index=True,
                column_config={
                    "Version": st.column_config.TextColumn(
                        "Version",
                        help="Numéro de version du modèle participant au test A/B.",
                    ),
                    "Observations actuelles": st.column_config.NumberColumn(
                        "Observations actuelles",
                        help="Nombre de prédictions enregistrées pour cette version sur la période sélectionnée.",
                    ),
                    "Minimum recommandé": st.column_config.TextColumn(
                        "Minimum recommandé",
                        help="Nombre minimal d'observations nécessaires pour que le test soit statistiquement fiable (puissance 80 %).",
                    ),
                    "Suffisant ?": st.column_config.TextColumn(
                        "Suffisant ?",
                        help="✅ Oui : assez de données pour conclure. ❌ Non : continuez à accumuler des prédictions.",
                    ),
                },
            )

        # --- Recommandation finale ---
        with st.expander("💡 Comment interpréter ce résultat ?"):
            st.markdown(f"""
**Test utilisé :** {test_label}
- **Chi-²** : compare les taux d'erreur entre deux versions via un tableau de contingence succès/erreur.
- **Mann-Whitney U** : compare les distributions de latence (utilisé en fallback si aucune erreur n'est observée).

**p-value = {p_value:.4f}**
- Si `p < {threshold:.2f}` → la différence observée a moins de {(1 - confidence):.0%} de chances d'être due au hasard.
- Si `p ≥ {threshold:.2f}` → pas assez de données ou pas de différence réelle.

**Minimum recommandé : {min_needed} observations/version** (pour une puissance statistique de 80 %)
- Calculé via l'effet de taille de Cohen h (taux d'erreur) ou Cohen d (latence).

{"✅ **Conclusion : vous pouvez promouvoir** `" + winner + "` **en production en toute confiance.**" if is_significant and winner else "⚠️ **Conclusion : continuez à accumuler des données** avant de prendre une décision de promotion."}
""")

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
            st.plotly_chart(fig_traffic, width='stretch')
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
            st.plotly_chart(fig_dist, width='stretch')
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
        st.plotly_chart(fig_agree, width='stretch')
    else:
        st.info(
            "💡 Le taux de concordance shadow/production sera calculé automatiquement "
            "dès que des prédictions avec `id_obs` seront enregistrées pour les deux versions."
        )

st.divider()

# ===========================================================================
# SECTION 3 — Analyse shadow enrichie
# ===========================================================================
st.subheader("🔮 Analyse shadow enrichie")
st.caption(
    "Métriques différentielles entre la version shadow et la version production "
    "sur les prédictions avec `id_obs` commun."
)

try:
    shadow_data = client.get_shadow_comparison(selected_model, period_days=days)
    shadow_available = True
except Exception:
    shadow_data = None
    shadow_available = False

if not shadow_available or shadow_data is None:
    st.info("Impossible de charger les données shadow.")
elif shadow_data.get("shadow_version") is None:
    st.info("Aucune version en mode shadow pour ce modèle.")
else:
    sv = shadow_data["shadow_version"]
    pv = shadow_data["production_version"]
    n_comparable = shadow_data["n_comparable"]
    agreement_rate = shadow_data.get("agreement_rate")
    conf_delta = shadow_data.get("shadow_confidence_delta")
    lat_delta = shadow_data.get("shadow_latency_delta_ms")
    shadow_acc = shadow_data.get("shadow_accuracy")
    prod_acc = shadow_data.get("production_accuracy")
    accuracy_available = shadow_data.get("accuracy_available", False)
    recommendation = shadow_data.get("recommendation", "insufficient_data")

    _rec_badge = {
        "shadow_better": ("🟢", "Le shadow semble meilleur"),
        "production_better": ("🔴", "La production reste meilleure"),
        "equivalent": ("🟡", "Performances équivalentes"),
        "insufficient_data": ("⚪", "Données insuffisantes pour conclure"),
    }.get(recommendation, ("⚪", recommendation))

    st.markdown(
        f"**Shadow `{sv}`** vs **Production `{pv}`** — "
        f"{_rec_badge[0]} *{_rec_badge[1]}*"
    )

    _cols = st.columns(5)
    _cols[0].metric("Paires comparables", n_comparable)
    _cols[1].metric(
        "Accord des prédictions",
        f"{agreement_rate:.1%}" if agreement_rate is not None else "—",
        help="Fraction des id_obs où shadow et production prédisent la même valeur.",
    )
    _cols[2].metric(
        "Δ Confiance",
        f"{conf_delta:+.3f}" if conf_delta is not None else "—",
        delta_color="normal" if conf_delta is not None else "off",
        delta=f"{conf_delta:+.3f}" if conf_delta is not None else None,
        help="max_confidence shadow − max_confidence production (positif = shadow plus confiant).",
    )
    _cols[3].metric(
        "Δ Latence (ms)",
        f"{lat_delta:+.1f}" if lat_delta is not None else "—",
        delta=f"{lat_delta:+.1f}" if lat_delta is not None else None,
        delta_color="inverse" if lat_delta is not None else "off",
        help="Latence moyenne shadow − latence moyenne production (négatif = shadow plus rapide).",
    )

    if accuracy_available and shadow_acc is not None and prod_acc is not None:
        _cols[4].metric(
            "Accuracy shadow vs prod",
            f"{shadow_acc:.1%} / {prod_acc:.1%}",
            help="Accuracy calculée sur les paires ayant un observed_result.",
        )
    else:
        _cols[4].metric(
            "Accuracy",
            "—",
            help="Non disponible : aucun observed_result pour les id_obs comparés.",
        )

    if agreement_rate is not None and n_comparable > 0:
        fig_agree_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=agreement_rate * 100,
                number={"valueformat": ".1f", "suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {
                        "color": (
                            "#16a34a"
                            if agreement_rate >= 0.9
                            else "#f59e0b" if agreement_rate >= 0.7 else "#dc2626"
                        )
                    },
                    "steps": [
                        {"range": [0, 70], "color": "#fee2e2"},
                        {"range": [70, 90], "color": "#fef3c7"},
                        {"range": [90, 100], "color": "#dcfce7"},
                    ],
                    "threshold": {
                        "line": {"color": "#7c3aed", "width": 3},
                        "thickness": 0.85,
                        "value": 90,
                    },
                },
                title={"text": "Accord shadow / production (%)"},
            )
        )
        fig_agree_gauge.update_layout(height=240, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig_agree_gauge, width='stretch')

    if recommendation == "shadow_better" and is_admin:
        with st.container(border=True):
            promo_col1, promo_col2 = st.columns([3, 1])
            promo_col1.markdown(
                f"🟢 **Le shadow `{sv}` semble meilleur que la production `{pv}`** — "
                "promouvoir pour qu'il devienne la version active ?"
            )
            if promo_col2.button(
                "🚀 Promouvoir le shadow",
                type="primary",
                key="shadow_promote_btn",
                width='stretch',
            ):
                errors_shadow_promo = []
                try:
                    client.update_model(selected_model, sv, {"is_production": True})
                except Exception as exc:
                    errors_shadow_promo.append(f"Promotion `{sv}` : {exc}")
                if errors_shadow_promo:
                    for err in errors_shadow_promo:
                        st.error(err)
                else:
                    st.success(f"✅ `{sv}` promu en production avec succès.")
                    st.cache_data.clear()
                    st.rerun()

st.divider()

# ===========================================================================
# SECTION 4 — Log des prédictions shadow récentes
# ===========================================================================
st.subheader("📋 Prédictions shadow récentes")

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
                "Timestamp": (
                    pd.to_datetime(p.get("timestamp")).strftime("%Y-%m-%d %H:%M:%S")
                    if p.get("timestamp")
                    else "—"
                ),
                "Version": p.get("model_version") or "—",
                "id_obs": p.get("id_obs") or "—",
                "Résultat": str(p.get("prediction_result", "")),
                "Latence (ms)": (
                    f"{p['response_time_ms']:.1f}" if p.get("response_time_ms") is not None else "—"
                ),
                "Statut": "✅" if p.get("status") == "success" else "❌",
            }
        )

    st.caption(f"{len(shadow_preds)} prédiction(s) shadow")
    st.dataframe(
        pd.DataFrame(shadow_rows),
        width='stretch',
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn(
                "ID",
                help="Identifiant unique de la prédiction en base de données.",
            ),
            "Timestamp": st.column_config.TextColumn(
                "Timestamp",
                help="Date et heure à laquelle la prédiction shadow a été effectuée.",
            ),
            "Version": st.column_config.TextColumn(
                "Version",
                help="Version du modèle qui a produit cette prédiction shadow.",
            ),
            "id_obs": st.column_config.TextColumn(
                "id_obs",
                help="Identifiant de l'observation, fourni par le client appelant. Permet d'apparier une prédiction shadow à sa contrepartie production.",
            ),
            "Résultat": st.column_config.TextColumn(
                "Résultat",
                help="Valeur prédite par la version shadow. Non retournée au client — sert uniquement à la comparaison interne.",
            ),
            "Latence (ms)": st.column_config.TextColumn(
                "Latence (ms)",
                help="Temps de calcul de la prédiction shadow en millisecondes.",
            ),
            "Statut": st.column_config.TextColumn(
                "Statut",
                help="✅ Succès ou ❌ Erreur lors du calcul de la prédiction.",
            ),
        },
    )
