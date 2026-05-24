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
from utils.i18n import t

st.set_page_config(page_title=t("ab_testing.page_title"), page_icon="🧪", layout="wide")
require_auth()

col_title, col_refresh = st.columns([8, 1])
col_title.title(t("ab_testing.title"))
if col_refresh.button(t("ab_testing.btn_refresh"), key="ab_refresh", width='stretch'):
    st.cache_data.clear()
    st.rerun()
st.caption(t("ab_testing.caption"))

client = get_client()
is_admin = st.session_state.get("is_admin", False)

# --- Charger la liste des modèles ---
try:
    all_models = get_models_cached(
        st.session_state.get("api_url"), st.session_state.get("api_token")
    )
except Exception as e:
    st.error(t("ab_testing.load_error", error=e))
    st.stop()

if not all_models:
    st.info(t("ab_testing.no_models"))
    st.stop()

# Grouper les versions par nom de modèle
model_groups: dict = defaultdict(list)
for m in all_models:
    model_groups[m["name"]].append(m)

model_names = sorted(model_groups.keys())
_ab_col1, _ab_col2 = st.columns([2, 3])
ab_search = _ab_col1.text_input(t("ab_testing.filter_name"), key="ab_model_search", placeholder=t("ab_testing.filter_placeholder"))
ab_filtered = [n for n in model_names if ab_search.lower() in n.lower()] if ab_search else model_names
selected_model = _ab_col2.selectbox(t("ab_testing.model_select"), ab_filtered or model_names, key="ab_model_select")
versions_for_model = model_groups[selected_model]

st.divider()

# ===========================================================================
# SECTION 1 — Configuration du déploiement (admin seulement)
# ===========================================================================
if is_admin:
    with st.expander(t("ab_testing.config.expander"), expanded=True):  # premier expander — ouvert par défaut
        st.markdown(t("ab_testing.config.description"))

        # Mapping label ↔ valeur API
        # "—" = aucun changement (utilisé quand le mode courant n'est pas dans la liste)
        _MODE_TO_LABEL = {
            "ab_test":    "🟠 A/B",
            "shadow":     "🟣 Shadow",
            "production": "🟢 Prod",
        }
        _LABEL_TO_MODE = {v: k for k, v in _MODE_TO_LABEL.items()}
        _MODE_LABELS = [t("ab_testing.config.mode_no_change")] + list(_MODE_TO_LABEL.values())
        configs: dict = {}

        cols_header = st.columns([2, 2, 1, 2, 1])
        cols_header[0].markdown(t("ab_testing.config.col_version"))
        cols_header[1].markdown(t("ab_testing.config.col_mode_current"))
        cols_header[2].markdown(t("ab_testing.config.col_weight_current"))
        cols_header[3].markdown(t("ab_testing.config.col_mode_new"))
        cols_header[4].markdown(t("ab_testing.config.col_weight_new"))

        for v in versions_for_model:
            ver = v["version"]
            mode_current = v.get("deployment_mode") or ""
            weight_current = v.get("traffic_weight")

            row_cols = st.columns([2, 2, 1, 2, 1])
            row_cols[0].markdown(f"`{ver}`")

            # Mode actuel (badge)
            row_cols[1].markdown(_MODE_TO_LABEL.get(mode_current, "⚪ —"))

            # Poids actuel
            row_cols[2].markdown(
                f"`{weight_current:.0%}`" if weight_current is not None else t("ab_testing.config.weight_no_change")
            )

            # Nouveau mode — badge pré-rempli ; "—" si mode inconnu ou non défini
            _cur_label = _MODE_TO_LABEL.get(mode_current)   # None si mode inconnu
            _default_idx = _MODE_LABELS.index(_cur_label) if _cur_label else 0  # 0 = "—"
            new_mode_label = row_cols[3].selectbox(
                "Mode",
                _MODE_LABELS,
                index=_default_idx,
                key=f"mode_{selected_model}_{ver}",
                label_visibility="collapsed",
            )
            new_mode = _LABEL_TO_MODE.get(new_mode_label)  # None si "—"

            # Nouveau poids — visible uniquement en mode A/B, pré-rempli
            new_weight = None
            if new_mode == "ab_test":
                _weight_key = f"weight_{selected_model}_{ver}"
                # Pré-remplir avec le poids actuel si l'état n'existe pas encore pour ce modèle
                if _weight_key not in st.session_state:
                    st.session_state[_weight_key] = float(weight_current if weight_current is not None else 0.5)
                new_weight = row_cols[4].number_input(
                    "Poids",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    key=_weight_key,
                    label_visibility="collapsed",
                )
            else:
                row_cols[4].markdown(t("ab_testing.config.weight_no_change"))

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
        st.markdown(t("ab_testing.config.weight_sum", badge=weight_color, value=f"{total_weight:.2f}"))
        if total_weight > 1.0:
            st.warning(t("ab_testing.config.weight_overflow"))

        if st.button(t("ab_testing.config.apply_btn"), type="primary", disabled=total_weight > 1.0):
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
                st.toast(t("ab_testing.config.updated", count=updated), icon="✅")
                st.cache_data.clear()
                st.rerun()

else:
    st.info(t("ab_testing.config.admin_only"))

st.divider()

# ===========================================================================
# SECTION 2 — Dashboard de comparaison
# ===========================================================================
st.subheader(t("ab_testing.comparison.subheader"))

_cmp_c1, _cmp_c2 = st.columns(2)
_ab_start = _cmp_c1.date_input(
    t("ab_testing.comparison.date_start"), value=date.today() - timedelta(days=30), key="ab_start_date"
)
_ab_end = _cmp_c2.date_input(t("ab_testing.comparison.date_end"), value=date.today(), key="ab_end_date")
days = max((_ab_end - _ab_start).days, 1)

# Lire la métrique depuis session_state AVANT l'appel API
# (Streamlit y stocke la nouvelle valeur du widget avant le rerun)
_METRIC_OPTIONS = {
    t("ab_testing.ab_test.metric_auto"):          None,
    t("ab_testing.ab_test.metric_error_rate"):    "error_rate",
    t("ab_testing.ab_test.metric_mae"):           "mae",
    t("ab_testing.ab_test.metric_response_time"): "response_time_ms",
}
_sig_metric_label = st.session_state.get("ab_sig_metric", t("ab_testing.ab_test.metric_auto"))
_sig_metric = _METRIC_OPTIONS.get(_sig_metric_label)

try:
    ab_data = client.get_ab_comparison(selected_model, days=days, metric=_sig_metric)
    versions_stats = ab_data.get("versions", [])
    ab_significance = ab_data.get("ab_significance")
except Exception as e:
    st.error(t("ab_testing.comparison.load_error", error=e))
    versions_stats = []
    ab_significance = None

def _output_summary(dist: dict, meta: dict) -> "str | None":
    """
    Résumé compact des prédictions d'une version.
    - Régression      → moyenne pondérée des valeurs prédites  (ex: µ = 3.42)
    - Classification  → répartition des classes top-3 avec labels réels
                        (ex: setosa: 45% · versicolor: 33%)
    Retourne None si la distribution est vide.
    """
    try:
        if not dist:
            return None
        total = sum(dist.values())
        if total == 0:
            return None

        # Détecter le type de tâche depuis les métadonnées du modèle
        task = meta.get("model_task") or ""
        classes_list: list = meta.get("classes") or []
        if not task:
            tm = meta.get("training_metrics") or {}
            if classes_list:
                task = "classification"
            elif tm.get("r2") is not None or tm.get("rmse") is not None or tm.get("mae") is not None:
                # Fallback : training_metrics contient des métriques de régression
                task = "regression"
            elif meta.get("accuracy") is not None or meta.get("f1_score") is not None:
                task = "classification"

        # Heuristique de secours : clés numériques nombreuses → régression
        if not task:
            try:
                float_vals = [float(k) for k in dist.keys()]
                task = "regression" if len(float_vals) > 5 else "classification"
            except (ValueError, TypeError):
                task = "classification"

        if "regression" in task:
            try:
                float_keys = [float(k) for k in dist.keys()]
                mean_val = sum(cnt * val for cnt, val in zip(dist.values(), float_keys)) / total
                return f"µ = {mean_val:.2f}"
            except (ValueError, TypeError):
                pass  # fall through to classification display

        # Classification — mapper les indices entiers vers les labels si disponibles
        def _resolve_label(key: str) -> str:
            if classes_list:
                try:
                    idx = int(key)
                    if 0 <= idx < len(classes_list):
                        return str(classes_list[idx])
                except (ValueError, TypeError):
                    pass
            return key

        # Top-3 classes triées par fréquence décroissante (exclure les classes à 0)
        sorted_classes = [(cls, cnt) for cls, cnt in sorted(dist.items(), key=lambda x: -x[1]) if cnt > 0]
        parts = [f"{_resolve_label(cls)}: {cnt / total:.0%}" for cls, cnt in sorted_classes[:3]]
        if len(sorted_classes) > 3:
            parts.append("…")
        return " · ".join(parts)
    except Exception:
        return None


if not versions_stats:
    st.info(t("ab_testing.comparison.no_predictions"))
else:
    # --- Tableau de comparaison ---
    _BADGE_CMP = {"ab_test": "🟠 A/B", "shadow": "🟣 Shadow", "production": "🟢 Production"}
    _vfm_lookup = {v["version"]: v for v in versions_for_model}

    # Détecter si le modèle sélectionné est de régression (depuis n'importe quelle version)
    def _model_is_regression(vfm: dict) -> bool:
        """Retourne True si model_task = 'regression', avec fallback sur training_metrics."""
        for v in vfm.values():
            mt = v.get("model_task") or ""
            if "regression" in mt:
                return True
            tm = v.get("training_metrics") or {}
            if any(tm.get(k) is not None for k in ("r2", "rmse", "mae")):
                return True
        return False

    _is_regression = _model_is_regression(_vfm_lookup)

    # Resolve column name strings once
    _col_version = t("ab_testing.comparison.col_version")
    _col_mode = t("ab_testing.comparison.col_mode")
    _col_weight = t("ab_testing.comparison.col_weight")
    _col_algorithm = t("ab_testing.comparison.col_algorithm")
    _col_created = t("ab_testing.comparison.col_created")
    _col_creator = t("ab_testing.comparison.col_creator")
    _col_pred_prod = t("ab_testing.comparison.col_pred_prod")
    _col_shadow = t("ab_testing.comparison.col_shadow")
    _col_output = t("ab_testing.comparison.col_output")
    _col_err_pct = t("ab_testing.comparison.col_err_pct")
    _col_lat_avg = t("ab_testing.comparison.col_lat_avg")
    _col_lat_p95 = t("ab_testing.comparison.col_lat_p95")
    _col_concordance = t("ab_testing.comparison.col_concordance")
    _col_accuracy = t("ab_testing.comparison.col_accuracy")
    _col_f1 = t("ab_testing.comparison.col_f1")
    _col_r2 = t("ab_testing.comparison.col_r2")
    _col_rmse = t("ab_testing.comparison.col_rmse")

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
            _col_version:     ver,
            _col_mode:        _BADGE_CMP.get(mode, "⚪ —"),
            _col_weight:      f"{weight:.0%}" if weight is not None else "—",
            _col_algorithm:   meta.get("algorithm") or "—",
            _col_created:     created_str,
            _col_creator:     meta.get("creator_username") or "—",
            _col_pred_prod:   vs.get("total_predictions", 0),
            _col_shadow:      vs.get("shadow_predictions", 0),
            _col_output:      _output_summary(vs.get("prediction_distribution", {}), meta),
            _col_err_pct:     round(vs.get("error_rate", 0) * 100, 2) if vs.get("error_rate") is not None else None,
            _col_lat_avg:     round(vs["avg_response_time_ms"], 1) if vs.get("avg_response_time_ms") is not None else None,
            _col_lat_p95:     round(vs["p95_response_time_ms"], 1) if vs.get("p95_response_time_ms") is not None else None,
            _col_concordance: round(vs["agreement_rate"] * 100, 1) if vs.get("agreement_rate") is not None else None,
            _col_accuracy:    meta.get("accuracy"),
            _col_f1:          meta.get("f1_score"),
            _col_r2:          meta.get("r2_score"),
            _col_rmse:        meta.get("rmse"),
        })

    _all_col_config = {
        _col_version:     st.column_config.TextColumn(_col_version),
        _col_mode:        st.column_config.TextColumn(_col_mode),
        _col_weight:      st.column_config.TextColumn(_col_weight, help=t("ab_testing.comparison.col_weight_help")),
        _col_algorithm:   st.column_config.TextColumn(_col_algorithm),
        _col_created:     st.column_config.TextColumn(_col_created),
        _col_creator:     st.column_config.TextColumn(_col_creator),
        _col_pred_prod:   st.column_config.NumberColumn(_col_pred_prod, help=t("metrics.predictions_prod")),
        _col_shadow:      st.column_config.NumberColumn(_col_shadow, help=t("metrics.shadow_predictions")),
        _col_output:      st.column_config.TextColumn(
            _col_output,
            help=t("ab_testing.comparison.col_output_help"),
        ),
        _col_err_pct:     st.column_config.NumberColumn(_col_err_pct, format="%.2f %%", help=t("metrics.taux_erreur")),
        _col_lat_avg:     st.column_config.NumberColumn(_col_lat_avg, format="%.1f", help=t("metrics.latence_avg")),
        _col_lat_p95:     st.column_config.NumberColumn(_col_lat_p95, format="%.1f", help=t("metrics.latence_p95")),
        _col_concordance: st.column_config.NumberColumn(_col_concordance, format="%.1f %%", help=t("metrics.concordance_shadow")),
        _col_accuracy:    st.column_config.NumberColumn(_col_accuracy, format="%.3f", help=t("ab_testing.comparison.col_accuracy_help")),
        _col_f1:          st.column_config.NumberColumn(_col_f1, format="%.3f", help=t("ab_testing.comparison.col_f1_help")),
        _col_r2:          st.column_config.NumberColumn(_col_r2, format="%.3f", help=t("ab_testing.comparison.col_r2_help")),
        _col_rmse:        st.column_config.NumberColumn(_col_rmse, format="%.4f", help=t("ab_testing.comparison.col_rmse_help")),
    }

    # Colonnes toujours visibles
    _base_cols = [_col_version, _col_mode, _col_algorithm, _col_created, _col_creator,
                  _col_pred_prod, _col_shadow]
    # Colonnes de métriques : masquées si toutes les valeurs sont null/None
    _metric_cols = [_col_output, _col_err_pct, _col_lat_avg, _col_lat_p95, _col_concordance,
                    _col_accuracy, _col_f1, _col_r2, _col_rmse]

    _df = pd.DataFrame(_rows)
    _visible_metrics = [c for c in _metric_cols if _df[c].notna().any()]
    _df = _df[_base_cols + _visible_metrics]
    _col_config = {k: v for k, v in _all_col_config.items() if k in _df.columns}

    st.dataframe(_df, width="stretch", hide_index=True, column_config=_col_config)

    with st.expander(t("ab_testing.ab_test.expander"), expanded=False):
        # ===========================================================================
        # Bloc significativité statistique
        # ===========================================================================
        st.divider()
        _sig_col_title, _sig_col_select = st.columns([3, 2])
        _sig_col_title.subheader(t("ab_testing.ab_test.sig_title"))
        _sig_metric_label = _sig_col_select.selectbox(
            t("ab_testing.ab_test.metric_select_label"),
            list(_METRIC_OPTIONS.keys()),
            key="ab_sig_metric",
            label_visibility="collapsed",
        )

        if ab_significance is None:
            _no_data_reasons = {
                "error_rate":       t("ab_testing.ab_test.no_data_reason_error_rate"),
                "mae":              t("ab_testing.ab_test.no_data_reason_mae"),
                "response_time_ms": t("ab_testing.ab_test.no_data_reason_response_time"),
            }
            if _sig_metric:
                st.warning(
                    t("ab_testing.ab_test.no_data_metric", metric=_sig_metric_label, reason=_no_data_reasons.get(_sig_metric, ""))
                )
            else:
                st.info(t("ab_testing.ab_test.no_data_auto"))
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
                    t("ab_testing.ab_test.metric_fallback_warning", requested=_sig_metric_label, actual=metric)
                )
            min_needed = sig.get("min_samples_needed", 0)
            current_samples: dict = sig.get("current_samples", {})

            # --- Bannière verdict ---
            if is_significant:
                if winner:
                    st.success(t("ab_testing.ab_test.significant_with_winner", winner=winner, metric=metric))
                else:
                    st.success(t("ab_testing.ab_test.significant_no_winner"))
            else:
                st.warning(
                    t("ab_testing.ab_test.not_significant", p_value=f"{p_value:.4f}", threshold=f"{1 - confidence:.2f}")
                )

            # --- Bandeau de promotion du gagnant (admin uniquement) ---
            if is_significant and winner and is_admin:
                with st.container(border=True):
                    promo_col1, promo_col2 = st.columns([3, 1])
                    promo_col1.markdown(
                        t("ab_testing.ab_test.winner_banner", winner=winner, p_value=f"{p_value:.4f}", confidence=f"{confidence:.0%}")
                    )
                    if promo_col2.button(
                        t("ab_testing.ab_test.promote_btn"),
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
                            st.success(t("ab_testing.ab_test.promote_success", winner=winner))
                            st.cache_data.clear()
                            st.rerun()

            # --- KPI de significativité ---
            sig_cols = st.columns(4)

            test_label = {"chi2": "Chi-²", "mann_whitney_u": "Mann-Whitney U"}.get(test, test)
            metric_label = {"error_rate": "Taux d'erreur", "response_time_ms": "Latence (ms)"}.get(
                metric, metric
            )

            sig_cols[0].metric(t("ab_testing.ab_test.kpi_test"), test_label, help=t("metrics.test_statistique"))
            sig_cols[1].metric(t("ab_testing.ab_test.kpi_metric"), metric_label, help=t("metrics.metrique_analysee"))
            sig_cols[2].metric(
                t("ab_testing.ab_test.kpi_pvalue"),
                f"{p_value:.4f}",
                delta=t("ab_testing.ab_test.kpi_pvalue_delta", threshold=f"{1 - confidence:.2f}"),
                delta_color="off",
                help=t("metrics.p_value"),
            )
            sig_cols[3].metric(t("ab_testing.ab_test.kpi_confidence"), f"{confidence:.0%}", help=t("metrics.niveau_confiance"))

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
                    title={"text": t("ab_testing.ab_test.gauge_pvalue_title", threshold=f"{threshold:.2f}")},
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
                    title={"text": t("ab_testing.ab_test.gauge_power_title")},
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
                st.markdown(t("ab_testing.ab_test.samples_title"))
                _col_samp_version = t("ab_testing.ab_test.col_samples_version")
                _col_samp_current = t("ab_testing.ab_test.col_samples_current")
                _col_samp_min = t("ab_testing.ab_test.col_samples_min")
                _col_samp_ok = t("ab_testing.ab_test.col_samples_ok")
                sample_rows = []
                for ver, n in current_samples.items():
                    enough = n >= min_needed if min_needed > 0 else True
                    sample_rows.append(
                        {
                            _col_samp_version: ver,
                            _col_samp_current: n,
                            _col_samp_min:     min_needed if min_needed > 0 else "—",
                            _col_samp_ok:      (
                                t("ab_testing.ab_test.samples_enough")
                                if enough
                                else t("ab_testing.ab_test.samples_not_enough", missing=min_needed - n)
                            ),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(sample_rows),
                    width='stretch',
                    hide_index=True,
                    column_config={
                        _col_samp_version: st.column_config.TextColumn(
                            _col_samp_version,
                            help=t("ab_testing.ab_test.col_samples_version_help"),
                        ),
                        _col_samp_current: st.column_config.NumberColumn(
                            _col_samp_current,
                            help=t("ab_testing.ab_test.col_samples_current_help"),
                        ),
                        _col_samp_min: st.column_config.TextColumn(
                            _col_samp_min,
                            help=t("ab_testing.ab_test.col_samples_min_help"),
                        ),
                        _col_samp_ok: st.column_config.TextColumn(
                            _col_samp_ok,
                            help=t("ab_testing.ab_test.col_samples_ok_help"),
                        ),
                    },
                )

            # --- Recommandation finale ---
            with st.expander(t("ab_testing.ab_test.interp_expander")):
                st.markdown(
                    t(
                        "ab_testing.ab_test.interp_body",
                        test_label=test_label,
                        p_value=f"{p_value:.4f}",
                        threshold=f"{threshold:.2f}",
                        risk=f"{(1 - confidence):.0%}",
                        min_needed=min_needed,
                    )
                )
                if is_significant and winner:
                    st.markdown(t("ab_testing.ab_test.interp_conclusion_promote", winner=winner))
                else:
                    st.markdown(t("ab_testing.ab_test.interp_conclusion_wait"))

        st.divider()

        # Graphique : distribution du trafic par version
        col_chart1, col_chart2 = st.columns(2)

        _col_traffic_version = t("ab_testing.ab_test.traffic_col_version")
        _col_traffic_pred = t("ab_testing.ab_test.traffic_col_pred")

        with col_chart1:
            st.markdown(t("ab_testing.ab_test.traffic_title"))
            traffic_df = pd.DataFrame(
                [
                    {_col_traffic_version: vs["version"], _col_traffic_pred: vs["total_predictions"]}
                    for vs in versions_stats
                ]
            )
            if traffic_df[_col_traffic_pred].sum() > 0:
                fig_traffic = px.bar(
                    traffic_df,
                    x=_col_traffic_version,
                    y=_col_traffic_pred,
                    color=_col_traffic_version,
                    text_auto=True,
                )
                fig_traffic.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_traffic, width='stretch')
            else:
                st.info(t("ab_testing.ab_test.traffic_no_prod"))

        _col_dist_version = t("ab_testing.ab_test.traffic_col_version")
        _col_dist_predicted = t("ab_testing.ab_test.dist_predicted_value")
        _col_dist_label = t("ab_testing.ab_test.dist_label")
        _col_dist_count = t("ab_testing.ab_test.dist_count")

        with col_chart2:
            if _is_regression:
                st.markdown(t("ab_testing.ab_test.dist_regression_title"))
                # Régression : histogramme des valeurs prédites (expand chaque valeur × count)
                hist_rows = []
                for vs in versions_stats:
                    for label, count in vs.get("prediction_distribution", {}).items():
                        try:
                            val = float(label)
                        except (ValueError, TypeError):
                            continue
                        for _ in range(count):
                            hist_rows.append({_col_dist_version: vs["version"], _col_dist_predicted: val})
                if hist_rows:
                    hist_df = pd.DataFrame(hist_rows)
                    fig_dist = px.histogram(
                        hist_df,
                        x=_col_dist_predicted,
                        color=_col_dist_version,
                        barmode="overlay",
                        opacity=0.6,
                        nbins=30,
                    )
                    fig_dist.update_layout(height=300)
                    st.plotly_chart(fig_dist, width='stretch')
                else:
                    st.info(t("ab_testing.ab_test.dist_no_data"))
            else:
                st.markdown(t("ab_testing.ab_test.dist_classification_title"))
                dist_rows = []
                for vs in versions_stats:
                    for label, count in vs.get("prediction_distribution", {}).items():
                        dist_rows.append({_col_dist_version: vs["version"], _col_dist_label: str(label), _col_dist_count: count})

                if dist_rows:
                    dist_df = pd.DataFrame(dist_rows)
                    fig_dist = px.bar(
                        dist_df,
                        x=_col_dist_label,
                        y=_col_dist_count,
                        color=_col_dist_version,
                        barmode="group",
                    )
                    fig_dist.update_layout(height=300)
                    st.plotly_chart(fig_dist, width='stretch')
                else:
                    st.info(t("ab_testing.ab_test.dist_no_data"))

    with st.expander(t("ab_testing.shadow.expander"), expanded=False):
        # Taux de concordance shadow
        shadow_versions = [vs for vs in versions_stats if vs.get("agreement_rate") is not None]
        if shadow_versions:
            st.divider()
            st.markdown(t("ab_testing.shadow.concordance_title"))
            st.caption(t("ab_testing.shadow.concordance_caption"))
            _col_conc_version = t("ab_testing.shadow.concordance_col_version")
            _col_conc_rate = t("ab_testing.shadow.concordance_col_rate")
            _col_conc_score = t("ab_testing.shadow.concordance_col_score")
            agree_data = [
                {
                    _col_conc_version: vs["version"],
                    _col_conc_rate:    f"{vs['agreement_rate']:.1%}",
                    _col_conc_score:   vs["agreement_rate"],
                }
                for vs in shadow_versions
            ]
            agree_df = pd.DataFrame(agree_data)
            fig_agree = px.bar(
                agree_df,
                x=_col_conc_version,
                y=_col_conc_score,
                text=_col_conc_rate,
                color_discrete_sequence=["#7c3aed"],
                range_y=[0, 1],
            )
            fig_agree.update_layout(height=250, yaxis_tickformat=".0%")
            st.plotly_chart(fig_agree, width='stretch')
        else:
            st.info(t("ab_testing.shadow.concordance_no_data"))

        # ── Analyse shadow enrichie ──────────────────────────────────────────
        # ===========================================================================
        # SECTION 3 — Analyse shadow enrichie
        # ===========================================================================
        st.subheader(t("ab_testing.shadow.enriched_title"))
        st.caption(t("ab_testing.shadow.enriched_caption"))

        try:
            shadow_data = client.get_shadow_comparison(selected_model, period_days=days)
            shadow_available = True
        except Exception:
            shadow_data = None
            shadow_available = False

        if not shadow_available or shadow_data is None:
            st.info(t("ab_testing.shadow.load_error"))
        elif shadow_data.get("shadow_version") is None:
            st.info(t("ab_testing.shadow.no_shadow_version"))
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
                "shadow_better":    ("🟢", t("ab_testing.shadow.rec_shadow_better")),
                "production_better": ("🔴", t("ab_testing.shadow.rec_prod_better")),
                "equivalent":       ("🟡", t("ab_testing.shadow.rec_equivalent")),
                "insufficient_data": ("⚪", t("ab_testing.shadow.rec_insufficient")),
            }.get(recommendation, ("⚪", recommendation))

            st.markdown(
                t("ab_testing.shadow.vs_banner", sv=sv, pv=pv, badge=_rec_badge[0], recommendation=_rec_badge[1])
            )

            _cols = st.columns(5)
            _cols[0].metric(t("ab_testing.shadow.kpi_pairs"), n_comparable)
            _cols[1].metric(
                t("ab_testing.shadow.kpi_agreement"),
                f"{agreement_rate:.1%}" if agreement_rate is not None else "—",
                help=t("ab_testing.shadow.kpi_agreement_help"),
            )
            _cols[2].metric(
                t("ab_testing.shadow.kpi_conf_delta"),
                f"{conf_delta:+.3f}" if conf_delta is not None else "—",
                delta_color="normal" if conf_delta is not None else "off",
                delta=f"{conf_delta:+.3f}" if conf_delta is not None else None,
                help=t("ab_testing.shadow.kpi_conf_delta_help"),
            )
            _cols[3].metric(
                t("ab_testing.shadow.kpi_lat_delta"),
                f"{lat_delta:+.1f}" if lat_delta is not None else "—",
                delta=f"{lat_delta:+.1f}" if lat_delta is not None else None,
                delta_color="inverse" if lat_delta is not None else "off",
                help=t("ab_testing.shadow.kpi_lat_delta_help"),
            )

            if accuracy_available and shadow_acc is not None and prod_acc is not None:
                _cols[4].metric(
                    t("ab_testing.shadow.kpi_accuracy"),
                    f"{shadow_acc:.1%} / {prod_acc:.1%}",
                    help=t("ab_testing.shadow.kpi_accuracy_help"),
                )
            else:
                _cols[4].metric(
                    t("ab_testing.shadow.kpi_accuracy_na"),
                    "—",
                    help=t("ab_testing.shadow.kpi_accuracy_na_help"),
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
                        title={"text": t("ab_testing.shadow.gauge_title")},
                    )
                )
                fig_agree_gauge.update_layout(height=240, margin=dict(t=40, b=10, l=20, r=20))
                st.plotly_chart(fig_agree_gauge, width='stretch')

            if recommendation == "shadow_better" and is_admin:
                with st.container(border=True):
                    promo_col1, promo_col2 = st.columns([3, 1])
                    promo_col1.markdown(
                        t("ab_testing.shadow.promote_banner", sv=sv, pv=pv)
                    )
                    if promo_col2.button(
                        t("ab_testing.shadow.promote_btn"),
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
                            st.success(t("ab_testing.shadow.promote_success", sv=sv))
                            st.cache_data.clear()
                            st.rerun()

        st.divider()

        # ── Log des prédictions shadow récentes ────────────────────────────
        # ===========================================================================
        # SECTION 4 — Log des prédictions shadow récentes
        # ===========================================================================
        st.subheader(t("ab_testing.shadow.recent_title"))

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
            st.warning(t("ab_testing.shadow.recent_load_error", error=e))
            shadow_preds = []

        if not shadow_preds:
            st.info(t("ab_testing.shadow.recent_no_data"))
        else:
            _col_sh_id = t("ab_testing.shadow.col_id")
            _col_sh_ts = t("ab_testing.shadow.col_timestamp")
            _col_sh_ver = t("ab_testing.shadow.col_version")
            _col_sh_idobs = t("ab_testing.shadow.col_id_obs")
            _col_sh_result = t("ab_testing.shadow.col_result")
            _col_sh_latency = t("ab_testing.shadow.col_latency")
            _col_sh_status = t("ab_testing.shadow.col_status")

            shadow_rows = []
            for p in shadow_preds:
                shadow_rows.append(
                    {
                        _col_sh_id:      p.get("id"),
                        _col_sh_ts:      (
                            pd.to_datetime(p.get("timestamp")).strftime("%Y-%m-%d %H:%M:%S")
                            if p.get("timestamp")
                            else "—"
                        ),
                        _col_sh_ver:     p.get("model_version") or "—",
                        _col_sh_idobs:   p.get("id_obs") or "—",
                        _col_sh_result:  str(p.get("prediction_result", "")),
                        _col_sh_latency: (
                            f"{p['response_time_ms']:.1f}" if p.get("response_time_ms") is not None else "—"
                        ),
                        _col_sh_status:  "✅" if p.get("status") == "success" else "❌",
                    }
                )

            st.caption(t("ab_testing.shadow.recent_count", count=len(shadow_preds)))
            st.dataframe(
                pd.DataFrame(shadow_rows),
                width='stretch',
                hide_index=True,
                column_config={
                    _col_sh_id: st.column_config.NumberColumn(
                        _col_sh_id,
                        help=t("ab_testing.shadow.col_id_help"),
                    ),
                    _col_sh_ts: st.column_config.TextColumn(
                        _col_sh_ts,
                        help=t("ab_testing.shadow.col_timestamp_help"),
                    ),
                    _col_sh_ver: st.column_config.TextColumn(
                        _col_sh_ver,
                        help=t("ab_testing.shadow.col_version_help"),
                    ),
                    _col_sh_idobs: st.column_config.TextColumn(
                        _col_sh_idobs,
                        help=t("ab_testing.shadow.col_id_obs_help"),
                    ),
                    _col_sh_result: st.column_config.TextColumn(
                        _col_sh_result,
                        help=t("ab_testing.shadow.col_result_help"),
                    ),
                    _col_sh_latency: st.column_config.TextColumn(
                        _col_sh_latency,
                        help=t("ab_testing.shadow.col_latency_help"),
                    ),
                    _col_sh_status: st.column_config.TextColumn(
                        _col_sh_status,
                        help=t("ab_testing.shadow.col_status_help"),
                    ),
                },
            )
