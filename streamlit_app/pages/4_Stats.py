"""
Statistiques et graphiques d'utilisation
"""

import re
from datetime import datetime, timedelta, date

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
if col_refresh.button("🔄 Rafraîchir", key="stats_refresh", width='stretch'):
    st.cache_data.clear()
    st.rerun()

client = get_client()

# --- Filtres date (globaux) ---
col_d1, col_d2 = st.columns([1, 1])

date_start = col_d1.date_input(
    "Date début",
    value=date.today() - timedelta(days=7),
    max_value=date.today(),
    key="stats_date_start",
)
date_end = col_d2.date_input(
    "Date fin",
    value=date.today(),
    min_value=date_start,
    max_value=date.today(),
    key="stats_date_end",
)

if date_end < date_start:
    st.error("La date de fin doit être après la date de début.")
    st.stop()

days = max(1, (date_end - date_start).days)
start_dt = datetime.combine(date_start, datetime.min.time())
end_dt   = datetime.combine(date_end,   datetime.max.time().replace(microsecond=0))

try:
    models = get_models_cached(
        st.session_state.get("api_url"), st.session_state.get("api_token")
    )
    model_names = sorted({m["name"] for m in models})
except Exception:
    model_names = []

if not model_names:
    st.warning("Aucun modèle disponible.")
    st.stop()

# --- Helpers Leaderboard ---
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


def _bg_r2(val):
    """Coloration R² : vert ≥ 0.90, jaune ≥ 0.70, rouge < 0.70, gris si None/NaN."""
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
            "r2": (m.get("training_metrics") or {}).get("r2"),
            "rmse": (m.get("training_metrics") or {}).get("rmse"),
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
    elif metric == "r2":
        rows.sort(key=lambda r: r["r2"] if r["r2"] is not None else -float("inf"), reverse=True)
    elif metric == "rmse":
        rows.sort(key=lambda r: r["rmse"] if r["rmse"] is not None else float("inf"))
    else:
        rows.sort(key=lambda r: r["accuracy"] if r["accuracy"] is not None else -1, reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


_DRIFT_COLOR = {
    "ok": "#2ECC71",
    "warning": "#F39C12",
    "critical": "#E74C3C",
    "no_baseline": "#95A5A6",
    "no_data": "#95A5A6",
    "insufficient_data": "#95A5A6",
    "unknown": "#95A5A6",
}


# ── Expander 1 : Leaderboard ─────────────────────────────────────────────────
with st.expander("🏆 Leaderboard — Modèles en production", expanded=True):
    lb_col_metric, _ = st.columns([2, 3])
    lb_metric = lb_col_metric.selectbox(
        "Trier par",
        options=["accuracy", "f1_score", "r2", "rmse", "latency_p95_ms", "predictions_count"],
        format_func=lambda x: {
            "accuracy": "Accuracy",
            "f1_score": "F1 Score",
            "r2": "R² (régression)",
            "rmse": "RMSE (régression)",
            "latency_p95_ms": "Latence p95",
            "predictions_count": "Volume de prédictions",
        }.get(x, x),
        key="lb_metric",
    )

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
                    "r2": "R²",
                    "rmse": "RMSE",
                    "latency_p95_ms": "Latence p95 (ms)",
                    "drift_status": "Drift",
                    "predictions_count": f"Prédictions ({days}j)",
                }
            )
            df_display["Drift"] = df_display["Drift"].map(lambda x: _DRIFT_EMOJI.get(x, x))
            df_display["Accuracy"] = df_display["Accuracy"].apply(
                lambda x: round(x, 4) if pd.notna(x) and x is not None else None
            )
            df_display["F1 Score"] = df_display["F1 Score"].apply(
                lambda x: round(x, 4) if pd.notna(x) and x is not None else None
            )
            df_display["R²"] = df_display["R²"].apply(
                lambda x: round(x, 4) if pd.notna(x) and x is not None else None
            )
            df_display["RMSE"] = df_display["RMSE"].apply(
                lambda x: round(x, 4) if pd.notna(x) and x is not None else None
            )
            df_display["Latence p95 (ms)"] = df_display["Latence p95 (ms)"].apply(
                lambda x: f"{x:.0f} ms" if pd.notna(x) and x is not None else "—"
            )

            # Colonnes avec valeur active selon la métrique de tri
            _style_subsets = ["Accuracy"]
            if "R²" in df_display.columns and df_display["R²"].notna().any():
                _style_subsets.append("R²")

            styled = (
                df_display.style
                .map(_bg_accuracy, subset=["Accuracy"])
                .map(_bg_r2, subset=["R²"])
                .hide(axis="index")
            )
            st.dataframe(
                styled,
                width='stretch',
                column_config={
                    "Accuracy": st.column_config.NumberColumn(
                        "Accuracy",
                        help="Proportion de prédictions correctes sur le jeu de test (classification). Entre 0 et 1 — plus c'est proche de 1, meilleur est le modèle.",
                        format="%.4f",
                    ),
                    "F1 Score": st.column_config.NumberColumn(
                        "F1 Score",
                        help="Moyenne harmonique précision/rappel sur le jeu de test (classification). Robuste aux classes déséquilibrées. Entre 0 et 1.",
                        format="%.4f",
                    ),
                    "R²": st.column_config.NumberColumn(
                        "R²",
                        help="Coefficient de détermination R² (régression). Mesure la part de variance expliquée par le modèle : 1.0 = parfait, 0.0 = modèle nul, < 0 = pire que la moyenne. '—' pour les modèles de classification.",
                        format="%.4f",
                    ),
                    "RMSE": st.column_config.NumberColumn(
                        "RMSE",
                        help="Root Mean Square Error (régression). Erreur moyenne en unité de la variable cible — plus c'est faible, meilleur est le modèle. Trié croissant quand sélectionné. '—' pour les modèles de classification.",
                        format="%.4f",
                    ),
                    "Latence p95 (ms)": st.column_config.TextColumn(
                        "Latence p95 (ms)",
                        help="95e percentile du temps de réponse sur la période sélectionnée. 95 % des requêtes ont été traitées en moins de cette durée.",
                    ),
                    "Drift": st.column_config.TextColumn(
                        "Drift",
                        help="Écart entre la distribution des features en production et la baseline d'entraînement. 🔴 critique = dérive significative, le modèle voit des données différentes de celles sur lesquelles il a été entraîné.",
                    ),
                    f"Prédictions ({days}j)": st.column_config.NumberColumn(
                        f"Prédictions ({days}j)",
                        help=f"Nombre total de prédictions servies sur les {days} derniers jours.",
                    ),
                },
            )
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
                    "r2": e.get("r2"),
                    "rmse": e.get("rmse"),
                    "latency_p95_ms": e["latency_p95_ms"],
                    "drift_status": e["drift_status"],
                    "predictions_count": e["predictions_count"],
                }
                for e in leaderboard
            ])

            # Config de chaque métrique : label, ratio [0-1] ou non, step et max pour le seuil
            _METRIC_CFG = {
                "latency_p95_ms":   {"label": "Latence p95 (ms)",    "is_ratio": False, "step": 10.0,  "max": None, "fmt_hover": lambda v: f"{v:.0f} ms"},
                "accuracy":         {"label": "Accuracy",            "is_ratio": True,  "step": 0.05,  "max": 1.0,  "fmt_hover": lambda v: f"{v:.4f}"},
                "f1_score":         {"label": "F1 Score",            "is_ratio": True,  "step": 0.05,  "max": 1.0,  "fmt_hover": lambda v: f"{v:.4f}"},
                "r2":               {"label": "R²",                  "is_ratio": True,  "step": 0.05,  "max": 1.0,  "fmt_hover": lambda v: f"{v:.4f}"},
                "rmse":             {"label": "RMSE",                "is_ratio": False, "step": 0.1,   "max": None, "fmt_hover": lambda v: f"{v:.4f}"},
                "predictions_count":{"label": "Volume prédictions",  "is_ratio": False, "step": 100.0, "max": None, "fmt_hover": lambda v: f"{int(v):,}"},
            }

            # Axe Y = métrique sélectionnée dans "Trier par" (latency/count → fallback accuracy)
            _Y_FALLBACK = {"latency_p95_ms": "accuracy", "predictions_count": "accuracy"}
            scatter_y_metric = _Y_FALLBACK.get(lb_metric, lb_metric)

            # Dropdown Axe X — toutes les métriques sauf celle déjà sur Y
            _x_options = [k for k in _METRIC_CFG if k != scatter_y_metric]
            _x_default = "latency_p95_ms" if "latency_p95_ms" in _x_options else _x_options[0]
            _x_default_idx = _x_options.index(_x_default)

            sc_col_x, sc_col_seuil_x, sc_col_seuil_y = st.columns([2, 2, 2])
            scatter_x_metric = sc_col_x.selectbox(
                "Axe X",
                options=_x_options,
                index=_x_default_idx,
                format_func=lambda k: _METRIC_CFG[k]["label"],
                key="scatter_x_metric",
            )

            cfg_x = _METRIC_CFG[scatter_x_metric]
            cfg_y = _METRIC_CFG[scatter_y_metric]
            x_label = cfg_x["label"]
            y_label = cfg_y["label"]

            # Seuil X (ligne verticale)
            _sx_is_lower_better = scatter_x_metric == "rmse"
            _sx_label = f"Seuil {x_label} (max)" if _sx_is_lower_better else f"Seuil {x_label} (min)"
            x_threshold = sc_col_seuil_x.number_input(
                _sx_label,
                min_value=0.0,
                max_value=float(cfg_x["max"]) if cfg_x["max"] else None,
                value=0.0,
                step=float(cfg_x["step"]),
                format="%.2f",
                help=f"Affiche une ligne verticale sur l'axe X ({x_label}). 0 = désactivé.",
                key="scatter_x_threshold",
            )

            # Seuil Y (ligne horizontale)
            _sy_is_lower_better = scatter_y_metric == "rmse"
            _sy_label = f"Seuil {y_label} (max)" if _sy_is_lower_better else f"Seuil {y_label} (min)"
            y_threshold = sc_col_seuil_y.number_input(
                _sy_label,
                min_value=0.0,
                max_value=float(cfg_y["max"]) if cfg_y["max"] else None,
                value=0.0,
                step=float(cfg_y["step"]),
                format="%.2f",
                help=f"Affiche une ligne horizontale sur l'axe Y ({y_label}). 0 = désactivé.",
                key="scatter_y_threshold",
            )

            df_plot = df_scatter.dropna(subset=[scatter_x_metric, scatter_y_metric]).copy()
            df_plot["color"] = df_plot["drift_status"].map(
                lambda s: _DRIFT_COLOR.get(s, _DRIFT_COLOR["unknown"])
            )
            df_plot["drift_label"] = df_plot["drift_status"].map(
                lambda s: _DRIFT_EMOJI.get(s, s)
            )
            df_plot["label"] = df_plot["name"] + " v" + df_plot["version"]

            if df_plot.empty:
                st.info(
                    f"Données insuffisantes : les modèles doivent avoir {x_label} et {y_label} renseignés."
                )
            else:
                fig = go.Figure()

                for _, row in df_plot.iterrows():
                    x_val   = row[scatter_x_metric]
                    y_val   = row[scatter_y_metric]
                    count   = int(row.get("predictions_count") or 0)
                    bubble  = max(10, min(60, count ** 0.5)) if count > 0 else 15

                    # Hover : X et Y en tête, puis toutes les autres métriques disponibles
                    hover_lines = [
                        f"<b>{row['label']}</b>",
                        f"{x_label} : {cfg_x['fmt_hover'](x_val)}",
                        f"{y_label} : {cfg_y['fmt_hover'](y_val)}",
                    ]
                    for key, cfg in _METRIC_CFG.items():
                        if key in (scatter_x_metric, scatter_y_metric, "predictions_count"):
                            continue
                        v = row.get(key)
                        if v is not None and not (isinstance(v, float) and v != v):
                            hover_lines.append(f"{cfg['label']} : {cfg['fmt_hover'](v)}")
                    hover_lines += [
                        f"Prédictions : {count:,}",
                        f"Drift : {row['drift_label']}",
                    ]
                    hover_text = "<br>".join(hover_lines)

                    fig.add_trace(go.Scatter(
                        x=[x_val],
                        y=[y_val],
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

                # Ligne verticale (seuil X)
                if x_threshold > 0:
                    _vline_label = (
                        f"Max {x_label} {x_threshold:.2f}" if _sx_is_lower_better
                        else f"Min {x_label} {x_threshold:.0%}" if cfg_x["is_ratio"]
                        else f"Seuil {x_label} {x_threshold:.0f}"
                    )
                    fig.add_vline(
                        x=x_threshold,
                        line_dash="dash",
                        line_color="#E74C3C",
                        annotation_text=_vline_label,
                        annotation_position="top right",
                        annotation_font_color="#E74C3C",
                    )

                # Ligne horizontale (seuil Y)
                if y_threshold > 0:
                    _hline_label = (
                        f"Max {y_label} {y_threshold:.2f}" if _sy_is_lower_better
                        else f"Min {y_label} {y_threshold:.0%}" if cfg_y["is_ratio"]
                        else f"Seuil {y_label} {y_threshold:.2f}"
                    )
                    fig.add_hline(
                        y=y_threshold,
                        line_dash="dash",
                        line_color="#E74C3C",
                        annotation_text=_hline_label,
                        annotation_position="bottom right",
                        annotation_font_color="#E74C3C",
                    )

                layout_kwargs = dict(
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    margin=dict(t=40, b=40),
                    hovermode="closest",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                )
                if cfg_x["is_ratio"]:
                    layout_kwargs["xaxis_range"] = [0, 1.05]
                    layout_kwargs["xaxis_tickformat"] = ".0%"
                if cfg_y["is_ratio"]:
                    layout_kwargs["yaxis_range"] = [0, 1.05]
                    layout_kwargs["yaxis_tickformat"] = ".0%"

                fig.update_layout(**layout_kwargs)
                st.plotly_chart(fig, width='stretch')


# ── Expander 2 : Statistiques agrégées par modèle ────────────────────────────
with st.expander("📊 Statistiques agrégées par modèle", expanded=True):
    # Filtre modèle local à cet expander
    _s2_col_search, _s2_col_model = st.columns([2, 2])
    with _s2_col_search:
        stats_search = st.text_input("Filtrer par nom", key="stats_model_search", placeholder="Rechercher…")
        stats_filtered = [n for n in model_names if stats_search.lower() in n.lower()] if stats_search else model_names
    with _s2_col_model:
        model_filter = st.selectbox("Modèle", ["(tous)"] + (stats_filtered or model_names), key="stats_model_filter")
    selected_model = None if model_filter == "(tous)" else model_filter

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

    # Statistiques agrégées (endpoint /predictions/stats)
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
            st.dataframe(
                df_stats,
                width='stretch',
                hide_index=True,
                column_config={
                    "Modèle": st.column_config.TextColumn(
                        "Modèle",
                        help="Nom du modèle ML pour lequel les statistiques sont calculées.",
                    ),
                    "Total": st.column_config.NumberColumn(
                        "Total",
                        help="Nombre total de prédictions servies par ce modèle sur la période sélectionnée.",
                    ),
                    "Erreurs": st.column_config.NumberColumn(
                        "Erreurs",
                        help="Nombre de prédictions ayant renvoyé une erreur (statut 'error') sur la période.",
                    ),
                    "Taux d'erreur": st.column_config.TextColumn(
                        "Taux d'erreur",
                        help="Pourcentage de prédictions en erreur. Un taux > 5 % mérite attention, > 10 % est critique.",
                    ),
                    "Moy. RT (ms)": st.column_config.NumberColumn(
                        "Moy. RT (ms)",
                        help="Temps de réponse moyen en millisecondes. Inclut le chargement du modèle et le calcul.",
                        format="%.1f",
                    ),
                    "p50 RT (ms)": st.column_config.NumberColumn(
                        "p50 RT (ms)",
                        help="Médiane du temps de réponse : 50 % des requêtes sont traitées en moins de cette durée.",
                        format="%.1f",
                    ),
                    "p95 RT (ms)": st.column_config.NumberColumn(
                        "p95 RT (ms)",
                        help="95e percentile du temps de réponse : 95 % des requêtes sont traitées en moins de cette durée. Indicateur clé pour les SLA.",
                        format="%.1f",
                    ),
                },
            )
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

    # --- Métriques KPI ---
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
        st.plotly_chart(fig, width='stretch')

    # Temps de réponse — courbe de densité (KDE) par modèle
    with row1_r:
        st.subheader("Distribution des temps de réponse")
        if "response_time_ms" in df.columns:
            import numpy as np

            df_rt = df[~df["is_error"] & df["response_time_ms"].notna()].copy()
            colors = px.colors.qualitative.Set2
            fig = go.Figure()

            for i, (model, grp) in enumerate(df_rt.groupby("model_name")):
                vals = grp["response_time_ms"].values
                if len(vals) < 5:
                    continue
                # Borner à p99 pour ne pas écraser la courbe avec des outliers rares
                p99 = float(np.percentile(vals, 99))
                vals_clipped = vals[vals <= p99]
                # Histogramme normalisé en densité → 50 bacs
                counts, edges = np.histogram(vals_clipped, bins=50, density=True)
                centers = (edges[:-1] + edges[1:]) / 2
                # Lissage gaussien léger (convolution) sans scipy
                sigma = 2
                k = np.arange(-3 * sigma, 3 * sigma + 1)
                kernel = np.exp(-0.5 * (k / sigma) ** 2)
                kernel /= kernel.sum()
                smoothed = np.convolve(counts, kernel, mode="same")

                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=centers,
                    y=smoothed,
                    mode="lines",
                    name=model,
                    line=dict(width=2, color=color),
                    fill="tozeroy",
                    opacity=0.6,
                ))

            fig.update_layout(
                xaxis_title="Temps (ms)",
                yaxis_title="Densité",
                margin=dict(t=20),
                legend=dict(orientation="h", y=-0.25),
                hovermode="x unified",
            )
            st.plotly_chart(fig, width='stretch')
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
        st.plotly_chart(fig, width='stretch')

    # Erreurs par jour par modèle
    with row2_r:
        st.subheader("Erreurs par jour — par modèle")
        errors_daily = (
            df[df["is_error"]]
            .groupby(["date", "model_name"])
            .size()
            .reset_index(name="Erreurs")
            .rename(columns={"date": "Date", "model_name": "Modèle"})
        )
        if errors_daily.empty:
            st.success("✅ Aucune erreur sur la période sélectionnée.")
        else:
            fig = px.line(
                errors_daily,
                x="Date",
                y="Erreurs",
                color="Modèle",
                markers=True,
                labels={"Date": "Date", "Erreurs": "Nb erreurs", "Modèle": "Modèle"},
            )
            fig.update_layout(margin=dict(t=20), legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, width='stretch')

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
        st.plotly_chart(fig, width='stretch')


# ── Expander 3 : Accuracy temporelle multi-modèles ───────────────────────────
with st.expander("📈 Accuracy temporelle — comparaison multi-modèles", expanded=True):
    acc_col_models, acc_col_gran = st.columns([4, 1])

    with acc_col_models:
        _default_acc = model_names[:6] if len(model_names) > 6 else model_names
        acc_models = st.multiselect(
            "Modèles à comparer",
            options=model_names,
            default=_default_acc,
            key="acc_models_multiselect",
        )

    with acc_col_gran:
        acc_gran = st.selectbox(
            "Granularité",
            options=["day", "week"],
            format_func=lambda x: {"day": "Jour", "week": "Semaine"}.get(x, x),
            key="acc_granularity",
        )

    if not acc_models:
        st.info("Sélectionnez au moins un modèle.")
    else:
        acc_rows: list[dict] = []
        acc_errors: list[str] = []
        with st.spinner("Chargement des métriques de performance…"):
            for mname in acc_models:
                try:
                    perf = client.get_model_performance(
                        model_name=mname,
                        start=start_dt.isoformat(),
                        end=end_dt.isoformat(),
                        granularity=acc_gran,
                    )
                    mtype = perf.get("model_type", "classification")
                    mcol = "accuracy" if mtype == "classification" else "mae"
                    for period in perf.get("by_period") or []:
                        v = period.get(mcol)
                        if v is not None:
                            acc_rows.append(
                                {
                                    "date": period["period"],
                                    "Modèle": mname,
                                    "metric": float(v),
                                    "model_type": mtype,
                                    "matched_count": int(period.get("matched_count") or 0),
                                }
                            )
                except Exception:
                    acc_errors.append(mname)

        if acc_errors:
            st.caption(f"⚠️ Données indisponibles pour : {', '.join(acc_errors)}")

        if not acc_rows:
            st.info(
                "Aucune donnée d'accuracy sur la période sélectionnée. "
                "Soumettez des résultats via **POST /observed-results** pour activer le suivi."
            )
        else:
            df_acc = pd.DataFrame(acc_rows)
            # L'API renvoie "2026-W19" (granularité semaine) ou "2026-05-11" (jour).
            # pd.to_datetime ne reconnaît pas le format ISO semaine — on détecte et parse
            # manuellement : "YYYY-Wnn" → lundi de la semaine via strptime %G-W%V-%u.
            _sample = str(df_acc["date"].iloc[0]) if len(df_acc) else ""
            if re.match(r"^\d{4}-W\d{2}$", _sample):
                df_acc["date"] = pd.to_datetime(
                    df_acc["date"] + "-1", format="%G-W%V-%u"
                )
            else:
                df_acc["date"] = pd.to_datetime(df_acc["date"])
            df_acc = df_acc.sort_values("date")

            # Détecter si tous les modèles sont du même type
            _types = df_acc["model_type"].unique().tolist()
            _all_classif = all(t == "classification" for t in _types)
            _all_regress = all(t == "regression" for t in _types)
            if _all_classif:
                metric_label = "Accuracy"
                y_fmt = ".0%"
                y_range = [0, 1.05]
                hover_fmt = ".1%"
            elif _all_regress:
                metric_label = "MAE"
                y_fmt = None
                y_range = None
                hover_fmt = ".4f"
            else:
                metric_label = "Accuracy / MAE"
                y_fmt = None
                y_range = None
                hover_fmt = ".4f"

            colors = px.colors.qualitative.Set2
            fig_acc = go.Figure()
            for i, (model, grp) in enumerate(df_acc.groupby("Modèle", sort=False)):
                grp = grp.sort_values("date")
                color = colors[i % len(colors)]
                mtype_grp = grp["model_type"].iloc[0]
                # suffixe "(MAE)" si types mixtes
                legend_label = model if len(_types) == 1 else (
                    f"{model} (MAE)" if mtype_grp == "regression" else model
                )
                h_fmt = ".1%" if mtype_grp == "classification" else ".4f"
                fig_acc.add_trace(
                    go.Scatter(
                        x=grp["date"],
                        y=grp["metric"],
                        mode="lines+markers",
                        name=legend_label,
                        line=dict(width=2, color=color),
                        marker=dict(size=6, color=color),
                        customdata=grp[["matched_count"]].values,
                        hovertemplate=(
                            f"<b>{model}</b><br>"
                            "%{x|%Y-%m-%d}<br>"
                            f"{metric_label} : %{{y:{h_fmt}}}<br>"
                            "Paires obs. : %{customdata[0]}<extra></extra>"
                        ),
                    )
                )

            fig_acc.update_layout(
                xaxis_title="Date",
                yaxis_title=metric_label,
                yaxis_tickformat=y_fmt,
                yaxis_range=y_range,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=50, b=20),
                hovermode="x unified",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(200,200,200,0.12)"),
                yaxis=dict(gridcolor="rgba(200,200,200,0.12)"),
            )
            st.plotly_chart(fig_acc, width='stretch')

            # Tableau résumé
            _summary = []
            for model, grp in df_acc.groupby("Modèle", sort=False):
                last = grp.sort_values("date").iloc[-1]
                mtype_grp = grp["model_type"].iloc[0]
                _summary.append(
                    {
                        "Modèle": model,
                        metric_label: round(last["metric"], 4),
                        "Paires observées": int(grp["matched_count"].sum()),
                    }
                )
            df_summary = (
                pd.DataFrame(_summary)
                .sort_values(metric_label, ascending=_all_regress)
                .reset_index(drop=True)
            )
            st.dataframe(
                df_summary,
                width='stretch',
                hide_index=True,
                column_config={
                    metric_label: st.column_config.NumberColumn(
                        metric_label,
                        help=(
                            "Dernière valeur connue sur la période. "
                            "Accuracy = proportion de prédictions correctes (classification). "
                            "MAE = erreur absolue moyenne (régression)."
                        ),
                        format="%.4f",
                    ),
                    "Paires observées": st.column_config.NumberColumn(
                        "Paires observées",
                        help="Nombre de prédictions pour lesquelles un résultat observé a été soumis.",
                    ),
                },
            )


# ── Expander 4 : Évolution multi-métriques — un modèle ───────────────────────
with st.expander("📊 Évolution multi-métriques — un modèle", expanded=True):

    # ── Contrôles ────────────────────────────────────────────────────────────
    _pm_col_model, _pm_col_ver, _pm_col_gran = st.columns([3, 2, 1])

    with _pm_col_model:
        perf_model = st.selectbox(
            "Modèle",
            options=model_names,
            key="pm_model_select",
        )

    with _pm_col_ver:
        _pm_versions = ["(toutes)"] + sorted(
            {m["version"] for m in models if m["name"] == perf_model},
            key=lambda v: [int(x) for x in v.split(".")],
            reverse=True,
        )
        perf_ver_sel = st.selectbox("Version", _pm_versions, key="pm_ver_select")
        perf_ver_arg = None if perf_ver_sel == "(toutes)" else perf_ver_sel

    with _pm_col_gran:
        perf_gran = st.selectbox(
            "Granularité",
            options=["day", "week", "month"],
            format_func=lambda x: {"day": "Jour", "week": "Semaine", "month": "Mois"}.get(x, x),
            key="pm_gran_select",
        )

    # ── Chargement ───────────────────────────────────────────────────────────
    try:
        _pm_perf = client.get_model_performance(
            model_name=perf_model,
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            version=perf_ver_arg,
            granularity=perf_gran,
        )
        _pm_mtype = _pm_perf.get("model_type", "classification")
        _pm_periods = _pm_perf.get("by_period") or []
    except Exception as _pm_exc:
        st.warning(f"Impossible de charger les métriques : {_pm_exc}")
        _pm_mtype = "classification"
        _pm_periods = []

    # ── Catalogue des métriques disponibles selon le type ────────────────────
    _PM_METRICS_CLASSIF: dict[str, dict] = {
        "accuracy":      {"label": "Accuracy",         "ratio": True},
        "f1_weighted":   {"label": "F1 pondéré",       "ratio": True},
        "matched_count": {"label": "Paires observées", "ratio": False},
    }
    _PM_METRICS_REGRESS: dict[str, dict] = {
        "mae":           {"label": "MAE",              "ratio": False},
        "rmse":          {"label": "RMSE",             "ratio": False},
        "matched_count": {"label": "Paires observées", "ratio": False},
    }
    _pm_catalog = _PM_METRICS_CLASSIF if _pm_mtype == "classification" else _PM_METRICS_REGRESS
    _pm_defaults = ["accuracy", "f1_weighted"] if _pm_mtype == "classification" else ["mae", "rmse"]

    perf_metrics = st.multiselect(
        "Métriques à afficher",
        options=list(_pm_catalog.keys()),
        default=_pm_defaults,
        format_func=lambda k: _pm_catalog[k]["label"],
        key="pm_metrics_select",
    )

    # ── Graphique ─────────────────────────────────────────────────────────────
    if not perf_metrics:
        st.info("Sélectionnez au moins une métrique.")
    elif not _pm_periods:
        st.info(
            "Pas de données d'observed-results pour ce modèle sur la période sélectionnée. "
            "Soumettez des résultats via **POST /observed-results** pour activer le suivi."
        )
    else:
        # Parse dates (jour / semaine ISO / mois)
        df_pm = pd.DataFrame(_pm_periods)
        _pm_sample = str(df_pm["period"].iloc[0])
        if re.match(r"^\d{4}-W\d{2}$", _pm_sample):
            df_pm["date"] = pd.to_datetime(df_pm["period"] + "-1", format="%G-W%V-%u")
        elif re.match(r"^\d{4}-\d{2}$", _pm_sample):
            df_pm["date"] = pd.to_datetime(df_pm["period"] + "-01", format="%Y-%m-%d")
        else:
            df_pm["date"] = pd.to_datetime(df_pm["period"])
        df_pm = df_pm.sort_values("date").reset_index(drop=True)

        # Métriques primaires (axe gauche) vs matched_count (axe droit, barres)
        _primary = [m for m in perf_metrics if m != "matched_count"]
        _show_count = "matched_count" in perf_metrics
        _all_ratio = bool(_primary) and all(_pm_catalog[m]["ratio"] for m in _primary)

        colors = px.colors.qualitative.Set2
        fig_pm = go.Figure()

        for i, metric in enumerate(_primary):
            cfg = _pm_catalog[metric]
            col = colors[i % len(colors)]
            vals = df_pm[metric] if metric in df_pm.columns else pd.Series([None] * len(df_pm))
            h_fmt = ".1%" if cfg["ratio"] else ".4f"
            fig_pm.add_trace(
                go.Scatter(
                    x=df_pm["date"],
                    y=vals,
                    mode="lines+markers",
                    name=cfg["label"],
                    line=dict(width=2, color=col),
                    marker=dict(size=7, color=col),
                    yaxis="y1",
                    hovertemplate=(
                        f"<b>{cfg['label']}</b><br>"
                        "%{x|%Y-%m-%d}<br>"
                        f"%{{y:{h_fmt}}}<extra></extra>"
                    ),
                )
            )

        if _show_count:
            fig_pm.add_trace(
                go.Bar(
                    x=df_pm["date"],
                    y=df_pm["matched_count"],
                    name="Paires observées",
                    yaxis="y2",
                    opacity=0.22,
                    marker_color="#999999",
                    hovertemplate=(
                        "<b>Paires observées</b><br>"
                        "%{x|%Y-%m-%d}<br>"
                        "%{y:d}<extra></extra>"
                    ),
                )
            )

        y1_title = " · ".join(_pm_catalog[m]["label"] for m in _primary) or ""
        fig_pm.update_layout(
            xaxis_title="Date",
            yaxis=dict(
                title=y1_title,
                tickformat=".0%" if _all_ratio else None,
                range=[0, 1.05] if _all_ratio else None,
                gridcolor="rgba(200,200,200,0.12)",
            ),
            yaxis2=dict(
                title="Paires observées",
                overlaying="y",
                side="right",
                showgrid=False,
                rangemode="tozero",
            ) if _show_count else None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=55, b=20),
            hovermode="x unified",
            barmode="overlay",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(200,200,200,0.12)"),
        )
        st.plotly_chart(fig_pm, width='stretch')

        # ── Résumé global de la période ───────────────────────────────────────
        _pm_global_metrics = {
            "accuracy": _pm_perf.get("accuracy"),
            "f1_weighted": _pm_perf.get("f1_weighted"),
            "mae": _pm_perf.get("mae"),
            "rmse": _pm_perf.get("rmse"),
            "matched_count": _pm_perf.get("matched_predictions"),
        }
        _summary_cols = [m for m in perf_metrics if _pm_global_metrics.get(m) is not None]
        if _summary_cols:
            st.caption("Valeurs agrégées sur toute la période sélectionnée :")
            _s_cols = st.columns(len(_summary_cols))
            for col_w, metric in zip(_s_cols, _summary_cols):
                cfg = _pm_catalog[metric]
                val = _pm_global_metrics[metric]
                fmt_val = f"{val:.1%}" if cfg["ratio"] else (
                    f"{int(val):,}" if metric == "matched_count" else f"{val:.4f}"
                )
                col_w.metric(cfg["label"], fmt_val)


# ── Expander 5 : Drift de performance ────────────────────────────────────────
with st.expander("📉 Drift de performance — accuracy rolling", expanded=True):
    drift_col_model, drift_col_threshold, drift_col_dates = st.columns([2, 2, 2])
    with drift_col_model:
        drift_search = st.text_input("Filtrer par nom", key="drift_model_search", placeholder="Rechercher…")
        drift_filtered = [n for n in model_names if drift_search.lower() in n.lower()] if drift_search else model_names
        drift_model = st.selectbox("Modèle (drift)", drift_filtered or model_names, key="drift_model")
    with drift_col_threshold:
        alert_enabled = st.checkbox("Activer alerte seuil", value=True)
        threshold = st.slider(
            "Seuil d'alerte accuracy",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            disabled=not alert_enabled,
        )
    with drift_col_dates:
        _default_end = datetime.utcnow().date()
        _default_start = _default_end - timedelta(days=45)
        drift_date_start = st.date_input("Date début", value=_default_start, key="drift_date_start")
        drift_date_end = st.date_input("Date fin", value=_default_end, key="drift_date_end")

    drift_end = datetime.combine(drift_date_end, datetime.max.time())
    drift_start = datetime.combine(drift_date_start, datetime.min.time())
    _n_drift_days = (drift_date_end - drift_date_start).days

    if drift_start >= drift_end:
        st.warning("⚠️ La date de début doit être antérieure à la date de fin.")
        st.stop()

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
            f"Pas assez de données observées pour ce modèle sur la période sélectionnée "
            f"({drift_date_start} → {drift_date_end}). "
            "Soumettez des résultats via POST /observed-results pour activer le suivi."
        )
    else:
        metric_col = "accuracy" if model_type == "classification" else "mae"
        metric_label = "Accuracy" if model_type == "classification" else "MAE"

        drift_df = pd.DataFrame(by_period)
        drift_df["date"] = pd.to_datetime(drift_df["period"])
        drift_df = drift_df.sort_values("date").reset_index(drop=True)

        if metric_col in drift_df.columns and drift_df[metric_col].notna().any():
            # Fenêtres mobiles adaptées à la plage sélectionnée
            win_short = min(7, max(1, _n_drift_days // 6))
            win_long  = min(30, max(3, _n_drift_days // 2))
            drift_df["rolling_7d"]  = drift_df[metric_col].rolling(win_short, min_periods=1).mean().round(4)
            drift_df["rolling_30d"] = drift_df[metric_col].rolling(win_long,  min_periods=1).mean().round(4)
            roll_short_label = f"Moy. mobile {win_short}j"
            roll_long_label  = f"Moy. mobile {win_long}j"

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
                roll_short_label,
                (
                    f"{drift_df['rolling_7d'].iloc[-1]:.1%}"
                    if model_type == "classification"
                    else f"{drift_df['rolling_7d'].iloc[-1]:.4f}"
                ),
                help=_perf_help,
            )
            m3.metric(f"Prédictions avec résultat observé ({_n_drift_days}j)", f"{matched_total:,}")

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
                    name=roll_short_label,
                    line=dict(color="#636EFA", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=drift_df["date"],
                    y=drift_df["rolling_30d"],
                    mode="lines",
                    name=roll_long_label,
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
            st.plotly_chart(fig, width='stretch')
        else:
            st.info(f"Métrique '{metric_col}' non disponible pour ce modèle (type : {model_type}).")
