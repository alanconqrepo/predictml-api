"""
Historique des prédictions avec filtres
"""

from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st
from utils.auth import get_client, require_auth

st.set_page_config(page_title="Predictions — PredictML", page_icon="📊", layout="wide")
require_auth()

st.title("📊 Historique des prédictions")

client = get_client()

# --- Couverture du ground truth ---
with st.expander("🏷️ Couverture du ground truth", expanded=True):
    try:
        coverage_data = client.get_observed_results_stats()
        total_pred = coverage_data.get("total_predictions", 0)
        labeled = coverage_data.get("labeled_count", 0)
        rate = coverage_data.get("coverage_rate", 0.0)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Prédictions totales", f"{total_pred:,}")
        col_b.metric("Labellisées", f"{labeled:,}")
        col_c.metric("Couverture", f"{rate * 100:.1f} %")

        by_model = coverage_data.get("by_model") or []
        if by_model:
            st.markdown("**Par modèle :**")
            for m in by_model:
                cov = m.get("coverage", 0.0)
                st.progress(
                    cov,
                    text=f"{m['model_name']} — {m['labeled']}/{m['predictions']} ({cov * 100:.1f} %)",
                )
    except Exception:
        st.caption("Données de couverture non disponibles.")

# --- Import CSV résultats observés ---
CSV_TEMPLATE = "id_obs,model_name,observed_result,date_time\n"

with st.expander("📤 Importer des résultats observés (CSV)"):
    st.download_button(
        "⬇️ Télécharger un template CSV",
        data=CSV_TEMPLATE,
        file_name="template_observed_results.csv",
        mime="text/csv",
    )
    uploaded_file = st.file_uploader("Fichier CSV", type=["csv"], key="csv_obs_upload")
    model_name_override = st.text_input(
        "Modèle (override colonne CSV — optionnel)",
        key="csv_obs_model_override",
    )
    if uploaded_file is not None and st.button("Importer", key="csv_obs_submit"):
        try:
            result = client.upload_observed_results_csv(
                file_bytes=uploaded_file.read(),
                filename=uploaded_file.name,
                model_name=model_name_override.strip() or None,
            )
            st.success(f"{result['upserted']} résultats importés depuis **{result['filename']}**")
            if result.get("skipped_rows", 0) > 0:
                st.warning(f"{result['skipped_rows']} ligne(s) ignorée(s)")
                errors = result.get("parse_errors", [])
                if errors:
                    st.dataframe(
                        pd.DataFrame(errors),
                        use_container_width=True,
                        hide_index=True,
                    )
        except Exception as exc:
            st.error(f"Erreur lors de l'import : {exc}")

# --- Export résultats observés ---
with st.expander("📥 Exporter les résultats observés (ground truth)"):
    col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
    ex_model = col_ex1.text_input("Modèle (optionnel)", key="ex_obs_model")
    ex_start = col_ex2.date_input(
        "Date début", value=date.today() - timedelta(days=30), key="ex_obs_start"
    )
    ex_end = col_ex3.date_input("Date fin", value=date.today(), key="ex_obs_end")
    ex_format = col_ex4.selectbox("Format", ["csv", "jsonl"], key="ex_obs_format")

    if st.button("Préparer l'export", key="ex_obs_btn"):
        if ex_start > ex_end:
            st.error("La date de début doit être avant la date de fin.")
        else:
            try:
                content = client.export_observed_results(
                    start=datetime.combine(ex_start, datetime.min.time()).isoformat(),
                    end=datetime.combine(ex_end, datetime.max.time()).isoformat(),
                    model_name=ex_model.strip() or None,
                    export_format=ex_format,
                )
                mime = "text/csv" if ex_format == "csv" else "application/x-ndjson"
                st.download_button(
                    label=f"⬇️ Télécharger observed_results_export.{ex_format}",
                    data=content,
                    file_name=f"observed_results_export.{ex_format}",
                    mime=mime,
                    key="ex_obs_download",
                )
            except Exception as exc:
                st.error(f"Erreur lors de l'export : {exc}")

# --- Filtres ---
with st.expander("🔍 Filtres", expanded=True):
    col1, col2, col3, col4, col5 = st.columns(5)

    # Liste des modèles disponibles
    try:
        models = client.list_models()
        model_names = sorted({m["name"] for m in models})
    except Exception:
        model_names = []

    model_name = col1.selectbox("Modèle", ["(tous)"] + model_names)
    if model_name == "(tous)":
        model_name = model_names[0] if model_names else None

    today = date.today()
    start_date = col2.date_input("Date début", value=today - timedelta(days=7))
    end_date = col3.date_input("Date fin", value=today)
    status_filter = col4.selectbox("Statut", ["Tous", "success", "error"])
    limit = col5.selectbox("Limite", [50, 100, 500], index=1)

if not model_name:
    st.warning("Aucun modèle disponible. Créez d'abord un modèle via l'API.")
    st.stop()

if start_date > end_date:
    st.error("La date de début doit être avant la date de fin.")
    st.stop()

# Pagination via session state
if "pred_offset" not in st.session_state:
    st.session_state["pred_offset"] = 0

start_iso = datetime.combine(start_date, datetime.min.time()).isoformat()
end_iso = datetime.combine(end_date, datetime.max.time()).isoformat()

# Fetch
try:
    data = client.get_predictions(
        model_name=model_name,
        start=start_iso,
        end=end_iso,
        limit=limit,
        offset=st.session_state["pred_offset"],
    )
except Exception as e:
    st.error(f"Erreur lors du chargement : {e}")
    st.stop()

total = data.get("total", 0)
predictions = data.get("predictions", [])

# Filtre statut côté client (l'API ne supporte pas le filtre statut)
if status_filter != "Tous":
    predictions = [p for p in predictions if p.get("status") == status_filter]

st.caption(
    f"**{total}** prédictions trouvées — affichage {st.session_state['pred_offset'] + 1}–{min(st.session_state['pred_offset'] + limit, total)}"
)

if not predictions:
    st.info("Aucune prédiction pour ces critères.")
else:
    rows = []
    for p in predictions:
        rows.append(
            {
                "ID": p.get("id"),
                "Timestamp": (
                    pd.to_datetime(p.get("timestamp")).strftime("%Y-%m-%d %H:%M:%S")
                    if p.get("timestamp")
                    else "—"
                ),
                "Modèle": p.get("model_name", ""),
                "Version": p.get("model_version") or "—",
                "id_obs": p.get("id_obs") or "—",
                "Résultat": str(p.get("prediction_result", "")),
                "Temps (ms)": (
                    f"{p['response_time_ms']:.1f}" if p.get("response_time_ms") is not None else "—"
                ),
                "Statut": "✅" if p.get("status") == "success" else "❌",
                "Shadow": "🔮" if p.get("is_shadow") else "—",
                "Utilisateur": p.get("username") or "—",
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        label="⬇️ Télécharger en CSV",
        data=df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv",
    )

    # Détail features
    with st.expander("🔍 Voir les features d'une prédiction"):
        pred_ids = {str(p["id"]): p for p in predictions}
        selected_id = st.selectbox("Prédiction ID", list(pred_ids.keys()))
        if selected_id:
            p = pred_ids[selected_id]
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**Features d'entrée :**")
                st.json(p.get("input_features", {}))
            with col_r:
                st.markdown("**Résultat :**")
                st.json(
                    {
                        "prediction": p.get("prediction_result"),
                        "probabilities": p.get("probabilities"),
                    }
                )
                if p.get("error_message"):
                    st.error(f"Erreur : {p['error_message']}")

# --- Pagination ---
st.divider()
col_prev, col_info, col_next = st.columns([1, 2, 1])
with col_prev:
    if st.session_state["pred_offset"] > 0:
        if st.button("← Précédent", use_container_width=True):
            st.session_state["pred_offset"] = max(0, st.session_state["pred_offset"] - limit)
            st.rerun()
with col_info:
    current_page = st.session_state["pred_offset"] // limit + 1
    total_pages = max(1, (total + limit - 1) // limit)
    st.caption(f"Page {current_page} / {total_pages}")
with col_next:
    if st.session_state["pred_offset"] + limit < total:
        if st.button("Suivant →", use_container_width=True):
            st.session_state["pred_offset"] += limit
            st.rerun()

# --- Maintenance RGPD (admin uniquement) ---
if st.session_state.get("is_admin", False):
    st.divider()
    with st.expander("🗑️ Maintenance RGPD — Purge des prédictions"):
        st.caption(
            "Supprime définitivement les prédictions anciennes. "
            "Utilisez **Simuler** avant de confirmer."
        )

        col_m1, col_m2 = st.columns(2)
        purge_days = col_m1.slider(
            "Purger les prédictions antérieures à",
            min_value=7,
            max_value=365,
            value=90,
            format="%d jours",
            key="purge_days_slider",
        )
        purge_model_sel = col_m2.selectbox(
            "Filtrer par modèle (optionnel)",
            ["(tous)"] + model_names,
            key="purge_model_sel",
        )
        purge_model_name = None if purge_model_sel == "(tous)" else purge_model_sel

        col_sim, col_purge = st.columns(2)

        if col_sim.button("🔍 Simuler (dry_run)", key="purge_simulate", use_container_width=True):
            try:
                result = client.purge_predictions(
                    older_than_days=purge_days,
                    model_name=purge_model_name,
                    dry_run=True,
                )
                st.info(
                    f"Simulation : **{result['deleted_count']}** prédiction(s) seraient supprimées."
                )
                if result.get("oldest_remaining"):
                    st.caption(f"Prédiction la plus ancienne restante : {result['oldest_remaining']}")
                if result.get("models_affected"):
                    st.caption(f"Modèles affectés : {', '.join(result['models_affected'])}")
                if result.get("linked_observed_results_count", 0) > 0:
                    st.warning(
                        f"⚠️ {result['linked_observed_results_count']} résultat(s) observé(s) "
                        "lié(s) seraient perdus (perte de données de performance historiques)."
                    )
            except Exception as exc:
                st.error(f"Erreur lors de la simulation : {exc}")

        @st.dialog("⚠️ Confirmer la purge définitive")
        def _confirm_purge_dialog():
            st.warning(
                f"Vous allez **supprimer définitivement** toutes les prédictions "
                f"antérieures à **{purge_days} jours**."
            )
            if purge_model_name:
                st.info(f"Modèle ciblé : **{purge_model_name}**")
            else:
                st.info("Tous les modèles sont ciblés.")
            st.markdown("Cette action est **irréversible**.")
            if st.button("Confirmer la suppression", type="primary", key="purge_dialog_confirm"):
                try:
                    result = client.purge_predictions(
                        older_than_days=purge_days,
                        model_name=purge_model_name,
                        dry_run=False,
                    )
                    st.success(f"✅ {result['deleted_count']} prédiction(s) supprimée(s).")
                    if result.get("linked_observed_results_count", 0) > 0:
                        st.warning(
                            f"{result['linked_observed_results_count']} résultat(s) observé(s) "
                            "liés ont été perdus."
                        )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Erreur lors de la purge : {exc}")

        if col_purge.button(
            "⚠️ Confirmer la purge",
            key="purge_open_dialog",
            type="primary",
            use_container_width=True,
        ):
            _confirm_purge_dialog()
