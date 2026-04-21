"""
Historique des prédictions avec filtres
"""
import json
from datetime import datetime, timedelta, date
import streamlit as st
import pandas as pd
from utils.auth import require_auth, get_client

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

st.caption(f"**{total}** prédictions trouvées — affichage {st.session_state['pred_offset'] + 1}–{min(st.session_state['pred_offset'] + limit, total)}")

if not predictions:
    st.info("Aucune prédiction pour ces critères.")
else:
    rows = []
    for p in predictions:
        rows.append({
            "ID": p.get("id"),
            "Timestamp": pd.to_datetime(p.get("timestamp")).strftime("%Y-%m-%d %H:%M:%S") if p.get("timestamp") else "—",
            "Modèle": p.get("model_name", ""),
            "Version": p.get("model_version") or "—",
            "id_obs": p.get("id_obs") or "—",
            "Résultat": str(p.get("prediction_result", "")),
            "Temps (ms)": f"{p['response_time_ms']:.1f}" if p.get("response_time_ms") is not None else "—",
            "Statut": "✅" if p.get("status") == "success" else "❌",
            "Shadow": "🔮" if p.get("is_shadow") else "—",
            "Utilisateur": p.get("username") or "—",
        })

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
                st.json({"prediction": p.get("prediction_result"), "probabilities": p.get("probabilities")})
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
