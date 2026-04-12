"""
Gestion des modèles ML
"""
import os
import json
import streamlit as st
import pandas as pd
from utils.auth import require_auth, get_client

st.set_page_config(page_title="Models — PredictML", page_icon="🤖", layout="wide")
require_auth()

st.title("🤖 Gestion des modèles")

client = get_client()
is_admin = st.session_state.get("is_admin", False)

MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://localhost:5000")


@st.cache_data(ttl=15, show_spinner=False)
def fetch_models(api_url, token):
    c = get_client()
    return c.list_models()


def reload():
    st.cache_data.clear()
    st.rerun()


try:
    models = fetch_models(st.session_state.get("api_url"), st.session_state.get("api_token"))
except Exception as e:
    st.error(f"Impossible de charger les modèles : {e}")
    st.stop()

if not models:
    st.info("Aucun modèle disponible.")
    st.stop()

# Résumé
col1, col2, col3 = st.columns(3)
col1.metric("Total modèles", len(models))
col2.metric("En production", sum(1 for m in models if m.get("is_production")))
col3.metric("Avec MLflow", sum(1 for m in models if m.get("mlflow_run_id")))

st.divider()

# Filtre par tag
all_tags = sorted({t for m in models for t in (m.get("tags") or [])})
if all_tags:
    tag_filter = st.selectbox("Filtrer par tag", ["(tous)"] + all_tags, key="tag_filter")
    if tag_filter != "(tous)":
        models = [m for m in models if tag_filter in (m.get("tags") or [])]

# Tableau de synthèse
rows = []
for m in models:
    rows.append({
        "Nom": m.get("name", ""),
        "Version": m.get("version", ""),
        "Tags": ", ".join(m.get("tags") or []) or "—",
        "Algorithme": m.get("algorithm") or "—",
        "Accuracy": f"{m['accuracy']:.3f}" if m.get("accuracy") is not None else "—",
        "F1": f"{m['f1_score']:.3f}" if m.get("f1_score") is not None else "—",
        "Statut": "🟢 Production" if m.get("is_production") else ("✅ Actif" if m.get("is_active") else "⚫ Inactif"),
        "Créateur": m.get("creator_username") or "—",
        "Créé le": pd.to_datetime(m.get("created_at")).strftime("%Y-%m-%d") if m.get("created_at") else "—",
        "Dernière préd.": pd.to_datetime(m.get("last_seen")).strftime("%Y-%m-%d %H:%M") if m.get("last_seen") else "—",
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Détail et actions")

model_options = {f"{m['name']} v{m['version']}": m for m in models}
selected_label = st.selectbox("Sélectionner un modèle", list(model_options.keys()))
selected = model_options[selected_label]

# Détails
with st.expander("📋 Détails complets", expanded=True):
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"**Nom :** `{selected.get('name')}`")
        st.markdown(f"**Version :** `{selected.get('version')}`")
        st.markdown(f"**Description :** {selected.get('description') or '—'}")
        st.markdown(f"**Algorithme :** {selected.get('algorithm') or '—'}")
        st.markdown(f"**Dataset d'entraînement :** {selected.get('training_dataset') or '—'}")
        st.markdown(f"**Entraîné par :** {selected.get('trained_by') or '—'}")
        tags = selected.get("tags")
        st.markdown(f"**Tags :** {', '.join(tags) if tags else '—'}")
        webhook = selected.get("webhook_url")
        st.markdown(f"**Webhook URL :** `{webhook}`" if webhook else "**Webhook URL :** —")
    with col_r:
        st.markdown(f"**Accuracy :** {selected.get('accuracy') or '—'}")
        st.markdown(f"**F1 Score :** {selected.get('f1_score') or '—'}")
        st.markdown(f"**Precision :** {selected.get('precision') or '—'}")
        st.markdown(f"**Recall :** {selected.get('recall') or '—'}")
        st.markdown(f"**Nb features :** {selected.get('features_count') or '—'}")
        last_seen = selected.get("last_seen")
        st.markdown(f"**Dernière prédiction :** {pd.to_datetime(last_seen).strftime('%Y-%m-%d %H:%M') if last_seen else '—'}")
        classes = selected.get("classes")
        st.markdown(f"**Classes :** {classes if classes else '—'}")

    if selected.get("training_params"):
        st.markdown("**Hyperparamètres :**")
        st.json(selected["training_params"])

    mlflow_id = selected.get("mlflow_run_id")
    if mlflow_id:
        mlflow_link = f"{MLFLOW_URL}/#/experiments/0/runs/{mlflow_id}"
        st.markdown(f"**MLflow run :** [{mlflow_id}]({mlflow_link})")
    else:
        st.markdown("**MLflow run :** —")

    minio_key = selected.get("minio_object_key")
    if minio_key:
        st.markdown(f"**MinIO object :** `{selected.get('minio_bucket')}/{minio_key}`")
        size = selected.get("file_size_bytes")
        if size:
            st.markdown(f"**Taille fichier :** {size / 1024:.1f} KB")

# Actions
if is_admin:
    st.subheader("Actions admin")
    col_p, col_d = st.columns(2)

    # Passer en production
    if not selected.get("is_production"):
        if col_p.button("🚀 Passer en production", use_container_width=True, type="primary"):
            try:
                client.update_model(selected["name"], selected["version"], {"is_production": True})
                st.success(f"**{selected['name']} v{selected['version']}** est maintenant en production.")
                reload()
            except Exception as e:
                st.error(f"Erreur : {e}")
    else:
        col_p.info("🟢 Déjà en production")

    # Supprimer
    if col_d.button("🗑️ Supprimer cette version", use_container_width=True, type="secondary"):
        st.session_state["confirm_delete_model"] = f"{selected['name']}:{selected['version']}"

    key = f"{selected['name']}:{selected['version']}"
    if st.session_state.get("confirm_delete_model") == key:
        st.warning(f"Supprimer **{selected['name']} v{selected['version']}** ? (fichier MinIO + run MLflow)")
        c1, c2 = st.columns(2)
        if c1.button("Oui, supprimer", type="primary"):
            try:
                client.delete_model_version(selected["name"], selected["version"])
                st.success("Modèle supprimé.")
                st.session_state.pop("confirm_delete_model", None)
                reload()
            except Exception as e:
                st.error(f"Erreur : {e}")
        if c2.button("Annuler"):
            st.session_state.pop("confirm_delete_model", None)
            st.rerun()

    # Modifier tags et webhook
    with st.expander("✏️ Modifier les métadonnées"):
        new_webhook = st.text_input(
            "Webhook URL",
            value=selected.get("webhook_url") or "",
            placeholder="https://example.com/webhook",
        )
        new_tags_raw = st.text_input(
            "Tags (séparés par des virgules)",
            value=", ".join(selected.get("tags") or []),
            placeholder="production, finance, v2",
        )
        if st.button("💾 Enregistrer", key="save_meta"):
            patch = {}
            current_webhook = selected.get("webhook_url") or ""
            if new_webhook != current_webhook:
                patch["webhook_url"] = new_webhook if new_webhook else None
            new_tags = [t.strip() for t in new_tags_raw.split(",") if t.strip()]
            if new_tags != (selected.get("tags") or []):
                patch["tags"] = new_tags if new_tags else None
            if patch:
                try:
                    client.update_model(selected["name"], selected["version"], patch)
                    st.success("Métadonnées mises à jour.")
                    reload()
                except Exception as e:
                    st.error(f"Erreur : {e}")
            else:
                st.info("Aucun changement détecté.")
