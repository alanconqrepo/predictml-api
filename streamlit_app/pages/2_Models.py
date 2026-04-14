"""
Gestion des modèles ML
"""

import os
import json
import streamlit as st
import pandas as pd
from utils.auth import require_auth, get_client

# --- Helpers historique ---

_ACTION_ICONS = {
    "created": "🟦",
    "updated": "🟨",
    "set_production": "🟩",
    "deprecated": "🟫",
    "deleted": "🟥",
    "rollback": "🔁",
}


def _action_badge(action: str) -> str:
    icon = _ACTION_ICONS.get(action, "⬜")
    return f"{icon} `{action}`"


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

_DEPLOY_BADGE = {
    "ab_test": "🟠 A/B",
    "shadow": "🟣 Shadow",
    "production": "🟢 Prod",
}


# Tableau de synthèse
rows = []
for m in models:
    mode = m.get("deployment_mode")
    weight = m.get("traffic_weight")
    if mode == "ab_test":
        statut = f"🟠 A/B ({weight:.0%})" if weight is not None else "🟠 A/B"
    elif mode == "shadow":
        statut = "🟣 Shadow"
    elif m.get("is_production"):
        statut = "🟢 Production"
    elif m.get("is_active"):
        statut = "✅ Actif"
    else:
        statut = "⚫ Inactif"

    rows.append(
        {
            "Nom": m.get("name", ""),
            "Version": m.get("version", ""),
            "Tags": ", ".join(m.get("tags") or []) or "—",
            "Algorithme": m.get("algorithm") or "—",
            "Accuracy": f"{m['accuracy']:.3f}" if m.get("accuracy") is not None else "—",
            "F1": f"{m['f1_score']:.3f}" if m.get("f1_score") is not None else "—",
            "Statut": statut,
            "Créateur": m.get("creator_username") or "—",
            "Créé le": (
                pd.to_datetime(m.get("created_at")).strftime("%Y-%m-%d")
                if m.get("created_at")
                else "—"
            ),
            "Dernière préd.": (
                pd.to_datetime(m.get("last_seen")).strftime("%Y-%m-%d %H:%M")
                if m.get("last_seen")
                else "—"
            ),
        }
    )

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
        st.markdown(
            f"**Dernière prédiction :** {pd.to_datetime(last_seen).strftime('%Y-%m-%d %H:%M') if last_seen else '—'}"
        )
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
                st.success(
                    f"**{selected['name']} v{selected['version']}** est maintenant en production."
                )
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
        st.warning(
            f"Supprimer **{selected['name']} v{selected['version']}** ? (fichier MinIO + run MLflow)"
        )
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

    # Modifier tags, webhook et déploiement
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

        st.markdown("**Déploiement A/B / Shadow**")
        deploy_options = ["(inchangé)", "ab_test", "shadow", "production"]
        new_deploy_mode = st.selectbox(
            "Mode de déploiement",
            deploy_options,
            key="deploy_mode_select",
            help="ab_test = routage pondéré, shadow = exécution silencieuse en background",
        )
        new_traffic_weight = None
        if new_deploy_mode == "ab_test":
            new_traffic_weight = st.number_input(
                "Poids du trafic (0.0 – 1.0)",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=float(selected.get("traffic_weight") or 0.5),
                key="traffic_weight_input",
                help="Fraction du trafic routé vers cette version (ex: 0.3 = 30%)",
            )

        if st.button("💾 Enregistrer", key="save_meta"):
            patch = {}
            current_webhook = selected.get("webhook_url") or ""
            if new_webhook != current_webhook:
                patch["webhook_url"] = new_webhook if new_webhook else None
            new_tags = [t.strip() for t in new_tags_raw.split(",") if t.strip()]
            if new_tags != (selected.get("tags") or []):
                patch["tags"] = new_tags if new_tags else None
            if new_deploy_mode != "(inchangé)":
                patch["deployment_mode"] = new_deploy_mode
                if new_deploy_mode == "ab_test" and new_traffic_weight is not None:
                    patch["traffic_weight"] = new_traffic_weight
            if patch:
                try:
                    client.update_model(selected["name"], selected["version"], patch)
                    st.success("Métadonnées mises à jour.")
                    reload()
                except Exception as e:
                    st.error(f"Erreur : {e}")
            else:
                st.info("Aucun changement détecté.")

    # Ré-entraînement
    if selected.get("train_script_object_key"):
        st.divider()
        if st.button("🔄 Ré-entraîner", use_container_width=True, key="retrain_btn"):
            current = st.session_state.get("show_retrain_form")
            toggle_key = f"{selected['name']}:{selected['version']}"
            if current == toggle_key:
                st.session_state.pop("show_retrain_form", None)
            else:
                st.session_state["show_retrain_form"] = toggle_key

        retrain_key = f"{selected['name']}:{selected['version']}"
        if st.session_state.get("show_retrain_form") == retrain_key:
            with st.form("retrain_form"):
                st.markdown(f"**Ré-entraîner** `{selected['name']}` v`{selected['version']}`")
                col_s, col_e = st.columns(2)
                with col_s:
                    start_date = st.date_input("Date de début", key="retrain_start")
                with col_e:
                    end_date = st.date_input("Date de fin", key="retrain_end")
                new_version_input = st.text_input(
                    "Nouvelle version (laisser vide = auto-généré)",
                    value="",
                    placeholder=f"{selected['version']}-retrain-YYYYMMDDHHMMSS",
                    key="retrain_new_version",
                )
                set_prod = st.checkbox(
                    "Mettre en production après entraînement",
                    value=False,
                    key="retrain_set_prod",
                )
                submitted = st.form_submit_button("🚀 Lancer le ré-entraînement", type="primary")

            if submitted:
                if start_date > end_date:
                    st.error("La date de début doit être antérieure à la date de fin.")
                else:
                    with st.spinner("Ré-entraînement en cours… (peut prendre jusqu'à 10 minutes)"):
                        try:
                            result = client.retrain_model(
                                name=selected["name"],
                                version=selected["version"],
                                start_date=str(start_date),
                                end_date=str(end_date),
                                new_version=new_version_input.strip() or None,
                                set_production=set_prod,
                            )
                            st.session_state.pop("show_retrain_form", None)
                            if result.get("success"):
                                st.success(
                                    f"Ré-entraînement réussi ! "
                                    f"Nouvelle version : **{result['new_version']}**"
                                )
                            else:
                                st.error(
                                    f"Échec du ré-entraînement : "
                                    f"{result.get('error', 'Erreur inconnue')}"
                                )
                            with st.expander("📋 Logs stdout", expanded=not result.get("success")):
                                st.code(result.get("stdout", "(vide)"), language="text")
                            with st.expander("⚠️ Logs stderr", expanded=not result.get("success")):
                                st.code(result.get("stderr", "(vide)"), language="text")
                            if result.get("success"):
                                reload()
                        except Exception as e:
                            st.error(f"Erreur lors du ré-entraînement : {e}")

    # Historique des modifications
    with st.expander("📜 Historique des modifications"):
        try:
            history_data = client.get_model_history(selected["name"], selected["version"], limit=20)
            entries = history_data.get("entries", [])
            total_hist = history_data.get("total", 0)
        except Exception as e:
            st.error(f"Impossible de charger l'historique : {e}")
            entries = []
            total_hist = 0

        if not entries:
            st.info("Aucun historique disponible pour cette version.")
        else:
            st.caption(f"{total_hist} entrée(s) au total — affichage des 20 dernières")
            for entry in entries:
                ts = pd.to_datetime(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S UTC")
                badge = _action_badge(entry["action"])
                changed = ", ".join(entry.get("changed_fields") or []) or "—"
                who = entry.get("changed_by_username") or "inconnu"

                col_info, col_btn = st.columns([5, 1])
                with col_info:
                    st.markdown(
                        f"**{ts}** — {badge} — par **{who}**  \n" f"Champs modifiés : `{changed}`"
                    )
                with col_btn:
                    if is_admin:
                        if st.button(
                            "↩ Rollback",
                            key=f"rollback_btn_{entry['id']}",
                            type="secondary",
                            help=f"Restaurer l'état de l'entrée #{entry['id']}",
                        ):
                            st.session_state["confirm_rollback_id"] = entry["id"]
                            st.session_state["confirm_rollback_model"] = selected["name"]
                            st.session_state["confirm_rollback_version"] = selected["version"]

                with st.expander(f"Snapshot #{entry['id']}", expanded=False):
                    st.json(entry["snapshot"])

                st.divider()

            # Dialog de confirmation du rollback
            confirm_id = st.session_state.get("confirm_rollback_id")
            confirm_model = st.session_state.get("confirm_rollback_model")
            confirm_version = st.session_state.get("confirm_rollback_version")
            if (
                confirm_id is not None
                and confirm_model == selected["name"]
                and confirm_version == selected["version"]
            ):
                st.warning(
                    f"Restaurer les métadonnées de **{confirm_model} v{confirm_version}** "
                    f"à l'état capturé dans l'entrée **#{confirm_id}** ?  \n"
                    "Cette action est irréversible (mais loguée dans l'historique)."
                )
                c1, c2 = st.columns(2)
                if c1.button("✅ Oui, restaurer", type="primary", key="confirm_rollback_yes"):
                    try:
                        result = client.rollback_model(confirm_model, confirm_version, confirm_id)
                        st.success(
                            f"Rollback effectué. Nouvelle entrée d'historique "
                            f"**#{result['new_history_id']}** créée."
                        )
                        st.session_state.pop("confirm_rollback_id", None)
                        st.session_state.pop("confirm_rollback_model", None)
                        st.session_state.pop("confirm_rollback_version", None)
                        reload()
                    except Exception as e:
                        st.error(f"Erreur lors du rollback : {e}")
                if c2.button("❌ Annuler", key="confirm_rollback_no"):
                    st.session_state.pop("confirm_rollback_id", None)
                    st.session_state.pop("confirm_rollback_model", None)
                    st.session_state.pop("confirm_rollback_version", None)
                    st.rerun()
