"""
Page d'aide avec chatbot LLM — PredictML Admin
"""

import os

import anthropic
import streamlit as st
from utils.auth import require_auth
from utils.docs_loader import build_system_prompt, load_all_docs
from utils.metrics_help import METRIC_HELP

st.set_page_config(page_title="Aide & Assistant IA — PredictML", page_icon="💬", layout="wide")
require_auth()

# ── Constantes ────────────────────────────────────────────────────────────────

MODEL_ID = "claude-sonnet-4-6"

QUICK_TOPICS = [
    ("🛠️ Générer un train.py sklearn", "Génère un script train.py complet compatible avec PredictML. Utilise RandomForestClassifier sur le dataset Iris, avec MLflow tracking, et respecte le contrat de variables d'environnement (TRAIN_START_DATE, TRAIN_END_DATE, OUTPUT_MODEL_PATH)."),
    ("🌐 Appels API Python", "Montre-moi un exemple complet en Python (requests) pour : uploader un modèle, faire une prédiction, et récupérer l'historique des prédictions."),
    ("📊 Expliquer les KPIs", "Explique les principaux indicateurs de performance disponibles dans le dashboard : accuracy, F1, taux d'erreur, latence P95, drift, et les métriques A/B."),
    ("🔄 Ré-entraînement planifié", "Comment configurer un ré-entraînement automatique chaque semaine avec APScheduler et l'endpoint PATCH /models/{name}/{version}/schedule ?"),
    ("🧪 Comprendre l'A/B testing", "Comment fonctionne le mode A/B testing dans PredictML ? Comment interpréter le résultat de ab-compare (p-value, winner, test Chi-²) ?"),
    ("🛡️ Purge RGPD", "Comment utiliser l'endpoint DELETE /predictions/purge pour supprimer les anciennes prédictions tout en respectant le RGPD ?"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def get_anthropic_client() -> anthropic.Anthropic | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)


def stream_answer(client: anthropic.Anthropic, messages: list, system_prompt: str):
    with client.messages.stream(
        model=MODEL_ID,
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=messages,
    ) as stream:
        for chunk in stream.text_stream:
            yield chunk


def render_chat_history():
    for msg in st.session_state["help_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ── Initialisation session state ──────────────────────────────────────────────

if "help_messages" not in st.session_state:
    st.session_state["help_messages"] = []

if "help_pending_prompt" not in st.session_state:
    st.session_state["help_pending_prompt"] = None

# ── Chargement docs & prompt système ─────────────────────────────────────────

docs = load_all_docs()

# Injecter aussi les définitions des KPIs dans le prompt
kpis_section = "\n".join(f"- **{k}** : {v}" for k, v in METRIC_HELP.items())
docs["KPIs_Dashboard"] = f"## Définitions des KPIs du dashboard\n\n{kpis_section}"

system_prompt = build_system_prompt(docs)
anthropic_client = get_anthropic_client()

# ── En-tête ───────────────────────────────────────────────────────────────────

st.title("💬 Aide & Assistant IA")
st.markdown(
    "Posez vos questions sur l'utilisation de PredictML : scripts d'entraînement, "
    "appels API, indicateurs, configuration du dashboard…"
)
st.divider()

# ── Layout principal ──────────────────────────────────────────────────────────

col_docs, col_chat = st.columns([4, 6], gap="large")

# ═══════════════════════════════════════════════════════════
# COLONNE GAUCHE — Visualiseur de documentation
# ═══════════════════════════════════════════════════════════

with col_docs:
    st.subheader("📚 Documentation")

    if not docs:
        st.warning(
            "Les fichiers de documentation ne sont pas accessibles.\n\n"
            "En mode Docker, assurez-vous que le volume `./documentation:/app/documentation:ro` "
            "est bien déclaré dans `docker-compose.yml` pour le service `streamlit`."
        )
    else:
        doc_names = [k for k in docs if k != "KPIs_Dashboard"]
        labels = {
            "API_REFERENCE": "📖 Référence API",
            "ARCHITECTURE": "🏗️ Architecture",
            "BEGINNER_GUIDE": "🎓 Guide débutant",
            "DATABASE": "🗄️ Base de données",
            "DOCKER": "🐳 Docker",
            "QUICKSTART": "⚡ Démarrage rapide",
            "README": "🏠 README",
            "CODING_STANDARDS": "✅ Standards de code",
        }
        display_names = [labels.get(n, n) for n in doc_names]
        name_map = dict(zip(display_names, doc_names))

        selected_label = st.selectbox(
            "Choisir un document",
            display_names,
            key="help_doc_select",
        )
        selected_key = name_map[selected_label]

        with st.container(height=700, border=True):
            st.markdown(docs[selected_key])

# ═══════════════════════════════════════════════════════════
# COLONNE DROITE — Chatbot LLM
# ═══════════════════════════════════════════════════════════

with col_chat:
    st.subheader("💬 Assistant IA")

    # ── Vérification de la clé API ────────────────────────
    if anthropic_client is None:
        st.warning(
            "**La clé `ANTHROPIC_API_KEY` n'est pas configurée.**\n\n"
            "Pour activer le chatbot :\n\n"
            "1. Créez ou éditez le fichier **`.env`** à la racine du projet :\n"
            "   ```\n"
            "   ANTHROPIC_API_KEY=sk-ant-...\n"
            "   ```\n"
            "2. Relancez le container Streamlit :\n"
            "   ```bash\n"
            "   docker-compose up -d streamlit\n"
            "   ```\n\n"
            "La clé est automatiquement transmise au container via `docker-compose.yml`.\n\n"
            "---\n"
            "En attendant, vous pouvez consulter la documentation à gauche."
        )

    # ── Boutons de sujets rapides ─────────────────────────
    with st.expander("⚡ Sujets rapides", expanded=True):
        cols = st.columns(2)
        for i, (label, prompt) in enumerate(QUICK_TOPICS):
            if cols[i % 2].button(label, key=f"quick_{i}", use_container_width=True):
                st.session_state["help_pending_prompt"] = prompt

    # ── Actions de gestion ────────────────────────────────
    btn_col1, btn_col2 = st.columns([1, 3])
    if btn_col1.button("🗑️ Nouvelle conversation", key="help_clear"):
        st.session_state["help_messages"] = []
        st.session_state["help_pending_prompt"] = None
        st.rerun()

    if st.session_state["help_messages"]:
        btn_col2.caption(
            f"{len(st.session_state['help_messages']) // 2} échange(s) dans la conversation"
        )

    st.divider()

    # ── Historique du chat ────────────────────────────────
    render_chat_history()

    # ── Traitement d'un prompt en attente (bouton rapide) ─
    if st.session_state["help_pending_prompt"] and anthropic_client:
        pending = st.session_state["help_pending_prompt"]
        st.session_state["help_pending_prompt"] = None

        st.session_state["help_messages"].append({"role": "user", "content": pending})
        with st.chat_message("user"):
            st.markdown(pending)

        with st.chat_message("assistant"):
            response_text = st.write_stream(
                stream_answer(
                    anthropic_client,
                    st.session_state["help_messages"],
                    system_prompt,
                )
            )
        st.session_state["help_messages"].append(
            {"role": "assistant", "content": response_text}
        )
        st.rerun()

    # ── Zone de saisie ────────────────────────────────────
    user_input = st.chat_input(
        "Posez votre question…",
        disabled=(anthropic_client is None),
        key="help_chat_input",
    )

    if user_input and anthropic_client:
        st.session_state["help_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response_text = st.write_stream(
                stream_answer(
                    anthropic_client,
                    st.session_state["help_messages"],
                    system_prompt,
                )
            )
        st.session_state["help_messages"].append(
            {"role": "assistant", "content": response_text}
        )
        st.rerun()
