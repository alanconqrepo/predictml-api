"""
Page d'aide avec chatbot LLM (function calling natif) — PredictML Admin
"""

import json
import os

import anthropic
import streamlit as st
from utils.auth import require_auth
from utils.docs_loader import build_system_prompt, load_all_docs, load_source_snippets
from utils.i18n import t
from utils.metrics_help import METRIC_HELP
from utils.tools import (
    TOOL_DEFINITIONS,
    build_tool_summary,
    execute_tool,
    render_tool_input,
    render_tool_result,
    tool_expander_label,
)

st.set_page_config(page_title=t("aide.page_title"), page_icon="💬", layout="wide")
require_auth()


@st.dialog("📚 Documentation", width="large")
def _doc_popup(content: str, label: str) -> None:
    st.caption(label)
    st.divider()
    st.markdown(content)


@st.dialog("🔧 Code source", width="large")
def _src_popup(filename: str, content: str) -> None:
    st.caption(filename)
    st.divider()
    st.code(content, language="python")

# ── Constantes ────────────────────────────────────────────────────────────────

MODEL_ID = "claude-sonnet-4-6"

QUICK_TOPICS = [
    (
        t("aide.quick_topics.prod_status"),
        "Quels modèles sont actuellement en production ? "
        "Montre-moi leurs métriques (accuracy, F1, latence) et indique s'il y a des problèmes de drift.",
    ),
    (
        t("aide.quick_topics.prediction_stats"),
        "Donne-moi un résumé des prédictions des 7 derniers jours : volume par modèle, "
        "taux d'erreur, latence moyenne et P95.",
    ),
    (
        t("aide.quick_topics.generate_train_py"),
        "Génère un script train.py complet et compatible PredictML pour un modèle de classification "
        "avec RandomForestClassifier. Respecte le contrat de variables d'environnement "
        "(TRAIN_START_DATE, TRAIN_END_DATE, OUTPUT_MODEL_PATH) et inclus la sortie JSON sur stdout.",
    ),
    (
        t("aide.quick_topics.api_calls"),
        "Montre-moi le workflow Python complet avec requests : uploader un modèle .joblib, "
        "le passer en production, faire une prédiction unitaire et batch, "
        "puis enregistrer les résultats observés.",
    ),
    (
        t("aide.quick_topics.interpret_kpis"),
        "Explique les principaux indicateurs de performance du dashboard PredictML : "
        "accuracy, F1, taux d'erreur, latence P95, Brier Score, drift Z-score et PSI, p-value A/B. "
        "Donne les seuils d'alerte recommandés.",
    ),
    (
        t("aide.quick_topics.configure_ab"),
        "Comment configurer un A/B test entre deux versions d'un modèle ? "
        "Explique deployment_mode, traffic_weight, l'interprétation statistique "
        "(p-value, winner, min_samples_needed) et quand promouvoir.",
    ),
    (
        t("aide.quick_topics.auto_retrain"),
        "Comment planifier un ré-entraînement automatique hebdomadaire avec une expression cron ? "
        "Inclus la configuration du schedule, la politique d'auto-promotion, "
        "et le ré-entraînement déclenché par drift.",
    ),
    (
        t("aide.quick_topics.detect_drift"),
        "Comment configurer et interpréter la détection de drift dans PredictML ? "
        "Explique feature_baseline, Z-score, PSI, statuts ok/warning/critical, "
        "et le retrain automatique quand le drift est critique.",
    ),
    (
        t("aide.quick_topics.manage_users"),
        "Comment gérer les utilisateurs ? Montre-moi les utilisateurs actuellement actifs, "
        "leur rôle et quota. Explique aussi comment créer un compte et renouveler un token.",
    ),
    (
        t("aide.quick_topics.golden_tests"),
        "Comment créer des golden tests et les intégrer dans la politique d'auto-promotion ? "
        "Montre les cas de test existants si disponibles.",
    ),
    (
        t("aide.quick_topics.architecture"),
        "Explique l'architecture de PredictML : les 7 services Docker, comment ils interagissent, "
        "et le flux de données complet d'une prédiction.",
    ),
    (
        t("aide.quick_topics.get_started"),
        "Je veux démarrer avec PredictML de zéro. Guide-moi étape par étape : "
        "installation Docker, premier modèle sklearn, upload, mise en production, prédiction.",
    ),
]


# ── Client Anthropic ──────────────────────────────────────────────────────────


def get_anthropic_client() -> anthropic.Anthropic | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return anthropic.Anthropic(api_key=key) if key else None


# ── Boucle agent avec tool use ────────────────────────────────────────────────


def run_agent_turn(
    client: anthropic.Anthropic,
    raw_messages: list,
    system_prompt: str,
    api_url: str,
    token: str,
) -> tuple[str, list]:
    """
    Exécute un tour complet de l'agent avec support du function calling.

    Render les appels d'outils inline dans le chat_message courant.
    Retourne (texte_final, liste_de_résumés_d'outils).
    """
    current_messages = list(raw_messages)
    final_text = ""
    tool_summaries = []

    while True:
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=TOOL_DEFINITIONS,
            messages=current_messages,
        )

        # Texte intermédiaire émis avant les appels d'outils
        turn_text = "".join(
            b.text for b in response.content if hasattr(b, "text") and b.text
        )

        if response.stop_reason == "tool_use":
            # Afficher le texte de réflexion intermédiaire
            if turn_text:
                st.markdown(turn_text)
                final_text += turn_text + "\n\n"

            # Traiter chaque appel d'outil
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                label = tool_expander_label(block.name, block.input)
                with st.expander(label, expanded=True):
                    render_tool_input(block.name, block.input)
                    with st.spinner(t("aide.tool_running")):
                        result = execute_tool(block.name, block.input, api_url, token)
                    st.divider()
                    render_tool_result(block.name, result)

                tool_summaries.append(build_tool_summary(block.name, block.input, result))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

            # Continuer la conversation avec les résultats
            current_messages = current_messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ]

        else:
            # Réponse finale
            final_text += turn_text
            return final_text, tool_summaries


# ── Rendu de l'historique ─────────────────────────────────────────────────────


def render_chat_history() -> None:
    for msg in st.session_state["help_messages"]:
        with st.chat_message(msg["role"]):
            # Pour les messages assistant, réafficher les résumés d'outils (repliés)
            if msg["role"] == "assistant":
                for summary in msg.get("tool_summaries", []):
                    with st.expander(summary["label"], expanded=False):
                        st.caption(t("aide.tool_summary_request"))
                        if summary["type"] == "sql":
                            st.code(summary.get("query", ""), language="sql")
                        else:
                            st.code(
                                f"{summary.get('method','')} {summary.get('endpoint','')}",
                                language="http",
                            )
                        if summary.get("result_preview"):
                            st.caption(t("aide.tool_summary_preview"))
                            st.code(summary["result_preview"])
            st.markdown(msg["content"])


# ── Helpers de gestion de l'état des messages ────────────────────────────────


def _user_to_raw(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant_to_raw(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _sync_raw_from_display() -> list:
    """Reconstruit raw_messages depuis help_messages (pour compatibilité Anthropic API)."""
    raw = []
    for msg in st.session_state["help_messages"]:
        raw.append({"role": msg["role"], "content": msg["content"]})
    return raw


# ── Initialisation session state ──────────────────────────────────────────────

if "help_messages" not in st.session_state:
    st.session_state["help_messages"] = []  # [{role, content, tool_summaries?}]

if "help_pending_prompt" not in st.session_state:
    st.session_state["help_pending_prompt"] = None

# ── Chargement docs, code source & prompt système ─────────────────────────────

docs = load_all_docs()
snippets = load_source_snippets()

kpis_section = "\n".join(f"- **{k}** : {v}" for k, v in METRIC_HELP.items())
docs["KPIs_Dashboard"] = f"## Définitions des KPIs du dashboard\n\n{kpis_section}"

system_prompt = build_system_prompt(docs, snippets)
anthropic_client = get_anthropic_client()

# URL et token de la session courante
_api_url: str = st.session_state.get("api_url", "http://localhost:8000")
_token: str = st.session_state.get("api_token", "")

# ── En-tête ───────────────────────────────────────────────────────────────────

st.title(t("aide.title"))
st.markdown(t("aide.subtitle"))

ctx_parts = []
if docs:
    ctx_parts.append(f"📚 {len(docs)} docs")
if snippets:
    ctx_parts.append(f"🔧 {len(snippets)} fichiers source")
ctx_parts.append("🗄️ SQL" if os.environ.get("POSTGRES_HOST") else "🗄️ SQL (local)")
ctx_parts.append("🌐 API")
st.caption(t("aide.tools_available", tools=" · ".join(ctx_parts)))

st.divider()

# ── Layout principal — 3 sections pleine largeur ─────────────────────────────

# ═══════════════════════════════════════════════════════════
# SECTION 1 — Documentation
# ═══════════════════════════════════════════════════════════

with st.expander(t("aide.doc_expander"), expanded=True):
    if not docs:
        st.warning(t("aide.doc_not_available"))
    else:
        labels = {
            "API_REFERENCE":      "📖 Référence API",
            "ARCHITECTURE":       "🏗️ Architecture",
            "BEGINNER_GUIDE":     "🎓 Guide débutant",
            "DATABASE":           "🗄️ Base de données",
            "DOCKER":             "🐳 Docker",
            "QUICKSTART":         "⚡ Démarrage rapide",
            "README":             "🏠 README",
            "CODING_STANDARDS":   "✅ Standards de code",
            "DASHBOARD_GUIDE":    "🖥️ Guide du dashboard",
            "TRAIN_SCRIPT_GUIDE": "🛠️ Guide train.py",
            "KPIS_REFERENCE":     "📊 Référence KPIs",
            "FAQ":                "❓ FAQ",
        }
        doc_names = [k for k in docs if k != "KPIs_Dashboard"]
        display_names = [labels.get(n, n) for n in doc_names]
        name_map = dict(zip(display_names, doc_names))

        col_sel, col_btn, col_dl = st.columns([6, 1, 1], vertical_alignment="bottom")
        selected_label = col_sel.selectbox(t("aide.doc_select_label"), display_names, key="help_doc_select")
        selected_key = name_map[selected_label]

        if col_btn.button(t("aide.doc_enlarge_btn"), key="open_doc_popup", width='stretch'):
            _doc_popup(docs[selected_key], selected_label)

        col_dl.download_button(
            t("aide.doc_download_btn"),
            data=docs[selected_key].encode("utf-8"),
            file_name=f"{selected_key.lower()}.md",
            mime="text/markdown",
            width='stretch',
            key="dl_doc",
        )

        with st.container(height=500, border=True):
            st.markdown(docs[selected_key])

# ═══════════════════════════════════════════════════════════
# SECTION 2 — Code source
# ═══════════════════════════════════════════════════════════

if snippets:
    with st.expander(t("aide.src_expander", count=len(snippets)), expanded=False):
        col_src, col_src_btn, col_src_dl = st.columns([6, 1, 1], vertical_alignment="bottom")
        selected_src = col_src.selectbox(t("aide.src_file_label"), list(snippets.keys()), key="help_src_select")
        if col_src_btn.button(t("aide.doc_enlarge_btn"), key="open_src_popup", width='stretch'):
            _src_popup(selected_src, snippets[selected_src])
        col_src_dl.download_button(
            t("aide.src_download_btn"),
            data=snippets[selected_src].encode("utf-8"),
            file_name=selected_src.split("/")[-1],
            mime="text/x-python",
            width='stretch',
            key="dl_src",
        )
        st.code(snippets[selected_src], language="python")

# ═══════════════════════════════════════════════════════════
# SECTION 3 — Chatbot LLM
# ═══════════════════════════════════════════════════════════

with st.expander(t("aide.chat_expander"), expanded=True):
    if anthropic_client is None:
        st.warning(t("aide.no_api_key"))

    with st.expander(
        t("aide.quick_topics_expander"),
        expanded=(len(st.session_state["help_messages"]) == 0),
    ):
        cols = st.columns(3)
        for i, (label, prompt) in enumerate(QUICK_TOPICS):
            if cols[i % 3].button(label, key=f"quick_{i}", width='stretch'):
                st.session_state["help_pending_prompt"] = prompt

    btn_col1, btn_col2 = st.columns([2, 5])
    if btn_col1.button(t("aide.new_conversation_btn"), key="help_clear"):
        st.session_state["help_messages"] = []
        st.session_state["help_pending_prompt"] = None
        st.rerun()

    n_exchanges = len(st.session_state["help_messages"]) // 2
    if n_exchanges > 0:
        btn_col2.caption(t("aide.exchange_count", count=n_exchanges))

    st.divider()

    render_chat_history()

    # ── Traitement du message (prompt rapide ou saisie) ───

    user_input: str | None = None

    if st.session_state["help_pending_prompt"] and anthropic_client:
        user_input = st.session_state["help_pending_prompt"]
        st.session_state["help_pending_prompt"] = None

    typed = st.chat_input(
        placeholder=t("aide.chat_placeholder"),
        disabled=(anthropic_client is None),
        key="help_chat_input",
    )
    if typed:
        user_input = typed

    if user_input and anthropic_client:
        # Afficher le message utilisateur
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state["help_messages"].append({"role": "user", "content": user_input})

        # Construire le contexte de conversation (format raw pour l'API)
        raw_messages = _sync_raw_from_display()

        # Exécuter le tour de l'agent
        with st.chat_message("assistant"):
            response_text, tool_summaries = run_agent_turn(
                client=anthropic_client,
                raw_messages=raw_messages,
                system_prompt=system_prompt,
                api_url=_api_url,
                token=_token,
            )
            # Afficher la réponse finale
            st.markdown(response_text)

        # Sauvegarder dans l'historique
        st.session_state["help_messages"].append(
            {
                "role": "assistant",
                "content": response_text,
                "tool_summaries": tool_summaries,
            }
        )
        st.rerun()
