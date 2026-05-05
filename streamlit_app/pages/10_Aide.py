"""
Page d'aide avec chatbot LLM — PredictML Admin
"""

import os

import anthropic
import streamlit as st
from utils.auth import require_auth
from utils.docs_loader import build_system_prompt, load_all_docs, load_source_snippets
from utils.metrics_help import METRIC_HELP

st.set_page_config(page_title="Aide & Assistant IA — PredictML", page_icon="💬", layout="wide")
require_auth()

# ── Constantes ────────────────────────────────────────────────────────────────

MODEL_ID = "claude-sonnet-4-6"

QUICK_TOPICS = [
    (
        "🛠️ Générer un train.py sklearn",
        "Génère un script train.py complet et compatible PredictML pour un modèle de classification "
        "avec RandomForestClassifier, MLflow tracking, et qui respecte le contrat de variables "
        "d'environnement (TRAIN_START_DATE, TRAIN_END_DATE, OUTPUT_MODEL_PATH). "
        "Inclus aussi la sortie JSON sur stdout avec accuracy, f1_score et feature_stats.",
    ),
    (
        "🌐 Appels API Python complets",
        "Montre-moi le workflow Python complet avec requests : uploader un modèle .pkl, "
        "le passer en production, faire une prédiction unitaire et une prédiction batch, "
        "puis enregistrer les résultats observés (feedback).",
    ),
    (
        "📊 Interpréter les KPIs du dashboard",
        "Explique les principaux indicateurs de performance disponibles dans le dashboard PredictML : "
        "accuracy, F1 Score, taux d'erreur, latence P95, Brier Score, drift Z-score et PSI, "
        "p-value A/B. Pour chacun, donne le seuil d'alerte recommandé et quand agir.",
    ),
    (
        "🔬 Configurer un A/B test",
        "Comment configurer un A/B test entre deux versions d'un modèle dans PredictML ? "
        "Explique le deployment_mode, traffic_weight, comment interpréter les résultats "
        "statistiques (p-value, winner, min_samples_needed) et quand promouvoir la nouvelle version.",
    ),
    (
        "🔄 Ré-entraînement automatique cron",
        "Comment planifier un ré-entraînement automatique hebdomadaire avec une expression cron ? "
        "Inclus la configuration via PATCH /models/{name}/{version}/schedule, "
        "la politique d'auto-promotion (min_accuracy, max_latency_p95_ms), "
        "et le ré-entraînement déclenché par drift (trigger_on_drift).",
    ),
    (
        "🗑️ Purge RGPD des prédictions",
        "Comment utiliser l'endpoint DELETE /predictions/purge pour supprimer les anciennes "
        "prédictions ? Montre dry_run=true pour simuler, puis dry_run=false pour confirmer. "
        "Explique aussi linked_observed_results_count et les risques associés.",
    ),
    (
        "🧪 Golden Tests de régression",
        "Comment créer des golden tests pour valider qu'un modèle produit toujours les mêmes "
        "sorties ? Montre comment créer des cas, les lancer via l'API, interpréter les résultats "
        "PASS/FAIL, et les intégrer dans la politique d'auto-promotion (min_golden_test_pass_rate).",
    ),
    (
        "🏗️ Architecture de PredictML",
        "Explique l'architecture de PredictML : les 7 services Docker (FastAPI, PostgreSQL, MinIO, "
        "Redis, MLflow, Streamlit, Grafana), comment ils interagissent, et le flux de données "
        "d'une prédiction de bout en bout (client → API → modèle → DB → réponse).",
    ),
    (
        "🔍 Détecter la dérive des données",
        "Comment configurer la détection de drift dans PredictML ? Explique feature_baseline, "
        "les métriques Z-score et PSI, les statuts ok/warning/critical, et l'output drift "
        "(label shift). Comment déclencher un ré-entraînement automatique quand le drift est critique ?",
    ),
    (
        "👥 Gestion des utilisateurs",
        "Comment gérer les utilisateurs dans PredictML ? Créer un compte avec rôle et quota, "
        "renouveler un token, désactiver un compte, vérifier le quota journalier. "
        "Explique aussi les rôles admin / user / readonly et leurs droits respectifs.",
    ),
    (
        "📝 Valider le schéma d'un modèle",
        "Comment utiliser POST /models/{name}/{version}/validate-input pour vérifier les features "
        "avant de prédire ? Explique les erreurs missing_feature, unexpected_feature, type_coercion "
        "et le mode strict ?strict_validation=true sur /predict.",
    ),
    (
        "🚀 Démarrer avec PredictML",
        "Je veux démarrer avec PredictML de zéro. Explique étape par étape : "
        "installation Docker, premier modèle sklearn, upload, passer en production, "
        "faire une prédiction, voir les résultats dans le dashboard.",
    ),
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

if "help_show_sources" not in st.session_state:
    st.session_state["help_show_sources"] = False

# ── Chargement docs, code source & prompt système ─────────────────────────────

docs = load_all_docs()
snippets = load_source_snippets()

# Injecter la définition des KPIs dans la documentation
kpis_section = "\n".join(f"- **{k}** : {v}" for k, v in METRIC_HELP.items())
docs["KPIs_Dashboard"] = f"## Définitions des KPIs du dashboard\n\n{kpis_section}"

system_prompt = build_system_prompt(docs, snippets)
anthropic_client = get_anthropic_client()

# ── En-tête ───────────────────────────────────────────────────────────────────

st.title("💬 Aide & Assistant IA")
st.markdown(
    "Posez vos questions sur PredictML : scripts d'entraînement, appels API, "
    "indicateurs KPIs, configuration du dashboard, architecture…"
)

# Indicateur de richesse du contexte
ctx_parts = []
if docs:
    ctx_parts.append(f"📚 {len(docs)} docs")
if snippets:
    ctx_parts.append(f"🔧 {len(snippets)} fichiers source")
if ctx_parts:
    st.caption(f"Contexte chargé : {' · '.join(ctx_parts)}")

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
            "En mode Docker, vérifiez que le volume `./documentation:/app/documentation:ro` "
            "est bien déclaré dans `docker-compose.yml` pour le service `streamlit`."
        )
    else:
        # Labels lisibles pour chaque document
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

        selected_label = st.selectbox(
            "Choisir un document",
            display_names,
            key="help_doc_select",
        )
        selected_key = name_map[selected_label]

        with st.container(height=680, border=True):
            st.markdown(docs[selected_key])

    # Affichage optionnel des fichiers source
    if snippets:
        with st.expander(f"🔧 Code source ({len(snippets)} fichiers)", expanded=False):
            src_names = list(snippets.keys())
            selected_src = st.selectbox("Fichier", src_names, key="help_src_select")
            st.code(snippets[selected_src], language="python")

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
            "La clé est transmise automatiquement via `docker-compose.yml`.\n\n"
            "---\n"
            "En attendant, la documentation est disponible dans la colonne de gauche."
        )

    # ── Boutons de sujets rapides ─────────────────────────
    with st.expander("⚡ Sujets rapides", expanded=(len(st.session_state["help_messages"]) == 0)):
        cols = st.columns(2)
        for i, (label, prompt) in enumerate(QUICK_TOPICS):
            if cols[i % 2].button(label, key=f"quick_{i}", use_container_width=True):
                st.session_state["help_pending_prompt"] = prompt

    # ── Actions de gestion ────────────────────────────────
    btn_col1, btn_col2 = st.columns([2, 3])
    if btn_col1.button("🗑️ Nouvelle conversation", key="help_clear"):
        st.session_state["help_messages"] = []
        st.session_state["help_pending_prompt"] = None
        st.rerun()

    n_exchanges = len(st.session_state["help_messages"]) // 2
    if n_exchanges > 0:
        btn_col2.caption(f"{n_exchanges} échange(s) en cours")

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
        placeholder="Posez votre question sur PredictML…",
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
