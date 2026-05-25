"""
AutoTrain Chatbot — Génération et test de scripts train.py assistés par IA.
Page réservée aux administrateurs.
"""

import json
import os
from datetime import date, timedelta

import anthropic
import pandas as pd
import streamlit as st
from utils.auth import require_admin
from utils.autotraining_tools import (
    AUTOTRAINING_TOOL_DEFINITIONS,
    _run_fetch_training_data,
    autotraining_tool_expander_label,
    build_autotraining_system_prompt,
    build_autotraining_tool_summary,
    execute_autotraining_tool,
    render_autotraining_tool_input,
    render_autotraining_tool_result,
)
from utils.i18n import t

st.set_page_config(page_title=t("autotrain.page_title"), page_icon="🤖", layout="wide")
require_admin()

# ── Constantes ────────────────────────────────────────────────────────────────

MODEL_ID = "claude-sonnet-4-6"

_DEFAULT_START = (date.today() - timedelta(days=90)).isoformat()
_DEFAULT_END = date.today().isoformat()

QUICK_TOPICS = [
    (
        t("autotrain.quick_topics.analyze_data"),
        "Analyse le dataset chargé en session (ou récupère les données disponibles) "
        "et dis-moi quelles features sont disponibles, la distribution des labels, "
        "et quel algorithme tu recommandes pour ce problème.",
    ),
    (
        t("autotrain.quick_topics.random_forest"),
        "Génère un script train.py complet avec RandomForestClassifier. "
        "Respecte le contrat PredictML (TRAIN_START_DATE, TRAIN_END_DATE, OUTPUT_MODEL_PATH). "
        "Inclus la gestion de TRAIN_DATA_PATH avec fallback dataset Iris. "
        "Puis exécute-le pour valider qu'il fonctionne.",
    ),
    (
        t("autotrain.quick_topics.gradient_boosting"),
        "Génère un script train.py avec GradientBoostingClassifier (100 estimateurs, "
        "learning_rate=0.1). Respecte le contrat PredictML et inclus feature_stats "
        "et label_distribution dans la sortie JSON. Exécute-le pour obtenir les métriques.",
    ),
    (
        t("autotrain.quick_topics.compare_versions"),
        "Compare les scripts déjà testés dans la session. "
        "Montre un tableau récapitulatif des métriques (accuracy, F1, n_rows) "
        "et recommande le meilleur à uploader.",
    ),
    (
        t("autotrain.quick_topics.optimize_script"),
        "Analyse le dernier script exécuté avec succès et propose des améliorations : "
        "optimisation des hyperparamètres, normalisation des features, "
        "ou algorithme alternatif. Génère et teste la version améliorée.",
    ),
    (
        t("autotrain.quick_topics.upload_model"),
        "Uploade le meilleur modèle de la session vers l'API PredictML. "
        "Vérifie d'abord les métriques disponibles, propose un nom et une version, "
        "puis exécute l'upload.",
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
            tools=AUTOTRAINING_TOOL_DEFINITIONS,
            messages=current_messages,
        )

        turn_text = "".join(b.text for b in response.content if hasattr(b, "text") and b.text)

        if response.stop_reason == "tool_use":
            if turn_text:
                st.markdown(turn_text)
                final_text += turn_text + "\n\n"

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                label = autotraining_tool_expander_label(block.name, block.input)
                with st.expander(label, expanded=True):
                    render_autotraining_tool_input(block.name, block.input)
                    with st.spinner(t("autotrain.tool_running")):
                        result = execute_autotraining_tool(block.name, block.input, api_url, token)
                    st.divider()
                    render_autotraining_tool_result(block.name, result)

                tool_summaries.append(
                    build_autotraining_tool_summary(block.name, block.input, result)
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

            current_messages = current_messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ]

        else:
            final_text += turn_text
            return final_text, tool_summaries


# ── Rendu de l'historique ─────────────────────────────────────────────────────


def render_chat_history() -> None:
    for msg in st.session_state["autotrain_messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                for summary in msg.get("tool_summaries", []):
                    with st.expander(summary["label"], expanded=False):
                        st.caption(t("autotrain.tool_summary_request"))
                        tool_type = summary.get("type", "other")
                        if tool_type == "execute_python":
                            st.code(summary.get("query", ""), language="python")
                        elif tool_type == "fetch_training_data":
                            st.caption(summary.get("query", ""))
                        elif tool_type == "upload_model":
                            st.caption(summary.get("query", ""))
                        elif tool_type == "sql":
                            st.code(summary.get("query", ""), language="sql")
                        else:
                            st.code(
                                f"{summary.get('method', '')} {summary.get('endpoint', '')}",
                                language="http",
                            )
                        if summary.get("result_preview"):
                            st.caption(t("autotrain.tool_summary_preview"))
                            st.code(summary["result_preview"])
            st.markdown(msg["content"])


# ── Helpers de gestion de l'état des messages ────────────────────────────────


def _sync_raw_from_display() -> list:
    """Reconstruit raw_messages depuis autotrain_messages."""
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state["autotrain_messages"]
    ]


# ── Initialisation session state ──────────────────────────────────────────────


def _init_session_state() -> None:
    defaults: dict = {
        "autotrain_messages": [],
        "autotrain_pending_prompt": None,
        "autotrain_dataset_path": None,
        "autotrain_dataset_info": None,
        "autotrain_scripts": [],
        "autotrain_last_model_path": None,
        "autotrain_last_script": None,
        "autotrain_train_start": _DEFAULT_START,
        "autotrain_train_end": _DEFAULT_END,
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)


def _cleanup_session_temps() -> None:
    """Supprime les fichiers temporaires de la session (modèles + datasets)."""
    dataset_path = st.session_state.get("autotrain_dataset_path")
    if dataset_path:
        try:
            os.unlink(dataset_path)
        except OSError:
            pass

    for entry in st.session_state.get("autotrain_scripts", []):
        model_path = entry.get("model_path")
        if model_path:
            try:
                import shutil

                parent = os.path.dirname(model_path)
                if os.path.exists(parent):
                    shutil.rmtree(parent, ignore_errors=True)
            except OSError:
                pass


# ── Page principale ───────────────────────────────────────────────────────────

_init_session_state()

system_prompt = build_autotraining_system_prompt()
anthropic_client = get_anthropic_client()

_api_url: str = st.session_state.get("api_url", "http://localhost:8000")
_token: str = st.session_state.get("api_token", "")

# En-tête
st.title(t("autotrain.title"))
st.markdown(t("autotrain.subtitle"))

# Badges d'état en en-tête
dataset_info = st.session_state.get("autotrain_dataset_info")
last_model = st.session_state.get("autotrain_last_model_path")
badges = []
if dataset_info:
    badges.append(
        t(
            "autotrain.dataset_badge",
            n_rows=dataset_info["n_rows"],
            n_labeled=dataset_info["n_labeled"],
        )
    )
if last_model and os.path.exists(last_model):
    badges.append(t("autotrain.model_ready_badge"))
if badges:
    st.success("  ·  ".join(badges))

st.divider()

# ── Layout 3 onglets ──────────────────────────────────────────────────────────

tab_chat, tab_data, tab_scripts = st.tabs(
    [
        t("autotrain.tab_chat"),
        t("autotrain.tab_data"),
        t("autotrain.tab_scripts"),
    ]
)

# ═══════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — CHAT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_chat:
    if anthropic_client is None:
        st.warning(t("autotrain.no_api_key"))

    # Dates d'entraînement (contexte pour execute_python)
    with st.expander("📅 Fenêtre d'entraînement (TRAIN_START/END_DATE)", expanded=False):
        col_s, col_e = st.columns(2)
        new_start = col_s.date_input(
            t("autotrain.dates.train_start"),
            value=date.fromisoformat(st.session_state["autotrain_train_start"]),
            key="at_date_start",
        )
        new_end = col_e.date_input(
            t("autotrain.dates.train_end"),
            value=date.fromisoformat(st.session_state["autotrain_train_end"]),
            key="at_date_end",
        )
        st.session_state["autotrain_train_start"] = new_start.isoformat()
        st.session_state["autotrain_train_end"] = new_end.isoformat()
        st.caption(
            "Ces dates seront injectées automatiquement comme `TRAIN_START_DATE` "
            "et `TRAIN_END_DATE` lors de l'exécution des scripts."
        )

    # Sujets rapides
    with st.expander(
        "⚡ Actions rapides",
        expanded=(len(st.session_state["autotrain_messages"]) == 0),
    ):
        cols = st.columns(3)
        for i, (label, prompt) in enumerate(QUICK_TOPICS):
            if cols[i % 3].button(label, key=f"at_quick_{i}", width="stretch"):
                st.session_state["autotrain_pending_prompt"] = prompt

    # Boutons de contrôle
    btn_col1, btn_col2 = st.columns([2, 5])
    if btn_col1.button(t("autotrain.new_conversation_btn"), key="at_clear"):
        _cleanup_session_temps()
        st.session_state["autotrain_messages"] = []
        st.session_state["autotrain_pending_prompt"] = None
        st.session_state["autotrain_scripts"] = []
        st.session_state["autotrain_last_model_path"] = None
        st.session_state["autotrain_last_script"] = None
        st.session_state["autotrain_dataset_path"] = None
        st.session_state["autotrain_dataset_info"] = None
        st.rerun()

    n_exchanges = len(st.session_state["autotrain_messages"]) // 2
    if n_exchanges > 0:
        btn_col2.caption(t("autotrain.exchange_count", count=n_exchanges))

    st.divider()

    # Historique de la conversation
    render_chat_history()

    # ── Traitement du message (prompt rapide ou saisie) ───

    user_input: str | None = None

    if st.session_state["autotrain_pending_prompt"] and anthropic_client:
        user_input = st.session_state["autotrain_pending_prompt"]
        st.session_state["autotrain_pending_prompt"] = None

    typed = st.chat_input(
        placeholder=t("autotrain.chat_placeholder"),
        disabled=(anthropic_client is None),
        key="at_chat_input",
    )
    if typed:
        user_input = typed

    if user_input and anthropic_client:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state["autotrain_messages"].append({"role": "user", "content": user_input})

        raw_messages = _sync_raw_from_display()

        with st.chat_message("assistant"):
            response_text, tool_summaries = run_agent_turn(
                client=anthropic_client,
                raw_messages=raw_messages,
                system_prompt=system_prompt,
                api_url=_api_url,
                token=_token,
            )
            st.markdown(response_text)

        st.session_state["autotrain_messages"].append(
            {
                "role": "assistant",
                "content": response_text,
                "tool_summaries": tool_summaries,
            }
        )
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

with tab_data:
    source = st.radio(
        t("autotrain.data.source_label"),
        [t("autotrain.data.source_upload"), t("autotrain.data.source_api")],
        horizontal=True,
        key="at_data_source",
    )

    if source == t("autotrain.data.source_upload"):
        # ── Upload CSV local ──────────────────────────────────────────────────
        uploaded = st.file_uploader(
            t("autotrain.data.upload_label"),
            type=["csv"],
            help=t("autotrain.data.upload_help"),
            key="at_csv_uploader",
        )
        if uploaded is not None:
            try:
                import tempfile

                df_up = pd.read_csv(uploaded)

                # Sauvegarder dans fichier temporaire
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".csv", delete=False, mode="w", encoding="utf-8"
                )
                df_up.to_csv(tmp, index=False)
                csv_path = tmp.name
                tmp.close()

                st.session_state["autotrain_dataset_path"] = csv_path

                # Calculer les infos
                labeled_col = "observed_result"
                n_labeled = (
                    int(df_up[labeled_col].notna().sum()) if labeled_col in df_up.columns else 0
                )
                feature_names = (
                    [
                        c
                        for c in df_up.columns
                        if c
                        not in [
                            labeled_col,
                            "id_obs",
                            "timestamp",
                            "model_version",
                            "response_time_ms",
                            "prediction_result",
                        ]
                    ]
                    if labeled_col in df_up.columns
                    else list(df_up.columns)
                )

                st.session_state["autotrain_dataset_info"] = {
                    "n_rows": len(df_up),
                    "n_labeled": n_labeled,
                    "feature_names": feature_names,
                    "preview": df_up.head(5).to_dict("records"),
                }
                st.success(
                    t(
                        "autotrain.data.fetch_success",
                        n_rows=len(df_up),
                        n_labeled=n_labeled,
                    )
                )
            except Exception as e:
                st.error(t("autotrain.data.fetch_error", error=str(e)))

    else:
        # ── Récupération depuis l'API ────────────────────────────────────────
        try:
            from utils.api_client import get_models

            models_list = get_models(_api_url, _token)
            model_names = sorted({m["name"] for m in models_list if m.get("name")})
        except Exception:
            model_names = []

        if not model_names:
            st.warning(t("autotrain.data.no_models"))
        else:
            col1, col2, col3 = st.columns(3)
            sel_model = col1.selectbox(
                t("autotrain.data.model_label"), model_names, key="at_fetch_model"
            )
            sel_start = col2.date_input(
                t("autotrain.data.date_start_label"),
                value=date.fromisoformat(_DEFAULT_START),
                key="at_fetch_start",
            )
            sel_end = col3.date_input(
                t("autotrain.data.date_end_label"),
                value=date.today(),
                key="at_fetch_end",
            )
            fetch_limit = st.number_input(
                t("autotrain.data.limit_label"),
                min_value=10,
                max_value=5000,
                value=1000,
                step=100,
                key="at_fetch_limit",
            )

            if st.button(t("autotrain.data.fetch_btn"), key="at_fetch_btn"):
                with st.spinner(t("autotrain.data.fetch_spinner")):
                    result = _run_fetch_training_data(
                        {
                            "model_name": sel_model,
                            "start_date": sel_start.isoformat(),
                            "end_date": sel_end.isoformat(),
                            "limit": fetch_limit,
                        },
                        _api_url,
                        _token,
                    )
                if "error" in result:
                    st.error(t("autotrain.data.fetch_error", error=result["error"]))
                else:
                    st.success(
                        t(
                            "autotrain.data.fetch_success",
                            n_rows=result["n_rows"],
                            n_labeled=result["n_labeled"],
                        )
                    )
                    st.rerun()

    # ── Affichage du dataset chargé ───────────────────────────────────────────

    st.divider()
    info = st.session_state.get("autotrain_dataset_info")
    if info:
        st.subheader("📊 Dataset chargé")
        c1, c2, c3 = st.columns(3)
        c1.metric(t("autotrain.data.n_rows"), f"{info['n_rows']:,}")
        c2.metric(t("autotrain.data.n_labeled"), f"{info['n_labeled']:,}")
        c3.metric(t("autotrain.data.n_features"), len(info.get("feature_names", [])))

        feature_names = info.get("feature_names", [])
        if feature_names:
            features_display = ", ".join(feature_names[:15])
            if len(feature_names) > 15:
                features_display += f" … (+{len(feature_names) - 15})"
            st.caption(f"**Features :** `{features_display}`")

        preview = info.get("preview", [])
        if preview:
            with st.expander(t("autotrain.data.preview_header"), expanded=True):
                try:
                    df_preview = pd.DataFrame(preview)
                    # Tronquer les colonnes JSON longues pour l'affichage
                    for col in [
                        "input_features",
                        "prediction_result",
                        "observed_result",
                    ]:
                        if col in df_preview.columns:
                            df_preview[col] = df_preview[col].astype(str).str[:60]
                    st.dataframe(df_preview, use_container_width=True, hide_index=True)
                except Exception:
                    st.json(preview[:3])

        # Bouton pour effacer le dataset
        if st.button("🗑️ Effacer le dataset", key="at_clear_dataset"):
            dataset_path = st.session_state.get("autotrain_dataset_path")
            if dataset_path:
                try:
                    os.unlink(dataset_path)
                except OSError:
                    pass
            st.session_state["autotrain_dataset_path"] = None
            st.session_state["autotrain_dataset_info"] = None
            st.rerun()
    else:
        st.info(t("autotrain.data.no_data"))

# ═══════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — SCRIPTS GÉNÉRÉS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_scripts:
    scripts = st.session_state.get("autotrain_scripts", [])

    if not scripts:
        st.info(t("autotrain.scripts.empty"))
    else:
        st.subheader(t("autotrain.scripts.title"))

        # Tableau récapitulatif
        summary_rows = []
        for s in scripts:
            metrics = s.get("metrics", {})
            summary_rows.append(
                {
                    "#": s["id"],
                    "Accuracy": (
                        f"{metrics['accuracy']:.4f}" if metrics and "accuracy" in metrics else "—"
                    ),
                    "F1 Score": (
                        f"{metrics['f1_score']:.4f}" if metrics and "f1_score" in metrics else "—"
                    ),
                    "Lignes": metrics.get("n_rows", "—") if metrics else "—",
                    "Temps (ms)": s.get("execution_time_ms", "—"),
                    "Modèle produit": "✅" if s.get("model_path") else "❌",
                }
            )
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # Détail de chaque script
        for script_entry in reversed(scripts):
            idx = script_entry["id"]
            metrics = script_entry.get("metrics", {})
            model_path = script_entry.get("model_path")
            exec_ms = script_entry.get("execution_time_ms", 0)

            acc_str = (
                f"acc={metrics['accuracy']:.3f}"
                if metrics and "accuracy" in metrics
                else "no metrics"
            )
            f1_str = f"F1={metrics['f1_score']:.3f}" if metrics and "f1_score" in metrics else ""
            summary_str = " | ".join(filter(None, [acc_str, f1_str, f"{exec_ms}ms"]))

            with st.expander(
                f"{t('autotrain.scripts.version', id=idx)} — {summary_str}",
                expanded=(idx == len(scripts)),  # Dernier script ouvert
            ):
                # Métriques
                if metrics:
                    m_cols = st.columns(3)
                    m_cols[0].metric(
                        "Accuracy",
                        f"{metrics['accuracy']:.4f}" if "accuracy" in metrics else "—",
                    )
                    m_cols[1].metric(
                        "F1 Score",
                        f"{metrics['f1_score']:.4f}" if "f1_score" in metrics else "—",
                    )
                    m_cols[2].metric(t("autotrain.scripts.execution_time"), f"{exec_ms} ms")

                # Code source
                with st.expander(t("autotrain.scripts.code_expander"), expanded=False):
                    st.code(script_entry["code"], language="python")

                # Télécharger le script
                st.download_button(
                    label=t("autotrain.scripts.download_btn"),
                    data=script_entry["code"].encode("utf-8"),
                    file_name=f"train_v{idx}.py",
                    mime="text/x-python",
                    key=f"dl_script_{idx}",
                )

                # Upload depuis cet onglet
                if model_path and os.path.exists(model_path):
                    st.divider()
                    st.caption("🚀 **Uploader ce modèle vers l'API PredictML**")

                    with st.form(key=f"upload_form_{idx}"):
                        u_col1, u_col2 = st.columns(2)
                        u_name = u_col1.text_input(
                            t("autotrain.scripts.upload_form_name"),
                            key=f"u_name_{idx}",
                        )
                        u_version = u_col2.text_input(
                            t("autotrain.scripts.upload_form_version"),
                            value=f"1.{idx}.0",
                            key=f"u_ver_{idx}",
                        )
                        u_desc = st.text_input(
                            t("autotrain.scripts.upload_form_desc"),
                            key=f"u_desc_{idx}",
                        )
                        u_algo = st.text_input(
                            t("autotrain.scripts.upload_form_algo"),
                            key=f"u_algo_{idx}",
                        )
                        submitted = st.form_submit_button(t("autotrain.scripts.upload_btn"))

                    if submitted and u_name and u_version:
                        # Pointer temporairement sur ce script/modèle
                        prev_path = st.session_state.get("autotrain_last_model_path")
                        prev_script = st.session_state.get("autotrain_last_script")
                        prev_scripts = list(st.session_state.get("autotrain_scripts", []))

                        st.session_state["autotrain_last_model_path"] = model_path
                        st.session_state["autotrain_last_script"] = script_entry["code"]

                        # Mettre à jour les métriques dans la liste des scripts
                        # pour que _run_upload_model les lise correctement
                        scripts_with_this = [
                            s
                            for s in st.session_state.get("autotrain_scripts", [])
                            if s["id"] == idx
                        ]
                        if not scripts_with_this:
                            st.session_state["autotrain_scripts"] = [script_entry]

                        from utils.autotraining_tools import _run_upload_model

                        with st.spinner(t("autotrain.scripts.upload_spinner")):
                            res = _run_upload_model(
                                {
                                    "name": u_name,
                                    "version": u_version,
                                    "description": u_desc or None,
                                    "algorithm": u_algo or None,
                                },
                                _api_url,
                                _token,
                            )

                        # Restaurer les valeurs précédentes
                        st.session_state["autotrain_last_model_path"] = prev_path
                        st.session_state["autotrain_last_script"] = prev_script
                        st.session_state["autotrain_scripts"] = prev_scripts

                        if "error" in res:
                            st.error(
                                t(
                                    "autotrain.scripts.upload_error",
                                    error=res["error"],
                                )
                            )
                        else:
                            st.success(
                                t(
                                    "autotrain.scripts.upload_success",
                                    name=u_name,
                                    version=u_version,
                                )
                            )
                else:
                    st.caption(t("autotrain.scripts.no_model_file"))
