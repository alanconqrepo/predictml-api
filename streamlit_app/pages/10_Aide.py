"""
Help page with LLM chatbot (native function calling) — PredictML Admin
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
        "Which models are currently in production? "
        "Show me their metrics (accuracy, F1, latency) and indicate whether there are any drift issues.",
    ),
    (
        t("aide.quick_topics.prediction_stats"),
        "Give me a summary of predictions from the last 7 days: volume by model, "
        "error rate, average latency, and P95.",
    ),
    (
        t("aide.quick_topics.generate_train_py"),
        "Generate a complete PredictML-compatible train.py script for a classification model "
        "using RandomForestClassifier. Respect the environment variable contract "
        "(TRAIN_START_DATE, TRAIN_END_DATE, OUTPUT_MODEL_PATH) and include JSON output on stdout.",
    ),
    (
        t("aide.quick_topics.api_calls"),
        "Show me the full Python workflow using requests: upload a .joblib model, "
        "promote it to production, make a single and batch prediction, "
        "then record the observed results.",
    ),
    (
        t("aide.quick_topics.interpret_kpis"),
        "Explain the main performance indicators in the PredictML dashboard: "
        "accuracy, F1, error rate, P95 latency, Brier Score, drift Z-score and PSI, A/B p-value. "
        "Give the recommended alert thresholds.",
    ),
    (
        t("aide.quick_topics.configure_ab"),
        "How do I configure an A/B test between two model versions? "
        "Explain deployment_mode, traffic_weight, statistical interpretation "
        "(p-value, winner, min_samples_needed) and when to promote.",
    ),
    (
        t("aide.quick_topics.auto_retrain"),
        "How do I schedule an automatic weekly retraining using a cron expression? "
        "Include the schedule configuration, the auto-promotion policy, "
        "and drift-triggered retraining.",
    ),
    (
        t("aide.quick_topics.detect_drift"),
        "How do I configure and interpret drift detection in PredictML? "
        "Explain feature_baseline, Z-score, PSI, ok/warning/critical statuses, "
        "and automatic retraining when drift is critical.",
    ),
    (
        t("aide.quick_topics.manage_users"),
        "How do I manage users? Show me the currently active users, "
        "their role and quota. Also explain how to create an account and renew a token.",
    ),
    (
        t("aide.quick_topics.golden_tests"),
        "How do I create golden tests and integrate them into the auto-promotion policy? "
        "Show existing test cases if available.",
    ),
    (
        t("aide.quick_topics.architecture"),
        "Explain the PredictML architecture: the 7 Docker services, how they interact, "
        "and the full data flow for a prediction.",
    ),
    (
        t("aide.quick_topics.get_started"),
        "I want to get started with PredictML from scratch. Guide me step by step: "
        "Docker installation, first sklearn model, upload, production deployment, prediction.",
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
    Execute a full agent turn with function calling support.

    Renders tool calls inline in the current chat_message.
    Returns (final_text, list_of_tool_summaries).
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

        # Intermediate text emitted before tool calls
        turn_text = "".join(
            b.text for b in response.content if hasattr(b, "text") and b.text
        )

        if response.stop_reason == "tool_use":
            # Display intermediate reasoning text
            if turn_text:
                st.markdown(turn_text)
                final_text += turn_text + "\n\n"

            # Process each tool call
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

            # Continue the conversation with tool results
            current_messages = current_messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ]

        else:
            # Final response
            final_text += turn_text
            return final_text, tool_summaries


# ── Chat history rendering ────────────────────────────────────────────────────


def render_chat_history() -> None:
    for msg in st.session_state["help_messages"]:
        with st.chat_message(msg["role"]):
            # For assistant messages, redisplay tool summaries (collapsed)
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


# ── Message state management helpers ─────────────────────────────────────────


def _user_to_raw(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant_to_raw(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _sync_raw_from_display() -> list:
    """Rebuild raw_messages from help_messages (for Anthropic API compatibility)."""
    raw = []
    for msg in st.session_state["help_messages"]:
        raw.append({"role": msg["role"], "content": msg["content"]})
    return raw


# ── Session state initialization ─────────────────────────────────────────────

if "help_messages" not in st.session_state:
    st.session_state["help_messages"] = []  # [{role, content, tool_summaries?}]

if "help_pending_prompt" not in st.session_state:
    st.session_state["help_pending_prompt"] = None

# ── Documentation, source code & system prompt loading ───────────────────────

docs = load_all_docs()
snippets = load_source_snippets()

kpis_section = "\n".join(f"- **{k}** : {v}" for k, v in METRIC_HELP.items())
docs["KPIs_Dashboard"] = f"## Définitions des KPIs du dashboard\n\n{kpis_section}"

system_prompt = build_system_prompt(docs, snippets)
anthropic_client = get_anthropic_client()

# URL and token of the current session
_api_url: str = st.session_state.get("api_url", "http://localhost:8000")
_token: str = st.session_state.get("api_token", "")

# ── Header ────────────────────────────────────────────────────────────────────

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

# ── Main layout — 3 full-width sections ──────────────────────────────────────

# ═══════════════════════════════════════════════════════════
# SECTION 1 — Chatbot LLM
# ═══════════════════════════════════════════════════════════

with st.expander(t("aide.chat_expander"), expanded=False):
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

    # ── Message processing (quick prompt or user input) ───

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
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state["help_messages"].append({"role": "user", "content": user_input})

        # Build conversation context (raw format for the API)
        raw_messages = _sync_raw_from_display()

        # Execute the agent turn
        with st.chat_message("assistant"):
            response_text, tool_summaries = run_agent_turn(
                client=anthropic_client,
                raw_messages=raw_messages,
                system_prompt=system_prompt,
                api_url=_api_url,
                token=_token,
            )
            # Display the final response
            st.markdown(response_text)

        # Save to history
        st.session_state["help_messages"].append(
            {
                "role": "assistant",
                "content": response_text,
                "tool_summaries": tool_summaries,
            }
        )
        st.rerun()

# ═══════════════════════════════════════════════════════════
# SECTION 2 — Documentation
# ═══════════════════════════════════════════════════════════

with st.expander(t("aide.doc_expander"), expanded=False):
    if not docs:
        st.warning(t("aide.doc_not_available"))
    else:
        labels = {
            "API_REFERENCE":      "📖 API Reference",
            "ARCHITECTURE":       "🏗️ Architecture",
            "BEGINNER_GUIDE":     "🎓 Beginner Guide",
            "DATABASE":           "🗄️ Database",
            "DOCKER":             "🐳 Docker",
            "QUICKSTART":         "⚡ Quick Start",
            "README":             "🏠 README",
            "CODING_STANDARDS":   "✅ Coding Standards",
            "DASHBOARD_GUIDE":    "🖥️ Dashboard Guide",
            "TRAIN_SCRIPT_GUIDE": "🛠️ train.py Guide",
            "KPIS_REFERENCE":     "📊 KPIs Reference",
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
# SECTION 3 — Code source
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
