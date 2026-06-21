"""
Services & Access — admin only
Redirection links to infrastructure interfaces with associated credentials.
Credentials are read from the Streamlit container's environment variables.
"""

import os

import streamlit as st
from utils.auth import require_admin
from utils.i18n import t

st.set_page_config(
    page_title=t("services.page_title"),
    page_icon="🔌",
    layout="wide",
)

require_admin()

# ── Reading environment variables ──────────────────────────────────────────────
GRAFANA_URL = os.environ.get("GRAFANA_PUBLIC_URL", "http://localhost:3000")
GRAFANA_USER = os.environ.get("GRAFANA_ADMIN_USER", "admin")
GRAFANA_PASS = os.environ.get("GRAFANA_ADMIN_PASSWORD", "")

MINIO_URL = os.environ.get("MINIO_CONSOLE_PUBLIC_URL", "http://localhost:9011")
MINIO_USER = os.environ.get("MINIO_ROOT_USER", "minioadmin")
MINIO_PASS = os.environ.get("MINIO_ROOT_PASSWORD", "")

MLFLOW_URL = os.environ.get("MLFLOW_PUBLIC_URL", "http://localhost:5000")
MLFLOW_USER = os.environ.get("MLFLOW_ADMIN_USER", "admin")
MLFLOW_PASS = os.environ.get("MLFLOW_ADMIN_PASSWORD", "")

_api_internal = os.environ.get("API_URL", "http://localhost:8000")
API_PUBLIC_URL = os.environ.get(
    "API_PUBLIC_URL",
    "http://localhost:80" if "api:" in _api_internal else _api_internal,
)
API_TOKEN = st.session_state.get("api_token", "")

# SMTP / Email alerts
SMTP_ENABLED = os.environ.get("ENABLE_EMAIL_ALERTS", "false").lower() == "true"
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = os.environ.get("SMTP_PORT", "587")
SMTP_STARTTLS = os.environ.get("SMTP_STARTTLS", "true").lower() == "true"
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "")
ALERT_EMAIL_TO = os.environ.get("ALERT_EMAIL_TO", "")
WEEKLY_REPORT_ENABLED = os.environ.get("WEEKLY_REPORT_ENABLED", "false").lower() == "true"
WEEKLY_REPORT_DAY = os.environ.get("WEEKLY_REPORT_DAY", "monday")
WEEKLY_REPORT_HOUR = os.environ.get("WEEKLY_REPORT_HOUR", "8")
PERF_DRIFT_THRESHOLD = os.environ.get("PERFORMANCE_DRIFT_ALERT_THRESHOLD", "0.10")
ERROR_RATE_THRESHOLD = os.environ.get("ERROR_RATE_ALERT_THRESHOLD", "0.10")

# Anthropic / Chatbot
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title(t("services.title"))
st.caption(t("services.caption"))

# Warning for missing variables (infra services)
_missing = []
if not GRAFANA_PASS:
    _missing.append("`GRAFANA_ADMIN_PASSWORD`")
if not MINIO_PASS:
    _missing.append("`MINIO_ROOT_PASSWORD`")
if not MLFLOW_PASS:
    _missing.append("`MLFLOW_ADMIN_PASSWORD`")
if SMTP_ENABLED and not SMTP_HOST:
    _missing.append("`SMTP_HOST`")
if SMTP_ENABLED and not SMTP_PASSWORD:
    _missing.append("`SMTP_PASSWORD`")

if _missing:
    st.warning(t("services.missing_vars_warning", vars=', '.join(_missing)))

st.divider()

# ── Toggle display of secrets ───────────────────────────────────────────────
_col_toggle, _ = st.columns([1, 3])
with _col_toggle:
    show_secrets = st.toggle(t("services.show_secrets_toggle"), value=False)

st.divider()

# ── Service definitions ────────────────────────────────────────────────────
_SERVICES = [
    {
        "name": "Grafana",
        "icon": "📊",
        "description": t("services.grafana_desc"),
        "url": GRAFANA_URL,
        "link_path": "/dashboards",
        "user": GRAFANA_USER,
        "password": GRAFANA_PASS,
        "env_user": "GRAFANA_ADMIN_USER",
        "env_pass": "GRAFANA_ADMIN_PASSWORD",
        "env_url": "GRAFANA_PUBLIC_URL",
    },
    {
        "name": "MinIO",
        "icon": "🪣",
        "description": t("services.minio_desc"),
        "url": MINIO_URL,
        "link_path": "/login",
        "user": MINIO_USER,
        "password": MINIO_PASS,
        "env_user": "MINIO_ROOT_USER",
        "env_pass": "MINIO_ROOT_PASSWORD",
        "env_url": "MINIO_CONSOLE_PUBLIC_URL",
    },
    {
        "name": "MLflow",
        "icon": "🧪",
        "description": t("services.mlflow_desc"),
        "url": MLFLOW_URL,
        "link_path": "/",
        "user": MLFLOW_USER,
        "password": MLFLOW_PASS,
        "env_user": "MLFLOW_ADMIN_USER",
        "env_pass": "MLFLOW_ADMIN_PASSWORD",
        "env_url": "MLFLOW_PUBLIC_URL",
    },
]

# ── Render service cards ────────────────────────────────────────────────────
cols = st.columns(4, gap="large")

for col, svc in zip(cols, _SERVICES):
    with col:
        st.subheader(f"{svc['icon']} {svc['name']}")
        st.caption(svc["description"])

        # Main access button
        _full_url = svc["url"].rstrip("/") + svc["link_path"]
        st.link_button(
            t("services.open_btn", name=svc['name']),
            _full_url,
            type="primary",
            width="stretch",
        )

        st.markdown("---")
        st.markdown(t("services.credentials_header"))

        # Username
        st.text_input(
            t("services.login_label"),
            value=svc["user"] if svc["user"] else f"—  (var: {svc['env_user']})",
            disabled=True,
            key=f"login_{svc['name']}",
            label_visibility="collapsed",
            placeholder=t("services.login_label"),
        )
        st.caption(t("services.env_var_caption", env_url=svc["env_user"]))

        # Password — hidden by default, st.code when visible (built-in copy button)
        if svc["password"]:
            if show_secrets:
                st.code(svc["password"], language=None)
            else:
                st.text_input(
                    t("services.password_label"),
                    value=svc["password"],
                    type="password",
                    disabled=True,
                    key=f"pass_{svc['name']}",
                    label_visibility="collapsed",
                )
            st.caption(t("services.env_var_caption", env_url=svc["env_pass"]))
        else:
            st.error(t("services.password_missing", env_var=svc['env_pass']))

        st.markdown("---")
        st.markdown(t("services.url_header"))
        st.code(svc["url"], language=None)
        st.caption(t("services.env_var_caption", env_url=svc['env_url']))

# ── 4th column — Swagger / API ──────────────────────────────────────────────
with cols[3]:
    st.subheader(t("services.swagger_name"))
    st.caption(t("services.swagger_desc"))

    st.link_button(
        t("services.open_btn", name="Swagger"),
        API_PUBLIC_URL.rstrip("/") + "/docs",
        type="primary",
        width="stretch",
    )

    st.markdown("---")
    st.markdown(t("services.swagger_token_header"))

    if API_TOKEN:
        if show_secrets:
            st.code(API_TOKEN, language=None)
        else:
            st.text_input(
                "token",
                value=API_TOKEN,
                type="password",
                disabled=True,
                key="swagger_token",
                label_visibility="collapsed",
            )
        st.caption(t("services.swagger_token_caption"))
        st.caption(t("services.env_var_caption", env_url="API_TOKEN"))
    else:
        st.warning(t("services.swagger_token_missing"))

    st.markdown("---")
    st.markdown(t("services.url_header"))
    st.code(API_PUBLIC_URL.rstrip("/") + "/docs", language=None)
    st.caption(t("services.swagger_env_caption"))

# ══════════════════════════════════════════════════════════════════════════════
# Configuration des services externes
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader(t("services.config_section_title"))
st.caption(t("services.config_section_caption"))

col_smtp, col_ai = st.columns([3, 2], gap="large")

# ── SMTP / Email alerts ────────────────────────────────────────────────────
with col_smtp:
    st.markdown(f"#### {t('services.smtp_section_title')}")
    st.caption(t("services.smtp_section_caption"))

    # Status badge
    if SMTP_ENABLED:
        st.success(t("services.smtp_status_enabled"))
    else:
        st.info(t("services.smtp_status_disabled"))

    if not SMTP_HOST:
        st.warning(t("services.smtp_not_configured"))
    else:
        _c1, _c2, _c3 = st.columns([3, 1, 1])
        with _c1:
            st.text_input(
                t("services.smtp_host_label"),
                value=SMTP_HOST,
                disabled=True,
                key="smtp_host",
            )
        with _c2:
            st.text_input(
                t("services.smtp_port_label"),
                value=SMTP_PORT,
                disabled=True,
                key="smtp_port",
            )
        with _c3:
            st.text_input(
                "STARTTLS",
                value="✅" if SMTP_STARTTLS else "❌",
                disabled=True,
                key="smtp_tls",
                help=t("services.smtp_tls_help"),
            )

        _c4, _c5 = st.columns(2)
        with _c4:
            st.text_input(
                t("services.smtp_user_label"),
                value=SMTP_USER or "—",
                disabled=True,
                key="smtp_user",
            )
            st.caption(t("services.env_var_caption", env_url="SMTP_USER"))
        with _c5:
            st.text_input(
                t("services.smtp_from_label"),
                value=SMTP_FROM or "—",
                disabled=True,
                key="smtp_from",
            )
            st.caption(t("services.env_var_caption", env_url="SMTP_FROM"))

        # Password
        st.markdown(f"**{t('services.smtp_password_label')}**")
        if SMTP_PASSWORD:
            if show_secrets:
                st.code(SMTP_PASSWORD, language=None)
            else:
                st.text_input(
                    t("services.smtp_password_label"),
                    value=SMTP_PASSWORD,
                    type="password",
                    disabled=True,
                    key="smtp_pass",
                    label_visibility="collapsed",
                )
            st.caption(t("services.env_var_caption", env_url="SMTP_PASSWORD"))
        else:
            st.error(t("services.password_missing", env_var="SMTP_PASSWORD"))

        # Destinataires
        st.text_input(
            t("services.smtp_to_label"),
            value=ALERT_EMAIL_TO or "—",
            disabled=True,
            key="smtp_to",
        )
        st.caption(t("services.env_var_caption", env_url="ALERT_EMAIL_TO"))

    st.markdown("---")
    # Thresholds & rapport hebdo
    _ct1, _ct2, _ct3 = st.columns(3)
    with _ct1:
        st.metric(
            t("services.smtp_perf_threshold_label"),
            f"{float(PERF_DRIFT_THRESHOLD) * 100:.0f}%",
            help=t("services.smtp_perf_threshold_help"),
        )
        st.caption(t("services.env_var_caption", env_url="PERFORMANCE_DRIFT_ALERT_THRESHOLD"))
    with _ct2:
        st.metric(
            t("services.smtp_error_threshold_label"),
            f"{float(ERROR_RATE_THRESHOLD) * 100:.0f}%",
            help=t("services.smtp_error_threshold_help"),
        )
        st.caption(t("services.env_var_caption", env_url="ERROR_RATE_ALERT_THRESHOLD"))
    with _ct3:
        if WEEKLY_REPORT_ENABLED:
            st.metric(
                t("services.smtp_weekly_report"),
                t("services.smtp_weekly_report_schedule", day=WEEKLY_REPORT_DAY, hour=WEEKLY_REPORT_HOUR),
            )
        else:
            st.metric(t("services.smtp_weekly_report"), t("services.smtp_weekly_report_disabled"))
        st.caption(t("services.env_var_caption", env_url="WEEKLY_REPORT_ENABLED"))

# ── Anthropic / Chatbot ────────────────────────────────────────────────────
with col_ai:
    st.markdown(f"#### {t('services.anthropic_section_title')}")
    st.caption(t("services.anthropic_section_caption"))

    if ANTHROPIC_API_KEY:
        st.success(t("services.anthropic_key_set"))

        st.markdown(f"**{t('services.anthropic_key_label')}**")
        if show_secrets:
            st.code(ANTHROPIC_API_KEY, language=None)
        else:
            st.text_input(
                t("services.anthropic_key_label"),
                value=ANTHROPIC_API_KEY,
                type="password",
                disabled=True,
                key="anthropic_key",
                label_visibility="collapsed",
            )
        st.caption(t("services.env_var_caption", env_url="ANTHROPIC_API_KEY"))
    else:
        st.warning(t("services.anthropic_key_missing"))

    st.markdown("---")
    st.markdown(f"**{t('services.anthropic_model_label')}**")
    st.code("claude-sonnet-4-6", language=None)
    st.caption(t("services.anthropic_model_caption"))

# ── Security note ───────────────────────────────────────────────────────────
st.divider()
with st.expander(t("services.config_expander"), expanded=False):
    st.markdown(t("services.config_body"))
