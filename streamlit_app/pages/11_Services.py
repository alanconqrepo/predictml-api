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

# ── Header ─────────────────────────────────────────────────────────────────────
st.title(t("services.title"))
st.caption(t("services.caption"))

# Warning for missing variables
_missing = []
if not GRAFANA_PASS:
    _missing.append("`GRAFANA_ADMIN_PASSWORD`")
if not MINIO_PASS:
    _missing.append("`MINIO_ROOT_PASSWORD`")
if not MLFLOW_PASS:
    _missing.append("`MLFLOW_ADMIN_PASSWORD`")

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
cols = st.columns(3, gap="large")

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
        else:
            st.error(t("services.password_missing", env_var=svc['env_pass']))

        st.markdown("---")
        st.markdown(t("services.url_header"))
        st.code(svc["url"], language=None)
        st.caption(t("services.env_var_caption", env_url=svc['env_url']))

# ── Security note ───────────────────────────────────────────────────────────
st.divider()
with st.expander(t("services.config_expander"), expanded=False):
    st.markdown(t("services.config_body"))
