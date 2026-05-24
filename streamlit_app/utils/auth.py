"""
Helpers d'authentification pour les pages Streamlit
"""

import streamlit as st

from utils.api_client import APIClient
from utils.i18n import t


def require_auth():
    """Vérifie que l'utilisateur est connecté. Arrête l'exécution sinon."""
    if "api_token" not in st.session_state or not st.session_state["api_token"]:
        st.error(t("auth.require_auth_error"))
        st.stop()


def require_admin():
    """Vérifie que l'utilisateur est admin. Arrête l'exécution sinon."""
    require_auth()
    if not st.session_state.get("is_admin", False):
        st.error(t("auth.require_admin_error"))
        st.stop()


def get_client() -> APIClient:
    """Retourne un APIClient configuré avec le token de session."""
    return APIClient(
        base_url=st.session_state.get("api_url", "http://localhost:8000"),
        token=st.session_state.get("api_token", ""),
    )


def logout():
    """Efface la session."""
    for key in ["api_token", "api_url", "is_admin"]:
        st.session_state.pop(key, None)
    st.rerun()
