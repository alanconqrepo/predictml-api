"""
Page d'accueil et login — PredictML Admin Dashboard
"""

import os
from urllib.parse import urlparse

import streamlit as st
from utils.api_client import APIClient
from utils.auth import logout


def _is_valid_api_url(url: str) -> bool:
    """Vérifie que l'URL est http/https et pointe vers un hôte non vide."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False

st.set_page_config(
    page_title="PredictML Admin",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")


def show_login():
    st.title("🤖 PredictML Admin Dashboard")
    st.markdown("Connectez-vous avec votre API token pour accéder au dashboard.")

    with st.form("login_form"):
        api_url = st.text_input("URL de l'API", value=DEFAULT_API_URL)
        token = st.text_input("API Token", type="password", placeholder="Votre token Bearer")
        submitted = st.form_submit_button("Se connecter", use_container_width=True)

    if submitted:
        if not token:
            st.error("Veuillez saisir un token.")
            return
        if not _is_valid_api_url(api_url):
            st.error("URL invalide. Elle doit commencer par http:// ou https://")
            return

        client = APIClient(base_url=api_url, token=token)
        is_valid, is_admin = client.check_auth()

        if not is_valid:
            st.error("Token invalide ou API inaccessible.")
            return

        st.session_state["api_token"] = token
        st.session_state["api_url"] = api_url
        st.session_state["is_admin"] = is_admin
        st.rerun()


def show_home():
    st.title("🤖 PredictML Admin Dashboard")

    role_badge = "👑 Admin" if st.session_state.get("is_admin") else "👤 Utilisateur"
    st.caption(f"Connecté — {role_badge}  |  API: `{st.session_state.get('api_url')}`")

    client = APIClient(
        base_url=st.session_state["api_url"],
        token=st.session_state["api_token"],
    )

    with st.sidebar:
        st.subheader("Mon compte")
        try:
            quota = client.get_my_quota()
            used = quota["used_today"]
            limit = quota["rate_limit_per_day"]
            remaining = quota["remaining_today"]
            st.progress(used / limit if limit > 0 else 0)
            st.caption(f"{used} / {limit} aujourd'hui")
            if remaining == 0:
                st.warning("Quota épuisé pour aujourd'hui.")
        except Exception:
            pass
        st.divider()

    # Statut API
    try:
        health = client.get_health()
        api_status = health.get("status", "unknown")
        db_status = health.get("database", "unknown")
        col_status, col_db, col_cache = st.columns(3)
        col_status.metric(
            "Statut API", "✅ En ligne" if api_status == "healthy" else f"⚠️ {api_status}"
        )
        col_db.metric("Base de données", "✅ OK" if db_status == "connected" else f"⚠️ {db_status}")
        cached = health.get("cached_models", 0)
        col_cache.metric("Modèles en cache", cached)
    except Exception:
        st.warning("Impossible de contacter l'API.")

    # Résumé des modèles
    try:
        models = client.list_models()
        n_prod = sum(1 for m in models if m.get("is_production"))
        col1, col2 = st.columns(2)
        col1.metric("Modèles actifs", len(models))
        col2.metric("Versions en production", n_prod)
    except Exception:
        pass

    st.divider()
    st.subheader("Navigation")
    st.markdown("""
| Page | Description |
|------|-------------|
| **1 - Users** | Gérer les utilisateurs, créer des comptes, renouveler les tokens *(admin)* |
| **2 - Models** | Consulter et administrer les modèles ML |
| **3 - Predictions** | Historique des prédictions avec filtres |
| **4 - Stats** | Statistiques et graphiques d'utilisation |
| **5 - Code Example** | Exemple de code MLflow + API |
| **6 - A/B Testing** | Configurer les tests A/B, déploiement shadow, comparer les métriques par version |
""")

    st.divider()
    if st.button("Se déconnecter", type="secondary"):
        logout()


# Router principal
if "api_token" not in st.session_state or not st.session_state["api_token"]:
    show_login()
else:
    show_home()
