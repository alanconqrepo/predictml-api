"""
Page d'accueil et login — PredictML Admin Dashboard
"""

import os
from urllib.parse import urlparse

import streamlit as st
from utils.api_client import APIClient
from utils.auth import logout
from utils.ui_helpers import show_token_with_copy


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
        elif not _is_valid_api_url(api_url):
            st.error("URL invalide. Elle doit commencer par http:// ou https://")
        else:
            client = APIClient(base_url=api_url, token=token)
            is_valid, is_admin = client.check_auth()

            if not is_valid:
                st.error("Token invalide ou API inaccessible.")
            else:
                st.session_state["api_token"] = token
                st.session_state["api_url"] = api_url
                st.session_state["is_admin"] = is_admin
                st.rerun()

    st.divider()

    with st.expander("Vous n'avez pas encore de token ?"):
        st.markdown("""
**Premier accès admin**

Le token admin est défini par la variable d'environnement **`ADMIN_TOKEN`**.

---

**Nouvel utilisateur**

Soumettez une demande via la page **"Demande d'accès"** dans le menu — un admin vous
communiquera votre token une fois approuvé.
""")
        st.markdown("📝 [Soumettre une demande d'accès](/Demande_Acces)")


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

        with st.expander("🔑 Mon token API"):
            try:
                me = client.get_me()
                show_token_with_copy(me["api_token"])
            except Exception:
                st.error("Impossible de charger le token.")

        if st.button("Se déconnecter", type="secondary", use_container_width=True):
            logout()

        # Badge demandes en attente (admin uniquement)
        if st.session_state.get("is_admin"):
            try:
                n_pending = client.get_pending_account_requests_count()
                if n_pending > 0:
                    st.warning(f"🔔 {n_pending} demande(s) d'accès en attente")
                    st.page_link("pages/1_Users.py", label="Gérer les demandes →")
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



# Router principal — navigation conditionnelle selon l'état de connexion
_logged_in = bool(st.session_state.get("api_token"))

_DARK_CSS = """
<style>
[data-testid="stApp"] { background-color: #0e1117; }
[data-testid="stHeader"] { background-color: #0e1117; }
[data-testid="stSidebar"] { background-color: #262730; }
body, p, span, label { color: #fafafa; }
h1, h2, h3, h4, h5, h6 { color: #fafafa; }
.stMarkdown, .stText, .stCaption { color: #fafafa; }
input[type="text"], input[type="password"], textarea {
    background-color: #262730 !important;
    color: #fafafa !important;
    border-color: #555 !important;
}
[data-baseweb="input"] { background-color: #262730 !important; }
[data-baseweb="select"] > div { background-color: #262730 !important; color: #fafafa !important; }
[data-testid="stForm"] { border-color: #555; }
button {
    background-color: #262730 !important;
    color: #fafafa !important;
    border-color: #555 !important;
}
button:hover {
    background-color: #3a3c4a !important;
    border-color: #888 !important;
}
button[kind="primary"] {
    background-color: #c0392b !important;
    border-color: #c0392b !important;
}
code { background-color: #262730; color: #e6e6e6; }
pre { background-color: #1a1c23 !important; }
hr { border-color: #555; }
[data-testid="stMetricValue"] { color: #fafafa; }
[data-testid="stMetricLabel"] { color: #a0a0a0; }
</style>
"""

if _logged_in:
    _pg = st.navigation([
        st.Page(show_home, title="Accueil", default=True),
        st.Page("pages/1_Users.py", title="Users"),
        st.Page("pages/2_Models.py", title="Models"),
        st.Page("pages/3_Predictions.py", title="Predictions"),
        st.Page("pages/4_Stats.py", title="Stats"),
        st.Page("pages/6_AB_Testing.py", title="AB Testing"),
        st.Page("pages/7_Supervision.py", title="Supervision"),
        st.Page("pages/8_Retrain.py", title="Retrain"),
        st.Page("pages/9_Golden_Tests.py", title="Golden Tests"),
        st.Page("pages/10_Aide.py", title="Aide"),
        st.Page("pages/5_Code_Example.py", title="Code Example"),
    ])
else:
    _pg = st.navigation([
        st.Page(show_login, title="Connexion", default=True),
        st.Page("pages/0_Demande_Acces.py", title="Demande d'accès"),
    ])

# Dark mode toggle — visible sur toutes les pages
if st.sidebar.toggle("Mode sombre", key="dark_mode"):
    st.markdown(_DARK_CSS, unsafe_allow_html=True)

_pg.run()
