"""
Services & Accès — admin uniquement
Liens de redirection vers les interfaces infrastructure avec credentials associés.
Les credentials sont lus depuis les variables d'environnement du conteneur Streamlit.
"""

import os

import streamlit as st
from utils.auth import require_admin

st.set_page_config(
    page_title="Services — PredictML",
    page_icon="🔌",
    layout="wide",
)

require_admin()

# ── Lecture des variables d'environnement ──────────────────────────────────────
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
st.title("🔌 Services & Accès")
st.caption(
    "Accès rapide aux interfaces d'administration de l'infrastructure — "
    "réservé aux administrateurs"
)

# Avertissement variables manquantes
_missing = []
if not GRAFANA_PASS:
    _missing.append("`GRAFANA_ADMIN_PASSWORD`")
if not MINIO_PASS:
    _missing.append("`MINIO_ROOT_PASSWORD`")
if not MLFLOW_PASS:
    _missing.append("`MLFLOW_ADMIN_PASSWORD`")

if _missing:
    st.warning(
        f"⚠️ Variable(s) d'environnement manquante(s) : {', '.join(_missing)}.  \n"
        "Définissez-les dans votre fichier `.env` et redémarrez le conteneur Streamlit."
    )

st.divider()

# ── Toggle affichage des secrets ───────────────────────────────────────────────
_col_toggle, _ = st.columns([1, 3])
with _col_toggle:
    show_secrets = st.toggle("🔓 Afficher les mots de passe", value=False)

st.divider()

# ── Définition des services ────────────────────────────────────────────────────
_SERVICES = [
    {
        "name": "Grafana",
        "icon": "📊",
        "description": (
            "Monitoring, dashboards et alertes — stack LGTM "
            "(Loki · Grafana · Tempo · Prometheus)"
        ),
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
        "description": (
            "Stockage objets S3 — modèles `.joblib`, "
            "artefacts MLflow, datasets d'entraînement"
        ),
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
        "description": (
            "Experiment tracking — runs, métriques, artefacts, "
            "comparaisons et registre de modèles"
        ),
        "url": MLFLOW_URL,
        "link_path": "/",
        "user": MLFLOW_USER,
        "password": MLFLOW_PASS,
        "env_user": "MLFLOW_ADMIN_USER",
        "env_pass": "MLFLOW_ADMIN_PASSWORD",
        "env_url": "MLFLOW_PUBLIC_URL",
    },
]

# ── Rendu des cartes de service ────────────────────────────────────────────────
cols = st.columns(3, gap="large")

for col, svc in zip(cols, _SERVICES):
    with col:
        st.subheader(f"{svc['icon']} {svc['name']}")
        st.caption(svc["description"])

        # Bouton d'accès principal
        _full_url = svc["url"].rstrip("/") + svc["link_path"]
        st.link_button(
            f"Ouvrir {svc['name']} →",
            _full_url,
            type="primary",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("**Identifiants**")

        # Login
        st.text_input(
            "Login",
            value=svc["user"] if svc["user"] else f"—  (var: {svc['env_user']})",
            disabled=True,
            key=f"login_{svc['name']}",
            label_visibility="collapsed",
            placeholder="Login",
        )

        # Mot de passe — masqué par défaut, st.code quand visible (bouton copier intégré)
        if svc["password"]:
            if show_secrets:
                st.code(svc["password"], language=None)
            else:
                st.text_input(
                    "Mot de passe",
                    value=svc["password"],
                    type="password",
                    disabled=True,
                    key=f"pass_{svc['name']}",
                    label_visibility="collapsed",
                )
        else:
            st.error(f"Variable `{svc['env_pass']}` non définie dans `.env`")

        st.markdown("---")
        st.markdown("**URL du service**")
        st.code(svc["url"], language=None)
        st.caption(f"Variable d'env : `{svc['env_url']}`")

# ── Note de sécurité ───────────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ Configuration & sécurité", expanded=False):
    st.markdown("""
**Comment configurer les URLs publiques ?**

Ces liens utilisent les **URLs publiques** des services (accessibles depuis votre navigateur),
distinctes des URLs internes Docker. Définissez-les dans votre `.env` :

| Variable | Défaut | Description |
|---|---|---|
| `GRAFANA_PUBLIC_URL` | `http://localhost:3000` | Interface Grafana |
| `MINIO_CONSOLE_PUBLIC_URL` | `http://localhost:9011` | Console MinIO |
| `MLFLOW_PUBLIC_URL` | `http://localhost:5000` | Interface MLflow |
| `GRAFANA_ADMIN_USER` | `admin` | Login Grafana |
| `GRAFANA_ADMIN_PASSWORD` | *(généré)* | Mot de passe Grafana |
| `MINIO_ROOT_USER` | `minioadmin` | Login MinIO |
| `MINIO_ROOT_PASSWORD` | *(généré)* | Mot de passe MinIO |
| `MLFLOW_ADMIN_USER` | `admin` | Login MLflow |
| `MLFLOW_ADMIN_PASSWORD` | *(généré)* | Mot de passe MLflow |

Les secrets sont générés automatiquement par `scripts/init_env.sh` lors du premier déploiement.

**Sécurité :** cette page est accessible uniquement aux utilisateurs avec le rôle **admin**.
Les mots de passe sont masqués par défaut — activez le toggle pour les copier.
""")
