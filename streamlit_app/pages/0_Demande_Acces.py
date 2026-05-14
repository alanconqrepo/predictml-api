"""
Page publique de demande de création de compte — accessible sans authentification
"""

import os

import requests
import streamlit as st

st.set_page_config(
    page_title="Demande d'accès — PredictML",
    page_icon="📝",
    layout="centered",
)

DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("📝 Demande d'accès à PredictML")
st.markdown(
    "Remplissez ce formulaire pour demander un compte. "
    "Un administrateur examinera votre demande et vous communiquera votre token d'accès."
)

api_url = st.session_state.get("api_url", DEFAULT_API_URL)

with st.form("account_request_form"):
    col1, col2 = st.columns(2)
    username = col1.text_input(
        "Nom d'utilisateur souhaité *",
        placeholder="jean_dupont",
        help="Entre 3 et 50 caractères.",
    )
    email = col2.text_input(
        "Email *",
        placeholder="jean@example.com",
    )
    role_requested = st.selectbox(
        "Type d'accès souhaité",
        ["user", "readonly"],
        format_func=lambda r: "Utilisateur (peut faire des prédictions)" if r == "user" else "Lecture seule",
    )
    message = st.text_area(
        "Message à l'administrateur (optionnel)",
        placeholder="Décrivez votre cas d'usage, votre équipe, etc.",
        max_chars=500,
    )
    submitted = st.form_submit_button("Soumettre ma demande", use_container_width=True, type="primary")

if submitted:
    if not username or not email:
        st.error("Le nom d'utilisateur et l'email sont obligatoires.")
    elif len(username) < 3:
        st.error("Le nom d'utilisateur doit contenir au moins 3 caractères.")
    else:
        try:
            resp = requests.post(
                f"{api_url}/account-requests",
                json={
                    "username": username,
                    "email": email,
                    "message": message or None,
                    "role_requested": role_requested,
                },
                timeout=10,
            )
            if resp.status_code == 201:
                st.success(
                    "✅ Votre demande a bien été enregistrée. "
                    "Un administrateur l'examinera et vous contactera directement "
                    "pour vous transmettre votre token d'accès."
                )
                st.info(f"Référence de votre demande : **#{resp.json()['id']}**")
            elif resp.status_code == 409:
                detail = resp.json().get("detail", "")
                if "en attente" in detail:
                    st.warning(
                        "Une demande est déjà en attente pour cet email. "
                        "Veuillez patienter qu'un administrateur la traite."
                    )
                else:
                    st.error(
                        "Un compte existe déjà avec cet email ou ce nom d'utilisateur. "
                        "Si vous avez perdu votre token, contactez un administrateur."
                    )
            elif resp.status_code == 422:
                st.error("Données invalides : vérifiez l'email et le nom d'utilisateur.")
            elif resp.status_code == 429:
                st.error(
                    "Trop de demandes soumises récemment. "
                    "Veuillez réessayer dans quelques instants."
                )
            else:
                st.error(f"Erreur inattendue ({resp.status_code}). Veuillez réessayer.")
        except requests.exceptions.ConnectionError:
            st.error(
                f"Impossible de joindre l'API ({api_url}). "
                "Vérifiez l'URL ou contactez un administrateur."
            )
        except Exception as e:
            st.error(f"Erreur : {e}")

st.divider()
st.markdown(
    "Vous avez déjà un token ? [Connectez-vous sur la page d'accueil](/) ou "
    "utilisez l'application directement via l'API."
)
