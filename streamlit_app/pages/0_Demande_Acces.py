"""
Page publique de demande de création de compte — accessible sans authentification
"""

import os

import requests
import streamlit as st
from utils.i18n import t

if st.session_state.get("api_token"):
    st.switch_page("app.py")

DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title(t("access_request.title"))
st.markdown(t("access_request.intro"))

api_url = st.session_state.get("api_url", DEFAULT_API_URL)

with st.form("account_request_form"):
    col1, col2 = st.columns(2)
    username = col1.text_input(
        t("access_request.username_label"),
        placeholder=t("access_request.username_placeholder"),
        help=t("access_request.username_help"),
    )
    email = col2.text_input(
        t("access_request.email_label"),
        placeholder=t("access_request.email_placeholder"),
    )
    role_requested = st.selectbox(
        t("access_request.role_label"),
        ["user", "readonly"],
        format_func=lambda r: t(f"access_request.role_{r}"),
    )
    message = st.text_area(
        t("access_request.message_label"),
        placeholder=t("access_request.message_placeholder"),
        max_chars=500,
    )
    submitted = st.form_submit_button(t("access_request.submit_btn"), width='stretch', type="primary")

if submitted:
    if not username or not email:
        st.error(t("access_request.error_required"))
    elif len(username) < 3:
        st.error(t("access_request.error_username_short"))
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
                st.success(t("access_request.success"))
                st.info(t("access_request.reference", id=resp.json()['id']))
            elif resp.status_code == 409:
                detail = resp.json().get("detail", "")
                if "en attente" in detail:
                    st.warning(t("access_request.error_pending"))
                else:
                    st.error(t("access_request.error_exists"))
            elif resp.status_code == 422:
                st.error(t("access_request.error_invalid"))
            elif resp.status_code == 429:
                st.error(t("access_request.error_rate_limit"))
            else:
                st.error(t("access_request.error_unexpected", code=resp.status_code))
        except requests.exceptions.ConnectionError:
            st.error(t("access_request.error_connection", api_url=api_url))
        except Exception as e:
            st.error(t("access_request.error_generic", error=e))

st.divider()
st.markdown(t("access_request.already_have_token"))
