"""
Shared UI components across Streamlit pages
"""

import streamlit as st

from utils.i18n import t


def show_token_with_copy(token: str) -> None:
    """Display a token with a one-click copy button."""
    st.code(token, language=None)
    st.iframe(
        f"""data:text/html,<button onclick="navigator.clipboard.writeText('{token}').then(() => {{
            this.innerText = '{t("ui.copy_success")}';
            setTimeout(() => this.innerText = '{t("ui.copy_btn")}', 2000);
        }})" style="
            background:%234CAF50; color:white; border:none; padding:6px 14px;
            border-radius:4px; cursor:pointer; font-size:14px;
        ">{t("ui.copy_btn")}</button>""",
        height=42,
    )
