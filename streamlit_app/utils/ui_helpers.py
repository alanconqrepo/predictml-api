"""
Composants UI partagés entre les pages Streamlit
"""

import streamlit as st


def show_token_with_copy(token: str) -> None:
    """Affiche un token avec un bouton copier en un clic."""
    st.code(token, language=None)
    st.iframe(
        f"""data:text/html,<button onclick="navigator.clipboard.writeText('{token}').then(() => {{
            this.innerText = '✅ Copié !';
            setTimeout(() => this.innerText = '📋 Copier le token', 2000);
        }})" style="
            background:%234CAF50; color:white; border:none; padding:6px 14px;
            border-radius:4px; cursor:pointer; font-size:14px;
        ">📋 Copier le token</button>""",
        height=42,
    )
