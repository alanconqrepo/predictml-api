"""
Shared UI components across Streamlit pages
"""

import streamlit as st

from utils.i18n import t


def show_token_with_copy(token: str) -> None:
    """Display a token with a one-click copy button."""
    st.code(token, language=None)
    label = t("ui.copy_btn")
    success = t("ui.copy_success")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"></head><body style="margin:0">
<button onclick="navigator.clipboard.writeText('{token}').then(() => {{
    this.innerText = '{success}';
    setTimeout(() => this.innerText = '{label}', 2000);
}})" style="background:#4CAF50;color:white;border:none;padding:6px 14px;
    border-radius:4px;cursor:pointer;font-size:14px;">{label}</button>
</body></html>"""
    import base64
    src = "data:text/html;base64," + base64.b64encode(html.encode("utf-8")).decode()
    st.iframe(src, height=42)
