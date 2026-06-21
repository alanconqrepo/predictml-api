"""
Internationalisation — PredictML Admin Dashboard

Usage in any page:
    from utils.i18n import t
    st.title(t("models.title"))
    st.info(t("models.upload.success", name=result["name"], version=result["version"]))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import yaml

# ── Constants ──────────────────────────────────────────────────────────────────

SUPPORTED_LANGS = ("fr", "en")
DEFAULT_LANG = "fr"
_TRANSLATIONS_DIR = Path(__file__).parent.parent / "translations"


# ── YAML loading — mtime-aware cache (auto-reloads when file changes on disk) ──

# { lang: (mtime_float, data_dict) }
_lang_cache: dict[str, tuple[float, dict]] = {}


def _load_lang(lang: str) -> dict:
    """Load and parse a YAML translation file.

    Cached in-process but invalidated automatically whenever the file's
    modification time changes — no container restart needed after editing a YAML.
    """
    path = _TRANSLATIONS_DIR / f"{lang}.yaml"
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    cached = _lang_cache.get(lang)
    if cached and cached[0] == mtime:
        return cached[1]
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _lang_cache[lang] = (mtime, data)
    return data


def _get_nested(data: dict, key: str) -> str | None:
    """Resolve a dot-notation key like 'models.upload.title' into the nested dict."""
    node: Any = data
    for part in key.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node if isinstance(node, str) else None


# ── Public API ─────────────────────────────────────────────────────────────────


def get_lang() -> str:
    """Return the current language from session state, defaulting to 'fr'."""
    return st.session_state.get("lang", DEFAULT_LANG)


def set_lang(lang: str) -> None:
    """Set the current language in session state."""
    if lang in SUPPORTED_LANGS:
        st.session_state["lang"] = lang


def t(key: str, **kwargs: Any) -> str:
    """
    Translate a dot-notation key to the current language.

    Falls back from EN → FR → raw key (in brackets) if not found.

    Supports named string interpolation via Python str.format_map():
        t("models.upload.success", name="iris", version="1.0.0")
        → "Modèle iris v1.0.0 uploadé avec succès."
    """
    lang = get_lang()
    value = _get_nested(_load_lang(lang), key)

    # Fallback to French if key missing in current language
    if value is None and lang != DEFAULT_LANG:
        value = _get_nested(_load_lang(DEFAULT_LANG), key)

    # Last resort: visible debug signal (never silently fails)
    if value is None:
        return f"[{key}]"

    # Apply named interpolations if provided
    if kwargs:
        try:
            return value.format_map(kwargs)
        except (KeyError, ValueError):
            return value  # Return un-interpolated string rather than crashing

    return value


def init_lang() -> None:
    """
    Initialize language in session_state if not already set.
    Call once at app startup (top of app.py, before any st.* calls).
    """
    st.session_state.setdefault("lang", DEFAULT_LANG)


def language_switcher() -> None:
    """
    Render a compact FR/EN radio switcher in the sidebar.
    Triggers st.rerun() when the language changes.
    Call from app.py so it appears on every page.
    """
    current = get_lang()
    selected = st.sidebar.radio(
        "🌐",
        options=list(SUPPORTED_LANGS),
        format_func=lambda x: "🇫🇷 FR" if x == "fr" else "🇬🇧 EN",
        index=SUPPORTED_LANGS.index(current),
        horizontal=True,
        key="_lang_switcher",
        label_visibility="collapsed",
    )
    if selected != current:
        set_lang(selected)
        st.rerun()
