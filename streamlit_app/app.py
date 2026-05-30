"""
Home page and login — PredictML Admin Dashboard
"""

import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from urllib.parse import urlparse

import streamlit as st
from utils.api_client import APIClient
from utils.auth import logout
from utils.i18n import init_lang, language_switcher, t
from utils.ui_helpers import show_token_with_copy

# Sessions stored in /tmp — survive Streamlit hot-reloads.
_SESSION_DIR = os.path.join(tempfile.gettempdir(), "predictml_sessions")
os.makedirs(_SESSION_DIR, exist_ok=True)
_SESSION_TTL = 8 * 3600  # 8 hours


def _session_path(sid: str) -> str:
    return os.path.join(_SESSION_DIR, f"{sid}.json")


def _save_session(token: str, api_url: str, is_admin: bool, role: str = "user") -> str:
    sid = str(uuid.uuid4())
    with open(_session_path(sid), "w") as f:
        json.dump(
            {
                "token": token,
                "api_url": api_url,
                "is_admin": is_admin,
                "role": role,
                "expires_at": time.time() + _SESSION_TTL,
            },
            f,
        )
    return sid


def _restore_session(sid: str) -> bool:
    path = _session_path(sid)
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return False
    if data.get("expires_at", 0) < time.time():
        os.remove(path)
        return False
    st.session_state["api_token"] = data["token"]
    st.session_state["api_url"] = data["api_url"]
    st.session_state["is_admin"] = data["is_admin"]
    st.session_state["role"] = data.get("role", "user")
    st.session_state["_sid"] = sid
    return True


def _clear_session() -> None:
    sid = st.session_state.pop("_sid", None) or st.query_params.get("sid")
    if sid:
        path = _session_path(sid)
        if os.path.exists(path):
            os.remove(path)
    st.query_params.clear()


def _is_valid_api_url(url: str) -> bool:
    """Check that the URL is http/https and points to a non-empty host."""
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

init_lang()

DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")


def show_login():
    st.title(t("login.title"))
    st.markdown(t("login.subtitle"))

    with st.form("login_form"):
        api_url = st.text_input(t("login.api_url_label"), value=DEFAULT_API_URL)
        token = st.text_input(
            t("login.token_label"), type="password", placeholder=t("login.token_placeholder")
        )
        submitted = st.form_submit_button(t("login.submit_btn"), width="stretch")

    if submitted:
        if not token:
            st.error(t("login.error_no_token"))
        elif not _is_valid_api_url(api_url):
            st.error(t("login.error_invalid_url"))
        else:
            client = APIClient(base_url=api_url, token=token)
            is_valid, is_admin = client.check_auth()

            if not is_valid:
                st.error(t("login.error_invalid_token"))
            else:
                try:
                    me = client.get_me()
                    role = me.get("role", "admin" if is_admin else "user")
                except Exception:
                    role = "admin" if is_admin else "user"
                st.session_state["api_token"] = token
                st.session_state["api_url"] = api_url
                st.session_state["is_admin"] = is_admin
                st.session_state["role"] = role
                sid = _save_session(token, api_url, is_admin, role)
                st.session_state["_sid"] = sid
                st.query_params["sid"] = sid
                st.rerun()

    st.divider()

    with st.expander(t("login.no_token_expander")):
        st.markdown(t("login.no_token_body"))
        st.markdown(t("login.no_token_link"))


def show_home():  # noqa: C901
    import pandas as pd

    st.title(t("home.title"))

    role_badge = t("home.role_admin") if st.session_state.get("is_admin") else t("home.role_user")
    st.caption(t("home.caption", role=role_badge, api_url=st.session_state.get("api_url")))

    client = APIClient(
        base_url=st.session_state["api_url"],
        token=st.session_state["api_token"],
    )

    now = datetime.utcnow()
    start_30 = (now - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00")
    end_now = now.strftime("%Y-%m-%dT23:59:59")
    is_admin = st.session_state.get("is_admin", False)

    # ── Data loading (each block isolated) ───────────────────────────
    health: dict = {}
    monitoring: dict = {}
    pred_stats: list = []
    coverage: dict = {}
    leaderboard: list = []
    all_models: list = []

    try:
        health = client.get_health()
    except Exception:
        pass

    try:
        monitoring = client.get_monitoring_overview(start=start_30, end=end_now)
    except Exception:
        pass

    try:
        pred_stats = client.get_prediction_stats(days=30)
    except Exception:
        pass

    try:
        coverage = client.get_observed_results_stats()
    except Exception:
        pass

    try:
        leaderboard = client.get_leaderboard(metric="accuracy", days=30)
    except Exception:
        pass

    try:
        all_models = client.list_models()
    except Exception:
        pass

    # ── 1. System status banner ────────────────────────────────────────────
    api_status = health.get("status", "unknown")
    db_status = health.get("database", "unknown")
    cached = health.get("models_cached", health.get("cached_models", 0))
    mon_models: list = monitoring.get("models", [])
    models_alert_count = sum(
        1 for m in mon_models if m.get("health_status") in ("warning", "critical")
    )

    st.subheader(t("home.status.subheader"))
    st.caption(t("home.status.subheader_caption"))
    col_s, col_db, col_cache, col_alerts = st.columns(4)
    col_s.metric(
        t("home.status.api_label"),
        (
            t("home.status.api_online")
            if api_status == "healthy"
            else (t("home.status.api_degraded") if api_status == "degraded" else t("home.status.api_unknown"))
        ),
        help=t("home.status.api_label_help"),
    )
    col_db.metric(
        t("home.status.db_label"),
        t("home.status.db_connected") if db_status == "connected" else f"⚠️ {db_status}",
        help=t("home.status.db_label_help"),
    )
    col_cache.metric(
        t("home.status.cache_label"),
        t("home.status.cache_value", count=cached) if cached else "—",
        help=t("home.status.cache_label_help"),
    )
    col_alerts.metric(
        t("home.status.alerts_label"),
        t("home.status.alerts_count", count=models_alert_count) if models_alert_count > 0 else t("home.status.alerts_none"),
        help=t("home.status.alerts_label_help"),
    )

    # ── 2. Global KPIs — last 30 days ─────────────────────────────────
    st.divider()
    st.subheader(t("home.kpis.subheader"))
    st.caption(t("home.kpis.subheader_caption"))

    total_preds = sum(s.get("total_predictions", 0) for s in pred_stats)
    total_err = sum(s.get("error_count", 0) for s in pred_stats)
    success_rate = round((1 - total_err / total_preds) * 100, 1) if total_preds > 0 else None
    p95_vals = [
        s["p95_response_time_ms"] for s in pred_stats if s.get("p95_response_time_ms") is not None
    ]
    latency_p95 = round(sum(p95_vals) / len(p95_vals), 0) if p95_vals else None
    cov_rate = coverage.get("coverage_rate")
    coverage_pct = f"{cov_rate * 100:.1f}%" if cov_rate is not None else "—"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t("home.kpis.predictions"), f"{total_preds:,}" if total_preds else "—", help=t("home.kpis.predictions_help"))
    k2.metric(
        t("home.kpis.success_rate"),
        f"{success_rate}%" if success_rate is not None else "—",
        help=t("home.kpis.success_rate_help"),
    )
    k3.metric(
        t("home.kpis.latency_p95"),
        f"{int(latency_p95)} ms" if latency_p95 is not None else "—",
        help=t("home.kpis.latency_p95_help"),
    )
    k4.metric(
        t("home.kpis.coverage"),
        coverage_pct,
        help=t("home.kpis.coverage_help"),
    )

    # ── 3. Active alerts (conditional) ────────────────────────────────────
    critical = [m for m in mon_models if m.get("health_status") == "critical"]
    warning = [m for m in mon_models if m.get("health_status") == "warning"]

    if critical or warning:
        st.divider()
        st.subheader(t("home.alerts.subheader"))
        st.caption(t("home.alerts.subheader_caption"))
        if critical:
            names_c = ", ".join(f"**{m.get('model_name', '?')}**" for m in critical)
            st.error(t("home.alerts.critical", count=len(critical), names=names_c))
        if warning:
            names_w = ", ".join(f"**{m.get('model_name', '?')}**" for m in warning)
            st.warning(t("home.alerts.warning", count=len(warning), names=names_w))
        st.page_link("pages/7_Supervision.py", label=t("home.alerts.supervision_link"))

    # ── 4. Leaderboard ────────────────────────────────────────────────────
    st.divider()
    st.subheader(t("home.leaderboard.subheader"))
    st.caption(t("home.leaderboard.subheader_caption"))
    if leaderboard:
        _mode_badge = {
            "ab":      "🟠 A/B",
            "ab_test": "🟠 A/B",
            "shadow":  "🟣 Shadow",
        }
        rows = []
        for item in leaderboard[:5]:
            acc = item.get("accuracy")
            drift = item.get("drift_status") or "—"
            mode = item.get("deployment_mode")
            if mode in _mode_badge:
                mode_label = _mode_badge[mode]
            elif item.get("is_production"):
                mode_label = "🟢 Production"
            else:
                mode_label = "✅ Actif"
            rows.append(
                {
                    "#": item.get("rank", "—"),
                    t("home.leaderboard.col_model"): item.get("name") or item.get("model_name", "—"),
                    t("home.leaderboard.col_version"): item.get("version", "—"),
                    t("home.leaderboard.col_mode"): mode_label,
                    t("home.leaderboard.col_accuracy"): f"{acc:.1%}" if acc is not None else "—",
                    t("home.leaderboard.col_predictions"): f"{item.get('predictions_count', 0):,}",
                    t("home.leaderboard.col_drift"): drift,
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.page_link("pages/4_Stats.py", label=t("home.leaderboard.full_link"))
    else:
        n_prod = sum(1 for m in all_models if m.get("is_production"))
        if all_models:
            st.metric(t("home.leaderboard.active_models"), len(all_models))
            st.metric(t("home.leaderboard.prod_versions"), n_prod)
        else:
            st.info(t("home.leaderboard.no_models"))
            st.page_link("pages/2_Models.py", label=t("home.leaderboard.upload_link"))

    # ── 5a. Health grid ───────────────────────────────────────────────────
    st.divider()
    st.subheader(t("home.health.subheader"))
    st.caption(t("home.health.subheader_caption"))
    if mon_models:
        for m in mon_models:
            status = m.get("health_status", "unknown")
            badge = {"ok": "✅", "warning": "⚠️", "critical": "🔴"}.get(status, "❓")
            name = m.get("model_name", "?")
            err = m.get("error_rate")
            p95 = m.get("latency_p95_ms") or m.get("response_time_p95_ms")
            preds = m.get("predictions_count") or m.get("total_predictions", 0)
            details = []
            if err is not None:
                details.append(t("home.health.detail_error", rate=f"{err:.1%}"))
            if p95 is not None:
                details.append(t("home.health.detail_p95", ms=int(p95)))
            if preds:
                details.append(t("home.health.detail_preds", count=f"{preds:,}"))
            detail_str = f" — {' · '.join(details)}" if details else ""
            st.markdown(f"{badge} **{name}**{detail_str}")
        st.page_link("pages/7_Supervision.py", label=t("home.health.supervision_link"))
    elif all_models:
        for m in all_models[:8]:
            badge = "🟢" if m.get("is_production") else "⚪"
            st.markdown(f"{badge} **{m.get('name')}** v{m.get('version')}")
        if len(all_models) > 8:
            st.caption(t("home.health.more_versions", count=len(all_models) - 8))
    else:
        st.info(t("home.health.no_models"))

    # ── 5. Active A/B tests and Shadow deployments (conditional) ──────────────────────────
    ab_versions = [m for m in all_models if m.get("deployment_mode") in ("ab", "ab_test")]
    shadow_versions = [m for m in all_models if m.get("deployment_mode") == "shadow"]

    if ab_versions or shadow_versions:
        st.divider()
        st.subheader(t("home.ab_shadow.subheader"))
        st.caption(t("home.ab_shadow.subheader_caption"))
        c1, c2 = st.columns(2)
        ab_names = ", ".join({m.get("name", "?") for m in ab_versions}) if ab_versions else "—"
        sh_names = ", ".join({m.get("name", "?") for m in shadow_versions}) if shadow_versions else "—"
        c1.info(t("home.ab_shadow.ab_info", count=len(ab_versions), names=ab_names))
        c2.info(t("home.ab_shadow.shadow_info", count=len(shadow_versions), names=sh_names))
        st.page_link("pages/6_AB_Testing.py", label=t("home.ab_shadow.manage_link"))

    # ── 6. Navigation cards ───────────────────────────────────────────────
    st.divider()
    st.subheader(t("home.nav.subheader"))
    st.caption(t("home.nav.subheader_caption"))

    nav_items = [
        (
            "🤖",
            t("home.nav.models_title"),
            t("home.nav.models_desc"),
            "pages/2_Models.py",
        ),
        (
            "🔮",
            t("home.nav.predictions_title"),
            t("home.nav.predictions_desc"),
            "pages/3_Predictions.py",
        ),
        (
            "📊",
            t("home.nav.stats_title"),
            t("home.nav.stats_desc"),
            "pages/4_Stats.py",
        ),
        (
            "🛡️",
            t("home.nav.supervision_title"),
            t("home.nav.supervision_desc"),
            "pages/7_Supervision.py",
        ),
        (
            "🧪",
            t("home.nav.ab_title"),
            t("home.nav.ab_desc"),
            "pages/6_AB_Testing.py",
        ),
        (
            "🔄",
            t("home.nav.retrain_title"),
            t("home.nav.retrain_desc"),
            "pages/8_Retrain.py",
        ),
        (
            "🏆",
            t("home.nav.golden_title"),
            t("home.nav.golden_desc"),
            "pages/9_Golden_Tests.py",
        ),
        (
            "📚",
            t("home.nav.help_title"),
            t("home.nav.help_desc"),
            "pages/10_Aide.py",
        ),
    ]
    if is_admin:
        nav_items.insert(
            0,
            (
                "👥",
                t("home.nav.users_title"),
                t("home.nav.users_desc"),
                "pages/1_Users.py",
            ),
        )

    for i in range(0, len(nav_items), 3):
        row = nav_items[i : i + 3]
        cols = st.columns(3)
        for col, (ico, title, desc, page_path) in zip(cols, row):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{ico} {title}**")
                    st.caption(desc)
                    st.page_link(page_path, label=t("home.nav.open_btn"))


# Main router — conditional navigation based on login state
# Session restore after F5: sid in session_state or in the URL
if not st.session_state.get("api_token"):
    _sid = st.session_state.get("_sid") or st.query_params.get("sid")
    if _sid:
        if not _restore_session(_sid):
            st.query_params.clear()

_logged_in = bool(st.session_state.get("api_token"))

# Rewrite sid to URL on each render (navigation removes it)
if _logged_in and st.session_state.get("_sid"):
    if st.query_params.get("sid") != st.session_state["_sid"]:
        st.query_params["sid"] = st.session_state["_sid"]

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
/* High specificity to beat generated st-emotion-cache-xxx classes */
html body [data-testid="stApp"] button {
    background-color: #262730 !important;
    color: #fafafa !important;
    border-color: #555 !important;
}
html body [data-testid="stApp"] button:hover {
    background-color: #3a3c4a !important;
    border-color: #888 !important;
}
html body [data-testid="stApp"] button[kind="primary"] {
    background-color: #c0392b !important;
    border-color: #c0392b !important;
}
html body [data-testid="stApp"] button p {
    color: #fafafa !important;
}
code { background-color: #262730; color: #e6e6e6; }
pre { background-color: #1a1c23 !important; }
hr { border-color: #555; }
[data-testid="stMetricValue"] { color: #fafafa; }
[data-testid="stMetricLabel"] { color: #a0a0a0; }
/* Expanders — broad targeting because Streamlit generates random classes */
details {
    background-color: #1e2029 !important;
    border-color: #555 !important;
}
details summary {
    background-color: #1e2029 !important;
    color: #fafafa !important;
}
details summary svg, details summary p {
    fill: #fafafa !important;
    color: #fafafa !important;
}
details > div, details > section {
    background-color: #1e2029 !important;
}
[data-testid="stExpander"],
[data-testid="stExpanderDetails"] {
    background-color: #1e2029 !important;
    border-color: #555 !important;
}
/* Inputs inside expanders */
details input, details textarea, details select {
    background-color: #262730 !important;
    color: #fafafa !important;
    border-color: #555 !important;
}
/* All numeric inputs */
input[type="number"] {
    background-color: #262730 !important;
    color: #fafafa !important;
    border-color: #555 !important;
}
</style>
"""

if _logged_in:
    _client = APIClient(
        base_url=st.session_state["api_url"],
        token=st.session_state["api_token"],
    )

    # Build navigation first so st.page_link calls in the sidebar are valid
    _base_pages = [
        st.Page(show_home, title="Home", default=True),
        st.Page("pages/1_Users.py", title="Users"),
        st.Page("pages/2_Models.py", title="Models"),
        st.Page("pages/3_Predictions.py", title="Predictions"),
        st.Page("pages/4_Stats.py", title="Stats"),
        st.Page("pages/6_AB_Testing.py", title="AB Testing"),
        st.Page("pages/7_Supervision.py", title="Supervision"),
        st.Page("pages/8_Retrain.py", title="Retrain"),
        st.Page("pages/9_Golden_Tests.py", title="Golden Tests"),
        st.Page("pages/10_Aide.py", title="Help"),
        st.Page("pages/5_Code_Example.py", title="Code Example"),
    ]
    if st.session_state.get("is_admin"):
        _base_pages.append(st.Page("pages/11_Services.py", title="Services"))
    _pg = st.navigation(_base_pages)

    # Sidebar "My account" — shown on all pages
    with st.sidebar:
        st.subheader(t("sidebar.account_title"))
        try:
            quota = _client.get_my_quota()
            used = quota["used_today"]
            limit = quota["rate_limit_per_day"]
            remaining = quota["remaining_today"]
            st.progress(used / limit if limit > 0 else 0)
            st.caption(t("sidebar.quota_caption", used=used, limit=limit))
            if remaining == 0:
                st.warning(t("sidebar.quota_exhausted"))
        except Exception:
            pass

        with st.expander(t("sidebar.token_expander")):
            try:
                me = _client.get_me()
                show_token_with_copy(me["api_token"])
            except Exception:
                st.error(t("sidebar.token_load_error"))

        if st.button(t("sidebar.logout_btn"), type="secondary", width="stretch"):
            _clear_session()
            logout()

        if st.session_state.get("is_admin"):
            try:
                n_pending = _client.get_pending_account_requests_count()
                if n_pending > 0:
                    st.warning(t("sidebar.pending_requests", count=n_pending))
                    st.page_link("pages/1_Users.py", label=t("sidebar.manage_requests_link"))
            except Exception:
                pass

        st.divider()
        if st.session_state.get("role") != "readonly":
            _grafana_url = os.environ.get("GRAFANA_PUBLIC_URL", "http://localhost:3000")
            _minio_url = os.environ.get("MINIO_CONSOLE_PUBLIC_URL", "http://localhost:9011")
            _mlflow_url = os.environ.get("MLFLOW_PUBLIC_URL", "http://localhost:5000")
            st.subheader(t("sidebar.services_title"))
            st.markdown(
                f"🔗 [Swagger](http://localhost/docs)  \n"
                f"📊 [Grafana]({_grafana_url}/dashboards)  \n"
                f"🪣 [MinIO]({_minio_url}/login)  \n"
                f"🧪 [MLflow]({_mlflow_url}/)"
            )
            if st.session_state.get("is_admin"):
                st.page_link("pages/11_Services.py", label=t("sidebar.credentials_link"))
            st.divider()
else:
    _pg = st.navigation(
        [
            st.Page(show_login, title="Login", default=True),
            st.Page("pages/0_Demande_Acces.py", title="Access Request"),
        ]
    )

# Language switcher — visible on all pages (login and logged-in)
language_switcher()

# Dark mode toggle — visible on all pages
if st.sidebar.toggle(t("sidebar.dark_mode_toggle"), key="dark_mode"):
    st.markdown(_DARK_CSS, unsafe_allow_html=True)

_pg.run()
