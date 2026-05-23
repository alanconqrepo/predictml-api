"""
Page d'accueil et login — PredictML Admin Dashboard
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
from utils.ui_helpers import show_token_with_copy

# Sessions stockées dans /tmp — survive aux hot-reloads Streamlit.
_SESSION_DIR = os.path.join(tempfile.gettempdir(), "predictml_sessions")
os.makedirs(_SESSION_DIR, exist_ok=True)
_SESSION_TTL = 8 * 3600  # 8 heures


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
        submitted = st.form_submit_button("Se connecter", width="stretch")

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


def show_home():  # noqa: C901
    import pandas as pd

    st.title("🤖 PredictML Admin Dashboard")

    role_badge = "👑 Admin" if st.session_state.get("is_admin") else "👤 Utilisateur"
    st.caption(f"Connecté — {role_badge}  |  API: `{st.session_state.get('api_url')}`")

    client = APIClient(
        base_url=st.session_state["api_url"],
        token=st.session_state["api_token"],
    )

    now = datetime.utcnow()
    start_30 = (now - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00")
    end_now = now.strftime("%Y-%m-%dT23:59:59")
    is_admin = st.session_state.get("is_admin", False)

    # ── Chargement des données (chaque bloc isolé) ───────────────────────────
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

    # ── 1. Bandeau statut système ────────────────────────────────────────────
    api_status = health.get("status", "unknown")
    db_status = health.get("database", "unknown")
    cached = health.get("models_cached", health.get("cached_models", 0))
    mon_models: list = monitoring.get("models", [])
    models_alert_count = sum(
        1 for m in mon_models if m.get("health_status") in ("warning", "critical")
    )

    col_s, col_db, col_cache, col_alerts = st.columns(4)
    col_s.metric(
        "Statut API",
        (
            "✅ En ligne"
            if api_status == "healthy"
            else ("⚠️ Dégradé" if api_status == "degraded" else "❓ Inconnu")
        ),
    )
    col_db.metric(
        "Base de données",
        "✅ Connectée" if db_status == "connected" else f"⚠️ {db_status}",
    )
    col_cache.metric("En cache", f"{cached} modèle(s)" if cached else "—")
    col_alerts.metric(
        "Modèles en alerte",
        f"🔴 {models_alert_count}" if models_alert_count > 0 else "✅ Aucune",
    )

    # ── 2. KPIs globaux — 30 derniers jours ─────────────────────────────────
    st.divider()
    st.subheader("📊 Activité — 30 derniers jours")

    total_preds = sum(s.get("total_predictions", 0) for s in pred_stats)
    total_err = sum(s.get("error_count", 0) for s in pred_stats)
    success_rate = round((1 - total_err / total_preds) * 100, 1) if total_preds > 0 else None
    p95_vals = [
        s["response_time_p95_ms"] for s in pred_stats if s.get("response_time_p95_ms") is not None
    ]
    latency_p95 = round(sum(p95_vals) / len(p95_vals), 0) if p95_vals else None
    cov_rate = coverage.get("coverage_rate")
    coverage_pct = f"{cov_rate * 100:.1f}%" if cov_rate is not None else "—"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📈 Prédictions", f"{total_preds:,}" if total_preds else "—")
    k2.metric(
        "✅ Taux de succès",
        f"{success_rate}%" if success_rate is not None else "—",
    )
    k3.metric(
        "⚡ Latence P95",
        f"{int(latency_p95)} ms" if latency_p95 is not None else "—",
    )
    k4.metric(
        "🔖 Couverture terrain",
        coverage_pct,
        help="Pourcentage de prédictions avec un résultat observé (ground truth)",
    )

    # ── 3. Alertes actives (conditionnel) ────────────────────────────────────
    critical = [m for m in mon_models if m.get("health_status") == "critical"]
    warning = [m for m in mon_models if m.get("health_status") == "warning"]

    if critical or warning:
        st.divider()
        if critical:
            names_c = ", ".join(f"**{m.get('model_name', '?')}**" for m in critical)
            st.error(f"🔴 {len(critical)} modèle(s) en état critique : {names_c}")
        if warning:
            names_w = ", ".join(f"**{m.get('model_name', '?')}**" for m in warning)
            st.warning(f"⚠️ {len(warning)} modèle(s) en alerte : {names_w}")
        st.page_link("pages/7_Supervision.py", label="→ Voir la supervision complète")

    # ── 4. Leaderboard + Grille santé côte à côte ────────────────────────────
    st.divider()
    col_lb, col_health = st.columns([3, 2])

    with col_lb:
        st.subheader("🏆 Classement des modèles")
        if leaderboard:
            rows = []
            for item in leaderboard[:5]:
                acc = item.get("accuracy")
                drift = item.get("drift_status") or "—"
                rows.append(
                    {
                        "#": item.get("rank", "—"),
                        "Modèle": item.get("model_name", "—"),
                        "Version": item.get("version", "—"),
                        "Accuracy": f"{acc:.1%}" if acc is not None else "—",
                        "Prédictions": f"{item.get('predictions_count', 0):,}",
                        "Drift": drift,
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.page_link("pages/4_Stats.py", label="📊 Leaderboard complet →")
        else:
            n_prod = sum(1 for m in all_models if m.get("is_production"))
            if all_models:
                st.metric("Modèles actifs", len(all_models))
                st.metric("Versions en production", n_prod)
            else:
                st.info("Aucun modèle enregistré. Commencez par uploader un modèle.")
                st.page_link("pages/2_Models.py", label="🤖 Uploader un modèle →")

    with col_health:
        st.subheader("🛡️ État des modèles")
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
                    details.append(f"erreur : {err:.1%}")
                if p95 is not None:
                    details.append(f"P95 : {int(p95)} ms")
                if preds:
                    details.append(f"{preds:,} pred.")
                detail_str = f" — {' · '.join(details)}" if details else ""
                st.markdown(f"{badge} **{name}**{detail_str}")
            st.page_link("pages/7_Supervision.py", label="🛡️ Supervision complète →")
        elif all_models:
            for m in all_models[:8]:
                badge = "🟢" if m.get("is_production") else "⚪"
                st.markdown(f"{badge} **{m.get('name')}** v{m.get('version')}")
            if len(all_models) > 8:
                st.caption(f"… et {len(all_models) - 8} autre(s) version(s)")
        else:
            st.info("Aucun modèle disponible.")

    # ── 5. Tests A/B et Shadow actifs (conditionnel) ──────────────────────────
    ab_versions = [m for m in all_models if m.get("deployment_mode") == "ab"]
    shadow_versions = [m for m in all_models if m.get("deployment_mode") == "shadow"]

    if ab_versions or shadow_versions:
        st.divider()
        c1, c2 = st.columns(2)
        if ab_versions:
            ab_names = ", ".join({m.get("name", "?") for m in ab_versions})
            c1.info(f"🧪 **{len(ab_versions)} version(s) en test A/B** — {ab_names}")
        if shadow_versions:
            sh_names = ", ".join({m.get("name", "?") for m in shadow_versions})
            c2.info(f"👻 **{len(shadow_versions)} version(s) en shadow** — {sh_names}")
        st.page_link("pages/6_AB_Testing.py", label="→ Gérer les tests A/B et shadow")

    # ── 6. Cartes de navigation ───────────────────────────────────────────────
    st.divider()
    st.subheader("🗺️ Naviguer dans le dashboard")

    nav_items = [
        (
            "🤖",
            "Modèles",
            "Uploader, versionner, déployer et comparer des modèles ML",
            "pages/2_Models.py",
        ),
        (
            "🔮",
            "Prédictions",
            "Historique, export CSV, prédiction batch et résultats observés",
            "pages/3_Predictions.py",
        ),
        (
            "📊",
            "Statistiques",
            "Leaderboard, évolution temporelle et comparaison de versions",
            "pages/4_Stats.py",
        ),
        (
            "🛡️",
            "Supervision",
            "Vue de santé globale, détection de drift et alertes en temps réel",
            "pages/7_Supervision.py",
        ),
        (
            "🧪",
            "A/B Testing",
            "Configurer les déploiements progressifs et comparer les variantes",
            "pages/6_AB_Testing.py",
        ),
        (
            "🔄",
            "Réentraînement",
            "Scheduler cron, retrain manuel, upload de scripts train.py",
            "pages/8_Retrain.py",
        ),
        (
            "🏆",
            "Golden Tests",
            "Suites de tests de non-régression avec assertions sur les prédictions",
            "pages/9_Golden_Tests.py",
        ),
        (
            "📚",
            "Aide",
            "Documentation, exemples de code et guide d'intégration API",
            "pages/10_Aide.py",
        ),
    ]
    if is_admin:
        nav_items.insert(
            0,
            (
                "👥",
                "Utilisateurs",
                "Gérer les comptes, quotas et demandes d'accès",
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
                    st.page_link(page_path, label="Ouvrir →")


# Router principal — navigation conditionnelle selon l'état de connexion
# Restauration de session après F5 : sid dans session_state ou dans l'URL
if not st.session_state.get("api_token"):
    _sid = st.session_state.get("_sid") or st.query_params.get("sid")
    if _sid:
        if not _restore_session(_sid):
            st.query_params.clear()

_logged_in = bool(st.session_state.get("api_token"))

# Ré-écrire le sid dans l'URL à chaque render (la navigation le supprime)
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
/* Spécificité élevée pour battre les classes générées st-emotion-cache-xxx */
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
/* Expanders — ciblage large car Streamlit génère des classes aléatoires */
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
/* Inputs dans les expanders */
details input, details textarea, details select {
    background-color: #262730 !important;
    color: #fafafa !important;
    border-color: #555 !important;
}
/* Tous les inputs numériques */
input[type="number"] {
    background-color: #262730 !important;
    color: #fafafa !important;
    border-color: #555 !important;
}
</style>
"""

if _logged_in:
    # Sidebar "Mon compte" — affiché sur toutes les pages
    _client = APIClient(
        base_url=st.session_state["api_url"],
        token=st.session_state["api_token"],
    )
    with st.sidebar:
        st.subheader("Mon compte")
        try:
            quota = _client.get_my_quota()
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
                me = _client.get_me()
                show_token_with_copy(me["api_token"])
            except Exception:
                st.error("Impossible de charger le token.")

        if st.button("Se déconnecter", type="secondary", width="stretch"):
            _clear_session()
            logout()

        if st.session_state.get("is_admin"):
            try:
                n_pending = _client.get_pending_account_requests_count()
                if n_pending > 0:
                    st.warning(f"🔔 {n_pending} demande(s) d'accès en attente")
                    st.page_link("pages/1_Users.py", label="Gérer les demandes →")
            except Exception:
                pass

        st.divider()
        if st.session_state.get("role") != "readonly":
            _grafana_url = os.environ.get("GRAFANA_PUBLIC_URL", "http://localhost:3000")
            _minio_url = os.environ.get("MINIO_CONSOLE_PUBLIC_URL", "http://localhost:9011")
            _mlflow_url = os.environ.get("MLFLOW_PUBLIC_URL", "http://localhost:5000")
            st.subheader("Services")
            st.markdown(
                f"🔗 [Swagger](http://localhost/docs)  \n"
                f"📊 [Grafana]({_grafana_url}/dashboards)  \n"
                f"🪣 [MinIO]({_minio_url}/login)  \n"
                f"🧪 [MLflow]({_mlflow_url}/)"
            )
            if st.session_state.get("is_admin"):
                st.page_link("pages/11_Services.py", label="🔑 Accès & credentials →")
            st.divider()

    _base_pages = [
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
    ]
    if st.session_state.get("is_admin"):
        _base_pages.append(st.Page("pages/11_Services.py", title="Services"))
    _pg = st.navigation(_base_pages)
else:
    _pg = st.navigation(
        [
            st.Page(show_login, title="Connexion", default=True),
            st.Page("pages/0_Demande_Acces.py", title="Demande d'accès"),
        ]
    )

# Dark mode toggle — visible sur toutes les pages
if st.sidebar.toggle("Mode sombre", key="dark_mode"):
    st.markdown(_DARK_CSS, unsafe_allow_html=True)

_pg.run()
