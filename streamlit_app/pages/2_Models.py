"""
Gestion des modèles ML
"""

import json as _json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from utils.api_client import get_model_detail as get_model_detail_cached
from utils.api_client import get_models as get_models_cached
from utils.auth import get_client, require_auth
from utils.metrics_help import METRIC_HELP

# --- Helpers historique ---

_ACTION_ICONS = {
    "created": "🟦",
    "updated": "🟨",
    "set_production": "🟩",
    "deprecated": "🟫",
    "deleted": "🟥",
    "rollback": "🔁",
}


def _action_badge(action: str) -> str:
    icon = _ACTION_ICONS.get(action, "⬜")
    return f"{icon} `{action}`"


# ── Scripts d'exemple ─────────────────────────────────────────────────────────

_SCRIPTS_DIR = Path(__file__).parent.parent / "documentation" / "Scripts"

# (chemin relatif depuis _SCRIPTS_DIR, description)
_EXAMPLE_SCRIPTS: list[tuple[str, str, str]] = [
    (
        "Iris",
        "iris/train_iris.py",
        "Script `train.py` RandomForest — à uploader pour activer le ré-entraînement automatique",
    ),
    (
        "Iris",
        "iris/upload_iris_model.py",
        "Upload complet RandomForest — entraîne localement et uploade via l'API",
    ),
    (
        "Wine",
        "wine/train_wine.py",
        "Script `train.py` GradientBoostingRegressor — régression sur la teneur en alcool",
    ),
    (
        "Wine",
        "wine/upload_wine_model.py",
        "Upload complet Wine Regressor — entraîne localement et uploade via l'API",
    ),
]


def _read_script(rel_path: str) -> str:
    try:
        return (_SCRIPTS_DIR / rel_path).read_text(encoding="utf-8")
    except Exception:
        return f"# Fichier introuvable : {rel_path}"


@st.dialog("Aperçu du script", width="large")
def _view_script_dialog(rel_path: str) -> None:
    content = _read_script(rel_path)
    basename = Path(rel_path).name
    st.code(content, language="python", line_numbers=True)
    st.download_button(
        "⬇️ Télécharger",
        data=content,
        file_name=basename,
        mime="text/x-python",
        key=f"dl_dialog_{rel_path}",
        width='stretch',
    )


st.set_page_config(page_title="Models — PredictML", page_icon="🤖", layout="wide")
require_auth()

col_title, col_refresh = st.columns([8, 1])
col_title.title("🤖 Gestion des modèles")
if col_refresh.button("🔄 Rafraîchir", key="models_refresh", width='stretch'):
    st.cache_data.clear()
    st.rerun()

client = get_client()
is_admin = st.session_state.get("is_admin", False)

MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://localhost:5000")
MLFLOW_PUBLIC_URL = os.environ.get("MLFLOW_PUBLIC_URL", "http://localhost:5000")


def fetch_models(api_url, token):
    return get_models_cached(api_url, token)


def fetch_model_detail(api_url, token, name, version):
    return get_model_detail_cached(api_url, token, name, version)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_feature_importance(api_url, token, name, version, last_n, days):
    c = get_client()
    return c.get_feature_importance(name, version=version, last_n=last_n, days=days)


@st.cache_data(ttl=30, show_spinner=False)
def fetch_model_performance(api_url, token, name, version, period_days=30):
    from datetime import datetime, timedelta, timezone
    start = (datetime.now(timezone.utc) - timedelta(days=period_days)).strftime("%Y-%m-%dT%H:%M:%S")
    c = get_client()
    return c.get_model_performance(name, version=version, start=start)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_output_drift(api_url, token, name, version, period_days):
    import requests

    r = requests.get(
        f"{api_url}/models/{name}/output-drift",
        headers={"Authorization": f"Bearer {token}"},
        params={"model_version": version, "period_days": period_days},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_prediction_count(api_url, token, name, version, start: str, end: str) -> int:
    """Retourne le nombre total de prédictions sur la période (appel léger limit=1)."""
    import requests

    try:
        r = requests.get(
            f"{api_url}/predictions",
            headers={"Authorization": f"Bearer {token}"},
            params={"model_name": name, "model_version": version, "start": start, "end": end, "limit": 1},
            timeout=5,
        )
        r.raise_for_status()
        return r.json().get("total", 500)
    except Exception:
        return 500


@st.cache_data(ttl=60, show_spinner=False)
def fetch_feature_drift(api_url, token, name, start: str, end: str) -> dict:
    """Récupère le drift des features via /monitoring/model/{name}."""
    import requests

    r = requests.get(
        f"{api_url}/monitoring/model/{name}",
        headers={"Authorization": f"Bearer {token}"},
        params={"start": start, "end": end},
        timeout=15,
    )
    r.raise_for_status()
    return r.json().get("feature_drift", {})


@st.cache_data(ttl=10, show_spinner=False)
def fetch_cached_models(api_url, token):
    """Retourne la liste des clés 'name:version' actuellement en cache Redis."""
    try:
        import requests

        r = requests.get(
            f"{api_url}/models/cached",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
        if r.status_code == 200:
            return r.json().get("cached_models", [])
    except Exception:
        pass
    return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_model_pkl(api_url: str, token: str, name: str, version: str) -> bytes | None:
    """Télécharge le .joblib depuis MinIO, mis en cache 5 min par version."""
    from utils.api_client import APIClient

    try:
        return APIClient(base_url=api_url, token=token).download_model(name, version)
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_train_script(api_url: str, token: str, name: str, version: str) -> bytes | None:
    """Télécharge le script train.py depuis MinIO, mis en cache 5 min par version."""
    from utils.api_client import APIClient

    try:
        return APIClient(base_url=api_url, token=token).download_train_script(name, version)
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_training_dataset(api_url: str, token: str, name: str, version: str) -> bytes | None:
    """Télécharge le dataset CSV depuis MinIO, mis en cache 5 min par version."""
    from utils.api_client import APIClient

    try:
        return APIClient(base_url=api_url, token=token).download_training_dataset(name, version)
    except Exception:
        return None


def reload():
    st.cache_data.clear()
    st.rerun()


try:
    models = fetch_models(st.session_state.get("api_url"), st.session_state.get("api_token"))
except Exception as e:
    st.error(f"Impossible de charger les modèles : {e}")
    st.stop()

cached_model_keys: list = []
if is_admin:
    cached_model_keys = fetch_cached_models(
        st.session_state.get("api_url"), st.session_state.get("api_token")
    )

if is_admin:
    with st.expander("📋 Scripts d'exemple", expanded=False):
        st.caption(
            "Scripts de référence pour prendre en main l'upload et le ré-entraînement. "
            "Téléchargez-les ou visualisez-les directement ici."
        )
        _current_category = None
        for _category, _script_rel, _script_desc in _EXAMPLE_SCRIPTS:
            if _category != _current_category:
                st.markdown(f"**{_category}**")
                _current_category = _category
            _col_desc, _col_view, _col_dl = st.columns([5, 1.5, 1.5])
            _script_basename = Path(_script_rel).name
            _col_desc.markdown(f"**`{_script_basename}`**  \n{_script_desc}")
            if _col_view.button("👁 Visualiser", key=f"view_{_script_rel}", width='stretch'):
                _view_script_dialog(_script_rel)
            _col_dl.download_button(
                "⬇️ Télécharger",
                data=_read_script(_script_rel),
                file_name=_script_basename,
                mime="text/x-python",
                key=f"dl_{_script_rel}",
                width='stretch',
            )

if is_admin:
    with st.expander("➕ Uploader un nouveau modèle", expanded=not models):
        with st.form("upload_model_form", clear_on_submit=True):
            st.markdown("##### Fichiers")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                pkl_file = st.file_uploader(
                    "Fichier modèle (.joblib) *",
                    type=["joblib", "pkl"],
                    help="Fichier modèle sérialisé (joblib recommandé). Aucune limite de taille imposée.",
                )
            with col_f2:
                train_file = st.file_uploader(
                    "Script d'entraînement (train.py) — optionnel",
                    type=["py"],
                    help="Doit référencer TRAIN_START_DATE, TRAIN_END_DATE, OUTPUT_MODEL_PATH.",
                )

            st.markdown("##### Identité")
            col_n, col_v = st.columns(2)
            with col_n:
                up_name = st.text_input("Nom du modèle *", placeholder="ex : iris-classifier")
            with col_v:
                up_version = st.text_input("Version *", placeholder="ex : 1.0.0")

            up_description = st.text_area(
                "Description", placeholder="Description courte du modèle…", height=80
            )

            _ALGO_OPTIONS = [
                "",
                "RandomForest",
                "GradientBoosting",
                "XGBoost",
                "LightGBM",
                "LogisticRegression",
                "SVM",
                "KNN",
                "DecisionTree",
                "NeuralNetwork",
                "LinearRegression",
                "Ridge",
                "Lasso",
                "ElasticNet",
                "Autre",
            ]
            up_algorithm = st.selectbox("Algorithme", _ALGO_OPTIONS)

            st.markdown("##### Métriques")
            col_acc, col_f1s = st.columns(2)
            with col_acc:
                up_accuracy = st.number_input(
                    "Accuracy", min_value=0.0, max_value=1.0, value=None, step=0.001, format="%.4f"
                )
            with col_f1s:
                up_f1 = st.number_input(
                    "F1 score", min_value=0.0, max_value=1.0, value=None, step=0.001, format="%.4f"
                )

            up_tags_raw = st.text_input(
                "Tags (séparés par des virgules)",
                placeholder="ex : production, finance, v2",
            )

            st.markdown("##### Baseline automatique")
            col_bl1, col_bl2 = st.columns([3, 1])
            with col_bl1:
                compute_baseline_auto = st.checkbox(
                    "Calculer la baseline depuis les données de production",
                    value=True,
                    help=(
                        "Après l'upload, calcule les statistiques de distribution "
                        "(mean, std, min, max) depuis les prédictions récentes. "
                        "Nécessite au moins 100 prédictions pour ce nom de modèle."
                    ),
                )
            with col_bl2:
                baseline_days = st.number_input(
                    "Fenêtre (jours)", min_value=1, max_value=180, value=30
                )

            submitted = st.form_submit_button("⬆️ Uploader le modèle", type="primary")

        if submitted:
            errors = []
            if not pkl_file:
                errors.append("Le fichier modèle (.joblib) est obligatoire.")
            if not up_name.strip():
                errors.append("Le nom du modèle est obligatoire.")
            if not up_version.strip():
                errors.append("La version est obligatoire.")

            if errors:
                for err in errors:
                    st.error(err)
            else:
                tags = (
                    [t.strip() for t in up_tags_raw.split(",") if t.strip()]
                    if up_tags_raw.strip()
                    else []
                )
                algorithm = up_algorithm if up_algorithm and up_algorithm != "Autre" else None

                with st.spinner(f"Upload de {pkl_file.name} en cours…"):
                    progress = st.progress(0, text="Envoi du fichier…")
                    try:
                        train_bytes = train_file.read() if train_file else None
                        train_fname = train_file.name if train_file else None
                        progress.progress(30, text="Envoi en cours…")
                        result = client.upload_model(
                            name=up_name.strip(),
                            version=up_version.strip(),
                            file_bytes=pkl_file.read(),
                            filename=pkl_file.name,
                            description=up_description.strip() or None,
                            algorithm=algorithm,
                            accuracy=up_accuracy,
                            f1_score=up_f1,
                            tags=tags or None,
                            train_file_bytes=train_bytes,
                            train_filename=train_fname,
                        )
                        progress.progress(100, text="Terminé.")
                        st.toast(
                            f"Modèle {result['name']} v{result['version']} uploadé.", icon="✅"
                        )
                        if compute_baseline_auto:
                            try:
                                bl = client.compute_baseline(
                                    up_name.strip(),
                                    up_version.strip(),
                                    days=int(baseline_days),
                                    dry_run=False,
                                )
                                st.info(
                                    f"Baseline calculée : {len(bl['baseline'])} features, "
                                    f"{bl['predictions_used']} prédictions utilisées."
                                )
                            except Exception as bl_exc:
                                _bl_detail = ""
                                try:
                                    if hasattr(bl_exc, "response") and bl_exc.response is not None:
                                        _bl_detail = bl_exc.response.json().get("detail", "")
                                except Exception:
                                    pass
                                st.warning(f"Baseline non calculée : {_bl_detail or bl_exc}")
                        reload()
                    except Exception as exc:
                        progress.empty()
                        detail = ""
                        try:
                            if hasattr(exc, "response") and exc.response is not None:
                                body = exc.response.json()
                                detail = body.get("detail") or str(body)
                        except Exception:
                            detail = str(exc)
                        st.error(f"Erreur lors de l'upload : {detail or exc}")

if not models:
    st.info("Aucun modèle disponible.")
    st.stop()

# Résumé
col1, col2, col3 = st.columns(3)
col1.metric("Total modèles", len(models))
col2.metric("En production", sum(1 for m in models if m.get("is_production")))
col3.metric("Avec MLflow", sum(1 for m in models if m.get("mlflow_run_id")))

st.divider()

# ── Barre de filtres (une seule ligne) ───────────────────────────────────────
_all_model_names = sorted({m["name"] for m in models})
_all_tags = sorted({t for m in models for t in (m.get("tags") or [])})
_filter_cols = st.columns([3, 2, 2]) if _all_tags else st.columns([3, 2])

with _filter_cols[0]:
    search_query = st.text_input(
        "Rechercher",
        placeholder="Nom ou description…",
        key="models_search",
        label_visibility="collapsed",
    )

with _filter_cols[1]:
    model_name_filter = st.selectbox(
        "Modèle",
        ["(tous les modèles)"] + _all_model_names,
        key="models_name_filter",
    )

if len(_filter_cols) == 3:
    with _filter_cols[2]:
        tag_filter = st.selectbox("Tag", ["(tous les tags)"] + _all_tags, key="tag_filter")
else:
    tag_filter = "(tous les tags)"

# Appliquer les filtres
if search_query:
    q = search_query.lower()
    models = [
        m for m in models
        if q in (m.get("name") or "").lower() or q in (m.get("description") or "").lower()
    ]
if model_name_filter != "(tous les modèles)":
    models = [m for m in models if m.get("name") == model_name_filter]
if tag_filter != "(tous les tags)":
    models = [m for m in models if tag_filter in (m.get("tags") or [])]

_DEPLOY_BADGE = {
    "ab_test": "🟠 A/B",
    "shadow": "🟣 Shadow",
    "production": "🟢 Prod",
}


def _statut(m: dict) -> str:
    mode = m.get("deployment_mode")
    weight = m.get("traffic_weight")
    if mode == "ab_test":
        return f"🟠 A/B ({weight:.0%})" if weight is not None else "🟠 A/B"
    if mode == "shadow":
        return "🟣 Shadow"
    if m.get("is_production"):
        return "🟢 Production"
    if m.get("is_active"):
        return "✅ Actif"
    return "⚫ Inactif"


# Tableau de synthèse
rows = []
for m in models:
    statut = _statut(m)

    in_cache = f"{m.get('name')}:{m.get('version')}" in cached_model_keys
    rows.append(
        {
            "Nom": m.get("name", ""),
            "Version": m.get("version", ""),
            "Tags": ", ".join(m.get("tags") or []) or "—",
            "Algorithme": m.get("algorithm") or "—",
            "Tâche": {
                "regression": "📈 Régression",
                "classification_binary": "🔵 Binaire",
                "classification_multiclass": "🟡 Multiclass",
            }.get(m.get("model_task") or "", "—"),
            "Baseline": "✅ Baseline" if m.get("feature_baseline") else "⚠️ No baseline",
            "Cache": "🔥 En cache" if in_cache else "❄️ Non chargé",
            "Statut": statut,
            "Créateur": m.get("creator_username") or "—",
            "Créé le": (
                pd.to_datetime(m.get("created_at")).strftime("%Y-%m-%d")
                if m.get("created_at")
                else "—"
            ),
            "Dernière préd.": (
                pd.to_datetime(m.get("last_seen")).strftime("%Y-%m-%d %H:%M")
                if m.get("last_seen")
                else "—"
            ),
            "Accuracy (eval)": f"{m['accuracy']:.3f}" if m.get("accuracy") is not None else "—",
            "F1 (eval)": f"{m['f1_score']:.3f}" if m.get("f1_score") is not None else "—",
            "R² (train)": (
                f"{(m.get('training_metrics') or {}).get('r2'):.3f}"
                if (m.get("training_metrics") or {}).get("r2") is not None
                else "—"
            ),
            "RMSE (train)": (
                f"{(m.get('training_metrics') or {}).get('rmse'):.4f}"
                if (m.get("training_metrics") or {}).get("rmse") is not None
                else "—"
            ),
        }
    )

st.dataframe(
    pd.DataFrame(rows),
    width='stretch',
    hide_index=True,
    column_config={
        "Nom": st.column_config.TextColumn(
            "Nom",
            help="Identifiant unique du modèle.",
        ),
        "Version": st.column_config.TextColumn(
            "Version",
            help="Version du modèle au format X.Y.Z.",
        ),
        "Tags": st.column_config.TextColumn(
            "Tags",
            help="Étiquettes libres associées au modèle (ex : Example, production, v2…).",
        ),
        "Algorithme": st.column_config.TextColumn(
            "Algorithme",
            help="Classe scikit-learn utilisée lors de l'entraînement (ex : RandomForest, GradientBoosting…).",
        ),
        "Tâche": st.column_config.TextColumn(
            "Tâche",
            help="Type de tâche ML : 📈 Régression, 🔵 Classification binaire ou 🟡 Classification multiclasse.",
        ),
        "Baseline": st.column_config.TextColumn(
            "Baseline",
            help=(
                "✅ Baseline : statistiques de référence (moyenne, std, min/max) calculées à l'entraînement. "
                "Utilisées pour la détection de drift et la validation du schéma d'entrée. "
                "⚠️ Absent : upload sans feature_baseline."
            ),
        ),
        "Cache": st.column_config.TextColumn(
            "Cache",
            help=(
                "🔥 En cache : modèle chargé en mémoire — prêt à servir sans latence de cold start. "
                "❄️ Non chargé : sera chargé depuis MinIO à la première prédiction."
            ),
        ),
        "Statut": st.column_config.TextColumn(
            "Statut",
            help=(
                "🟢 Production : version principale. "
                "🟠 A/B (x%) : reçoit x% du trafic en test A/B. "
                "🟣 Shadow : reçoit les requêtes sans retourner sa prédiction au client. "
                "✅ Actif : disponible mais non en production. "
                "⚫ Inactif : désactivé."
            ),
        ),
        "Créateur": st.column_config.TextColumn(
            "Créateur",
            help="Utilisateur ayant uploadé ce modèle.",
        ),
        "Créé le": st.column_config.TextColumn(
            "Créé le",
            help="Date d'upload du modèle (heure locale).",
        ),
        "Dernière préd.": st.column_config.TextColumn(
            "Dernière préd.",
            help="Date et heure de la dernière prédiction servie par ce modèle.",
        ),
        "Accuracy (eval)": st.column_config.TextColumn(
            "Accuracy (eval)",
            help=(
                "Accuracy = (vrais positifs + vrais négatifs) / total des observations.\n\n"
                "Mesure la proportion de prédictions correctes sur le jeu de test (hold-out) "
                "séparé avant l'entraînement.\n\n"
                "• 1.0 → 100 % de prédictions correctes\n"
                "• 0.5 → équivalent à un tirage aléatoire sur 2 classes\n\n"
                "⚠️ Trompeuse sur des classes très déséquilibrées : un modèle qui "
                "prédit toujours la classe majoritaire peut afficher 0.95 sans rien apprendre. "
                "Dans ce cas, préférer le F1. Métrique classification uniquement."
            ),
        ),
        "F1 (eval)": st.column_config.TextColumn(
            "F1 (eval)",
            help=(
                "F1 = 2 × (Précision × Rappel) / (Précision + Rappel)\n\n"
                "• Précision = parmi les exemples prédits positifs, combien le sont vraiment.\n"
                "• Rappel = parmi les exemples réellement positifs, combien sont détectés.\n\n"
                "Le F1 est la moyenne harmonique des deux : il pénalise fortement "
                "un score très faible sur l'un ou l'autre.\n\n"
                "• 1.0 → précision et rappel parfaits\n"
                "• 0.0 → modèle inutilisable\n\n"
                "Recommandé quand les classes sont déséquilibrées ou quand les faux négatifs "
                "et faux positifs ont un coût différent (ex : détection de fraude, diagnostic médical). "
                "Métrique classification uniquement."
            ),
        ),
        "R² (train)": st.column_config.TextColumn(
            "R² (train)",
            help=(
                "R² = 1 − (Σ(y − ŷ)²) / (Σ(y − ȳ)²)\n\n"
                "Mesure la part de variance de la variable cible expliquée par le modèle, "
                "calculée sur le jeu d'entraînement.\n\n"
                "• 1.0 → le modèle prédit parfaitement toutes les valeurs\n"
                "• 0.0 → le modèle n'explique rien (équivalent à prédire la moyenne)\n"
                "• < 0 → le modèle est moins bon que de prédire la moyenne — signe "
                "d'un problème (sur-régularisation, mauvais features, bug)\n\n"
                "⚠️ Un R² élevé sur le train peut masquer du surapprentissage. "
                "Comparer avec les métriques live en production. Métrique régression uniquement."
            ),
        ),
        "RMSE (train)": st.column_config.TextColumn(
            "RMSE (train)",
            help=(
                "RMSE = √( Σ(y − ŷ)² / n )\n\n"
                "Erreur quadratique moyenne, exprimée dans la même unité que la variable cible. "
                "Calculée sur le jeu d'entraînement.\n\n"
                "• Un RMSE de 2.5 sur un prix en € signifie que le modèle se trompe "
                "en moyenne de ±2.5 €.\n"
                "• Plus la valeur est faible, meilleur est le modèle.\n"
                "• Pas de borne supérieure : dépend de l'échelle de la cible.\n\n"
                "Le RMSE pénalise davantage les grandes erreurs que la MAE "
                "(Mean Absolute Error), car les écarts sont mis au carré avant la racine. "
                "Métrique régression uniquement."
            ),
        ),
    },
)
st.caption("Accuracy (eval), F1 (eval), R² (eval), RMSE (eval) : métriques issues de l'évaluation sur le jeu de test lors de l'entraînement.")

# ---------------------------------------------------------------------------
# Comparaison multi-versions
# ---------------------------------------------------------------------------

with st.expander("📊 Comparaison multi-versions", expanded=False):
    model_names = sorted({m["name"] for m in models})
    compare_search = st.text_input(
        "Filtrer par nom", key="compare_search", placeholder="Rechercher un modèle…"
    )
    compare_filtered = (
        [n for n in model_names if compare_search.lower() in n.lower()]
        if compare_search
        else model_names
    )
    compare_name = st.selectbox(
        "Modèle à comparer", compare_filtered or model_names, key="compare_model_name"
    )

    versions_for_model = [m["version"] for m in models if m["name"] == compare_name]
    all_versions_label = f"Toutes ({len(versions_for_model)})"
    version_options = [all_versions_label] + versions_for_model
    selected_versions = st.multiselect(
        "Versions à inclure (vide = toutes)",
        versions_for_model,
        default=[],
        key="compare_versions_select",
    )
    _cmp_col_start, _cmp_col_end = st.columns(2)
    _cmp_default_end = pd.Timestamp.now().date()
    _cmp_default_start = _cmp_default_end - pd.Timedelta(days=6)
    compare_date_start = _cmp_col_start.date_input("Date début", value=_cmp_default_start, key="compare_date_start")
    compare_date_end = _cmp_col_end.date_input("Date fin", value=_cmp_default_end, key="compare_date_end")
    compare_days = max(1, (compare_date_end - compare_date_start).days + 1)

    if st.button("🔍 Comparer", key="compare_btn", type="primary"):
        if compare_date_start > compare_date_end:
            st.warning("La date de début doit être antérieure à la date de fin.")
        else:
            versions_param = ",".join(selected_versions) if selected_versions else None
            with st.spinner("Comparaison en cours…"):
                try:
                    cmp = client.compare_model_versions(
                        compare_name,
                        versions=versions_param,
                        days=compare_days,
                        start_date=compare_date_start.isoformat(),
                        end_date=compare_date_end.isoformat(),
                    )
                    cmp_versions = cmp.get("versions", [])
                    if not cmp_versions:
                        st.info("Aucune version active trouvée.")
                    else:
                        _DRIFT_BADGE = {
                            "ok": "🟢 ok",
                            "warning": "🟡 warning",
                            "critical": "🔴 critical",
                            "no_baseline": "⚫ no baseline",
                            "insufficient_data": "⬜ insuff. data",
                        }
                        _full_meta = {
                            m["version"]: m
                            for m in models
                            if m.get("name") == compare_name
                        }
                        cmp_rows = []
                        # Régression si training_metrics contient mae/rmse/r2
                        is_regression = any(v.get("mae_eval") is not None for v in cmp_versions)

                        def _r(val, n=3):
                            return round(val, n) if val is not None else None

                        # Règle : afficher une paire (eval, live) seulement si l'eval est présent
                        show = {}
                        if is_regression:
                            show["mae"]  = any(v.get("mae_eval")  is not None for v in cmp_versions)
                            show["rmse"] = any(v.get("rmse_eval") is not None for v in cmp_versions)
                            show["r2"]   = any(v.get("r2_eval")   is not None for v in cmp_versions)
                        else:
                            show["accuracy"] = any(v.get("accuracy")  is not None for v in cmp_versions)
                            show["f1"]       = any(v.get("f1_score")  is not None for v in cmp_versions)
                            show["brier"]    = any(v.get("brier_score") is not None for v in cmp_versions)

                        for v in cmp_versions:
                            full = _full_meta.get(v["version"], v)
                            row = {
                                "Version": v["version"],
                                "Statut": _statut(full),
                                "Tâche": {
                                    "regression": "📈 Régression",
                                    "classification_binary": "🔵 Binaire",
                                    "classification_multiclass": "🟡 Multiclass",
                                }.get(v.get("model_task") or "", "—"),
                                "Nb préd.": v.get("prediction_count") or 0,
                                "Nb préd. shadow": v.get("shadow_prediction_count") or 0,
                                "Latence p50 (ms)": _r(v.get("latency_p50_ms"), 1),
                                "Latence p95 (ms)": _r(v.get("latency_p95_ms"), 1),
                                "Entraîné le": (
                                    pd.to_datetime(v["trained_at"]).strftime("%Y-%m-%d")
                                    if v.get("trained_at") else "—"
                                ),
                                "Drift": _DRIFT_BADGE.get(v.get("drift_status") or "", "—"),
                            }
                            if is_regression:
                                if show["mae"]:
                                    row["MAE (eval)"]  = _r(v.get("mae_eval"), 4)
                                    row["MAE (live)"]  = _r(v.get("live_mae"), 4)
                                if show["rmse"]:
                                    row["RMSE (eval)"] = _r(v.get("rmse_eval"), 4)
                                    row["RMSE (live)"] = _r(v.get("live_rmse"), 4)
                                if show["r2"]:
                                    row["R² (eval)"]   = _r(v.get("r2_eval"), 3)
                                    row["R² (live)"]   = _r(v.get("live_r2"), 3)
                            else:
                                if show["accuracy"]:
                                    row["Accuracy (eval)"] = _r(v.get("accuracy"))
                                    row["Accuracy (live)"] = _r(v.get("live_accuracy"))
                                if show["f1"]:
                                    row["F1 (eval)"]       = _r(v.get("f1_score"))
                                    row["F1 (live)"]       = _r(v.get("live_f1"))
                                if show["brier"]:
                                    row["Brier score"]     = _r(v.get("brier_score"), 4)
                            cmp_rows.append(row)

                        col_config = {
                            "Nb préd.": st.column_config.NumberColumn(
                                "Nb préd.",
                                help=f"Prédictions production (is_shadow=False) sur la période {compare_date_start} → {compare_date_end}.",
                            ),
                            "Nb préd. shadow": st.column_config.NumberColumn(
                                "Nb préd. shadow",
                                help=f"Prédictions shadow (is_shadow=True) sur la période {compare_date_start} → {compare_date_end}.",
                            ),
                            "Latence p50 (ms)": st.column_config.NumberColumn(format="%.1f"),
                            "Latence p95 (ms)": st.column_config.NumberColumn(format="%.1f"),
                        }
                        if is_regression:
                            for col, fmt in [("MAE", "%.4f"), ("RMSE", "%.4f"), ("R²", "%.3f")]:
                                col_config[f"{col} (eval)"] = st.column_config.NumberColumn(f"{col} (eval)", help=f"{col} sur le jeu de test d'entraînement.", format=fmt)
                                col_config[f"{col} (live)"] = st.column_config.NumberColumn(f"{col} (live)", help=f"{col} calculé sur les prédictions réelles avec ground truth.", format=fmt)
                        else:
                            for col, fmt in [("Accuracy", "%.3f"), ("F1", "%.3f")]:
                                col_config[f"{col} (eval)"] = st.column_config.NumberColumn(f"{col} (eval)", help=f"{col} sur le jeu de test d'entraînement.", format=fmt)
                                col_config[f"{col} (live)"] = st.column_config.NumberColumn(f"{col} (live)", help=f"{col} sur les prédictions réelles avec ground truth.", format=fmt)
                            if show.get("brier"):
                                col_config["Brier score"] = st.column_config.NumberColumn(
                                    "Brier score (?)",
                                    help=(
                                        "Erreur quadratique entre probabilité prédite et résultat réel (0 = parfait, 1 = pire).\n\n"
                                        "• < 0.10 → très bon\n• 0.10–0.25 → correct\n• > 0.25 → à améliorer\n\n"
                                        "Classifieurs avec probabilités, ≥ 30 paires ground truth."
                                    ),
                                    format="%.4f",
                                )

                        st.dataframe(
                            pd.DataFrame(cmp_rows),
                            width='stretch',
                            hide_index=True,
                            column_config=col_config,
                        )
                        st.caption(
                            f"Comparé le {pd.to_datetime(cmp['compared_at']).strftime('%Y-%m-%d %H:%M')} UTC"
                            f" — période : {compare_date_start} → {compare_date_end}"
                        )
                except Exception as e:
                    st.error(f"Erreur lors de la comparaison : {e}")

st.divider()
st.subheader("Détail et actions")

model_options = {f"{m['name']} v{m['version']}": m for m in models}
detail_search = st.text_input(
    "Filtrer par nom", key="detail_search", placeholder="Rechercher un modèle…"
)
filtered_keys = (
    [k for k in model_options if detail_search.lower() in k.lower()]
    if detail_search
    else list(model_options.keys())
)

_preselect = st.session_state.pop("_nav_model", None) or st.query_params.get("model")
_detail_keys = filtered_keys or list(model_options.keys())
_detail_idx = 0
if _preselect:
    _hits = [i for i, k in enumerate(_detail_keys) if k.split(" v")[0] == _preselect]
    if _hits:
        _detail_idx = _hits[0]
selected_label = st.selectbox("Sélectionner un modèle", _detail_keys, index=_detail_idx)
selected = model_options[selected_label]

# Détails
_detail_tm = selected.get("training_metrics") or {}
_detail_is_regression = any(k in _detail_tm for k in ("mae", "rmse", "r2"))

with st.expander("📋 Détails complets", expanded=True):
    col_l, col_r = st.columns(2)
    # Pré-chargement des ressources téléchargeables (mis en cache 5 min par version)
    _api_url = client.base_url
    _token = client.token
    _ds = selected.get("training_dataset") or ""
    _csv_bytes = None
    if _ds.endswith(".csv") and "/" in _ds:
        _csv_bytes = fetch_training_dataset(_api_url, _token, selected["name"], selected["version"])

    _script_bytes = None
    _script_filename = None
    if selected.get("train_script_object_key"):
        _script_bytes = fetch_train_script(_api_url, _token, selected["name"], selected["version"])
        _script_filename = selected["train_script_object_key"].split("/")[-1]

    _pkl_bytes = None
    _pkl_size_label = ""
    minio_key = selected.get("minio_object_key")
    size = selected.get("file_size_bytes")
    if minio_key and is_admin:
        _pkl_size_label = f" ({size / 1024:.1f} KB)" if size else ""
        _pkl_bytes = fetch_model_pkl(_api_url, _token, selected["name"], selected["version"])

    with col_l:
        st.markdown(f"**Nom :** `{selected.get('name')}`")
        st.markdown(f"**Version :** `{selected.get('version')}`")
        st.markdown(f"**Description :** {selected.get('description') or '—'}")
        st.markdown(f"**Algorithme :** {selected.get('algorithm') or '—'}")
        if _ds and not _csv_bytes:
            st.markdown(f"**Dataset d'entraînement :** `{_ds}`")
        st.markdown(f"**Entraîné par :** {selected.get('trained_by') or '—'}")
        parent_v = selected.get("parent_version")
        if parent_v:
            st.markdown(f"**Dérivé de :** `v{parent_v}`")
        tags = selected.get("tags")
        if tags:
            _tag_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            _tag_html = " ".join(
                f'<span style="background:{_tag_colors[i % len(_tag_colors)]};color:white;'
                f'padding:2px 10px;border-radius:12px;font-size:0.82em;font-weight:600;'
                f'white-space:nowrap;">{t}</span>'
                for i, t in enumerate(tags)
            )
            st.markdown(f"**Tags :** {_tag_html}", unsafe_allow_html=True)
        else:
            st.markdown("**Tags :** —")
        webhook = selected.get("webhook_url")
        st.markdown(f"**Webhook URL :** `{webhook}`" if webhook else "**Webhook URL :** —")

        mlflow_id = selected.get("mlflow_run_id")
        if mlflow_id:
            mlflow_link = f"{MLFLOW_PUBLIC_URL}/#/experiments/0/runs/{mlflow_id}"
            st.markdown(f"**MLflow run :** [{mlflow_id}]({mlflow_link})")
        else:
            st.markdown("**MLflow run :** —")

        if minio_key:
            st.markdown(f"**MinIO object :** `{selected.get('minio_bucket')}/{minio_key}`")
            if size:
                st.markdown(f"**Taille fichier :** {size / 1024:.1f} KB")

    with col_r:
        # ── Statut de déploiement ─────────────────────────────────────────────
        _d_mode   = selected.get("deployment_mode")
        _d_prod   = selected.get("is_production")
        _d_active = selected.get("is_active")
        _d_weight = selected.get("traffic_weight")

        if _d_mode == "ab_test":
            _badge_color = "#e67e00"
            _badge_label = f"🟠 A/B test actif — trafic {_d_weight:.0%}" if _d_weight is not None else "🟠 A/B test actif"
        elif _d_mode == "shadow":
            _badge_color = "#7c3aed"
            _badge_label = "🟣 Shadow actif"
        elif _d_prod:
            _badge_color = "#1a7f37"
            _badge_label = "🟢 En production"
        elif _d_active:
            _badge_color = "#0ea5e9"
            _badge_label = "✅ Actif (non production)"
        else:
            _badge_color = "#6b7280"
            _badge_label = "⚫ Inactif"

        st.markdown(
            f'<div style="display:inline-block;background:{_badge_color};color:white;'
            f'padding:4px 14px;border-radius:16px;font-size:0.9em;font-weight:600;'
            f'margin-bottom:8px">{_badge_label}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(f"**Nb features :** {selected.get('features_count') or '—'}")
        last_seen = selected.get("last_seen")
        st.markdown(
            f"**Dernière prédiction :** {pd.to_datetime(last_seen).strftime('%Y-%m-%d %H:%M') if last_seen else '—'}"
        )
        classes = selected.get("classes")
        if not _detail_is_regression or classes:
            st.markdown(f"**Classes :** {classes if classes else '—'}")
        ct = selected.get("confidence_threshold")
        if not _detail_is_regression or ct is not None:
            st.markdown(f"**Confidence threshold :** {f'{ct:.2f}' if ct is not None else '—'}")

        hp = selected.get("hyperparameters")
        if hp:
            st.markdown("**Hyperparamètres :**")
            st.dataframe(
                pd.DataFrame(
                    [{"Paramètre": k, "Valeur": str(v)} for k, v in hp.items()],
                ),
                width='stretch',
                hide_index=True,
            )
        else:
            st.markdown("**Hyperparamètres :** —")

    # ── 4 boutons alignés en bas de l'expander ────────────────────────────────
    _btn_labels = []
    if _csv_bytes:
        _btn_labels.append("dataset")
    if _script_bytes:
        _btn_labels.append("script_dl")
        _btn_labels.append("script_view")
    if _pkl_bytes:
        _btn_labels.append("pkl")

    if _btn_labels:
        _n_btns = len(_btn_labels)
        _btn_cols = st.columns(_n_btns)
        _btn_idx = 0
        if _csv_bytes:
            with _btn_cols[_btn_idx]:
                st.download_button(
                    "⬇ Dataset d'entraînement",
                    data=_csv_bytes,
                    file_name=_ds.split("/")[-1],
                    mime="text/csv",
                    width='stretch',
                    key=f"dl_dataset_{selected['name']}_{selected['version']}",
                )
            _btn_idx += 1
        if _script_bytes:
            with _btn_cols[_btn_idx]:
                st.download_button(
                    "⬇ Script d'entraînement",
                    data=_script_bytes,
                    file_name=_script_filename or "train.py",
                    mime="text/x-python",
                    width='stretch',
                    key=f"dl_script_{selected['name']}_{selected['version']}",
                )
            _btn_idx += 1
            with _btn_cols[_btn_idx]:
                _show_key = f"show_train_script_{selected['name']}_{selected['version']}"
                _is_visible = st.session_state.get(_show_key, False)
                if st.button(
                    "🙈 Masquer le script" if _is_visible else "👁 Visualiser le script",
                    width='stretch',
                    key=f"toggle_script_{selected['name']}_{selected['version']}",
                ):
                    st.session_state[_show_key] = not _is_visible
                    st.rerun()
            _btn_idx += 1
        if _pkl_bytes:
            with _btn_cols[_btn_idx]:
                st.download_button(
                    f"⬇️ Télécharger le modèle{_pkl_size_label}",
                    data=_pkl_bytes,
                    file_name=f"{selected['name']}_{selected['version']}.joblib",
                    mime="application/octet-stream",
                    width='stretch',
                    key=f"dl_pkl_{selected['name']}_{selected['version']}",
                )

    _show_script_key = f"show_train_script_{selected['name']}_{selected['version']}"
    if st.session_state.get(_show_script_key) and _script_bytes:
        st.code(_script_bytes.decode("utf-8", errors="replace"), language="python", line_numbers=True)

# ── Métriques en 2 blocs côte à côte ─────────────────────────────────────────
_tm = selected.get("training_metrics") or {}
_is_regression = any(k in _tm for k in ("mae", "rmse", "r2"))

# ── Analyse & Monitoring (métriques, SHAP, performance, drift) ───────────────
with st.expander("📊 Analyse & Monitoring", expanded=False):
    from datetime import date as _date, timedelta as _td

    _today = _date.today()
    _dcol1, _dcol2 = st.columns(2)
    _date_debut = _dcol1.date_input(
        "Date début",
        value=_today - _td(days=30),
        max_value=_today,
        key=f"ana_date_debut_{selected['name']}_{selected['version']}",
    )
    _date_fin = _dcol2.date_input(
        "Date fin",
        value=_today,
        max_value=_today,
        key=f"ana_date_fin_{selected['name']}_{selected['version']}",
    )
    _ana_days = max(1, (_date_fin - _date_debut).days)

    # ── Gate lazy-load ────────────────────────────────────────────────────────
    _ana_key = f"ana_loaded_{selected['name']}_{selected['version']}"
    _ana_loaded = st.session_state.get(_ana_key, False)
    if not _ana_loaded:
        _gcol1, _gcol2 = st.columns([1, 3])
        if _gcol1.button(
            "📊 Charger l'analyse",
            key=f"btn_ana_{selected['name']}_{selected['version']}",
            type="primary",
            width='stretch',
        ):
            st.session_state[_ana_key] = True
            st.rerun()
        _gcol2.caption(
            "Charge les métriques SHAP, la performance, le drift de sortie et le drift des features."
        )

    # ── Métriques ─────────────────────────────────────────────────────────────
    if _ana_loaded:
        st.divider()
        st.markdown("#### 📈 Métriques")

        # Récupération ground truth
        _perf_gt = None
        _gt_n = 0
        try:
            _perf_gt = fetch_model_performance(
                st.session_state.get("api_url"),
                st.session_state.get("api_token"),
                selected["name"],
                selected["version"],
                period_days=_ana_days,
            )
            _gt_n = _perf_gt.get("matched_predictions", 0) if _perf_gt else 0
        except Exception:
            pass

        # Construction du tableau comparatif entraînement vs production
        if _is_regression:
            _metrics_keys = [
                ("MAE",  "mae",  "mae",             None),
                ("RMSE", "rmse", "rmse",             None),
                ("R²",   "r2",   "r2",               None),
            ]
        else:
            _metrics_keys = [
                ("Accuracy",  "accuracy",  "accuracy",           selected.get("accuracy")),
                ("F1 Score",  "f1_score",  "f1_weighted",        selected.get("f1_score")),
                ("Precision", "precision", "precision_weighted",  None),
                ("Recall",    "recall",    "recall_weighted",     None),
            ]

        _table_rows = []
        for _label, _train_key, _gt_key, _train_fallback in _metrics_keys:
            _train_val = _tm.get(_train_key) or _train_fallback
            _gt_val = None
            if _perf_gt and _gt_n > 0:
                _gt_val = _perf_gt.get(_gt_key)
            _train_str = f"{round(float(_train_val), 4)}" if _train_val is not None else "—"
            _gt_str    = f"{round(float(_gt_val), 4)}"    if _gt_val    is not None else "—"
            _delta_str = "—"
            if _train_val is not None and _gt_val is not None:
                _d = round(float(_gt_val) - float(_train_val), 4)
                _delta_str = f"{_d:+.4f}"
            _table_rows.append({"Métrique": _label, "Entraînement": _train_str, "Production": _gt_str, "Δ": _delta_str})

        st.dataframe(
            pd.DataFrame(_table_rows),
            width='stretch',
            hide_index=True,
            column_config={
                "Métrique": st.column_config.TextColumn("Métrique"),
                "Entraînement": st.column_config.TextColumn(
                    "Entraînement",
                    help="Métriques évaluées sur le jeu de test lors de l'entraînement initial.",
                ),
                "Production": st.column_config.TextColumn(
                    "Production",
                    help=(
                        f"Métriques calculées sur les prédictions appariées à des résultats observés "
                        f"du {_date_debut} au {_date_fin}.\n\n"
                        f"{_gt_n} prédiction(s) appariée(s). "
                        "Alimenté via `POST /observed-results`."
                    ),
                ),
                "Δ": st.column_config.TextColumn(
                    "Δ",
                    help=(
                        "Écart Production − Entraînement.\n\n"
                        "- **Négatif** : dégradation en production\n"
                        "- **Positif** : amélioration (ou distribution différente)\n"
                        "- **—** : données insuffisantes"
                    ),
                ),
            },
        )
        if _gt_n == 0:
            st.caption("Aucune donnée de ground truth — envoyez des résultats via `POST /observed-results` pour voir la colonne Production.")

        try:
            md_content = client.get_model_card(selected["name"], selected["version"], format="markdown")
            st.download_button(
                label="📄 Exporter la model card",
                data=md_content,
                file_name=f"{selected['name']}_{selected['version']}_model_card.md",
                mime="text/markdown",
                key="dl_model_card",
            )
        except Exception as e:
            st.warning(f"Model card indisponible : {e}")

    # ── Importance des features (SHAP agrégé) ─────────────────────────────────
    if _ana_loaded:
        st.divider()
        st.markdown("#### 📊 Importance des features (SHAP)")
        _shap_start = _date_debut.strftime("%Y-%m-%dT00:00:00")
        _shap_end = _date_fin.strftime("%Y-%m-%dT23:59:59")
        _shap_total = fetch_prediction_count(
            st.session_state.get("api_url"),
            st.session_state.get("api_token"),
            selected["name"],
            selected["version"],
            _shap_start,
            _shap_end,
        )
        _shap_max = max(10, _shap_total)
        _shap_default = min(100, _shap_max)
        fi_last_n = st.slider(
            f"Échantillon SHAP ({_shap_total} prédictions disponibles sur la période)",
            min_value=10,
            max_value=_shap_max,
            value=_shap_default,
            step=max(1, _shap_max // 50),
            key=f"fi_last_n_slider_{selected['name']}_{selected['version']}",
            help=(
                "Nombre de prédictions récentes utilisées pour calculer l'importance SHAP.\n\n"
                "**SHAP** estime la contribution de chaque feature à chaque prédiction — "
                "c'est coûteux en calcul, d'où cette limite.\n\n"
                "- **Petit** : rapide, moins représentatif\n"
                "- **Moyen** : bon compromis vitesse / précision ✅\n"
                "- **Grand** : plus précis, plus lent\n\n"
                "Les prédictions sont sélectionnées par **ordre chronologique inverse** "
                "(les N plus récentes dans la période sélectionnée, `ORDER BY id DESC`)."
            ),
        )
        try:
            fi_data = fetch_feature_importance(
                st.session_state.get("api_url"),
                st.session_state.get("api_token"),
                selected["name"],
                selected["version"],
                last_n=fi_last_n,
                days=_ana_days,
            )
            fi = fi_data.get("feature_importance", {})
            sample_size = fi_data.get("sample_size", 0)
            if not fi or sample_size == 0:
                st.info("Pas encore assez de prédictions pour calculer l'importance des features.")
            else:
                fi_rows = [
                    {"Feature": feat, "Importance SHAP": vals["mean_abs_shap"]}
                    for feat, vals in sorted(fi.items(), key=lambda x: x[1]["rank"])
                ]
                fi_df = pd.DataFrame(fi_rows).head(15).sort_values("Importance SHAP")
                try:
                    import plotly.express as px

                    fig = px.bar(
                        fi_df,
                        x="Importance SHAP",
                        y="Feature",
                        orientation="h",
                        title=(
                            f"Top {len(fi_df)} features — "
                            f"{fi_data['model_name']} v{fi_data['version']}"
                        ),
                    )
                    fig.update_layout(
                        yaxis_title="",
                        xaxis_title="Importance SHAP (valeur absolue moyenne)",
                        margin={"l": 10, "r": 10, "t": 40, "b": 10},
                    )
                    st.plotly_chart(fig, width='stretch')
                except ImportError:
                    st.bar_chart(fi_df.set_index("Feature")["Importance SHAP"])
                st.caption(
                    f"{sample_size} prédictions analysées"
                    f" — du {_date_debut} au {_date_fin} — version : {fi_data.get('version')}"
                )
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 422:
                st.info(
                    "Ce modèle ne supporte pas l'importance des features SHAP "
                    "(entraîner avec un DataFrame pandas pour activer cette fonctionnalité)."
                )
            elif status == 404:
                st.error("Modèle introuvable.")
            else:
                st.warning(f"Impossible de calculer l'importance des features : {e}")

    # ── Métriques de performance (matrice de confusion + métriques par classe) ─
    if _ana_loaded:
        st.divider()
        st.markdown("#### 📈 Métriques de performance")
        try:
            import numpy as np
            import plotly.express as px

            perf = fetch_model_performance(
                st.session_state.get("api_url"),
                st.session_state.get("api_token"),
                selected["name"],
                selected["version"],
                period_days=_ana_days,
            )

            model_type = perf.get("model_type", "classification")
            matched = perf.get("matched_predictions", 0)

            if matched == 0:
                st.info(
                    "Aucune paire (prédiction, résultat observé) disponible. "
                    "Ajoutez des résultats via POST /observed-results pour activer cette vue."
                )
            elif model_type != "classification":
                col_mae, col_rmse, col_r2 = st.columns(3)
                col_mae.metric(
                    "MAE",
                    f"{perf['mae']:.4f}" if perf.get("mae") is not None else "—",
                    help=METRIC_HELP["mae"],
                )
                col_rmse.metric(
                    "RMSE",
                    f"{perf['rmse']:.4f}" if perf.get("rmse") is not None else "—",
                    help=METRIC_HELP["rmse"],
                )
                col_r2.metric(
                    "R²",
                    f"{perf['r2']:.4f}" if perf.get("r2") is not None else "—",
                    help=METRIC_HELP["r2"],
                )
                st.caption(f"{matched} prédictions appariées")
            else:
                col_acc, col_prec, col_rec, col_f1 = st.columns(4)
                col_acc.metric(
                    "Accuracy",
                    f"{perf['accuracy']:.3f}" if perf.get("accuracy") is not None else "—",
                    help=METRIC_HELP["accuracy"],
                )
                col_prec.metric(
                    "Precision (w.)",
                    (
                        f"{perf['precision_weighted']:.3f}"
                        if perf.get("precision_weighted") is not None
                        else "—"
                    ),
                    help=METRIC_HELP["precision"],
                )
                col_rec.metric(
                    "Recall (w.)",
                    (
                        f"{perf['recall_weighted']:.3f}"
                        if perf.get("recall_weighted") is not None
                        else "—"
                    ),
                    help=METRIC_HELP["recall"],
                )
                col_f1.metric(
                    "F1 (w.)",
                    f"{perf['f1_weighted']:.3f}" if perf.get("f1_weighted") is not None else "—",
                    help=METRIC_HELP["f1"],
                )
                st.caption(f"{matched} prédictions appariées")

                # Mapping index → label depuis selected.classes
                _named_classes = selected.get("classes") or []
                def _cls_label(idx) -> str:
                    try:
                        i = int(str(idx))
                        if _named_classes and 0 <= i < len(_named_classes):
                            return f"{_named_classes[i]} ({i})"
                    except (ValueError, TypeError):
                        pass
                    return str(idx)

                cm = perf.get("confusion_matrix")
                classes = perf.get("classes") or []
                if cm and classes:
                    cm_arr = np.array(cm)
                    class_labels = [_cls_label(c) for c in classes]
                    fig = px.imshow(
                        cm_arr,
                        x=class_labels,
                        y=class_labels,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        title="Matrice de confusion",
                        labels={"x": "Prédit", "y": "Réel", "color": "Compte"},
                    )
                    fig.update_layout(
                        xaxis_title="Prédit",
                        yaxis_title="Réel",
                        margin={"l": 10, "r": 10, "t": 40, "b": 10},
                    )
                    st.plotly_chart(fig, width='stretch')

                per_class = perf.get("per_class_metrics")
                if per_class:
                    st.markdown("**Métriques par classe**")
                    pc_rows = [
                        {
                            "Classe": _cls_label(label),
                            "Precision": f"{m['precision']:.3f}",
                            "Recall": f"{m['recall']:.3f}",
                            "F1": f"{m['f1_score']:.3f}",
                            "Support": m["support"],
                        }
                        for label, m in per_class.items()
                    ]
                    st.dataframe(
                        pd.DataFrame(pc_rows),
                        width='stretch',
                        hide_index=True,
                        column_config={
                            "Classe": st.column_config.TextColumn(
                                "Classe",
                                help="Classe prédite. Le chiffre entre parenthèses est l'indice numérique stocké en base.",
                            ),
                            "Precision": st.column_config.TextColumn(
                                "Precision",
                                help=(
                                    "**Precision** : parmi toutes les prédictions de cette classe, "
                                    "quelle fraction était correcte.\n\n"
                                    "`TP / (TP + FP)`\n\n"
                                    "Élevée = peu de faux positifs pour cette classe."
                                ),
                            ),
                            "Recall": st.column_config.TextColumn(
                                "Recall",
                                help=(
                                    "**Recall (sensibilité)** : parmi tous les vrais exemples de cette classe, "
                                    "quelle fraction a été correctement détectée.\n\n"
                                    "`TP / (TP + FN)`\n\n"
                                    "Élevé = peu de faux négatifs pour cette classe."
                                ),
                            ),
                            "F1": st.column_config.TextColumn(
                                "F1",
                                help=(
                                    "**F1-score** : moyenne harmonique de la Precision et du Recall.\n\n"
                                    "`2 × (Precision × Recall) / (Precision + Recall)`\n\n"
                                    "Bon compromis quand les classes sont déséquilibrées."
                                ),
                            ),
                            "Support": st.column_config.NumberColumn(
                                "Support",
                                help=(
                                    "Nombre de prédictions appariées à un résultat observé pour cette classe "
                                    "sur la période sélectionnée.\n\n"
                                    "Un support faible rend les métriques moins fiables statistiquement."
                                ),
                            ),
                        },
                    )

        except Exception as e:
            st.warning(f"Impossible de charger les métriques de performance : {e}")

    # ── Drift de sortie (label shift) ─────────────────────────────────────────
    if _ana_loaded:
        st.divider()
        st.markdown("#### 📊 Drift de sortie (label shift)")
        try:
            import plotly.express as px

            od = fetch_output_drift(
                st.session_state.get("api_url"),
                st.session_state.get("api_token"),
                selected["name"],
                selected["version"],
                _ana_days,
            )
            od_status = od.get("status", "no_baseline")
            _OD_BADGE = {
                "ok": "🟢 ok",
                "warning": "🟡 warning",
                "critical": "🔴 critical",
                "no_baseline": "⬜ pas de baseline",
                "insufficient_data": "⬜ données insuffisantes",
            }
            st.markdown(f"**Statut :** {_OD_BADGE.get(od_status, od_status)}")

            if od_status in ("no_baseline", "insufficient_data"):
                if od_status == "no_baseline":
                    st.info(
                        "Aucune distribution de labels d'entraînement disponible. "
                        "Assurez-vous que le script `train.py` imprime un JSON avec `label_distribution`."
                    )
                else:
                    st.info(
                        f"Données insuffisantes ({od.get('predictions_analyzed', 0)} prédictions "
                        f"du {_date_debut} au {_date_fin} — minimum : 30)."
                    )
            else:
                col_psi, col_n = st.columns(2)
                psi_val = od.get("psi")
                col_psi.metric(
                    "PSI",
                    f"{psi_val:.4f}" if psi_val is not None else "—",
                    help="Population Stability Index : ok < 0.1 | warning 0.1–0.2 | critical ≥ 0.2",
                )
                col_n.metric("Prédictions analysées", od.get("predictions_analyzed", 0))

                _od_named = selected.get("classes") or []
                def _od_label(idx) -> str:
                    try:
                        i = int(str(idx))
                        if _od_named and 0 <= i < len(_od_named):
                            return f"{_od_named[i]} ({i})"
                    except (ValueError, TypeError):
                        pass
                    return str(idx)

                by_class = od.get("by_class") or []
                if by_class:
                    st.markdown("**Distribution par classe**")
                    bc_rows = [
                        {
                            "Classe": _od_label(row["label"]),
                            "Baseline": f"{row['baseline_ratio']:.3f}",
                            "Actuel": f"{row['current_ratio']:.3f}",
                            "Δ": f"{row['delta']:+.3f}",
                        }
                        for row in by_class
                    ]
                    st.dataframe(
                        pd.DataFrame(bc_rows),
                        width='stretch',
                        hide_index=True,
                        column_config={
                            "Classe": st.column_config.TextColumn(
                                "Classe",
                                help="Classe prédite par le modèle. Le chiffre entre parenthèses est l'indice numérique stocké en base.",
                            ),
                            "Baseline": st.column_config.TextColumn(
                                "Baseline",
                                help=(
                                    "Proportion de cette classe dans le **dataset d'entraînement** "
                                    "(stockée dans `training_stats.label_distribution`).\n\n"
                                    "Source : script `train.py` → JSON stdout → clé `label_distribution`."
                                ),
                            ),
                            "Actuel": st.column_config.TextColumn(
                                "Actuel",
                                help=(
                                    "Proportion de cette classe dans les **prédictions de production** "
                                    "sur la période sélectionnée.\n\n"
                                    "Calculé via `GROUP BY prediction_result` sur la table `predictions`."
                                ),
                            ),
                            "Δ": st.column_config.TextColumn(
                                "Δ",
                                help=(
                                    "Écart entre la proportion actuelle et la baseline (Actuel − Baseline).\n\n"
                                    "- **Positif** : surreprésentation de cette classe en production\n"
                                    "- **Négatif** : sous-représentation\n\n"
                                    "Un Δ élevé contribue à un PSI critique."
                                ),
                            ),
                        },
                    )

                    baseline_dist = od.get("baseline_distribution") or {}
                    current_dist = od.get("current_distribution") or {}
                    raw_labels = [row["label"] for row in by_class]
                    named_labels = [_od_label(l) for l in raw_labels]
                    df_bar = pd.DataFrame(
                        {
                            "Classe": named_labels + named_labels,
                            "Ratio": [baseline_dist.get(str(l), 0.0) for l in raw_labels]
                            + [current_dist.get(str(l), 0.0) for l in raw_labels],
                            "Source": ["Baseline"] * len(raw_labels) + ["Actuel"] * len(raw_labels),
                        }
                    )
                    fig_od = px.bar(
                        df_bar,
                        x="Classe",
                        y="Ratio",
                        color="Source",
                        barmode="group",
                        title="Distribution baseline vs actuelle",
                        color_discrete_map={"Baseline": "#636EFA", "Actuel": "#EF553B"},
                    )
                    fig_od.update_layout(yaxis_tickformat=".0%", yaxis_title="Proportion")
                    st.plotly_chart(fig_od, width='stretch')
        except Exception as e:
            st.warning(f"Impossible de calculer le drift de sortie : {e}")

    # ── Drift des features (inputs) ───────────────────────────────────────────
    if _ana_loaded:
        st.divider()
        st.markdown("#### 🌡️ Drift des features (inputs)")
        try:
            import plotly.express as px

            _fd_start = _date_debut.strftime("%Y-%m-%d")
            _fd_end = _date_fin.strftime("%Y-%m-%d")
            fd = fetch_feature_drift(
                st.session_state.get("api_url"),
                st.session_state.get("api_token"),
                selected["name"],
                _fd_start,
                _fd_end,
            )

            _FD_BADGE = {
                "ok": "🟢 ok",
                "warning": "🟡 warning",
                "critical": "🔴 critical",
                "no_data": "⬜ pas de données",
                "no_baseline": "⬜ pas de baseline",
                "insufficient_data": "⬜ données insuffisantes",
            }
            fd_summary = fd.get("drift_summary", "no_data")
            fd_baseline = fd.get("baseline_available", False)
            fd_analyzed = fd.get("predictions_analyzed", 0)

            _fd_features_raw = fd.get("features", {})
            _fd_n_preds = (
                next(iter(_fd_features_raw.values()), {}).get("production_count")
                or fd_analyzed
            )
            st.markdown(f"**Statut global :** {_FD_BADGE.get(fd_summary, fd_summary)}")
            st.caption(f"{_fd_n_preds} prédictions analysées du {_fd_start} au {_fd_end}")

            features_dict = fd.get("features", {})
            if not fd_baseline:
                st.info(
                    "Aucune baseline de features enregistrée pour ce modèle. "
                    "Assurez-vous que le script `train.py` imprime un JSON avec `feature_stats`."
                )
            elif not features_dict:
                st.info("Aucune donnée de drift de features disponible sur cette période.")
            else:
                _STATUS_COLOR = {"ok": "🟢", "warning": "🟡", "critical": "🔴", "no_data": "⬜"}
                rows_fd = []
                for feat_name, feat_data in features_dict.items():
                    ds = feat_data.get("drift_status", "no_data")
                    rows_fd.append(
                        {
                            "Feature": feat_name,
                            "Statut": _STATUS_COLOR.get(ds, "⬜") + f" {ds}",
                            "Moy. prod.": (
                                round(feat_data["production_mean"], 4)
                                if feat_data.get("production_mean") is not None
                                else "—"
                            ),
                            "Moy. baseline": (
                                round(feat_data["baseline_mean"], 4)
                                if feat_data.get("baseline_mean") is not None
                                else "—"
                            ),
                            "Z-score": (
                                round(feat_data["z_score"], 3)
                                if feat_data.get("z_score") is not None
                                else "—"
                            ),
                            "PSI": (
                                round(feat_data["psi"], 4)
                                if feat_data.get("psi") is not None
                                else "—"
                            ),
                            "N prod.": feat_data.get("production_count", 0),
                        }
                    )
                df_fd = pd.DataFrame(rows_fd)
                st.dataframe(
                    df_fd,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        "Z-score": st.column_config.TextColumn(
                            "Z-score",
                            help=(
                                "**Z-score** : écart entre la moyenne de production et la moyenne baseline, "
                                "normalisé par l'écart-type baseline.\n\n"
                                "- 🟢 **ok** : Z < 2σ\n"
                                "- 🟡 **warning** : 2σ ≤ Z < 3σ\n"
                                "- 🔴 **critical** : Z ≥ 3σ\n\n"
                                "Un Z-score élevé indique que la distribution de la feature en production "
                                "s'est significativement éloignée des données d'entraînement."
                            ),
                        ),
                        "PSI": st.column_config.TextColumn(
                            "PSI",
                            help=(
                                "**PSI (Population Stability Index)** : mesure le changement global de "
                                "distribution d'une feature entre baseline et production.\n\n"
                                "- 🟢 **ok** : PSI < 0.1 — distribution stable\n"
                                "- 🟡 **warning** : 0.1 ≤ PSI < 0.2 — changement modéré\n"
                                "- 🔴 **critical** : PSI ≥ 0.2 — changement majeur, ré-entraînement recommandé\n\n"
                                "Calculé sur les **5 000 dernières prédictions** de la période sélectionnée "
                                "via une approximation gaussienne des bins baseline.\n\n"
                                "— = non disponible (nécessite une `feature_baseline` enregistrée sur le modèle)."
                            ),
                        ),
                        "Moy. prod.": st.column_config.TextColumn(
                            "Moy. prod.",
                            help="Moyenne de la feature sur les prédictions de production de la période sélectionnée.",
                        ),
                        "Moy. baseline": st.column_config.TextColumn(
                            "Moy. baseline",
                            help="Moyenne de la feature sur le dataset d'entraînement (stockée dans `feature_baseline`).",
                        ),
                        "Statut": st.column_config.TextColumn(
                            "Statut",
                            help=(
                                "**Statut de drift** de la feature, calculé à partir du Z-score et du PSI.\n\n"
                                "- 🟢 **ok** : aucun drift significatif détecté\n"
                                "- 🟡 **warning** : drift modéré — à surveiller\n"
                                "- 🔴 **critical** : drift important — ré-entraînement recommandé\n"
                                "- ⬜ **no_data** : pas assez de prédictions pour calculer le drift"
                            ),
                        ),
                        "N prod.": st.column_config.NumberColumn(
                            "N prod.",
                            help=(
                                "Nombre de prédictions de production utilisées pour calculer le drift "
                                "sur la période sélectionnée. "
                                "Un faible N rend le drift moins fiable statistiquement."
                            ),
                        ),
                    },
                )

                _color_map = {"ok": "#2ECC71", "warning": "#F39C12", "critical": "#E74C3C", "no_data": "#95A5A6"}

                _rows_z = [r for r in rows_fd if r["Z-score"] != "—"]
                if _rows_z:
                    _df_z = pd.DataFrame(
                        {
                            "Feature": [r["Feature"] for r in _rows_z],
                            "Z-score": [abs(float(r["Z-score"])) for r in _rows_z],
                            "Drift": [r["Statut"].split(" ", 1)[-1].strip() for r in _rows_z],
                        }
                    ).sort_values("Z-score", ascending=True)
                    fig_z = px.bar(
                        _df_z, x="Z-score", y="Feature", color="Drift", orientation="h",
                        title="Z-score par feature (|Δμ / σ baseline|)",
                        color_discrete_map=_color_map,
                    )
                    fig_z.add_vline(x=2.0, line_dash="dash", line_color="#F39C12",
                                    annotation_text="seuil warning (2σ)", annotation_position="top right")
                    fig_z.add_vline(x=3.0, line_dash="dash", line_color="#E74C3C",
                                    annotation_text="seuil critical (3σ)", annotation_position="top right")
                    fig_z.update_layout(height=max(250, len(_rows_z) * 32), yaxis_title="")
                    st.plotly_chart(fig_z, width='stretch')

                _rows_psi = [r for r in rows_fd if r["PSI"] != "—"]
                if _rows_psi:
                    _df_psi = pd.DataFrame(
                        {
                            "Feature": [r["Feature"] for r in _rows_psi],
                            "PSI": [float(r["PSI"]) for r in _rows_psi],
                            "Drift": [r["Statut"].split(" ", 1)[-1].strip() for r in _rows_psi],
                        }
                    ).sort_values("PSI", ascending=True)
                    fig_psi = px.bar(
                        _df_psi, x="PSI", y="Feature", color="Drift", orientation="h",
                        title="PSI par feature (Population Stability Index)",
                        color_discrete_map=_color_map,
                    )
                    fig_psi.add_vline(x=0.1, line_dash="dash", line_color="#F39C12",
                                      annotation_text="seuil warning (0.1)", annotation_position="top right")
                    fig_psi.add_vline(x=0.2, line_dash="dash", line_color="#E74C3C",
                                      annotation_text="seuil critical (0.2)", annotation_position="top right")
                    fig_psi.update_layout(height=max(250, len(_rows_psi) * 32), yaxis_title="")
                    st.plotly_chart(fig_psi, width='stretch')

        except Exception as e:
            st.warning(f"Impossible de charger le drift des features : {e}")

# Résolution des features pour les blocs Valider / Golden Tests
feature_baseline = selected.get("feature_baseline") or {}
feature_names_list: list = []
if feature_baseline:
    feature_names_list = list(feature_baseline.keys())
else:
    try:
        _feat_detail = fetch_model_detail(
            st.session_state.get("api_url"),
            st.session_state.get("api_token"),
            selected["name"],
            selected["version"],
        )
        feature_names_list = _feat_detail.get("feature_names") or []
        feature_baseline = _feat_detail.get("feature_baseline") or {}
    except Exception:
        pass

with st.expander("🔍 Valider le schéma JSON", expanded=False):
    st.markdown(
        "Testez un payload JSON avant d'envoyer une prédiction pour détecter "
        "les features manquantes, inattendues ou mal typées."
    )

    # Build example JSON from feature_baseline or feature_names_list
    if feature_baseline:
        example_payload = {
            feat: float(info.get("mean") or 0.0) for feat, info in feature_baseline.items()
        }
    elif feature_names_list:
        example_payload = {feat: 0.0 for feat in feature_names_list}
    else:
        example_payload = {}

    raw_json = st.text_area(
        "Payload JSON à valider",
        value=_json.dumps(example_payload, indent=2),
        height=180,
        key=f"validate_json_{selected['name']}_{selected['version']}",
    )

    if st.button(
        "✅ Valider",
        key=f"validate_btn_{selected['name']}_{selected['version']}",
    ):
        try:
            parsed = _json.loads(raw_json)
        except _json.JSONDecodeError as exc:
            st.error(f"JSON invalide : {exc}")
            parsed = None

        if parsed is not None:
            with st.spinner("Validation en cours…"):
                try:
                    result = client.validate_input(
                        selected["name"], selected["version"], parsed
                    )

                    if result.get("valid"):
                        st.success("✅ Schéma valide")
                    else:
                        st.error("❌ Schéma invalide")

                    errors = result.get("errors") or []
                    warnings = result.get("warnings") or []
                    expected = result.get("expected_features")

                    for err in errors:
                        etype = err.get("type", "")
                        feat = err.get("feature", "")
                        if etype == "missing_feature":
                            st.markdown(f"❌ **Feature manquante** : `{feat}`")
                        elif etype == "unexpected_feature":
                            st.markdown(f"❌ **Feature inattendue** : `{feat}`")
                        else:
                            st.markdown(f"❌ `{feat}` — {etype}")

                    for warn in warnings:
                        wtype = warn.get("type", "")
                        feat = warn.get("feature", "")
                        from_t = warn.get("from_type", "")
                        to_t = warn.get("to_type", "")
                        if wtype == "type_coercion":
                            st.markdown(
                                f"⚠️ **Coercition de type** : `{feat}` " f"({from_t} → {to_t})"
                            )
                        else:
                            st.markdown(f"⚠️ `{feat}` — {wtype}")

                    if expected:
                        st.markdown("**Features attendues :**")
                        st.markdown(" ".join(f"`{f}`" for f in expected))

                except Exception as e:
                    st.error(f"Erreur lors de la validation : {e}")

    st.divider()
    st.markdown("**💻 Exemple Python — appel `/predict`**")
    _predict_payload = {
        "model_name": selected["name"],
        "features": example_payload,
    }
    _python_code = (
        "import requests\n\n"
        'url = "http://localhost:8000/predict"\n'
        "headers = {\n"
        '    "Authorization": "Bearer <VOTRE_TOKEN>",\n'
        '    "Content-Type": "application/json",\n'
        "}\n"
        f"payload = {_json.dumps(_predict_payload, indent=4)}\n\n"
        "response = requests.post(url, headers=headers, json=payload)\n"
        "print(response.json())"
    )
    st.code(_python_code, language="python")

    st.markdown("**Exemple de réponse :**")
    _example_response = _json.dumps(
        {
            "model_name": selected["name"],
            "model_version": selected["version"],
            "prediction": "<résultat>",
            "probability": [0.98, 0.01, 0.01],
            "low_confidence": False,
            "id_obs": None,
            "selected_version": None,
            "shap_values": None,
            "shap_base_value": None,
        },
        indent=4,
        ensure_ascii=False,
    )
    st.code(_example_response, language="json")

# Tests de régression (Golden Test Set)
with st.expander("🧪 Tests de régression", expanded=False):
    is_admin_gt = st.session_state.get("is_admin", False)
    gt_model_name = selected["name"]
    gt_version = selected["version"]

    try:
        golden_tests = client.list_golden_tests(gt_model_name)
    except Exception:
        golden_tests = []

    # --- Liste des cas existants ---
    if golden_tests:
        import pandas as _pd_gt
        st.markdown(f"**{len(golden_tests)} cas enregistré(s)**")
        for t in golden_tests:
            tid = t["id"]
            expected = t.get("expected_output", "—")
            desc = t.get("description") or ""
            features = t.get("input_features") or {}
            created_at = t.get("created_at", "")

            label = f"#{tid}  →  {expected}"
            if desc:
                label += f"  —  {desc}"
            try:
                label += f"  ·  {_pd_gt.to_datetime(created_at).strftime('%Y-%m-%d %H:%M')}"
            except Exception:
                pass

            with st.expander(label, expanded=False):
                c_feat, c_meta = st.columns([3, 1])
                with c_feat:
                    st.markdown("**Features d'entrée**")
                    st.json(features)
                with c_meta:
                    st.markdown("**Sortie attendue**")
                    st.code(expected, language=None)
                    if desc:
                        st.markdown(f"*{desc}*")
                    if created_at:
                        try:
                            st.caption(_pd_gt.to_datetime(created_at).strftime("%Y-%m-%d %H:%M"))
                        except Exception:
                            st.caption(str(created_at)[:16])
                    if is_admin_gt:
                        if st.button("🗑 Supprimer", key=f"del_gt_{tid}", type="secondary"):
                            try:
                                client.delete_golden_test(gt_model_name, tid)
                                st.toast(f"Test #{tid} supprimé.", icon="✅")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Erreur : {e}")
    else:
        st.info("Aucun cas de test golden enregistré pour ce modèle.")

    if is_admin_gt:
        st.divider()

        # --- Ajouter un cas ---
        with st.expander("➕ Ajouter un cas de test", expanded=False):
            if feature_baseline:
                _gt_default = _json.dumps(
                    {feat: float(info.get("mean") or 0.0) for feat, info in feature_baseline.items()},
                    indent=2,
                )
            elif feature_names_list:
                _gt_default = _json.dumps({feat: 0.0 for feat in feature_names_list}, indent=2)
            else:
                _gt_default = '{\n  "feature1": 0.0\n}'

            with st.form("add_golden_test_form", clear_on_submit=True):
                raw_features_gt = st.text_area(
                    "Features (JSON)",
                    value=_gt_default,
                    height=130,
                    key="gt_features_input",
                )
                expected_gt = st.text_input("Expected output", placeholder="setosa", key="gt_expected")
                desc_gt = st.text_input("Description (optionnel)", value="", key="gt_desc")
                submitted_gt = st.form_submit_button("Ajouter")

            if submitted_gt:
                if not expected_gt.strip():
                    st.error("La sortie attendue (Expected output) est obligatoire.")
                else:
                    try:
                        import json as _json

                        features_gt = _json.loads(raw_features_gt)
                        client.create_golden_test(
                            gt_model_name,
                            {
                                "input_features": features_gt,
                                "expected_output": expected_gt.strip(),
                                "description": desc_gt.strip() or None,
                            },
                        )
                        st.success("Cas de test ajouté.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur : {e}")

        # --- Upload CSV ---
        with st.expander("📥 Importer depuis un CSV", expanded=False):
            st.caption(
                "Format : colonnes features + `expected_output` (requis) + `description` (optionnel). "
                "Exemple : `sepal_length,sepal_width,petal_length,petal_width,expected_output,description`"
            )
            csv_file_gt = st.file_uploader("Choisir un fichier CSV", type=["csv"], key="gt_csv_upload")
            if csv_file_gt and st.button("Importer", key="import_gt_csv"):
                try:
                    result_csv = client.upload_golden_tests_csv(
                        gt_model_name, csv_file_gt.read(), csv_file_gt.name
                    )
                    st.success(f"{result_csv.get('created', 0)} cas importés.")
                    if result_csv.get("errors"):
                        st.warning(f"Erreurs : {result_csv['errors']}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur lors de l'import : {e}")

    st.divider()

    # --- Lancer les tests ---
    if st.button("▶️ Lancer les tests", key="run_golden_tests", type="primary"):
        try:
            gt_result = client.run_golden_tests(gt_model_name, gt_version)
            col_p, col_f, col_r = st.columns(3)
            col_p.metric("✅ Passés", gt_result["passed"])
            col_f.metric("❌ Échoués", gt_result["failed"])
            col_r.metric("Taux", f"{gt_result['pass_rate']:.1%}")

            if gt_result["total_tests"] == 0:
                st.info("Aucun cas de test enregistré.")
            else:
                for d in gt_result["details"]:
                    passed = d["passed"]
                    icon = "✅" if passed else "❌"
                    desc_str = f" — {d['description']}" if d.get("description") else ""
                    label = f"{icon} #{d['test_id']}{desc_str} &nbsp; attendu : `{d['expected']}` / obtenu : `{d['actual']}`"
                    with st.expander(label, expanded=not passed):
                        c_l, c_r = st.columns(2)
                        with c_l:
                            st.markdown("**Attendu**")
                            st.code(str(d.get("expected", "—")), language=None)
                        with c_r:
                            st.markdown("**Obtenu**")
                            if passed:
                                st.code(str(d.get("actual", "—")), language=None)
                            else:
                                st.error(str(d.get("actual", "—")))
                        if d.get("input"):
                            st.markdown("**Features utilisées**")
                            st.json(d["input"])
        except Exception as e:
            st.error(f"Erreur lors de l'exécution des tests : {e}")

# Explorateur What-if
with st.expander("🔮 Explorateur What-if", expanded=False):
    _wif_api_url = st.session_state.get("api_url")
    _wif_api_token = st.session_state.get("api_token")

    wif_baseline = selected.get("feature_baseline") or {}
    wif_classes = selected.get("classes") or []
    if not wif_baseline:
        try:
            _wif_detail = fetch_model_detail(
                _wif_api_url, _wif_api_token, selected["name"], selected["version"]
            )
            wif_baseline = _wif_detail.get("feature_baseline") or {}
        except Exception:
            pass

    if not wif_baseline:
        st.info(
            "⚠️ Ce modèle n'a pas de baseline de features. "
            "Calculez-en une dans **Actions admin → Calculer le baseline** "
            "pour activer l'explorateur What-if."
        )
    else:
        _wif_key = f"whatif_history_{selected['name']}_{selected['version']}"
        if _wif_key not in st.session_state:
            st.session_state[_wif_key] = []

        st.caption(f"{len(wif_baseline)} features — ajustez les sliders puis cliquez **Prédire**.")

        _wif_cols = st.columns(2)
        wif_feature_values: dict = {}
        for _wif_i, (_wif_feat, _wif_stats) in enumerate(wif_baseline.items()):
            with _wif_cols[_wif_i % 2]:
                _wif_min = float(_wif_stats.get("min") or 0.0)
                _wif_max = float(_wif_stats.get("max") or 1.0)
                _wif_mean = float(_wif_stats.get("mean") or (_wif_min + _wif_max) / 2)
                if _wif_min == _wif_max:
                    st.metric(_wif_feat, _wif_mean)
                    wif_feature_values[_wif_feat] = _wif_mean
                else:
                    _wif_range = _wif_max - _wif_min
                    _wif_is_int = (
                        _wif_min == int(_wif_min) and _wif_max == int(_wif_max) and _wif_range <= 50
                    )
                    if _wif_is_int:
                        _wif_step = 1.0
                        _wif_default = float(round(_wif_mean))
                    else:
                        _wif_step = max(0.001, round(_wif_range / 100, 4))
                        _wif_default = _wif_mean
                    wif_feature_values[_wif_feat] = st.slider(
                        _wif_feat,
                        min_value=_wif_min,
                        max_value=_wif_max,
                        value=_wif_default,
                        step=_wif_step,
                        key=f"whatif_slider_{selected['name']}_{selected['version']}_{_wif_feat}",
                    )

        wif_use_shap = st.checkbox(
            "Afficher les contributions SHAP",
            value=True,
            key=f"whatif_shap_{selected['name']}_{selected['version']}",
        )

        if st.button(
            "🔮 Prédire",
            key=f"whatif_btn_{selected['name']}_{selected['version']}",
            type="primary",
        ):
            with st.spinner("Prédiction en cours…"):
                try:
                    wif_result = client.predict(
                        model_name=selected["name"],
                        model_version=selected["version"],
                        features=wif_feature_values,
                        explain=wif_use_shap,
                    )
                    st.session_state[_wif_key].append(
                        {
                            "features": wif_feature_values.copy(),
                            "prediction": wif_result.get("prediction"),
                            "probability": wif_result.get("probability"),
                            "low_confidence": wif_result.get("low_confidence"),
                            "shap_values": wif_result.get("shap_values"),
                        }
                    )

                    _wc1, _wc2, _wc3 = st.columns(3)
                    _wif_raw_pred = wif_result.get("prediction", "—")
                    _wif_probs = wif_result.get("probability")

                    # Résoudre index → label si le modèle retourne un entier
                    _wif_pred_label = str(_wif_raw_pred)
                    if wif_classes:
                        _wif_raw_str = str(_wif_raw_pred)
                        if _wif_raw_str.lstrip("-").isdigit():
                            try:
                                _wif_idx = int(_wif_raw_str)
                                if 0 <= _wif_idx < len(wif_classes):
                                    _wif_pred_label = f"{wif_classes[_wif_idx]}"
                            except (ValueError, IndexError):
                                pass

                    _wc1.metric("Prédiction", _wif_pred_label)
                    if _wif_probs:
                        _wc2.metric("Probabilité max", f"{max(_wif_probs):.2%}")
                    if wif_result.get("low_confidence"):
                        _wc3.warning("⚠️ Confiance faible")

                    if _wif_probs:
                        st.markdown("**Probabilités par classe :**")
                        if wif_classes and len(wif_classes) == len(_wif_probs):
                            _wif_prob_rows = [
                                {"Classe": str(c), "Probabilité": f"{p:.4f}"}
                                for c, p in zip(wif_classes, _wif_probs)
                            ]
                        else:
                            _wif_prob_rows = [
                                {"Classe": f"Classe {i}", "Probabilité": f"{p:.4f}"}
                                for i, p in enumerate(_wif_probs)
                            ]
                        st.dataframe(
                            pd.DataFrame(_wif_prob_rows),
                            width='stretch',
                            hide_index=True,
                        )

                    _wif_shap = wif_result.get("shap_values")
                    if wif_use_shap and _wif_shap:
                        st.markdown("**Contributions SHAP :**")
                        try:
                            import plotly.express as px

                            _wif_shap_df = pd.DataFrame(
                                [
                                    {
                                        "Feature": f,
                                        "Contribution": v,
                                        "Signe": "Positif" if v >= 0 else "Négatif",
                                    }
                                    for f, v in _wif_shap.items()
                                ]
                            ).sort_values("Contribution")
                            _wif_shap_fig = px.bar(
                                _wif_shap_df,
                                x="Contribution",
                                y="Feature",
                                orientation="h",
                                color="Signe",
                                color_discrete_map={
                                    "Positif": "#e74c3c",
                                    "Négatif": "#3498db",
                                },
                            )
                            _wif_shap_fig.update_layout(
                                yaxis_title="",
                                margin={"l": 10, "r": 10, "t": 10, "b": 10},
                                showlegend=False,
                            )
                            st.plotly_chart(_wif_shap_fig, width='stretch')
                        except ImportError:
                            st.bar_chart(pd.DataFrame({"SHAP": _wif_shap}))
                    elif wif_use_shap:
                        st.info("Les valeurs SHAP ne sont pas disponibles pour ce type de modèle.")

                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {e}")

        if st.session_state[_wif_key]:
            st.divider()
            st.markdown("**Historique des combinaisons testées**")
            _wif_feat_options = list(wif_baseline.keys())
            _wif_sel_feat = st.selectbox(
                "Feature à analyser",
                _wif_feat_options,
                key=f"whatif_feat_sel_{selected['name']}_{selected['version']}",
            )
            _wif_chart_rows = []
            for _wif_entry in st.session_state[_wif_key]:
                _wif_x = _wif_entry["features"].get(_wif_sel_feat)
                _wif_entry_probs = _wif_entry.get("probability")
                _wif_y = max(_wif_entry_probs) if _wif_entry_probs else _wif_entry.get("prediction")
                if _wif_x is not None and _wif_y is not None:
                    _wif_chart_rows.append({"x": _wif_x, "y": _wif_y})

            if _wif_chart_rows:
                _wif_chart_df = pd.DataFrame(_wif_chart_rows).sort_values("x")
                _wif_has_probs = st.session_state[_wif_key][-1].get("probability")
                _wif_y_label = "Probabilité max" if _wif_has_probs else "Prédiction"
                try:
                    import plotly.express as px

                    _wif_evo_fig = px.line(
                        _wif_chart_df,
                        x="x",
                        y="y",
                        markers=True,
                        labels={"x": _wif_sel_feat, "y": _wif_y_label},
                        title=f"Évolution vs {_wif_sel_feat}",
                    )
                    st.plotly_chart(_wif_evo_fig, width='stretch')
                except ImportError:
                    st.line_chart(_wif_chart_df.set_index("x")["y"])

            if st.button(
                "🗑️ Effacer l'historique",
                key=f"whatif_clear_{selected['name']}_{selected['version']}",
            ):
                st.session_state[_wif_key] = []
                st.rerun()

# Actions
if is_admin:
    with st.expander("⚙️ Actions admin", expanded=True):
        col_p, col_d = st.columns(2)

        # Readiness checklist
        try:
            readiness = client.get_model_readiness(selected["name"], selected["version"])
            checks = readiness.get("checks", {})
            check_labels = {
                "file_accessible": "Fichier accessible dans MinIO",
                "baseline_computed": "Baseline des features calculée",
                "no_critical_drift": "Aucun drift critique (24h)",
                "is_production": "Marqué en production",
            }
            st.markdown("**Checklist de production-readiness**")
            for key, label in check_labels.items():
                check = checks.get(key, {})
                passed = check.get("pass", False)
                detail = check.get("detail")
                icon = "✅" if passed else "❌"
                suffix = f" — `{detail}`" if detail else ""
                st.markdown(f"{icon} {label}{suffix}")
            ready = readiness.get("ready", False)
        except Exception:
            ready = True  # ne pas bloquer si l'endpoint échoue

        # Passer en production
        if not selected.get("is_production"):
            promote_help = None if ready else "Résoudre les checks ❌ ci-dessus avant de promouvoir"
            if col_p.button(
                "🚀 Passer en production",
                width='stretch',
                type="primary",
                disabled=not ready,
                help=promote_help,
            ):
                try:
                    client.update_model(selected["name"], selected["version"], {"is_production": True})
                    st.toast(
                        f"{selected['name']} v{selected['version']} est maintenant en production.",
                        icon="✅",
                    )
                    reload()
                except Exception as e:
                    st.toast(f"Erreur : {e}", icon="❌")
        else:
            col_p.info("🟢 Déjà en production")

        # Préchauffage du cache
        selected_cache_key = f"{selected['name']}:{selected['version']}"
        is_selected_cached = selected_cache_key in cached_model_keys
        col_w1, col_w2 = st.columns([1, 3])
        if is_selected_cached:
            col_w1.success("🔥 En cache")
        else:
            col_w1.warning("❄️ Non chargé")
        if not is_selected_cached:
            if col_w2.button(
                "🔥 Préchauffer le cache",
                width='stretch',
                key="warmup_btn",
                help="Charge le modèle en mémoire pour éliminer la latence de cold-start",
            ):
                with st.spinner("Chargement du modèle en cache…"):
                    try:
                        result = client.warmup_model(selected["name"], selected["version"])
                        st.toast(
                            f"Modèle chargé en {result['load_time_ms']:.0f} ms "
                            f"— clé cache : {result['cache_key']}",
                            icon="✅",
                        )
                        reload()
                    except Exception as e:
                        st.toast(f"Erreur lors du préchauffage : {e}", icon="❌")
        else:
            col_w2.info("Le modèle est déjà en cache — aucun préchauffage nécessaire.")

        # Supprimer
        if col_d.button("🗑️ Supprimer cette version", width='stretch', type="secondary"):
            st.session_state["confirm_delete_model"] = f"{selected['name']}:{selected['version']}"

        key = f"{selected['name']}:{selected['version']}"
        if st.session_state.get("confirm_delete_model") == key:
            st.warning(
                f"Supprimer **{selected['name']} v{selected['version']}** ? (fichier MinIO + run MLflow)"
            )
            c1, c2 = st.columns(2)
            if c1.button("Oui, supprimer", type="primary"):
                try:
                    client.delete_model_version(selected["name"], selected["version"])
                    st.toast("Modèle supprimé.", icon="✅")
                    st.session_state.pop("confirm_delete_model", None)
                    reload()
                except Exception as e:
                    st.toast(f"Erreur : {e}", icon="❌")
            if c2.button("Annuler"):
                st.session_state.pop("confirm_delete_model", None)
                st.rerun()

        # Modifier tags, webhook et déploiement
        with st.expander("✏️ Modifier les métadonnées"):
            new_webhook = st.text_input(
                "Webhook URL",
                value=selected.get("webhook_url") or "",
                placeholder="https://example.com/webhook",
            )
            new_tags_raw = st.text_input(
                "Tags (séparés par des virgules)",
                value=", ".join(selected.get("tags") or []),
                placeholder="production, finance, v2",
            )

            _cur_ct = selected.get("confidence_threshold")
            new_confidence_threshold = st.slider(
                "Confidence threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(_cur_ct) if _cur_ct is not None else 0.5,
                step=0.01,
                help="Seuil en dessous duquel la prédiction est marquée `low_confidence=True` dans la réponse.",
            )

            st.markdown("**Déploiement A/B / Shadow**")
            deploy_options = ["(inchangé)", "ab_test", "shadow", "production"]
            new_deploy_mode = st.selectbox(
                "Mode de déploiement",
                deploy_options,
                key="deploy_mode_select",
                help="ab_test = routage pondéré, shadow = exécution silencieuse en background",
            )
            new_traffic_weight = None
            if new_deploy_mode == "ab_test":
                new_traffic_weight = st.number_input(
                    "Poids du trafic (0.0 – 1.0)",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    value=float(selected.get("traffic_weight") or 0.5),
                    key="traffic_weight_input",
                    help="Fraction du trafic routé vers cette version (ex: 0.3 = 30%)",
                )

            if st.button("💾 Enregistrer", key="save_meta"):
                patch = {}
                current_webhook = selected.get("webhook_url") or ""
                if new_webhook != current_webhook:
                    patch["webhook_url"] = new_webhook if new_webhook else None
                new_tags = [t.strip() for t in new_tags_raw.split(",") if t.strip()]
                if new_tags != (selected.get("tags") or []):
                    patch["tags"] = new_tags if new_tags else None
                _stored_ct = float(_cur_ct) if _cur_ct is not None else 0.5
                if abs(new_confidence_threshold - _stored_ct) > 1e-9:
                    patch["confidence_threshold"] = new_confidence_threshold
                if new_deploy_mode != "(inchangé)":
                    patch["deployment_mode"] = new_deploy_mode
                    if new_deploy_mode == "ab_test" and new_traffic_weight is not None:
                        patch["traffic_weight"] = new_traffic_weight
                if patch:
                    try:
                        client.update_model(selected["name"], selected["version"], patch)
                        st.toast("Métadonnées mises à jour.", icon="✅")
                        reload()
                    except Exception as e:
                        st.toast(f"Erreur : {e}", icon="❌")
                else:
                    st.info("Aucun changement détecté.")

        # Ré-entraînement
        if selected.get("train_script_object_key"):
            with st.expander("🔄 Ré-entraîner", expanded=False):
                with st.form("retrain_form"):
                    st.markdown(f"**Ré-entraîner** `{selected['name']}` v`{selected['version']}`")
                    col_s, col_e = st.columns(2)
                    with col_s:
                        start_date = st.date_input("Date de début", key="retrain_start")
                    with col_e:
                        end_date = st.date_input("Date de fin", key="retrain_end")
                    new_version_input = st.text_input(
                        "Nouvelle version (laisser vide = auto-généré)",
                        value="",
                        placeholder=f"{selected['version']}-retrain-YYYYMMDDHHMMSS",
                        key="retrain_new_version",
                    )
                    set_prod = st.checkbox(
                        "Mettre en production après entraînement",
                        value=False,
                        key="retrain_set_prod",
                    )
                    submitted = st.form_submit_button("🚀 Lancer le ré-entraînement", type="primary")

                if submitted:
                    if start_date > end_date:
                        st.error("La date de début doit être antérieure à la date de fin.")
                    else:
                        with st.spinner("Ré-entraînement en cours… (peut prendre jusqu'à 10 minutes)"):
                            try:
                                result = client.retrain_model(
                                    name=selected["name"],
                                    version=selected["version"],
                                    start_date=str(start_date),
                                    end_date=str(end_date),
                                    new_version=new_version_input.strip() or None,
                                    set_production=set_prod,
                                )
                                if result.get("success"):
                                    st.toast(
                                        f"Ré-entraînement réussi ! "
                                        f"Nouvelle version : {result['new_version']}",
                                        icon="✅",
                                    )
                                else:
                                    st.toast(
                                        f"Échec du ré-entraînement : "
                                        f"{result.get('error', 'Erreur inconnue')}",
                                        icon="❌",
                                    )
                                with st.expander("📋 Logs stdout", expanded=not result.get("success")):
                                    st.code(result.get("stdout", "(vide)"), language="text")
                                with st.expander("⚠️ Logs stderr", expanded=not result.get("success")):
                                    st.code(result.get("stderr", "(vide)"), language="text")
                                if result.get("success"):
                                    reload()
                            except Exception as e:
                                st.toast(f"Erreur lors du ré-entraînement : {e}", icon="❌")

        # Calcul du baseline depuis la production
        with st.expander("📐 Calculer le baseline depuis la production"):
            st.markdown(
                "Calcule `{mean, std, min, max}` par feature depuis les prédictions de production "
                "récentes et sauvegarde le résultat comme **feature_baseline** du modèle, "
                "activant ainsi la détection de drift."
            )
            if selected.get("feature_baseline"):
                st.warning(
                    "⚠️ Une baseline existe déjà pour ce modèle. "
                    "Recalculer écrasera les valeurs actuelles et réinitialisera la détection de drift."
                )
            baseline_days = st.slider("Fenêtre temporelle (jours)", 7, 180, 30, key="baseline_days")
            baseline_dry_run = st.checkbox(
                "dry_run (simuler sans sauvegarder)", value=True, key="baseline_dry_run"
            )
            if st.button("Calculer", key="baseline_compute_btn", type="primary"):
                with st.spinner("Calcul en cours…"):
                    try:
                        result = client.compute_baseline(
                            name=selected["name"],
                            version=selected["version"],
                            days=baseline_days,
                            dry_run=baseline_dry_run,
                        )
                        st.markdown(
                            f"**Prédictions utilisées :** {result.get('predictions_used')} "
                            f"— fenêtre : {baseline_days} jours"
                        )
                        st.json(result.get("baseline", {}))
                        if baseline_dry_run:
                            st.info("Décochez **dry_run** pour sauvegarder le baseline.")
                        else:
                            st.toast("Baseline sauvegardé — drift actif pour ce modèle.", icon="✅")
                            st.cache_data.clear()
                    except Exception as e:
                        st.toast(f"Erreur : {e}", icon="❌")

        # Historique des modifications
        with st.expander("📜 Historique des modifications"):
            try:
                history_data = client.get_model_history(selected["name"], selected["version"], limit=20)
                entries = history_data.get("entries", [])
                total_hist = history_data.get("total", 0)
            except Exception as e:
                st.error(f"Impossible de charger l'historique : {e}")
                entries = []
                total_hist = 0

            if not entries:
                st.info("Aucun historique disponible pour cette version.")
            else:
                st.caption(f"{total_hist} entrée(s) au total — affichage des 20 dernières")
                for entry in entries:
                    ts = pd.to_datetime(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S UTC")
                    badge = _action_badge(entry["action"])
                    changed = ", ".join(entry.get("changed_fields") or []) or "—"
                    who = entry.get("changed_by_username") or "inconnu"

                    col_info, col_btn = st.columns([5, 1])
                    with col_info:
                        st.markdown(
                            f"**{ts}** — {badge} — par **{who}**  \n" f"Champs modifiés : `{changed}`"
                        )
                    with col_btn:
                        if is_admin:
                            if st.button(
                                "↩ Rollback",
                                key=f"rollback_btn_{entry['id']}",
                                type="secondary",
                                help=f"Restaurer l'état de l'entrée #{entry['id']}",
                            ):
                                st.session_state["confirm_rollback_id"] = entry["id"]
                                st.session_state["confirm_rollback_model"] = selected["name"]
                                st.session_state["confirm_rollback_version"] = selected["version"]

                    with st.expander(f"Snapshot #{entry['id']}", expanded=False):
                        st.json(entry["snapshot"])

                    st.divider()

                # Dialog de confirmation du rollback
                confirm_id = st.session_state.get("confirm_rollback_id")
                confirm_model = st.session_state.get("confirm_rollback_model")
                confirm_version = st.session_state.get("confirm_rollback_version")
                if (
                    confirm_id is not None
                    and confirm_model == selected["name"]
                    and confirm_version == selected["version"]
                ):
                    st.warning(
                        f"Restaurer les métadonnées de **{confirm_model} v{confirm_version}** "
                        f"à l'état capturé dans l'entrée **#{confirm_id}** ?  \n"
                        "Cette action est irréversible (mais loguée dans l'historique)."
                    )
                    c1, c2 = st.columns(2)
                    if c1.button("✅ Oui, restaurer", type="primary", key="confirm_rollback_yes"):
                        try:
                            result = client.rollback_model(confirm_model, confirm_version, confirm_id)
                            st.toast(
                                f"Rollback effectué — entrée #{result['new_history_id']} créée.",
                                icon="✅",
                            )
                            st.session_state.pop("confirm_rollback_id", None)
                            st.session_state.pop("confirm_rollback_model", None)
                            st.session_state.pop("confirm_rollback_version", None)
                            reload()
                        except Exception as e:
                            st.toast(f"Erreur lors du rollback : {e}", icon="❌")
                    if c2.button("❌ Annuler", key="confirm_rollback_no"):
                        st.session_state.pop("confirm_rollback_id", None)
                        st.session_state.pop("confirm_rollback_model", None)
                        st.session_state.pop("confirm_rollback_version", None)
                        st.rerun()
