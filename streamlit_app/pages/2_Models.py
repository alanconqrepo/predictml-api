"""
Gestion des modèles ML
"""

import json as _json
import os
import time

import pandas as pd
import streamlit as st
from utils.auth import get_client, require_auth

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


st.set_page_config(page_title="Models — PredictML", page_icon="🤖", layout="wide")
require_auth()

st.title("🤖 Gestion des modèles")

client = get_client()
is_admin = st.session_state.get("is_admin", False)

MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://localhost:5000")


@st.cache_data(ttl=15, show_spinner=False)
def fetch_models(api_url, token):
    c = get_client()
    return c.list_models()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_model_detail(api_url, token, name, version):
    c = get_client()
    return c.get_model(name, version)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_feature_importance(api_url, token, name, version, last_n, days):
    c = get_client()
    return c.get_feature_importance(name, version=version, last_n=last_n, days=days)


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
    with st.expander("➕ Uploader un nouveau modèle", expanded=not models):
        with st.form("upload_model_form", clear_on_submit=True):
            st.markdown("##### Fichiers")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                pkl_file = st.file_uploader(
                    "Fichier modèle (.pkl) *",
                    type=["pkl"],
                    help="Fichier pickle sérialisé. Aucune limite de taille imposée.",
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

            submitted = st.form_submit_button("⬆️ Uploader le modèle", type="primary")

        if submitted:
            errors = []
            if not pkl_file:
                errors.append("Le fichier .pkl est obligatoire.")
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
                        st.success(
                            f"Modèle **{result['name']}** v{result['version']} uploadé avec succès."
                        )
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

# Filtre par tag
all_tags = sorted({t for m in models for t in (m.get("tags") or [])})
if all_tags:
    tag_filter = st.selectbox("Filtrer par tag", ["(tous)"] + all_tags, key="tag_filter")
    if tag_filter != "(tous)":
        models = [m for m in models if tag_filter in (m.get("tags") or [])]

_DEPLOY_BADGE = {
    "ab_test": "🟠 A/B",
    "shadow": "🟣 Shadow",
    "production": "🟢 Prod",
}


# Tableau de synthèse
rows = []
for m in models:
    mode = m.get("deployment_mode")
    weight = m.get("traffic_weight")
    if mode == "ab_test":
        statut = f"🟠 A/B ({weight:.0%})" if weight is not None else "🟠 A/B"
    elif mode == "shadow":
        statut = "🟣 Shadow"
    elif m.get("is_production"):
        statut = "🟢 Production"
    elif m.get("is_active"):
        statut = "✅ Actif"
    else:
        statut = "⚫ Inactif"

    in_cache = f"{m.get('name')}:{m.get('version')}" in cached_model_keys
    rows.append(
        {
            "Nom": m.get("name", ""),
            "Version": m.get("version", ""),
            "Tags": ", ".join(m.get("tags") or []) or "—",
            "Algorithme": m.get("algorithm") or "—",
            "Accuracy": f"{m['accuracy']:.3f}" if m.get("accuracy") is not None else "—",
            "F1": f"{m['f1_score']:.3f}" if m.get("f1_score") is not None else "—",
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
        }
    )

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Comparaison multi-versions
# ---------------------------------------------------------------------------

st.subheader("📊 Comparaison multi-versions")

model_names = sorted({m["name"] for m in models})
compare_name = st.selectbox("Modèle à comparer", model_names, key="compare_model_name")

versions_for_model = [m["version"] for m in models if m["name"] == compare_name]
all_versions_label = f"Toutes ({len(versions_for_model)})"
version_options = [all_versions_label] + versions_for_model
selected_versions = st.multiselect(
    "Versions à inclure (vide = toutes)",
    versions_for_model,
    default=[],
    key="compare_versions_select",
)
compare_days = st.slider("Fenêtre latence (jours)", 1, 30, 7, key="compare_days")

if st.button("🔍 Comparer", key="compare_btn", type="primary"):
    versions_param = ",".join(selected_versions) if selected_versions else None
    with st.spinner("Comparaison en cours…"):
        try:
            cmp = client.compare_model_versions(
                compare_name, versions=versions_param, days=compare_days
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
                cmp_rows = []
                for v in cmp_versions:
                    cmp_rows.append(
                        {
                            "Version": v["version"],
                            "Production": "🟢 Oui" if v["is_production"] else "—",
                            "Accuracy": (
                                f"{v['accuracy']:.3f}" if v.get("accuracy") is not None else "—"
                            ),
                            "F1": f"{v['f1_score']:.3f}" if v.get("f1_score") is not None else "—",
                            "Latence p50 (ms)": (
                                f"{v['latency_p50_ms']:.1f}"
                                if v.get("latency_p50_ms") is not None
                                else "—"
                            ),
                            "Latence p95 (ms)": (
                                f"{v['latency_p95_ms']:.1f}"
                                if v.get("latency_p95_ms") is not None
                                else "—"
                            ),
                            "Drift": _DRIFT_BADGE.get(v.get("drift_status") or "", "—"),
                            "Brier score": (
                                f"{v['brier_score']:.4f}"
                                if v.get("brier_score") is not None
                                else "—"
                            ),
                            "Entraîné le": (
                                pd.to_datetime(v["trained_at"]).strftime("%Y-%m-%d")
                                if v.get("trained_at")
                                else "—"
                            ),
                            "Lignes entraîn.": (
                                f"{v['n_rows_trained']:,}"
                                if v.get("n_rows_trained") is not None
                                else "—"
                            ),
                        }
                    )
                st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
                st.caption(
                    f"Comparé le {pd.to_datetime(cmp['compared_at']).strftime('%Y-%m-%d %H:%M')} UTC"
                    f" — fenêtre latence : {compare_days} j"
                )
        except Exception as e:
            st.error(f"Erreur lors de la comparaison : {e}")

st.divider()
st.subheader("Détail et actions")

model_options = {f"{m['name']} v{m['version']}": m for m in models}
selected_label = st.selectbox("Sélectionner un modèle", list(model_options.keys()))
selected = model_options[selected_label]

# Détails
with st.expander("📋 Détails complets", expanded=True):
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"**Nom :** `{selected.get('name')}`")
        st.markdown(f"**Version :** `{selected.get('version')}`")
        st.markdown(f"**Description :** {selected.get('description') or '—'}")
        st.markdown(f"**Algorithme :** {selected.get('algorithm') or '—'}")
        st.markdown(f"**Dataset d'entraînement :** {selected.get('training_dataset') or '—'}")
        st.markdown(f"**Entraîné par :** {selected.get('trained_by') or '—'}")
        parent_v = selected.get("parent_version")
        if parent_v:
            st.markdown(f"**Dérivé de :** `v{parent_v}`")
        tags = selected.get("tags")
        st.markdown(f"**Tags :** {', '.join(tags) if tags else '—'}")
        webhook = selected.get("webhook_url")
        st.markdown(f"**Webhook URL :** `{webhook}`" if webhook else "**Webhook URL :** —")
    with col_r:
        st.markdown(f"**Accuracy :** {selected.get('accuracy') or '—'}")
        st.markdown(f"**F1 Score :** {selected.get('f1_score') or '—'}")
        st.markdown(f"**Precision :** {selected.get('precision') or '—'}")
        st.markdown(f"**Recall :** {selected.get('recall') or '—'}")
        st.markdown(f"**Nb features :** {selected.get('features_count') or '—'}")
        last_seen = selected.get("last_seen")
        st.markdown(
            f"**Dernière prédiction :** {pd.to_datetime(last_seen).strftime('%Y-%m-%d %H:%M') if last_seen else '—'}"
        )
        classes = selected.get("classes")
        st.markdown(f"**Classes :** {classes if classes else '—'}")

    if selected.get("training_params"):
        st.markdown("**Hyperparamètres :**")
        st.json(selected["training_params"])

    mlflow_id = selected.get("mlflow_run_id")
    if mlflow_id:
        mlflow_link = f"{MLFLOW_URL}/#/experiments/0/runs/{mlflow_id}"
        st.markdown(f"**MLflow run :** [{mlflow_id}]({mlflow_link})")
    else:
        st.markdown("**MLflow run :** —")

    minio_key = selected.get("minio_object_key")
    if minio_key:
        st.markdown(f"**MinIO object :** `{selected.get('minio_bucket')}/{minio_key}`")
        size = selected.get("file_size_bytes")
        if size:
            st.markdown(f"**Taille fichier :** {size / 1024:.1f} KB")
        if is_admin:
            size_label = f" ({size / 1024:.1f} KB)" if size else ""
            try:
                pkl_bytes = client.download_model(selected["name"], selected["version"])
                st.download_button(
                    label=f"⬇️ Télécharger le .pkl{size_label}",
                    data=pkl_bytes,
                    file_name=f"{selected['name']}_{selected['version']}.pkl",
                    mime="application/octet-stream",
                )
            except Exception as e:
                st.error(f"Erreur lors du téléchargement : {e}")

# Importance des features (SHAP agrégé)
with st.expander("📊 Importance des features (SHAP)", expanded=False):
    fi_col1, fi_col2 = st.columns(2)
    fi_days = fi_col1.slider("Fenêtre (jours)", 1, 30, 7, key="fi_days_slider")
    fi_last_n = fi_col2.slider("Prédictions max", 10, 500, 100, step=10, key="fi_last_n_slider")
    try:
        fi_data = fetch_feature_importance(
            st.session_state.get("api_url"),
            st.session_state.get("api_token"),
            selected["name"],
            selected["version"],
            last_n=fi_last_n,
            days=fi_days,
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
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(fi_df.set_index("Feature")["Importance SHAP"])
            st.caption(
                f"{sample_size} prédictions analysées"
                f" — fenêtre : {fi_days} j — version : {fi_data.get('version')}"
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

# Test interactif de prédiction
with st.expander("🧪 Tester le modèle", expanded=False):
    api_url_key = st.session_state.get("api_url")
    api_token_key = st.session_state.get("api_token")

    # Resolve feature list: prefer feature_baseline from list cache, else call get_model
    feature_baseline = selected.get("feature_baseline") or {}
    feature_names_list: list = []

    if feature_baseline:
        feature_names_list = list(feature_baseline.keys())
    else:
        try:
            detail = fetch_model_detail(
                api_url_key, api_token_key, selected["name"], selected["version"]
            )
            feature_names_list = detail.get("feature_names") or []
            feature_baseline = detail.get("feature_baseline") or {}
        except Exception as e:
            st.warning(f"Impossible de charger les features : {e}")

    classes = selected.get("classes") or []

    if not feature_names_list:
        st.info(
            "Aucune information de features disponible pour ce modèle. "
            "Uploadez le modèle avec un `feature_baseline` ou entraînez-le avec un DataFrame pandas."
        )
    else:
        st.markdown(f"**{len(feature_names_list)} features attendues**")

        with_shap = st.checkbox(
            "Avec explication SHAP",
            value=False,
            key=f"shap_cb_{selected['name']}_{selected['version']}",
        )

        n_cols = 3 if len(feature_names_list) > 4 else 2
        cols = st.columns(n_cols)
        feature_values: dict = {}
        for i, feat in enumerate(feature_names_list):
            with cols[i % n_cols]:
                baseline_info = feature_baseline.get(feat)
                if baseline_info is not None:
                    default_val = float(baseline_info.get("mean") or 0.0)
                    feature_values[feat] = st.number_input(
                        feat,
                        value=default_val,
                        key=f"feat_{selected['name']}_{selected['version']}_{feat}",
                        format="%.4f",
                    )
                else:
                    raw = st.text_input(
                        feat,
                        value="",
                        key=f"feat_{selected['name']}_{selected['version']}_{feat}",
                    )
                    try:
                        feature_values[feat] = float(raw) if raw != "" else 0.0
                    except ValueError:
                        feature_values[feat] = raw

        if st.button(
            "🔮 Prédire",
            key=f"predict_btn_{selected['name']}_{selected['version']}",
            type="primary",
        ):
            with st.spinner("Prédiction en cours…"):
                try:
                    t0 = time.time()
                    result = client.predict(
                        model_name=selected["name"],
                        model_version=selected["version"],
                        features=feature_values,
                        explain=with_shap,
                    )
                    latency_ms = (time.time() - t0) * 1000

                    col_pred, col_ver, col_lat = st.columns(3)
                    col_pred.metric("Prédiction", str(result.get("prediction", "—")))
                    col_ver.metric(
                        "Version utilisée",
                        result.get("selected_version") or result.get("model_version", "—"),
                    )
                    col_lat.metric("Latence", f"{latency_ms:.0f} ms")

                    if result.get("low_confidence"):
                        st.warning("⚠️ Confiance faible (en dessous du seuil configuré)")

                    probs = result.get("probability")
                    if probs:
                        st.markdown("**Probabilités par classe :**")
                        if classes and len(classes) == len(probs):
                            prob_rows = [
                                {"Classe": str(c), "Probabilité": f"{p:.4f}"}
                                for c, p in zip(classes, probs)
                            ]
                        else:
                            prob_rows = [
                                {"Classe": f"Classe {i}", "Probabilité": f"{p:.4f}"}
                                for i, p in enumerate(probs)
                            ]
                        st.dataframe(
                            pd.DataFrame(prob_rows),
                            use_container_width=True,
                            hide_index=True,
                        )

                    shap_vals = result.get("shap_values")
                    if with_shap and shap_vals:
                        st.markdown("**Contributions SHAP :**")
                        sorted_shap = dict(
                            sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
                        )
                        st.bar_chart(pd.DataFrame({"SHAP": sorted_shap}))
                    elif with_shap:
                        st.info("Les valeurs SHAP ne sont pas disponibles pour ce type de modèle.")

                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {e}")

    st.divider()
    with st.expander("🔍 Valider le schéma JSON", expanded=False):
        st.markdown(
            "Testez un payload JSON avant d'envoyer une prédiction pour détecter "
            "les features manquantes, inattendues ou mal typées."
        )

        # Build example JSON from feature_baseline or feature_names_list
        if feature_baseline:
            example_payload = {
                feat: float(info.get("mean") or 0.0)
                for feat, info in feature_baseline.items()
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
                                    f"⚠️ **Coercition de type** : `{feat}` "
                                    f"({from_t} → {to_t})"
                                )
                            else:
                                st.markdown(f"⚠️ `{feat}` — {wtype}")

                        if expected:
                            st.markdown("**Features attendues :**")
                            st.markdown(
                                " ".join(f"`{f}`" for f in expected)
                            )

                    except Exception as e:
                        st.error(f"Erreur lors de la validation : {e}")

# Actions
if is_admin:
    st.subheader("Actions admin")
    col_p, col_d = st.columns(2)

    # Passer en production
    if not selected.get("is_production"):
        if col_p.button("🚀 Passer en production", use_container_width=True, type="primary"):
            try:
                client.update_model(selected["name"], selected["version"], {"is_production": True})
                st.success(
                    f"**{selected['name']} v{selected['version']}** est maintenant en production."
                )
                reload()
            except Exception as e:
                st.error(f"Erreur : {e}")
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
            use_container_width=True,
            key="warmup_btn",
            help="Charge le modèle en mémoire pour éliminer la latence de cold-start",
        ):
            with st.spinner("Chargement du modèle en cache…"):
                try:
                    result = client.warmup_model(selected["name"], selected["version"])
                    st.success(
                        f"Modèle chargé en {result['load_time_ms']:.0f} ms "
                        f"— clé cache : `{result['cache_key']}`"
                    )
                    reload()
                except Exception as e:
                    st.error(f"Erreur lors du préchauffage : {e}")
    else:
        col_w2.info("Le modèle est déjà en cache — aucun préchauffage nécessaire.")

    # Supprimer
    if col_d.button("🗑️ Supprimer cette version", use_container_width=True, type="secondary"):
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
                st.success("Modèle supprimé.")
                st.session_state.pop("confirm_delete_model", None)
                reload()
            except Exception as e:
                st.error(f"Erreur : {e}")
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
            if new_deploy_mode != "(inchangé)":
                patch["deployment_mode"] = new_deploy_mode
                if new_deploy_mode == "ab_test" and new_traffic_weight is not None:
                    patch["traffic_weight"] = new_traffic_weight
            if patch:
                try:
                    client.update_model(selected["name"], selected["version"], patch)
                    st.success("Métadonnées mises à jour.")
                    reload()
                except Exception as e:
                    st.error(f"Erreur : {e}")
            else:
                st.info("Aucun changement détecté.")

    # Ré-entraînement
    if selected.get("train_script_object_key"):
        st.divider()
        if st.button("🔄 Ré-entraîner", use_container_width=True, key="retrain_btn"):
            current = st.session_state.get("show_retrain_form")
            toggle_key = f"{selected['name']}:{selected['version']}"
            if current == toggle_key:
                st.session_state.pop("show_retrain_form", None)
            else:
                st.session_state["show_retrain_form"] = toggle_key

        retrain_key = f"{selected['name']}:{selected['version']}"
        if st.session_state.get("show_retrain_form") == retrain_key:
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
                            st.session_state.pop("show_retrain_form", None)
                            if result.get("success"):
                                st.success(
                                    f"Ré-entraînement réussi ! "
                                    f"Nouvelle version : **{result['new_version']}**"
                                )
                            else:
                                st.error(
                                    f"Échec du ré-entraînement : "
                                    f"{result.get('error', 'Erreur inconnue')}"
                                )
                            with st.expander("📋 Logs stdout", expanded=not result.get("success")):
                                st.code(result.get("stdout", "(vide)"), language="text")
                            with st.expander("⚠️ Logs stderr", expanded=not result.get("success")):
                                st.code(result.get("stderr", "(vide)"), language="text")
                            if result.get("success"):
                                reload()
                        except Exception as e:
                            st.error(f"Erreur lors du ré-entraînement : {e}")

    # Calcul du baseline depuis la production
    with st.expander("📐 Calculer le baseline depuis la production"):
        st.markdown(
            "Calcule `{mean, std, min, max}` par feature depuis les prédictions de production "
            "récentes et sauvegarde le résultat comme **feature_baseline** du modèle, "
            "activant ainsi la détection de drift."
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
                        st.success(
                            "Baseline sauvegardé — le drift est maintenant actif pour ce modèle."
                        )
                        st.cache_data.clear()
                except Exception as e:
                    st.error(f"Erreur : {e}")

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
                        st.success(
                            f"Rollback effectué. Nouvelle entrée d'historique "
                            f"**#{result['new_history_id']}** créée."
                        )
                        st.session_state.pop("confirm_rollback_id", None)
                        st.session_state.pop("confirm_rollback_model", None)
                        st.session_state.pop("confirm_rollback_version", None)
                        reload()
                    except Exception as e:
                        st.error(f"Erreur lors du rollback : {e}")
                if c2.button("❌ Annuler", key="confirm_rollback_no"):
                    st.session_state.pop("confirm_rollback_id", None)
                    st.session_state.pop("confirm_rollback_model", None)
                    st.session_state.pop("confirm_rollback_version", None)
                    st.rerun()
