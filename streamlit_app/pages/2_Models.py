"""
ML model management
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
from utils.i18n import t

# --- History helpers ---

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


# ── Example scripts ───────────────────────────────────────────────────────────

_SCRIPTS_DIR = Path(__file__).parent.parent / "documentation" / "Scripts"

# (category, relative path from _SCRIPTS_DIR, i18n description key)
_EXAMPLE_SCRIPTS: list[tuple[str, str, str]] = [
    ("Iris", "iris/train_iris.py", "iris_train_desc"),
    ("Iris", "iris/upload_iris_model.py", "iris_upload_desc"),
    ("Wine", "wine/train_wine.py", "wine_train_desc"),
    ("Wine", "wine/upload_wine_model.py", "wine_upload_desc"),
]


def _read_script(rel_path: str) -> str:
    try:
        return (_SCRIPTS_DIR / rel_path).read_text(encoding="utf-8")
    except Exception:
        return t("models.scripts_example.script_not_found", path=rel_path)


@st.dialog(t("models.scripts_example.dialog_title"), width="large")
def _view_script_dialog(rel_path: str) -> None:
    content = _read_script(rel_path)
    basename = Path(rel_path).name
    st.code(content, language="python", line_numbers=True)
    st.download_button(
        t("models.scripts_example.btn_download"),
        data=content,
        file_name=basename,
        mime="text/x-python",
        key=f"dl_dialog_{rel_path}",
        width='stretch',
    )


st.set_page_config(page_title=t("models.page_title"), page_icon="🤖", layout="wide")
require_auth()

col_title, col_refresh = st.columns([8, 1])
col_title.title(t("models.title"))
if col_refresh.button(t("models.btn_refresh"), key="models_refresh", width='stretch'):
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
    """Returns the total number of predictions over the period (lightweight call with limit=1)."""
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
    """Fetches feature drift via /monitoring/model/{name}."""
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
    """Returns the list of 'name:version' keys currently in Redis cache."""
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
    """Downloads the .joblib from MinIO, cached 5 min per version."""
    from utils.api_client import APIClient

    try:
        return APIClient(base_url=api_url, token=token).download_model(name, version)
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_train_script(api_url: str, token: str, name: str, version: str) -> bytes | None:
    """Downloads the train.py script from MinIO, cached 5 min per version."""
    from utils.api_client import APIClient

    try:
        return APIClient(base_url=api_url, token=token).download_train_script(name, version)
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_training_dataset(api_url: str, token: str, name: str, version: str) -> bytes | None:
    """Downloads the CSV dataset from MinIO, cached 5 min per version."""
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
    st.error(t("models.load_error", error=e))
    st.stop()

cached_model_keys: list = []
if is_admin:
    cached_model_keys = fetch_cached_models(
        st.session_state.get("api_url"), st.session_state.get("api_token")
    )

if is_admin:
    with st.expander(t("models.scripts_example.expander"), expanded=False):
        st.caption(t("models.scripts_example.caption"))
        _current_category = None
        for _category, _script_rel, _desc_key in _EXAMPLE_SCRIPTS:
            if _category != _current_category:
                st.markdown(f"**{_category}**")
                _current_category = _category
            _col_desc, _col_view, _col_dl = st.columns([5, 1.5, 1.5])
            _script_basename = Path(_script_rel).name
            _col_desc.markdown(f"**`{_script_basename}`**  \n{t(f'models.scripts_example.{_desc_key}')}")
            if _col_view.button(t("models.scripts_example.btn_view"), key=f"view_{_script_rel}", width='stretch'):
                _view_script_dialog(_script_rel)
            _col_dl.download_button(
                t("models.scripts_example.btn_download"),
                data=_read_script(_script_rel),
                file_name=_script_basename,
                mime="text/x-python",
                key=f"dl_{_script_rel}",
                width='stretch',
            )

if is_admin:
    with st.expander(t("models.upload.expander"), expanded=not models):
        with st.form("upload_model_form", clear_on_submit=True):
            st.markdown(t("models.upload.section_files"))
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                pkl_file = st.file_uploader(
                    t("models.upload.pkl_label"),
                    type=["joblib", "pkl"],
                    help=t("models.upload.pkl_help"),
                )
            with col_f2:
                train_file = st.file_uploader(
                    t("models.upload.train_label"),
                    type=["py"],
                    help=t("models.upload.train_help"),
                )

            st.markdown(t("models.upload.section_identity"))
            col_n, col_v = st.columns(2)
            with col_n:
                up_name = st.text_input(t("models.upload.name_label"), placeholder=t("models.upload.name_placeholder"))
            with col_v:
                up_version = st.text_input(t("models.upload.version_label"), placeholder=t("models.upload.version_placeholder"))

            up_description = st.text_area(
                t("models.upload.description_label"), placeholder=t("models.upload.description_placeholder"), height=80
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
                "Other",
            ]
            up_algorithm = st.selectbox(t("models.upload.algorithm_label"), _ALGO_OPTIONS)

            st.markdown(t("models.upload.section_metrics"))
            col_acc, col_auc, col_f1s = st.columns(3)
            with col_acc:
                up_accuracy = st.number_input(
                    t("models.upload.accuracy_label"), min_value=0.0, max_value=1.0, value=None, step=0.001, format="%.4f"
                )
            with col_auc:
                up_auc = st.number_input(
                    t("models.upload.auc_label"),
                    min_value=0.0,
                    max_value=1.0,
                    value=None,
                    step=0.001,
                    format="%.4f",
                    help=t("models.upload.auc_help"),
                )
            with col_f1s:
                up_f1 = st.number_input(
                    t("models.upload.f1_label"), min_value=0.0, max_value=1.0, value=None, step=0.001, format="%.4f"
                )

            up_tags_raw = st.text_input(
                t("models.upload.tags_label"),
                placeholder=t("models.upload.tags_placeholder"),
            )

            st.markdown(t("models.upload.section_baseline"))
            col_bl1, col_bl2 = st.columns([3, 1])
            with col_bl1:
                compute_baseline_auto = st.checkbox(
                    t("models.upload.baseline_checkbox"),
                    value=True,
                    help=t("models.upload.baseline_checkbox_help"),
                )
            with col_bl2:
                baseline_days = st.number_input(
                    t("models.upload.baseline_days_label"), min_value=1, max_value=180, value=30
                )

            submitted = st.form_submit_button(t("models.upload.submit_btn"), type="primary")

        if submitted:
            errors = []
            if not pkl_file:
                errors.append(t("models.upload.error_no_file"))
            if not up_name.strip():
                errors.append(t("models.upload.error_no_name"))
            if not up_version.strip():
                errors.append(t("models.upload.error_no_version"))

            if errors:
                for err in errors:
                    st.error(err)
            else:
                tags = (
                    [_tag.strip() for _tag in up_tags_raw.split(",") if _tag.strip()]
                    if up_tags_raw.strip()
                    else []
                )
                algorithm = up_algorithm if up_algorithm and up_algorithm != "Other" else None

                with st.spinner(t("models.upload.spinner", filename=pkl_file.name)):
                    progress = st.progress(0, text=t("models.upload.progress_sending"))
                    try:
                        train_bytes = train_file.read() if train_file else None
                        train_fname = train_file.name if train_file else None
                        progress.progress(30, text=t("models.upload.progress_in_progress"))
                        result = client.upload_model(
                            name=up_name.strip(),
                            version=up_version.strip(),
                            file_bytes=pkl_file.read(),
                            filename=pkl_file.name,
                            description=up_description.strip() or None,
                            algorithm=algorithm,
                            accuracy=up_accuracy,
                            auc=up_auc,
                            f1_score=up_f1,
                            tags=tags or None,
                            train_file_bytes=train_bytes,
                            train_filename=train_fname,
                        )
                        progress.progress(100, text=t("models.upload.progress_done"))
                        st.toast(
                            t("models.upload.success_toast", name=result['name'], version=result['version']), icon="✅"
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
                                    t("models.upload.baseline_success", n_features=len(bl['baseline']), n_predictions=bl['predictions_used'])
                                )
                            except Exception as bl_exc:
                                _bl_detail = ""
                                try:
                                    if hasattr(bl_exc, "response") and bl_exc.response is not None:
                                        _bl_detail = bl_exc.response.json().get("detail", "")
                                except Exception:
                                    pass
                                st.warning(t("models.upload.baseline_warning", error=_bl_detail or bl_exc))
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
                        st.error(t("models.upload.upload_error", detail=detail or exc))

if not models:
    st.info(t("models.no_models"))
    st.stop()

# Summary
col1, col2, col3 = st.columns(3)
col1.metric(t("models.summary.total_models"), len(models))
col2.metric(t("models.summary.in_production"), sum(1 for m in models if m.get("is_production")))
col3.metric(t("models.summary.with_mlflow"), sum(1 for m in models if m.get("mlflow_run_id")))

st.divider()

# ── Filter bar (single row) ───────────────────────────────────────────────────
_all_model_names = sorted({m["name"] for m in models})
_all_tags = sorted({_tag for m in models for _tag in (m.get("tags") or [])})
_filter_cols = st.columns([3, 2, 2]) if _all_tags else st.columns([3, 2])

with _filter_cols[0]:
    search_query = st.text_input(
        t("models.filters.search_label"),
        placeholder=t("models.filters.search_placeholder"),
        key="models_search",
    )

with _filter_cols[1]:
    model_name_filter = st.selectbox(
        t("models.filters.model_label"),
        [t("models.filters.model_all")] + _all_model_names,
        key="models_name_filter",
    )

if len(_filter_cols) == 3:
    with _filter_cols[2]:
        tag_filter = st.selectbox(t("models.filters.tag_label"), [t("models.filters.tag_all")] + _all_tags, key="tag_filter")
else:
    tag_filter = t("models.filters.tag_all")

# Apply filters
if search_query:
    q = search_query.lower()
    models = [
        m for m in models
        if q in (m.get("name") or "").lower() or q in (m.get("description") or "").lower()
    ]
if model_name_filter != t("models.filters.model_all"):
    models = [m for m in models if m.get("name") == model_name_filter]
if tag_filter != t("models.filters.tag_all"):
    models = [m for m in models if tag_filter in (m.get("tags") or [])]

_DEPLOY_BADGE = {
    "ab_test": "🟠 A/B",
    "shadow": "🟣 Shadow",
    "production": "🟢 Prod",
}


def _infer_task(m: dict) -> str:
    """Returns the task label from model_task, or infers it from classes/metrics."""
    task = m.get("model_task")
    if not task:
        classes = m.get("classes")
        tm = m.get("training_metrics") or {}
        if classes:
            task = "classification_binary" if len(classes) == 2 else "classification_multiclass"
        elif tm.get("r2") is not None or tm.get("rmse") is not None or tm.get("mae") is not None:
            task = "regression"
        elif m.get("accuracy") is not None or m.get("f1_score") is not None:
            task = "classification_multiclass"
    _TASK_LABELS = {
        "regression": t("models.table.task_regression"),
        "classification_binary": t("models.table.task_binary"),
        "classification_multiclass": t("models.table.task_multiclass"),
    }
    return _TASK_LABELS.get(task or "", "—")


def _statut(m: dict) -> str:
    mode = m.get("deployment_mode")
    weight = m.get("traffic_weight")
    if mode == "ab_test":
        return f"🟠 A/B ({weight:.0%})" if weight is not None else "🟠 A/B"
    if mode == "shadow":
        return "🟣 Shadow"
    if m.get("is_production"):
        return t("models.table.status_production")
    if m.get("is_active"):
        return t("models.table.status_active")
    return t("models.table.status_inactive")


# Summary table
_col_name = t("models.table.col_name")
_col_version = t("models.table.col_version")
_col_tags = t("models.table.col_tags")
_col_algorithm = t("models.table.col_algorithm")
_col_task = t("models.table.col_task")
_col_baseline = t("models.table.col_baseline")
_col_cache = t("models.table.col_cache")
_col_status = t("models.table.col_status")
_col_creator = t("models.table.col_creator")
_col_created_at = t("models.table.col_created_at")
_col_last_pred = t("models.table.col_last_pred")
_col_accuracy_eval = t("models.table.col_accuracy_eval")
_col_auc_eval = t("models.table.col_auc_eval")
_col_f1_eval = t("models.table.col_f1_eval")
_col_r2_eval = t("models.table.col_r2_eval")
_col_rmse_eval = t("models.table.col_rmse_eval")

rows = []
for m in models:
    statut = _statut(m)

    in_cache = f"{m.get('name')}:{m.get('version')}" in cached_model_keys
    rows.append(
        {
            _col_name: m.get("name", ""),
            _col_version: m.get("version", ""),
            _col_tags: ", ".join(m.get("tags") or []) or "—",
            _col_algorithm: m.get("algorithm") or "—",
            _col_task: _infer_task(m),
            _col_baseline: t("models.table.baseline_ok") if m.get("feature_baseline") else t("models.table.baseline_missing"),
            _col_cache: t("models.table.cache_hot") if in_cache else t("models.table.cache_cold"),
            _col_status: statut,
            _col_creator: m.get("creator_username") or "—",
            _col_created_at: (
                pd.to_datetime(m.get("created_at")).strftime("%Y-%m-%d")
                if m.get("created_at")
                else "—"
            ),
            _col_last_pred: (
                pd.to_datetime(m.get("last_seen")).strftime("%Y-%m-%d %H:%M")
                if m.get("last_seen")
                else "—"
            ),
            _col_accuracy_eval: f"{m['accuracy']:.3f}" if m.get("accuracy") is not None else "—",
            _col_auc_eval: f"{m['auc']:.3f}" if m.get("auc") is not None else "—",
            _col_f1_eval: f"{m['f1_score']:.3f}" if m.get("f1_score") is not None else "—",
            _col_r2_eval: (
                f"{(m.get('training_metrics') or {}).get('r2'):.3f}"
                if (m.get("training_metrics") or {}).get("r2") is not None
                else "—"
            ),
            _col_rmse_eval: (
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
        _col_name: st.column_config.TextColumn(
            _col_name,
            help=t("models.table.col_name_help"),
        ),
        _col_version: st.column_config.TextColumn(
            _col_version,
            help=t("models.table.col_version_help"),
        ),
        _col_tags: st.column_config.TextColumn(
            _col_tags,
            help=t("models.table.col_tags_help"),
        ),
        _col_algorithm: st.column_config.TextColumn(
            _col_algorithm,
            help=t("models.table.col_algorithm_help"),
        ),
        _col_task: st.column_config.TextColumn(
            _col_task,
            help=t("models.table.col_task_help"),
        ),
        _col_baseline: st.column_config.TextColumn(
            _col_baseline,
            help=t("models.table.col_baseline_help"),
        ),
        _col_cache: st.column_config.TextColumn(
            _col_cache,
            help=t("models.table.col_cache_help"),
        ),
        _col_status: st.column_config.TextColumn(
            _col_status,
            help=t("models.table.col_status_help"),
        ),
        _col_creator: st.column_config.TextColumn(
            _col_creator,
            help=t("models.table.col_creator_help"),
        ),
        _col_created_at: st.column_config.TextColumn(
            _col_created_at,
            help=t("models.table.col_created_at_help"),
        ),
        _col_last_pred: st.column_config.TextColumn(
            _col_last_pred,
            help=t("models.table.col_last_pred_help"),
        ),
        _col_accuracy_eval: st.column_config.TextColumn(
            _col_accuracy_eval,
            help=t("models.table.col_accuracy_eval_help"),
        ),
        _col_f1_eval: st.column_config.TextColumn(
            _col_f1_eval,
            help=t("models.table.col_f1_eval_help"),
        ),
        _col_r2_eval: st.column_config.TextColumn(
            _col_r2_eval,
            help=t("models.table.col_r2_eval_help"),
        ),
        _col_rmse_eval: st.column_config.TextColumn(
            _col_rmse_eval,
            help=t("models.table.col_rmse_eval_help"),
        ),
    },
)
st.caption(t("models.summary.caption"))

# ---------------------------------------------------------------------------
# Multi-version comparison
# ---------------------------------------------------------------------------

with st.expander(t("models.compare.expander"), expanded=False):
    model_names = sorted({m["name"] for m in models})
    compare_search = st.text_input(
        t("models.compare.filter_label"), key="compare_search", placeholder=t("models.compare.filter_placeholder")
    )
    compare_filtered = (
        [n for n in model_names if compare_search.lower() in n.lower()]
        if compare_search
        else model_names
    )
    compare_name = st.selectbox(
        t("models.compare.model_label"), compare_filtered or model_names, key="compare_model_name"
    )

    versions_for_model = [m["version"] for m in models if m["name"] == compare_name]
    all_versions_label = f"All ({len(versions_for_model)})"
    version_options = [all_versions_label] + versions_for_model
    selected_versions = st.multiselect(
        t("models.compare.versions_label"),
        versions_for_model,
        default=[],
        key="compare_versions_select",
    )
    _cmp_col_start, _cmp_col_end = st.columns(2)
    _cmp_default_end = pd.Timestamp.now().date()
    _cmp_default_start = _cmp_default_end - pd.Timedelta(days=6)
    compare_date_start = _cmp_col_start.date_input(t("models.compare.date_start"), value=_cmp_default_start, key="compare_date_start")
    compare_date_end = _cmp_col_end.date_input(t("models.compare.date_end"), value=_cmp_default_end, key="compare_date_end")
    compare_days = max(1, (compare_date_end - compare_date_start).days + 1)

    if st.button(t("models.compare.btn_compare"), key="compare_btn", type="primary"):
        if compare_date_start > compare_date_end:
            st.warning(t("models.compare.date_order_warning"))
        else:
            versions_param = ",".join(selected_versions) if selected_versions else None
            with st.spinner(t("models.compare.spinner")):
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
                        st.info(t("models.compare.no_versions"))
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
                        # Regression if training_metrics contains mae/rmse/r2
                        is_regression = any(v.get("mae_eval") is not None for v in cmp_versions)

                        def _r(val, n=3):
                            return round(val, n) if val is not None else None

                        # Rule: display a (eval, live) pair only if eval is present
                        show = {}
                        if is_regression:
                            show["mae"]  = any(v.get("mae_eval")  is not None for v in cmp_versions)
                            show["rmse"] = any(v.get("rmse_eval") is not None for v in cmp_versions)
                            show["r2"]   = any(v.get("r2_eval")   is not None for v in cmp_versions)
                        else:
                            show["accuracy"] = any(v.get("accuracy")  is not None for v in cmp_versions)
                            show["auc"]      = any(v.get("auc") is not None or v.get("live_auc") is not None for v in cmp_versions)
                            show["f1"]       = any(v.get("f1_score")  is not None for v in cmp_versions)
                            show["brier"]    = any(v.get("brier_score") is not None for v in cmp_versions)

                        _cv = t("models.compare.col_version")
                        _cst = t("models.table.col_status")
                        _ctask = t("models.compare.col_task")
                        _cnp = t("models.compare.col_nb_pred")
                        _cns = t("models.compare.col_nb_shadow")
                        _clp50 = t("models.compare.col_lat_p50")
                        _clp95 = t("models.compare.col_lat_p95")
                        _cta = t("models.compare.col_trained_at")
                        _cd = t("models.compare.col_drift")
                        _cmae_eval = t("models.compare.col_mae_eval")
                        _cmae_live = t("models.compare.col_mae_live")
                        _crmse_eval = t("models.compare.col_rmse_eval")
                        _crmse_live = t("models.compare.col_rmse_live")
                        _cr2_eval = t("models.compare.col_r2_eval")
                        _cr2_live = t("models.compare.col_r2_live")
                        _cacc_eval = t("models.compare.col_accuracy_eval")
                        _cacc_live = t("models.compare.col_accuracy_live")
                        _cauc_eval = t("models.compare.col_auc_eval")
                        _cauc_live = t("models.compare.col_auc_live")
                        _cf1_eval = t("models.compare.col_f1_eval")
                        _cf1_live = t("models.compare.col_f1_live")
                        _cbrier = t("models.compare.col_brier")

                        for v in cmp_versions:
                            full = _full_meta.get(v["version"], v)
                            row = {
                                _cv: v["version"],
                                _cst: _statut(full),
                                _ctask: _infer_task(v),
                                _cnp: v.get("prediction_count") or 0,
                                _cns: v.get("shadow_prediction_count") or 0,
                                _clp50: _r(v.get("latency_p50_ms"), 1),
                                _clp95: _r(v.get("latency_p95_ms"), 1),
                                _cta: (
                                    pd.to_datetime(v["trained_at"]).strftime("%Y-%m-%d")
                                    if v.get("trained_at") else "—"
                                ),
                                _cd: _DRIFT_BADGE.get(v.get("drift_status") or "", "—"),
                            }
                            if is_regression:
                                if show["mae"]:
                                    row[_cmae_eval]  = _r(v.get("mae_eval"), 4)
                                    row[_cmae_live]  = _r(v.get("live_mae"), 4)
                                if show["rmse"]:
                                    row[_crmse_eval] = _r(v.get("rmse_eval"), 4)
                                    row[_crmse_live] = _r(v.get("live_rmse"), 4)
                                if show["r2"]:
                                    row[_cr2_eval]   = _r(v.get("r2_eval"), 3)
                                    row[_cr2_live]   = _r(v.get("live_r2"), 3)
                            else:
                                if show["accuracy"]:
                                    row[_cacc_eval] = _r(v.get("accuracy"))
                                    row[_cacc_live] = _r(v.get("live_accuracy"))
                                if show.get("auc"):
                                    row[_cauc_eval] = _r(v.get("auc"))
                                    row[_cauc_live] = _r(v.get("live_auc"))
                                if show["f1"]:
                                    row[_cf1_eval]  = _r(v.get("f1_score"))
                                    row[_cf1_live]  = _r(v.get("live_f1"))
                                if show["brier"]:
                                    row[_cbrier]    = _r(v.get("brier_score"), 4)
                            cmp_rows.append(row)

                        col_config = {
                            _cnp: st.column_config.NumberColumn(
                                _cnp,
                                help=t("models.compare.col_nb_pred_help", start=compare_date_start, end=compare_date_end),
                            ),
                            _cns: st.column_config.NumberColumn(
                                _cns,
                                help=t("models.compare.col_nb_shadow_help", start=compare_date_start, end=compare_date_end),
                            ),
                            _clp50: st.column_config.NumberColumn(format="%.1f"),
                            _clp95: st.column_config.NumberColumn(format="%.1f"),
                        }
                        if is_regression:
                            for _ckey, _lkey, fmt in [(_cmae_eval, _cmae_live, "%.4f"), (_crmse_eval, _crmse_live, "%.4f"), (_cr2_eval, _cr2_live, "%.3f")]:
                                col_config[_ckey] = st.column_config.NumberColumn(_ckey, help=t("models.compare.col_mae_eval_help") if _ckey == _cmae_eval else (t("models.compare.col_rmse_eval_help") if _ckey == _crmse_eval else t("models.compare.col_r2_eval_help")), format=fmt)
                                col_config[_lkey] = st.column_config.NumberColumn(_lkey, help=t("models.compare.col_mae_live_help") if _lkey == _cmae_live else (t("models.compare.col_rmse_live_help") if _lkey == _crmse_live else t("models.compare.col_r2_live_help")), format=fmt)
                        else:
                            for _ckey, _lkey, fmt in [(_cacc_eval, _cacc_live, "%.3f"), (_cf1_eval, _cf1_live, "%.3f")]:
                                col_config[_ckey] = st.column_config.NumberColumn(_ckey, help=t("models.compare.col_accuracy_eval_help") if _ckey == _cacc_eval else t("models.compare.col_f1_eval_help"), format=fmt)
                                col_config[_lkey] = st.column_config.NumberColumn(_lkey, help=t("models.compare.col_accuracy_live_help") if _lkey == _cacc_live else t("models.compare.col_f1_live_help"), format=fmt)
                            if show.get("auc"):
                                col_config[_cauc_eval] = st.column_config.NumberColumn(
                                    _cauc_eval,
                                    help=t("models.compare.col_auc_eval_help"),
                                    format="%.3f",
                                )
                                col_config[_cauc_live] = st.column_config.NumberColumn(
                                    _cauc_live,
                                    help=t("models.compare.col_auc_live_help"),
                                    format="%.3f",
                                )
                            if show.get("brier"):
                                col_config[_cbrier] = st.column_config.NumberColumn(
                                    _cbrier,
                                    help=t("models.compare.col_brier_help"),
                                    format="%.4f",
                                )

                        st.dataframe(
                            pd.DataFrame(cmp_rows),
                            width='stretch',
                            hide_index=True,
                            column_config=col_config,
                        )
                        st.caption(
                            t("models.compare.caption", compared_at=pd.to_datetime(cmp['compared_at']).strftime('%Y-%m-%d %H:%M'), start=compare_date_start, end=compare_date_end)
                        )

                        # ── Multi-version ROC curve (classification with probabilities) ──
                        if not is_regression:
                            import plotly.graph_objects as _go

                            roc_data = {}
                            for _v in cmp_versions:
                                try:
                                    _perf = fetch_model_performance(
                                        st.session_state.get("api_url", ""),
                                        st.session_state.get("api_token", ""),
                                        compare_name,
                                        _v["version"],
                                        period_days=compare_days,
                                    )
                                    if (
                                        _perf
                                        and _perf.get("roc_curve_fpr")
                                        and _perf.get("roc_curve_tpr")
                                    ):
                                        roc_data[_v["version"]] = {
                                            "fpr": _perf["roc_curve_fpr"],
                                            "tpr": _perf["roc_curve_tpr"],
                                            "auc": _perf.get("auc"),
                                        }
                                except Exception:
                                    pass

                            if roc_data:
                                st.markdown(t("models.compare.roc_title"))
                                st.caption(t("models.compare.roc_caption", start=compare_date_start, end=compare_date_end))
                                fig_roc = _go.Figure()
                                fig_roc.add_trace(
                                    _go.Scatter(
                                        x=[0, 1],
                                        y=[0, 1],
                                        mode="lines",
                                        line=dict(dash="dash", color="gray", width=1),
                                        name=t("models.compare.roc_random"),
                                    )
                                )
                                for _ver, _d in sorted(roc_data.items()):
                                    _auc_lbl = (
                                        f" — AUC {_d['auc']:.3f}"
                                        if _d["auc"] is not None
                                        else ""
                                    )
                                    fig_roc.add_trace(
                                        _go.Scatter(
                                            x=_d["fpr"],
                                            y=_d["tpr"],
                                            mode="lines",
                                            name=f"v{_ver}{_auc_lbl}",
                                            line=dict(width=2),
                                        )
                                    )
                                fig_roc.update_layout(
                                    xaxis_title="False Positive Rate (FPR)",
                                    yaxis_title="True Positive Rate (TPR)",
                                    xaxis=dict(range=[0, 1]),
                                    yaxis=dict(range=[0, 1]),
                                    legend=dict(
                                        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
                                    ),
                                    margin=dict(l=40, r=20, t=60, b=40),
                                    height=420,
                                )
                                st.plotly_chart(fig_roc, width="stretch")
                            else:
                                st.info(t("models.compare.roc_unavailable"))
                except Exception as e:
                    st.error(t("models.compare.compare_error", error=e))

st.divider()
st.subheader(t("models.detail.subheader"))

model_options = {f"{m['name']} v{m['version']}": m for m in models}
_detail_col_search, _detail_col_select = st.columns([1, 2])
with _detail_col_search:
    detail_search = st.text_input(
        t("models.detail.filter_label"), key="detail_search", placeholder=t("models.detail.filter_placeholder")
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
with _detail_col_select:
    selected_label = st.selectbox(t("models.detail.select_label"), _detail_keys, index=_detail_idx)
selected = model_options[selected_label]

# Details
_detail_tm = selected.get("training_metrics") or {}
_detail_is_regression = any(k in _detail_tm for k in ("mae", "rmse", "r2"))

with st.expander(t("models.details_expander"), expanded=True):
    col_l, col_r = st.columns(2)
    # Pre-loading downloadable resources (cached 5 min per version)
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
        st.markdown(t("models.detail_name", name=selected.get('name')))
        st.markdown(t("models.detail_version", version=selected.get('version')))
        st.markdown(t("models.detail_description", value=selected.get('description') or '—'))
        st.markdown(t("models.detail_algorithm", value=selected.get('algorithm') or '—'))
        if _ds and not _csv_bytes:
            st.markdown(t("models.detail_training_dataset", value=_ds))
        st.markdown(t("models.detail_trained_by", value=selected.get('trained_by') or '—'))
        parent_v = selected.get("parent_version")
        if parent_v:
            st.markdown(t("models.detail_derived_from", version=parent_v))
        tags = selected.get("tags")
        if tags:
            _tag_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            _tag_html = " ".join(
                f'<span style="background:{_tag_colors[i % len(_tag_colors)]};color:white;'
                f'padding:2px 10px;border-radius:12px;font-size:0.82em;font-weight:600;'
                f'white-space:nowrap;">{_tg}</span>'
                for i, _tg in enumerate(tags)
            )
            st.markdown(t("models.detail_tags", tags=_tag_html), unsafe_allow_html=True)
        else:
            st.markdown(t("models.detail_tags_empty"))
        webhook = selected.get("webhook_url")
        st.markdown(t("models.detail_webhook", url=webhook) if webhook else t("models.detail_webhook_empty"))

        mlflow_id = selected.get("mlflow_run_id")
        if mlflow_id:
            mlflow_link = f"{MLFLOW_PUBLIC_URL}/#/experiments/0/runs/{mlflow_id}"
            st.markdown(t("models.detail_mlflow", run_id=mlflow_id, link=mlflow_link))
        else:
            st.markdown(t("models.detail_mlflow_empty"))

        if minio_key:
            st.markdown(t("models.detail_minio", bucket=selected.get('minio_bucket'), minio_key=minio_key))
            if size:
                st.markdown(t("models.detail_file_size", size=f"{size / 1024:.1f}"))

    with col_r:
        # ── Deployment status ─────────────────────────────────────────────────
        _d_mode   = selected.get("deployment_mode")
        _d_prod   = selected.get("is_production")
        _d_active = selected.get("is_active")
        _d_weight = selected.get("traffic_weight")

        if _d_mode == "ab_test":
            _badge_color = "#e67e00"
            _badge_label = t("models.badge_ab_test", weight=f"{_d_weight:.0%}") if _d_weight is not None else t("models.badge_ab_test_no_weight")
        elif _d_mode == "shadow":
            _badge_color = "#7c3aed"
            _badge_label = t("models.badge_shadow")
        elif _d_prod:
            _badge_color = "#1a7f37"
            _badge_label = t("models.badge_production")
        elif _d_active:
            _badge_color = "#0ea5e9"
            _badge_label = t("models.badge_active")
        else:
            _badge_color = "#6b7280"
            _badge_label = t("models.badge_inactive")

        st.markdown(
            f'<div style="display:inline-block;background:{_badge_color};color:white;'
            f'padding:4px 14px;border-radius:16px;font-size:0.9em;font-weight:600;'
            f'margin-bottom:8px">{_badge_label}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(t("models.detail_nb_features", value=selected.get('features_count') or '—'))
        last_seen = selected.get("last_seen")
        st.markdown(
            t("models.detail_last_pred", value=pd.to_datetime(last_seen).strftime('%Y-%m-%d %H:%M') if last_seen else '—')
        )
        classes = selected.get("classes")
        if not _detail_is_regression or classes:
            st.markdown(t("models.detail_classes", value=classes if classes else '—'))
        ct = selected.get("confidence_threshold")
        if not _detail_is_regression or ct is not None:
            st.markdown(t("models.detail_confidence_threshold", value=f'{ct:.2f}' if ct is not None else '—'))

        hp = selected.get("hyperparameters")
        if hp:
            st.markdown(t("models.detail_hyperparams"))
            _col_param = t("models.detail_param_col")
            _col_val = t("models.detail_value_col")
            st.dataframe(
                pd.DataFrame(
                    [{_col_param: k, _col_val: str(v)} for k, v in hp.items()],
                ),
                width='stretch',
                hide_index=True,
            )
        else:
            st.markdown(t("models.detail_hyperparams_empty"))

    # ── 4 buttons aligned at the bottom of the expander ──────────────────────
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
                    t("models.btn_download_dataset"),
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
                    t("models.btn_download_script"),
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
                    t("models.btn_hide_script") if _is_visible else t("models.btn_view_script"),
                    width='stretch',
                    key=f"toggle_script_{selected['name']}_{selected['version']}",
                ):
                    st.session_state[_show_key] = not _is_visible
                    st.rerun()
            _btn_idx += 1
        if _pkl_bytes:
            with _btn_cols[_btn_idx]:
                st.download_button(
                    t("models.btn_download_model", size=_pkl_size_label),
                    data=_pkl_bytes,
                    file_name=f"{selected['name']}_{selected['version']}.joblib",
                    mime="application/octet-stream",
                    width='stretch',
                    key=f"dl_pkl_{selected['name']}_{selected['version']}",
                )

    _show_script_key = f"show_train_script_{selected['name']}_{selected['version']}"
    if st.session_state.get(_show_script_key) and _script_bytes:
        st.code(_script_bytes.decode("utf-8", errors="replace"), language="python", line_numbers=True)

# ── Metrics in 2 side-by-side blocks ─────────────────────────────────────────
_tm = selected.get("training_metrics") or {}
_is_regression = any(k in _tm for k in ("mae", "rmse", "r2"))

# ── Analysis & Monitoring (metrics, SHAP, performance, drift) ────────────────
with st.expander(t("models.analysis.expander"), expanded=False):
    from datetime import date as _date, timedelta as _td

    _today = _date.today()
    _dcol1, _dcol2 = st.columns(2)
    _date_debut = _dcol1.date_input(
        t("models.analysis.date_start"),
        value=_today - _td(days=30),
        max_value=_today,
        key=f"ana_date_debut_{selected['name']}_{selected['version']}",
    )
    _date_fin = _dcol2.date_input(
        t("models.analysis.date_end"),
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
            t("models.analysis.btn_load"),
            key=f"btn_ana_{selected['name']}_{selected['version']}",
            type="primary",
            width='stretch',
        ):
            st.session_state[_ana_key] = True
            st.rerun()
        _gcol2.caption(t("models.analysis.btn_load_caption"))

    # ── Metrics ───────────────────────────────────────────────────────────────
    if _ana_loaded:
        st.divider()
        st.markdown(t("models.analysis.metrics_header"))

        # Fetch ground truth
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

        # Build comparison table: training vs production
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

        _col_metric = t("models.analysis.col_metric")
        _col_training = t("models.analysis.col_training")
        _col_production = t("models.analysis.col_production")
        _col_delta = t("models.analysis.col_delta")

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
            _table_rows.append({_col_metric: _label, _col_training: _train_str, _col_production: _gt_str, _col_delta: _delta_str})

        st.dataframe(
            pd.DataFrame(_table_rows),
            width='stretch',
            hide_index=True,
            column_config={
                _col_metric: st.column_config.TextColumn(_col_metric),
                _col_training: st.column_config.TextColumn(
                    _col_training,
                    help=t("models.analysis.col_training_help"),
                ),
                _col_production: st.column_config.TextColumn(
                    _col_production,
                    help=t("models.analysis.col_production_help", start=_date_debut, end=_date_fin, n=_gt_n),
                ),
                _col_delta: st.column_config.TextColumn(
                    _col_delta,
                    help=t("models.analysis.col_delta_help"),
                ),
            },
        )
        if _gt_n == 0:
            st.caption(t("models.analysis.no_ground_truth"))

        try:
            md_content = client.get_model_card(selected["name"], selected["version"], format="markdown")
            st.download_button(
                label=t("models.analysis.export_model_card_btn"),
                data=md_content,
                file_name=f"{selected['name']}_{selected['version']}_model_card.md",
                mime="text/markdown",
                key="dl_model_card",
            )
        except Exception as e:
            st.warning(t("models.analysis.model_card_error", error=e))

    # ── Feature importance (aggregated SHAP) ──────────────────────────────────
    if _ana_loaded:
        st.divider()
        st.markdown(t("models.analysis.shap_header"))
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
            t("models.analysis.shap_slider_label", total=_shap_total),
            min_value=10,
            max_value=_shap_max,
            value=_shap_default,
            step=max(1, _shap_max // 50),
            key=f"fi_last_n_slider_{selected['name']}_{selected['version']}",
            help=t("models.analysis.shap_slider_help"),
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
                st.info(t("models.analysis.shap_no_data"))
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
                        title=t("models.analysis.shap_chart_title", n=len(fi_df), model=fi_data['model_name'], version=fi_data['version']),
                    )
                    fig.update_layout(
                        yaxis_title="",
                        xaxis_title=t("models.analysis.shap_xaxis"),
                        margin={"l": 10, "r": 10, "t": 40, "b": 10},
                    )
                    st.plotly_chart(fig, width='stretch')
                except ImportError:
                    st.bar_chart(fi_df.set_index("Feature")["Importance SHAP"])
                st.caption(t("models.analysis.shap_caption", n=sample_size, start=_date_debut, end=_date_fin, version=fi_data.get('version')))
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 422:
                st.info(t("models.analysis.shap_error_422"))
            elif status == 404:
                st.error(t("models.analysis.shap_error_404"))
            else:
                st.warning(t("models.analysis.shap_error_generic", error=e))

    # ── Performance metrics (confusion matrix + per-class metrics) ────────────
    if _ana_loaded:
        st.divider()
        st.markdown(t("models.analysis.perf_header"))
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
                st.info(t("models.analysis.perf_no_data"))
            elif model_type != "classification":
                col_mae, col_rmse, col_r2 = st.columns(3)
                col_mae.metric(
                    "MAE",
                    f"{perf['mae']:.4f}" if perf.get("mae") is not None else "—",
                    help=t("metrics.mae"),
                )
                col_rmse.metric(
                    "RMSE",
                    f"{perf['rmse']:.4f}" if perf.get("rmse") is not None else "—",
                    help=t("metrics.rmse"),
                )
                col_r2.metric(
                    "R²",
                    f"{perf['r2']:.4f}" if perf.get("r2") is not None else "—",
                    help=t("metrics.r2"),
                )
                st.caption(t("models.analysis.perf_n_paired", n=matched))
            else:
                col_acc, col_prec, col_rec, col_f1 = st.columns(4)
                col_acc.metric(
                    "Accuracy",
                    f"{perf['accuracy']:.3f}" if perf.get("accuracy") is not None else "—",
                    help=t("metrics.accuracy"),
                )
                col_prec.metric(
                    "Precision (w.)",
                    (
                        f"{perf['precision_weighted']:.3f}"
                        if perf.get("precision_weighted") is not None
                        else "—"
                    ),
                    help=t("metrics.precision"),
                )
                col_rec.metric(
                    "Recall (w.)",
                    (
                        f"{perf['recall_weighted']:.3f}"
                        if perf.get("recall_weighted") is not None
                        else "—"
                    ),
                    help=t("metrics.recall"),
                )
                col_f1.metric(
                    "F1 (w.)",
                    f"{perf['f1_weighted']:.3f}" if perf.get("f1_weighted") is not None else "—",
                    help=t("metrics.f1"),
                )
                st.caption(t("models.analysis.perf_n_paired", n=matched))

                # Map index → label from selected.classes
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
                        title=t("models.analysis.confusion_title"),
                        labels={"x": t("models.analysis.confusion_predicted"), "y": t("models.analysis.confusion_actual"), "color": t("models.analysis.confusion_count")},
                    )
                    fig.update_layout(
                        xaxis_title=t("models.analysis.confusion_predicted"),
                        yaxis_title=t("models.analysis.confusion_actual"),
                        margin={"l": 10, "r": 10, "t": 40, "b": 10},
                    )
                    st.plotly_chart(fig, width='stretch')

                per_class = perf.get("per_class_metrics")
                if per_class:
                    st.markdown(t("models.analysis.per_class_header"))
                    _col_class = t("models.analysis.col_class")
                    _col_prec = t("models.analysis.col_precision")
                    _col_rec = t("models.analysis.col_recall")
                    _col_f1c = t("models.analysis.col_f1")
                    _col_supp = t("models.analysis.col_support")
                    pc_rows = [
                        {
                            _col_class: _cls_label(label),
                            _col_prec: f"{m['precision']:.3f}",
                            _col_rec: f"{m['recall']:.3f}",
                            _col_f1c: f"{m['f1_score']:.3f}",
                            _col_supp: m["support"],
                        }
                        for label, m in per_class.items()
                    ]
                    st.dataframe(
                        pd.DataFrame(pc_rows),
                        width='stretch',
                        hide_index=True,
                        column_config={
                            _col_class: st.column_config.TextColumn(
                                _col_class,
                                help=t("models.analysis.col_class_help"),
                            ),
                            _col_prec: st.column_config.TextColumn(
                                _col_prec,
                                help=t("models.analysis.col_precision_help"),
                            ),
                            _col_rec: st.column_config.TextColumn(
                                _col_rec,
                                help=t("models.analysis.col_recall_help"),
                            ),
                            _col_f1c: st.column_config.TextColumn(
                                _col_f1c,
                                help=t("models.analysis.col_f1_help"),
                            ),
                            _col_supp: st.column_config.NumberColumn(
                                _col_supp,
                                help=t("models.analysis.col_support_help"),
                            ),
                        },
                    )

        except Exception as e:
            st.warning(t("models.analysis.perf_error", error=e))

    # ── Output drift (label shift) ────────────────────────────────────────────
    if _ana_loaded:
        st.divider()
        st.markdown(t("models.analysis.output_drift_header"))
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
                "no_baseline": t("models.analysis.output_drift_od_no_baseline"),
                "insufficient_data": t("models.analysis.output_drift_od_insufficient"),
            }
            st.markdown(t("models.analysis.output_drift_status", status=_OD_BADGE.get(od_status, od_status)))

            if od_status in ("no_baseline", "insufficient_data"):
                if od_status == "no_baseline":
                    st.info(t("models.analysis.output_drift_no_baseline"))
                else:
                    st.info(t("models.analysis.output_drift_insufficient", n=od.get('predictions_analyzed', 0), start=_date_debut, end=_date_fin))
            else:
                col_psi, col_n = st.columns(2)
                psi_val = od.get("psi")
                _col_psi_metric = t("models.analysis.output_drift_psi")
                _col_n_analyzed = t("models.analysis.output_drift_n_analyzed")
                col_psi.metric(
                    _col_psi_metric,
                    f"{psi_val:.4f}" if psi_val is not None else "—",
                    help=t("models.analysis.output_drift_psi_help"),
                )
                col_n.metric(_col_n_analyzed, od.get("predictions_analyzed", 0))

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
                    st.markdown(t("models.analysis.output_drift_by_class_header"))
                    _odc_class = t("models.analysis.output_drift_col_class")
                    _odc_baseline = t("models.analysis.output_drift_col_baseline")
                    _odc_current = t("models.analysis.output_drift_col_current")
                    _odc_delta = t("models.analysis.output_drift_col_delta")
                    bc_rows = [
                        {
                            _odc_class: _od_label(row["label"]),
                            _odc_baseline: f"{row['baseline_ratio']:.3f}",
                            _odc_current: f"{row['current_ratio']:.3f}",
                            _odc_delta: f"{row['delta']:+.3f}",
                        }
                        for row in by_class
                    ]
                    st.dataframe(
                        pd.DataFrame(bc_rows),
                        width='stretch',
                        hide_index=True,
                        column_config={
                            _odc_class: st.column_config.TextColumn(
                                _odc_class,
                                help=t("models.analysis.output_drift_col_class_help"),
                            ),
                            _odc_baseline: st.column_config.TextColumn(
                                _odc_baseline,
                                help=t("models.analysis.output_drift_col_baseline_help"),
                            ),
                            _odc_current: st.column_config.TextColumn(
                                _odc_current,
                                help=t("models.analysis.output_drift_col_current_help"),
                            ),
                            _odc_delta: st.column_config.TextColumn(
                                _odc_delta,
                                help=t("models.analysis.output_drift_col_delta_help"),
                            ),
                        },
                    )

                    baseline_dist = od.get("baseline_distribution") or {}
                    current_dist = od.get("current_distribution") or {}
                    raw_labels = [row["label"] for row in by_class]
                    named_labels = [_od_label(l) for l in raw_labels]
                    df_bar = pd.DataFrame(
                        {
                            t("models.analysis.output_drift_col_class"): named_labels + named_labels,
                            "Ratio": [baseline_dist.get(str(l), 0.0) for l in raw_labels]
                            + [current_dist.get(str(l), 0.0) for l in raw_labels],
                            "Source": [t("models.analysis.output_drift_col_baseline")] * len(raw_labels) + [t("models.analysis.output_drift_col_current")] * len(raw_labels),
                        }
                    )
                    fig_od = px.bar(
                        df_bar,
                        x=t("models.analysis.output_drift_col_class"),
                        y="Ratio",
                        color="Source",
                        barmode="group",
                        title=t("models.analysis.output_drift_chart_title"),
                        color_discrete_map={t("models.analysis.output_drift_col_baseline"): "#636EFA", t("models.analysis.output_drift_col_current"): "#EF553B"},
                    )
                    fig_od.update_layout(yaxis_tickformat=".0%", yaxis_title="Proportion")
                    st.plotly_chart(fig_od, width='stretch')
        except Exception as e:
            st.warning(t("models.analysis.output_drift_error", error=e))

    # ── Feature drift (inputs) ────────────────────────────────────────────────
    if _ana_loaded:
        st.divider()
        st.markdown(t("models.analysis.feature_drift_header"))
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
                "no_data": "⬜ no data",
                "no_baseline": "⬜ no baseline",
                "insufficient_data": "⬜ insufficient data",
            }
            fd_summary = fd.get("drift_summary", "no_data")
            fd_baseline = fd.get("baseline_available", False)
            fd_analyzed = fd.get("predictions_analyzed", 0)

            _fd_features_raw = fd.get("features", {})
            _fd_n_preds = (
                next(iter(_fd_features_raw.values()), {}).get("production_count")
                or fd_analyzed
            )
            st.markdown(t("models.analysis.feature_drift_global_status", status=_FD_BADGE.get(fd_summary, fd_summary)))
            st.caption(t("models.analysis.feature_drift_n_analyzed", n=_fd_n_preds, start=_fd_start, end=_fd_end))

            features_dict = fd.get("features", {})
            if not fd_baseline:
                st.info(t("models.analysis.feature_drift_no_baseline"))
            elif not features_dict:
                st.info(t("models.analysis.feature_drift_no_data"))
            else:
                _STATUS_COLOR = {"ok": "🟢", "warning": "🟡", "critical": "🔴", "no_data": "⬜"}
                _fdc_feature = t("models.analysis.feature_drift_col_feature")
                _fdc_status = t("models.analysis.feature_drift_col_status")
                _fdc_prod_mean = t("models.analysis.feature_drift_col_prod_mean")
                _fdc_bl_mean = t("models.analysis.feature_drift_col_baseline_mean")
                _fdc_zscore = t("models.analysis.feature_drift_col_zscore")
                _fdc_psi = t("models.analysis.feature_drift_col_psi")
                _fdc_n_prod = t("models.analysis.feature_drift_col_n_prod")
                rows_fd = []
                for feat_name, feat_data in features_dict.items():
                    ds = feat_data.get("drift_status", "no_data")
                    rows_fd.append(
                        {
                            _fdc_feature: feat_name,
                            _fdc_status: _STATUS_COLOR.get(ds, "⬜") + f" {ds}",
                            _fdc_prod_mean: (
                                round(feat_data["production_mean"], 4)
                                if feat_data.get("production_mean") is not None
                                else "—"
                            ),
                            _fdc_bl_mean: (
                                round(feat_data["baseline_mean"], 4)
                                if feat_data.get("baseline_mean") is not None
                                else "—"
                            ),
                            _fdc_zscore: (
                                round(feat_data["z_score"], 3)
                                if feat_data.get("z_score") is not None
                                else "—"
                            ),
                            _fdc_psi: (
                                round(feat_data["psi"], 4)
                                if feat_data.get("psi") is not None
                                else "—"
                            ),
                            _fdc_n_prod: feat_data.get("production_count", 0),
                        }
                    )
                df_fd = pd.DataFrame(rows_fd)
                st.dataframe(
                    df_fd,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        _fdc_zscore: st.column_config.TextColumn(
                            _fdc_zscore,
                            help=t("models.analysis.feature_drift_col_zscore_help"),
                        ),
                        _fdc_psi: st.column_config.TextColumn(
                            _fdc_psi,
                            help=t("models.analysis.feature_drift_col_psi_help"),
                        ),
                        _fdc_prod_mean: st.column_config.TextColumn(
                            _fdc_prod_mean,
                            help=t("models.analysis.feature_drift_col_prod_mean_help"),
                        ),
                        _fdc_bl_mean: st.column_config.TextColumn(
                            _fdc_bl_mean,
                            help=t("models.analysis.feature_drift_col_baseline_mean_help"),
                        ),
                        _fdc_status: st.column_config.TextColumn(
                            _fdc_status,
                            help=t("models.analysis.feature_drift_col_status_help"),
                        ),
                        _fdc_n_prod: st.column_config.NumberColumn(
                            _fdc_n_prod,
                            help=t("models.analysis.feature_drift_col_n_prod_help"),
                        ),
                    },
                )

                _color_map = {"ok": "#2ECC71", "warning": "#F39C12", "critical": "#E74C3C", "no_data": "#95A5A6"}

                _rows_z = [r for r in rows_fd if r[_fdc_zscore] != "—"]
                if _rows_z:
                    _df_z = pd.DataFrame(
                        {
                            "Feature": [r[_fdc_feature] for r in _rows_z],
                            "Z-score": [abs(float(r[_fdc_zscore])) for r in _rows_z],
                            "Drift": [r[_fdc_status].split(" ", 1)[-1].strip() for r in _rows_z],
                        }
                    ).sort_values("Z-score", ascending=True)
                    fig_z = px.bar(
                        _df_z, x="Z-score", y="Feature", color="Drift", orientation="h",
                        title=t("models.analysis.feature_drift_zscore_chart_title"),
                        color_discrete_map=_color_map,
                    )
                    fig_z.add_vline(x=2.0, line_dash="dash", line_color="#F39C12",
                                    annotation_text=t("models.analysis.feature_drift_warning_2sigma"), annotation_position="top right")
                    fig_z.add_vline(x=3.0, line_dash="dash", line_color="#E74C3C",
                                    annotation_text=t("models.analysis.feature_drift_critical_3sigma"), annotation_position="top right")
                    fig_z.update_layout(height=max(250, len(_rows_z) * 32), yaxis_title="")
                    st.plotly_chart(fig_z, width='stretch')

                _rows_psi = [r for r in rows_fd if r[_fdc_psi] != "—"]
                if _rows_psi:
                    _df_psi = pd.DataFrame(
                        {
                            "Feature": [r[_fdc_feature] for r in _rows_psi],
                            "PSI": [float(r[_fdc_psi]) for r in _rows_psi],
                            "Drift": [r[_fdc_status].split(" ", 1)[-1].strip() for r in _rows_psi],
                        }
                    ).sort_values("PSI", ascending=True)
                    fig_psi = px.bar(
                        _df_psi, x="PSI", y="Feature", color="Drift", orientation="h",
                        title=t("models.analysis.feature_drift_psi_chart_title"),
                        color_discrete_map=_color_map,
                    )
                    fig_psi.add_vline(x=0.1, line_dash="dash", line_color="#F39C12",
                                      annotation_text=t("models.analysis.feature_drift_warning_01"), annotation_position="top right")
                    fig_psi.add_vline(x=0.2, line_dash="dash", line_color="#E74C3C",
                                      annotation_text=t("models.analysis.feature_drift_critical_02"), annotation_position="top right")
                    fig_psi.update_layout(height=max(250, len(_rows_psi) * 32), yaxis_title="")
                    st.plotly_chart(fig_psi, width='stretch')

        except Exception as e:
            st.warning(t("models.analysis.feature_drift_error", error=e))

# Feature resolution for the Validate / Golden Tests blocks
# Always fetch model detail to get the full feature_names_in_ list (includes
# categorical features that are absent from feature_baseline, which only stores
# numerical stats — mean/std/min/max cannot be computed for categoricals).
feature_baseline = selected.get("feature_baseline") or {}
categorical_baseline = selected.get("categorical_baseline") or {}
feature_names_list: list = []
try:
    _feat_detail = fetch_model_detail(
        st.session_state.get("api_url"),
        st.session_state.get("api_token"),
        selected["name"],
        selected["version"],
    )
    feature_names_list = _feat_detail.get("feature_names") or []
    if not feature_baseline:
        feature_baseline = _feat_detail.get("feature_baseline") or {}
    if not categorical_baseline:
        categorical_baseline = _feat_detail.get("categorical_baseline") or {}
except Exception:
    pass

if not feature_names_list:
    feature_names_list = list(feature_baseline.keys()) + [
        k for k in categorical_baseline if k not in feature_baseline
    ]


def _cat_default(feat: str) -> str:
    """Return the most frequent category for a categorical feature, or '' if unknown."""
    dist = categorical_baseline.get(feat, {})
    if not dist:
        return ""
    return max(dist, key=lambda k: dist[k])


with st.expander(t("models.validate.expander"), expanded=False):
    st.markdown(t("models.validate.intro"))

    # Build example JSON: use all feature_names from the model (feature_names_in_).
    # - Numerical features (in feature_baseline): use their stored mean value.
    # - Categorical features (in categorical_baseline): use the most frequent category.
    # - Unknown features: fall back to 0.0.
    if feature_names_list:
        example_payload = {
            feat: float(feature_baseline[feat].get("mean") or 0.0)
            if feat in feature_baseline
            else _cat_default(feat)
            for feat in feature_names_list
        }
    elif feature_baseline:
        example_payload = {
            feat: float(info.get("mean") or 0.0) for feat, info in feature_baseline.items()
        }
    else:
        example_payload = {}

    raw_json = st.text_area(
        t("models.validate.json_label"),
        value=_json.dumps(example_payload, indent=2),
        height=180,
        key=f"validate_json_{selected['name']}_{selected['version']}",
    )

    if st.button(
        t("models.validate.btn_validate"),
        key=f"validate_btn_{selected['name']}_{selected['version']}",
    ):
        try:
            parsed = _json.loads(raw_json)
        except _json.JSONDecodeError as exc:
            st.error(t("models.validate.json_invalid", error=exc))
            parsed = None

        if parsed is not None:
            with st.spinner(t("models.validate.spinner")):
                try:
                    result = client.validate_input(
                        selected["name"], selected["version"], parsed
                    )

                    if result.get("valid"):
                        st.success(t("models.validate.valid"))
                    else:
                        st.error(t("models.validate.invalid"))

                    errors = result.get("errors") or []
                    warnings = result.get("warnings") or []
                    expected = result.get("expected_features")

                    for err in errors:
                        etype = err.get("type", "")
                        feat = err.get("feature", "")
                        if etype == "missing_feature":
                            st.markdown(t("models.validate.missing_feature", feature=feat))
                        elif etype == "unexpected_feature":
                            st.markdown(t("models.validate.unexpected_feature", feature=feat))
                        else:
                            st.markdown(f"❌ `{feat}` — {etype}")

                    for warn in warnings:
                        wtype = warn.get("type", "")
                        feat = warn.get("feature", "")
                        from_t = warn.get("from_type", "")
                        to_t = warn.get("to_type", "")
                        if wtype == "type_coercion":
                            st.markdown(t("models.validate.type_coercion", feature=feat, from_type=from_t, to_type=to_t))
                        else:
                            st.markdown(f"⚠️ `{feat}` — {wtype}")

                    if expected:
                        st.markdown(t("models.validate.expected_features_header"))
                        st.markdown(" ".join(f"`{f}`" for f in expected))

                except Exception as e:
                    st.error(t("models.validate.validate_error", error=e))

    st.divider()
    st.markdown(t("models.validate.code_example_header"))
    _predict_payload = {
        "model_name": selected["name"],
        "features": example_payload,
    }
    _python_code = (
        "import requests\n\n"
        'url = "http://localhost:8000/predict"\n'
        "headers = {\n"
        '    "Authorization": "Bearer <YOUR_TOKEN>",\n'
        '    "Content-Type": "application/json",\n'
        "}\n"
        f"payload = {_json.dumps(_predict_payload, indent=4)}\n\n"
        "response = requests.post(url, headers=headers, json=payload)\n"
        "print(response.json())"
    )
    st.code(_python_code, language="python")

    st.markdown(t("models.validate.response_example_header"))
    _example_response = _json.dumps(
        {
            "model_name": selected["name"],
            "model_version": selected["version"],
            "prediction": "<result>",
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

# Regression tests (Golden Test Set)
with st.expander(t("models.golden_tests.expander"), expanded=False):
    is_admin_gt = st.session_state.get("is_admin", False)
    gt_model_name = selected["name"]
    gt_version = selected["version"]

    try:
        golden_tests = client.list_golden_tests(gt_model_name)
    except Exception:
        golden_tests = []

    # --- List of existing cases ---
    if golden_tests:
        import pandas as _pd_gt
        st.markdown(t("models.golden_tests.n_cases", n=len(golden_tests)))
        for t_item in golden_tests:
            tid = t_item["id"]
            expected = t_item.get("expected_output", "—")
            desc = t_item.get("description") or ""
            features = t_item.get("input_features") or {}
            created_at = t_item.get("created_at", "")

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
                    st.markdown(t("models.golden_tests.input_features_header"))
                    st.json(features)
                with c_meta:
                    st.markdown(t("models.golden_tests.expected_output_header"))
                    st.code(expected, language=None)
                    if desc:
                        st.markdown(f"*{desc}*")
                    if created_at:
                        try:
                            st.caption(_pd_gt.to_datetime(created_at).strftime("%Y-%m-%d %H:%M"))
                        except Exception:
                            st.caption(str(created_at)[:16])
                    if is_admin_gt:
                        if st.button(t("models.golden_tests.btn_delete"), key=f"del_gt_{tid}", type="secondary"):
                            try:
                                client.delete_golden_test(gt_model_name, tid)
                                st.toast(t("models.golden_tests.delete_toast", id=tid), icon="✅")
                                st.rerun()
                            except Exception as e:
                                st.error(t("models.golden_tests.delete_error", error=e))
    else:
        st.info(t("models.golden_tests.no_cases"))

    if is_admin_gt:
        st.divider()

        # --- Add a case ---
        with st.expander(t("models.golden_tests.add_expander"), expanded=False):
            if feature_names_list:
                _gt_default = _json.dumps(
                    {
                        feat: float(feature_baseline[feat].get("mean") or 0.0)
                        if feat in feature_baseline
                        else _cat_default(feat)
                        for feat in feature_names_list
                    },
                    indent=2,
                )
            elif feature_baseline:
                _gt_default = _json.dumps(
                    {feat: float(info.get("mean") or 0.0) for feat, info in feature_baseline.items()},
                    indent=2,
                )
            else:
                _gt_default = '{}\n  "feature1": 0.0\n}'

            with st.form("add_golden_test_form", clear_on_submit=True):
                raw_features_gt = st.text_area(
                    t("models.golden_tests.features_label"),
                    value=_gt_default,
                    height=130,
                    key="gt_features_input",
                )
                expected_gt = st.text_input(t("models.golden_tests.expected_label"), placeholder=t("models.golden_tests.expected_placeholder"), key="gt_expected")
                desc_gt = st.text_input(t("models.golden_tests.description_label"), value="", key="gt_desc")
                submitted_gt = st.form_submit_button(t("models.golden_tests.submit_btn"))

            if submitted_gt:
                if not expected_gt.strip():
                    st.error(t("models.golden_tests.error_no_expected"))
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
                        st.success(t("models.golden_tests.add_success"))
                        st.rerun()
                    except Exception as e:
                        st.error(t("models.golden_tests.add_error", error=e))

        # --- CSV Upload ---
        with st.expander(t("models.golden_tests.csv_expander"), expanded=False):
            st.caption(t("models.golden_tests.csv_caption"))
            csv_file_gt = st.file_uploader(t("models.golden_tests.csv_uploader_label"), type=["csv"], key="gt_csv_upload")
            if csv_file_gt and st.button(t("models.golden_tests.csv_import_btn"), key="import_gt_csv"):
                try:
                    result_csv = client.upload_golden_tests_csv(
                        gt_model_name, csv_file_gt.read(), csv_file_gt.name
                    )
                    st.success(t("models.golden_tests.csv_import_success", n=result_csv.get('created', 0)))
                    if result_csv.get("errors"):
                        st.warning(t("models.golden_tests.csv_import_errors", errors=result_csv['errors']))
                    st.rerun()
                except Exception as e:
                    st.error(t("models.golden_tests.csv_import_error", error=e))

    st.divider()

    # --- Run tests ---
    if st.button(t("models.golden_tests.run_btn"), key="run_golden_tests", type="primary"):
        try:
            gt_result = client.run_golden_tests(gt_model_name, gt_version)
            col_p, col_f, col_r = st.columns(3)
            col_p.metric(t("models.golden_tests.col_passed"), gt_result["passed"])
            col_f.metric(t("models.golden_tests.col_failed"), gt_result["failed"])
            col_r.metric(t("models.golden_tests.col_rate"), f"{gt_result['pass_rate']:.1%}")

            if gt_result["total_tests"] == 0:
                st.info(t("models.golden_tests.no_test_cases"))
            else:
                for d in gt_result["details"]:
                    passed = d["passed"]
                    icon = "✅" if passed else "❌"
                    desc_str = f" — {d['description']}" if d.get("description") else ""
                    label = f"{icon} #{d['test_id']}{desc_str} &nbsp; expected: `{d['expected']}` / received: `{d['actual']}`"
                    with st.expander(label, expanded=not passed):
                        c_l, c_r = st.columns(2)
                        with c_l:
                            st.markdown(t("models.golden_tests.result_expected_header"))
                            st.code(str(d.get("expected", "—")), language=None)
                        with c_r:
                            st.markdown(t("models.golden_tests.result_actual_header"))
                            if passed:
                                st.code(str(d.get("actual", "—")), language=None)
                            else:
                                st.error(str(d.get("actual", "—")))
                        if d.get("input"):
                            st.markdown(t("models.golden_tests.result_features_header"))
                            st.json(d["input"])
        except Exception as e:
            st.error(t("models.golden_tests.run_error", error=e))

# Explorateur What-if
with st.expander(t("models.whatif.expander"), expanded=False):
    _wif_api_url = st.session_state.get("api_url")
    _wif_api_token = st.session_state.get("api_token")

    wif_baseline = selected.get("feature_baseline") or {}
    wif_cat_baseline = selected.get("categorical_baseline") or {}
    wif_classes = selected.get("classes") or []
    wif_feature_names: list = []
    if not wif_baseline and not wif_cat_baseline:
        try:
            _wif_detail = fetch_model_detail(
                _wif_api_url, _wif_api_token, selected["name"], selected["version"]
            )
            wif_baseline = _wif_detail.get("feature_baseline") or {}
            wif_cat_baseline = _wif_detail.get("categorical_baseline") or {}
            wif_feature_names = _wif_detail.get("feature_names") or []
        except Exception:
            pass

    # Determine ordered feature list: model order from feature_names_in_, else baseline keys
    if not wif_feature_names:
        wif_feature_names = list(wif_baseline.keys()) + [
            k for k in wif_cat_baseline if k not in wif_baseline
        ]

    if not wif_baseline and not wif_cat_baseline:
        st.info(t("models.whatif.no_baseline"))
    else:
        _wif_key = f"whatif_history_{selected['name']}_{selected['version']}"
        if _wif_key not in st.session_state:
            st.session_state[_wif_key] = []

        _wif_n_total = len(wif_feature_names) or len(wif_baseline) + len(wif_cat_baseline)
        st.caption(t("models.whatif.caption", n=_wif_n_total))

        _wif_cols = st.columns(2)
        wif_feature_values: dict = {}
        _wif_i = 0
        for _wif_feat in wif_feature_names:
            with _wif_cols[_wif_i % 2]:
                if _wif_feat in wif_baseline:
                    # Numerical feature — slider
                    _wif_stats = wif_baseline[_wif_feat]
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
                elif _wif_feat in wif_cat_baseline:
                    # Categorical feature — selectbox sorted by frequency (most common first)
                    _wif_dist = wif_cat_baseline[_wif_feat]
                    _wif_options = sorted(_wif_dist.keys(), key=lambda c: -_wif_dist[c])
                    wif_feature_values[_wif_feat] = st.selectbox(
                        _wif_feat,
                        options=_wif_options,
                        index=0,
                        key=f"whatif_select_{selected['name']}_{selected['version']}_{_wif_feat}",
                    )
            _wif_i += 1

        wif_use_shap = st.checkbox(
            t("models.whatif.shap_checkbox"),
            value=True,
            key=f"whatif_shap_{selected['name']}_{selected['version']}",
        )

        if st.button(
            t("models.whatif.btn_predict"),
            key=f"whatif_btn_{selected['name']}_{selected['version']}",
            type="primary",
        ):
            with st.spinner(t("models.whatif.spinner")):
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

                    # Resolve index → label if the model returns an integer
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

                    _wc1.metric(t("models.whatif.metric_prediction"), _wif_pred_label)
                    if _wif_probs:
                        _wc2.metric(t("models.whatif.metric_max_prob"), f"{max(_wif_probs):.2%}")
                    if wif_result.get("low_confidence"):
                        _wc3.warning(t("models.whatif.low_confidence"))

                    if _wif_probs:
                        st.markdown(t("models.whatif.class_probs_header"))
                        _wif_col_class = t("models.whatif.col_class")
                        _wif_col_prob = t("models.whatif.col_probability")
                        if wif_classes and len(wif_classes) == len(_wif_probs):
                            _wif_prob_rows = [
                                {_wif_col_class: str(c), _wif_col_prob: f"{p:.4f}"}
                                for c, p in zip(wif_classes, _wif_probs)
                            ]
                        else:
                            _wif_prob_rows = [
                                {_wif_col_class: t("models.whatif.col_class_i", i=i), _wif_col_prob: f"{p:.4f}"}
                                for i, p in enumerate(_wif_probs)
                            ]
                        st.dataframe(
                            pd.DataFrame(_wif_prob_rows),
                            width='stretch',
                            hide_index=True,
                        )

                    _wif_shap = wif_result.get("shap_values")
                    if wif_use_shap and _wif_shap:
                        st.markdown(t("models.whatif.shap_header"))
                        try:
                            import plotly.express as px

                            _wif_shap_df = pd.DataFrame(
                                [
                                    {
                                        "Feature": f,
                                        "Contribution": v,
                                        "Signe": t("models.whatif.shap_positive") if v >= 0 else t("models.whatif.shap_negative"),
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
                                    t("models.whatif.shap_positive"): "#e74c3c",
                                    t("models.whatif.shap_negative"): "#3498db",
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
                        st.info(t("models.whatif.shap_unavailable"))

                except Exception as e:
                    st.error(t("models.whatif.predict_error", error=e))

        if st.session_state[_wif_key]:
            st.divider()
            st.markdown(t("models.whatif.history_header"))
            _wif_feat_options = list(wif_baseline.keys())
            _wif_sel_feat = st.selectbox(
                t("models.whatif.feature_select_label"),
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
                _wif_y_label = t("models.whatif.y_label_prob") if _wif_has_probs else t("models.whatif.y_label_pred")
                try:
                    import plotly.express as px

                    _wif_evo_fig = px.line(
                        _wif_chart_df,
                        x="x",
                        y="y",
                        markers=True,
                        labels={"x": _wif_sel_feat, "y": _wif_y_label},
                        title=t("models.whatif.evolution_title", feature=_wif_sel_feat),
                    )
                    st.plotly_chart(_wif_evo_fig, width='stretch')
                except ImportError:
                    st.line_chart(_wif_chart_df.set_index("x")["y"])

            if st.button(
                t("models.whatif.btn_clear_history"),
                key=f"whatif_clear_{selected['name']}_{selected['version']}",
            ):
                st.session_state[_wif_key] = []
                st.rerun()

# Actions
if is_admin:
    with st.expander(t("models.admin.expander"), expanded=True):
        col_p, col_d = st.columns(2)

        # Readiness checklist
        try:
            readiness = client.get_model_readiness(selected["name"], selected["version"])
            checks = readiness.get("checks", {})
            check_labels = {
                "file_accessible": t("models.admin.check_file_accessible"),
                "baseline_computed": t("models.admin.check_baseline_computed"),
                "no_critical_drift": t("models.admin.check_no_critical_drift"),
                "is_production": t("models.admin.check_is_production"),
            }
            st.markdown(t("models.admin.readiness_header"))
            for key, label in check_labels.items():
                check = checks.get(key, {})
                passed = check.get("pass", False)
                detail = check.get("detail")
                icon = "✅" if passed else "❌"
                suffix = f" — `{detail}`" if detail else ""
                st.markdown(f"{icon} {label}{suffix}")
            ready = readiness.get("ready", False)
        except Exception:
            ready = True  # do not block if the endpoint fails

        # Promote to production
        if not selected.get("is_production"):
            promote_help = None if ready else t("models.admin.btn_promote_help")
            if col_p.button(
                t("models.admin.btn_promote"),
                width='stretch',
                type="primary",
                disabled=not ready,
                help=promote_help,
                key=f"promote_btn_{selected['name']}_{selected['version']}",
            ):
                try:
                    client.update_model(selected["name"], selected["version"], {"is_production": True})
                    st.toast(
                        t("models.admin.promote_toast", name=selected['name'], version=selected['version']),
                        icon="✅",
                    )
                    reload()
                except Exception as e:
                    st.toast(t("models.admin.promote_error", error=e), icon="❌")
        else:
            col_p.info(t("models.admin.already_production"))

        # Cache warm-up
        selected_cache_key = f"{selected['name']}:{selected['version']}"
        is_selected_cached = selected_cache_key in cached_model_keys
        col_w1, col_w2 = st.columns([1, 3])
        if is_selected_cached:
            col_w1.success(t("models.admin.cache_hot"))
        else:
            col_w1.warning(t("models.admin.cache_cold"))
        if not is_selected_cached:
            if col_w2.button(
                t("models.admin.btn_warmup"),
                width='stretch',
                key="warmup_btn",
                help=t("models.admin.btn_warmup_help"),
            ):
                with st.spinner(t("models.admin.warmup_spinner")):
                    try:
                        result = client.warmup_model(selected["name"], selected["version"])
                        st.toast(
                            t("models.admin.warmup_toast", ms=result['load_time_ms'], key=result['cache_key']),
                            icon="✅",
                        )
                        reload()
                    except Exception as e:
                        st.toast(t("models.admin.warmup_error", error=e), icon="❌")
        else:
            col_w2.info(t("models.admin.cache_already_loaded"))

        # Delete
        if col_d.button(t("models.admin.btn_delete"), width='stretch', type="secondary"):
            st.session_state["confirm_delete_model"] = f"{selected['name']}:{selected['version']}"

        key = f"{selected['name']}:{selected['version']}"
        if st.session_state.get("confirm_delete_model") == key:
            st.warning(t("models.admin.confirm_delete", name=selected['name'], version=selected['version']))
            c1, c2 = st.columns(2)
            if c1.button(t("models.admin.confirm_yes"), type="primary"):
                try:
                    client.delete_model_version(selected["name"], selected["version"])
                    st.toast(t("models.admin.delete_toast"), icon="✅")
                    st.session_state.pop("confirm_delete_model", None)
                    reload()
                except Exception as e:
                    st.toast(t("models.admin.delete_error", error=e), icon="❌")
            if c2.button(t("models.admin.confirm_cancel")):
                st.session_state.pop("confirm_delete_model", None)
                st.rerun()

        # Edit metadata (all PATCH /models/{name}/{version} fields)
        with st.expander(t("models.edit_meta.expander")):

            # ── Description & information ──────────────────────────────────────
            st.markdown(t("models.edit_meta.section_description"))
            _col_desc, _col_ds = st.columns(2)
            with _col_desc:
                new_description = st.text_area(
                    t("models.edit_meta.description_label"),
                    value=selected.get("description") or "",
                    placeholder=t("models.edit_meta.description_placeholder"),
                    height=80,
                )
            with _col_ds:
                new_training_dataset = st.text_input(
                    t("models.edit_meta.dataset_label"),
                    value=selected.get("training_dataset") or "",
                    placeholder=t("models.edit_meta.dataset_placeholder"),
                    help=t("models.edit_meta.dataset_help"),
                )

            # ── Tags & Webhook ─────────────────────────────────────────────────
            st.markdown(t("models.edit_meta.section_tags"))
            _col_tags, _col_wh = st.columns(2)
            with _col_tags:
                new_tags_raw = st.text_input(
                    t("models.edit_meta.tags_label"),
                    value=", ".join(selected.get("tags") or []),
                    placeholder=t("models.edit_meta.tags_placeholder"),
                )
            with _col_wh:
                new_webhook = st.text_input(
                    t("models.edit_meta.webhook_label"),
                    value=selected.get("webhook_url") or "",
                    placeholder=t("models.edit_meta.webhook_placeholder"),
                )

            # ── Inference ─────────────────────────────────────────────────────
            st.markdown(t("models.edit_meta.section_inference"))
            _cur_ct = selected.get("confidence_threshold")
            new_confidence_threshold = st.slider(
                t("models.edit_meta.confidence_label"),
                min_value=0.0,
                max_value=1.0,
                value=float(_cur_ct) if _cur_ct is not None else 0.5,
                step=0.01,
                help=t("models.edit_meta.confidence_help"),
            )

            # ── A/B / Shadow deployment ───────────────────────────────────────
            st.markdown(t("models.edit_meta.section_deployment"))
            _deploy_opts = ["production", "ab_test", "shadow"]
            _cur_deploy = selected.get("deployment_mode") or "production"
            _deploy_idx = _deploy_opts.index(_cur_deploy) if _cur_deploy in _deploy_opts else 0
            new_deploy_mode = st.selectbox(
                t("models.edit_meta.deploy_mode_label"),
                _deploy_opts,
                index=_deploy_idx,
                key="deploy_mode_select",
                format_func=lambda m: {
                    "production": t("models.edit_meta.deploy_production"),
                    "ab_test": t("models.edit_meta.deploy_ab_test"),
                    "shadow": t("models.edit_meta.deploy_shadow"),
                }.get(m, m),
                help=t("models.edit_meta.deploy_mode_help"),
            )
            new_traffic_weight = None
            if new_deploy_mode == "ab_test":
                new_traffic_weight = st.number_input(
                    t("models.edit_meta.traffic_weight_label"),
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    value=float(selected.get("traffic_weight") or 0.5),
                    key="traffic_weight_input",
                    help=t("models.edit_meta.traffic_weight_help"),
                )

            # ── Alert thresholds ──────────────────────────────────────────────
            st.markdown(t("models.edit_meta.section_alerts"))
            st.caption(t("models.edit_meta.alerts_caption"))
            _cur_at = selected.get("alert_thresholds") or {}
            _col_at1, _col_at2, _col_at3 = st.columns(3)
            with _col_at1:
                _acc_min_cur = _cur_at.get("accuracy_min")
                new_accuracy_min = st.number_input(
                    t("models.edit_meta.accuracy_min_label"),
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    value=float(_acc_min_cur) if _acc_min_cur is not None else 0.0,
                    help=t("models.edit_meta.accuracy_min_help"),
                )
            with _col_at2:
                _err_max_cur = _cur_at.get("error_rate_max")
                new_error_rate_max = st.number_input(
                    t("models.edit_meta.error_rate_max_label"),
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    value=float(_err_max_cur) if _err_max_cur is not None else 0.0,
                    help=t("models.edit_meta.error_rate_max_help"),
                )
            with _col_at3:
                new_drift_auto_alert = st.checkbox(
                    t("models.edit_meta.drift_auto_alert_label"),
                    value=bool(_cur_at.get("drift_auto_alert", True)),
                    help=t("models.edit_meta.drift_auto_alert_help"),
                )

            # ── Save ──────────────────────────────────────────────────────────
            if st.button(t("models.edit_meta.btn_save"), key="save_meta"):
                patch = {}

                # Description
                new_desc = new_description.strip() or None
                if new_desc != (selected.get("description") or None):
                    patch["description"] = new_desc

                # Training dataset
                new_td = new_training_dataset.strip() or None
                if new_td != (selected.get("training_dataset") or None):
                    patch["training_dataset"] = new_td

                # Webhook
                _cur_wh = selected.get("webhook_url") or ""
                if new_webhook != _cur_wh:
                    patch["webhook_url"] = new_webhook if new_webhook else None

                # Tags
                new_tags = [_tag.strip() for _tag in new_tags_raw.split(",") if _tag.strip()]
                if new_tags != (selected.get("tags") or []):
                    patch["tags"] = new_tags if new_tags else None

                # Confidence threshold
                _stored_ct = float(_cur_ct) if _cur_ct is not None else 0.5
                if abs(new_confidence_threshold - _stored_ct) > 1e-9:
                    patch["confidence_threshold"] = new_confidence_threshold

                # Deployment mode
                _stored_dm = selected.get("deployment_mode") or "production"
                _stored_tw = float(selected.get("traffic_weight") or 0.5)
                if new_deploy_mode != _stored_dm:
                    patch["deployment_mode"] = new_deploy_mode
                if new_deploy_mode == "ab_test" and new_traffic_weight is not None:
                    if new_deploy_mode != _stored_dm or abs(new_traffic_weight - _stored_tw) > 1e-9:
                        patch["traffic_weight"] = new_traffic_weight

                # Alert thresholds
                new_at = {
                    "accuracy_min": new_accuracy_min if new_accuracy_min > 0 else None,
                    "error_rate_max": new_error_rate_max if new_error_rate_max > 0 else None,
                    "drift_auto_alert": new_drift_auto_alert,
                }
                _stored_at = {
                    "accuracy_min": _cur_at.get("accuracy_min"),
                    "error_rate_max": _cur_at.get("error_rate_max"),
                    "drift_auto_alert": _cur_at.get("drift_auto_alert", True),
                }
                if new_at != _stored_at:
                    patch["alert_thresholds"] = new_at

                if patch:
                    try:
                        client.update_model(selected["name"], selected["version"], patch)
                        st.toast(t("models.edit_meta.save_toast"), icon="✅")
                        reload()
                    except Exception as e:
                        st.toast(t("models.edit_meta.save_error", error=e), icon="❌")
                else:
                    st.info(t("models.edit_meta.no_change"))

        # Retraining
        if selected.get("train_script_object_key"):
            with st.expander(t("models.retrain.expander"), expanded=False):
                with st.form("retrain_form"):
                    st.markdown(t("models.retrain.title", name=selected['name'], version=selected['version']))
                    col_s, col_e = st.columns(2)
                    with col_s:
                        start_date = st.date_input(t("models.retrain.start_date"), key="retrain_start")
                    with col_e:
                        end_date = st.date_input(t("models.retrain.end_date"), key="retrain_end")
                    new_version_input = st.text_input(
                        t("models.retrain.new_version_label"),
                        value="",
                        placeholder=f"{selected['version']}-retrain-YYYYMMDDHHMMSS",
                        key="retrain_new_version",
                    )
                    set_prod = st.checkbox(
                        t("models.retrain.set_prod_label"),
                        value=False,
                        key="retrain_set_prod",
                    )
                    submitted = st.form_submit_button(t("models.retrain.submit_btn"), type="primary")

                if submitted:
                    if start_date > end_date:
                        st.error(t("models.retrain.date_order_error"))
                    else:
                        with st.spinner(t("models.retrain.spinner")):
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
                                        t("models.retrain.success_toast", new_version=result['new_version']),
                                        icon="✅",
                                    )
                                else:
                                    st.toast(
                                        t("models.retrain.failure_toast", error=result.get('error', t("models.retrain.unknown_error"))),
                                        icon="❌",
                                    )
                                with st.expander(t("models.retrain.stdout_expander"), expanded=not result.get("success")):
                                    st.code(result.get("stdout", t("models.retrain.logs_empty")), language="text")
                                with st.expander(t("models.retrain.stderr_expander"), expanded=not result.get("success")):
                                    st.code(result.get("stderr", t("models.retrain.logs_empty")), language="text")
                                if result.get("success"):
                                    reload()
                            except Exception as e:
                                st.toast(t("models.retrain.error_toast", error=e), icon="❌")

        # Auto-promotion policy / circuit breaker
        if is_admin:
            _pp = selected.get("promotion_policy") or {}
            _pp_promote_on = _pp.get("auto_promote", False)
            _pp_demote_on  = _pp.get("auto_demote",  False)

            if _pp_promote_on and _pp_demote_on:
                _policy_exp_title = t("models.policy.expander_active")
            elif _pp_promote_on:
                _policy_exp_title = t("models.policy.expander_promote_only")
            elif _pp_demote_on:
                _policy_exp_title = t("models.policy.expander_cb_only")
            else:
                _policy_exp_title = t("models.policy.expander_default")

            with st.expander(_policy_exp_title, expanded=False):
                st.caption(t("models.policy.caption", name=selected['name']))

                # ── Auto-promotion ─────────────────────────────────────────────
                st.markdown(t("models.policy.promote_header_active") if _pp_promote_on else t("models.policy.promote_header_inactive"))
                _pp_c1, _pp_c2 = st.columns(2)
                with _pp_c1:
                    _pp_new_promote = st.checkbox(
                        t("models.policy.auto_promote_label"),
                        value=_pp_promote_on,
                        key="pp_auto_promote",
                    )
                    _pp_min_acc = _pp.get("min_accuracy")
                    _pp_new_min_acc = st.number_input(
                        t("models.policy.min_accuracy_label"),
                        min_value=0.0, max_value=1.0, step=0.01,
                        value=float(_pp_min_acc) if _pp_min_acc is not None else 0.0,
                        key="pp_min_accuracy",
                        help=t("models.policy.min_accuracy_help"),
                    )
                    _pp_min_golden = _pp.get("min_golden_test_pass_rate")
                    _pp_new_golden = st.number_input(
                        t("models.policy.min_golden_label"),
                        min_value=0.0, max_value=1.0, step=0.01,
                        value=float(_pp_min_golden) if _pp_min_golden is not None else 0.0,
                        key="pp_min_golden",
                    )
                with _pp_c2:
                    _pp_max_mae = _pp.get("max_mae")
                    _pp_new_max_mae = st.number_input(
                        t("models.policy.max_mae_label"),
                        min_value=0.0, step=0.01,
                        value=float(_pp_max_mae) if _pp_max_mae is not None else 0.0,
                        key="pp_max_mae",
                        help=t("models.policy.max_mae_help"),
                    )
                    _pp_max_lat = _pp.get("max_latency_p95_ms")
                    _pp_new_max_lat = st.number_input(
                        t("models.policy.max_latency_label"),
                        min_value=0.0, step=10.0,
                        value=float(_pp_max_lat) if _pp_max_lat is not None else 0.0,
                        key="pp_max_latency",
                    )
                    _pp_new_min_samples = st.number_input(
                        t("models.policy.min_samples_label"),
                        min_value=1, step=1,
                        value=int(_pp.get("min_sample_validation", 10)),
                        key="pp_min_samples",
                    )

                st.divider()

                # ── Circuit breaker ────────────────────────────────────────────
                st.markdown(t("models.policy.cb_header_active") if _pp_demote_on else t("models.policy.cb_header_inactive"))
                _pp_cb1, _pp_cb2 = st.columns(2)
                with _pp_cb1:
                    _pp_new_demote = st.checkbox(
                        t("models.policy.auto_demote_label"),
                        value=_pp_demote_on,
                        key="pp_auto_demote",
                    )
                    _pp_new_drift = st.selectbox(
                        t("models.policy.drift_level_label"),
                        ["warning", "critical"],
                        index=0 if _pp.get("demote_on_drift", "critical") == "warning" else 1,
                        key="pp_demote_on_drift",
                        format_func=lambda x: t("models.policy.drift_warning") if x == "warning" else t("models.policy.drift_critical"),
                        help=t("models.policy.drift_level_help"),
                    )
                with _pp_cb2:
                    _pp_acc_thr = _pp.get("demote_on_accuracy_below")
                    _pp_new_demote_acc = st.number_input(
                        t("models.policy.demote_accuracy_label"),
                        min_value=0.0, max_value=1.0, step=0.01,
                        value=float(_pp_acc_thr) if _pp_acc_thr is not None else 0.0,
                        key="pp_demote_accuracy",
                        help=t("models.policy.demote_accuracy_help"),
                    )
                    _pp_new_cooldown = st.number_input(
                        t("models.policy.cooldown_label"),
                        min_value=0, step=1,
                        value=int(_pp.get("demote_cooldown_hours", 24)),
                        key="pp_cooldown",
                        help=t("models.policy.cooldown_help"),
                    )

                # ── Save (once, entire policy at once) ────────────────────────
                if st.button(t("models.policy.btn_save"), key="save_policy"):
                    try:
                        client.set_policy(
                            selected["name"],
                            auto_promote=_pp_new_promote,
                            min_accuracy=_pp_new_min_acc if _pp_new_min_acc > 0 else None,
                            max_mae=_pp_new_max_mae if _pp_new_max_mae > 0 else None,
                            max_latency_p95_ms=_pp_new_max_lat if _pp_new_max_lat > 0 else None,
                            min_sample_validation=_pp_new_min_samples,
                            min_golden_test_pass_rate=_pp_new_golden if _pp_new_golden > 0 else None,
                            auto_demote=_pp_new_demote,
                            demote_on_drift=_pp_new_drift,
                            demote_on_accuracy_below=_pp_new_demote_acc if _pp_new_demote_acc > 0 else None,
                            demote_cooldown_hours=_pp_new_cooldown,
                        )
                        st.toast(t("models.policy.save_toast"), icon="✅")
                        reload()
                    except Exception as e:
                        st.toast(t("models.policy.save_error", error=e), icon="❌")

        # Compute baseline from production
        with st.expander(t("models.baseline.expander")):
            st.markdown(t("models.baseline.description"))
            if selected.get("feature_baseline"):
                st.warning(t("models.baseline.existing_warning"))
            baseline_days = st.slider(t("models.baseline.days_slider_label"), 7, 180, 30, key="baseline_days")
            baseline_dry_run = st.checkbox(
                t("models.baseline.dry_run_label"), value=True, key="baseline_dry_run"
            )
            if st.button(t("models.baseline.btn_compute"), key="baseline_compute_btn", type="primary"):
                with st.spinner(t("models.baseline.spinner")):
                    try:
                        result = client.compute_baseline(
                            name=selected["name"],
                            version=selected["version"],
                            days=baseline_days,
                            dry_run=baseline_dry_run,
                        )
                        st.markdown(t("models.baseline.result_caption", n=result.get('predictions_used'), days=baseline_days))
                        st.json(result.get("baseline", {}))
                        if baseline_dry_run:
                            st.info(t("models.baseline.dry_run_info"))
                        else:
                            st.toast(t("models.baseline.save_toast"), icon="✅")
                            st.cache_data.clear()
                    except Exception as e:
                        st.toast(t("models.baseline.error", error=e), icon="❌")

        # Modification history
        with st.expander(t("models.history.expander")):
            try:
                history_data = client.get_model_history(selected["name"], selected["version"], limit=20)
                entries = history_data.get("entries", [])
                total_hist = history_data.get("total", 0)
            except Exception as e:
                st.error(t("models.history.load_error", error=e))
                entries = []
                total_hist = 0

            if not entries:
                st.info(t("models.history.no_history"))
            else:
                st.caption(t("models.history.caption", total=total_hist))
                for entry in entries:
                    ts = pd.to_datetime(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S UTC")
                    badge = _action_badge(entry["action"])
                    changed = ", ".join(entry.get("changed_fields") or []) or "—"
                    who = entry.get("changed_by_username") or t("models.history.changed_by_unknown")

                    col_info, col_btn = st.columns([5, 1])
                    with col_info:
                        st.markdown(t("models.history.entry_line", ts=ts, badge=badge, who=who, changed=changed))
                    with col_btn:
                        if is_admin:
                            if st.button(
                                t("models.history.btn_rollback"),
                                key=f"rollback_btn_{entry['id']}",
                                type="secondary",
                                help=t("models.history.rollback_help", id=entry['id']),
                            ):
                                st.session_state["confirm_rollback_id"] = entry["id"]
                                st.session_state["confirm_rollback_model"] = selected["name"]
                                st.session_state["confirm_rollback_version"] = selected["version"]

                    with st.expander(t("models.history.snapshot_expander", id=entry['id']), expanded=False):
                        st.json(entry["snapshot"])

                    st.divider()

                # Rollback confirmation dialog
                confirm_id = st.session_state.get("confirm_rollback_id")
                confirm_model = st.session_state.get("confirm_rollback_model")
                confirm_version = st.session_state.get("confirm_rollback_version")
                if (
                    confirm_id is not None
                    and confirm_model == selected["name"]
                    and confirm_version == selected["version"]
                ):
                    st.warning(t("models.history.confirm_rollback", model=confirm_model, version=confirm_version, id=confirm_id))
                    c1, c2 = st.columns(2)
                    if c1.button(t("models.history.confirm_yes"), type="primary", key="confirm_rollback_yes"):
                        try:
                            result = client.rollback_model(confirm_model, confirm_version, confirm_id)
                            st.toast(
                                t("models.history.rollback_toast", id=result['new_history_id']),
                                icon="✅",
                            )
                            st.session_state.pop("confirm_rollback_id", None)
                            st.session_state.pop("confirm_rollback_model", None)
                            st.session_state.pop("confirm_rollback_version", None)
                            reload()
                        except Exception as e:
                            st.toast(t("models.history.rollback_error", error=e), icon="❌")
                    if c2.button(t("models.history.confirm_cancel"), key="confirm_rollback_no"):
                        st.session_state.pop("confirm_rollback_id", None)
                        st.session_state.pop("confirm_rollback_model", None)
                        st.session_state.pop("confirm_rollback_version", None)
                        st.rerun()
