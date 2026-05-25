"""
Prediction history with filters
"""

import io
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.api_client import get_models as get_models_cached
from utils.auth import get_client, require_auth
from utils.i18n import t

_SCRIPTS_DIR = Path(__file__).parent.parent / "documentation" / "Scripts"

_EXAMPLE_SCRIPTS = [
    (
        "send_predictions_iris.py",
        t("predictions.scripts.iris_desc"),
    ),
    (
        "send_ground_truth.py",
        t("predictions.scripts.ground_truth_desc"),
    ),
]


def _read_script(filename: str) -> str:
    try:
        return (_SCRIPTS_DIR / filename).read_text(encoding="utf-8")
    except Exception:
        return t("predictions.scripts.not_found", filename=filename)


@st.dialog(t("predictions.scripts.dialog_title"), width="large")
def _view_script_dialog(filename: str) -> None:
    st.code(_read_script(filename), language="python", line_numbers=True)
    st.download_button(
        t("predictions.scripts.download_btn"),
        data=_read_script(filename),
        file_name=filename,
        mime="text/x-python",
        key=f"dl_dialog_{filename}",
        width='stretch',
    )


st.set_page_config(page_title="Predictions — PredictML", page_icon="📊", layout="wide")
require_auth()

col_title, col_refresh = st.columns([8, 1])
col_title.title(t("predictions.page_title"))
if col_refresh.button(t("predictions.refresh_btn"), key="pred_refresh", width='stretch'):
    st.cache_data.clear()
    st.rerun()

client = get_client()

with st.expander(t("predictions.scripts.expander_title"), expanded=False):
    st.caption(t("predictions.scripts.expander_caption"))
    for _script_name, _script_desc in _EXAMPLE_SCRIPTS:
        _col_desc, _col_view, _col_dl = st.columns([5, 1.5, 1.5])
        _col_desc.markdown(f"**`{_script_name}`**  \n{_script_desc}")
        if _col_view.button(t("predictions.scripts.view_btn"), key=f"view_{_script_name}", width='stretch'):
            _view_script_dialog(_script_name)
        _col_dl.download_button(
            t("predictions.scripts.download_btn"),
            data=_read_script(_script_name),
            file_name=_script_name,
            mime="text/x-python",
            key=f"dl_{_script_name}",
            width='stretch',
        )

tab_history, tab_batch = st.tabs([t("predictions.tab_history"), t("predictions.tab_batch")])

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — History
# ───────────────────────────────────────────────────────────────────────────────
with tab_history:
    # --- Ground truth coverage ---
    with st.expander(t("predictions.coverage.expander_title"), expanded=True):
        try:
            coverage_data = client.get_observed_results_stats()
            total_pred = coverage_data.get("total_predictions", 0)
            labeled = coverage_data.get("labeled_count", 0)
            rate = coverage_data.get("coverage_rate", 0.0)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric(t("predictions.coverage.total_predictions"), f"{total_pred:,}")
            col_b.metric(t("predictions.coverage.labeled"), f"{labeled:,}")
            col_c.metric(t("predictions.coverage.coverage_rate"), f"{rate * 100:.1f} %")

            by_model = coverage_data.get("by_model") or []
            if by_model:
                st.markdown(t("predictions.coverage.by_model"))
                for m in by_model:
                    cov = m.get("coverage", 0.0)
                    st.progress(
                        cov,
                        text=f"{m['model_name']} — {m['labeled']}/{m['predictions']} ({cov * 100:.1f} %)",
                    )
        except Exception:
            st.caption(t("predictions.coverage.unavailable"))

    # --- Filters ---
    with st.expander(t("predictions.filters.expander_title"), expanded=False):
        col_search, col_model, col2, col3, col4, col5 = st.columns([1.2, 1.8, 1.2, 1.2, 1.2, 1.2])

        # List of available models
        try:
            models = get_models_cached(
                st.session_state.get("api_url"), st.session_state.get("api_token")
            )
            model_names = sorted({m["name"] for m in models})
        except Exception:
            models = []
            model_names = []

        hist_search = col_search.text_input(
            t("predictions.filters.search_label"), key="hist_model_search", placeholder=t("predictions.filters.search_placeholder")
        )
        hist_filtered = (
            [n for n in model_names if hist_search.lower() in n.lower()]
            if hist_search
            else model_names
        )
        model_name = col_model.selectbox(t("predictions.filters.model_label"), [t("predictions.filters.all_models")] + (hist_filtered or model_names))
        if model_name == t("predictions.filters.all_models"):
            model_name = model_names[0] if model_names else None

        today = date.today()
        start_date = col2.date_input(t("predictions.filters.start_date"), value=today - timedelta(days=7))
        end_date = col3.date_input(t("predictions.filters.end_date"), value=today)
        status_filter = col4.selectbox(t("predictions.filters.status_label"), [t("predictions.filters.status_all"), "success", "error"])
        _LIMIT_LABELS = ["50", "100", "500", t("predictions.filters.limit_all")]
        _LIMIT_VALUES = [50, 100, 500, None]
        _limit_label = col5.selectbox(t("predictions.filters.limit_label"), _LIMIT_LABELS, index=1)
        limit = _LIMIT_VALUES[_LIMIT_LABELS.index(_limit_label)]

        is_classifier = any(
            m.get("name") == model_name and m.get("classes")
            for m in models
        ) if model_name else False

        col_conf1, col_conf2, col_conf3 = st.columns(3)
        conf_min = col_conf1.slider(
            t("predictions.filters.conf_min_label"),
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            format="%.2f",
            key="hist_conf_min",
            disabled=not is_classifier,
            help=t("predictions.filters.conf_min_help"),
        )
        conf_max = col_conf2.slider(
            t("predictions.filters.conf_max_label"),
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            format="%.2f",
            key="hist_conf_max",
            disabled=not is_classifier,
            help=t("predictions.filters.conf_max_help"),
        )
        filter_mismatch_only = col_conf3.checkbox(
            t("predictions.filters.mismatch_only_label"),
            key="hist_mismatch_only",
            help=t("predictions.filters.mismatch_only_help"),
        )
        filter_min_conf = conf_min if (is_classifier and conf_min > 0.0) else None
        filter_max_conf = conf_max if (is_classifier and conf_max < 1.0) else None

    if not model_name:
        st.warning(t("predictions.warnings.no_model"))
    elif start_date > end_date:
        st.error(t("predictions.warnings.date_order"))
    else:
        start_iso = datetime.combine(start_date, datetime.min.time()).isoformat()
        end_iso = datetime.combine(end_date, datetime.max.time()).isoformat()

        _API_MAX = 1000
        try:
            if limit is None:
                _probe = client.get_predictions(
                    model_name=model_name,
                    start=start_iso,
                    end=end_iso,
                    limit=1,
                    offset=0,
                    min_confidence=filter_min_conf,
                    max_confidence=filter_max_conf,
                )
                _total_count = _probe.get("total", 0)
                _all_preds: list = []
                _offset = 0
                _bar = st.progress(0, text=t("predictions.loading.progress", loaded=0, total=_total_count))
                while _offset < max(_total_count, 1):
                    _chunk = client.get_predictions(
                        model_name=model_name,
                        start=start_iso,
                        end=end_iso,
                        limit=_API_MAX,
                        offset=_offset,
                        min_confidence=filter_min_conf,
                        max_confidence=filter_max_conf,
                    )
                    _preds = _chunk.get("predictions", [])
                    _all_preds.extend(_preds)
                    _offset += _API_MAX
                    _pct = min(len(_all_preds) / _total_count, 1.0) if _total_count else 1.0
                    _bar.progress(_pct, text=t("predictions.loading.progress", loaded=len(_all_preds), total=_total_count))
                    if not _preds:
                        break
                _bar.empty()
                data = {"total": _total_count, "predictions": _all_preds}
            else:
                data = client.get_predictions(
                    model_name=model_name,
                    start=start_iso,
                    end=end_iso,
                    limit=limit,
                    offset=0,
                    min_confidence=filter_min_conf,
                    max_confidence=filter_max_conf,
                )
        except Exception as e:
            st.error(t("predictions.errors.load_error", error=e))
            st.stop()

        total = data.get("total", 0)
        predictions = data.get("predictions", [])

        if status_filter != t("predictions.filters.status_all"):
            predictions = [p for p in predictions if p.get("status") == status_filter]

        with st.expander(t("predictions.results.expander_title"), expanded=False):
            st.caption(t("predictions.results.filter_note"))
            st.caption(t("predictions.results.count_caption", total=total, shown=len(predictions)))

            if not predictions:
                st.info(t("predictions.results.no_predictions"))
            else:
                # Fetch ground truth for visible id_obs values
                gt_lookup: dict = {}
                id_obs_list = [p["id_obs"] for p in predictions if p.get("id_obs")]
                if id_obs_list:
                    try:
                        obs_data = client.get_observed_results(
                            model_name=model_name or None,
                            limit=len(id_obs_list) + 50,
                        )
                        for obs in obs_data.get("results", obs_data if isinstance(obs_data, list) else []):
                            if obs.get("id_obs"):
                                gt_lookup[obs["id_obs"]] = str(obs.get("observed_result", ""))
                    except Exception:
                        pass

                rows = []
                for p in predictions:
                    mc = p.get("max_confidence")
                    id_obs_val = p.get("id_obs")
                    gt_val = gt_lookup.get(id_obs_val, "—") if id_obs_val else "—"
                    pred_val = str(p.get("prediction_result", ""))
                    mismatch = gt_val != "—" and pred_val != gt_val
                    rows.append(
                        {
                            t("predictions.table.col_id"): p.get("id"),
                            "id_obs": id_obs_val or "—",
                            t("predictions.table.col_timestamp"): (
                                pd.to_datetime(p.get("timestamp")).strftime("%Y-%m-%d %H:%M:%S")
                                if p.get("timestamp")
                                else "—"
                            ),
                            t("predictions.table.col_model"): p.get("model_name", ""),
                            t("predictions.table.col_version"): p.get("model_version") or "—",
                            t("predictions.table.col_result"): pred_val,
                            t("predictions.table.col_ground_truth"): gt_val,
                            t("predictions.table.col_confidence"): f"{mc:.2%}" if mc is not None else "—",
                            t("predictions.table.col_time_ms"): (
                                f"{p['response_time_ms']:.1f}"
                                if p.get("response_time_ms") is not None
                                else "—"
                            ),
                            t("predictions.table.col_status"): "✅" if p.get("status") == "success" else "❌",
                            t("predictions.table.col_shadow"): "🔮" if p.get("is_shadow") else "—",
                            t("predictions.table.col_user"): p.get("username") or "—",
                            "_mismatch": mismatch,
                        }
                    )

                df = pd.DataFrame(rows)

                if filter_mismatch_only:
                    df = df[df["_mismatch"]].reset_index(drop=True)
                    if df.empty:
                        st.info(t("predictions.results.no_mismatch"))

                mismatch_flags = df["_mismatch"].to_numpy()
                df_display = df.drop(columns=["_mismatch"])

                def _highlight_mismatch(row):
                    return (
                        ["background-color: #ffcccc"] * len(row)
                        if mismatch_flags[row.name]
                        else [""] * len(row)
                    )

                styled = df_display.style.apply(_highlight_mismatch, axis=1)
                sel = st.dataframe(
                    styled,
                    width='stretch',
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    column_config={
                        t("predictions.table.col_id"): st.column_config.NumberColumn(
                            t("predictions.table.col_id"),
                            help=t("predictions.table.help_id"),
                        ),
                        "id_obs": st.column_config.TextColumn(
                            "id_obs",
                            help=t("predictions.table.help_id_obs"),
                        ),
                        t("predictions.table.col_timestamp"): st.column_config.TextColumn(
                            t("predictions.table.col_timestamp"),
                            help=t("predictions.table.help_timestamp"),
                        ),
                        t("predictions.table.col_model"): st.column_config.TextColumn(
                            t("predictions.table.col_model"),
                            help=t("predictions.table.help_model"),
                        ),
                        t("predictions.table.col_version"): st.column_config.TextColumn(
                            t("predictions.table.col_version"),
                            help=t("predictions.table.help_version"),
                        ),
                        t("predictions.table.col_result"): st.column_config.TextColumn(
                            t("predictions.table.col_result"),
                            help=t("predictions.table.help_result"),
                        ),
                        t("predictions.table.col_ground_truth"): st.column_config.TextColumn(
                            t("predictions.table.col_ground_truth"),
                            help=t("predictions.table.help_ground_truth"),
                        ),
                        t("predictions.table.col_confidence"): st.column_config.TextColumn(
                            t("predictions.table.col_confidence"),
                            help=t("predictions.table.help_confidence"),
                        ),
                        t("predictions.table.col_time_ms"): st.column_config.TextColumn(
                            t("predictions.table.col_time_ms"),
                            help=t("predictions.table.help_time_ms"),
                        ),
                        t("predictions.table.col_status"): st.column_config.TextColumn(
                            t("predictions.table.col_status"),
                            help=t("predictions.table.help_status"),
                        ),
                        t("predictions.table.col_shadow"): st.column_config.TextColumn(
                            t("predictions.table.col_shadow"),
                            help=t("predictions.table.help_shadow"),
                        ),
                        t("predictions.table.col_user"): st.column_config.TextColumn(
                            t("predictions.table.col_user"),
                            help=t("predictions.table.help_user"),
                        ),
                    },
                )

                # ── Detail panel (selected row) ───────────────────────────
                selected_rows = sel.selection.rows if sel.selection else []
                if selected_rows:
                    import plotly.graph_objects as go

                    row_idx = selected_rows[0]
                    pred_id_at_row = df_display.iloc[row_idx][t("predictions.table.col_id")]
                    p = next((x for x in predictions if x.get("id") == pred_id_at_row), None)
                    if p is None:
                        st.info(t("predictions.detail.not_found"))
                        st.stop()
                    pred_id = pred_id_at_row
                    st.divider()
                    st.markdown(t("predictions.detail.title", pred_id=pred_id))

                    col_l, col_r = st.columns(2)
                    with col_l:
                        st.markdown(t("predictions.detail.input_features"))
                        st.json(p.get("input_features", {}))
                    with col_r:
                        st.markdown(t("predictions.detail.result_label"))
                        st.json({
                            "prediction": p.get("prediction_result"),
                            "probabilities": p.get("probabilities"),
                        })
                        if p.get("error_message"):
                            st.error(t("predictions.detail.error_msg", msg=p['error_message']))

                    # Ground truth
                    st.divider()
                    st.markdown(t("predictions.detail.observed_section"))
                    id_obs_val = p.get("id_obs")
                    if not id_obs_val:
                        st.caption(t("predictions.detail.no_id_obs"))
                    else:
                        obs_cache_key = f"obs_result_{pred_id}"
                        if obs_cache_key not in st.session_state:
                            try:
                                resp = client.get_observed_results(
                                    model_name=p.get("model_name"), id_obs=id_obs_val, limit=1
                                )
                                results = resp.get("results", [])
                                st.session_state[obs_cache_key] = results[0] if results else None
                            except Exception:
                                st.session_state[obs_cache_key] = None

                        existing = st.session_state.get(obs_cache_key)
                        if existing is not None:
                            st.success(t("predictions.detail.obs_recorded", value=existing['observed_result']))
                        else:
                            obs_input_val = st.text_input(
                                t("predictions.detail.obs_input_label"),
                                key=f"obs_input_{pred_id}",
                                placeholder=t("predictions.detail.obs_input_placeholder"),
                            )
                            if st.button(t("predictions.detail.obs_submit_btn"), key=f"obs_btn_{pred_id}"):
                                if not obs_input_val.strip():
                                    st.warning(t("predictions.detail.obs_empty_warning"))
                                else:
                                    try:
                                        parsed_val = int(obs_input_val)
                                    except ValueError:
                                        try:
                                            parsed_val = float(obs_input_val)
                                        except ValueError:
                                            parsed_val = obs_input_val
                                    try:
                                        client.submit_observed_result(
                                            id_obs=id_obs_val,
                                            model_name=p.get("model_name"),
                                            observed_result=parsed_val,
                                        )
                                        st.session_state[obs_cache_key] = {"observed_result": parsed_val}
                                        st.rerun()
                                    except Exception as exc:
                                        st.error(t("predictions.errors.generic", error=exc))

                    # SHAP
                    if p.get("status") == "success":
                        st.divider()
                        st.markdown(t("predictions.detail.shap_section"))
                        if st.button(t("predictions.detail.shap_btn"), key=f"shap_btn_{pred_id}"):
                            with st.spinner(t("predictions.detail.shap_spinner")):
                                try:
                                    st.session_state[f"shap_{pred_id}"] = client.explain_prediction(pred_id)
                                except Exception as exc:
                                    st.error(t("predictions.detail.shap_error", error=exc))

                        shap_data = st.session_state.get(f"shap_{pred_id}")
                        if shap_data:
                            shap_values: dict = shap_data.get("shap_values", {})
                            base_value: float = shap_data.get("base_value", 0.0)
                            model_type: str = shap_data.get("model_type", "")
                            col_s1, col_s2, col_s3 = st.columns(3)
                            col_s1.metric("E[f(X)]", f"{base_value:.4f}")
                            col_s2.metric(t("predictions.detail.shap_prediction_metric"), str(shap_data.get("prediction")))
                            col_s3.metric(t("predictions.detail.shap_type_metric"), model_type)
                            if shap_values:
                                sorted_features = sorted(
                                    shap_values.items(), key=lambda x: abs(x[1]), reverse=True
                                )[:10]
                                feat_names = [f for f, _ in sorted_features]
                                shap_vals = [v for _, v in sorted_features]
                                fig = go.Figure(go.Bar(
                                    x=shap_vals, y=feat_names, orientation="h",
                                    marker_color=["#e05252" if v >= 0 else "#5282e0" for v in shap_vals],
                                    text=[f"{v:+.4f}" for v in shap_vals],
                                    textposition="outside",
                                ))
                                fig.update_layout(
                                    title=t("predictions.detail.shap_chart_title"),
                                    xaxis_title=t("predictions.detail.shap_chart_xaxis"),
                                    yaxis={"autorange": "reversed"},
                                    height=max(300, len(sorted_features) * 40 + 100),
                                    margin={"l": 20, "r": 60, "t": 50, "b": 40},
                                    showlegend=False,
                                )
                                st.plotly_chart(fig, width='stretch')
                                if len(shap_values) > 10:
                                    with st.expander(t("predictions.detail.shap_all_features_expander")):
                                        st.dataframe(
                                            pd.DataFrame(
                                                sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True),
                                                columns=["Feature", "SHAP"],
                                            ),
                                            width='stretch', hide_index=True,
                                        )
                            else:
                                st.info(t("predictions.detail.shap_no_values"))

                    # Deletion
                    if st.session_state.get("is_admin", False):
                        st.divider()
                        if st.button(
                            t("predictions.detail.delete_btn", pred_id=pred_id),
                            key=f"del_pred_{pred_id}",
                            type="primary",
                        ):
                            try:
                                client.delete_prediction(pred_id)
                                st.toast(t("predictions.detail.delete_toast", pred_id=pred_id), icon="✅")
                                st.rerun()
                            except Exception as exc:
                                st.error(t("predictions.errors.generic", error=exc))

                with st.expander(t("predictions.export.expander_title"), expanded=False):
                    st.caption(t("predictions.export.caption"))
                    col_exp1, col_exp2 = st.columns(2)
                    export_fmt = col_exp1.selectbox(
                        t("predictions.export.format_label"), ["csv", "jsonl", "parquet"], key="pred_export_fmt"
                    )
                    export_status = col_exp2.selectbox(
                        t("predictions.export.status_label"), [t("predictions.filters.all_models_export"), "success", "error"], key="pred_export_status"
                    )
                    if st.button(t("predictions.export.prepare_btn"), key="pred_export_btn"):
                        with st.spinner(t("predictions.export.spinner")):
                            try:
                                content = client.export_predictions(
                                    start=start_iso,
                                    end=end_iso,
                                    model_name=model_name,
                                    export_format=export_fmt,
                                    status=None if export_status == t("predictions.filters.all_models_export") else export_status,
                                )
                                mime_map = {
                                    "csv": "text/csv",
                                    "jsonl": "application/x-ndjson",
                                    "parquet": "application/octet-stream",
                                }
                                st.download_button(
                                    label=t("predictions.export.download_btn", fmt=export_fmt),
                                    data=content,
                                    file_name=f"predictions_export.{export_fmt}",
                                    mime=mime_map[export_fmt],
                                    key="pred_export_download",
                                )
                            except Exception as exc:
                                st.error(t("predictions.export.error", error=exc))


        # --- Activity by user & model (admin only) ---
        if st.session_state.get("is_admin", False):
            with st.expander(t("predictions.activity.expander_title"), expanded=False):
                st.caption(t("predictions.activity.caption"))

                try:
                    _all_users = client.list_users()
                except Exception as _e:
                    st.error(t("predictions.activity.load_users_error", error=_e))
                    _all_users = []

                if not _all_users:
                    st.info(t("predictions.activity.no_users"))
                else:
                    _usage_summary: list[dict] = []
                    _by_model_rows: list[dict] = []
                    _by_day_rows:   list[dict] = []

                    with st.spinner(t("predictions.activity.spinner")):
                        for _u in _all_users:
                            try:
                                _udata = client.get_user_usage(
                                    _u["id"],
                                    start_date=start_date.isoformat(),
                                    end_date=end_date.isoformat(),
                                )
                            except Exception:
                                continue

                            _total = _udata.get("total_calls", 0)
                            if _total == 0:
                                continue

                            _uname  = _udata.get("username", _u.get("username", f"user_{_u['id']}"))
                            _bm     = _udata.get("by_model", [])
                            _errors = sum(m.get("errors", 0) for m in _bm)
                            _lats   = [m["avg_latency_ms"] for m in _bm if m.get("avg_latency_ms")]
                            _avg_lat = round(sum(_lats) / len(_lats), 1) if _lats else None

                            _usage_summary.append({
                                "username":       _uname,
                                "total_calls":    _total,
                                "nb_modeles":     len(_bm),
                                "errors":         _errors,
                                "avg_latency_ms": _avg_lat,
                            })
                            for _m in _bm:
                                _by_model_rows.append({
                                    "username":       _uname,
                                    "model_name":     _m["model_name"],
                                    "calls":          _m["calls"],
                                    "errors":         _m.get("errors", 0),
                                    "avg_latency_ms": _m.get("avg_latency_ms"),
                                })
                            for _d in _udata.get("by_day", []):
                                _by_day_rows.append({
                                    "username": _uname,
                                    "date":     _d["date"],
                                    "calls":    _d["calls"],
                                })

                    if not _usage_summary:
                        st.info(t("predictions.activity.no_activity"))
                    else:
                        df_usumm = pd.DataFrame(_usage_summary).sort_values(
                            "total_calls", ascending=False
                        ).reset_index(drop=True)
                        _total_all = int(df_usumm["total_calls"].sum())

                        # ── KPIs ──────────────────────────────────────────────────────
                        _ku1, _ku2, _ku3, _ku4 = st.columns(4)
                        _ku1.metric(t("predictions.activity.kpi_total_calls"), f"{_total_all:,}")
                        _ku2.metric(
                            t("predictions.activity.kpi_active_users"), len(df_usumm),
                            help=t("predictions.activity.kpi_active_users_help")
                        )
                        _top = df_usumm.iloc[0]
                        _top_pct = _top["total_calls"] / _total_all * 100
                        _ku3.metric(
                            t("predictions.activity.kpi_top_user"),
                            _top["username"],
                            delta=t("predictions.activity.kpi_top_user_delta", calls=_top['total_calls'], pct=_top_pct),
                            delta_color="off",
                        )
                        _top3_pct = df_usumm.head(3)["total_calls"].sum() / _total_all * 100
                        _ku4.metric(
                            t("predictions.activity.kpi_top3_concentration"),
                            f"{_top3_pct:.0f} %",
                            help=t("predictions.activity.kpi_top3_help"),
                        )

                        st.divider()

                        # ── Summary table by user ─────────────────────────
                        st.markdown(t("predictions.activity.summary_header"))
                        _df_disp = df_usumm.copy()
                        _df_disp[t("predictions.activity.col_traffic_share")] = _df_disp["total_calls"].apply(
                            lambda v: f"{v / _total_all * 100:.1f} %"
                        )
                        _df_disp[t("predictions.activity.col_error_rate")] = _df_disp.apply(
                            lambda r: f"{r['errors'] / r['total_calls'] * 100:.1f} %"
                            if r["total_calls"] > 0 else "—",
                            axis=1,
                        )
                        _df_disp[t("predictions.activity.col_avg_latency")] = _df_disp["avg_latency_ms"].apply(
                            lambda v: f"{v} ms" if v is not None else "—"
                        )
                        st.dataframe(
                            _df_disp.rename(columns={
                                "username":    t("predictions.activity.col_user"),
                                "total_calls": t("predictions.activity.col_calls"),
                                "nb_modeles":  t("predictions.activity.col_models_used"),
                                "errors":      t("predictions.activity.col_errors"),
                            })[[t("predictions.activity.col_user"), t("predictions.activity.col_calls"),
                                t("predictions.activity.col_traffic_share"),
                                t("predictions.activity.col_models_used"),
                                t("predictions.activity.col_errors"),
                                t("predictions.activity.col_error_rate"),
                                t("predictions.activity.col_avg_latency")]],
                            hide_index=True,
                            width='stretch',
                            column_config={
                                t("predictions.activity.col_calls"): st.column_config.NumberColumn(
                                    t("predictions.activity.col_calls"),
                                    help=t("predictions.activity.col_calls_help"),
                                ),
                                t("predictions.activity.col_traffic_share"): st.column_config.TextColumn(
                                    t("predictions.activity.col_traffic_share"),
                                    help=t("predictions.activity.col_traffic_share_help"),
                                ),
                                t("predictions.activity.col_models_used"): st.column_config.NumberColumn(
                                    t("predictions.activity.col_models_used"),
                                    help=t("predictions.activity.col_models_used_help"),
                                ),
                                t("predictions.activity.col_error_rate"): st.column_config.TextColumn(
                                    t("predictions.activity.col_error_rate"),
                                    help=t("predictions.activity.col_error_rate_help"),
                                ),
                                t("predictions.activity.col_avg_latency"): st.column_config.TextColumn(
                                    t("predictions.activity.col_avg_latency"),
                                    help=t("predictions.activity.col_avg_latency_help"),
                                ),
                            },
                        )

                        st.divider()

                        # ── Charts ────────────────────────────────────────────────
                        _gc1, _gc2 = st.columns([1, 2])

                        _fig_top = px.bar(
                            df_usumm.head(10).sort_values("total_calls"),
                            y="username", x="total_calls",
                            orientation="h",
                            title=t("predictions.activity.chart_top_users_title"),
                            labels={"username": "", "total_calls": t("predictions.activity.col_calls")},
                            color="total_calls",
                            color_continuous_scale=["#74b9ff", "#0984e3", "#2d3436"],
                            text="total_calls",
                        )
                        _fig_top.update_traces(textposition="outside")
                        _fig_top.update_layout(
                            coloraxis_showscale=False,
                            margin=dict(l=10, r=30, t=50, b=10),
                        )
                        _gc1.plotly_chart(_fig_top, use_container_width=True)

                        if _by_day_rows:
                            _df_day = pd.DataFrame(_by_day_rows)
                            _df_day["date"] = pd.to_datetime(_df_day["date"])
                            _df_day = _df_day.sort_values("date")
                            _fig_day = px.line(
                                _df_day,
                                x="date", y="calls", color="username",
                                title=t("predictions.activity.chart_daily_title"),
                                labels={"date": t("predictions.activity.chart_date_label"), "calls": t("predictions.activity.col_calls"), "username": t("predictions.activity.col_user")},
                                markers=True,
                            )
                            _fig_day.update_layout(legend_title_text=t("predictions.activity.col_user"))
                            _gc2.plotly_chart(_fig_day, use_container_width=True)

                        if _by_model_rows:
                            st.markdown(t("predictions.activity.detail_by_user_model_header"))
                            _df_bm = pd.DataFrame(_by_model_rows)

                            _col_tbl, _col_chart = st.columns([1, 1])

                            _df_bm_disp = _df_bm.copy()
                            _df_bm_disp[t("predictions.activity.col_error_rate")] = _df_bm_disp.apply(
                                lambda r: f"{r['errors'] / r['calls'] * 100:.1f} %"
                                if r["calls"] > 0 else "—",
                                axis=1,
                            )
                            _df_bm_disp[t("predictions.activity.col_avg_latency")] = _df_bm_disp["avg_latency_ms"].apply(
                                lambda v: f"{v} ms" if v is not None else "—"
                            )
                            _col_tbl.dataframe(
                                _df_bm_disp.sort_values(
                                    ["username", "calls"], ascending=[True, False]
                                ).rename(columns={
                                    "username":   t("predictions.activity.col_user"),
                                    "model_name": t("predictions.activity.col_model"),
                                    "calls":      t("predictions.activity.col_calls"),
                                    "errors":     t("predictions.activity.col_errors"),
                                })[[t("predictions.activity.col_user"),
                                    t("predictions.activity.col_model"),
                                    t("predictions.activity.col_calls"),
                                    t("predictions.activity.col_errors"),
                                    t("predictions.activity.col_error_rate"),
                                    t("predictions.activity.col_avg_latency")]],
                                hide_index=True,
                                use_container_width=True,
                            )

                            _fig_stack = px.bar(
                                _df_bm.sort_values("calls", ascending=False),
                                x="username", y="calls", color="model_name",
                                title=t("predictions.activity.chart_stack_title"),
                                labels={
                                    "username":   t("predictions.activity.col_user"),
                                    "calls":      t("predictions.activity.col_calls"),
                                    "model_name": t("predictions.activity.col_model"),
                                },
                                barmode="stack",
                            )
                            _fig_stack.update_layout(
                                legend_title_text=t("predictions.activity.col_model"),
                                xaxis_title=t("predictions.activity.col_user"),
                            )
                            _col_chart.plotly_chart(_fig_stack, use_container_width=True)

        # --- Import / Export observed results ---
        CSV_TEMPLATE = (
            "id_obs,model_name,observed_result,date_time\n"
            "obs-001,iris-classifier,2,2026-05-20 14:32:00\n"
            "obs-002,iris-classifier,1,2026-05-20 14:33:10\n"
            "obs-003,wine-regressor,13.5,2026-05-20\n"
        )

        with st.expander(t("predictions.csv_import.expander_title"), expanded=False):
            st.markdown(t("predictions.csv_import.intro"))

            st.markdown(t("predictions.csv_import.format_header"))
            st.code(
                "id_obs,model_name,observed_result,date_time\n"
                "obs-001,iris-classifier,2,2026-05-20 14:32:00\n"
                "obs-002,iris-classifier,1,2026-05-20\n"
                "obs-003,wine-regressor,13.5,2026-05-20T09:15:00",
                language="text",
            )
            st.markdown(t("predictions.csv_import.columns_table"))
            st.caption(t("predictions.csv_import.format_note"))

            st.download_button(
                t("predictions.csv_import.template_download_btn"),
                data=CSV_TEMPLATE,
                file_name="template_observed_results.csv",
                mime="text/csv",
            )

            uploaded_file = st.file_uploader(t("predictions.csv_import.file_uploader_label"), type=["csv"], key="csv_obs_upload")

            model_name_override = st.text_input(
                t("predictions.csv_import.model_override_label"),
                key="csv_obs_model_override",
                help=t("predictions.csv_import.model_override_help"),
            )
            if uploaded_file is not None and st.button(t("predictions.csv_import.import_btn"), key="csv_obs_submit"):
                try:
                    result = client.upload_observed_results_csv(
                        file_bytes=uploaded_file.read(),
                        filename=uploaded_file.name,
                        model_name=model_name_override.strip() or None,
                    )
                    st.toast(
                        t("predictions.csv_import.import_success", count=result['upserted'], filename=result['filename']),
                        icon="✅",
                    )
                    if result.get("skipped_rows", 0) > 0:
                        st.warning(t("predictions.csv_import.skipped_rows", count=result['skipped_rows']))
                        errors = result.get("parse_errors", [])
                        if errors:
                            st.dataframe(
                                pd.DataFrame(errors),
                                width='stretch',
                                hide_index=True,
                            )
                except Exception as exc:
                    st.error(t("predictions.csv_import.import_error", error=exc))

        # --- GDPR maintenance (admin only) ---
        if st.session_state.get("is_admin", False):
            with st.expander(t("predictions.purge.expander_title"), expanded=False):
                st.caption(t("predictions.purge.caption"))

                col_m1, col_m2 = st.columns(2)
                purge_days = col_m1.slider(
                    t("predictions.purge.days_slider_label"),
                    min_value=7,
                    max_value=365,
                    value=90,
                    format="%d days",
                    key="purge_days_slider",
                )
                purge_model_sel = col_m2.selectbox(
                    t("predictions.purge.model_filter_label"),
                    [t("predictions.filters.all_models")] + (model_names if model_names else []),
                    key="purge_model_sel",
                )
                purge_model_name = None if purge_model_sel == t("predictions.filters.all_models") else purge_model_sel

                col_sim, col_purge = st.columns(2)

                if col_sim.button(
                    t("predictions.purge.simulate_btn"), key="purge_simulate", width='stretch'
                ):
                    try:
                        result = client.purge_predictions(
                            older_than_days=purge_days,
                            model_name=purge_model_name,
                            dry_run=True,
                        )
                        st.info(
                            t("predictions.purge.simulate_result", count=result['deleted_count'])
                        )
                        if result.get("oldest_remaining"):
                            st.caption(
                                t("predictions.purge.oldest_remaining", ts=result['oldest_remaining'])
                            )
                        if result.get("models_affected"):
                            st.caption(t("predictions.purge.models_affected", models=', '.join(result['models_affected'])))
                        if result.get("linked_observed_results_count", 0) > 0:
                            st.warning(
                                t("predictions.purge.linked_obs_warning", count=result['linked_observed_results_count'])
                            )
                    except Exception as exc:
                        st.error(t("predictions.purge.simulate_error", error=exc))

                @st.dialog(t("predictions.purge.dialog_title"))
                def _confirm_purge_dialog():
                    st.warning(
                        t("predictions.purge.dialog_warning", days=purge_days)
                    )
                    if purge_model_name:
                        st.info(t("predictions.purge.dialog_target_model", model=purge_model_name))
                    else:
                        st.info(t("predictions.purge.dialog_all_models"))
                    st.markdown(t("predictions.purge.dialog_irreversible"))
                    if st.button(
                        t("predictions.purge.dialog_confirm_btn"), type="primary", key="purge_dialog_confirm"
                    ):
                        try:
                            result = client.purge_predictions(
                                older_than_days=purge_days,
                                model_name=purge_model_name,
                                dry_run=False,
                            )
                            st.toast(
                                t("predictions.purge.purge_toast", count=result['deleted_count']), icon="✅"
                            )
                            if result.get("linked_observed_results_count", 0) > 0:
                                st.warning(
                                    t("predictions.purge.linked_obs_lost", count=result['linked_observed_results_count'])
                                )
                            st.rerun()
                        except Exception as exc:
                            st.toast(t("predictions.purge.purge_error", error=exc), icon="❌")

                if col_purge.button(
                    t("predictions.purge.confirm_btn"),
                    key="purge_open_dialog",
                    type="primary",
                    width='stretch',
                ):
                    _confirm_purge_dialog()


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — Batch predictions
# ───────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.subheader(t("predictions.batch.subheader"))
    st.caption(t("predictions.batch.caption"))

    with st.expander(t("predictions.batch.format_expander_title"), expanded=False):
        st.markdown(t("predictions.batch.format_description"))
        example_df = pd.DataFrame(
            {
                "id_obs": ["obs-001", "obs-002", "obs-003"],
                "sepal_length": [5.1, 6.3, 4.7],
                "sepal_width": [3.5, 2.9, 3.2],
                "petal_length": [1.4, 5.6, 1.3],
                "petal_width": [0.2, 1.8, 0.2],
            }
        )
        st.dataframe(example_df, width='stretch', hide_index=True)
        st.download_button(
            t("predictions.batch.example_download_btn"),
            data=example_df.to_csv(index=False),
            file_name="exemple_batch_iris.csv",
            mime="text/csv",
            key="batch_example_download",
        )

    # --- Model + version selection ---
    try:
        all_models = get_models_cached(
            st.session_state.get("api_url"), st.session_state.get("api_token")
        )
        model_names_batch = sorted({m["name"] for m in all_models})
    except Exception:
        all_models = []
        model_names_batch = []

    if not model_names_batch:
        st.warning(t("predictions.warnings.no_model"))
        st.stop()

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        batch_search = st.text_input(
            t("predictions.filters.search_label"), key="batch_model_search", placeholder=t("predictions.filters.search_placeholder")
        )
        batch_filtered = (
            [n for n in model_names_batch if batch_search.lower() in n.lower()]
            if batch_search
            else model_names_batch
        )
        batch_model = st.selectbox(
            t("predictions.batch.target_model_label"), batch_filtered or model_names_batch, key="batch_model_sel"
        )

    batch_versions = [t("predictions.batch.production_auto")] + sorted(
        {m["version"] for m in all_models if m["name"] == batch_model},
        reverse=True,
    )
    batch_version_sel = col_b2.selectbox(t("predictions.batch.version_label"), batch_versions, key="batch_version_sel")
    batch_version = None if batch_version_sel == t("predictions.batch.production_auto") else batch_version_sel

    batch_file = st.file_uploader(
        t("predictions.batch.file_uploader_label"),
        type=["csv", "parquet"],
        key="batch_file_uploader",
        help=t("predictions.batch.file_uploader_help"),
    )

    if batch_file is not None:
        try:
            fname = batch_file.name.lower()
            if fname.endswith(".parquet"):
                df_input = pd.read_parquet(io.BytesIO(batch_file.read()))
            else:
                df_input = pd.read_csv(io.BytesIO(batch_file.read()))
        except Exception as exc:
            st.error(t("predictions.batch.file_read_error", error=exc))
            st.stop()

        st.caption(
            t("predictions.batch.file_loaded_caption", filename=batch_file.name, rows=len(df_input), cols=len(df_input.columns))
        )
        st.dataframe(df_input.head(10), width='stretch', hide_index=True)

        if st.button(t("predictions.batch.run_btn"), type="primary", key="batch_run"):
            id_obs_col = None
            if "id_obs" in df_input.columns:
                id_obs_col = df_input["id_obs"].astype(str).tolist()
                feature_df = df_input.drop(columns=["id_obs"])
            else:
                feature_df = df_input

            rows_payload = feature_df.to_dict(orient="records")

            if id_obs_col is not None:
                inputs = [
                    {"features": row, "id_obs": obs_id}
                    for row, obs_id in zip(rows_payload, id_obs_col)
                ]
            else:
                inputs = [{"features": row} for row in rows_payload]

            with st.spinner(t("predictions.batch.scoring_spinner", count=len(rows_payload))):
                try:
                    import requests as _requests

                    result = client.predict_batch_from_df(
                        model_name=batch_model,
                        rows=[inp["features"] for inp in inputs],
                        model_version=batch_version,
                    )
                except _requests.exceptions.Timeout:
                    st.error(t("predictions.batch.timeout_error"))
                    st.stop()
                except Exception as exc:
                    detail = str(exc)
                    try:
                        import json as _json

                        body = _json.loads(str(exc).split(" - ", 1)[-1])
                        detail = body.get("detail", detail)
                    except Exception:
                        pass
                    st.error(t("predictions.batch.scoring_error", detail=detail))
                    st.stop()

            predictions_out = result.get("predictions", [])
            used_version = result.get("model_version", "—")

            st.toast(
                t("predictions.batch.scoring_toast", count=len(predictions_out), model=batch_model, version=used_version),
                icon="✅",
            )

            df_result = df_input.copy()
            df_result["prediction"] = [p.get("prediction") for p in predictions_out]

            first_proba = next(
                (p.get("probability") for p in predictions_out if p.get("probability")), None
            )
            if first_proba is not None:
                n_classes = len(first_proba)
                for i in range(n_classes):
                    df_result[f"proba_class_{i}"] = [
                        (p.get("probability") or [None] * n_classes)[i] for p in predictions_out
                    ]

            if any(p.get("low_confidence") is not None for p in predictions_out):
                df_result["low_confidence"] = [p.get("low_confidence") for p in predictions_out]

            st.markdown(t("predictions.batch.preview_header"))
            st.dataframe(df_result.head(50), width='stretch', hide_index=True)

            csv_out = df_result.to_csv(index=False)
            st.download_button(
                label=t("predictions.batch.download_full_btn", count=len(df_result)),
                data=csv_out,
                file_name=f"predictions_batch_{batch_model}.csv",
                mime="text/csv",
                key="batch_download_btn",
            )

            with st.expander(t("predictions.batch.summary_expander_title")):
                pred_series = df_result["prediction"]
                st.markdown(t("predictions.batch.summary_count", count=len(pred_series)))
                try:
                    numeric_preds = pred_series.astype(float)
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    col_s1.metric(t("predictions.batch.stat_mean"), f"{numeric_preds.mean():.4f}")
                    col_s2.metric(t("predictions.batch.stat_median"), f"{numeric_preds.median():.4f}")
                    col_s3.metric(t("predictions.batch.stat_min"), f"{numeric_preds.min():.4f}")
                    col_s4.metric(t("predictions.batch.stat_max"), f"{numeric_preds.max():.4f}")
                except (ValueError, TypeError):
                    dist = pred_series.value_counts()
                    st.dataframe(
                        dist.rename_axis(t("predictions.batch.class_col")).reset_index(name="Count"),
                        width='stretch',
                        hide_index=True,
                    )
