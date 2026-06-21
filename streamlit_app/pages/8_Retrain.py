"""
Centralized management of retraining jobs and cron schedules
"""

import re
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from utils.api_client import get_models as get_models_cached
from utils.auth import get_client, require_auth
from utils.i18n import t

st.set_page_config(page_title=t("retrain.page_title"), page_icon="🔄", layout="wide")
require_auth()

is_admin = st.session_state.get("is_admin", False)
if not is_admin:
    st.error(t("retrain.admin_only"))
    st.stop()

st.title(t("retrain.title"))
st.caption(t("retrain.caption"))

client = get_client()


def fetch_models(api_url, token):
    return get_models_cached(api_url, token)


def reload():
    st.cache_data.clear()
    st.rerun()


try:
    models = fetch_models(st.session_state.get("api_url"), st.session_state.get("api_token"))
except Exception as e:
    st.error(t("retrain.error_load_models", error=e))
    st.stop()

if not models:
    st.info(t("retrain.no_models"))
    st.stop()

# Pre-select model from navigation link or ?model= query param
_nav = st.session_state.pop("_nav_model", None) or st.query_params.get("model")
if _nav:
    _trainable = [m for m in models if m.get("train_script_object_key")]
    _match = next((f"{m['name']} v{m['version']}" for m in _trainable if m["name"] == _nav), None)
    if _match:
        st.session_state["retrain_select"] = _match

tab_overview, tab_manual, tab_schedule, tab_policy, tab_history = st.tabs(
    [
        t("retrain.tab_overview"),
        t("retrain.tab_manual"),
        t("retrain.tab_schedule"),
        t("retrain.tab_policy"),
        t("retrain.tab_history"),
    ]
)

# ─── Tab 1 — Schedule overview ──────────────────────────────

with tab_overview:
    st.subheader(t("retrain.overview.subheader"))

    rows = []
    for m in models:
        sched = m.get("retrain_schedule") or {}
        if not sched:
            continue
        has_script = bool(m.get("train_script_object_key"))

        cron = sched.get("cron") or "—"
        last_run = sched.get("last_run_at")
        next_run = sched.get("next_run_at")
        enabled = sched.get("enabled", True)
        badge = t("retrain.overview.badge_active") if enabled else t("retrain.overview.badge_disabled")
        lookback = sched.get("lookback_days")
        auto_promote = sched.get("auto_promote", False)

        rows.append(
            {
                t("retrain.overview.col_model"): m.get("name", ""),
                t("retrain.overview.col_version"): m.get("version", ""),
                t("retrain.overview.col_script"): "✅" if has_script else "❌",
                t("retrain.overview.col_cron"): cron,
                t("retrain.overview.col_lookback"): lookback,
                t("retrain.overview.col_auto_promote"): "✅" if auto_promote else "❌",
                t("retrain.overview.col_last_retrain"): (
                    pd.to_datetime(last_run).strftime("%Y-%m-%d %H:%M") if last_run else "—"
                ),
                t("retrain.overview.col_next_retrain"): (
                    pd.to_datetime(next_run).strftime("%Y-%m-%d %H:%M") if next_run else "—"
                ),
                t("retrain.overview.col_status"): badge,
            }
        )

    col_model = t("retrain.overview.col_model")
    col_version = t("retrain.overview.col_version")
    col_script = t("retrain.overview.col_script")
    col_cron = t("retrain.overview.col_cron")
    col_lookback = t("retrain.overview.col_lookback")
    col_auto_promote = t("retrain.overview.col_auto_promote")
    col_last_retrain = t("retrain.overview.col_last_retrain")
    col_next_retrain = t("retrain.overview.col_next_retrain")
    col_status = t("retrain.overview.col_status")

    st.dataframe(
        pd.DataFrame(rows),
        width='stretch',
        hide_index=True,
        column_config={
            col_model: st.column_config.TextColumn(
                col_model,
                help=t("retrain.overview.help_model"),
            ),
            col_version: st.column_config.TextColumn(
                col_version,
                help=t("retrain.overview.help_version"),
            ),
            col_script: st.column_config.TextColumn(
                col_script,
                help=t("retrain.overview.help_script"),
            ),
            col_cron: st.column_config.TextColumn(
                col_cron,
                help=t("retrain.overview.help_cron"),
            ),
            col_lookback: st.column_config.NumberColumn(
                col_lookback,
                help=t("retrain.overview.help_lookback"),
                format="%d j",
            ),
            col_auto_promote: st.column_config.TextColumn(
                col_auto_promote,
                help=t("retrain.overview.help_auto_promote"),
            ),
            col_last_retrain: st.column_config.TextColumn(
                col_last_retrain,
                help=t("retrain.overview.help_last_retrain"),
            ),
            col_next_retrain: st.column_config.TextColumn(
                col_next_retrain,
                help=t("retrain.overview.help_next_retrain"),
            ),
            col_status: st.column_config.TextColumn(
                col_status,
                help=t("retrain.overview.help_status"),
            ),
        },
    )

    active = sum(
        1
        for m in models
        if (m.get("retrain_schedule") or {}).get("enabled")
        and (m.get("retrain_schedule") or {}).get("cron")
    )
    trainable_count = sum(1 for m in models if m.get("train_script_object_key"))

    col1, col2, col3 = st.columns(3)
    col1.metric(t("retrain.overview.metric_active_schedules"), active)
    col2.metric(t("retrain.overview.metric_trainable_models"), trainable_count)
    col3.metric(t("retrain.overview.metric_total_models"), len(models))

# ─── Tab 2 — Manual retrain ────────────────────────────────────────────

with tab_manual:
    st.subheader(t("retrain.manual.subheader"))

    trainable = [m for m in models if m.get("train_script_object_key")]
    if not trainable:
        st.warning(t("retrain.manual.no_trainable"))
    else:
        model_opts = {f"{m['name']} v{m['version']}": m for m in trainable}
        _col_s, _col_sel = st.columns([1, 2])
        with _col_s:
            retrain_search = st.text_input(
                t("retrain.filter_by_name"), key="retrain_search", placeholder=t("retrain.search_placeholder")
            )
        retrain_keys = (
            [k for k in model_opts if retrain_search.lower() in k.lower()]
            if retrain_search
            else list(model_opts.keys())
        )
        with _col_sel:
            selected_label = st.selectbox(
                t("retrain.manual.select_model"), retrain_keys or list(model_opts.keys()), key="retrain_select"
            )
        sel = model_opts[selected_label]

        with st.form("manual_retrain_form"):
            col_s, col_e = st.columns(2)
            with col_s:
                start_date = st.date_input(
                    t("retrain.manual.start_date"),
                    value=date.today() - timedelta(days=30),
                    key="manual_start",
                )
            with col_e:
                end_date = st.date_input(
                    t("retrain.manual.end_date"),
                    value=date.today(),
                    key="manual_end",
                )
            new_version_input = st.text_input(
                t("retrain.manual.new_version_label"),
                value="",
                placeholder=f"{sel['version']}-retrain-YYYYMMDDHHMMSS",
                key="manual_new_version",
                help=t("retrain.manual.help_new_version"),
            )
            set_prod = st.checkbox(
                t("retrain.manual.set_production"),
                value=False,
                key="manual_set_prod",
            )
            submitted = st.form_submit_button(t("retrain.manual.submit_btn"), type="primary")

        if submitted:
            _version_re = re.compile(r"^\d+\.\d+(\.\d+)?(-[a-zA-Z0-9]+)*$")
            _version_val = new_version_input.strip()
            _version_ok = not _version_val or _version_re.match(_version_val)
            if start_date > end_date:
                st.error(t("retrain.manual.error_date_order"))
            elif not _version_ok:
                st.error(t("retrain.manual.error_version_format"))
            else:
                with st.spinner(t("retrain.manual.spinner")):
                    try:
                        import time as _time
                        enqueued = client.retrain_model(
                            name=sel["name"],
                            version=sel["version"],
                            start_date=str(start_date),
                            end_date=str(end_date),
                            new_version=new_version_input.strip() or None,
                            set_production=set_prod,
                        )
                        job_id = enqueued["job_id"]
                        _TERMINAL = {"success", "failed", "cancelled"}
                        _deadline = _time.monotonic() + 750
                        job = enqueued
                        while job.get("status") not in _TERMINAL:
                            if _time.monotonic() > _deadline:
                                break
                            _time.sleep(3)
                            job = client.get_job_status(job_id)
                        _job_result = job.get("result") or {}
                        result = {
                            "success": job.get("status") == "success",
                            "new_version": job.get("new_version") or _job_result.get("new_version"),
                            "error": job.get("error"),
                            "stdout": job.get("logs", ""),
                            "stderr": "",
                            "auto_promoted": _job_result.get("auto_promoted"),
                            "auto_promote_reason": _job_result.get("auto_promote_reason"),
                        }
                        if result.get("success"):
                            st.toast(
                                t("retrain.manual.toast_success", new_version=result['new_version']),
                                icon="✅",
                            )
                            auto_promoted = result.get("auto_promoted")
                            if auto_promoted is True:
                                st.info(t("retrain.manual.auto_promoted_yes"))
                            elif auto_promoted is False:
                                st.warning(
                                    t(
                                        "retrain.manual.auto_promoted_no",
                                        reason=result.get('auto_promote_reason') or '—',
                                    )
                                )
                        else:
                            st.toast(
                                t(
                                    "retrain.manual.toast_failure",
                                    error=result.get('error', t("retrain.manual.unknown_error")),
                                ),
                                icon="❌",
                            )
                        with st.expander(t("retrain.manual.logs_stdout"), expanded=not result.get("success")):
                            st.code(result.get("stdout", t("retrain.manual.logs_empty")), language="text")
                        with st.expander(t("retrain.manual.logs_stderr"), expanded=not result.get("success")):
                            st.code(result.get("stderr", t("retrain.manual.logs_empty")), language="text")
                        if result.get("success"):
                            reload()
                    except Exception as e:
                        st.toast(t("retrain.manual.toast_error", error=e), icon="❌")

# ─── Tab 3 — Cron schedule ─────────────────────────────────────────────

with tab_schedule:
    st.subheader(t("retrain.schedule.subheader"))

    trainable_sched = [m for m in models if m.get("train_script_object_key")]
    if not trainable_sched:
        st.warning(t("retrain.manual.no_trainable"))
    else:
        with st.expander(t("retrain.schedule.help_expander"), expanded=False):
            st.markdown(t("retrain.schedule.help_table"))

        sched_opts = {f"{m['name']} v{m['version']}": m for m in trainable_sched}
        _col_s, _col_sel = st.columns([1, 2])
        with _col_s:
            sched_search = st.text_input(
                t("retrain.filter_by_name"), key="sched_search", placeholder=t("retrain.search_placeholder")
            )
        sched_keys = (
            [k for k in sched_opts if sched_search.lower() in k.lower()]
            if sched_search
            else list(sched_opts.keys())
        )
        with _col_sel:
            sched_label = st.selectbox(
                t("retrain.schedule.select_model"), sched_keys or list(sched_opts.keys()), key="sched_select"
            )
        sched_sel = sched_opts[sched_label]
        existing_sched = sched_sel.get("retrain_schedule") or {}

        if existing_sched:
            _s_enabled = existing_sched.get("enabled", True)
            _s_cron = existing_sched.get("cron") or "—"
            _s_lookback = existing_sched.get("lookback_days")
            _s_auto_promote = existing_sched.get("auto_promote", False)
            _s_last_run = existing_sched.get("last_run_at")
            _s_next_run = existing_sched.get("next_run_at")

            st.markdown(t("retrain.schedule.current_schedule"))
            _s_icon = "🟢" if _s_enabled else "🔴"
            st.markdown(
                f"{_s_icon} "
                + (t("retrain.schedule.field_enabled_on") if _s_enabled else t("retrain.schedule.field_enabled_off"))
                + f"  —  {t('retrain.schedule.cron_label')} : `{_s_cron}`"
            )
            _sc1, _sc2, _sc3, _sc4 = st.columns(4)
            with _sc1:
                st.metric(
                    t("retrain.schedule.field_lookback_label"),
                    f"{_s_lookback} j" if _s_lookback is not None else "—",
                )
            with _sc2:
                st.metric(
                    t("retrain.schedule.field_auto_promote_label"),
                    t("retrain.schedule.field_yes") if _s_auto_promote else t("retrain.schedule.field_no"),
                )
            with _sc3:
                st.metric(
                    t("retrain.schedule.field_last_run_label"),
                    pd.to_datetime(_s_last_run).strftime("%Y-%m-%d %H:%M") if _s_last_run else "—",
                )
            with _sc4:
                st.metric(
                    t("retrain.schedule.field_next_run_label"),
                    pd.to_datetime(_s_next_run).strftime("%Y-%m-%d %H:%M") if _s_next_run else "—",
                )

        with st.form("schedule_form"):
            col_cron, col_lb = st.columns(2)
            with col_cron:
                cron_val = st.text_input(
                    t("retrain.schedule.cron_label"),
                    value=existing_sched.get("cron") or "",
                    placeholder="0 3 * * 1",
                    help=t("retrain.schedule.cron_help"),
                )
            with col_lb:
                lookback = st.slider(
                    t("retrain.schedule.lookback_label"),
                    min_value=1,
                    max_value=365,
                    value=int(existing_sched.get("lookback_days") or 30),
                    help=t("retrain.schedule.lookback_help"),
                )
            col_ap, col_en, col_btn = st.columns([2, 2, 1])
            with col_ap:
                auto_promote_sched = st.checkbox(
                    t("retrain.schedule.auto_promote_label"),
                    value=bool(existing_sched.get("auto_promote", False)),
                    key="sched_auto_promote",
                    help=t("retrain.schedule.auto_promote_help"),
                )
            with col_en:
                enabled_sched = st.checkbox(
                    t("retrain.schedule.enabled_label"),
                    value=bool(existing_sched.get("enabled", True)),
                    key="sched_enabled",
                    help=t("retrain.schedule.enabled_help"),
                )
            with col_btn:
                save_sched = st.form_submit_button(
                    t("retrain.schedule.save_btn"), type="primary", use_container_width=True
                )

        if save_sched:
            try:
                result = client.set_schedule(
                    name=sched_sel["name"],
                    version=sched_sel["version"],
                    cron=cron_val.strip() or None,
                    lookback_days=lookback,
                    auto_promote=auto_promote_sched,
                    enabled=enabled_sched,
                )
                st.toast(
                    t("retrain.schedule.toast_saved", name=sched_sel['name'], version=sched_sel['version']),
                    icon="✅",
                )
                saved_sched = result.get("retrain_schedule") or {}
                if saved_sched.get("next_run_at"):
                    st.info(t("retrain.schedule.next_run_info", next_run=saved_sched['next_run_at']))
                reload()
            except Exception as e:
                st.toast(t("retrain.schedule.toast_error", error=e), icon="❌")

# ─── Tab 4 — Auto-promotion policy ───────────────────────────────

with tab_policy:
    st.subheader(t("retrain.policy.subheader"))
    st.caption(t("retrain.policy.caption"))

    model_names = sorted({m["name"] for m in models})
    _col_s, _col_sel = st.columns([1, 2])
    with _col_s:
        policy_search = st.text_input(
            t("retrain.filter_by_name"), key="policy_model_search", placeholder=t("retrain.search_placeholder")
        )
    policy_filtered = (
        [n for n in model_names if policy_search.lower() in n.lower()]
        if policy_search
        else model_names
    )
    with _col_sel:
        policy_name = st.selectbox(t("retrain.policy.model_label"), policy_filtered or model_names, key="policy_model_select")

    matching = [m for m in models if m["name"] == policy_name]
    current_policy: dict = {}
    for m in matching:
        if m.get("promotion_policy"):
            current_policy = m["promotion_policy"]
            break

    is_classification = any(m.get("classes") for m in matching)

    # ── Pretty display of current policy ──────────────────────────
    if current_policy:
        _auto_promote = current_policy.get("auto_promote", False)
        _min_accuracy = current_policy.get("min_accuracy")
        _min_auc = current_policy.get("min_auc")
        _max_mae = current_policy.get("max_mae")
        _max_latency = current_policy.get("max_latency_p95_ms")
        _min_samples = current_policy.get("min_sample_validation", 100)

        st.markdown(t("retrain.policy.current_policy_title", name=policy_name))
        _status_icon = "🟢" if _auto_promote else "🔴"
        st.markdown(
            f"{_status_icon} **{t('retrain.policy.field_auto_promote_label')} :** "
            + (t("retrain.policy.field_auto_promote_on") if _auto_promote else t("retrain.policy.field_auto_promote_off"))
        )
        _c1, _c2, _c3, _c4 = st.columns(4)
        with _c1:
            st.metric(
                t("retrain.policy.field_min_accuracy_label"),
                f"{_min_accuracy:.0%}" if _min_accuracy is not None else t("retrain.policy.field_not_set"),
            )
        with _c2:
            if is_classification:
                st.metric(
                    t("retrain.policy.field_min_auc_label"),
                    f"{_min_auc:.2f}" if _min_auc is not None else t("retrain.policy.field_not_set"),
                )
            else:
                st.metric(
                    t("retrain.policy.field_max_mae_label"),
                    f"{_max_mae:.4f}" if _max_mae is not None else t("retrain.policy.field_not_set"),
                )
        with _c3:
            st.metric(
                t("retrain.policy.field_max_latency_label"),
                f"{_max_latency:.0f} ms" if _max_latency is not None else t("retrain.policy.field_not_set"),
            )
        with _c4:
            st.metric(
                t("retrain.policy.field_min_samples_label"),
                str(_min_samples),
            )
    else:
        st.info(t("retrain.policy.no_policy"))

    st.divider()

    # ── Form ──────────────────────────────────────────────────────
    st.info(t("retrain.policy.form_intro"))

    # Defaults for widgets that may not be rendered
    min_auc_enabled = False
    min_auc = float(current_policy.get("min_auc") or 0.8)
    max_mae_enabled = False
    max_mae = min(float(current_policy.get("max_mae") or 0.1), 1.0)

    with st.form("policy_form"):
        st.markdown(f"#### 🚀 {t('retrain.policy.section_activation')}")
        auto_promote_policy = st.checkbox(
            t("retrain.policy.label_auto_promote"),
            value=bool(current_policy.get("auto_promote", False)),
            help=t("retrain.policy.help_auto_promote"),
        )

        st.divider()
        st.markdown(f"#### 📊 {t('retrain.policy.section_performance')}")
        col_l, col_r = st.columns(2)
        with col_l:
            min_acc_enabled = st.checkbox(
                t("retrain.policy.enable_min_accuracy"),
                value=current_policy.get("min_accuracy") is not None,
            )
            min_acc = st.slider(
                t("retrain.policy.label_min_accuracy"),
                min_value=0.0,
                max_value=1.0,
                value=float(current_policy.get("min_accuracy") or 0.9),
                step=0.01,
                help=t("retrain.policy.help_min_accuracy"),
            )
        with col_r:
            if is_classification:
                min_auc_enabled = st.checkbox(
                    t("retrain.policy.enable_min_auc"),
                    value=current_policy.get("min_auc") is not None,
                )
                min_auc = st.slider(
                    t("retrain.policy.label_min_auc"),
                    min_value=0.5,
                    max_value=1.0,
                    value=float(current_policy.get("min_auc") or 0.8),
                    step=0.01,
                    help=t("retrain.policy.help_min_auc"),
                )
            else:
                max_mae_enabled = st.checkbox(
                    t("retrain.policy.enable_max_mae"),
                    value=current_policy.get("max_mae") is not None,
                )
                max_mae = st.slider(
                    t("retrain.policy.label_max_mae"),
                    min_value=0.0,
                    max_value=1.0,
                    value=min(float(current_policy.get("max_mae") or 0.1), 1.0),
                    step=0.001,
                    help=t("retrain.policy.help_max_mae"),
                )

        st.divider()
        st.markdown(f"#### ⚡ {t('retrain.policy.section_speed')}")
        max_latency_enabled = st.checkbox(
            t("retrain.policy.enable_max_latency"),
            value=current_policy.get("max_latency_p95_ms") is not None,
        )
        max_latency = st.number_input(
            t("retrain.policy.label_max_latency"),
            min_value=1,
            value=int(current_policy.get("max_latency_p95_ms") or 200),
            step=10,
            help=t("retrain.policy.help_max_latency"),
        )

        st.divider()
        st.markdown(f"#### 🔢 {t('retrain.policy.section_validation')}")
        min_samples = st.number_input(
            t("retrain.policy.label_min_samples"),
            min_value=1,
            value=int(current_policy.get("min_sample_validation") or 100),
            step=1,
            help=t("retrain.policy.help_min_samples"),
        )

        save_policy = st.form_submit_button(t("retrain.policy.save_btn"), type="primary")

    if save_policy:
        try:
            result = client.set_policy(
                name=policy_name,
                min_accuracy=min_acc if min_acc_enabled else None,
                min_auc=min_auc if min_auc_enabled else None,
                max_mae=max_mae if max_mae_enabled else None,
                max_latency_p95_ms=max_latency if max_latency_enabled else None,
                min_sample_validation=int(min_samples),
                auto_promote=auto_promote_policy,
            )
            updated = result.get("updated_versions", 0)
            st.toast(
                t("retrain.policy.toast_saved", name=policy_name, updated=updated),
                icon="✅",
            )
            reload()
        except Exception as e:
            st.toast(t("retrain.policy.toast_error", error=e), icon="❌")

# ─── Tab 5 — Retrain history ──────────────────────────────────

with tab_history:
    st.subheader(t("retrain.history.subheader"))

    model_names_hist = sorted({m["name"] for m in models})
    _col_search, _col_select = st.columns([1, 2])
    with _col_search:
        hist_search = st.text_input(
            t("retrain.filter_by_name"), key="hist_model_search", placeholder=t("retrain.search_placeholder")
        )
    hist_filtered = (
        [n for n in model_names_hist if hist_search.lower() in n.lower()]
        if hist_search
        else model_names_hist
    )
    with _col_select:
        hist_model_name = st.selectbox(
            t("retrain.history.model_label"), hist_filtered or model_names_hist, key="hist_model_select"
        )

    try:
        retrain_data = client.get_retrain_history(name=hist_model_name, limit=100)
    except Exception as e:
        st.error(t("retrain.history.error_load", error=e))
        retrain_data = {"history": [], "total": 0}

    retrain_entries = retrain_data.get("history", [])
    total_retrains = retrain_data.get("total", 0)

    if not retrain_entries:
        st.info(t("retrain.history.no_entries"))
    else:
        rows = []
        for e in retrain_entries:
            accuracy = e.get("accuracy")
            auc = e.get("auc")
            f1 = e.get("f1_score")
            auto_promoted = e.get("auto_promoted")
            if auto_promoted is True:
                promo_badge = t("retrain.history.promo_auto")
            elif e.get("trained_by") and auto_promoted is None:
                promo_badge = "—"
            else:
                promo_badge = t("retrain.history.promo_failed")
            rows.append(
                {
                    t("retrain.history.col_date"): pd.to_datetime(e["timestamp"]).strftime("%Y-%m-%d %H:%M"),
                    t("retrain.history.col_new_version"): e.get("new_version", "—"),
                    t("retrain.history.col_trained_by"): e.get("trained_by") or "—",
                    t("retrain.history.col_source_version"): e.get("source_version") or "—",
                    t("retrain.history.col_accuracy"): round(accuracy, 4) if accuracy is not None else None,
                    t("retrain.history.col_auc"): round(auc, 4) if auc is not None else None,
                    t("retrain.history.col_f1"): round(f1, 4) if f1 is not None else None,
                    t("retrain.history.col_auto_promotion"): promo_badge,
                    t("retrain.history.col_reason"): e.get("auto_promote_reason") or "—",
                    "n_rows": e.get("n_rows"),
                    t("retrain.history.col_train_start"): e.get("train_start_date") or "—",
                    t("retrain.history.col_train_end"): e.get("train_end_date") or "—",
                }
            )

        col_date = t("retrain.history.col_date")
        col_new_version = t("retrain.history.col_new_version")
        col_trained_by = t("retrain.history.col_trained_by")
        col_source_version = t("retrain.history.col_source_version")
        col_accuracy = t("retrain.history.col_accuracy")
        col_auc = t("retrain.history.col_auc")
        col_f1 = t("retrain.history.col_f1")
        col_auto_promotion = t("retrain.history.col_auto_promotion")
        col_reason = t("retrain.history.col_reason")
        col_train_start = t("retrain.history.col_train_start")
        col_train_end = t("retrain.history.col_train_end")

        df_hist = pd.DataFrame(rows)
        st.dataframe(
            df_hist,
            width='stretch',
            hide_index=True,
            column_config={
                col_date: st.column_config.TextColumn(
                    col_date,
                    help=t("retrain.history.help_date"),
                ),
                col_new_version: st.column_config.TextColumn(
                    col_new_version,
                    help=t("retrain.history.help_new_version"),
                ),
                col_trained_by: st.column_config.TextColumn(
                    col_trained_by,
                    help=t("retrain.history.help_trained_by"),
                ),
                col_source_version: st.column_config.TextColumn(
                    col_source_version,
                    help=t("retrain.history.help_source_version"),
                ),
                col_accuracy: st.column_config.NumberColumn(
                    col_accuracy,
                    help=t("retrain.history.help_accuracy"),
                    format="%.4f",
                ),
                col_auc: st.column_config.NumberColumn(
                    col_auc,
                    help=t("retrain.history.help_auc"),
                    format="%.4f",
                ),
                col_f1: st.column_config.NumberColumn(
                    col_f1,
                    help=t("retrain.history.help_f1"),
                    format="%.4f",
                ),
                col_auto_promotion: st.column_config.TextColumn(
                    col_auto_promotion,
                    help=t("retrain.history.help_auto_promotion"),
                ),
                col_reason: st.column_config.TextColumn(
                    col_reason,
                    help=t("retrain.history.help_reason"),
                ),
                "n_rows": st.column_config.NumberColumn(
                    "n_rows",
                    help=t("retrain.history.help_n_rows"),
                ),
                col_train_start: st.column_config.TextColumn(
                    col_train_start,
                    help=t("retrain.history.help_train_start"),
                ),
                col_train_end: st.column_config.TextColumn(
                    col_train_end,
                    help=t("retrain.history.help_train_end"),
                ),
            },
        )
        st.caption(t("retrain.history.total_caption", total=total_retrains))

        # Accuracy progression chart
        chart_df = df_hist[df_hist[col_accuracy].notna()].copy()
        if not chart_df.empty:
            st.markdown(t("retrain.history.accuracy_chart_title"))
            chart_df = chart_df.sort_values(col_date)
            chart_df = chart_df.rename(columns={col_date: "index"}).set_index("index")
            st.line_chart(chart_df[[col_accuracy, col_f1]].dropna(how="all"))

        # ─── Feature importance delta ─────────────────────────────
        st.markdown("---")
        st.markdown(t("retrain.history.fi_delta_title"))
        st.caption(t("retrain.history.fi_delta_caption"))

        retrain_options = {
            f"{e.get('new_version', '—')} ← {e.get('source_version', '—')} "
            f"({pd.to_datetime(e['timestamp']).strftime('%Y-%m-%d %H:%M')})": e
            for e in retrain_entries
        }

        selected_retrain_label = st.selectbox(
            t("retrain.history.fi_select_label"),
            list(retrain_options.keys()),
            key="hist_fi_select",
        )
        selected_event = retrain_options[selected_retrain_label]
        source_version = selected_event.get("source_version")
        new_version = selected_event.get("new_version")

        fi_baseline = None
        fi_new = None
        fi_error_baseline = None
        fi_error_new = None

        with st.spinner(t("retrain.history.fi_spinner")):
            try:
                fi_baseline = client.get_feature_importance(hist_model_name, version=source_version, days=365)
            except Exception as exc:
                fi_error_baseline = str(exc)
            try:
                fi_new = client.get_feature_importance(hist_model_name, version=new_version, days=365)
            except Exception as exc:
                fi_error_new = str(exc)

        if fi_error_baseline:
            st.warning(
                t("retrain.history.fi_error_baseline", version=source_version, error=fi_error_baseline)
            )
        if fi_error_new:
            st.warning(
                t("retrain.history.fi_error_new", version=new_version, error=fi_error_new)
            )

        if fi_baseline and fi_new:
            import plotly.graph_objects as go

            baseline_fi = fi_baseline.get("feature_importance") or {}
            new_fi = fi_new.get("feature_importance") or {}
            baseline_sample = fi_baseline.get("sample_size", 0)
            new_sample = fi_new.get("sample_size", 0)

            # Fallback: use model-metadata feature_importances when no SHAP predictions available
            def _meta_fi(version_str):
                meta = next(
                    (m for m in models if m["name"] == hist_model_name and str(m.get("version", "")) == str(version_str or "")),
                    None,
                )
                raw = (meta or {}).get("feature_importances") or {}
                return {f: {"mean_abs_shap": v} for f, v in raw.items()} if raw else {}

            if not baseline_fi:
                baseline_fi = _meta_fi(source_version)
            if not new_fi:
                new_fi = _meta_fi(new_version)

            all_features = set(baseline_fi.keys()) | set(new_fi.keys())

            if not all_features:
                st.info(t("retrain.history.fi_no_data"))
            else:
                comparison = []
                for feature in all_features:
                    base_val = (baseline_fi.get(feature) or {}).get("mean_abs_shap", 0.0)
                    new_val = (new_fi.get(feature) or {}).get("mean_abs_shap", 0.0)
                    delta = new_val - base_val
                    if base_val > 0:
                        delta_pct = abs(delta) / base_val * 100
                    elif new_val > 0:
                        delta_pct = 100.0
                    else:
                        delta_pct = 0.0
                    comparison.append(
                        {
                            "feature": feature,
                            "baseline": base_val,
                            "new": new_val,
                            "delta": delta,
                            "delta_pct": delta_pct,
                        }
                    )

                comparison.sort(key=lambda x: max(x["baseline"], x["new"]), reverse=True)
                top10 = comparison[:10]

                total_features = len(comparison)
                stable_count = sum(1 for c in comparison if c["delta_pct"] < 10)
                stability_pct = round(stable_count / total_features * 100) if total_features else 0

                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric(
                    t("retrain.history.fi_stability_metric"),
                    f"{stability_pct}%",
                    help=t("retrain.history.fi_stability_help"),
                )
                col_m2.metric(t("retrain.history.fi_sample_baseline", version=source_version), baseline_sample)
                col_m3.metric(t("retrain.history.fi_sample_new", version=new_version), new_sample)

                features_names = [c["feature"] for c in top10]
                baseline_vals = [c["baseline"] for c in top10]
                new_vals = [c["new"] for c in top10]
                delta_pcts = [c["delta_pct"] for c in top10]

                bar_colors_new = [
                    "#EF553B" if dp > 30 else ("#FF7F0E" if dp > 15 else "#00CC96")
                    for dp in delta_pcts
                ]

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        name=t("retrain.history.fi_trace_baseline", version=source_version),
                        x=features_names,
                        y=baseline_vals,
                        marker_color="#636EFA",
                    )
                )
                fig.add_trace(
                    go.Bar(
                        name=t("retrain.history.fi_trace_new", version=new_version),
                        x=features_names,
                        y=new_vals,
                        marker_color=bar_colors_new,
                    )
                )
                fig.update_layout(
                    barmode="group",
                    title=t("retrain.history.fi_chart_title", model=hist_model_name),
                    xaxis_title=t("retrain.history.fi_xaxis"),
                    yaxis_title=t("retrain.history.fi_yaxis"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=420,
                )
                st.plotly_chart(fig, width='stretch')

                with st.expander(t("retrain.history.fi_detail_expander")):
                    detail_rows = sorted(comparison, key=lambda x: x["delta_pct"], reverse=True)
                    col_feat = t("retrain.history.fi_col_feature")
                    col_delta = t("retrain.history.fi_col_delta")
                    col_delta_pct = t("retrain.history.fi_col_delta_pct")
                    col_alert = t("retrain.history.fi_col_alert")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    col_feat: c["feature"],
                                    f"v{source_version}": round(c["baseline"], 5),
                                    f"v{new_version}": round(c["new"], 5),
                                    col_delta: round(c["delta"], 5),
                                    col_delta_pct: f"{c['delta_pct']:.1f}%",
                                    col_alert: (
                                        "🔴 >30%"
                                        if c["delta_pct"] > 30
                                        else ("🟠 >15%" if c["delta_pct"] > 15 else "✅ <15%")
                                    ),
                                }
                                for c in detail_rows
                            ]
                        ),
                        width='stretch',
                        hide_index=True,
                        column_config={
                            col_feat: st.column_config.TextColumn(
                                col_feat,
                                help=t("retrain.history.fi_help_feature"),
                            ),
                            f"v{source_version}": st.column_config.NumberColumn(
                                f"v{source_version}",
                                help=t("retrain.history.fi_help_baseline_col", version=source_version),
                                format="%.5f",
                            ),
                            f"v{new_version}": st.column_config.NumberColumn(
                                f"v{new_version}",
                                help=t("retrain.history.fi_help_new_col", version=new_version),
                                format="%.5f",
                            ),
                            col_delta: st.column_config.NumberColumn(
                                col_delta,
                                help=t("retrain.history.fi_help_delta"),
                                format="%.5f",
                            ),
                            col_delta_pct: st.column_config.TextColumn(
                                col_delta_pct,
                                help=t("retrain.history.fi_help_delta_pct"),
                            ),
                            col_alert: st.column_config.TextColumn(
                                col_alert,
                                help=t("retrain.history.fi_help_alert"),
                            ),
                        },
                    )
