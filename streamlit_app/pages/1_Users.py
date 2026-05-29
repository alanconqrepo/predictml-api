"""
User management — admin only
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.auth import get_client, logout, require_admin, require_auth
from utils.i18n import t
from utils.ui_helpers import show_token_with_copy


def show_quota_progress(used: int, limit: int) -> None:
    if limit <= 0:
        st.info(t("users.quota_unlimited"))
        return
    pct = used / limit
    pct_clamped = min(pct * 100, 100)
    remaining = max(limit - used, 0)

    if pct < 0.70:
        bar_color = "#21c354"
    elif pct < 0.90:
        bar_color = "#ff9f0a"
    else:
        bar_color = "#ff3b30"

    tooltip = t("users.quota_tooltip", used=used, limit=limit)
    st.markdown(
        f"""<div title="{tooltip}" style="margin-bottom:6px">
  <div style="background:rgba(255,255,255,0.15);border-radius:6px;height:18px;width:100%;overflow:hidden">
    <div style="background:{bar_color};width:{pct_clamped:.1f}%;height:100%;border-radius:6px;transition:width .3s"></div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )
    caption = t("users.quota_caption", used=used, limit=limit, remaining=remaining)
    if pct >= 0.90:
        st.error(caption)
    elif pct >= 0.70:
        st.warning(caption)
    else:
        st.success(caption)


st.set_page_config(page_title=t("users.page_title"), page_icon="👥", layout="wide")

require_auth()

# ── Non-admin view ──────────────────────────────────────────────────────────────
if not st.session_state.get("is_admin"):
    client = get_client()
    with st.expander(t("users.my_profile"), expanded=True):
        try:
            me = client.get_me()
            quota = client.get_my_quota()
            st.markdown(t("users.profile_username_role", username=me['username'], role=me['role']))
            st.markdown(t("users.profile_email", email=me['email']))

            st.markdown(t("users.profile_api_token"))
            show_token_with_copy(me["api_token"])

            st.markdown(t("users.profile_daily_quota"))
            used = quota["used_today"]
            limit = quota["rate_limit_per_day"]
            show_quota_progress(used, limit)
        except Exception as e:
            st.error(t("users.error_load_profile", error=e))

    st.divider()

    with st.expander(t("users.regen_token_expander")):
        st.warning(t("users.regen_token_warning"))
        if st.button(t("users.regen_token_btn"), key="rotate_self_token"):
            try:
                result = client.regenerate_my_token()
                new_token = result["api_token"]
                st.session_state["api_token"] = new_token
                st.success(t("users.regen_token_success"))
                show_token_with_copy(new_token)
            except Exception as e:
                st.error(t("users.error_generic", error=e))

    st.stop()

# ── Admin view ──────────────────────────────────────────────────────────────────
require_admin()

st.title(t("users.title"))

client = get_client()


@st.cache_data(ttl=10, show_spinner=False)
def fetch_users(api_url, token):
    c = get_client()
    return c.list_users()


@st.cache_data(ttl=15, show_spinner=False)
def fetch_account_requests(api_url, token, status_filter=None):
    c = get_client()
    return c.get_account_requests(status=status_filter)


def reload():
    st.cache_data.clear()
    st.rerun()


tab_users, tab_requests = st.tabs([t("users.tab_users"), t("users.tab_requests")])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — User management
# ═══════════════════════════════════════════════════════════════════════════════
with tab_users:

    # Load users (executed before expanders)
    try:
        users = fetch_users(st.session_state.get("api_url"), st.session_state.get("api_token"))
    except Exception as e:
        st.error(t("users.error_load_users", error=e))
        st.stop()

    if not users:
        st.info(t("users.no_users"))
        st.stop()

    _col_id       = t("users.col.id")
    _col_username = t("users.col.username")
    _col_email    = t("users.col.email")
    _col_role     = t("users.col.role")
    _col_status   = t("users.col.status")
    _col_quota    = t("users.col.quota")
    _col_last_login = t("users.col.last_login")
    _col_created  = t("users.col.created")

    df = pd.DataFrame(users)[[
        "id", "username", "email", "role", "is_active",
        "rate_limit_per_day", "last_login", "created_at",
    ]]
    df["last_login"] = pd.to_datetime(df["last_login"]).dt.strftime("%Y-%m-%d %H:%M").fillna("—")
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d")
    df["is_active"]  = df["is_active"].map({True: t("users.status_active"), False: t("users.status_inactive")})
    df = df.rename(columns={
        "id": _col_id,
        "username": _col_username,
        "email": _col_email,
        "role": _col_role,
        "is_active": _col_status,
        "rate_limit_per_day": _col_quota,
        "last_login": _col_last_login,
        "created_at": _col_created,
    })

    # 1. User list (open) -------------------------------------------
    with st.expander(t("users.expander_list"), expanded=True):
        st.dataframe(
            df,
            width='stretch',
            hide_index=True,
            column_config={
                _col_id: st.column_config.NumberColumn(_col_id, help=t("users.col.id_help")),
                _col_username: st.column_config.TextColumn(_col_username, help=t("users.col.username_help")),
                _col_email: st.column_config.TextColumn(_col_email, help=t("users.col.email_help")),
                _col_role: st.column_config.TextColumn(_col_role, help=t("users.col.role_help")),
                _col_status: st.column_config.TextColumn(_col_status, help=t("users.col.status_help")),
                _col_quota: st.column_config.NumberColumn(_col_quota, help=t("users.col.quota_help")),
                _col_last_login: st.column_config.TextColumn(_col_last_login, help=t("users.col.last_login_help")),
                _col_created: st.column_config.TextColumn(_col_created, help=t("users.col.created_help")),
            },
        )

    # The user selectbox is needed here to define `selected` before Analytics
    user_options = {f"{u['username']} (id:{u['id']})": u for u in users}

    # 2. Actions (closed) -----------------------------------------------------------
    with st.expander(t("users.expander_actions"), expanded=False):
        selected_label = st.selectbox(t("users.select_user"), list(user_options.keys()))
        selected = user_options[selected_label]

        col_a, col_b, col_c, col_d = st.columns(4)

        current_active = selected["is_active"]
        toggle_label = t("users.btn_deactivate") if current_active else t("users.btn_activate")
        if col_a.button(f"{'🔴' if current_active else '🟢'} {toggle_label}", width='stretch'):
            try:
                client.update_user(selected["id"], {"is_active": not current_active})
                st.toast(t("users.toast_deactivated") if current_active else t("users.toast_activated"), icon="✅")
                reload()
            except Exception as e:
                st.toast(t("users.error_generic", error=e), icon="❌")

        if col_b.button(t("users.btn_renew_token"), width='stretch'):
            try:
                result = client.update_user(selected["id"], {"regenerate_token": True})
                new_token = result["api_token"]
                st.session_state[f"regen_token_{selected['id']}"] = new_token
                st.toast(t("users.toast_token_generated"), icon="✅")
            except Exception as e:
                st.toast(t("users.error_generic", error=e), icon="❌")

        regen_key = f"regen_token_{selected['id']}"
        if st.session_state.get(regen_key):
            st.info(t("users.new_token_info"))
            show_token_with_copy(st.session_state[regen_key])
            if st.button(t("users.btn_clear_token"), key="clear_regen_token"):
                del st.session_state[regen_key]
                st.rerun()

        new_roles = [r for r in ["user", "admin", "readonly"] if r != selected["role"]]
        _role_val = st.session_state.get("role_select", new_roles[0] if new_roles else None)
        if _role_val not in new_roles and new_roles:
            _role_val = new_roles[0]
        if col_c.button(t("users.btn_apply_role"), width='stretch'):
            try:
                client.update_user(selected["id"], {"role": _role_val})
                st.toast(t("users.toast_role_updated", role=_role_val), icon="✅")
                reload()
            except Exception as e:
                st.toast(t("users.error_generic", error=e), icon="❌")
        col_c.selectbox(
            t("users.select_change_role"),
            new_roles,
            key="role_select",
            format_func=lambda r: t(f"users.role.{r}"),
        )

        with col_d:
            if st.button(t("users.btn_delete"), width='stretch', type="secondary"):
                st.session_state["confirm_delete_user"] = selected["id"]

        if st.session_state.get("confirm_delete_user") == selected["id"]:
            st.warning(t("users.confirm_delete_warning", username=selected['username']))
            c1, c2 = st.columns(2)
            if c1.button(t("users.btn_confirm_delete"), type="primary"):
                try:
                    client.delete_user(selected["id"])
                    st.toast(t("users.toast_deleted"), icon="✅")
                    st.session_state.pop("confirm_delete_user", None)
                    reload()
                except Exception as e:
                    st.toast(t("users.error_generic", error=e), icon="❌")
            if c2.button(t("users.btn_cancel")):
                st.session_state.pop("confirm_delete_user", None)
                st.rerun()

    # `selected` is set by the selectbox in the with block above
    # (with expander blocks always execute, even when the expander is closed)
    _selected = user_options[st.session_state.get(
        "pm_model_select",
        list(user_options.keys())[0]
    )] if "selected" not in vars() else selected
    selected = _selected if "selected" not in vars() else selected

    # 3. Create a new user (closed) ---------------------------------------
    with st.expander(t("users.expander_create"), expanded=False):
        with st.form("create_user_form"):
            col1, col2 = st.columns(2)
            username = col1.text_input(t("users.form_username"), placeholder="john_doe")
            email    = col2.text_input(t("users.form_email"), placeholder="john@example.com")
            col3, col4 = st.columns(2)
            role       = col3.selectbox(t("users.form_role"), ["user", "admin", "readonly"])
            rate_limit = col4.number_input(t("users.form_quota"), min_value=1, max_value=100000, value=1000)
            submitted  = st.form_submit_button(t("users.form_submit"), width='stretch', type="primary")

        if submitted:
            if not username or not email:
                st.error(t("users.form_error_required"))
            else:
                try:
                    result = client.create_user({
                        "username": username, "email": email,
                        "role": role, "rate_limit": rate_limit,
                    })
                    st.toast(t("users.toast_created", username=result['username']), icon="✅")
                    st.info(t("users.info_save_token"))
                    show_token_with_copy(result["api_token"])
                    reload()
                except Exception as e:
                    st.toast(t("users.error_generic", error=e), icon="❌")

    # 4. Usage analytics (closed) -------------------------------------------------
    _default_end   = pd.Timestamp.now().date()
    _default_start = _default_end - pd.Timedelta(days=29)
    with st.expander(t("users.expander_analytics", username=selected['username']), expanded=False):
        daily_limit = selected["rate_limit_per_day"]

        try:
            _quota_usage = client.get_user_usage(
                selected["id"], start_date=str(_default_end), end_date=str(_default_end),
            )
            today_calls = sum(d["calls"] for d in _quota_usage.get("by_day", []))
        except Exception:
            today_calls = 0
        pct = today_calls / daily_limit * 100 if daily_limit > 0 else 0
        st.metric(
            label=t("users.analytics_quota_today"),
            value=f"{pct:.1f}%",
            delta=t("users.analytics_quota_delta", calls=today_calls, limit=daily_limit),
            delta_color="off",
        )

        col_date_start, col_date_end = st.columns(2)
        usage_date_start = col_date_start.date_input(
            t("users.analytics_date_start"), value=_default_start, key=f"usage_start_{selected['id']}"
        )
        usage_date_end = col_date_end.date_input(
            t("users.analytics_date_end"), value=_default_end, key=f"usage_end_{selected['id']}"
        )

        if usage_date_start > usage_date_end:
            st.warning(t("users.analytics_date_error"))
        else:
            try:
                usage = client.get_user_usage(
                    selected["id"],
                    start_date=str(usage_date_start),
                    end_date=str(usage_date_end),
                )

                _col_model_name  = t("users.analytics.col_model")
                _col_vol_total   = t("users.analytics.col_volume_total")
                _col_max_day     = t("users.analytics.col_max_day")
                _col_avg_day     = t("users.analytics.col_avg_day")
                _col_median_day  = t("users.analytics.col_median_day")

                by_model_day = usage.get("by_model_day", [])
                if by_model_day:
                    df_md = pd.DataFrame(by_model_day)
                    df_md["date"] = pd.to_datetime(df_md["date"]).dt.date
                    rows_table = []
                    for model_name, grp in df_md.groupby("model_name"):
                        s = grp["calls"]
                        rows_table.append({
                            _col_model_name: model_name,
                            _col_vol_total:  int(s.sum()),
                            _col_max_day:    int(s.max()),
                            _col_avg_day:    round(float(s.mean()), 1),
                            _col_median_day: round(float(s.median()), 1),
                        })
                    by_day_data = usage.get("by_day", [])
                    if by_day_data:
                        s_all = pd.DataFrame(by_day_data)["calls"]
                        rows_table.append({
                            _col_model_name: t("users.analytics.col_total_label"),
                            _col_vol_total:  int(s_all.sum()),
                            _col_max_day:    int(s_all.max()),
                            _col_avg_day:    round(float(s_all.mean()), 1),
                            _col_median_day: round(float(s_all.median()), 1),
                        })
                    st.subheader(t("users.analytics.subheader_by_model"))
                    st.dataframe(
                        pd.DataFrame(rows_table), width='stretch', hide_index=True,
                        column_config={
                            _col_model_name: st.column_config.TextColumn(_col_model_name),
                            _col_vol_total:  st.column_config.NumberColumn(_col_vol_total),
                            _col_max_day:    st.column_config.NumberColumn(_col_max_day),
                            _col_avg_day:    st.column_config.NumberColumn(_col_avg_day, format="%.1f"),
                            _col_median_day: st.column_config.NumberColumn(_col_median_day, format="%.1f"),
                        },
                    )
                elif usage.get("by_model"):
                    _col_m = t("users.analytics.col_model")
                    _col_v = t("users.analytics.col_volume_total")
                    _col_e = t("users.analytics.col_errors")
                    df_model = pd.DataFrame(usage["by_model"]).rename(
                        columns={"model_name": _col_m, "calls": _col_v, "errors": _col_e}
                    )[[_col_m, _col_v, _col_e]]
                    st.subheader(t("users.analytics.subheader_by_model"))
                    st.dataframe(df_model, width='stretch', hide_index=True)
                else:
                    st.info(t("users.analytics.no_predictions"))

                if by_model_day:
                    df_md_plot = pd.DataFrame(by_model_day)
                    df_md_plot["date"] = pd.to_datetime(df_md_plot["date"])
                    df_md_plot = df_md_plot.sort_values("date")
                    fig_day = go.Figure()
                    colors = px.colors.qualitative.Set2
                    for i, (model_name, grp) in enumerate(df_md_plot.groupby("model_name")):
                        color = colors[i % len(colors)]
                        fig_day.add_trace(go.Scatter(
                            x=grp["date"], y=grp["calls"],
                            mode="lines", name=model_name,
                            stackgroup="one",
                            line={"color": color, "width": 1},
                            fillcolor=color,
                        ))
                    y_max = df_md_plot["calls"].max()
                    total_max = y_max * df_md_plot["model_name"].nunique()
                    if daily_limit > 0 and daily_limit <= total_max * 3:
                        fig_day.add_hline(
                            y=daily_limit, line_dash="dash", line_color="red",
                            annotation_text=t("users.analytics.quota_annotation", limit=daily_limit),
                            annotation_position="top right",
                        )
                    else:
                        fig_day.add_annotation(
                            text=t("users.analytics.quota_annotation", limit=daily_limit),
                            xref="paper", yref="paper", x=1, y=1.08,
                            showarrow=False, font=dict(color="red", size=11),
                        )
                    fig_day.update_layout(
                        title=t("users.analytics.chart_title"),
                        xaxis_title=t("users.analytics.chart_x"),
                        yaxis_title=t("users.analytics.chart_y"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                        margin={"t": 60, "r": 20},
                        yaxis=dict(rangemode="tozero"),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_day, width='stretch')
                else:
                    st.info(t("users.analytics.no_daily_data"))

            except Exception as e:
                st.error(t("users.analytics.error_load", error=e))

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Access requests
# ═══════════════════════════════════════════════════════════════════════════════
with tab_requests:
    try:
        pending_requests = fetch_account_requests(
            st.session_state.get("api_url"),
            st.session_state.get("api_token"),
            status_filter="pending",
        )
    except Exception as e:
        st.error(t("users.requests.error_load", error=e))
        pending_requests = []

    n_pending = len(pending_requests)
    if n_pending > 0:
        st.warning(t("users.requests.pending_count", count=n_pending))
    else:
        st.success(t("users.requests.none_pending"))

    if pending_requests:
        st.subheader(t("users.requests.subheader_pending"))

        _req_col_id         = t("users.requests.col.id")
        _req_col_username   = t("users.requests.col.username")
        _req_col_email      = t("users.requests.col.email")
        _req_col_role       = t("users.requests.col.role_requested")
        _req_col_message    = t("users.requests.col.message")
        _req_col_requested  = t("users.requests.col.requested_at")

        df_req = pd.DataFrame(pending_requests)[
            ["id", "username", "email", "role_requested", "message", "requested_at"]
        ]
        df_req["requested_at"] = pd.to_datetime(df_req["requested_at"]).dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        df_req["message"] = df_req["message"].fillna("—")
        df_req = df_req.rename(columns={
            "id": _req_col_id,
            "username": _req_col_username,
            "email": _req_col_email,
            "role_requested": _req_col_role,
            "message": _req_col_message,
            "requested_at": _req_col_requested,
        })
        st.dataframe(
            df_req,
            width="stretch",
            hide_index=True,
            column_config={
                _req_col_id: st.column_config.NumberColumn(
                    _req_col_id,
                    help=t("users.requests.col.id_help"),
                ),
                _req_col_username: st.column_config.TextColumn(
                    _req_col_username,
                    help=t("users.requests.col.username_help"),
                ),
                _req_col_email: st.column_config.TextColumn(
                    _req_col_email,
                    help=t("users.requests.col.email_help"),
                ),
                _req_col_role: st.column_config.TextColumn(
                    _req_col_role,
                    help=t("users.requests.col.role_requested_help"),
                ),
                _req_col_message: st.column_config.TextColumn(
                    _req_col_message,
                    help=t("users.requests.col.message_help"),
                ),
                _req_col_requested: st.column_config.TextColumn(
                    _req_col_requested,
                    help=t("users.requests.col.requested_at_help"),
                ),
            },
        )

        st.divider()
        req_options = {
            f"{r['username']} — {r['email']} (id:{r['id']})": r for r in pending_requests
        }
        selected_req_label = st.selectbox(t("users.requests.select_request"), list(req_options.keys()))
        selected_req = req_options[selected_req_label]

        st.markdown(t("users.requests.detail_username", username=selected_req['username']))
        st.markdown(t("users.requests.detail_email", email=selected_req['email']))
        st.markdown(t("users.requests.detail_role", role=selected_req['role_requested']))
        if selected_req.get("message"):
            st.markdown(t("users.requests.detail_message", message=selected_req['message']))

        col_approve, col_reject = st.columns(2)

        with col_approve:
            if st.button(t("users.requests.btn_approve"), width="stretch", type="primary"):
                try:
                    result = client.approve_account_request(selected_req["id"])
                    created_user = result["created_user"]
                    st.success(t(
                        "users.requests.approve_success",
                        username=created_user['username'],
                        email=created_user['email'],
                    ))
                    show_token_with_copy(created_user["api_token"])
                    reload()
                except Exception as e:
                    st.error(t("users.requests.error_approve", error=e))

        with col_reject:
            reject_reason = st.text_input(
                t("users.requests.reject_reason_label"), key="reject_reason_input"
            )
            if st.button(t("users.requests.btn_reject"), width="stretch", type="secondary"):
                try:
                    client.reject_account_request(
                        selected_req["id"], reason=reject_reason or None
                    )
                    st.toast(t("users.requests.toast_rejected"), icon="✅")
                    reload()
                except Exception as e:
                    st.error(t("users.requests.error_reject", error=e))

    st.divider()
    with st.expander(t("users.requests.expander_history")):
        try:
            all_requests = fetch_account_requests(
                st.session_state.get("api_url"),
                st.session_state.get("api_token"),
            )
            past = [r for r in all_requests if r["status"] != "pending"]
            if past:
                _hist_col_id     = t("users.requests.col.id")
                _hist_col_user   = t("users.requests.col.username")
                _hist_col_email  = t("users.requests.col.email")
                _hist_col_role   = t("users.requests.col.role_requested")
                _hist_col_status = t("users.requests.col.status")
                _hist_col_review = t("users.requests.col.reviewed_at")
                _hist_col_reason = t("users.requests.col.rejection_reason")

                df_past = pd.DataFrame(past)[
                    ["id", "username", "email", "role_requested", "status", "reviewed_at", "rejection_reason"]
                ]
                df_past["reviewed_at"] = pd.to_datetime(df_past["reviewed_at"]).dt.strftime(
                    "%Y-%m-%d %H:%M"
                ).fillna("—")
                df_past["rejection_reason"] = df_past["rejection_reason"].fillna("—")
                df_past["status"] = df_past["status"].map(
                    {"approved": t("users.requests.status_approved"), "rejected": t("users.requests.status_rejected")}
                )
                df_past = df_past.rename(columns={
                    "id": _hist_col_id,
                    "username": _hist_col_user,
                    "email": _hist_col_email,
                    "role_requested": _hist_col_role,
                    "status": _hist_col_status,
                    "reviewed_at": _hist_col_review,
                    "rejection_reason": _hist_col_reason,
                })
                st.dataframe(
                    df_past,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        _hist_col_id: st.column_config.NumberColumn(
                            _hist_col_id,
                            help=t("users.requests.col.id_help"),
                        ),
                        _hist_col_user: st.column_config.TextColumn(
                            _hist_col_user,
                            help=t("users.requests.col.username_hist_help"),
                        ),
                        _hist_col_email: st.column_config.TextColumn(
                            _hist_col_email,
                            help=t("users.requests.col.email_help"),
                        ),
                        _hist_col_role: st.column_config.TextColumn(
                            _hist_col_role,
                            help=t("users.requests.col.role_hist_help"),
                        ),
                        _hist_col_status: st.column_config.TextColumn(
                            _hist_col_status,
                            help=t("users.requests.col.status_help"),
                        ),
                        _hist_col_review: st.column_config.TextColumn(
                            _hist_col_review,
                            help=t("users.requests.col.reviewed_at_help"),
                        ),
                        _hist_col_reason: st.column_config.TextColumn(
                            _hist_col_reason,
                            help=t("users.requests.col.rejection_reason_help"),
                        ),
                    },
                )
            else:
                st.info(t("users.requests.no_history"))
        except Exception as e:
            st.error(t("users.error_generic", error=e))
