"""
Validation of reference test cases (Golden Test Set) for ML models
"""

import json

import pandas as pd
import streamlit as st
from utils.api_client import get_golden_tests as get_golden_tests_cached
from utils.api_client import get_models as get_models_cached
from utils.auth import get_client, require_auth
from utils.i18n import t

st.set_page_config(page_title=t("golden_tests.page_title"), page_icon="🧪", layout="wide")
require_auth()

is_admin = st.session_state.get("is_admin", False)

st.title(t("golden_tests.title"))
st.caption(t("golden_tests.caption"))

client = get_client()


def reload():
    st.cache_data.clear()
    st.rerun()


# ─── Model loading ───────────────────────────────────────────────────
try:
    models = get_models_cached(
        st.session_state.get("api_url"), st.session_state.get("api_token")
    )
except Exception as e:
    st.error(t("golden_tests.error_load_models", error=e))
    st.stop()

if not models:
    st.info(t("golden_tests.no_models"))
    st.stop()

# ─── Section 1 — Model selection ─────────────────────────────────────────
st.subheader(t("golden_tests.section_select_model"))

model_names = sorted({m["name"] for m in models})
col_name, col_version = st.columns([2, 1])

with col_name:
    selected_name = st.selectbox(t("golden_tests.model_label"), model_names, key="gt_model_name")

versions = [m["version"] for m in models if m["name"] == selected_name]
with col_version:
    selected_version = st.selectbox(t("golden_tests.version_label"), versions, key="gt_model_version")

# ─── Section 2 — Existing test cases ──────────────────────────────────────
st.markdown("---")
st.subheader(t("golden_tests.section_existing_tests"))

try:
    golden_tests = get_golden_tests_cached(
        st.session_state.get("api_url"),
        st.session_state.get("api_token"),
        selected_name,
    )
except Exception as e:
    st.error(t("golden_tests.error_load_tests", error=e))
    golden_tests = []

if not golden_tests:
    st.info(t("golden_tests.no_tests", name=selected_name))
else:
    _col_id = t("golden_tests.col_id")
    _col_desc = t("golden_tests.col_description")
    _col_input = t("golden_tests.col_input_features")
    _col_expected = t("golden_tests.col_expected_output")
    _col_date = t("golden_tests.col_date")
    _col_created_by = t("golden_tests.col_created_by")

    rows = []
    for test in golden_tests:
        rows.append(
            {
                _col_id: test.get("id"),
                _col_desc: test.get("description") or "—",
                _col_input: json.dumps(test.get("input_features", {})),
                _col_expected: str(test.get("expected_output", "—")),
                _col_date: (
                    pd.to_datetime(test["created_at"]).strftime("%Y-%m-%d %H:%M")
                    if test.get("created_at")
                    else "—"
                ),
                _col_created_by: test.get("created_by_username") or "—",
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        width='stretch',
        hide_index=True,
        column_config={
            _col_id: st.column_config.NumberColumn(_col_id, help=t("golden_tests.col_id_help")),
            _col_desc: st.column_config.TextColumn(_col_desc, help=t("golden_tests.col_description_help")),
            _col_input: st.column_config.TextColumn(_col_input, help=t("golden_tests.col_input_features_help")),
            _col_expected: st.column_config.TextColumn(_col_expected, help=t("golden_tests.col_expected_output_help")),
            _col_date: st.column_config.TextColumn(_col_date, help=t("golden_tests.col_date_help")),
            _col_created_by: st.column_config.TextColumn(_col_created_by, help=t("golden_tests.col_created_by_help")),
        },
    )
    st.caption(t("golden_tests.tests_count", count=len(golden_tests)))

    col_run, col_spacer = st.columns([1, 3])
    with col_run:
        run_clicked = st.button(
            t("golden_tests.run_all_btn", version=selected_version),
            type="primary",
            key="run_tests_btn",
        )

    if is_admin:
        with st.expander(t("golden_tests.delete_expander"), expanded=False):
            test_opts = {
                f"#{test['id']} — {test.get('description') or test.get('expected_output', '')}": test["id"]
                for test in golden_tests
            }
            to_delete_label = st.selectbox(
                t("golden_tests.delete_select_label"), list(test_opts.keys()), key="gt_delete_select"
            )
            if st.button(t("golden_tests.delete_btn"), type="secondary", key="gt_delete_btn"):
                test_id = test_opts[to_delete_label]
                try:
                    client.delete_golden_test(selected_name, test_id)
                    st.toast(t("golden_tests.delete_success", id=test_id), icon="✅")
                    reload()
                except Exception as e:
                    st.toast(t("golden_tests.delete_error", error=e), icon="❌")

    if run_clicked:
        with st.spinner(t("golden_tests.run_spinner", count=len(golden_tests), name=selected_name, version=selected_version)):
            try:
                run_result = client.run_golden_tests(selected_name, selected_version)

                total = run_result.get("total_tests", len(golden_tests))
                passed = run_result.get("passed", 0)
                failed = run_result.get("failed", 0)
                pass_rate = run_result.get("pass_rate", 0.0)
                details = run_result.get("details", [])

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric(t("golden_tests.metric_total"), total)
                col_m2.metric(t("golden_tests.metric_passed"), passed)
                col_m3.metric(t("golden_tests.metric_failed"), failed)
                col_m4.metric(t("golden_tests.metric_pass_rate"), f"{pass_rate * 100:.1f}%")

                if not details:
                    st.info(t("golden_tests.no_details"))
                else:
                    st.markdown(t("golden_tests.detailed_results_header"))
                    for i, r in enumerate(details):
                        test_passed = r.get("passed", False)
                        icon = "✅" if test_passed else "❌"
                        desc = r.get("description") or t("golden_tests.test_id_label", id=r.get('test_id'))
                        label = f"{icon} {desc}"

                        with st.expander(label, expanded=(i == 0)):
                            col_exp, col_rec = st.columns(2)
                            with col_exp:
                                st.markdown(t("golden_tests.expected_label"))
                                st.code(str(r.get("expected", "—")))
                            with col_rec:
                                st.markdown(t("golden_tests.received_label"))
                                actual = str(r.get("actual", "—"))
                                if not test_passed:
                                    st.error(actual)
                                else:
                                    st.code(actual)

                            if not test_passed:
                                expected_val = str(r.get("expected", ""))
                                actual_val = str(r.get("actual", ""))
                                if expected_val != actual_val:
                                    st.markdown(t("golden_tests.diff_label"))
                                    st.markdown(
                                        t("golden_tests.diff_content",
                                          expected=expected_val, actual=actual_val)
                                    )

                            with st.expander(t("golden_tests.input_used_label"), expanded=False):
                                st.json(r.get("input", {}))

            except Exception as e:
                st.error(t("golden_tests.run_error", error=e))

# ─── Section 3 — Add a test case (admin only) ───────────────────────────────────────
st.markdown("---")
st.subheader(t("golden_tests.section_add_test"))

if not is_admin:
    st.info(t("golden_tests.admin_only"))
else:
    with st.expander(t("golden_tests.add_expander"), expanded=False):
        with st.form("add_golden_test_form"):
            selected_model_meta = next(
                (m for m in models if m["name"] == selected_name and m["version"] == selected_version),
                None,
            )
            _meta = selected_model_meta or {}
            feature_baseline = _meta.get("feature_baseline") or {}
            _classes = _meta.get("classes") or []

            description = st.text_input(
                t("golden_tests.form_description_label"),
                placeholder=t("golden_tests.form_description_placeholder", name=selected_name),
                key="gt_description",
            )
            if feature_baseline:
                default_features = json.dumps(
                    {k: 0.0 for k in feature_baseline.keys()}, indent=2
                )
            else:
                default_features = '{\n  "feature1": 1.0,\n  "feature2": 2.0\n}'
            features_json = st.text_area(
                t("golden_tests.form_features_label"),
                value=default_features,
                height=130,
                key="gt_features",
                help=t("golden_tests.form_features_help"),
            )
            if _classes:
                _expected_placeholder = " | ".join(str(c) for c in _classes[:3])
            else:
                _expected_placeholder = t("golden_tests.form_expected_placeholder")
            expected_output = st.text_input(
                t("golden_tests.form_expected_label"),
                placeholder=_expected_placeholder,
                key="gt_expected",
                help=t("golden_tests.form_expected_help"),
            )
            submitted = st.form_submit_button(t("golden_tests.form_save_btn"), type="primary")

        if submitted:
            try:
                features = json.loads(features_json)
            except json.JSONDecodeError as e:
                st.error(t("golden_tests.error_invalid_json", error=e))
            else:
                if not description.strip():
                    st.error(t("golden_tests.error_description_required"))
                elif not expected_output.strip():
                    st.error(t("golden_tests.error_expected_required"))
                elif not features:
                    st.error(t("golden_tests.error_features_empty"))
                elif _classes and expected_output.strip() not in [str(c) for c in _classes]:
                    st.error(t("golden_tests.error_expected_not_in_classes", classes=", ".join(str(c) for c in _classes)))
                else:
                    try:
                        payload = {
                            "input_features": features,
                            "expected_output": expected_output.strip(),
                            "description": description.strip(),
                        }
                        client.create_golden_test(selected_name, payload)
                        st.toast(t("golden_tests.save_success"), icon="✅")
                        reload()
                    except Exception as e:
                        st.toast(t("golden_tests.save_error", error=e), icon="❌")

    with st.expander(t("golden_tests.csv_expander"), expanded=False):
        # Build a dynamic example using the selected model's actual feature names
        _selected_model_meta = next(
            (m for m in models if m["name"] == selected_name and m["version"] == selected_version),
            None,
        )
        _feature_baseline = (_selected_model_meta or {}).get("feature_baseline") or {}
        if _feature_baseline:
            _feat_cols = list(_feature_baseline.keys())
            _feat_vals = ["0.0"] * len(_feat_cols)
        else:
            _feat_cols = ["feature1", "feature2"]
            _feat_vals = ["1.0", "2.0"]
        _header = ",".join(_feat_cols + ["expected_output", "description"])
        _row = ",".join(_feat_vals + ["label", "example description"])
        _csv_example = f"{_header}\n{_row}"
        st.markdown(t("golden_tests.csv_format_intro"))
        st.code(_csv_example, language="text")
        st.markdown(t("golden_tests.csv_format_columns", cols=",".join(f"`{c}`" for c in _feat_cols)))
        uploaded_csv = st.file_uploader(
            t("golden_tests.csv_uploader_label"), type=["csv"], key="gt_csv_upload"
        )
        if uploaded_csv is not None:
            if st.button(t("golden_tests.csv_import_btn"), type="primary", key="gt_csv_import_btn"):
                try:
                    result = client.upload_golden_tests_csv(
                        selected_name,
                        uploaded_csv.read(),
                        uploaded_csv.name,
                    )
                    imported = result.get("imported", result.get("count", "?"))
                    st.toast(t("golden_tests.csv_import_success", count=imported), icon="✅")
                    reload()
                except Exception as e:
                    st.toast(t("golden_tests.csv_import_error", error=e), icon="❌")
